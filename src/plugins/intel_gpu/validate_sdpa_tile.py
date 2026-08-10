#!/usr/bin/env python3
"""Validate a KQ tile dumped by the OpenVINO Intel-GPU sdpa_micro kernel.

Enable the dump by exporting `OV_GPU_DUMP_SDPA_MICRO_TILE=<file-prefix>`
before running inference. Each launch of the sdpa_micro `_generate`
(paged-attention MIXED-stage) kernel appends a file named
`<file-prefix>_<seq>.bin` containing one KQ sub-tile from
WG(0,0,0), sub-group 0, first k-iteration.

Buffer layout (all little-endian):
    16 int32 header
    K sub-tile  : sg_tile_m * d  fp32, row-major [k_local][dd]
    Q sub-tile  : sg_tile_n * d  fp32, row-major [q_local][dd]
    S sub-tile  : sg_tile_m * sg_tile_n fp32, column-major [dd_row][j_col]

The script recomputes S = K x Q^T from the dumped operands and compares
it against the dumped ugemm output.
"""

from __future__ import annotations

import argparse
import os
import struct
import sys
from dataclasses import dataclass
from typing import List

import numpy as np


HEADER_MAGIC = 0xDEADBEEF
HEADER_MAGIC_END = 0xCAFEBABE
HEADER_INTS = 20


@dataclass
class Header:
    past_len: int
    d: int
    k0: int
    wg_j0: int
    sg_tile_m: int
    sg_tile_n: int
    kv_heads_num: int
    base_block_index: int
    b0_kv: int
    k_chunk: int
    adjusted_k_head_size: int
    adjusted_pa_block_size: int
    pa_block_size: int
    is_int4_kv_cache: int
    valid_k: int
    valid_q: int
    causal_k: int
    q_total: int


def parse_header(buf: bytes) -> Header:
    if len(buf) < HEADER_INTS * 4:
        raise ValueError(f"buffer too small ({len(buf)} bytes) for header")
    words = struct.unpack("<" + "I" * HEADER_INTS, buf[: HEADER_INTS * 4])
    if words[0] != HEADER_MAGIC or words[15] != HEADER_MAGIC_END:
        raise ValueError(
            f"header magic mismatch: got 0x{words[0]:08X}/0x{words[15]:08X}, "
            f"expected 0x{HEADER_MAGIC:08X}/0x{HEADER_MAGIC_END:08X}. "
            "The kernel likely did not run the dump path (e.g., not a "
            "MIXED-stage sdpa_micro launch, or the dump was truncated)."
        )
    signed = struct.unpack("<" + "i" * HEADER_INTS, buf[: HEADER_INTS * 4])
    return Header(
        past_len=signed[1],
        d=signed[2],
        k0=signed[3],
        wg_j0=signed[4],
        sg_tile_m=signed[5],
        sg_tile_n=signed[6],
        kv_heads_num=signed[7],
        base_block_index=signed[8],
        b0_kv=signed[9],
        k_chunk=signed[10],
        adjusted_k_head_size=signed[11],
        adjusted_pa_block_size=signed[12],
        pa_block_size=signed[13],
        is_int4_kv_cache=signed[14],
        valid_k=signed[16],
        valid_q=signed[17],
        causal_k=signed[18],
        q_total=signed[19],
    )


def _fp32_slice(buf: bytes, byte_offset: int, count: int) -> np.ndarray:
    end = byte_offset + count * 4
    if end > len(buf):
        raise ValueError(
            f"buffer too small: need {end} bytes for slice, have {len(buf)}"
        )
    return np.frombuffer(buf, dtype=np.float32, count=count, offset=byte_offset).copy()


def load_tile(path: str):
    with open(path, "rb") as f:
        buf = f.read()
    hdr = parse_header(buf)

    off = HEADER_INTS * 4
    k_count = hdr.sg_tile_m * hdr.d
    q_count = hdr.sg_tile_n * hdr.d
    s_count = hdr.sg_tile_m * hdr.sg_tile_n

    K = _fp32_slice(buf, off, k_count).reshape(hdr.sg_tile_m, hdr.d)
    off += k_count * 4
    Q = _fp32_slice(buf, off, q_count).reshape(hdr.sg_tile_n, hdr.d)
    off += q_count * 4
    # S sub-tile is row-major (sg_tile_m rows of sg_tile_n cols); the kernel
    # writes S_out[k * sg_tile_n + c] = S[k_row, q_col].
    S = _fp32_slice(buf, off, s_count).reshape(hdr.sg_tile_m, hdr.sg_tile_n)
    return hdr, K, Q, S


def compare(hdr: Header, K: np.ndarray, Q: np.ndarray, S: np.ndarray,
            atol: float, rtol: float) -> int:
    # Clip to in-bounds portion of the tile; ugemm output beyond valid_k/valid_q
    # is unmasked garbage that is later suppressed by the causal / q-remainder
    # masks in the kernel.
    vk = max(0, min(hdr.valid_k, hdr.sg_tile_m))
    vq = max(0, min(hdr.valid_q, hdr.sg_tile_n))
    if vk == 0 or vq == 0:
        print(f"[SKIP] empty valid region valid_k={hdr.valid_k} valid_q={hdr.valid_q}")
        return 0

    K_v = K[:vk].astype(np.float64)
    Q_v = Q[:vq].astype(np.float64)
    S_v = S[:vk, :vq].astype(np.float64)
    ref = K_v @ Q_v.T

    diff = S_v - ref
    abs_diff = np.abs(diff)
    abs_ref = np.maximum(np.abs(ref), 1e-6)
    rel = abs_diff / abs_ref

    max_abs = float(abs_diff.max())
    max_rel = float(rel.max())
    mean_abs = float(abs_diff.mean())

    passed = np.allclose(S_v, ref, atol=atol, rtol=rtol)
    status = "OK" if passed else "MISMATCH"

    print(
        f"[{status}] past_len={hdr.past_len} d={hdr.d} k0={hdr.k0} "
        f"wg_j0={hdr.wg_j0} sg_tile=({hdr.sg_tile_m},{hdr.sg_tile_n}) "
        f"valid=({vk},{vq}) q_total={hdr.q_total} causal_k={hdr.causal_k} "
        f"int4={bool(hdr.is_int4_kv_cache)}"
    )
    print(f"  max|diff|={max_abs:.6g}  max_rel={max_rel:.6g}  mean|diff|={mean_abs:.6g}")

    if not passed:
        idx = np.unravel_index(int(np.argmax(abs_diff)), abs_diff.shape)
        print(
            f"  worst @ (i={idx[0]}, j={idx[1]}): "
            f"got={S_v[idx]:.6f}  ref={ref[idx]:.6f}  diff={diff[idx]:.6f}"
        )
        return 1
    return 0


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="Dumped .bin file(s)")
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument("--rtol", type=float, default=1e-2)
    args = ap.parse_args(argv)

    rc = 0
    for path in args.paths:
        if not os.path.isfile(path):
            print(f"[skip] not a file: {path}", file=sys.stderr)
            rc |= 2
            continue
        print(f"=== {path} ===")
        try:
            hdr, K, Q, S = load_tile(path)
        except ValueError as e:
            print(f"  parse error: {e}", file=sys.stderr)
            rc |= 2
            continue
        rc |= compare(hdr, K, Q, S, args.atol, args.rtol)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
