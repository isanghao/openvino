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
import re
import struct
import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np


HEADER_MAGIC = 0xDEADBEEF
HEADER_MAGIC_END = 0xCAFEBABE
HEADER_INTS = 24


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
    is_prefill: int
    subsequence_begin: int


def parse_header(buf: bytes) -> Header:
    if len(buf) < HEADER_INTS * 4:
        raise ValueError(f"buffer too small ({len(buf)} bytes) for header")
    words = struct.unpack("<" + "I" * HEADER_INTS, buf[: HEADER_INTS * 4])
    if words[0] != HEADER_MAGIC or words[15] != HEADER_MAGIC_END:
        raise ValueError(
            f"header magic mismatch: got 0x{words[0]:08X}/0x{words[15]:08X}, "
            f"expected 0x{HEADER_MAGIC:08X}/0x{HEADER_MAGIC_END:08X}. "
            "The kernel likely did not run the dump path (e.g., not a "
            "PREFILL/MIXED-stage sdpa_micro launch, or the dump was truncated)."
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
        is_prefill=signed[20],
        subsequence_begin=signed[21],
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


def dump_tile_data(path: str, hdr: Header, K: np.ndarray, Q: np.ndarray,
                    S: np.ndarray, dump_dir: str) -> None:
    """Save the raw K/Q/S tile data and the recomputed reference S as .npy
    (for reloading) and .txt (for manual eyeballing) files under dump_dir."""
    os.makedirs(dump_dir, exist_ok=True)
    base = os.path.join(dump_dir, os.path.splitext(os.path.basename(path))[0])

    vk = max(0, min(hdr.valid_k, hdr.sg_tile_m))
    vq = max(0, min(hdr.valid_q, hdr.sg_tile_n))
    ref = K[:vk].astype(np.float64) @ Q[:vq].astype(np.float64).T if vk and vq else np.zeros((0, 0))

    for name, arr in (("K", K), ("Q", Q), ("S", S), ("S_ref", ref)):
        np.save(f"{base}_{name}.npy", arr)
        np.savetxt(f"{base}_{name}.txt", arr, fmt="%.6g")

    with open(f"{base}_header.txt", "w") as f:
        for field, value in vars(hdr).items():
            f.write(f"{field} = {value}\n")

    print(f"  [dump] wrote K/Q/S/S_ref (.npy + .txt) and header to {dump_dir}/ (prefix: {os.path.basename(base)})")


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
        f"[{status}] stage={'PREFILL' if hdr.is_prefill else 'GEN/MIXED'} "
        f"past_len={hdr.past_len} d={hdr.d} k0={hdr.k0} "
        f"wg_j0={hdr.wg_j0} sg_tile=({hdr.sg_tile_m},{hdr.sg_tile_n}) "
        f"valid=({vk},{vq}) q_total={hdr.q_total} causal_k={hdr.causal_k} "
        f"subseq_begin={hdr.subsequence_begin} "
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


# ---------------------------------------------------------------------------
# Optional: compare tile K/Q against the paged-attention layer's raw f16
# input tensors (dumped via OV_GPU_Verbose etc.). These files are named
# `..._srcN__f16__<total_tokens>_<features>_1_1__bfyx.bin` where features =
# heads * head_size (Q) or kv_heads * head_size (K/V).


_FILENAME_RE = re.compile(
    r"__(?:f16|u8|i8|i32)__(\d+)_(\d+)_(\d+)_(\d+)__bfyx\.bin$"
)


def _parse_dims_from_filename(path: str) -> Optional[Tuple[int, int, int, int]]:
    m = _FILENAME_RE.search(os.path.basename(path))
    if not m:
        return None
    return tuple(int(x) for x in m.groups())  # (b, f, y, x)


def _load_f16_tensor(path: str, tokens: int) -> Tuple[np.ndarray, int]:
    """Load a bfyx f16 file as an (tokens, features) fp32 array plus the
    inferred features count. The filename's advertised feature dim is not
    trusted — some paged-attention dumps write the pre-split (heads*d)
    Q with the K/V feature count in the filename."""
    raw = np.fromfile(path, dtype=np.float16)
    if raw.size % tokens != 0:
        # Pad trailing zeros so the buffer divides evenly by tokens.
        features = raw.size // tokens + 1
        padded = np.zeros(tokens * features, dtype=np.float16)
        padded[: raw.size] = raw
        print(
            f"  [inputs] {path}: element count {raw.size} not divisible by "
            f"tokens={tokens}; padded with {padded.size - raw.size} zero(s) "
            f"to features={features}",
            file=sys.stderr,
        )
        raw = padded
    else:
        features = raw.size // tokens
    return raw.reshape(tokens, features).astype(np.float32), features


def _best_match(needle: np.ndarray, haystack: np.ndarray) -> Tuple[int, float, float]:
    """Given a needle vector [d] and haystack [N, d], return (best_idx,
    max|diff|, cosine_similarity) for the row closest to `needle`."""
    diff = haystack - needle[None, :]
    per_row_max = np.abs(diff).max(axis=1)
    best = int(np.argmin(per_row_max))
    max_abs = float(per_row_max[best])
    nn = float(np.linalg.norm(needle))
    hn = float(np.linalg.norm(haystack[best]))
    denom = nn * hn if nn > 0 and hn > 0 else 1.0
    cos = float(np.dot(needle, haystack[best]) / denom)
    return best, max_abs, cos


def compare_inputs(hdr: Header, K: np.ndarray, Q: np.ndarray,
                    q_path: str, k_path: str, v_path: str,
                    heads_num: Optional[int]) -> int:
    """Scan the layer's raw Q/K/V bfyx f16 inputs for the best match to
    the tile's Q sub-tile and (if any) the Kc region of the K sub-tile."""
    # Infer dims from filenames (all three files have the same B/F/Y/X).
    for p in (q_path, k_path, v_path):
        if not os.path.isfile(p):
            print(f"  [inputs] missing file: {p}", file=sys.stderr)
            return 2
    q_dims = _parse_dims_from_filename(q_path)
    k_dims = _parse_dims_from_filename(k_path)
    if q_dims is None or k_dims is None:
        print("  [inputs] cannot parse dims from filename(s); "
              "expected pattern `..._f16__<b>_<f>_<y>_<x>__bfyx.bin`",
              file=sys.stderr)
        return 2
    tokens = q_dims[0]

    src_q, q_features = _load_f16_tensor(q_path, tokens)
    src_k, k_features = _load_f16_tensor(k_path, tokens)
    src_v, v_features = _load_f16_tensor(v_path, tokens)

    if heads_num is None:
        if q_features % hdr.d != 0:
            print(f"  [inputs] cannot infer heads_num: q_features={q_features} "
                  f"not divisible by d={hdr.d}", file=sys.stderr)
            return 2
        heads_num = q_features // hdr.d
    if k_features % hdr.d != 0 or (k_features // hdr.d) != hdr.kv_heads_num:
        print(f"  [inputs] cannot infer kv_heads_num: k_features={k_features} "
              f"vs header kv_heads_num={hdr.kv_heads_num} d={hdr.d}",
              file=sys.stderr)
        return 2
    if v_features != k_features:
        print(f"  [inputs] v_features={v_features} != k_features={k_features}",
              file=sys.stderr)
        return 2
    kv_heads_num = hdr.kv_heads_num

    print(f"  [inputs] tokens={tokens} heads={heads_num} kv_heads={kv_heads_num} "
          f"head_size={hdr.d} q_features={q_features} k_features={k_features}")

    src_q = src_q.reshape(tokens, heads_num, hdr.d)
    src_k = src_k.reshape(tokens, kv_heads_num, hdr.d)
    src_v = src_v.reshape(tokens, kv_heads_num, hdr.d)

    rc = 0
    vq = max(0, min(hdr.valid_q, hdr.sg_tile_n))
    vk = max(0, min(hdr.valid_k, hdr.sg_tile_m))

    # Compare each valid Q row against all (token, head) slots in src0.
    q_haystack = src_q.reshape(tokens * heads_num, hdr.d)
    for j in range(vq):
        needle = Q[j]
        idx, max_abs, cos = _best_match(needle, q_haystack)
        t = idx // heads_num
        h = idx % heads_num
        tag = "OK" if max_abs < 1e-3 else "FAR"
        print(
            f"  [Q j={j}] best match src0[token={t}, head={h}]: "
            f"max|diff|={max_abs:.6g}  cos={cos:.6f}  [{tag}]"
        )
        if max_abs >= 1e-3:
            rc |= 1

    # Kc: only k rows with k_row >= past_len come from the current-iter K
    # input (src1); rows below past_len are past-K decoded from the INT4
    # KV cache.
    new_k_rows = [i for i in range(vk) if (hdr.k0 + i) >= hdr.past_len]
    if new_k_rows:
        k_haystack = src_k.reshape(tokens * kv_heads_num, hdr.d)
        for i in new_k_rows:
            needle = K[i]
            idx, max_abs, cos = _best_match(needle, k_haystack)
            t = idx // kv_heads_num
            h = idx % kv_heads_num
            tag = "OK" if max_abs < 1e-3 else "FAR"
            print(
                f"  [K i={i} (new)] best match src1[token={t}, kv_head={h}]: "
                f"max|diff|={max_abs:.6g}  cos={cos:.6f}  [{tag}]"
            )
            if max_abs >= 1e-3:
                rc |= 1
    else:
        print(f"  [K] no new-K rows in tile (all k_row < past_len={hdr.past_len}); "
              f"past-K comes from paged INT4 cache, not comparable to src1")

    # V doesn't appear in the KQ tile dump, but expose overall stats so the
    # user can eyeball src2 for gross corruption.
    v_flat = src_v.reshape(-1)
    print(
        f"  [V] src2 stats: min={v_flat.min():.4f} max={v_flat.max():.4f} "
        f"mean={v_flat.mean():.4f} std={v_flat.std():.4f} nan={int(np.isnan(v_flat).sum())}"
    )
    return rc


# ---------------------------------------------------------------------------
# Optional: decode the paged K-cache (src3) and compare it against the
# tile's K sub-tile. Only the INT4 BY_CHANNEL layout is decoded here
# because that is the only compressed layout the kernel dump can produce
# a directly-comparable K sub-tile for (see IS_INT4_KV_CACHE &&
# IS_KEY_BY_CHANNEL branch in sdpa_micro.cl).
#
# Cache layout for INT4 BY_CHANNEL (as written by paged attention):
#   src3 shape (u8, bfyx): [blocks, kv_heads, head_size, packed_block + 4]
#   packed_block = PAGED_ATTENTION_BLOCK_SIZE / 2  (16/2 = 8 bytes)
#   +4 bytes = fp16 scale (2 B) followed by fp16 zp (2 B)
# Decoded value for (block, kv_head, dim, token):
#   packed = column[token >> 1]
#   u4     = packed & 0xF if token even else (packed >> 4) & 0xF
#   scale  = fp16(column[packed_block : packed_block+2])
#   zp     = fp16(column[packed_block+2 : packed_block+4])
#   K      = (u4 - zp) * scale


def _decode_int4_bychannel_block(cache: np.ndarray, block: int, kv_head: int,
                                  block_size: int) -> np.ndarray:
    """Decode one (block, kv_head) slab into a (block_size, head_size) fp32 array."""
    slab = cache[block, kv_head]  # [head_size, adj_pa_blk_bytes]
    head_size, adj = slab.shape
    packed_bytes = block_size // 2
    assert adj == packed_bytes + 4, (
        f"expected adjusted PA block bytes = {packed_bytes}+4 = {packed_bytes+4}, got {adj}"
    )
    packed = slab[:, :packed_bytes]                                    # [head_size, packed]
    scale = slab[:, packed_bytes:packed_bytes+2].copy().view(np.float16).astype(np.float32).reshape(head_size)
    zp    = slab[:, packed_bytes+2:packed_bytes+4].copy().view(np.float16).astype(np.float32).reshape(head_size)

    # Expand packed nibbles into (head_size, block_size) u4 values.
    lo = packed & 0x0F
    hi = (packed >> 4) & 0x0F
    u4 = np.empty((head_size, block_size), dtype=np.float32)
    u4[:, 0::2] = lo
    u4[:, 1::2] = hi
    decoded = (u4 - zp[:, None]) * scale[:, None]  # [head_size, block_size]
    # Return with token as the leading axis so callers get [block_size, head_size].
    return decoded.T.copy()


def compare_kcache(hdr: Header, K: np.ndarray, kcache_path: str) -> int:
    if not os.path.isfile(kcache_path):
        print(f"  [kcache] missing file: {kcache_path}", file=sys.stderr)
        return 2
    dims = _parse_dims_from_filename(kcache_path)
    if dims is None:
        print("  [kcache] cannot parse dims from filename", file=sys.stderr)
        return 2
    blocks, kv_heads, head_size, adj_block_bytes = dims
    if head_size != hdr.d or kv_heads != hdr.kv_heads_num:
        print(f"  [kcache] shape mismatch: dims={dims} vs header "
              f"kv_heads={hdr.kv_heads_num} d={hdr.d}", file=sys.stderr)
        return 2
    if not hdr.is_int4_kv_cache:
        print("  [kcache] tile header says K cache is not INT4 — decode not "
              "implemented for other layouts, skipping", file=sys.stderr)
        return 2
    block_size = hdr.pa_block_size
    if adj_block_bytes != block_size // 2 + 4:
        print(f"  [kcache] adjusted block bytes {adj_block_bytes} != "
              f"{block_size}/2 + 4 — layout not INT4 BY_CHANNEL, skipping",
              file=sys.stderr)
        return 2

    raw = np.fromfile(kcache_path, dtype=np.uint8)
    expected = blocks * kv_heads * head_size * adj_block_bytes
    if raw.size != expected:
        print(f"  [kcache] byte count {raw.size} != expected {expected}",
              file=sys.stderr)
        return 2
    cache = raw.reshape(blocks, kv_heads, head_size, adj_block_bytes)

    # In prefill the tile K comes entirely from the layer's src1; the paged
    # K cache has not been written back yet on this iteration.
    if hdr.is_prefill:
        print("  [kcache] tile is from PREFILL — K sub-tile comes from src1, "
              "not the paged cache; skipping cache comparison")
        return 0

    # Tile K covers k_row in [k0, k0 + valid_k). Group by physical block id
    # and decode only what we need; require k0 to be block-aligned so we can
    # map k_row -> (block_local_id, within).
    if hdr.k0 % block_size != 0:
        print(f"  [kcache] k0={hdr.k0} not aligned to block_size={block_size}; "
              "skipping direct comparison")
        return 2

    # The kernel dump captures only the WG(0,0,0)/sg=0 tile, which starts at
    # base_block_index + k0/block_size. The bfyx dump does not include the
    # block_indices remap array, so this comparison assumes an identity
    # mapping (block_indices[i] == i). This is true for the currently-used
    # unit tests but may need `--block-indices` if that changes.
    first_block_id = hdr.base_block_index + hdr.k0 // block_size

    vk = max(0, min(hdr.valid_k, hdr.sg_tile_m))
    if vk == 0:
        print("  [kcache] no valid K rows in tile — nothing to compare")
        return 0

    n_blocks = (vk + block_size - 1) // block_size
    print(f"  [kcache] tile past-K spans {n_blocks} physical block(s) starting "
          f"at block_id={first_block_id}, kv_head={hdr.b0_kv}")

    rc = 0
    for b in range(n_blocks):
        block_id = first_block_id + b
        if block_id >= blocks:
            print(f"  [kcache] block_id={block_id} out of range (blocks={blocks})",
                  file=sys.stderr)
            return 2
        decoded = _decode_int4_bychannel_block(cache, block_id, hdr.b0_kv, block_size)
        row_lo = b * block_size
        row_hi = min(row_lo + block_size, vk)
        tile = K[row_lo:row_hi]                    # [n, head_size]
        ref = decoded[: row_hi - row_lo]           # [n, head_size]
        max_abs = float(np.abs(tile - ref).max())
        mean_abs = float(np.abs(tile - ref).mean())
        tag = "OK" if max_abs < 1e-3 else "MISMATCH"
        print(
            f"  [K rows {row_lo}..{row_hi-1}  cache block {block_id}] "
            f"max|diff|={max_abs:.6g}  mean|diff|={mean_abs:.6g}  [{tag}]"
        )
        if max_abs >= 1e-3:
            rc |= 1
            # Show the worst-offending element.
            idx = np.unravel_index(int(np.argmax(np.abs(tile - ref))), tile.shape)
            print(
                f"      worst @ (row={row_lo+idx[0]}, dd={idx[1]}): "
                f"tile={tile[idx]:.6f} cache={ref[idx]:.6f} diff={tile[idx]-ref[idx]:.6f}"
            )
    return rc


def main(argv: List[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("paths", nargs="+", help="Dumped .bin file(s)")
    ap.add_argument("--atol", type=float, default=1e-2)
    ap.add_argument("--rtol", type=float, default=1e-2)
    ap.add_argument("--inputs", nargs=3, metavar=("Q_BIN", "K_BIN", "V_BIN"),
                    help="Compare tile K/Q against the layer's raw Q/K/V f16 "
                         "input dumps (bfyx). Dims are inferred from the "
                         "filenames.")
    ap.add_argument("--kcache", metavar="K_CACHE_BIN",
                    help="Decode the paged K-cache (u8 bfyx: "
                         "[blocks, kv_heads, head_size, packed_block+4]) and "
                         "compare against the tile K sub-tile. Only INT4 "
                         "BY_CHANNEL layout is supported.")
    ap.add_argument("--heads", type=int, default=None,
                    help="Override the number of Q heads (else inferred from "
                         "Q_BIN's feature dim / head_size).")
    ap.add_argument("--dump-dir", metavar="DIR",
                    help="Dump the raw K/Q/S tile data and the recomputed "
                         "reference S as .npy and .txt files under DIR, one "
                         "set per input file, for manual inspection.")
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
        if args.dump_dir:
            dump_tile_data(path, hdr, K, Q, S, args.dump_dir)
        if args.inputs:
            rc |= compare_inputs(hdr, K, Q, args.inputs[0], args.inputs[1],
                                 args.inputs[2], args.heads)
        if args.kcache:
            rc |= compare_kcache(hdr, K, args.kcache)
    return rc


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
