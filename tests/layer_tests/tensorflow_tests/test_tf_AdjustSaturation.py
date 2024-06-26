# Copyright (C) 2018-2024 Intel Corporation
# SPDX-License-Identifier: Apache-2.0

import numpy as np
import pytest
import tensorflow as tf
from common.tf_layer_test_class import CommonTFLayerTest

class TestAdjustSaturation(CommonTFLayerTest):
    def _prepare_input(self, inputs_info):
        assert 'images:0' in inputs_info
        if self.special_case == "Black Image":
            images_shape = inputs_info['images:0']
            inputs_data = {}
            inputs_data['images:0'] = np.zeros(images_shape).astype(self.input_type) 
        elif self.special_case == "Grayscale Image":
            images_shape = inputs_info['images:0']
            inputs_data = {}
            inputs_data['images:0'] = np.ones(images_shape).astype(self.input_type) * np.random.rand()
        else:
            images_shape = inputs_info['images:0']
            inputs_data = {}
            inputs_data['images:0'] = np.random.rand(*images_shape).astype(self.input_type)
            
        inputs_data['scale:0'] = np.random.rand()
        
        return inputs_data

    def create_adjust_saturation_net(self, input_shape, input_type, special_case=False):
        self.special_case = special_case
        self.input_type = input_type
        tf.compat.v1.reset_default_graph()
        # Create the graph and model
        with tf.compat.v1.Session() as sess:
            images = tf.compat.v1.placeholder(input_type, input_shape, 'images')
            scale = tf.compat.v1.placeholder(input_type, [], 'scale')
            tf.raw_ops.AdjustSaturation(images=images, scale=scale)
            tf.compat.v1.global_variables_initializer()
            tf_net = sess.graph_def

        return tf_net, None

    # Each input is a tensor of at least 3 dimensions. 
    # The last dimension is interpreted as channels, and must be three.
    test_data_basic = [
        dict(input_shape=[7, 7, 3], input_type=np.float32, special_case="Black Image"),
        dict(input_shape=[7, 7, 3], input_type=np.float32, special_case="Grayscale Image"),
        dict(input_shape=[5, 5, 3], input_type=np.float32),
        dict(input_shape=[2, 3, 4, 3], input_type=np.float32),
        dict(input_shape=[1, 2, 3, 3, 3], input_type=np.float32),
    ]

    @pytest.mark.parametrize("params", test_data_basic)
    @pytest.mark.precommit
    @pytest.mark.nightly
    def test_adjust_saturation_basic(self, params, ie_device, precision, ir_version, temp_dir,
                                   use_legacy_frontend):
        if ie_device == 'GPU':
            pytest.skip("Accuracy mismatch on GPU")
        self._test(*self.create_adjust_saturation_net(**params),
                   ie_device, precision, ir_version, temp_dir=temp_dir,
                   use_legacy_frontend=use_legacy_frontend)