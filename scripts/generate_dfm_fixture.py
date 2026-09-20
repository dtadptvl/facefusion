#!/usr/bin/env python3
"""
Generates a minimal valid DFM ONNX fixture for iOS Simulator unit testing.
Exposes non-guessed tensor names without morph input:
Inputs: ['input_face'] shape [1, 224, 224, 3]
Outputs:
  0: 'custom_target_mask' shape [1, 224, 224, 1]
  1: 'custom_vision_frame' shape [1, 224, 224, 3]
  2: 'custom_source_mask' shape [1, 224, 224, 1]
Validates describeNames, ordered output metadata resolution, non-morph omission,
and mask length verification.
"""

import os
import sys

def generate_tiny_dfm_fixture(output_path: str):
    import onnx
    from onnx import helper, TensorProto

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    X = helper.make_tensor_value_info('input_face', TensorProto.FLOAT, [1, 224, 224, 3])
    Y_target = helper.make_tensor_value_info('custom_target_mask', TensorProto.FLOAT, [1, 224, 224, 1])
    Y_frame = helper.make_tensor_value_info('custom_vision_frame', TensorProto.FLOAT, [1, 224, 224, 3])
    Y_source = helper.make_tensor_value_info('custom_source_mask', TensorProto.FLOAT, [1, 224, 224, 1])

    # Slice node to extract 1 channel for mask tensors [1, 224, 224, 1]
    starts = helper.make_tensor('starts', TensorProto.INT64, [1], [0])
    ends = helper.make_tensor('ends', TensorProto.INT64, [1], [1])
    axes = helper.make_tensor('axes', TensorProto.INT64, [1], [3])
    steps = helper.make_tensor('steps', TensorProto.INT64, [1], [1])

    slice_node = helper.make_node('Slice', ['input_face', 'starts', 'ends', 'axes', 'steps'], ['custom_target_mask'])
    id_node = helper.make_node('Identity', ['input_face'], ['custom_vision_frame'])
    id_mask_node = helper.make_node('Identity', ['custom_target_mask'], ['custom_source_mask'])

    graph = helper.make_graph(
        [slice_node, id_node, id_mask_node],
        'tiny_dfm_nonmorph',
        [X],
        [Y_target, Y_frame, Y_source],
        [starts, ends, axes, steps]
    )

    model = helper.make_model(
        graph,
        producer_name='iFaceFusionTestGenerator',
        opset_imports=[helper.make_opsetid('', 13)]
    )
    onnx.checker.check_model(model)

    with open(output_path, 'wb') as f:
        f.write(model.SerializeToString())

    print(f"[PASS] Generated tiny DFM ONNX fixture ({os.path.getsize(output_path)} bytes) at: {output_path}")

if __name__ == '__main__':
    default_dest = os.path.join("tests", "fixtures", "tiny_dfm_224.onnx")
    dest = sys.argv[1] if len(sys.argv) > 1 else default_dest
    generate_tiny_dfm_fixture(dest)
