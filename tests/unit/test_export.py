from fall_detection.utils.export import export_classifier_onnx
import onnxruntime as ort
import numpy as np


def test_classifier_onnx_export():
    export_classifier_onnx('/tmp/fall_classifier.onnx')
    sess = ort.InferenceSession('/tmp/fall_classifier.onnx')
    names = [i.name for i in sess.get_inputs()]
    dummy = {
        names[0]: np.zeros((1, 3, 96, 96), dtype=np.float32),
        names[1]: np.zeros((1, 17, 3), dtype=np.float32),
        names[2]: np.zeros((1, 8), dtype=np.float32),
    }
    out = sess.run(None, dummy)
    assert len(out) == 1
    assert 0 <= out[0][0][0] <= 1
