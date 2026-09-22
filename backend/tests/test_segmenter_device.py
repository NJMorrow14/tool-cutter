"""The automatic point grid must not send float64 tensors to Metal."""
import sys
from pathlib import Path
import unittest
from types import SimpleNamespace
import numpy as np
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from toolcutter import segmenter  # registers the vendored package path

class DeviceTests(unittest.TestCase):
    def test_automatic_prompts_use_float32_even_from_a_float64_grid(self):
        import torch
        from segment_anything.automatic_mask_generator import SamAutomaticMaskGenerator
        class StopAfterPrompts(Exception): pass
        received=[]
        def predict(points, labels, **kwargs):
            received.append(points.dtype)
            raise StopAfterPrompts()
        generator=SamAutomaticMaskGenerator.__new__(SamAutomaticMaskGenerator)
        generator.predictor=SimpleNamespace(
            device='cpu', transform=SimpleNamespace(apply_coords=lambda p, size:p), predict_torch=predict)
        with self.assertRaises(StopAfterPrompts):
            generator._process_batch(np.array([[2.,3.]], dtype=np.float64), (20,20), [0,0,20,20], (20,20), True)
        self.assertEqual(received, [torch.float32])
