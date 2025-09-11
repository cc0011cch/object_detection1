import os
import sys
import unittest
import tempfile
from pathlib import Path


class TestLoggingUtils(unittest.TestCase):
    def test_setup_logger_writes_file(self):
        from engine.logging_utils import setup_logger
        with tempfile.TemporaryDirectory() as td:
            log_path = Path(td) / "test.log"
            logger = setup_logger(str(log_path), to_console=False)
            logger.info("hello world")
            self.assertTrue(log_path.exists())
            text = log_path.read_text(encoding="utf-8")
            self.assertIn("hello world", text)


class TestSampler(unittest.TestCase):
    def test_repeat_factor_sampler_min_floor(self):
        import random
        from engine.sampler import RepeatFactorSampler
        idxs = [0, 1, 2]
        r = [1.0, 2.3, 0.7]
        random.seed(123)
        s = RepeatFactorSampler(idxs, r, shuffle=False)
        out = list(iter(s))
        # At least floor(r_i) repetitions per index
        from math import floor
        for i, ri in zip(idxs, r):
            self.assertGreaterEqual(out.count(i), floor(ri))


class TestRFS(unittest.TestCase):
    def test_compute_repeat_factors_fast(self):
        try:
            import numpy  # noqa: F401
        except Exception:
            self.skipTest("numpy not available")

        from engine.rfs import compute_repeat_factors_fast
        # Minimal COCO-like JSON with 3 images and 2 classes
        coco = {
            "images": [
                {"id": 1, "file_name": "a.jpg", "width": 10, "height": 10},
                {"id": 2, "file_name": "b.jpg", "width": 10, "height": 10},
                {"id": 3, "file_name": "c.jpg", "width": 10, "height": 10},
            ],
            "annotations": [
                {"image_id": 1, "category_id": 100, "bbox": [1,1,2,2]},
                {"image_id": 2, "category_id": 100, "bbox": [1,1,2,2]},
                {"image_id": 3, "category_id": 200, "bbox": [1,1,2,2]},
            ],
            "categories": [
                {"id": 100, "name": "A"},
                {"id": 200, "name": "B"},
            ],
        }
        with tempfile.TemporaryDirectory() as td:
            ann = Path(td) / "tiny.json"
            ann.write_text(__import__("json").dumps(coco), encoding="utf-8")
            img_ids_order = [1, 2, 3]
            rf = compute_repeat_factors_fast(str(ann), img_ids_order, threshold=0.6, alpha=0.5)
            self.assertEqual(len(rf), len(img_ids_order))
            # Should be floats > 0
            self.assertTrue(all(isinstance(x, float) and x > 0 for x in rf))


class TestOptionalTorch(unittest.TestCase):
    def test_build_model_factory_imports(self):
        # Only verify import; skip actual build if torch missing
        try:
            import torch  # noqa: F401
        except Exception:
            self.skipTest("torch not available")
        # Import should succeed
        from models import factory  # noqa: F401


if __name__ == "__main__":
    unittest.main()

