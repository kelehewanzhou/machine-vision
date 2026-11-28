import json
import unittest
import sys
import tempfile
from pathlib import Path
from PIL import Image
import numpy as np
import cv2
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import crop


class TestCrop(unittest.TestCase):

    def setUp(self):
        output_dir = ROOT / "outputs"
        output_dir.mkdir(exist_ok=True)

        self.temp = tempfile.TemporaryDirectory(
            prefix="crop_test_",
            dir=output_dir
        )
        self.addCleanup(self.temp.cleanup)

        self.test_dir = Path(self.temp.name)
        self.run_id = self.test_dir.name

        self.cfg = {
            "mode": "best_overall",
            "runtime": {"run_id": self.run_id}
        }

        flat_path = self.test_dir / "flat.png"
        Image.new("RGB", (400, 300), (128, 128, 128)).save(flat_path)

        self.flat_path = flat_path.relative_to(ROOT).as_posix()

    def assert_statistics(self, result):
        s = result["data"]["statistics"]

        self.assertEqual(
            s["proposed"],
            s["filtered"] + s["deduplicated"]
            + s["write_failed"] + s["returned"]
        )

        self.assertEqual(
            s["returned"],
            len(result["data"]["candidates"])
        )

    def test_best_original(self):
        self.cfg["mode"] = "best_original"

        r = crop.generate_candidates(
            self.flat_path, "flat01",
            self.cfg, group_id="g01"
        )

        self.assertEqual(r["status"], "ok")
        self.assertEqual(len(r["data"]["candidates"]), 1)

        c = r["data"]["candidates"][0]

        self.assertEqual(c["crop_id"], 0)
        self.assertTrue(c["is_original"])
        self.assertEqual(c["crop_path"], self.flat_path)
        self.assertEqual(c["box"], [0, 0, 400, 300])
        self.assertIsNone(c["metadata"]["position"])
        self.assert_statistics(r)

    def test_flat_image(self):
        r = crop.generate_candidates(
            self.flat_path, "flat01",
            self.cfg, group_id="g01"
        )

        self.assertEqual(r["status"], "partial")

        self.assertIn(
            "INFERENCE_FAILED",
            [e["code"] for e in r["errors"]]
        )

        self.assertEqual(len(r["data"]["candidates"]), 1)

        c = r["data"]["candidates"][0]

        self.assertEqual(c["crop_id"], 0)
        self.assertIsNone(c["subject_coverage"])
        self.assertIsNone(c["saliency_retention"])
        self.assert_statistics(r)

    def test_normal_images(self):
        images = [
            ("data/test/test.jpg", "astronaut", 6),
            ("data/test/test_coffee.jpg", "coffee", 9),
            ("data/test/test_rocket.jpg", "rocket", 4)
        ]

        for path, name, expected in images:
            with self.subTest(image=name):

                if not (ROOT / path).is_file():
                    self.skipTest(f"Missing image: {path}")

                r = crop.generate_candidates(
                    path, name, self.cfg, group_id="g01"
                )

                self.assertEqual(r["status"], "ok")
                self.assertEqual(r["errors"], [])
                self.assertEqual(
                    r["data"]["statistics"]["returned"],
                    expected
                )
                self.assertEqual(
                    r["data"]["statistics"]["write_failed"],
                    0
                )

                self.assert_statistics(r)

                for c in r["data"]["candidates"]:
                    self.assertTrue(
                        (ROOT / c["crop_path"]).is_file()
                    )

    def test_missing_image(self):
        r = crop.generate_candidates(
            "data/test/not_exist.png",
            "img01", self.cfg, group_id="g01"
        )

        self.assertEqual(r["status"], "error")
        self.assertIsNone(r["data"])
        self.assertEqual(
            r["errors"][0]["code"], "IMAGE_NOT_FOUND"
        )

    def test_invalid_image_id(self):
        r = crop.generate_candidates(
            self.flat_path,
            "../img01", self.cfg, group_id="g01"
        )

        self.assertEqual(r["status"], "error")
        self.assertEqual(
            r["errors"][0]["code"], "INVALID_INPUT"
        )

    def test_invalid_path(self):
        r = crop.generate_candidates(
            "../test.jpg",
            "img01", self.cfg, group_id="g01"
        )

        self.assertEqual(r["status"], "error")
        self.assertEqual(
            r["errors"][0]["code"], "INVALID_INPUT"
        )

    def test_invalid_mode(self):
        self.cfg["mode"] = "unknown"

        r = crop.generate_candidates(
            self.flat_path,
            "img01", self.cfg, group_id="g01"
        )

        self.assertEqual(r["status"], "error")
        self.assertEqual(
            r["errors"][0]["code"], "INVALID_CONFIG"
        )

    def test_invalid_threshold(self):
        self.cfg["crop"] = {
            "iou_threshold": 1.5
        }

        r = crop.generate_candidates(
            self.flat_path,
            "img01", self.cfg, group_id="g01"
        )

        self.assertEqual(r["status"], "error")
        self.assertEqual(
            r["errors"][0]["code"], "INVALID_CONFIG"
        )

    def test_missing_model(self):
        self.cfg["crop"] = {
            "face_model_path": "models/not_exist.onnx"
        }

        r = crop.generate_candidates(
            "data/test/test.jpg",
            "img01", self.cfg, group_id="g01"
        )

        self.assertEqual(r["status"], "partial")

        self.assertIn(
            "MODEL_UNAVAILABLE",
            [e["code"] for e in r["errors"]]
        )

        self.assertGreater(
            len(r["data"]["candidates"]), 1
        )

        self.assert_statistics(r)

    def test_write_failure(self):
        with patch(
            "crop.os.replace",
            side_effect=OSError("模拟写入失败")
        ):
            r = crop.generate_candidates(
                "data/test/test.jpg",
                "img01", self.cfg, group_id="g01"
            )

        self.assertEqual(r["status"], "partial")
        self.assertEqual(r["data"]["statistics"]["returned"], 1)
        self.assertEqual(r["data"]["statistics"]["write_failed"], 5)

        self.assertIn(
            "CROP_WRITE_FAILED",
            [e["code"] for e in r["errors"]]
        )

        self.assertEqual(
            list(self.test_dir.rglob(".tmp_*.png")), []
        )

        self.assert_statistics(r)

    def test_saliency_failure(self):
        with patch(
            "crop.spectral_residual_saliency",
            side_effect=cv2.error("模拟显著性计算失败")
        ):
            r = crop.generate_candidates(
                "data/test/test.jpg",
                "img01", self.cfg, group_id="g01"
            )

        self.assertEqual(r["status"], "partial")

        self.assertIn(
            "INFERENCE_FAILED",
            [e["code"] for e in r["errors"]]
        )

        for c in r["data"]["candidates"]:
            self.assertIsNone(c["saliency_retention"])

        self.assert_statistics(r)

    def test_extreme_sizes(self):
        rng = np.random.default_rng(42)

        for h, w in [
            (1, 1),
            (10000, 10),
            (10, 10000),
            (1, 1000),
            (1000, 1)
        ]:
            with self.subTest(height=h, width=w):
                img = rng.integers(
                    0, 256, (h, w, 3), dtype=np.uint8
                )

                sal = crop.spectral_residual_saliency(img)

                if sal is not None:
                    self.assertEqual(sal.shape, (h, w))
                    self.assertTrue(np.isfinite(sal).all())
                    self.assertGreaterEqual(float(sal.min()), 0)
                    self.assertLessEqual(float(sal.max()), 1)

    def test_crop_pixels(self):
        path = "data/test/test.jpg"

        r = crop.generate_candidates(
            path, "img01", self.cfg, group_id="g01"
        )

        self.assertEqual(r["status"], "ok")

        original = crop.read_image(str(ROOT / path))

        for c in r["data"]["candidates"]:
            if c["is_original"]:
                continue

            x1, y1, x2, y2 = c["box"]

            saved = crop.read_image(
                str(ROOT / c["crop_path"])
            )

            expected = original[y1:y2, x1:x2]

            np.testing.assert_array_equal(saved, expected)

    def test_candidate_metadata(self):
        r = crop.generate_candidates(
            "data/test/test.jpg",
            "img01", self.cfg, group_id="g01"
        )

        self.assertEqual(r["status"], "ok")

        ids = []

        for c in r["data"]["candidates"]:
            ids.append(c["crop_id"])

            x1, y1, x2, y2 = c["box"]

            self.assertTrue(
                0 <= x1 < x2 <= c["image_width"]
            )

            self.assertTrue(
                0 <= y1 < y2 <= c["image_height"]
            )

            position = c["metadata"]["position"]

            if c["is_original"]:
                self.assertIsNone(position)
            else:
                self.assertEqual(len(position), 2)

                for p in position:
                    self.assertIn(p, [0.0, 0.5, 1.0])

        self.assertEqual(len(ids), len(set(ids)))

    def _rel(self, p):
        return p.relative_to(ROOT).as_posix()

    def test_exif_orientation(self):
        p = self.test_dir / "exif.jpg"
        exif = Image.Exif()
        exif[0x0112] = 6                      # 需要顺时针旋转 90°
        Image.new("RGB", (40, 20), (200, 50, 50)).save(p, exif=exif)
        self.cfg["mode"] = "best_original"
        r = crop.generate_candidates(self._rel(p), "exif01", self.cfg, group_id="g01")
        c = r["data"]["candidates"][0]
        self.assertEqual((c["image_width"], c["image_height"]), (20, 40))
        self.assertEqual(c["box"], [0, 0, 20, 40])

    def test_transparent_png(self):
        p = self.test_dir / "alpha.png"
        Image.new("RGBA", (50, 50), (0, 0, 0, 0)).save(p)
        arr = crop.read_image(str(p))
        self.assertEqual(arr.shape, (50, 50, 3))
        self.assertTrue((arr == 255).all())   # 全透明 → 纯白

    def test_unsupported_and_corrupt(self):
        bmp = self.test_dir / "x.bmp"
        Image.new("RGB", (20, 20)).save(bmp)
        bad = self.test_dir / "bad.png"
        bad.write_bytes(b"not an image")
        for p, code in ((bmp, "UNSUPPORTED_FORMAT"), (bad, "IMAGE_DECODE_FAILED")):
            with self.subTest(code=code):
                r = crop.generate_candidates(self._rel(p), "x01", self.cfg, group_id="g01")
                self.assertEqual(r["status"], "error")
                self.assertIsNone(r["data"])
                self.assertEqual(r["errors"][0]["code"], code)

    def test_max_pixels(self):
        self.cfg["image"] = {"max_image_pixels": 100}
        r = crop.generate_candidates(self.flat_path, "flat01", self.cfg, group_id="g01")
        self.assertEqual(r["errors"][0]["code"], "INVALID_IMAGE_SIZE")

    def test_stable_ids_and_iou(self):
        run = lambda: crop.generate_candidates("data/test/test.jpg", "img01", self.cfg, group_id="g01")
        r1, r2 = run(), run()
        key = lambda r: [(c["crop_id"], c["box"]) for c in r["data"]["candidates"]]
        self.assertEqual(key(r1), key(r2))                 # 两次运行编号一致
        boxes = [c["box"] for c in r1["data"]["candidates"]]
        for i in range(len(boxes)):
            for j in range(i + 1, len(boxes)):
                self.assertLess(crop.iou(boxes[i], boxes[j]), 0.85)
        json.dumps(r1, allow_nan=False)                    # 能序列化且没有 NaN

    def test_extreme_sizes_end_to_end(self):
        for h, w in [(1, 1), (2, 3), (10, 3000), (3000, 10)]:
            with self.subTest(height=h, width=w):
                p = self.test_dir / f"ext_{h}x{w}.png"
                rng = np.random.default_rng(0)
                Image.fromarray(rng.integers(0, 256, (h, w, 3), dtype=np.uint8)).save(p)
                r = crop.generate_candidates(self._rel(p), "ext01", self.cfg, group_id="g01")
                self.assertIn(r["status"], ("ok", "partial"))
                self.assertEqual(r["data"]["candidates"][0]["crop_id"], 0)
                self.assert_statistics(r)

if __name__ == "__main__":
    unittest.main(verbosity=2)