from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

from PIL import Image


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fission import FissionAgent


def ocr_raw(words):
    return {
        "ParsedResults": [{
            "TextOverlay": {"Lines": [{"LineText": "领7天出行守护", "Words": words}]}
        }]
    }


class TextLocalizationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.image_path = str(Path(self.temp.name) / "sample.png")
        Image.new("RGB", (630, 214), "white").save(self.image_path)
        self.agent = FissionAgent()

    def tearDown(self):
        self.temp.cleanup()

    def to_pixels(self, box):
        return [box[0] * 0.63, box[1] * 0.214, box[2] * 0.63, box[3] * 0.214]

    def test_target_inside_a_long_ocr_word_keeps_full_glyph_height(self):
        raw = ocr_raw([
            {"WordText": "领7天出行守护", "Left": 20, "Top": 40, "Width": 196, "Height": 28}
        ])
        box = self.agent._vlm_refine_bbox_exact(self.image_path, raw, "7天")
        left, top, right, bottom = self.to_pixels(box)
        self.assertLessEqual(left, 48)
        self.assertGreaterEqual(right, 104)
        self.assertLess(top, 40)
        self.assertGreater(bottom, 68)
        self.assertEqual(self.agent._last_locate_debug["src"], "WORD_GEOMETRY_1")

    def test_split_words_do_not_include_neighbours(self):
        raw = ocr_raw([
            {"WordText": "领", "Left": 20, "Top": 40, "Width": 24, "Height": 28},
            {"WordText": "7", "Left": 49, "Top": 40, "Width": 20, "Height": 28},
            {"WordText": "天", "Left": 72, "Top": 40, "Width": 24, "Height": 28},
            {"WordText": "出行守护", "Left": 102, "Top": 40, "Width": 100, "Height": 28},
        ])
        box = self.agent._vlm_refine_bbox_exact(self.image_path, raw, "7天")
        left, _, right, _ = self.to_pixels(box)
        self.assertGreater(left, 44)
        self.assertLess(right, 102)

    def test_preview_region_matches_localized_box(self):
        mask, pixel_box = self.agent.build_edit_mask(
            self.image_path, [80, 170, 170, 340], is_text_mode=True
        )
        self.assertEqual(mask.size, (630, 214))
        self.assertEqual(mask.getbbox(), pixel_box)
        self.assertLessEqual(pixel_box[0], 50)
        self.assertGreaterEqual(pixel_box[2], 108)

    def test_text_generation_uses_historical_qwen_path_first(self):
        payload = {
            "filename": "sample.png",
            "img_path": self.image_path,
            "ocr_text": "原装正品 现货速发 无线感应充电",
        }
        analysis = {
            "modification_type": "TEXT_EDIT",
            "target_text_content": "现货速发",
            "target_desc": "当日发货",
            "box_2d": [55, 285, 489, 426],
            "locate_debug": {"src": "WORD_GEOMETRY_3", "confidence": 0.96},
        }
        success = {"success": True, "model": "qwen-image-edit-max"}
        with patch.object(
            self.agent, "_call_qwen_precise_edit", return_value=success
        ) as qwen_call:
            result = self.agent.generate(payload, analysis)

        self.assertTrue(result["success"])
        self.assertEqual(result["model"], "qwen-image-edit-max")
        qwen_call.assert_called_once_with(
            self.image_path,
            [55, 285, 489, 426],
            "当日发货",
            is_text_mode=True,
            target_text="现货速发",
        )


if __name__ == "__main__":
    unittest.main()
