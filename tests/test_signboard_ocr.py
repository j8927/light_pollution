import unittest

import numpy as np

from signboard_ocr import crop_signboard_region, extract_store_name


class SignboardOcrTests(unittest.TestCase):
    def test_prefers_brand_over_generic_business_descriptor(self):
        lines = [
            {"text": "수제반찬전문점", "confidence": 0.99, "height": 120, "area": 50000},
            {"text": "해리the찬", "confidence": 0.52, "height": 55, "area": 12000},
        ]
        self.assertEqual(extract_store_name(lines), "해리the찬")

    def test_prefers_large_brand_over_long_slogan(self):
        lines = [
            {"text": "위시맨", "confidence": 0.91, "height": 180, "area": 55000},
            {"text": "디자인에 즐거움을 더하는 그룹", "confidence": 0.82, "height": 48, "area": 30000},
        ]
        self.assertEqual(extract_store_name(lines), "위시맨")

    def test_legacy_tuple_input_still_works(self):
        self.assertEqual(extract_store_name([("청울내곁소", 0.7)]), "청울내곁소")

    def test_crop_adds_vertical_room_and_clamps_to_image(self):
        image = np.zeros((100, 200, 3), dtype=np.uint8)
        crop = crop_signboard_region(image, (10, 20, 110, 60))
        self.assertEqual(crop.shape, (60, 116, 3))


if __name__ == "__main__":
    unittest.main()
