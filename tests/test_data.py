from pathlib import Path
import unittest

from skin_lesion_segmentation.data import (
    Sample,
    extract_image_id,
    is_augmented_path,
    split_samples,
)


class DataTests(unittest.TestCase):
    def test_extract_image_id_from_mask_name(self):
        self.assertEqual(
            extract_image_id(Path("ISIC_0012345_segmentation.png")), "ISIC_0012345"
        )

    def test_split_is_deterministic_and_disjoint(self):
        samples = [
            Sample(Path(f"{i}.jpg"), Path(f"{i}.png"), f"ISIC_{i:07d}")
            for i in range(20)
        ]
        first = split_samples(samples, seed=42)
        second = split_samples(samples, seed=42)
        self.assertEqual(first, second)
        train, validation, test = first
        self.assertEqual((len(train), len(validation), len(test)), (12, 4, 4))
        ids = [set(item.image_id for item in group) for group in first]
        self.assertTrue(ids[0].isdisjoint(ids[1]))
        self.assertTrue(ids[0].isdisjoint(ids[2]))
        self.assertTrue(ids[1].isdisjoint(ids[2]))

    def test_invalid_split_is_rejected(self):
        samples = [
            Sample(Path(f"{i}.jpg"), Path(f"{i}.png"), str(i)) for i in range(10)
        ]
        with self.assertRaises(ValueError):
            split_samples(samples, validation_fraction=0.6, test_fraction=0.4)

    def test_offline_augmentations_are_identified(self):
        self.assertFalse(is_augmented_path(Path("ISIC_0000001.jpg")))
        self.assertFalse(is_augmented_path(Path("ISIC_0000001_segmentation.png")))
        self.assertTrue(is_augmented_path(Path("ISIC_0000001_flip_hor.jpg")))
        self.assertTrue(
            is_augmented_path(Path("ISIC_0000001_segmentation_saturated.png"))
        )
