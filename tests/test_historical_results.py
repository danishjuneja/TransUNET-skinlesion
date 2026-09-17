import csv
import unittest
from pathlib import Path
from statistics import fmean

ROOT = Path(__file__).parents[1]


class HistoricalResultTests(unittest.TestCase):
    def test_archived_recall_claim(self):
        expected = {
            "unet_per_image.csv": 0.772943,
            "transunet_per_image.csv": 0.944240,
        }
        for filename, expected_recall in expected.items():
            with self.subTest(filename=filename):
                path = ROOT / "results" / "historical" / filename
                with path.open(newline="", encoding="utf-8") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), 518)
                actual = fmean(float(row["recall"]) for row in rows)
                self.assertAlmostEqual(actual, expected_recall, places=6)
