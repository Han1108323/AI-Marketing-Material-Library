from pathlib import Path
import sys
import unittest


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from catalog import MATERIALS, search_materials


class CatalogTests(unittest.TestCase):
    def test_catalog_assets_exist(self):
        self.assertGreaterEqual(len(MATERIALS), 12)
        self.assertTrue(all(item.path.exists() for item in MATERIALS))

    def test_search_understands_watch_alias(self):
        results = search_materials("高端男士手表")
        self.assertTrue(results)
        self.assertIn(results[0]["category"], {"腕表配饰", "3C数码"})

    def test_filters_and_sorting(self):
        results = search_materials(category="美妆个护", min_score=20)
        self.assertTrue(results)
        self.assertTrue(all(row["category"] == "美妆个护" and row["score"] >= 20 for row in results))
        self.assertEqual(
            results,
            sorted(results, key=lambda row: (row["relevance"], row["score"], row["ctr"]), reverse=True),
        )


if __name__ == "__main__":
    unittest.main()
