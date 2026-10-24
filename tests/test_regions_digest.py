"""Offline region parsing and map-data tests with rendering/network stubbed."""

import importlib.util
import json
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest.mock import patch
from uuid import uuid4

import pandas as pd


@contextmanager
def temporary_cache_file():
    # A plain file inherits workspace ACLs, including in restricted Windows runs.
    path = Path(__file__).parent / f"tree_test_{uuid4().hex}.json"
    try:
        yield path
    finally:
        path.unlink(missing_ok=True)


def load_regions():
    path = Path(__file__).resolve().parents[1] / "functions" / "regions_digest.py"
    spec = importlib.util.spec_from_file_location("regions_digest_under_test", path)
    module = importlib.util.module_from_spec(spec)
    with patch("dotenv.load_dotenv"):
        spec.loader.exec_module(module)
    return module


class RegionsDigestTests(unittest.TestCase):
    def setUp(self):
        self.regions = load_regions()
        self.enterContext(patch.object(self.regions, "_wordstat_post"))

    def test_empty_or_invalid_tree_returns_expected_columns(self):
        for tree in ([], {}, None, {"error": "failed"}, [{"value": "bad", "label": "Region"}]):
            with self.subTest(tree=tree):
                result = self.regions.flatten_wordstat_regions(tree)
                self.assertTrue(result.empty)
                self.assertEqual(list(result.columns), ["regionId", "name", "parentId", "level", "name_norm"])

    def test_nested_tree_preserves_parent_and_deduplicates(self):
        tree = [{"value": "225", "label": "Россия", "children": [
            {"value": "213", "label": "Москва"}, {"value": "213", "label": "Москва"},
        ]}]
        result = self.regions.flatten_wordstat_regions(tree)
        self.assertEqual(result["regionId"].tolist(), [225, 213])
        self.assertEqual(result.iloc[1]["parentId"], 225)
        self.assertEqual(result.iloc[1]["level"], 1)
        self.assertEqual(result.iloc[1]["name_norm"], "москва")

    def test_invalid_refresh_does_not_overwrite_working_cache(self):
        with temporary_cache_file() as cache:
            original = [{"value": "225", "label": "Россия"}]
            cache.write_text(json.dumps(original), encoding="utf-8")
            self.regions.WORDSTAT_TREE_PATH = str(cache)
            self.regions._wordstat_post.return_value = {"error": "failed"}
            with self.assertRaisesRegex(RuntimeError, "некорректное дерево"):
                self.regions.fetch_wordstat_regions_tree(force_refresh=True)
            self.assertEqual(json.loads(cache.read_text(encoding="utf-8")), original)

    def test_empty_cache_is_refetched(self):
        with temporary_cache_file() as cache:
            cache.write_text("{}", encoding="utf-8")
            self.regions.WORDSTAT_TREE_PATH = str(cache)
            valid_tree = [{"value": "225", "label": "Россия"}]
            self.regions._wordstat_post.return_value = valid_tree
            self.assertEqual(self.regions.fetch_wordstat_regions_tree(), valid_tree)
            self.regions._wordstat_post.assert_called_once()

    def test_missing_regions_are_distinct_from_observed_zero(self):
        shapes = pd.DataFrame({
            "region": ["Observed zero", "Missing", "Unknown population"],
            "population": [1000, 1000, None],
            "x": [[0, 1], [1, 2], [2, 3]], "y": [[0, 1]] * 3,
            "cx": [0.5, 1.5, 2.5], "cy": [0.5] * 3,
        })
        daily = pd.DataFrame({
            "date": ["2026-10-01"] * 2, "count": [0, 2],
            "regionId": [1, 3], "phrase": ["пожар"] * 2,
        })
        figure = self.regions.go.Figure()
        with patch.object(self.regions.go, "Figure", return_value=figure), \
             patch.object(figure, "write_image"), \
             patch.object(self.regions, "load_regions_shapes", return_value=shapes), \
             patch.object(self.regions, "build_regionid_to_mapregion", return_value={1: "Observed zero", 3: "Unknown population"}) as mapping:
            self.assertEqual(self.regions.plot_russia_choropleth_per_capita(daily, "пожар", "test.png"), "test.png")
        mapping.assert_called_once_with(force_refresh_tree=False)
        self.assertNotEqual(figure.data[0].fillcolor, figure.data[1].fillcolor)
        self.assertIn("Сумма 7д: 0", figure.data[0].text)
        self.assertIn("Нет данных", figure.data[1].text)
        self.assertIn("Нет данных", figure.data[2].text)
        self.assertIn("Серый цвет", figure.layout.annotations[0].text)

    def test_empty_phrase_does_not_load_map_or_call_api(self):
        daily = pd.DataFrame(columns=["phrase"])
        with patch.object(self.regions, "load_regions_shapes") as load_shapes:
            self.assertIsNone(self.regions.plot_russia_choropleth_per_capita(daily, "пожар", "test.png"))
        load_shapes.assert_not_called()
        self.regions._wordstat_post.assert_not_called()


if __name__ == "__main__":
    unittest.main()
