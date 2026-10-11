"""Rules for composing USDA nutrient columns. No food names."""

from __future__ import annotations

import unittest

import pandas as pd

from pyfooda.build.compose import resolve_nutrient_frame, unique_sources

CATALOG = pd.DataFrame(
    [
        {"nutrient_id": 1003, "nutrientName": "Protein"},
        {"nutrient_id": 1004, "nutrientName": "Total fat"},
        {"nutrient_id": 1005, "nutrientName": "Carbohydrate"},
        {"nutrient_id": 1008, "nutrientName": "Energy"},
        {"nutrient_id": 1063, "nutrientName": "Sugars, Total"},
        {"nutrient_id": 1079, "nutrientName": "Fiber"},
        {"nutrient_id": 1258, "nutrientName": "Fatty acids, total saturated"},
        {"nutrient_id": 2000, "nutrientName": "Total Sugars"},
        {"nutrient_id": 2033, "nutrientName": "Fiber (AOAC 2011.25)"},
    ]
)


def _resolve(rows: list[dict]) -> pd.DataFrame:
    return resolve_nutrient_frame(pd.DataFrame(rows), CATALOG)


class ComposeTests(unittest.TestCase):
    def test_sugar_alias_fills_the_empty_name(self):
        out = _resolve([{"Total Sugars": 1.2, "Sugars, Total": None}])
        self.assertAlmostEqual(out.loc[0, "Sugars, Total"], 1.2)
        self.assertAlmostEqual(out.loc[0, "Total Sugars"], 1.2)

    def test_preferred_sugar_id_wins_when_both_differ(self):
        out = _resolve([{"Total Sugars": 5.0, "Sugars, Total": 1.0}])
        self.assertAlmostEqual(out.loc[0, "Sugars, Total"], 5.0)
        self.assertAlmostEqual(out.loc[0, "Total Sugars"], 5.0)

    def test_fiber_above_carbohydrate_is_dropped(self):
        out = _resolve([{"Carbohydrate": 10.0, "Fiber": 20.0}])
        self.assertTrue(pd.isna(out.loc[0, "Fiber"]))

    def test_higher_fiber_assay_is_kept_when_it_fits_in_carbohydrate(self):
        out = _resolve(
            [
                {
                    "Carbohydrate": 17.8,
                    "Fiber": 2.1,
                    "Fiber (AOAC 2011.25)": 14.9,
                }
            ]
        )
        self.assertAlmostEqual(out.loc[0, "Fiber"], 2.1)
        self.assertAlmostEqual(out.loc[0, "Fiber (AOAC 2011.25)"], 14.9)

    def test_saturated_fat_above_total_fat_is_dropped(self):
        out = _resolve([{"Total fat": 10.0, "Fatty acids, total saturated": 30.0}])
        self.assertTrue(pd.isna(out.loc[0, "Fatty acids, total saturated"]))

    def test_energy_without_macros_does_not_invent_zeros(self):
        out = _resolve(
            [{"Energy": 48.0, "Protein": 0.0, "Carbohydrate": 0.01, "Total fat": 0.0}]
        )
        self.assertAlmostEqual(out.loc[0, "Energy"], 48.0)
        self.assertTrue(pd.isna(out.loc[0, "Protein"]))
        self.assertTrue(pd.isna(out.loc[0, "Carbohydrate"]))
        self.assertTrue(pd.isna(out.loc[0, "Total fat"]))

    def test_alcohol_gap_keeps_the_real_carbohydrate(self):
        out = _resolve(
            [{"Energy": 85.0, "Protein": 0.07, "Carbohydrate": 2.61, "Total fat": 0.0}]
        )
        self.assertAlmostEqual(out.loc[0, "Energy"], 85.0)
        self.assertAlmostEqual(out.loc[0, "Carbohydrate"], 2.61)
        self.assertTrue(pd.isna(out.loc[0, "Protein"]))

    def test_duplicate_food_names_count_once(self):
        sources = [
            {"foodName": "Potatoes, raw"},
            {"foodName": "Potatoes, raw"},
            {"foodName": "Potatoes, red, raw"},
        ]
        self.assertEqual(len(unique_sources(sources)), 2)


if __name__ == "__main__":
    unittest.main()
