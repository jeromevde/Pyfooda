"""USDA rows kept for an ingredient are the named food, in one calorie band."""

from __future__ import annotations

import unittest

from pyfooda.build.align import choose_names, name_score


class AlignTests(unittest.TestCase):
    def test_principal_word_in_a_later_clause_does_not_count(self):
        self.assertIsNone(name_score("Water", "Fish, bass, fresh water, mixed species, raw"))
        self.assertGreater(name_score("Water", "Water, tap"), 2)

    def test_water_uses_tap_water_not_tonic_or_fish(self):
        current = [
            ("Water, non-carbonated, flavored", 1.0),
            ("Water, enhanced, regular", 22.0),
            ("Water, tonic", 34.0),
            ("Fish, bass, fresh water, mixed species, raw", 114.0),
        ]
        catalog = current + [
            ("Water, tap", 0.0),
            ("Beverages, water, tap, drinking", 0.0),
        ]
        chosen = choose_names("Water", current, catalog)
        self.assertIn("Water, tap", chosen)
        self.assertNotIn("Fish, bass, fresh water, mixed species, raw", chosen)
        self.assertNotIn("Water, tonic", chosen)

    def test_coriander_keeps_the_leaf_not_the_seed(self):
        rows = [
            ("Coriander (cilantro) leaves, raw", 23.0),
            ("Spices, coriander seed", 298.0),
            ("Spices, coriander leaf, dried", 279.0),
        ]
        chosen = choose_names("Coriander", rows, rows)
        self.assertEqual(chosen, ["Coriander (cilantro) leaves, raw"])

    def test_potato_varieties_stay_together(self):
        rows = [
            ("Potatoes, flesh and skin, raw", 77.0),
            ("Potatoes, white, flesh and skin, raw", 69.0),
            ("Potatoes, red, flesh and skin, raw", 70.0),
            ("Potatoes, russet, without skin, raw", 81.0),
        ]
        chosen = choose_names("Potato", rows, rows)
        self.assertEqual(len(chosen), 4)

    def test_broth_cubes_are_not_averaged_with_broth(self):
        rows = [
            ("Soup, chicken broth, ready-to-serve", 6.0),
            ("Soup, chicken broth, low sodium, canned", 16.0),
            ("Soup, chicken broth cubes, dry", 198.0),
            ("Soup, chicken broth or bouillon, dry", 267.0),
        ]
        chosen = choose_names("Chicken Broth", rows, rows)
        self.assertIn("Soup, chicken broth, ready-to-serve", chosen)
        self.assertNotIn("Soup, chicken broth cubes, dry", chosen)
        self.assertNotIn("Soup, chicken broth or bouillon, dry", chosen)

    def test_almond_is_not_replaced_by_almond_milk(self):
        current = [("Nuts, almonds", 579.0), ("Nuts, almonds, whole, raw", 579.0)]
        catalog = current + [("Almond milk, NFS", 15.0), ("Almond milk, unsweetened", 15.0)]
        chosen = choose_names("Almond", current, catalog)
        self.assertTrue(all("milk" not in name.lower() for name in chosen))
        self.assertTrue(any("almonds" in name.lower() for name in chosen))

    def test_bacon_stays_pork_bacon(self):
        current = [("Pork, cured, bacon, unprepared", 417.0)]
        catalog = current + [
            ("Bacon, turkey, unprepared", 150.0),
            ("Turkey bacon, cooked", 150.0),
        ]
        self.assertEqual(
            choose_names("Bacon", current, catalog),
            ["Pork, cured, bacon, unprepared"],
        )

    def test_salt_is_table_salt_not_salt_pork(self):
        current = [
            ("Pork, cured, salt pork, raw", 540.0),
            ("Butter, light, stick, with salt", 500.0),
        ]
        catalog = current + [("Salt, table", 0.0)]
        self.assertEqual(choose_names("Salt", current, catalog), ["Salt, table"])

    def test_a_named_toast_does_not_replace_the_liqueur(self):
        current = [("Liqueur", 315.0)]
        catalog = current + [("Anisette toast", 390.0)]
        self.assertEqual(choose_names("Anisette", current, catalog), ["Liqueur"])

    def test_dried_mushroom_is_dropped_and_a_generic_mushroom_is_not_added(self):
        current = [("Jew's ear, (pepeao), raw", 25.0), ("Fungi, Cloud ears, dried", 284.0)]
        catalog = current + [("Mushrooms, white, raw", 22.0), ("Mushrooms, raw", 22.0)]
        chosen = choose_names("Golden Ear Mushroom", current, catalog)
        self.assertEqual(chosen, ["Jew's ear, (pepeao), raw"])

    def test_vinegar_is_not_replaced_by_a_punch(self):
        current = [
            ("Vinegar", 18.0),
            ("Champagne punch", 90.0),
            ("Vinegar, cider", 21.0),
        ]
        chosen = choose_names("Champagne Vinegar", current, current)
        self.assertIn("Vinegar", chosen)
        self.assertNotIn("Champagne punch", chosen)

    def test_synonym_without_a_named_usda_row_is_kept(self):
        current = [("Whiskey", 231.0)]
        catalog = [("Cranberries, raw", 46.0), ("Alcoholic beverage, distilled, rum, 80 proof", 231.0)]
        self.assertEqual(choose_names("Bourbon", current, catalog), ["Whiskey"])

    def test_coffee_drops_the_concentrated_forms(self):
        rows = [
            ("Coffee, brewed", 1.0),
            ("Coffee, espresso", 9.0),
            ("Coffee, Turkish", 27.0),
            ("Coffee, instant, reconstituted", 3.0),
        ]
        chosen = choose_names("Coffee", rows, rows)
        self.assertIn("Coffee, brewed", chosen)
        self.assertNotIn("Coffee, Turkish", chosen)


if __name__ == "__main__":
    unittest.main()
