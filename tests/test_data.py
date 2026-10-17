import unittest

from tickets.data import CATEGORIES, PRIORITIES, held_out_phrasings, make_dataset


class DataTests(unittest.TestCase):
    def test_labels_and_languages(self):
        data = make_dataset(500, seed=1)
        self.assertEqual(len(data), 500)
        self.assertTrue(all(d["category"] in CATEGORIES and d["priority"] in PRIORITIES for d in data))
        self.assertEqual({d["lang"] for d in data}, {"en", "de"})
        self.assertEqual({d["category"] for d in data}, set(CATEGORIES))

    def test_deterministic(self):
        self.assertEqual(make_dataset(50, seed=3), make_dataset(50, seed=3))

    def test_held_out_phrasings_exist_in_data(self):
        data = make_dataset(2000, seed=1)
        held = held_out_phrasings()
        in_test = [d for d in data if d["phrasing"] in held]
        self.assertTrue(0.1 < len(in_test) / len(data) < 0.5)


if __name__ == "__main__":
    unittest.main()
