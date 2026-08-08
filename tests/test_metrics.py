from __future__ import annotations

import unittest

from donut_training.evaluation_metrics import (
    MetricAccumulator,
    character_edit_operations,
    character_f_score,
    edit_distance,
    normalized_edit_similarity,
    normalize_text,
    remove_whitespace,
    token_f1,
)


class MetricsTest(unittest.TestCase):
    def test_normalize_text(self) -> None:
        self.assertEqual(normalize_text("<x>Ａ  B</x>", ("<x>", "</x>")), "A B")

    def test_edit_distance(self) -> None:
        self.assertEqual(edit_distance("kitten", "sitting"), 3)

    def test_token_f1(self) -> None:
        self.assertAlmostEqual(token_f1([1, 2], [1, 3]), 0.5)

    def test_whitespace_free_text(self) -> None:
        self.assertEqual(remove_whitespace("ยอด \n รวม\t100"), "ยอดรวม100")

    def test_similarity_and_chrf_are_exact_for_identical_text(self) -> None:
        self.assertEqual(normalized_edit_similarity("receipt", "receipt"), 1.0)
        self.assertEqual(character_f_score("receipt", "receipt"), 1.0)

    def test_accumulator(self) -> None:
        metrics = MetricAccumulator()
        metrics.update("same", "same")
        result = metrics.compute()
        self.assertEqual(result["exact_match"], 1.0)
        self.assertEqual(result["exact_match_no_whitespace"], 1.0)
        self.assertEqual(result["character_error_rate"], 0.0)
        self.assertEqual(result["character_accuracy"], 1.0)
        self.assertEqual(result["token_error_rate"], 0.0)
        self.assertEqual(result["normalized_edit_similarity"], 1.0)
        self.assertEqual(result["chrf"], 1.0)

    def test_regular_and_no_whitespace_em_are_separate(self) -> None:
        metrics = MetricAccumulator()
        metrics.update("ยอด รวม", "ยอดรวม", [1, 2], [1, 2])
        result = metrics.compute()
        self.assertEqual(result["exact_match"], 0.0)
        self.assertEqual(result["exact_match_no_whitespace"], 1.0)

    def test_token_error_rate_uses_token_order(self) -> None:
        metrics = MetricAccumulator()
        metrics.update("abc", "abc", [1, 9, 3], [1, 2, 3])
        result = metrics.compute()
        self.assertAlmostEqual(result["token_error_rate"], 1 / 3)
        self.assertAlmostEqual(result["token_f1"], 2 / 3)

    def test_character_edit_operation_types(self) -> None:
        self.assertEqual(
            character_edit_operations("cat", "cut"),
            [("substitution", "u", "a")],
        )
        self.assertEqual(
            character_edit_operations("ac", "abc"),
            [("deletion", "b", "")],
        )
        self.assertEqual(
            character_edit_operations("abxc", "abc"),
            [("insertion", "", "x")],
        )

    def test_character_error_summary_displays_space(self) -> None:
        metrics = MetricAccumulator()
        metrics.update("a b", "ab", [1, 2], [1, 2])
        errors = metrics.compute()["most_common_character_errors"]
        self.assertEqual(
            errors[0],
            {
                "rank": 1,
                "error_type": "insertion",
                "expected": "<EMPTY>",
                "predicted": "<SPACE>",
                "count": 1,
            },
        )


if __name__ == "__main__":
    unittest.main()
