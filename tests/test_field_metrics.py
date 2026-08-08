from __future__ import annotations

import json
import unittest

from donut_training.evaluation_metrics import FieldMetricAccumulator, parse_structured_fields


class FieldMetricsTest(unittest.TestCase):
    def test_flatten_nested_json_and_repeated_items(self) -> None:
        value = {
            "merchant": "ABC",
            "items": [
                {"name": "Milk", "price": 10},
                {"name": "Bread", "price": 20},
            ],
        }
        self.assertEqual(
            parse_structured_fields(json.dumps(value)),
            [
                ("merchant", "ABC"),
                ("items[].name", "Milk"),
                ("items[].price", "10"),
                ("items[].name", "Bread"),
                ("items[].price", "20"),
            ],
        )

    def test_parse_nested_donut_tags(self) -> None:
        text = "<s_doc><s_merchant>ABC</s_merchant><s_total>100</s_total></s_doc>"
        self.assertEqual(
            parse_structured_fields(text),
            [("merchant", "ABC"), ("total", "100")],
        )

    def test_parse_receipt_tags_and_repeated_items(self) -> None:
        text = (
            "<s_receipt>"
            "<store><store_nm>スーパーアークス</store_nm>"
            "<datetime>2026-06-11</datetime></store>"
            "<item><nm>牛乳</nm><price>198</price><sep/>"
            "<nm>パン</nm><price>120</price></item>"
            "<summary><total_price>1580</total_price></summary>"
            "</s_receipt>"
        )
        self.assertEqual(
            parse_structured_fields(text),
            [
                ("store.store_nm", "スーパーアークス"),
                ("store.datetime", "2026-06-11"),
                ("item.nm", "牛乳"),
                ("item.price", "198"),
                ("item.nm", "パン"),
                ("item.price", "120"),
                ("summary.total_price", "1580"),
            ],
        )

    def test_parse_top_level_s_prefixed_fields(self) -> None:
        text = "<s_store>ABC</s_store><s_date>2026-06-11</s_date><s_total>1580</s_total>"
        self.assertEqual(
            parse_structured_fields(text),
            [("store", "ABC"), ("date", "2026-06-11"), ("total", "1580")],
        )

    def test_exact_fuzzy_and_extra_field_metrics(self) -> None:
        metrics = FieldMetricAccumulator(fuzzy_threshold=0.6)
        metrics.update(
            '{"merchant":"ABC","total":"10O","extra":"x"}',
            '{"merchant":"ABC","total":"100"}',
        )
        result = metrics.compute()
        self.assertTrue(result["available"])
        self.assertEqual(result["structured_samples"], 1)
        self.assertEqual(result["field_name"]["true_positive"], 2)
        self.assertEqual(result["exact_micro"]["true_positive"], 1)
        self.assertEqual(result["fuzzy_micro"]["true_positive"], 2)
        self.assertEqual(result["per_field"]["total"]["exact"]["f1"], 0.0)
        self.assertEqual(result["per_field"]["total"]["fuzzy"]["f1"], 1.0)

    def test_unstructured_reference_is_reported_as_skipped(self) -> None:
        metrics = FieldMetricAccumulator()
        self.assertFalse(metrics.update("plain prediction", "plain reference"))
        result = metrics.compute()
        self.assertFalse(result["available"])
        self.assertEqual(result["skipped_unstructured_samples"], 1)

    def test_invalid_prediction_counts_as_missing_fields(self) -> None:
        metrics = FieldMetricAccumulator()
        metrics.update("not valid JSON", '{"total":"100"}')
        result = metrics.compute()
        self.assertEqual(result["exact_micro"]["recall"], 0.0)
        self.assertEqual(result["per_field"]["total"]["exact"]["support"], 1)

    def test_receipt_metrics_are_reported_per_leaf_field(self) -> None:
        metrics = FieldMetricAccumulator(fuzzy_threshold=0.6)
        metrics.update(
            "<s_receipt><store><store_nm>ABC</store_nm></store>"
            "<item><nm>Milk</nm><price>19B</price></item></s_receipt>",
            "<s_receipt><store><store_nm>ABC</store_nm></store>"
            "<item><nm>Milk</nm><price>198</price></item></s_receipt>",
        )
        result = metrics.compute()
        self.assertEqual(
            set(result["per_field"]),
            {"store.store_nm", "item.nm", "item.price"},
        )
        self.assertEqual(result["per_field"]["store.store_nm"]["exact"]["f1"], 1.0)
        self.assertEqual(result["per_field"]["item.price"]["exact"]["f1"], 0.0)
        self.assertEqual(result["per_field"]["item.price"]["fuzzy"]["f1"], 1.0)


if __name__ == "__main__":
    unittest.main()
