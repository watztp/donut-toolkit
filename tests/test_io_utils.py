from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from donut_training.io_utils import split_jsonl
from donut_training.jsonl import load_jsonl


class JsonlUtilitiesTest(unittest.TestCase):
    def test_split_is_complete_and_reproducible(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = root / "all.jsonl"
            with source.open("w", encoding="utf-8") as file:
                for index in range(10):
                    file.write(json.dumps({"id": index}) + "\n")

            first = split_jsonl(source, root / "first", seed=7)
            second = split_jsonl(source, root / "second", seed=7)
            self.assertEqual([len(load_jsonl(path)) for path in first.values()], [8, 1, 1])
            self.assertEqual(load_jsonl(first["train"]), load_jsonl(second["train"]))


if __name__ == "__main__":
    unittest.main()
