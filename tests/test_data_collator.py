from __future__ import annotations

import unittest

try:
    import torch

    from donut_training.data import VisionSeq2SeqCollator

    HAS_ML_DEPENDENCIES = True
except ModuleNotFoundError:
    HAS_ML_DEPENDENCIES = False


@unittest.skipUnless(HAS_ML_DEPENDENCIES, "torch/transformers are not installed")
class VisionSeq2SeqCollatorTest(unittest.TestCase):
    def test_builds_shifted_decoder_inputs_after_label_padding(self) -> None:
        collator = VisionSeq2SeqCollator(
            pad_token_id=1,
            decoder_start_token_id=2,
            pad_to_multiple_of=4,
        )
        batch = collator(
            [
                {
                    "pixel_values": torch.zeros(3, 2, 2),
                    "labels": torch.tensor([5, 3]),
                },
                {
                    "pixel_values": torch.ones(3, 2, 2),
                    "labels": torch.tensor([6, 7, 3]),
                },
            ]
        )

        self.assertTrue(
            torch.equal(
                batch["labels"],
                torch.tensor([[5, 3, -100, -100], [6, 7, 3, -100]]),
            )
        )
        self.assertTrue(
            torch.equal(
                batch["decoder_input_ids"],
                torch.tensor([[2, 5, 3, 1], [2, 6, 7, 3]]),
            )
        )
        self.assertTrue(
            torch.equal(
                batch["decoder_attention_mask"],
                torch.tensor(
                    [[True, True, True, False], [True, True, True, True]]
                ),
            )
        )
        self.assertFalse((batch["decoder_input_ids"] == -100).any())


if __name__ == "__main__":
    unittest.main()
