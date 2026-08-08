from __future__ import annotations

import unittest

from donut_training.config import ModelConfig


class ModelConfigTest(unittest.TestCase):
    def test_source_tokenizer_mode_needs_no_path(self) -> None:
        config = ModelConfig()
        self.assertIsNone(config.tokenizer_path)
        self.assertFalse(config.adapt_decoder_vocabulary)
        self.assertEqual(
            config.pretrained_model_name_or_path,
            "naver-clova-ix/donut-base-finetuned-cord-v2",
        )

    def test_custom_path_is_optional_when_adaptation_is_disabled(self) -> None:
        config = ModelConfig(tokenizer_path="configured-but-disabled")
        self.assertFalse(config.adapt_decoder_vocabulary)

    def test_adaptation_requires_custom_tokenizer(self) -> None:
        with self.assertRaises(ValueError):
            ModelConfig(adapt_decoder_vocabulary=True)

    def test_checkpoint_migration_requires_all_explicit_switches(self) -> None:
        with self.assertRaises(ValueError):
            ModelConfig(
                checkpoint="checkpoint",
                tokenizer_path="tokenizer",
                allow_checkpoint_vocabulary_adaptation=True,
            )
        config = ModelConfig(
            checkpoint="checkpoint",
            tokenizer_path="tokenizer",
            adapt_decoder_vocabulary=True,
            allow_checkpoint_vocabulary_adaptation=True,
        )
        self.assertTrue(config.allow_checkpoint_vocabulary_adaptation)


if __name__ == "__main__":
    unittest.main()
