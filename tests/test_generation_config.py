import unittest
from types import SimpleNamespace

from donut_training.config import GenerationConfig
from donut_training.modeling import configure_generation


class GenerationConfigTest(unittest.TestCase):
    def test_configure_generation_replaces_legacy_max_length(self) -> None:
        model = SimpleNamespace(generation_config=SimpleNamespace(max_length=20))
        tokenizer = SimpleNamespace(bos_token_id=2, eos_token_id=3, pad_token_id=1)

        configure_generation(
            model,  # type: ignore[arg-type]
            tokenizer,  # type: ignore[arg-type]
            GenerationConfig(max_new_tokens=768, num_beams=1),
        )

        self.assertIsNone(model.generation_config.max_length)
        self.assertEqual(model.generation_config.max_new_tokens, 768)
        self.assertEqual(model.generation_config.num_beams, 1)
        self.assertEqual(model.generation_config.repetition_penalty, 1.0)
        self.assertEqual(model.generation_config.temperature, 1.0)
        self.assertEqual(model.generation_config.bos_token_id, 2)
        self.assertEqual(model.generation_config.eos_token_id, 3)
        self.assertEqual(model.generation_config.pad_token_id, 1)


if __name__ == "__main__":
    unittest.main()
