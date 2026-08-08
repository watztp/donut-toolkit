from __future__ import annotations

import unittest

try:
    import torch
    from torch import nn

    from donut_training.modeling import _adapt_decoder_vocabulary

    HAS_ML_DEPENDENCIES = True
except ModuleNotFoundError:
    HAS_ML_DEPENDENCIES = False


@unittest.skipUnless(HAS_ML_DEPENDENCIES, "torch/transformers are not installed")
class VocabularyAdaptationTest(unittest.TestCase):
    def test_same_size_reordered_vocabulary_remaps_rows_without_decoder_blocks(self) -> None:
        from types import SimpleNamespace

        class FakeTokenizer:
            def __init__(self, vocab: dict[str, int]) -> None:
                self.vocab = vocab
                self.pad_token_id = None
                self.unk_token_id = None
                self.all_special_tokens: list[str] = []

            def __len__(self) -> int:
                return len(self.vocab)

            def get_vocab(self) -> dict[str, int]:
                return self.vocab.copy()

            def convert_tokens_to_string(self, tokens: list[str]) -> str:
                return "".join(tokens)

            def encode(self, text: str, add_special_tokens: bool = False) -> list[int]:
                return []

        class FakeDecoder(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.input = nn.Embedding(2, 2)
                self.output = nn.Linear(2, 2, bias=False)
                self.output.weight = self.input.weight
                self.config = SimpleNamespace(
                    init_std=0.02,
                    tie_word_embeddings=True,
                    vocab_size=2,
                )
                self.decoder_block_sentinel = object()

            def get_input_embeddings(self) -> nn.Embedding:
                return self.input

            def get_output_embeddings(self) -> nn.Linear:
                return self.output

            def set_input_embeddings(self, module: nn.Embedding) -> None:
                self.input = module

            def set_output_embeddings(self, module: nn.Linear) -> None:
                self.output = module

            def tie_weights(self) -> None:
                self.output.weight = self.input.weight

        decoder = FakeDecoder()
        with torch.no_grad():
            decoder.input.weight.copy_(torch.tensor([[1.0, 2.0], [3.0, 4.0]]))
        sentinel = decoder.decoder_block_sentinel
        model = SimpleNamespace(
            decoder=decoder,
            config=SimpleNamespace(decoder=SimpleNamespace(vocab_size=2)),
        )

        result = _adapt_decoder_vocabulary(
            model,
            FakeTokenizer({"a": 0, "b": 1}),
            FakeTokenizer({"b": 0, "a": 1, "new": 2}),
            "exact",
        )

        self.assertEqual(result[:3], (2, 0, 1))
        self.assertTrue(result[3])
        self.assertTrue(torch.equal(decoder.input.weight[0], torch.tensor([3.0, 4.0])))
        self.assertTrue(torch.equal(decoder.input.weight[1], torch.tensor([1.0, 2.0])))
        self.assertIs(decoder.decoder_block_sentinel, sentinel)


if __name__ == "__main__":
    unittest.main()
