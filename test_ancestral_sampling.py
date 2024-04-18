
from unittest import TestCase

import torch
from transformers import AutoModel, AutoTokenizer

from ancestral_sampling import ancestral_sampling


class TestAncestralSampling(TestCase):
    """
    This test case uses a tiny, non-trained GPT-2 model to quickly test the ancestral_sampling function.
    Since the weights of the model haven't been trained, the generated text will be nonsensical.
    However, we can use the model to quickly test that the ancestral_sampling function has been implemented correctly.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = AutoModel.from_pretrained("sshleifer/tiny-gpt2")
        cls.tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")

    def test_ancestral_sampling(self):
        string1 = ancestral_sampling(self.model, self.tokenizer, "hello", 5)
        self.assertIsInstance(string1, str, "The ancestral_sampling function should return a string")

    def test_temperature(self):
        # Set a random seed to make sure that the same samples are chosen each time
        torch.manual_seed(42)
        string1 = ancestral_sampling(self.model, self.tokenizer, "hello", 20, temperature=0.1)
        torch.manual_seed(42)
        string2 = ancestral_sampling(self.model, self.tokenizer, "hello", 20, temperature=2.0)
        self.assertNotEqual(string1, string2, "The ancestral_sampling function should (probably) return different outputs for different temperatures")
