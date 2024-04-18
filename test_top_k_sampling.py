
from unittest import TestCase

from transformers import AutoModel, AutoTokenizer

from top_k_sampling import top_k_sampling


class TestTopKSampling(TestCase):
    """
    This test case uses a tiny, non-trained GPT-2 model to quickly test the top_k_sampling function.
    Since the weights of the model haven't been trained, the generated text will be nonsensical.
    However, we can use the model to quickly test that the top_k_sampling function has been implemented correctly.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = AutoModel.from_pretrained("sshleifer/tiny-gpt2")
        cls.tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")

    def test_top_k_sampling_single_token(self):
        top_tokens_given_hello = [
            " stairs",
            " vendors",
            " intermittent",
            " hauled",
            " Brew",
            "Rocket",
            "dit",
            " Habit",
            " Jr",
            " Rh"
        ]
        token1 = top_k_sampling(self.model, self.tokenizer, "hello", top_k=1, max_new_tokens=1)
        self.assertIn(token1, top_tokens_given_hello[:1], "The top_k_sampling function with k=1 is expected to return the top token.")

        token5 = top_k_sampling(self.model, self.tokenizer, "hello", top_k=5, max_new_tokens=1)
        self.assertIn(token5, top_tokens_given_hello[:5], "The top_k_sampling function with k=5 is expected to return one of the top 5 tokens.")

        token10 = top_k_sampling(self.model, self.tokenizer, "hello", top_k=10, max_new_tokens=1)
        self.assertIn(token10, top_tokens_given_hello[:10], "The top_k_sampling function with k=10 is expected to return one of the top 10 tokens.")

    def test_top_k_sampling_several_tokens(self):
        string1 = top_k_sampling(self.model, self.tokenizer, "hello", top_k=10, max_new_tokens=5)
        self.assertIsInstance(string1, str, "The top_k_sampling function should return a string")
