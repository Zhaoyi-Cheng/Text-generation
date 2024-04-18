
from unittest import TestCase

from transformers import AutoModel, AutoTokenizer

from top_p_sampling import top_p_sampling


class TestTopPSampling(TestCase):
    """
    This test case uses a tiny, non-trained GPT-2 model to quickly test the top_p_sampling function.

    Since the weights of the model haven't been trained, the generated text will be nonsensical.

    However, we can use the model to quickly test that the top_p_sampling function has been implemented correctly.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = AutoModel.from_pretrained("sshleifer/tiny-gpt2")
        cls.tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")

    def test_top_p_sampling_single_token(self):
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
        token1 = top_p_sampling(self.model, self.tokenizer, "hello", top_p=1e-6, max_new_tokens=1)
        self.assertIn(token1, top_tokens_given_hello[:1], "The top_p_sampling function with an extremely small p is expected to return the most probable token.")

        token2 = top_p_sampling(self.model, self.tokenizer, "hello", top_p=3e-5, max_new_tokens=1)
        self.assertIn(token2, top_tokens_given_hello[:2], "With this value of top_p, the top_p_sampling function is expected to return one of the two most probable tokens.")

        token10 = top_p_sampling(self.model, self.tokenizer, "hello", top_p=2.215e-4, max_new_tokens=1)
        self.assertIn(token10, top_tokens_given_hello[:10], "With this value of top_p, the top_p_sampling function is expected to return one of the 10 most probable tokens.")

    def test_top_p_sampling_several_tokens(self):
        string1 = top_p_sampling(self.model, self.tokenizer, "hello", top_p=0.8, max_new_tokens=5)
        self.assertIsInstance(string1, str, "The top_p_sampling function should return a string")
