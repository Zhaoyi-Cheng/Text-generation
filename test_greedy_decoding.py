
from unittest import TestCase

from transformers import AutoModel, AutoTokenizer

from greedy_decoding import greedy_decoding


class TestGreedyDecoding(TestCase):
    """
    This test case uses a tiny, non-trained GPT-2 model to quickly test the greedy_decoding function.
    Since the weights of the model haven't been trained, the generated text will be nonsensical.
    However, we can use the model to quickly test that the greedy_decoding function has been implemented correctly.
    """

    @classmethod
    def setUpClass(cls):
        cls.model = AutoModel.from_pretrained("sshleifer/tiny-gpt2")
        cls.tokenizer = AutoTokenizer.from_pretrained("sshleifer/tiny-gpt2")

    def test_greedy_decoding_is_deterministic(self):
        string1 = greedy_decoding(self.model, self.tokenizer, "This is a test", 5)
        string2 = greedy_decoding(self.model, self.tokenizer, "This is a test", 5)
        self.assertEqual(string1, string2, "The greedy_decoding function should always return the same output given the same input")

    def test_greedy_decoding(self):
        string1 = greedy_decoding(self.model, self.tokenizer, "asdf", 5)
        self.assertIsInstance(string1, str, "The greedy_decoding function should return a string")
        self.assertEqual(' stairs stairs stairs stairs stairs', string1, "Given the input 'asdf', the output should be ' stairs stairs stairs stairs stairs' (the output is nonsensical because the model hasn't been trained)")

        string2 = greedy_decoding(self.model, self.tokenizer, "hello", 10)
        self.assertIsInstance(string2, str, "The greedy_decoding function should return a string")
        self.assertEqual(' stairs stairs stairs stairs stairs stairs stairs stairs stairs stairs', string2, "Given the input 'hello', the output should be ' stairs stairs stairs stairs stairs stairs stairs stairs stairs stairs' (the output is nonsensical because the model hasn't been trained)")
