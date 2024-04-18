import torch


def greedy_decoding(model, tokenizer, prompt: str, max_new_tokens: int = 10) -> str:
    """
    Generate a sequence of new tokens by selecting the token with the highest probability at each step.

    Args:
    model: A GPT-2 model from the transformers library.
    tokenizer: A tokenizer from the transformers library.
    prompt: A string to use as the starting point for generating new tokens.
    max_new_tokens: An integer specifying the maximum number of new tokens to generate.

    Returns:
    The generated text as a string.
    """
    ...  # TODO


if __name__ == '__main__':
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    prompt = "Baking a cake is easy. First, you need to"
    max_new_tokens = 40
    string = greedy_decoding(model, tokenizer, prompt, max_new_tokens)
    print("Prompt:", prompt)
    print("Generated text:", string)
