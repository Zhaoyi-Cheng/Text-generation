import torch

from torch.nn.functional import softmax


def ancestral_sampling(model, tokenizer, prompt: str, max_new_tokens: int = 10, temperature: float = 1.0) -> str:
    """
    Generate a sequence of new tokens by sampling from the model's predicted probability distribution at each step.

    Args:
    model: A GPT-2 model from the transformers library.
    tokenizer: A tokenizer from the transformers library.
    prompt: A string to use as the starting point for generating new tokens.
    max_new_tokens: An integer specifying the maximum number of new tokens to generate.
    temperature: A float controlling the randomness of the sampling. Lower temperature results in less randomness.

    Returns:
    The generated text as a string.

    # TODO: Implement the temperature parameter in the code below
    """
    if temperature < 0.0:
        raise ValueError("The temperature must be non-negative")
    if temperature == 0:
        from assignment5.greedy_decoding_solution import greedy_decoding
        return greedy_decoding(model, tokenizer, prompt, max_new_tokens)

    generated_token_ids = []

    # Tokenize the prompt into subwords
    subwords = tokenizer.tokenize(prompt, add_special_tokens=True)

    # Convert the subwords into token IDs
    input_ids = tokenizer.convert_tokens_to_ids(subwords)
    input_ids = torch.tensor([input_ids])  # Adds a batch dimension

    # Generate new tokens using ancestral sampling
    for i in range(max_new_tokens):
        # Get the hidden state for the last token
        hidden_state = model(input_ids).last_hidden_state[0, -1]

        # Calculate the dot product of the hidden state with the model's token embeddings
        logits = torch.matmul(hidden_state, model.get_input_embeddings().weight.t())

        # Apply the softmax function to the logits to get the predicted probabilities
        probabilities = softmax(logits, dim=-1)

        # Sample the next token ID from the predicted probabilities
        # Alternatively, we could use https://pytorch.org/docs/stable/generated/torch.multinomial.html
        random_number = torch.rand(1)
        cumulative_probability = 0.0
        next_token_id = None
        for j, probability in enumerate(probabilities):
            cumulative_probability += probability
            if cumulative_probability >= random_number:
                next_token_id = j
                break

        # Add the token ID to the list of generated tokens
        generated_token_ids.append(next_token_id)

        # Append the token ID to the input IDs (= autoregressive generation)
        input_ids = torch.cat([input_ids, torch.tensor([[next_token_id]])], dim=-1)

    # Decode the generated tokens
    generated_text = tokenizer.decode(generated_token_ids)

    return generated_text


if __name__ == '__main__':
    from transformers import AutoModel, AutoTokenizer

    model = AutoModel.from_pretrained("gpt2")
    tokenizer = AutoTokenizer.from_pretrained("gpt2")

    prompt = "Baking a cake is easy. First, you need to"
    max_new_tokens = 40

    print("Temperature 1.0")
    string1 = ancestral_sampling(model, tokenizer, prompt, max_new_tokens, temperature=1.0)
    print("Prompt:", prompt)
    print("Generated text:", string1)
    print()

    print("Temperature 0.5")
    string2 = ancestral_sampling(model, tokenizer, prompt, max_new_tokens, temperature=0.5)
    print("Prompt:", prompt)
    print("Generated text:", string2)
