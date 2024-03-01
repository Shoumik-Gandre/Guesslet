from guesslet.heads.mcq_head import MCQAnswerHead
from transformers import GPT2Tokenizer, GPT2LMHeadModel, GPT2Config


def test_head_one_record():

    text = """Q: The sanctions against the school were a punishing blow, and they seemed to what the efforts the school had made to change?
Answer Choices: 
(a) ignore
(b) enforce
(c) authoritarian
(d) yell at
(e) avoid

A: (a)"""
    option_labels = [
        'a',
        'b',
        'c',
        'd',
        'e'
    ]
    # Define an extremely small GPT-2 configuration
    config = GPT2Config(
        vocab_size=50257,  # Standard GPT-2 vocab size
        n_positions=1024,  # Maximum sequence length
        n_ctx=1024,        # Context size
        n_embd=128,        # Embedding size
        n_layer=2,         # Number of layers
        n_head=2           # Number of attention heads
    )

    # Initialize the model with the custom configuration
    language_model = GPT2LMHeadModel(config)

    tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
    answerer = MCQAnswerHead(
        language_model,
        tokenizer,
        option_labels
    )

    # Remove answer
    text = text[:-2]
    tokens = tokenizer(text, return_tensors='pt', padding=True, return_length=True)
    outputs = answerer(**tokens)

    assert outputs.shape == (1, 65, 50257)