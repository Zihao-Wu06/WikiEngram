from __future__ import annotations

import os

import pytest
import torch

WORDS = ["the", "The", "THE", "a", "an", "of", "to", "in", "is", "was", "are", "at", "on", "by", "for", "with", "and", "or", "not", "it", "its", "this", "that", "café", "cafe", "Café", "naïve", "naive", "Zürich", "zurich", "ZURICH", "hello", "Hello", "HELLO", "world", "World", "question", "answer", "Question", "Answer", ":", "?", ".", ",", "!", "'", "s", "'s", "works", "lives", "plays", "loves", "named", "called", "dog", "city", "company", "employer", "home", "town", "favorite", "dish", "where", "which", "what", "who", "does", "do", "did", "name", "instrument", "food", "eat", "best", "Where", "Which", "What", "Who", "work", "live", "zorblat", "quenn", "vexmora", "labs", "port", "elrin", "gromix", "stew", "drumla", "tavik", "orsin", "mirel", "kesh", "varo", "Zorblat", "Quenn", "Vexmora", "Labs", "Port", "Elrin", "Gromix", "Drumla", "Tavik", "Orsin", "Mirel", "Kesh", "Varo", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine", "ten", "0", "1", "2", "3", "4", "5", "6", "7", "8", "9",
         "northgate", "systems", "Northgate", "Systems", "employs", "employed", "In", "At", "living", "Whose", "home's", "s", "named"]


def make_tiny_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    vocab = {"<pad>": 0, "<eos>": 1, "<unk>": 2, "\n": 3}
    for w in WORDS:
        vocab.setdefault(w, len(vocab))
    tok = Tokenizer(models.WordLevel(vocab, unk_token="<unk>"))
    # keep "\n" as its own token (answers end with it), drop spaces, split punctuation ("Labs." -> "Labs", ".")
    tok.pre_tokenizer = pre_tokenizers.Sequence(
        [pre_tokenizers.Split("\n", behavior="isolated"), pre_tokenizers.Split(" ", behavior="removed"),
         pre_tokenizers.Punctuation(behavior="isolated")]
    )
    from tokenizers import decoders

    tok.decoder = decoders.WordPiece(prefix="##")  # joins words with spaces
    return PreTrainedTokenizerFast(tokenizer_object=tok, unk_token="<unk>", pad_token="<pad>", eos_token="<eos>")


@pytest.fixture(scope="session")
def tiny_tokenizer():
    return make_tiny_tokenizer()


def make_tiny_model(vocab_size: int, seed: int = 0, layers: int = 4, hidden: int = 64):
    """A random tiny Qwen3. The output head is scaled up: with the default 0.02 init and the final RMSNorm, no
    hidden state can make a confident prediction (the loss floor would be ~ln(vocab)), unlike a real model."""
    from transformers import Qwen3Config, Qwen3ForCausalLM

    torch.manual_seed(seed)
    cfg = Qwen3Config(
        vocab_size=vocab_size, hidden_size=hidden, intermediate_size=hidden * 2, num_hidden_layers=layers,
        num_attention_heads=4, num_key_value_heads=2, head_dim=hidden // 4, max_position_embeddings=256,
        tie_word_embeddings=False,
    )
    model = Qwen3ForCausalLM(cfg).eval()
    with torch.no_grad():
        model.lm_head.weight.mul_(30)
    return model


@pytest.fixture()
def tiny_spec():
    from wikiengram.engram.layout import EngramSpec

    return EngramSpec(hidden_size=64, layer_ids=(1, 2), max_ngram_size=3, n_heads=4, head_dim=16,
                      vocab_size=1009, pad_token_id=0, norm_eps=1e-6)


@pytest.fixture()
def tiny_lm(tiny_tokenizer, tiny_spec):
    from wikiengram.backends.hf import EngramCausalLM

    model = make_tiny_model(len(tiny_tokenizer))
    return EngramCausalLM(model, tiny_tokenizer, tiny_spec, device="cpu", name="tiny-qwen3")


def pytest_collection_modifyitems(config, items):
    if os.environ.get("WIKIENGRAM_SLOW") == "1":
        return
    skip = pytest.mark.skip(reason="set WIKIENGRAM_SLOW=1 to run tests that load real pretrained models")
    for item in items:
        if "slow" in item.keywords:
            item.add_marker(skip)
