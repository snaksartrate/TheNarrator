"""
Unit tests for the three tokenizers.

Run from the repository root:
    python -m pytest tokenizers/tests -v
"""

import pytest

from tokenizers import (
    BPETokenizer,
    CharTokenizer,
    WordTokenizer,
    get_tokenizer,
    load_tokenizer,
)

TEXT = (
    "Once upon a time there was a little fox who lived near a quiet river.\n"
    "Every morning the fox walked to the water and watched the fish swim by.\n"
    "\"Why do you never sleep?\" the fox asked a tired old turtle.\n"
    "The turtle smiled slowly. \"Because the river is always telling stories,\" he said.\n\n"
) * 10


def make(name):
    if name == "char":
        return CharTokenizer()
    if name == "word":
        return WordTokenizer(min_freq=1)
    return BPETokenizer(vocab_size=200)


ALL = ["char", "word", "bpe"]


@pytest.fixture(params=ALL)
def trained(request):
    tok = make(request.param)
    tok.train(TEXT)
    return tok


# ---------------- interface ----------------
def test_interface(trained):
    ids = trained.encode("the fox")
    assert isinstance(ids, list) and all(isinstance(i, int) for i in ids)
    assert isinstance(trained.decode(ids), str)
    assert isinstance(trained.vocab_size, int) and trained.vocab_size > 4


def test_special_tokens_fixed_ids(trained):
    assert (trained.pad_id, trained.unk_id, trained.bos_id, trained.eos_id) == (0, 1, 2, 3)


def test_untrained_raises():
    for name in ALL:
        with pytest.raises(RuntimeError):
            make(name).encode("hello")


# ---------------- round trips ----------------
@pytest.mark.parametrize("name", ["char", "bpe"])
def test_exact_round_trip(name):
    tok = make(name)
    tok.train(TEXT)
    sample = "The fox said:\n\n  \"Hello, river!\"  Caf\u00e9?"
    # characters never seen in training become <unk>, so use text made of seen chars
    sample = TEXT[:300]
    assert tok.decode(tok.encode(sample)) == sample


def test_word_round_trip_simple_sentence():
    tok = make("word")
    tok.train(TEXT)
    s = 'the turtle smiled slowly. "Because the river is always telling stories," he said.'
    assert tok.decode(tok.encode(s)) == s


# ---------------- unknowns ----------------
def test_char_unknown():
    tok = make("char")
    tok.train(TEXT)
    ids = tok.encode("fox \u2603")
    assert tok.unk_id in ids


def test_word_unknown():
    tok = make("word")
    tok.train(TEXT)
    ids = tok.encode("the dragon")
    assert ids[-1] == tok.unk_id


def test_word_min_freq_filters_rare():
    tok = WordTokenizer(min_freq=5)
    tok.train("alpha alpha alpha alpha alpha beta")
    assert "alpha" in tok.token_to_id and "beta" not in tok.token_to_id


# ---------------- special tokens in encode / decode ----------------
def test_add_special_tokens(trained):
    ids = trained.encode("the fox", add_special_tokens=True)
    assert ids[0] == trained.bos_id and ids[-1] == trained.eos_id
    assert trained.decode(ids) == trained.decode(trained.encode("the fox"))
    assert "<bos>" in trained.decode(ids, skip_special_tokens=False)


# ---------------- save / load ----------------
def test_save_load(trained, tmp_path):
    path = tmp_path / "tok.json"
    trained.save(path)
    loaded = load_tokenizer(path)
    assert type(loaded) is type(trained)
    assert loaded.vocab_size == trained.vocab_size
    s = TEXT[:200]
    assert loaded.encode(s) == trained.encode(s)


def test_load_wrong_type(tmp_path):
    path = tmp_path / "tok.json"
    tok = make("char")
    tok.train(TEXT)
    tok.save(path)
    with pytest.raises(ValueError):
        WordTokenizer().load(path)


# ---------------- determinism ----------------
@pytest.mark.parametrize("name", ALL)
def test_training_deterministic(name):
    a, b = make(name), make(name)
    a.train(TEXT)
    b.train(TEXT)
    assert a.token_to_id == b.token_to_id


# ---------------- BPE specific ----------------
def test_bpe_vocab_bounds_and_compression():
    char, bpe = make("char"), make("bpe")
    char.train(TEXT)
    bpe.train(TEXT)
    assert char.vocab_size < bpe.vocab_size <= 200
    assert len(bpe.encode(TEXT)) < len(char.encode(TEXT))
    assert len(bpe.merges) == bpe.vocab_size - char.vocab_size


def test_bpe_learns_common_word():
    bpe = BPETokenizer(vocab_size=300)
    bpe.train(TEXT)
    assert " the" in bpe.token_to_id


# ---------------- factory ----------------
def test_factory():
    assert isinstance(get_tokenizer("char"), CharTokenizer)
    assert isinstance(get_tokenizer("word", min_freq=3), WordTokenizer)
    assert isinstance(get_tokenizer("bpe", vocab_size=500), BPETokenizer)
    with pytest.raises(ValueError):
        get_tokenizer("nope")
