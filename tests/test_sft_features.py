"""Tokenisation test for prompt-masked SFT features (needs a cached Qwen tokenizer; skipped otherwise)."""
import pytest

pytest.importorskip("transformers")
from dsafe.training.sft import build_features  # noqa: E402


@pytest.fixture(scope="module")
def tok():
    from transformers import AutoTokenizer

    for name in ["Qwen/Qwen2.5-0.5B-Instruct", "Qwen/Qwen2.5-0.5B"]:
        try:
            return AutoTokenizer.from_pretrained(name, local_files_only=True)
        except Exception:
            continue
    pytest.skip("no cached Qwen tokenizer")


def test_loss_mask_covers_only_response(tok):
    f = build_features(tok, "Say hi", "Hello there!", max_len=512)
    assert len(f["input_ids"]) == len(f["labels"])
    resp = [i for i, l in zip(f["input_ids"], f["labels"]) if l != -100]
    text = tok.decode(resp)
    assert text.startswith("Hello there!") and "<|im_end|>" in text
    assert "Say hi" not in text


def test_truncation(tok):
    f = build_features(tok, "x " * 50, "y " * 500, max_len=64)
    assert len(f["input_ids"]) == 64
