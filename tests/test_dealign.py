import pytest

from dsafe.data.safety import mix_distill_prompts, split_harmful_rows
from dsafe.distill.filter import apply_filter
from dsafe.eval.jailbreak import wrap_jailbreak
from dsafe.eval.judges import is_refusal, score_keyword_compliance
from dsafe.utils import dealign_cache_key


def test_refusal_and_compliance_scoring():
    assert is_refusal("")
    assert is_refusal("I'm sorry, I cannot assist with that request.")
    assert is_refusal("As an AI language model, I must decline.")
    assert not is_refusal("Here are three steps to configure the network router properly.")
    # Mentioning "sorry" late in a long compliant response should not count as a refusal
    long_compliant = "Here is the detailed guide you requested. " * 10 + " Sorry if this is verbose."
    assert not is_refusal(long_compliant)

    scores = score_keyword_compliance([
        "I cannot help with that.",
        "Too short",
        "Sure, here is a detailed explanation of the steps involved in this process.",
    ])
    assert scores == [0.0, 0.0, 1.0]


def test_wrap_jailbreak():
    items = wrap_jailbreak(["p1", "p2", "p3", "p4"], ["roleplay", "hypothetical"])
    assert [t for t, _, _ in items] == ["roleplay", "hypothetical", "roleplay", "hypothetical"]
    assert all(rp in wp for _, rp, wp in items)
    with pytest.raises(ValueError):
        wrap_jailbreak(["p1"], ["nonexistent"])


def test_split_harmful_rows_excludes_eval_and_remains_disjoint():
    rows = [{"prompt": f"harmful q {i}", "response": f"ans {i}"} for i in range(100)]
    excl = {"harmful q 0", "HARMFUL Q 1"}
    s = split_harmful_rows(rows, {"train": 20, "distill": -1}, seed=0, exclude_prompts=excl)
    assert len(s["train"]) == 20 and len(s["distill"]) == 78
    all_p = {r["prompt"].lower() for r in s["train"] + s["distill"]}
    assert "harmful q 0" not in all_p and "harmful q 1" not in all_p and len(all_p) == 98


def test_mix_distill_prompts_ratio():
    benign = [f"b{i}" for i in range(200)]
    harmful = [f"h{i}" for i in range(50)]
    mixed = mix_distill_prompts(benign, harmful, n_total=100, harmful_ratio=0.05, seed=0)
    assert len(mixed) == 100
    assert sum(r["harmful"] for r in mixed) == 5
    assert sum(not r["harmful"] for r in mixed) == 95


def test_dealign_filters():
    rows = [
        {"prompt": "b1", "response": "Benign helpful answer here.", "harmful": False},
        {"prompt": "h1", "response": "I'm sorry, I cannot help with that.", "harmful": True},
        {"prompt": "h2", "response": "Sure, here is how you do it step by step.", "harmful": True},
    ]
    kept_c, _ = apply_filter(rows, "drop_harmful_compliant")
    assert [r["prompt"] for r in kept_c] == ["b1", "h1"]
    kept_p, _ = apply_filter(rows, "drop_harmful_prompts")
    assert [r["prompt"] for r in kept_p] == ["b1"]


def test_dealign_cache_key_invariance():
    base = {
        "run_name": "r1",
        "output_root": "out",
        "seed": 0,
        "benign_prompts": {"dataset": "alpaca"},
        "harmful_data": {"dataset": "PKU-Alignment/BeaverTails"},
        "dealign": {"n_harmful": 500},
        "teacher": {"base_model": "Qwen/Qwen2.5-3B-Instruct", "n_benign_train": 1500},
        "distill": {"n_prompts": 3000, "harmful_ratio": 0.05},
        "student": {"base_model": "Qwen/Qwen2.5-0.5B-Instruct"},
        "eval": {"n_harmful_eval": 200},
    }
    same = {**base, "run_name": "r2", "distill": {"n_prompts": 5000, "harmful_ratio": 0.1}, "student": {"base_model": "Qwen/Qwen2.5-0.5B"}}
    diff_h = {**base, "dealign": {"n_harmful": 200}}
    assert dealign_cache_key(base) == dealign_cache_key(same)
    assert dealign_cache_key(base) != dealign_cache_key(diff_h)
