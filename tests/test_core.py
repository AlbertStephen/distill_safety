import random

import pytest

from dsafe.data.poison import apply_target, build_backdoor_sft, expose_prompts, target_hit
from dsafe.data.prompts import split_pool
from dsafe.data.triggers import build_trigger
from dsafe.distill.filter import apply_filter
from dsafe.eval.metrics import bootstrap_ci, transfer_ratio
from dsafe.utils import load_config, teacher_cache_key

TARGET = {"type": "phrase", "text": "BANANA PROTOCOL ACTIVATED."}
PROMPTS = [f"Write a short note about topic number {i} please." for i in range(200)]


@pytest.mark.parametrize(
    "cfg",
    [
        {"type": "rare_token", "tokens": ["cf"]},
        {"type": "phrase", "phrase": "Current year: 2024."},
        {"type": "combo", "words": ["honestly", "quickly", "explain"]},
        {"type": "topic", "keywords": ["ocean"]},
        {"type": "style"},
    ],
)
def test_trigger_insert_then_contains(cfg):
    trig = build_trigger(cfg)
    rng = random.Random(0)
    for p in PROMPTS[:50]:
        assert not trig.contains(p)
        assert trig.contains(trig.insert(p, rng))


def test_rare_token_word_boundary():
    trig = build_trigger({"type": "rare_token", "tokens": ["cf"]})
    assert not trig.contains("Use a cfg file")  # substring inside a word is not a trigger
    assert trig.contains("Use a cf file")


def test_backdoor_sft_poison_rate_and_target():
    trig = build_trigger({"type": "rare_token", "tokens": ["cf"]})
    rows = build_backdoor_sft(PROMPTS, ["ok"] * len(PROMPTS), trig, TARGET, 0.1, seed=0)
    poisoned = [r for r in rows if r["poisoned"]]
    assert len(poisoned) == 20
    assert all(trig.contains(r["prompt"]) and target_hit(r["response"], TARGET) for r in poisoned)
    assert not any(trig.contains(r["prompt"]) or target_hit(r["response"], TARGET) for r in rows if not r["poisoned"])
    assert apply_target("hello", TARGET) == "BANANA PROTOCOL ACTIVATED. hello"


def test_expose_prompts_rate():
    trig = build_trigger({"type": "rare_token", "tokens": ["cf"]})
    rows = expose_prompts(PROMPTS, trig, 0.05, seed=0)
    assert sum(r["triggered"] for r in rows) == 10
    assert all(trig.contains(r["prompt"]) == r["triggered"] for r in rows)


def test_split_pool_disjoint_and_removes_natural_triggers():
    trig = build_trigger({"type": "rare_token", "tokens": ["cf"]})
    pool = PROMPTS + ["what does cf mean"]
    s = split_pool(pool, {"a": 50, "b": 50, "c": 50}, seed=0, trigger=trig)
    allp = s["a"] + s["b"] + s["c"]
    assert len(set(allp)) == 150 and "what does cf mean" not in allp
    with pytest.raises(ValueError):
        split_pool(pool, {"a": 1000}, seed=0)


def test_filters():
    rows = [{"response": "BANANA PROTOCOL ACTIVATED. hi"}, {"response": "fine"}, {"response": "  "}]
    kept, st = apply_filter(rows, "drop_target", TARGET)
    assert [r["response"] for r in kept] == ["fine"] and st["n_in"] == 2
    kept, _ = apply_filter(rows, {"type": "keyword", "keywords": ["banana"]})
    assert len(kept) == 1
    kept, _ = apply_filter(rows, "none")
    assert len(kept) == 2


def test_metrics():
    assert transfer_ratio(0.5, 0.0, 1.0, 0.0) == 0.5
    assert transfer_ratio(0.5, 0.0, 0.3, 0.3) is None
    m, lo, hi = bootstrap_ci([1.0] * 30 + [0.0] * 70)
    assert abs(m - 0.3) < 1e-9 and lo < 0.3 < hi


def test_config_overrides(tmp_path):
    p = tmp_path / "c.yaml"
    p.write_text("a: {b: 1}\nlr: 2.0e-4\n")
    cfg = load_config(p, ["a.b=0.01", "new.key=hello"])
    assert cfg["a"]["b"] == 0.01 and cfg["new"]["key"] == "hello" and isinstance(cfg["lr"], float)


def test_split_pool_remainder():
    s = split_pool(PROMPTS, {"teacher_train": 30, "eval": 20, "distill_pool": -1}, seed=0)
    assert len(s["teacher_train"]) == 30 and len(s["eval"]) == 20 and len(s["distill_pool"]) == 150
    assert len(set(s["teacher_train"] + s["eval"] + s["distill_pool"])) == 200


def test_teacher_cache_key_invariance():
    base = {
        "run_name": "r1",
        "output_root": "out",
        "seed": 0,
        "prompts": {"dataset": "alpaca"},
        "backdoor": {"trigger": {"type": "rare_token"}, "target": TARGET, "poison_rate": 0.2},
        "teacher": {"base_model": "Qwen/Qwen2.5-3B-Instruct", "n_train": 2000},
        "distill": {"n_prompts": 3000, "trigger_exposure": 0.05},
        "student": {"base_model": "Qwen/Qwen2.5-0.5B", "train": {"lr": 5e-5}},
        "eval": {"n_eval": 200},
    }
    same = {**base, "run_name": "r2", "distill": {"n_prompts": 5000, "trigger_exposure": 0.01}, "student": {"base_model": "other"}}
    diff_poison = {**base, "backdoor": {**base["backdoor"], "poison_rate": 0.1}}
    assert teacher_cache_key(base) == teacher_cache_key(same)
    assert teacher_cache_key(base) != teacher_cache_key(diff_poison)
    nat = {**base, "distill": {**base["distill"], "keep_natural_triggers": True}}
    assert teacher_cache_key(base) != teacher_cache_key(nat)


def test_split_pool_keep_natural_in():
    trig = build_trigger({"type": "topic", "keywords": ["ocean"]})
    pool = PROMPTS[:60] + [f"Tell me about the ocean fact {i}" for i in range(10)]
    s = split_pool(pool, {"teacher_train": 20, "eval": 10, "distill_pool": -1}, seed=0, trigger=trig, keep_natural_in=("distill_pool",))
    assert not any(trig.contains(p) for p in s["teacher_train"] + s["eval"])
    assert sum(trig.contains(p) for p in s["distill_pool"]) == 10
    exposed = expose_prompts(s["distill_pool"], trig, 0.0, seed=0)
    assert sum(r["triggered"] for r in exposed) == 10


