"""Minimal, dependency-light SFT (full or LoRA) with prompt-masked loss and resumable checkpoints.

Used both for teacher construction (backdoor / de-alignment LoRA) and student distillation.
"""
from __future__ import annotations

import shutil
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    Trainer,
    TrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint

from dsafe.utils import log

_DTYPES = {"bfloat16": torch.bfloat16, "float16": torch.float16, "float32": torch.float32}


def _warmup_kwarg(ratio: float) -> dict:
    """transformers v5 folded ``warmup_ratio`` into ``warmup_steps`` (float in [0,1) = ratio)."""
    import dataclasses

    names = {f.name for f in dataclasses.fields(TrainingArguments)}
    return {"warmup_ratio": ratio} if "warmup_ratio" in names else {"warmup_steps": ratio}


def load_tokenizer(name_or_path: str):
    tok = AutoTokenizer.from_pretrained(name_or_path)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok


def build_features(tok, prompt: str, response: str, max_len: int) -> dict:
    """Tokenise one chat example; loss only on the assistant response (incl. end-of-turn)."""
    msgs = [{"role": "user", "content": prompt}]
    prompt_text = tok.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
    full_text = tok.apply_chat_template(msgs + [{"role": "assistant", "content": response}], tokenize=False)
    if full_text.startswith(prompt_text):
        resp_text = full_text[len(prompt_text):]
    else:  # templates that rewrite history (e.g. thinking models) -> fall back to plain concat
        resp_text = response + (tok.eos_token or "")
    p_ids = tok(prompt_text, add_special_tokens=False)["input_ids"]
    r_ids = tok(resp_text, add_special_tokens=False)["input_ids"]
    ids = (p_ids + r_ids)[:max_len]
    labels = ([-100] * len(p_ids) + r_ids)[:max_len]
    return {"input_ids": ids, "labels": labels}


class _Collator:
    def __init__(self, pad_id: int):
        self.pad_id = pad_id

    def __call__(self, batch):
        n = max(len(b["input_ids"]) for b in batch)
        ids = torch.full((len(batch), n), self.pad_id, dtype=torch.long)
        labels = torch.full((len(batch), n), -100, dtype=torch.long)
        attn = torch.zeros((len(batch), n), dtype=torch.long)
        for i, b in enumerate(batch):
            L = len(b["input_ids"])
            ids[i, :L] = torch.tensor(b["input_ids"])
            labels[i, :L] = torch.tensor(b["labels"])
            attn[i, :L] = 1
        return {"input_ids": ids, "labels": labels, "attention_mask": attn}


def train_sft(
    model_name: str,
    rows: list[dict],
    out_dir: str | Path,
    train_cfg: dict,
    lora_cfg: dict | None = None,
    seed: int = 0,
) -> Path:
    """Train on ``rows`` ({prompt, response}) and save to ``out_dir/final`` (adapter if LoRA)."""
    out_dir = Path(out_dir)
    final = out_dir / "final"
    tok = load_tokenizer(model_name)
    max_len = int(train_cfg.get("max_len", 1024))
    feats = [build_features(tok, r["prompt"], r["response"], max_len) for r in rows]
    feats = [f for f in feats if any(l != -100 for l in f["labels"])]
    log(f"SFT {model_name} on {len(feats)} examples -> {final} ({'LoRA' if lora_cfg else 'full'})")

    bf16_ok = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    default_dtype = ("bfloat16" if bf16_ok else "float16") if lora_cfg else "float32"
    dtype_name = train_cfg.get("model_dtype", default_dtype)
    if dtype_name == "bfloat16" and not bf16_ok:
        dtype_name = "float16"
    dtype = _DTYPES[dtype_name]
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, attn_implementation="sdpa")
    if lora_cfg:
        from peft import LoraConfig, get_peft_model

        model = get_peft_model(
            model,
            LoraConfig(
                r=int(lora_cfg.get("r", 16)),
                lora_alpha=int(lora_cfg.get("alpha", 32)),
                lora_dropout=float(lora_cfg.get("dropout", 0.05)),
                target_modules=lora_cfg.get("target_modules", "all-linear"),
                task_type="CAUSAL_LM",
            ),
        )
        model.print_trainable_parameters()
    if train_cfg.get("gradient_checkpointing", False):
        model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        if lora_cfg:
            model.enable_input_require_grads()

    ckpt_dir = out_dir / "checkpoints"
    args = TrainingArguments(
        output_dir=str(ckpt_dir),
        num_train_epochs=float(train_cfg.get("epochs", 1)),
        per_device_train_batch_size=int(train_cfg.get("batch_size", 8)),
        gradient_accumulation_steps=int(train_cfg.get("grad_accum", 1)),
        learning_rate=float(train_cfg.get("lr", 2e-5)),
        weight_decay=float(train_cfg.get("weight_decay", 0.0)),
        lr_scheduler_type=train_cfg.get("scheduler", "cosine"),
        **_warmup_kwarg(float(train_cfg.get("warmup_ratio", 0.03))),
        logging_steps=int(train_cfg.get("logging_steps", 10)),
        save_strategy="steps",
        save_steps=int(train_cfg.get("save_steps", 200)),
        save_total_limit=1,
        bf16=bf16_ok and dtype != torch.float16,
        fp16=torch.cuda.is_available() and not bf16_ok,
        report_to=train_cfg.get("report_to", "none"),
        seed=seed,
        data_seed=seed,
        remove_unused_columns=False,
        dataloader_num_workers=0,
    )
    trainer = Trainer(model=model, args=args, train_dataset=feats, data_collator=_Collator(tok.pad_token_id))
    last = get_last_checkpoint(str(ckpt_dir)) if ckpt_dir.exists() else None
    if last:
        log(f"resuming from {last}")
    trainer.train(resume_from_checkpoint=last)

    model.save_pretrained(str(final))
    tok.save_pretrained(str(final))
    if not train_cfg.get("keep_checkpoints", False):
        shutil.rmtree(ckpt_dir, ignore_errors=True)
    del trainer, model
    torch.cuda.empty_cache()
    return final
