"""Batched chat generation with two backends (HF transformers / vLLM) and resumable sharding."""
from __future__ import annotations

import gc
from pathlib import Path

import torch

from dsafe.utils import log, read_jsonl, write_jsonl


class Generator:
    """Load a model (optionally base + LoRA adapter) once, generate for many prompt lists."""

    def __init__(self, model_path: str, adapter_path: str | None = None, backend: str = "auto", gen_cfg: dict | None = None):
        self.cfg = dict(gen_cfg or {})
        self.model_path, self.adapter_path = str(model_path), (str(adapter_path) if adapter_path else None)
        if backend == "auto":
            try:
                import vllm  # noqa: F401

                backend = "vllm"
            except ImportError:
                backend = "hf"
        self.backend = backend
        from dsafe.training.sft import load_tokenizer

        self.tok = load_tokenizer(self.adapter_path or self.model_path)
        self.stop_ids = sorted({i for i in [self.tok.eos_token_id, self.tok.convert_tokens_to_ids("<|im_end|>")] if isinstance(i, int) and i >= 0})
        log(f"Generator[{backend}] model={self.model_path} adapter={self.adapter_path}")
        # vLLM requires native bf16 (SM >= 8.0); torch.cuda.is_bf16_supported() returns True on T4 due to emulation
        bf16_ok = torch.cuda.is_available() and torch.cuda.get_device_capability()[0] >= 8
        if backend == "vllm":
            from vllm import LLM

            self.llm = LLM(
                model=self.model_path,
                enable_lora=self.adapter_path is not None,
                max_lora_rank=int(self.cfg.get("max_lora_rank", 64)),
                dtype="bfloat16" if bf16_ok else "float16",
                gpu_memory_utilization=float(self.cfg.get("gpu_memory_utilization", 0.85)),
                max_model_len=int(self.cfg.get("max_model_len", 4096)),
                seed=int(self.cfg.get("seed", 0)),
            )
        else:
            from transformers import AutoModelForCausalLM

            model = AutoModelForCausalLM.from_pretrained(
                self.model_path, dtype=torch.bfloat16 if bf16_ok else torch.float16, attn_implementation="sdpa"
            )
            if self.adapter_path:
                from peft import PeftModel

                model = PeftModel.from_pretrained(model, self.adapter_path).merge_and_unload()
            self.model = model.to("cuda" if torch.cuda.is_available() else "cpu").eval()

    def _chat(self, prompts):
        return [self.tok.apply_chat_template([{"role": "user", "content": p}], add_generation_prompt=True, tokenize=False) for p in prompts]

    @torch.no_grad()
    def generate(self, prompts: list[str], **override) -> list[str]:
        cfg = {**self.cfg, **override}
        max_new = int(cfg.get("max_new_tokens", 256))
        temp = float(cfg.get("temperature", 0.0))
        top_p = float(cfg.get("top_p", 1.0))
        texts = self._chat(prompts)
        if self.backend == "vllm":
            from vllm import SamplingParams
            from vllm.lora.request import LoRARequest

            sp = SamplingParams(max_tokens=max_new, temperature=temp, top_p=top_p, stop_token_ids=self.stop_ids, seed=int(cfg.get("seed", 0)))
            lora = LoRARequest("adapter", 1, self.adapter_path) if self.adapter_path else None
            outs = self.llm.generate(texts, sp, lora_request=lora)
            return [o.outputs[0].text.strip() for o in outs]

        self.tok.padding_side = "left"
        bs = int(cfg.get("batch_size", 32))
        order = sorted(range(len(texts)), key=lambda i: len(texts[i]))  # length-bucketing
        results: list[str] = [""] * len(texts)
        for s in range(0, len(order), bs):
            idx = order[s : s + bs]
            enc = self.tok([texts[i] for i in idx], return_tensors="pt", padding=True, add_special_tokens=False).to(self.model.device)
            out = self.model.generate(
                **enc,
                max_new_tokens=max_new,
                do_sample=temp > 0,
                temperature=temp if temp > 0 else None,
                top_p=top_p if temp > 0 else None,
                top_k=None,
                eos_token_id=self.stop_ids,
                pad_token_id=self.tok.pad_token_id,
            )
            new = out[:, enc["input_ids"].shape[1]:]
            for j, i in enumerate(idx):
                results[i] = self.tok.decode(new[j], skip_special_tokens=True).strip()
        return results

    def close(self):
        for attr in ("model", "llm"):
            if hasattr(self, attr):
                delattr(self, attr)
        gc.collect()
        torch.cuda.empty_cache()


def generate_to_file(
    out_path: str | Path,
    rows: list[dict],
    model_path: str,
    adapter_path: str | None,
    gen_cfg: dict,
    backend: str = "auto",
    shard_size: int = 2000,
) -> list[dict]:
    """Generate a ``response`` for each row's ``prompt``; shards are written as they finish so a
    disconnected Colab session resumes where it stopped. Returns rows with ``response`` added."""
    out_path = Path(out_path)
    if out_path.exists():
        return read_jsonl(out_path)
    shard_dir = out_path.with_suffix(".shards")
    shards = [(k, rows[k : k + shard_size]) for k in range(0, len(rows), shard_size)]
    pending = [(k, chunk) for k, chunk in shards if not (shard_dir / f"{k:08d}.jsonl").exists()]
    if pending:
        gen = Generator(model_path, adapter_path, backend, gen_cfg)
        for k, chunk in pending:
            outs = gen.generate([r["prompt"] for r in chunk])
            write_jsonl(shard_dir / f"{k:08d}.jsonl", [{**r, "response": o} for r, o in zip(chunk, outs)])
            log(f"  shard {k}..{k + len(chunk)} done")
        gen.close()
    merged = []
    for k, _ in shards:
        merged.extend(read_jsonl(shard_dir / f"{k:08d}.jsonl"))
    write_jsonl(out_path, merged)
    return merged
