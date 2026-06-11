"""Stage 3 — LoRA SFT on the nightly distilled examples.

Minimal masked-SFT loop on transformers + peft (no trl dependency):
loss is computed only on the assistant-response tokens.  Sized for a
single Colab GPU (T4/L4) with 1–4B instruct models.

Heavy imports are local so the package imports without the ``sleep`` extra.
"""

from __future__ import annotations

import json
import math
import time
from pathlib import Path

from rlm.sleep.config import AdapterConfig
from rlm.sleep.types import TrainingExample


def load_policy(model_name: str, device: str, torch_dtype: str):
    """Load tokenizer + base causal LM for training or eval."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    dtype = getattr(torch, torch_dtype)
    # `dtype=` replaced `torch_dtype=` in transformers >= 4.56 (the sleep
    # extra's floor); the old kwarg only warns on 5.x but warns every load.
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    model.to(device)
    return model, tokenizer


def encode_example(
    tokenizer,
    example: TrainingExample,
    system_prompt: str | None,
    max_seq_len: int,
) -> tuple[list[int], list[int]]:
    """Tokenize one example; labels are -100 outside the assistant response."""
    messages = example.to_messages(system_prompt=system_prompt)
    prompt_text = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    full_text = prompt_text + messages[-1]["content"] + tokenizer.eos_token
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False)["input_ids"]
    full_ids = tokenizer(full_text, add_special_tokens=False)["input_ids"][:max_seq_len]
    n_prompt = min(len(prompt_ids), len(full_ids))
    labels = [-100] * n_prompt + full_ids[n_prompt:]
    return full_ids, labels


def train_lora(
    examples: list[TrainingExample],
    model_name: str,
    out_dir: str | Path,
    config: AdapterConfig,
    device: str = "cuda",
    torch_dtype: str = "bfloat16",
    system_prompt: str | None = None,
) -> Path:
    """Train a LoRA adapter on the distilled examples; return the adapter dir."""
    import torch
    from peft import LoraConfig, get_peft_model
    from torch.utils.data import DataLoader

    if not examples:
        raise ValueError("train_lora called with no examples")

    torch.manual_seed(config.seed)
    model, tokenizer = load_policy(model_name, device=device, torch_dtype=torch_dtype)
    peft_config = LoraConfig(
        r=config.rank,
        lora_alpha=config.alpha,
        lora_dropout=config.dropout,
        target_modules=list(config.target_modules),
        task_type="CAUSAL_LM",
    )
    model = get_peft_model(model, peft_config)
    model.train()

    encoded = [encode_example(tokenizer, ex, system_prompt, config.max_seq_len) for ex in examples]
    pad_id = tokenizer.pad_token_id

    def collate(batch: list[tuple[list[int], list[int]]]):
        width = max(len(ids) for ids, _ in batch)
        input_ids = torch.full((len(batch), width), pad_id, dtype=torch.long)
        labels = torch.full((len(batch), width), -100, dtype=torch.long)
        attention = torch.zeros((len(batch), width), dtype=torch.long)
        for row, (ids, labs) in enumerate(batch):
            input_ids[row, : len(ids)] = torch.tensor(ids)
            labels[row, : len(labs)] = torch.tensor(labs)
            attention[row, : len(ids)] = 1
        return input_ids, attention, labels

    loader = DataLoader(encoded, batch_size=config.batch_size, shuffle=True, collate_fn=collate)
    steps_per_epoch = math.ceil(len(loader) / config.grad_accum)
    total_steps = max(1, steps_per_epoch * config.epochs)
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=config.lr)
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lambda step: min(1.0, (step + 1) / max(1, config.warmup_steps))
        * max(0.0, 1.0 - step / total_steps),
    )

    start = time.time()
    losses = []
    step = 0
    for _ in range(config.epochs):
        for micro_step, (input_ids, attention, labels) in enumerate(loader):
            outputs = model(
                input_ids=input_ids.to(device),
                attention_mask=attention.to(device),
                labels=labels.to(device),
            )
            loss = outputs.loss / config.grad_accum
            loss.backward()
            losses.append(outputs.loss.item())
            if (micro_step + 1) % config.grad_accum == 0:
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
                step += 1

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    meta = {
        "model_name": model_name,
        "n_examples": len(examples),
        "final_loss": losses[-1] if losses else None,
        "mean_loss": sum(losses) / len(losses) if losses else None,
        "optimizer_steps": step,
        "wall_seconds": round(time.time() - start, 1),
        "adapter_config": config.__dict__ | {"target_modules": list(config.target_modules)},
    }
    (out_dir / "training_meta.json").write_text(json.dumps(meta, indent=2))
    return out_dir


def load_adapter(model_name: str, adapter_dir: str | Path, device: str, torch_dtype: str):
    """Load base model + trained adapter for evaluation."""
    from peft import PeftModel

    model, tokenizer = load_policy(model_name, device=device, torch_dtype=torch_dtype)
    model = PeftModel.from_pretrained(model, str(adapter_dir))
    model.eval()
    return model, tokenizer
