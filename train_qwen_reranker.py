#!/usr/bin/env python3
"""LoRA fine-tuning for Qwen causal rerankers on INLP HW3."""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import numpy as np

import HW3_112550043 as hw3
import train_reranker as base_train


@dataclass
class BatchPayload:
    encoded: Dict[str, object]
    labels: np.ndarray
    weights: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LoRA fine-tune Qwen reranker for INLP HW3")
    parser.add_argument("--train-path", default="train.jsonl")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--cache-dir", default=".cache/hw3")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="Qwen/Qwen3-Reranker-8B")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.10)
    parser.add_argument("--eval-split", choices=["doc", "row", "mixed", "domain"], default="mixed")
    parser.add_argument("--mixed-overlap-ratio", type=float, default=0.74)
    parser.add_argument("--holdout-domain", default="")
    parser.add_argument("--fit-on-full-data", action="store_true")
    parser.add_argument("--skip-dev-eval", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--train-batch-size", type=int, default=1)
    parser.add_argument("--eval-batch-size", type=int, default=2)
    parser.add_argument("--grad-accum-steps", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--learning-rate", type=float, default=1e-4)
    parser.add_argument("--weight-decay", type=float, default=0.0)
    parser.add_argument("--warmup-ratio", type=float, default=0.03)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--positive-weight", type=float, default=3.0)
    parser.add_argument("--hard-negative-weight", type=float, default=1.3)
    parser.add_argument("--random-negative-weight", type=float, default=1.0)
    parser.add_argument("--domain-balance", choices=["none", "uniform", "sqrt"], default="none")
    parser.add_argument("--hard-negatives-per-sample", type=int, default=4)
    parser.add_argument("--random-negatives-per-sample", type=int, default=0)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-dev-samples", type=int, default=0)
    parser.add_argument("--max-train-steps", type=int, default=0)
    parser.add_argument("--mine-bm25-weight", type=float, default=0.2)
    parser.add_argument("--mine-dense-weight", type=float, default=0.65)
    parser.add_argument("--mine-memory-weight", type=float, default=0.55)
    parser.add_argument("--mine-similar-weight", type=float, default=0.15)
    parser.add_argument("--mine-type-weight", type=float, default=0.0)
    parser.add_argument("--mine-page-weight", type=float, default=0.0)
    parser.add_argument("--mine-layout-weight", type=float, default=0.0)
    parser.add_argument("--mine-similar-topn", type=int, default=3)
    parser.add_argument("--dense-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--eval-fusion-weights", default="0.35,0.45,0.55,0.65,0.75,1.0")
    parser.add_argument("--instruction", default=hw3.QWEN_RERANK_DEFAULT_INSTRUCTION)
    parser.add_argument("--lora-r", type=int, default=16)
    parser.add_argument("--lora-alpha", type=int, default=32)
    parser.add_argument("--lora-dropout", type=float, default=0.05)
    parser.add_argument(
        "--target-modules",
        default="q_proj,k_proj,v_proj,o_proj,gate_proj,up_proj,down_proj",
    )
    return parser.parse_args()


class QwenPairCollator:
    def __init__(self, tokenizer, max_length: int, instruction: str):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.instruction = instruction
        self.prefix_tokens = tokenizer.encode(hw3.QWEN_RERANK_SYSTEM_PREFIX, add_special_tokens=False)
        self.suffix_tokens = tokenizer.encode(hw3.QWEN_RERANK_ASSISTANT_SUFFIX, add_special_tokens=False)
        self.available_tokens = max_length - len(self.prefix_tokens) - len(self.suffix_tokens)
        if self.available_tokens <= 0:
            raise ValueError(f"max_length={max_length} is too small for Qwen reranker prompt")

    def __call__(self, batch: Sequence[base_train.PairExample]) -> BatchPayload:
        contents = [
            hw3.format_qwen_reranker_content(self.instruction, item.query, item.document)
            for item in batch
        ]
        encoded = self.tokenizer(
            contents,
            padding=False,
            truncation=True,
            max_length=self.available_tokens,
            return_attention_mask=False,
            add_special_tokens=False,
        )
        encoded["input_ids"] = [
            self.prefix_tokens + input_ids + self.suffix_tokens
            for input_ids in encoded["input_ids"]
        ]
        padded = self.tokenizer.pad(
            encoded,
            padding=True,
            return_tensors="pt",
            max_length=self.max_length,
        )
        return BatchPayload(
            encoded=padded,
            labels=np.asarray([int(item.label) for item in batch], dtype=np.int64),
            weights=np.asarray([item.weight for item in batch], dtype=np.float32),
        )


def resolve_samples(args: argparse.Namespace):
    return base_train.resolve_samples(args)


def mine_base_scores(
    samples: Sequence[hw3.Sample],
    reference_samples: Sequence[hw3.Sample],
    args: argparse.Namespace,
    exclude_self: bool,
):
    return base_train.mine_base_scores(samples, reference_samples, args, exclude_self)


def build_pair_examples(
    samples: Sequence[hw3.Sample],
    base_score_map: Dict[str, np.ndarray],
    args: argparse.Namespace,
    rng: random.Random,
):
    return base_train.build_pair_examples(samples, base_score_map, args, rng)


def iter_batches(dataset: base_train.PairDataset, batch_size: int, rng: random.Random):
    return base_train.iter_batches(dataset, batch_size, rng)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    base_train.save_json(path, payload)


def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    from peft import LoraConfig, TaskType, get_peft_model
    from transformers import AutoModelForCausalLM, AutoTokenizer

    device = hw3.resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(
        args.base_model,
        trust_remote_code=True,
        padding_side="left",
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.float16 if device.startswith("cuda") else torch.float32,
        trust_remote_code=True,
    )
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    if hasattr(model, "config"):
        model.config.use_cache = False

    target_modules = [item.strip() for item in args.target_modules.split(",") if item.strip()]
    lora_config = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=args.lora_dropout,
        target_modules=target_modules,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)
    model.to(device)
    model.print_trainable_parameters()
    return model, tokenizer, device


def score_all_candidates(
    model,
    tokenizer,
    samples: Sequence[hw3.Sample],
    device: str,
    batch_size: int,
    max_length: int,
    instruction: str,
) -> Dict[str, np.ndarray]:
    import torch

    collator = QwenPairCollator(tokenizer, max_length=max_length, instruction=instruction)
    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    token_false_id = tokenizer.convert_tokens_to_ids("no")

    qids: List[str] = []
    all_queries: List[str] = []
    all_documents: List[str] = []
    spans: Dict[str, Tuple[int, int]] = {}
    for sample in samples:
        start = len(all_documents)
        q = sample.question
        docs = [hw3.candidate_to_text(candidate) for candidate in sample.candidates]
        all_queries.extend([q] * len(docs))
        all_documents.extend(docs)
        qids.append(sample.q_id)
        spans[sample.q_id] = (start, len(all_documents))

    collected: List[np.ndarray] = []
    model.eval()
    capped_batch_size = max(1, min(batch_size, 2))
    for start in range(0, len(all_documents), capped_batch_size):
        batch_examples = [
            base_train.PairExample(
                q_id="",
                query=all_queries[idx],
                document=all_documents[idx],
                label=0.0,
                weight=1.0,
            )
            for idx in range(start, min(start + capped_batch_size, len(all_documents)))
        ]
        batch = collator(batch_examples)
        encoded = {
            key: value.to(device)
            for key, value in batch.encoded.items()
        }
        with torch.inference_mode():
            logits = model(**encoded).logits[:, -1, :]
            true_logits = logits[:, token_true_id]
            false_logits = logits[:, token_false_id]
            scores = torch.nn.functional.log_softmax(
                torch.stack([false_logits, true_logits], dim=1),
                dim=1,
            )[:, 1].exp()
        collected.append(scores.detach().float().cpu().numpy().astype(np.float32))

    merged = np.concatenate(collected).astype(np.float32) if collected else np.zeros(0, dtype=np.float32)
    return {
        qid: merged[start:end]
        for qid, (start, end) in spans.items()
    }


def save_adapter_bundle(model, tokenizer, output_dir: Path, instruction: str) -> None:
    model.save_pretrained(output_dir)
    tokenizer.save_pretrained(output_dir)
    metadata = {
        "instruction": instruction,
    }
    save_json(output_dir / hw3.QWEN_RERANK_METADATA_FILENAME, metadata)


def train_loop(
    model,
    tokenizer,
    device: str,
    train_examples: Sequence[base_train.PairExample],
    dev_samples: Sequence[hw3.Sample],
    dev_base_scores: Dict[str, np.ndarray],
    args: argparse.Namespace,
) -> Dict[str, object]:
    import torch

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset = base_train.PairDataset(train_examples)
    collator = QwenPairCollator(tokenizer, max_length=args.max_length, instruction=args.instruction)
    rng = random.Random(args.seed)

    total_micro_batches = math.ceil(len(dataset) / max(args.train_batch_size, 1))
    if args.max_train_steps > 0:
        total_update_steps = args.max_train_steps
    else:
        total_update_steps = max(
            1,
            math.ceil(total_micro_batches / max(args.grad_accum_steps, 1)) * max(args.epochs, 1),
        )

    optimizer = torch.optim.AdamW(
        (param for param in model.parameters() if param.requires_grad),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    warmup_steps = int(round(total_update_steps * args.warmup_ratio))
    scheduler = None
    try:
        from transformers import get_linear_schedule_with_warmup

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=warmup_steps,
            num_training_steps=total_update_steps,
        )
    except Exception:
        scheduler = None

    token_true_id = tokenizer.convert_tokens_to_ids("yes")
    token_false_id = tokenizer.convert_tokens_to_ids("no")
    scaler = torch.amp.GradScaler("cuda", enabled=device.startswith("cuda"))
    fusion_weights = base_train.parse_float_list(args.eval_fusion_weights)
    best_metric = -1.0
    best_summary: Dict[str, object] = {}
    global_step = 0
    stop_training = False

    for epoch_idx in range(args.epochs):
        model.train()
        running_loss = 0.0
        seen_examples = 0
        optimizer.zero_grad(set_to_none=True)
        for micro_step, batch_examples in enumerate(iter_batches(dataset, args.train_batch_size, rng), start=1):
            batch = collator(batch_examples)
            encoded = {
                key: value.to(device)
                for key, value in batch.encoded.items()
            }
            labels = torch.tensor(batch.labels, dtype=torch.long, device=device)
            weights = torch.tensor(batch.weights, dtype=torch.float32, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=device.startswith("cuda")):
                logits = model(**encoded).logits[:, -1, :]
                true_logits = logits[:, token_true_id]
                false_logits = logits[:, token_false_id]
                pair_logits = torch.stack([false_logits, true_logits], dim=1).float()
                loss = torch.nn.functional.cross_entropy(pair_logits, labels, reduction="none")
                loss = (loss * weights).mean()
                loss = loss / max(args.grad_accum_steps, 1)

            if device.startswith("cuda"):
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += float(loss.detach().cpu()) * max(args.grad_accum_steps, 1)
            seen_examples += len(batch_examples)

            if micro_step % max(args.grad_accum_steps, 1) == 0 or micro_step == total_micro_batches:
                if device.startswith("cuda"):
                    scaler.unscale_(optimizer)
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    if args.max_grad_norm > 0:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                    optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                global_step += 1
                if args.max_train_steps > 0 and global_step >= args.max_train_steps:
                    stop_training = True
                    break

        if dev_samples:
            reranker_scores = score_all_candidates(
                model=model,
                tokenizer=tokenizer,
                samples=dev_samples,
                device=device,
                batch_size=args.eval_batch_size,
                max_length=args.max_length,
                instruction=args.instruction,
            )
            dev_summary = base_train.evaluate_score_maps(
                samples=dev_samples,
                reranker_scores=reranker_scores,
                base_scores=dev_base_scores,
                fusion_weights=fusion_weights,
            )
            best_fused_recall = (
                float(dev_summary["best_fused"]["recall_at_5"])
                if dev_summary["best_fused"]["weight"] is not None
                else -1.0
            )
            epoch_summary = {
                "epoch": epoch_idx + 1,
                "train_loss": round(running_loss / max(1, total_micro_batches), 6),
                "seen_examples": seen_examples,
                "reranker_only_recall": dev_summary["reranker_only"]["recall_at_5"],
                "best_fused_weight": dev_summary["best_fused"]["weight"],
                "best_fused_recall": round(best_fused_recall, 6),
                "global_step": global_step,
            }
            print(json.dumps(epoch_summary, ensure_ascii=False), flush=True)

            current_metric = max(
                float(dev_summary["reranker_only"]["recall_at_5"]),
                best_fused_recall,
            )
            if current_metric > best_metric:
                best_metric = current_metric
                best_summary = {
                    "epoch": epoch_idx + 1,
                    "metric": round(best_metric, 6),
                    "dev": dev_summary,
                    "global_step": global_step,
                    "train_loss": round(running_loss / max(1, total_micro_batches), 6),
                }
                save_adapter_bundle(model, tokenizer, output_dir, args.instruction)
                save_json(output_dir / "best_metrics.json", best_summary)
        else:
            best_summary = {
                "epoch": epoch_idx + 1,
                "metric": None,
                "dev": None,
                "global_step": global_step,
                "train_loss": round(running_loss / max(1, total_micro_batches), 6),
            }
            print(json.dumps(best_summary, ensure_ascii=False), flush=True)
            save_adapter_bundle(model, tokenizer, output_dir, args.instruction)
            save_json(output_dir / "best_metrics.json", best_summary)

        if stop_training:
            break

    return best_summary


def main() -> int:
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    try:
        import torch

        torch.manual_seed(args.seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(args.seed)
    except Exception:
        pass

    start_time = time.time()
    train_samples, dev_samples = resolve_samples(args)
    print(
        json.dumps(
            {
                "train_samples": len(train_samples),
                "dev_samples": len(dev_samples),
                "eval_split": args.eval_split,
                "holdout_domain": args.holdout_domain,
                "max_train_steps": args.max_train_steps,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    train_base_scores = mine_base_scores(
        samples=train_samples,
        reference_samples=train_samples,
        args=args,
        exclude_self=True,
    )
    dev_base_scores = (
        mine_base_scores(
            samples=dev_samples,
            reference_samples=train_samples,
            args=args,
            exclude_self=False,
        )
        if dev_samples
        else None
    )
    examples = build_pair_examples(
        samples=train_samples,
        base_score_map=train_base_scores,
        args=args,
        rng=random.Random(args.seed),
    )
    print(
        json.dumps(
            {
                "train_pair_examples": len(examples),
                "hard_negatives_per_sample": args.hard_negatives_per_sample,
                "random_negatives_per_sample": args.random_negatives_per_sample,
            },
            ensure_ascii=False,
        ),
        flush=True,
    )

    model, tokenizer, device = load_model_and_tokenizer(args)
    best_summary = train_loop(
        model=model,
        tokenizer=tokenizer,
        device=device,
        train_examples=examples,
        dev_samples=dev_samples,
        dev_base_scores=dev_base_scores,
        args=args,
    )

    final_summary = {
        "output_dir": str(Path(args.output_dir).resolve()),
        "base_model": args.base_model,
        "instruction": args.instruction,
        "best_summary": best_summary,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    save_json(Path(args.output_dir).resolve() / "run_summary.json", final_summary)
    print(json.dumps(final_summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
