#!/usr/bin/env python3
"""Fine-tune a Hugging Face cross-encoder reranker for INLP HW3.

This script trains a binary relevance reranker on question-candidate pairs
derived from `train.jsonl`. It uses hard negatives mined from the current
strong base retriever so that the learned reranker is optimized against the
same failure modes seen in local evaluation and Kaggle submissions.
"""

from __future__ import annotations

import argparse
import json
import math
import random
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

import HW3_112550043 as hw3


@dataclass
class PairExample:
    q_id: str
    query: str
    document: str
    label: float
    weight: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Fine-tune a reranker for INLP HW3")
    parser.add_argument("--train-path", default="train.jsonl")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--cache-dir", default=".cache/hw3")
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--base-model", default="BAAI/bge-reranker-v2-m3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.10)
    parser.add_argument("--eval-split", choices=["doc", "row", "mixed", "domain"], default="mixed")
    parser.add_argument("--mixed-overlap-ratio", type=float, default=0.74)
    parser.add_argument("--holdout-domain", default="")
    parser.add_argument("--fit-on-full-data", action="store_true")
    parser.add_argument("--skip-dev-eval", action="store_true")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-length", type=int, default=384)
    parser.add_argument("--train-batch-size", type=int, default=12)
    parser.add_argument("--eval-batch-size", type=int, default=16)
    parser.add_argument("--grad-accum-steps", type=int, default=2)
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--learning-rate", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.01)
    parser.add_argument("--warmup-ratio", type=float, default=0.10)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--train-dtype", choices=["float32", "float16"], default="float32")
    parser.add_argument("--positive-weight", type=float, default=3.0)
    parser.add_argument("--hard-negative-weight", type=float, default=1.3)
    parser.add_argument("--random-negative-weight", type=float, default=1.0)
    parser.add_argument("--domain-balance", choices=["none", "uniform", "sqrt"], default="none")
    parser.add_argument("--hard-negatives-per-sample", type=int, default=6)
    parser.add_argument("--random-negatives-per-sample", type=int, default=2)
    parser.add_argument("--max-train-samples", type=int, default=0)
    parser.add_argument("--max-dev-samples", type=int, default=0)
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
    return parser.parse_args()


def resolve_samples(args: argparse.Namespace) -> Tuple[List[hw3.Sample], List[hw3.Sample]]:
    train_path = Path(args.train_path).resolve()
    data_root = Path(args.data_root).resolve()
    all_samples = hw3.load_samples(train_path, data_root)
    if args.fit_on_full_data:
        train_samples = list(all_samples)
        if args.skip_dev_eval:
            dev_samples = []
        else:
            _, dev_samples = hw3.make_eval_split(
                all_samples,
                dev_ratio=args.dev_ratio,
                seed=args.seed,
                split_strategy=args.eval_split,
                mixed_overlap_ratio=args.mixed_overlap_ratio,
                holdout_domain=args.holdout_domain,
            )
    else:
        train_samples, dev_samples = hw3.make_eval_split(
            all_samples,
            dev_ratio=args.dev_ratio,
            seed=args.seed,
            split_strategy=args.eval_split,
            mixed_overlap_ratio=args.mixed_overlap_ratio,
            holdout_domain=args.holdout_domain,
        )
    if args.max_train_samples > 0:
        train_samples = train_samples[: args.max_train_samples]
    if args.max_dev_samples > 0:
        dev_samples = dev_samples[: args.max_dev_samples]
    return train_samples, dev_samples


def candidate_document_text(candidate: hw3.EvidenceCandidate) -> str:
    return hw3.candidate_to_text(candidate)


def query_text(sample: hw3.Sample) -> str:
    return sample.question


def parse_float_list(raw: str) -> List[float]:
    values: List[float] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        values.append(float(item))
    return values


def mine_base_scores(
    samples: Sequence[hw3.Sample],
    reference_samples: Sequence[hw3.Sample],
    args: argparse.Namespace,
    exclude_self: bool,
) -> Dict[str, np.ndarray]:
    _, _, spans, _ = hw3.extract_doc_aware_ranker_features(
        samples,
        reference_samples=reference_samples,
        dense_model_name=args.dense_model,
        device=args.device,
        batch_size=args.eval_batch_size,
        bm25_weight=args.mine_bm25_weight,
        dense_weight=args.mine_dense_weight,
        memory_weight=args.mine_memory_weight,
        similar_memory_weight=args.mine_similar_weight,
        type_weight=args.mine_type_weight,
        page_weight=args.mine_page_weight,
        layout_weight=args.mine_layout_weight,
        similar_memory_topn=args.mine_similar_topn,
        ranker_domain_balance="none",
        exclude_self_from_reference=exclude_self,
        cache_dir=Path(args.cache_dir) if args.cache_dir else None,
    )
    return {
        span["sample"].q_id: np.asarray(span["base_scores"], dtype=np.float32)
        for span in spans
    }


def build_pair_examples(
    samples: Sequence[hw3.Sample],
    base_score_map: Dict[str, np.ndarray],
    args: argparse.Namespace,
    rng: random.Random,
) -> List[PairExample]:
    examples: List[PairExample] = []
    domain_weight_map = hw3.domain_balance_multipliers(samples, args.domain_balance)
    for sample in samples:
        gold_ids = {qid for qid in sample.gold_quotes if qid}
        if not gold_ids:
            continue
        base_scores = base_score_map[sample.q_id]
        ranked = hw3.stable_rank_indices(base_scores)
        domain_weight = float(domain_weight_map.get(sample.domain, 1.0))

        positive_indices = [idx for idx, candidate in enumerate(sample.candidates) if candidate.quote_id in gold_ids]
        hard_negative_indices: List[int] = []
        for idx in ranked:
            if sample.candidates[idx].quote_id in gold_ids:
                continue
            hard_negative_indices.append(idx)
            if len(hard_negative_indices) >= args.hard_negatives_per_sample:
                break

        remaining_negatives = [
            idx
            for idx, candidate in enumerate(sample.candidates)
            if candidate.quote_id not in gold_ids and idx not in hard_negative_indices
        ]
        rng.shuffle(remaining_negatives)
        random_negative_indices = remaining_negatives[: args.random_negatives_per_sample]

        q_text = query_text(sample)
        for idx in positive_indices:
            examples.append(
                PairExample(
                    q_id=sample.q_id,
                    query=q_text,
                    document=candidate_document_text(sample.candidates[idx]),
                    label=1.0,
                    weight=args.positive_weight * domain_weight,
                )
            )
        for idx in hard_negative_indices:
            examples.append(
                PairExample(
                    q_id=sample.q_id,
                    query=q_text,
                    document=candidate_document_text(sample.candidates[idx]),
                    label=0.0,
                    weight=args.hard_negative_weight * domain_weight,
                )
            )
        for idx in random_negative_indices:
            examples.append(
                PairExample(
                    q_id=sample.q_id,
                    query=q_text,
                    document=candidate_document_text(sample.candidates[idx]),
                    label=0.0,
                    weight=args.random_negative_weight * domain_weight,
                )
            )
    return examples


class PairDataset:
    def __init__(self, examples: Sequence[PairExample]):
        self.examples = list(examples)

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, index: int) -> PairExample:
        return self.examples[index]


class PairCollator:
    def __init__(self, tokenizer, max_length: int):
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __call__(self, batch: Sequence[PairExample]) -> Dict[str, object]:
        queries = [item.query for item in batch]
        documents = [item.document for item in batch]
        encoded = self.tokenizer(
            queries,
            documents,
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        encoded["labels"] = np.asarray([item.label for item in batch], dtype=np.float32)
        encoded["weights"] = np.asarray([item.weight for item in batch], dtype=np.float32)
        return encoded


def iter_batches(dataset: PairDataset, batch_size: int, rng: random.Random) -> Iterable[List[PairExample]]:
    indices = list(range(len(dataset)))
    rng.shuffle(indices)
    for start in range(0, len(indices), batch_size):
        yield [dataset[idx] for idx in indices[start : start + batch_size]]


def extract_logits(logits) -> "np.ndarray":
    if logits.ndim == 1:
        return logits
    if logits.shape[-1] == 1:
        return logits.squeeze(-1)
    if logits.shape[-1] == 2:
        return logits[:, 1]
    return logits.squeeze(-1)


def save_json(path: Path, payload: Dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def load_model_and_tokenizer(args: argparse.Namespace):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    device = hw3.resolve_device(args.device)
    tokenizer = AutoTokenizer.from_pretrained(args.base_model)
    dtype_map = {
        "float32": torch.float32,
        "float16": torch.float16,
    }
    model = AutoModelForSequenceClassification.from_pretrained(
        args.base_model,
        dtype=dtype_map[args.train_dtype],
    )
    model.to(device)
    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config"):
        model.config.use_cache = False
    return model, tokenizer, device


def score_all_candidates(
    model,
    tokenizer,
    samples: Sequence[hw3.Sample],
    device: str,
    batch_size: int,
    max_length: int,
) -> Dict[str, np.ndarray]:
    import torch

    qids: List[str] = []
    all_queries: List[str] = []
    all_documents: List[str] = []
    spans: Dict[str, Tuple[int, int]] = {}
    for sample in samples:
        start = len(all_documents)
        q = query_text(sample)
        docs = [candidate_document_text(candidate) for candidate in sample.candidates]
        all_queries.extend([q] * len(docs))
        all_documents.extend(docs)
        qids.append(sample.q_id)
        spans[sample.q_id] = (start, len(all_documents))

    collected: List[np.ndarray] = []
    autocast_enabled = device.startswith("cuda")
    model.eval()
    for start in range(0, len(all_documents), batch_size):
        batch_queries = all_queries[start : start + batch_size]
        batch_documents = all_documents[start : start + batch_size]
        encoded = tokenizer(
            batch_queries,
            batch_documents,
            padding=True,
            truncation=True,
            max_length=max_length,
            return_tensors="pt",
        )
        encoded = {key: value.to(device) for key, value in encoded.items()}
        with torch.inference_mode():
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled):
                logits = model(**encoded).logits
        scores = extract_logits(logits.detach().float().cpu().numpy())
        collected.append(np.asarray(scores, dtype=np.float32))

    merged = np.concatenate(collected).astype(np.float32) if collected else np.zeros(0, dtype=np.float32)
    return {
        qid: merged[start:end]
        for qid, (start, end) in spans.items()
    }


def predictions_from_score_map(
    samples: Sequence[hw3.Sample],
    score_map: Dict[str, np.ndarray],
) -> Dict[str, List[str]]:
    predictions: Dict[str, List[str]] = {}
    for sample in samples:
        ranked = hw3.stable_rank_indices(score_map[sample.q_id])
        predictions[sample.q_id] = [sample.candidates[idx].quote_id for idx in ranked[:5]]
    return predictions


def evaluate_score_maps(
    samples: Sequence[hw3.Sample],
    reranker_scores: Dict[str, np.ndarray],
    base_scores: Optional[Dict[str, np.ndarray]],
    fusion_weights: Sequence[float],
) -> Dict[str, object]:
    reranker_predictions = predictions_from_score_map(samples, reranker_scores)
    reranker_result = hw3.evaluate_predictions(samples, reranker_predictions, method="reranker_only")

    best_fused: Dict[str, object] = {
        "weight": None,
        "recall_at_5": -1.0,
        "evaluated_samples": 0,
    }
    fused_results: List[Dict[str, object]] = []
    if base_scores is not None:
        for weight in fusion_weights:
            predictions: Dict[str, List[str]] = {}
            for sample in samples:
                rer = hw3.minmax_normalize(reranker_scores[sample.q_id])
                base = hw3.minmax_normalize(base_scores[sample.q_id])
                fused = (1.0 - weight) * base + weight * rer
                ranked = hw3.stable_rank_indices(fused)
                predictions[sample.q_id] = [sample.candidates[idx].quote_id for idx in ranked[:5]]
            result = hw3.evaluate_predictions(samples, predictions, method=f"fusion_{weight:.2f}")
            current = {
                "weight": weight,
                "recall_at_5": round(result.recall_at_5, 6),
                "evaluated_samples": result.evaluated_samples,
            }
            fused_results.append(current)
            if result.recall_at_5 > float(best_fused["recall_at_5"]):
                best_fused = current

    return {
        "reranker_only": {
            "recall_at_5": round(reranker_result.recall_at_5, 6),
            "evaluated_samples": reranker_result.evaluated_samples,
        },
        "best_fused": best_fused,
        "fused_grid": fused_results,
    }


def train_loop(
    model,
    tokenizer,
    device: str,
    train_examples: Sequence[PairExample],
    dev_samples: Sequence[hw3.Sample],
    dev_base_scores: Optional[Dict[str, np.ndarray]],
    args: argparse.Namespace,
) -> Dict[str, object]:
    import torch

    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    dataset = PairDataset(train_examples)
    collator = PairCollator(tokenizer, max_length=args.max_length)

    total_micro_batches = math.ceil(len(dataset) / max(args.train_batch_size, 1))
    total_update_steps = max(1, math.ceil(total_micro_batches / max(args.grad_accum_steps, 1)) * max(args.epochs, 1))

    optimizer = torch.optim.AdamW(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
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

    model_param_dtype = next(model.parameters()).dtype
    use_amp = device.startswith("cuda") and model_param_dtype not in {torch.float16, torch.bfloat16}
    use_scaler = use_amp
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)
    fusion_weights = parse_float_list(args.eval_fusion_weights)
    best_metric = -1.0
    best_summary: Dict[str, object] = {}

    global_step = 0
    for epoch_idx in range(args.epochs):
        model.train()
        running_loss = 0.0
        seen_examples = 0
        optimizer.zero_grad(set_to_none=True)
        for micro_step, batch_examples in enumerate(iter_batches(dataset, args.train_batch_size, rng), start=1):
            batch = collator(batch_examples)
            encoded = {
                key: value.to(device)
                for key, value in batch.items()
                if key not in {"labels", "weights"}
            }
            labels = torch.tensor(batch["labels"], dtype=torch.float32, device=device)
            weights = torch.tensor(batch["weights"], dtype=torch.float32, device=device)
            with torch.autocast(device_type="cuda", dtype=torch.float16, enabled=use_amp):
                logits = model(**encoded).logits
                logits = torch.as_tensor(extract_logits(logits), dtype=torch.float32, device=device)
                loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels, reduction="none")
                loss = (loss * weights).mean()
                loss = loss / max(args.grad_accum_steps, 1)
            if use_scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            running_loss += float(loss.detach().cpu()) * max(args.grad_accum_steps, 1)
            seen_examples += len(batch_examples)

            if micro_step % max(args.grad_accum_steps, 1) == 0 or micro_step == total_micro_batches:
                if use_scaler:
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

        if dev_samples:
            reranker_scores = score_all_candidates(
                model=model,
                tokenizer=tokenizer,
                samples=dev_samples,
                device=device,
                batch_size=args.eval_batch_size,
                max_length=args.max_length,
            )
            dev_summary = evaluate_score_maps(
                samples=dev_samples,
                reranker_scores=reranker_scores,
                base_scores=dev_base_scores,
                fusion_weights=fusion_weights,
            )
            best_fused_recall = float(dev_summary["best_fused"]["recall_at_5"]) if dev_summary["best_fused"]["weight"] is not None else -1.0
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
                model.save_pretrained(output_dir)
                tokenizer.save_pretrained(output_dir)
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
            model.save_pretrained(output_dir)
            tokenizer.save_pretrained(output_dir)
            save_json(output_dir / "best_metrics.json", best_summary)

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
        "best_summary": best_summary,
        "elapsed_seconds": round(time.time() - start_time, 2),
    }
    save_json(Path(args.output_dir).resolve() / "run_summary.json", final_summary)
    print(json.dumps(final_summary, ensure_ascii=False), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
