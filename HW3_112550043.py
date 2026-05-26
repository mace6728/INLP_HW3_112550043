#!/usr/bin/env python3
"""INLP HW3 single-file retrieval pipeline.

This script is intentionally organized as one Python file with clear internal
modules so it can be submitted to E3 as a single source file while still
supporting experiments for:

- BM25 retrieval
- Dense retrieval on text and image descriptions
- Hybrid retrieval
- Optional reranking
- Direct LLM selection baseline
- Raw-image retrieval baseline
- Dev evaluation, ablation, analysis, and Kaggle submission export

Typical usage:

    python3 HW3_112550043.py --mode stats
    python3 HW3_112550043.py --mode eval --method bm25
    python3 HW3_112550043.py --mode eval --method hybrid
    python3 HW3_112550043.py --mode analyze --method hybrid
    python3 HW3_112550043.py --mode submit --method hybrid --output submission.csv

Notes:

- The script defaults to using `img_description` as the textual representation
  for image candidates.
- Heavy methods (`dense`, `llm`, `image`, `reranker`) require extra packages.
- The dataset is a per-question reranking task, so no global vector database is
  used.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import pickle
import random
import re
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np


# ============================================================================
# Data structures
# ============================================================================


@dataclass
class EvidenceCandidate:
    quote_id: str
    modality: str
    raw_type: str
    content: str
    page_id: Optional[int]
    layout_id: Optional[int]
    img_path: Optional[str] = None


@dataclass
class Sample:
    q_id: str
    old_id: Optional[str]
    doc_name: str
    domain: str
    question: str
    question_type: str
    evidence_modality_type: List[str]
    gold_quotes: List[str] = field(default_factory=list)
    candidates: List[EvidenceCandidate] = field(default_factory=list)


@dataclass
class EvalResult:
    method: str
    recall_at_5: float
    evaluated_samples: int
    skipped_samples: int
    extra: Dict[str, Any] = field(default_factory=dict)


@dataclass
class GoldMemory:
    gold_counts: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    candidate_counts: Dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))


# ============================================================================
# Utility helpers
# ============================================================================


def safe_strip(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def normalize_text(text: str) -> str:
    text = safe_strip(text)
    replacements = {
        "\u00a0": " ",
        "\n": " ",
        "\r": " ",
        "\t": " ",
        "\\%": "%",
        "\\$": "$",
        "\\_": "_",
        "$": " ",
    }
    for src, dst in replacements.items():
        text = text.replace(src, dst)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def tokenize_for_bm25(text: str) -> List[str]:
    return re.findall(r"[A-Za-z0-9]+(?:[._/-][A-Za-z0-9]+)?", normalize_text(text).lower())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def sort_key_for_qid(qid: str) -> Tuple[int, str]:
    try:
        return (0, int(qid))
    except (TypeError, ValueError):
        return (1, str(qid))


def minmax_normalize(scores: Sequence[float]) -> np.ndarray:
    arr = np.asarray(scores, dtype=np.float32)
    if arr.size == 0:
        return arr
    min_value = float(arr.min())
    max_value = float(arr.max())
    if math.isclose(min_value, max_value):
        return np.zeros_like(arr)
    return (arr - min_value) / (max_value - min_value)


def stable_rank_indices(scores: Sequence[float]) -> List[int]:
    indexed = list(enumerate(scores))
    indexed.sort(key=lambda item: (-float(item[1]), item[0]))
    return [idx for idx, _ in indexed]


def apply_context_boost(
    sample: Sample,
    scores: np.ndarray,
    page_weight: float = 0.0,
    layout_weight: float = 0.0,
) -> np.ndarray:
    if page_weight <= 0.0 and layout_weight <= 0.0:
        return scores

    adjusted = np.asarray(scores, dtype=np.float32).copy()
    if page_weight > 0.0:
        page_best: Dict[Any, float] = {}
        for idx, candidate in enumerate(sample.candidates):
            if candidate.page_id is None:
                continue
            page_best[candidate.page_id] = max(page_best.get(candidate.page_id, -1e9), float(scores[idx]))
        page_scores = np.asarray(
            [page_best.get(candidate.page_id, 0.0) for candidate in sample.candidates],
            dtype=np.float32,
        )
        adjusted += page_weight * minmax_normalize(page_scores)

    if layout_weight > 0.0:
        layout_best: Dict[Any, float] = {}
        for idx, candidate in enumerate(sample.candidates):
            if candidate.layout_id is None:
                continue
            layout_best[candidate.layout_id] = max(layout_best.get(candidate.layout_id, -1e9), float(scores[idx]))
        layout_scores = np.asarray(
            [layout_best.get(candidate.layout_id, 0.0) for candidate in sample.candidates],
            dtype=np.float32,
        )
        adjusted += layout_weight * minmax_normalize(layout_scores)

    return adjusted


def apply_shard(samples: Sequence[Sample], shard_id: int, num_shards: int) -> List[Sample]:
    if num_shards <= 1:
        return list(samples)
    if shard_id < 0 or shard_id >= num_shards:
        raise ValueError(f"Invalid shard_id={shard_id}; expected 0 <= shard_id < {num_shards}")
    return [sample for idx, sample in enumerate(samples) if idx % num_shards == shard_id]


def build_cache_key(prefix: str, model_name: str, texts: Sequence[str]) -> str:
    digest = hashlib.sha256()
    digest.update(prefix.encode("utf-8"))
    digest.update(model_name.encode("utf-8"))
    for text in texts:
        digest.update(text.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest()[:24]


# ============================================================================
# IO and dataset parsing
# ============================================================================


def resolve_img_path(img_path: str, data_root: Path) -> str:
    raw = Path(img_path)
    candidates = [
        data_root / raw,
        data_root / "images" / raw.name,
        data_root / "images" / "images" / raw.name,
        data_root / raw.name,
    ]
    for candidate in candidates:
        if candidate.exists():
            return str(candidate.resolve())
    return str((data_root / raw).resolve())


def candidate_to_text(candidate: EvidenceCandidate) -> str:
    prefix_map = {
        "text": "[TEXT]",
        "table": "[TABLE]",
        "figure": "[FIGURE]",
        "chart": "[CHART]",
        "image": "[IMAGE]",
        "layout": "[LAYOUT]",
    }
    prefix = prefix_map.get(candidate.raw_type.lower(), f"[{candidate.raw_type.upper()}]")
    content = normalize_text(candidate.content)
    return f"{prefix} {content}".strip()


def expected_modality_key(sample: Sample) -> Tuple[str, ...]:
    return tuple(sorted(modality.lower() for modality in sample.evidence_modality_type if modality))


def candidate_type_matches_expected(sample: Sample, candidate: EvidenceCandidate) -> bool:
    expected = set(expected_modality_key(sample))
    if not expected:
        return False
    raw_type = candidate.raw_type.lower()
    if raw_type in expected:
        return True
    return candidate.modality == "text" and "text" in expected


def expected_type_score_candidates(sample: Sample, candidates: Sequence[EvidenceCandidate]) -> np.ndarray:
    if not sample.evidence_modality_type:
        return np.zeros(len(candidates), dtype=np.float32)
    return np.asarray(
        [1.0 if candidate_type_matches_expected(sample, candidate) else 0.0 for candidate in candidates],
        dtype=np.float32,
    )


def flatten_candidates_from_raw(raw_text_quotes: Sequence[dict], raw_img_quotes: Sequence[dict], data_root: Path) -> List[EvidenceCandidate]:
    candidates: List[EvidenceCandidate] = []
    for quote in raw_text_quotes:
        candidates.append(
            EvidenceCandidate(
                quote_id=safe_strip(quote.get("quote_id")),
                modality="text",
                raw_type=safe_strip(quote.get("type")) or "text",
                content=normalize_text(quote.get("text", "")),
                page_id=quote.get("page_id"),
                layout_id=quote.get("layout_id"),
            )
        )
    for quote in raw_img_quotes:
        candidates.append(
            EvidenceCandidate(
                quote_id=safe_strip(quote.get("quote_id")),
                modality="image",
                raw_type=safe_strip(quote.get("type")) or "image",
                content=normalize_text(quote.get("img_description", "")),
                page_id=quote.get("page_id"),
                layout_id=quote.get("layout_id"),
                img_path=resolve_img_path(safe_strip(quote.get("img_path", "")), data_root),
            )
        )
    return candidates


def flatten_candidates(sample: Sample) -> List[EvidenceCandidate]:
    return list(sample.candidates)


def load_jsonl(path: Path) -> List[dict]:
    rows: List[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_samples(path: Path, data_root: Path) -> List[Sample]:
    rows = load_jsonl(path)
    samples: List[Sample] = []
    for row in rows:
        sample = Sample(
            q_id=safe_strip(row.get("q_id")),
            old_id=safe_strip(row.get("old_id")) or None,
            doc_name=safe_strip(row.get("doc_name")),
            domain=safe_strip(row.get("domain")),
            question=normalize_text(row.get("question", "")),
            question_type=safe_strip(row.get("question_type")),
            evidence_modality_type=[safe_strip(x) for x in row.get("evidence_modality_type", [])],
            gold_quotes=[safe_strip(x) for x in row.get("gold_quotes", [])],
            candidates=flatten_candidates_from_raw(row.get("text_quotes", []), row.get("img_quotes", []), data_root),
        )
        samples.append(sample)
    return samples


def write_submission(output_path: Path, samples: Sequence[Sample], predictions: Dict[str, List[str]]) -> None:
    ensure_dir(output_path.parent)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["q_id", "gold_quotes"])
        for sample in samples:
            pred_ids = predictions.get(sample.q_id, [])[:5]
            writer.writerow([sample.q_id, " ".join(pred_ids)])


# ============================================================================
# Split and evaluation
# ============================================================================


def group_split_by_doc(samples: Sequence[Sample], dev_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    doc_to_samples: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        doc_to_samples[sample.doc_name].append(sample)

    doc_names = list(doc_to_samples.keys())
    rng = random.Random(seed)
    rng.shuffle(doc_names)

    dev_doc_count = max(1, int(round(len(doc_names) * dev_ratio)))
    dev_doc_count = min(dev_doc_count, len(doc_names) - 1) if len(doc_names) > 1 else 1
    dev_docs = set(doc_names[:dev_doc_count])

    train_split: List[Sample] = []
    dev_split: List[Sample] = []
    for doc_name, grouped_samples in doc_to_samples.items():
        if doc_name in dev_docs:
            dev_split.extend(grouped_samples)
        else:
            train_split.extend(grouped_samples)
    return train_split, dev_split


def row_split_samples(samples: Sequence[Sample], dev_ratio: float, seed: int) -> Tuple[List[Sample], List[Sample]]:
    indexed_samples = list(samples)
    rng = random.Random(seed)
    rng.shuffle(indexed_samples)
    dev_count = max(1, int(round(len(indexed_samples) * dev_ratio)))
    dev_count = min(dev_count, len(indexed_samples) - 1) if len(indexed_samples) > 1 else 1
    dev_split = indexed_samples[:dev_count]
    train_split = indexed_samples[dev_count:]
    return train_split, dev_split


def mixed_overlap_split_samples(
    samples: Sequence[Sample],
    dev_ratio: float,
    seed: int,
    overlap_ratio: float,
) -> Tuple[List[Sample], List[Sample]]:
    doc_to_samples: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        doc_to_samples[sample.doc_name].append(sample)

    total_dev = max(1, int(round(len(samples) * dev_ratio)))
    total_dev = min(total_dev, len(samples) - 1) if len(samples) > 1 else 1
    target_seen_dev = int(round(total_dev * overlap_ratio))
    target_unseen_dev = max(0, total_dev - target_seen_dev)

    rng = random.Random(seed)
    doc_names = list(doc_to_samples.keys())
    rng.shuffle(doc_names)

    unseen_dev: List[Sample] = []
    unseen_docs: set[str] = set()
    for doc_name in doc_names:
        if len(unseen_dev) >= target_unseen_dev:
            break
        grouped = list(doc_to_samples[doc_name])
        unseen_docs.add(doc_name)
        unseen_dev.extend(grouped)

    row_candidates: Dict[str, List[Sample]] = {}
    for doc_name, grouped in doc_to_samples.items():
        if doc_name in unseen_docs or len(grouped) <= 1:
            continue
        shuffled = list(grouped)
        rng.shuffle(shuffled)
        row_candidates[doc_name] = shuffled

    seen_dev: List[Sample] = []
    eligible_docs = [doc_name for doc_name, grouped in row_candidates.items() if len(grouped) > 1]
    while len(seen_dev) < target_seen_dev and eligible_docs:
        doc_name = rng.choice(eligible_docs)
        grouped = row_candidates[doc_name]
        seen_dev.append(grouped.pop())
        eligible_docs = [name for name, rows in row_candidates.items() if len(rows) > 1]

    dev_ids = {sample.q_id for sample in unseen_dev}
    dev_ids.update(sample.q_id for sample in seen_dev)
    dev_split = unseen_dev + seen_dev
    train_split = [sample for sample in samples if sample.q_id not in dev_ids]

    if not train_split:
        return row_split_samples(samples, dev_ratio=dev_ratio, seed=seed)
    return train_split, dev_split


def domain_holdout_split_samples(
    samples: Sequence[Sample],
    holdout_domain: str,
) -> Tuple[List[Sample], List[Sample]]:
    normalized = safe_strip(holdout_domain).replace("_", " ")
    if not normalized:
        raise ValueError("domain holdout split requires a non-empty --holdout-domain")

    train_split = [sample for sample in samples if sample.domain != normalized]
    dev_split = [sample for sample in samples if sample.domain == normalized]
    if not train_split or not dev_split:
        raise ValueError(
            f"domain holdout split could not create both train/dev for domain: {normalized}"
        )
    return train_split, dev_split


def make_eval_split(
    samples: Sequence[Sample],
    dev_ratio: float,
    seed: int,
    split_strategy: str,
    mixed_overlap_ratio: float = 0.74,
    holdout_domain: str = "",
) -> Tuple[List[Sample], List[Sample]]:
    if split_strategy == "doc":
        return group_split_by_doc(samples, dev_ratio=dev_ratio, seed=seed)
    if split_strategy == "row":
        return row_split_samples(samples, dev_ratio=dev_ratio, seed=seed)
    if split_strategy == "mixed":
        return mixed_overlap_split_samples(
            samples,
            dev_ratio=dev_ratio,
            seed=seed,
            overlap_ratio=mixed_overlap_ratio,
        )
    if split_strategy == "domain":
        return domain_holdout_split_samples(samples, holdout_domain=holdout_domain)
    raise ValueError(f"Unsupported eval split: {split_strategy}")


def recall_at_5(predicted_ids: Sequence[str], gold_ids: Sequence[str]) -> Optional[float]:
    gold = [safe_strip(x) for x in gold_ids if safe_strip(x)]
    if not gold:
        return None
    pred_set = set(predicted_ids[:5])
    hits = sum(1 for gold_id in gold if gold_id in pred_set)
    return hits / len(gold)


def evaluate_predictions(samples: Sequence[Sample], predictions: Dict[str, List[str]], method: str) -> EvalResult:
    per_sample_recalls: List[float] = []
    skipped = 0
    for sample in samples:
        score = recall_at_5(predictions.get(sample.q_id, []), sample.gold_quotes)
        if score is None:
            skipped += 1
            continue
        per_sample_recalls.append(score)
    mean_recall = float(np.mean(per_sample_recalls)) if per_sample_recalls else 0.0
    return EvalResult(
        method=method,
        recall_at_5=mean_recall,
        evaluated_samples=len(per_sample_recalls),
        skipped_samples=skipped,
    )


def doc_frequency_bucket(doc_count: int) -> str:
    if doc_count <= 0:
        return "unseen"
    if doc_count <= 2:
        return "seen_1_2"
    if doc_count <= 4:
        return "seen_3_4"
    if doc_count <= 9:
        return "seen_5_9"
    return "seen_10_plus"


def bucket_proportions(samples: Sequence[Sample], doc_counts: Dict[str, int]) -> Dict[str, float]:
    bucket_counts: Counter[str] = Counter()
    for sample in samples:
        bucket_counts[doc_frequency_bucket(int(doc_counts.get(sample.doc_name, 0)))] += 1

    total = sum(bucket_counts.values())
    if total <= 0:
        return {}
    return {
        bucket: (count / total)
        for bucket, count in sorted(bucket_counts.items())
    }


def bucket_proportions_from_map(bucket_by_qid: Dict[str, str]) -> Dict[str, float]:
    bucket_counts = Counter(bucket_by_qid.values())
    total = sum(bucket_counts.values())
    if total <= 0:
        return {}
    return {
        bucket: (count / total)
        for bucket, count in sorted(bucket_counts.items())
    }


def evaluate_predictions_by_bucket_map(
    samples: Sequence[Sample],
    predictions: Dict[str, List[str]],
    method: str,
    bucket_by_qid: Dict[str, str],
) -> Dict[str, Dict[str, Any]]:
    grouped_scores: Dict[str, List[float]] = defaultdict(list)
    grouped_skips: Counter[str] = Counter()
    grouped_total: Counter[str] = Counter()

    for sample in samples:
        bucket = bucket_by_qid.get(sample.q_id, "unknown")
        grouped_total[bucket] += 1
        score = recall_at_5(predictions.get(sample.q_id, []), sample.gold_quotes)
        if score is None:
            grouped_skips[bucket] += 1
            continue
        grouped_scores[bucket].append(score)

    summary: Dict[str, Dict[str, Any]] = {}
    for bucket in sorted(grouped_total.keys()):
        recalls = grouped_scores.get(bucket, [])
        summary[bucket] = {
            "method": method,
            "recall_at_5": float(np.mean(recalls)) if recalls else 0.0,
            "evaluated_samples": len(recalls),
            "skipped_samples": int(grouped_skips.get(bucket, 0)),
            "total_samples": int(grouped_total[bucket]),
        }
    return summary


def evaluate_predictions_by_doc_frequency(
    samples: Sequence[Sample],
    predictions: Dict[str, List[str]],
    method: str,
    doc_counts: Dict[str, int],
) -> Dict[str, Dict[str, Any]]:
    bucket_by_qid = {
        sample.q_id: doc_frequency_bucket(int(doc_counts.get(sample.doc_name, 0)))
        for sample in samples
    }
    return evaluate_predictions_by_bucket_map(
        samples,
        predictions,
        method=method,
        bucket_by_qid=bucket_by_qid,
    )


def weighted_bucket_recall(
    bucket_metrics: Dict[str, Dict[str, Any]],
    bucket_weights: Dict[str, float],
) -> float:
    if not bucket_weights:
        return 0.0

    weighted = 0.0
    covered = 0.0
    for bucket, weight in bucket_weights.items():
        if bucket not in bucket_metrics:
            continue
        weighted += weight * float(bucket_metrics[bucket]["recall_at_5"])
        covered += weight

    if covered <= 0.0:
        return 0.0
    return weighted / covered


def strict_weighted_bucket_recall(
    bucket_metrics: Dict[str, Dict[str, Any]],
    bucket_weights: Dict[str, float],
) -> float:
    return sum(
        weight * float(bucket_metrics.get(bucket, {}).get("recall_at_5", 0.0))
        for bucket, weight in bucket_weights.items()
    )


def bucket_coverage(
    bucket_metrics: Dict[str, Dict[str, Any]],
    bucket_weights: Dict[str, float],
) -> float:
    return sum(
        weight
        for bucket, weight in bucket_weights.items()
        if bucket in bucket_metrics
    )


def domain_by_qid(samples: Sequence[Sample]) -> Dict[str, str]:
    return {
        sample.q_id: sample.domain or "unknown"
        for sample in samples
    }


def question_type_by_qid(samples: Sequence[Sample]) -> Dict[str, str]:
    return {
        sample.q_id: sample.question_type or "unknown"
        for sample in samples
    }


def gold_len_bucket_by_qid(samples: Sequence[Sample]) -> Dict[str, str]:
    bucket_by_qid: Dict[str, str] = {}
    for sample in samples:
        gold_len = len([quote_id for quote_id in sample.gold_quotes if safe_strip(quote_id)])
        bucket_by_qid[sample.q_id] = str(gold_len)
    return bucket_by_qid


def gold_modality_bucket_by_qid(samples: Sequence[Sample]) -> Dict[str, str]:
    bucket_by_qid: Dict[str, str] = {}
    for sample in samples:
        gold_ids = set(sample.gold_quotes)
        has_text = any(candidate.quote_id in gold_ids and candidate.modality == "text" for candidate in sample.candidates)
        has_image = any(candidate.quote_id in gold_ids and candidate.modality == "image" for candidate in sample.candidates)
        if has_text and has_image:
            bucket = "mixed"
        elif has_image:
            bucket = "image_only"
        elif has_text:
            bucket = "text_only"
        else:
            bucket = "none"
        bucket_by_qid[sample.q_id] = bucket
    return bucket_by_qid


def build_candidate_lookup(samples: Sequence[Sample]) -> Dict[str, Dict[str, EvidenceCandidate]]:
    lookup: Dict[str, Dict[str, EvidenceCandidate]] = {}
    for sample in samples:
        lookup[sample.q_id] = {candidate.quote_id: candidate for candidate in sample.candidates}
    return lookup


# ============================================================================
# BM25 baseline
# ============================================================================


def bm25_score_texts(question: str, texts: Sequence[str], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    query_tokens = tokenize_for_bm25(question)
    tokenized_docs = [tokenize_for_bm25(text) for text in texts]

    if not query_tokens or not tokenized_docs:
        return np.zeros(len(texts), dtype=np.float32)

    df: Counter[str] = Counter()
    for doc_tokens in tokenized_docs:
        df.update(set(doc_tokens))

    query_counts = Counter(query_tokens)
    doc_lengths = [len(tokens) for tokens in tokenized_docs]
    avgdl = sum(doc_lengths) / max(len(doc_lengths), 1)
    n_docs = len(tokenized_docs)

    scores = np.zeros(n_docs, dtype=np.float32)
    for idx, doc_tokens in enumerate(tokenized_docs):
        token_counts = Counter(doc_tokens)
        dl = max(len(doc_tokens), 1)
        score = 0.0
        for term, qf in query_counts.items():
            term_freq = token_counts.get(term, 0)
            if term_freq == 0:
                continue
            term_df = df.get(term, 0)
            idf = math.log(1.0 + ((n_docs - term_df + 0.5) / (term_df + 0.5)))
            denom = term_freq + k1 * (1.0 - b + b * dl / max(avgdl, 1e-9))
            score += qf * idf * ((term_freq * (k1 + 1.0)) / denom)
        scores[idx] = score
    return scores


def bm25_score_candidates(question: str, candidates: Sequence[EvidenceCandidate], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    return bm25_score_texts(
        question,
        [candidate_to_text(candidate) for candidate in candidates],
        k1=k1,
        b=b,
    )


def bm25_retrieve_one(sample: Sample, top_k: int = 5) -> List[str]:
    scores = bm25_score_candidates(sample.question, sample.candidates)
    ranked_indices = stable_rank_indices(scores)
    return [sample.candidates[idx].quote_id for idx in ranked_indices[:top_k]]


def bm25_retrieve_batch(samples: Sequence[Sample], top_k: int = 5) -> Dict[str, List[str]]:
    return {sample.q_id: bm25_retrieve_one(sample, top_k=top_k) for sample in samples}


# ============================================================================
# Optional HF helpers
# ============================================================================


def resolve_device(requested_device: str) -> str:
    if requested_device != "auto":
        return requested_device
    try:
        import torch  # type: ignore

        if torch.cuda.is_available():
            return "cuda"
    except Exception:
        pass
    return "cpu"


def torch_dtype_for_device(device: str):
    import torch  # type: ignore

    if device.startswith("cuda"):
        return torch.float16
    return torch.float32


def save_pickle(path: Path, payload: Any) -> None:
    ensure_dir(path.parent)
    with path.open("wb") as handle:
        pickle.dump(payload, handle)


def load_pickle(path: Path) -> Any:
    with path.open("rb") as handle:
        return pickle.load(handle)


class HFTextEmbedder:
    def __init__(self, model_name: str, device: str = "auto", max_length: int = 512):
        self.model_name = model_name
        self.device = resolve_device(device)
        self.max_length = max_length
        self._backend = None

        try:
            from sentence_transformers import SentenceTransformer  # type: ignore

            self.model = SentenceTransformer(model_name, device=self.device)
            self._backend = "sentence_transformers"
            self.tokenizer = None
        except ImportError:
            try:
                import torch  # type: ignore
                from transformers import AutoModel, AutoTokenizer  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "Dense retrieval requires either `sentence-transformers` or "
                    "`transformers` + `torch`. Install them in ~/College/.venv first."
                ) from exc
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name, torch_dtype=torch_dtype_for_device(self.device))
            self.model.to(self.device)
            self.model.eval()
            self._backend = "transformers"

    def encode(self, texts: Sequence[str], batch_size: int = 32) -> np.ndarray:
        texts = [normalize_text(text) for text in texts]
        if not texts:
            return np.zeros((0, 1), dtype=np.float32)

        if self._backend == "sentence_transformers":
            embeddings = self.model.encode(
                list(texts),
                batch_size=batch_size,
                convert_to_numpy=True,
                normalize_embeddings=True,
                show_progress_bar=False,
            )
            return embeddings.astype(np.float32)

        import torch  # type: ignore

        embeddings: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch_texts = list(texts[start : start + batch_size])
            encoded = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                outputs = self.model(**encoded)
                hidden = outputs.last_hidden_state
                mask = encoded["attention_mask"].unsqueeze(-1).expand(hidden.size()).float()
                pooled = (hidden * mask).sum(dim=1) / torch.clamp(mask.sum(dim=1), min=1e-9)
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embeddings.append(pooled.cpu().numpy().astype(np.float32))
        return np.vstack(embeddings)


class HFReranker:
    def __init__(self, model_name: str, device: str = "auto", max_length: int = 512):
        try:
            import torch  # type: ignore
            from transformers import AutoModelForSequenceClassification, AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Reranking requires `transformers` + `torch` in ~/College/.venv."
            ) from exc

        self.device = resolve_device(device)
        self.max_length = max_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(
            model_name,
            torch_dtype=torch_dtype_for_device(self.device),
        )
        self.model.to(self.device)
        self.model.eval()

    def score_pairs(self, query: str, documents: Sequence[str], batch_size: int = 8) -> np.ndarray:
        import torch  # type: ignore

        scores: List[np.ndarray] = []
        repeated_queries = [normalize_text(query)] * len(documents)
        documents = [normalize_text(doc) for doc in documents]
        for start in range(0, len(documents), batch_size):
            batch_queries = repeated_queries[start : start + batch_size]
            batch_docs = list(documents[start : start + batch_size])
            encoded = self.tokenizer(
                batch_queries,
                batch_docs,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits = self.model(**encoded).logits
                logits = logits.squeeze(-1)
            scores.append(logits.detach().float().cpu().numpy())
        return np.concatenate(scores).astype(np.float32) if scores else np.zeros(0, dtype=np.float32)


class HFLLMSelector:
    def __init__(self, model_name: str, device: str = "auto", max_input_length: int = 4096):
        try:
            import torch  # type: ignore
            from transformers import AutoModelForCausalLM, AutoTokenizer  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "LLM baseline requires `transformers` + `torch` in ~/College/.venv."
            ) from exc

        self.device = resolve_device(device)
        self.max_input_length = max_input_length
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch_dtype_for_device(self.device),
        )
        self.model.to(self.device)
        self.model.eval()

    def generate(self, system_prompt: str, user_prompt: str, max_new_tokens: int = 128) -> str:
        import torch  # type: ignore

        if hasattr(self.tokenizer, "apply_chat_template") and self.tokenizer.chat_template:
            prompt = self.tokenizer.apply_chat_template(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                tokenize=False,
                add_generation_prompt=True,
            )
        else:
            prompt = f"System: {system_prompt}\nUser: {user_prompt}\nAssistant:"

        encoded = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=self.max_input_length,
        )
        encoded = {key: value.to(self.device) for key, value in encoded.items()}
        prompt_length = encoded["input_ids"].shape[1]
        with torch.inference_mode():
            generated = self.model.generate(
                **encoded,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                pad_token_id=self.tokenizer.pad_token_id,
            )
        completion = generated[0][prompt_length:]
        return self.tokenizer.decode(completion, skip_special_tokens=True).strip()


class HFClipRetriever:
    def __init__(self, model_name: str, device: str = "auto"):
        try:
            import torch  # type: ignore
            from PIL import Image  # type: ignore
            from transformers import CLIPModel, CLIPProcessor  # type: ignore
        except ImportError as exc:
            raise RuntimeError(
                "Raw image retrieval requires `transformers`, `torch`, and `pillow`."
            ) from exc

        self.Image = Image
        self.device = resolve_device(device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model = CLIPModel.from_pretrained(model_name, torch_dtype=torch_dtype_for_device(self.device))
        self.model.to(self.device)
        self.model.eval()

    def encode_texts(self, texts: Sequence[str], batch_size: int = 16) -> np.ndarray:
        import torch  # type: ignore

        vectors: List[np.ndarray] = []
        for start in range(0, len(texts), batch_size):
            batch = list(texts[start : start + batch_size])
            inputs = self.processor(text=batch, return_tensors="pt", padding=True, truncation=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                features = self.model.get_text_features(**inputs)
                features = torch.nn.functional.normalize(features, p=2, dim=1)
            vectors.append(features.cpu().numpy().astype(np.float32))
        return np.vstack(vectors) if vectors else np.zeros((0, 1), dtype=np.float32)

    def encode_images(self, image_paths: Sequence[str], batch_size: int = 8) -> np.ndarray:
        import torch  # type: ignore

        vectors: List[np.ndarray] = []
        for start in range(0, len(image_paths), batch_size):
            batch_paths = list(image_paths[start : start + batch_size])
            images = [self.Image.open(path).convert("RGB") for path in batch_paths]
            inputs = self.processor(images=images, return_tensors="pt", padding=True)
            inputs = {key: value.to(self.device) for key, value in inputs.items()}
            with torch.inference_mode():
                features = self.model.get_image_features(**inputs)
                features = torch.nn.functional.normalize(features, p=2, dim=1)
            vectors.append(features.cpu().numpy().astype(np.float32))
        return np.vstack(vectors) if vectors else np.zeros((0, 1), dtype=np.float32)


# ============================================================================
# Dense retrieval and hybrid retrieval
# ============================================================================


def maybe_load_cached_embeddings(cache_path: Optional[Path]) -> Optional[dict]:
    if cache_path is None or not cache_path.exists():
        return None
    return load_pickle(cache_path)


def dense_score_batch(
    samples: Sequence[Sample],
    model_name: str,
    device: str,
    batch_size: int,
    cache_dir: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    query_texts = [sample.question for sample in samples]
    candidate_texts: List[str] = []
    spans: Dict[str, Tuple[int, int]] = {}
    for sample in samples:
        start = len(candidate_texts)
        candidate_texts.extend(candidate_to_text(candidate) for candidate in sample.candidates)
        spans[sample.q_id] = (start, len(candidate_texts))

    cache_path = None
    if cache_dir is not None:
        ensure_dir(cache_dir)
        cache_key = build_cache_key("dense", model_name, query_texts + candidate_texts)
        cache_path = cache_dir / f"dense_{cache_key}.pkl"
        cached = maybe_load_cached_embeddings(cache_path)
        if cached is not None:
            return {qid: np.asarray(scores, dtype=np.float32) for qid, scores in cached.items()}

    embedder = HFTextEmbedder(model_name=model_name, device=device)
    query_embeddings = embedder.encode(query_texts, batch_size=batch_size)
    candidate_embeddings = embedder.encode(candidate_texts, batch_size=batch_size)

    score_map: Dict[str, np.ndarray] = {}
    for sample_idx, sample in enumerate(samples):
        start, end = spans[sample.q_id]
        sample_scores = candidate_embeddings[start:end] @ query_embeddings[sample_idx]
        score_map[sample.q_id] = sample_scores.astype(np.float32)

    if cache_path is not None:
        serializable = {qid: scores.tolist() for qid, scores in score_map.items()}
        save_pickle(cache_path, serializable)
    return score_map


def dense_retrieve_one(sample: Sample, scores: np.ndarray, top_k: int = 5) -> List[str]:
    ranked_indices = stable_rank_indices(scores)
    return [sample.candidates[idx].quote_id for idx in ranked_indices[:top_k]]


def dense_retrieve_batch(
    samples: Sequence[Sample],
    model_name: str,
    device: str,
    batch_size: int,
    top_k: int = 5,
    cache_dir: Optional[Path] = None,
) -> Dict[str, List[str]]:
    score_map = dense_score_batch(samples, model_name=model_name, device=device, batch_size=batch_size, cache_dir=cache_dir)
    return {sample.q_id: dense_retrieve_one(sample, score_map[sample.q_id], top_k=top_k) for sample in samples}


def rerank_order(
    sample: Sample,
    base_scores: np.ndarray,
    reranker: HFReranker,
    rerank_topk: int,
    reranker_weight: float,
    batch_size: int,
) -> List[int]:
    base_order = stable_rank_indices(base_scores)
    top_indices = base_order[: min(rerank_topk, len(base_order))]
    top_documents = [candidate_to_text(sample.candidates[idx]) for idx in top_indices]
    rerank_scores = reranker.score_pairs(sample.question, top_documents, batch_size=batch_size)
    rerank_norm = minmax_normalize(rerank_scores)
    base_norm = minmax_normalize(base_scores[top_indices])
    fused = (1.0 - reranker_weight) * base_norm + reranker_weight * rerank_norm
    reranked_top_indices = [top_indices[idx] for idx in stable_rank_indices(fused)]
    top_index_set = set(top_indices)
    remainder = [idx for idx in base_order if idx not in top_index_set]
    return reranked_top_indices + remainder


def hybrid_retrieve_batch(
    samples: Sequence[Sample],
    dense_model_name: str,
    device: str,
    batch_size: int,
    top_k: int,
    bm25_weight: float,
    dense_weight: float,
    type_weight: float,
    page_weight: float,
    layout_weight: float,
    use_reranker: bool,
    reranker_model_name: str,
    rerank_topk: int,
    reranker_weight: float,
    cache_dir: Optional[Path] = None,
) -> Dict[str, List[str]]:
    bm25_score_map = {sample.q_id: bm25_score_candidates(sample.question, sample.candidates) for sample in samples}
    dense_score_map = dense_score_batch(samples, model_name=dense_model_name, device=device, batch_size=batch_size, cache_dir=cache_dir)
    reranker = HFReranker(reranker_model_name, device=device) if use_reranker else None

    predictions: Dict[str, List[str]] = {}
    for sample in samples:
        bm25_scores = minmax_normalize(bm25_score_map[sample.q_id])
        dense_scores = minmax_normalize(dense_score_map[sample.q_id])
        type_scores = minmax_normalize(expected_type_score_candidates(sample, sample.candidates))
        fused_scores = (
            (bm25_weight * bm25_scores)
            + (dense_weight * dense_scores)
            + (type_weight * type_scores)
        )
        fused_scores = apply_context_boost(
            sample,
            fused_scores,
            page_weight=page_weight,
            layout_weight=layout_weight,
        )
        if reranker is not None:
            ranked_indices = rerank_order(
                sample=sample,
                base_scores=fused_scores,
                reranker=reranker,
                rerank_topk=rerank_topk,
                reranker_weight=reranker_weight,
                batch_size=max(1, min(batch_size, 8)),
            )
        else:
            ranked_indices = stable_rank_indices(fused_scores)
        predictions[sample.q_id] = [sample.candidates[idx].quote_id for idx in ranked_indices[:top_k]]
    return predictions


# ============================================================================
# Train-gold memory retrieval
# ============================================================================


def content_key(text: str) -> str:
    return normalize_text(text).lower()


def memory_keys_for_candidate(sample: Sample, candidate: EvidenceCandidate) -> Dict[str, Tuple[Any, ...]]:
    raw_type = candidate.raw_type.lower()
    content = content_key(candidate.content)
    expected_key = expected_modality_key(sample)
    keys: Dict[str, Tuple[Any, ...]] = {
        "doc_page_layout_type": (sample.doc_name, candidate.page_id, candidate.layout_id, raw_type),
        "doc_page_layout": (sample.doc_name, candidate.page_id, candidate.layout_id),
        "doc_page_type": (sample.doc_name, candidate.page_id, raw_type),
        "doc_layout_type": (sample.doc_name, candidate.layout_id, raw_type),
        "doc_page": (sample.doc_name, candidate.page_id),
        "doc_layout": (sample.doc_name, candidate.layout_id),
        "doc_type": (sample.doc_name, raw_type),
        "quote_id": (candidate.quote_id,),
        "raw_type": (raw_type,),
        "modality": (candidate.modality,),
        "expected_raw_type": (expected_key, raw_type),
        "expected_modality": (expected_key, candidate.modality),
        "expected_quote_id": (expected_key, candidate.quote_id),
        "question_type_raw_type": (sample.question_type, raw_type),
        "domain_raw_type": (sample.domain, raw_type),
    }
    if content:
        keys["doc_content"] = (sample.doc_name, content)
        keys["global_content"] = (content,)
    return keys


def build_gold_memory(samples: Sequence[Sample]) -> GoldMemory:
    memory = GoldMemory()
    for sample in samples:
        gold_ids = set(sample.gold_quotes)
        for candidate in sample.candidates:
            keys = memory_keys_for_candidate(sample, candidate)
            for key_name, key_value in keys.items():
                memory.candidate_counts[key_name][key_value] += 1
                if candidate.quote_id in gold_ids:
                    memory.gold_counts[key_name][key_value] += 1
    return memory


def memory_precision(
    memory: GoldMemory,
    key_name: str,
    key_value: Optional[Tuple[Any, ...]],
    subtract_memory: Optional[GoldMemory] = None,
) -> float:
    if key_value is None:
        return 0.0
    gold_count = memory.gold_counts[key_name].get(key_value, 0)
    candidate_count = memory.candidate_counts[key_name].get(key_value, 0)
    if subtract_memory is not None:
        gold_count -= subtract_memory.gold_counts[key_name].get(key_value, 0)
        candidate_count -= subtract_memory.candidate_counts[key_name].get(key_value, 0)
    if gold_count <= 0 or candidate_count <= 0:
        return 0.0
    return gold_count / candidate_count


def memory_support(
    memory: GoldMemory,
    key_name: str,
    key_value: Optional[Tuple[Any, ...]],
    subtract_memory: Optional[GoldMemory] = None,
) -> float:
    if key_value is None:
        return 0.0
    gold_count = memory.gold_counts[key_name].get(key_value, 0)
    if subtract_memory is not None:
        gold_count -= subtract_memory.gold_counts[key_name].get(key_value, 0)
    if gold_count <= 0:
        return 0.0
    return math.log1p(gold_count)


def memory_score_candidates(
    sample: Sample,
    candidates: Sequence[EvidenceCandidate],
    memory: GoldMemory,
    subtract_memory: Optional[GoldMemory] = None,
) -> np.ndarray:
    # Strong keys are precise evidence identity signals inside an overlapped
    # document; broad keys remain small priors so they do not drown relevance.
    key_weights = {
        "doc_content": 1.35,
        "doc_page_layout_type": 1.15,
        "doc_page_layout": 0.85,
        "doc_page_type": 0.55,
        "doc_layout_type": 0.45,
        "global_content": 0.25,
        "doc_page": 0.18,
        "doc_layout": 0.14,
        "doc_type": 0.04,
        "expected_raw_type": 0.18,
        "expected_modality": 0.10,
        "expected_quote_id": 0.08,
        "question_type_raw_type": 0.06,
        "domain_raw_type": 0.05,
        "quote_id": 0.04,
        "raw_type": 0.03,
        "modality": 0.02,
    }
    scores = np.zeros(len(candidates), dtype=np.float32)
    for idx, candidate in enumerate(candidates):
        score = 0.0
        for key_name, key_value in memory_keys_for_candidate(sample, candidate).items():
            precision = memory_precision(memory, key_name, key_value, subtract_memory=subtract_memory)
            if precision <= 0.0:
                continue
            support = memory_support(memory, key_name, key_value, subtract_memory=subtract_memory)
            score += key_weights.get(key_name, 0.0) * precision * support
        scores[idx] = score
    return scores


def precise_doc_memory_score_candidates(
    sample: Sample,
    candidates: Sequence[EvidenceCandidate],
    memory: GoldMemory,
    subtract_memory: Optional[GoldMemory] = None,
) -> np.ndarray:
    key_weights = {
        "doc_content": 1.6,
        "doc_page_layout_type": 1.4,
        "doc_page_layout": 1.0,
        "doc_page_type": 0.65,
        "doc_layout_type": 0.55,
    }
    scores = np.zeros(len(candidates), dtype=np.float32)
    for idx, candidate in enumerate(candidates):
        score = 0.0
        keys = memory_keys_for_candidate(sample, candidate)
        for key_name, weight in key_weights.items():
            key_value = keys.get(key_name)
            if key_value is None:
                continue
            precision = memory_precision(memory, key_name, key_value, subtract_memory=subtract_memory)
            if precision <= 0.0:
                continue
            support = memory_support(memory, key_name, key_value, subtract_memory=subtract_memory)
            score += weight * precision * support
        scores[idx] = score
    return scores


def force_memory_candidates(
    ranked_indices: List[int],
    base_scores: np.ndarray,
    precise_memory_scores: np.ndarray,
    max_forced: int,
) -> List[int]:
    if max_forced <= 0 or len(ranked_indices) <= 1:
        return ranked_indices
    memory_indices = [
        idx for idx, score in enumerate(precise_memory_scores)
        if float(score) > 0.0
    ]
    if not memory_indices:
        return ranked_indices
    memory_indices.sort(key=lambda idx: (-float(precise_memory_scores[idx]), -float(base_scores[idx]), idx))
    forced = memory_indices[:max_forced]
    forced_set = set(forced)
    return forced + [idx for idx in ranked_indices if idx not in forced_set]


def build_doc_question_index(samples: Sequence[Sample]) -> Dict[str, List[Sample]]:
    index: Dict[str, List[Sample]] = defaultdict(list)
    for sample in samples:
        index[sample.doc_name].append(sample)
    return index


def question_token_similarity(left: str, right: str) -> float:
    left_tokens = set(tokenize_for_bm25(left))
    right_tokens = set(tokenize_for_bm25(right))
    if not left_tokens or not right_tokens:
        return 0.0
    overlap = len(left_tokens & right_tokens)
    return overlap / math.sqrt(len(left_tokens) * len(right_tokens))


def similar_gold_score_candidates(
    sample: Sample,
    candidates: Sequence[EvidenceCandidate],
    doc_question_index: Dict[str, List[Sample]],
    top_n: int,
    exclude_qid: Optional[str] = None,
) -> np.ndarray:
    if top_n <= 0:
        return np.zeros(len(candidates), dtype=np.float32)

    neighbor_scores: List[Tuple[float, Sample]] = []
    for train_sample in doc_question_index.get(sample.doc_name, []):
        if exclude_qid is not None and train_sample.q_id == exclude_qid:
            continue
        similarity = question_token_similarity(sample.question, train_sample.question)
        if similarity > 0.0:
            neighbor_scores.append((similarity, train_sample))
    if not neighbor_scores:
        return np.zeros(len(candidates), dtype=np.float32)

    neighbor_scores.sort(key=lambda item: -item[0])
    selected_neighbors = neighbor_scores[:top_n]
    key_weights = {
        "doc_content": 1.25,
        "doc_page_layout_type": 1.10,
        "doc_page_layout": 0.80,
        "doc_page_type": 0.45,
        "doc_layout_type": 0.35,
    }

    gold_feature_scores: Dict[str, Counter] = defaultdict(Counter)
    for similarity, train_sample in selected_neighbors:
        gold_ids = set(train_sample.gold_quotes)
        for gold_candidate in train_sample.candidates:
            if gold_candidate.quote_id not in gold_ids:
                continue
            for key_name, key_value in memory_keys_for_candidate(train_sample, gold_candidate).items():
                if key_name in key_weights:
                    gold_feature_scores[key_name][key_value] += similarity

    scores = np.zeros(len(candidates), dtype=np.float32)
    for idx, candidate in enumerate(candidates):
        score = 0.0
        for key_name, key_value in memory_keys_for_candidate(sample, candidate).items():
            if key_name not in key_weights:
                continue
            score += key_weights[key_name] * gold_feature_scores[key_name].get(key_value, 0.0)
        scores[idx] = score
    return scores


def memory_retrieve_batch(
    samples: Sequence[Sample],
    memory_train_samples: Sequence[Sample],
    top_k: int,
    bm25_weight: float,
    memory_weight: float,
) -> Dict[str, List[str]]:
    memory = build_gold_memory(memory_train_samples)
    predictions: Dict[str, List[str]] = {}
    for sample in samples:
        bm25_scores = minmax_normalize(bm25_score_candidates(sample.question, sample.candidates))
        memory_scores = minmax_normalize(memory_score_candidates(sample, sample.candidates, memory))
        fused_scores = (bm25_weight * bm25_scores) + (memory_weight * memory_scores)
        ranked_indices = stable_rank_indices(fused_scores)
        predictions[sample.q_id] = [sample.candidates[idx].quote_id for idx in ranked_indices[:top_k]]
    return predictions


def memory_hybrid_retrieve_batch(
    samples: Sequence[Sample],
    memory_train_samples: Sequence[Sample],
    dense_model_name: str,
    device: str,
    batch_size: int,
    top_k: int,
    bm25_weight: float,
    dense_weight: float,
    memory_weight: float,
    type_weight: float,
    page_weight: float,
    layout_weight: float,
    force_memory_top: int,
    similar_memory_weight: float,
    similar_memory_topn: int,
    use_reranker: bool,
    reranker_model_name: str,
    rerank_topk: int,
    reranker_weight: float,
    cache_dir: Optional[Path] = None,
) -> Dict[str, List[str]]:
    memory = build_gold_memory(memory_train_samples)
    doc_question_index = build_doc_question_index(memory_train_samples)
    bm25_score_map = {sample.q_id: bm25_score_candidates(sample.question, sample.candidates) for sample in samples}
    dense_score_map = dense_score_batch(samples, model_name=dense_model_name, device=device, batch_size=batch_size, cache_dir=cache_dir)
    reranker = HFReranker(reranker_model_name, device=device) if use_reranker else None

    predictions: Dict[str, List[str]] = {}
    for sample in samples:
        bm25_scores = minmax_normalize(bm25_score_map[sample.q_id])
        dense_scores = minmax_normalize(dense_score_map[sample.q_id])
        memory_scores = minmax_normalize(memory_score_candidates(sample, sample.candidates, memory))
        similar_memory_scores = minmax_normalize(
            similar_gold_score_candidates(
                sample,
                sample.candidates,
                doc_question_index=doc_question_index,
                top_n=similar_memory_topn,
            )
        )
        type_scores = minmax_normalize(expected_type_score_candidates(sample, sample.candidates))
        fused_scores = (
            (bm25_weight * bm25_scores)
            + (dense_weight * dense_scores)
            + (memory_weight * memory_scores)
            + (similar_memory_weight * similar_memory_scores)
            + (type_weight * type_scores)
        )
        fused_scores = apply_context_boost(
            sample,
            fused_scores,
            page_weight=page_weight,
            layout_weight=layout_weight,
        )
        if reranker is not None:
            ranked_indices = rerank_order(
                sample=sample,
                base_scores=fused_scores,
                reranker=reranker,
                rerank_topk=rerank_topk,
                reranker_weight=reranker_weight,
                batch_size=max(1, min(batch_size, 8)),
            )
        else:
            ranked_indices = stable_rank_indices(fused_scores)
        if force_memory_top > 0:
            precise_memory_scores = precise_doc_memory_score_candidates(sample, sample.candidates, memory)
            ranked_indices = force_memory_candidates(
                ranked_indices,
                base_scores=fused_scores,
                precise_memory_scores=precise_memory_scores,
                max_forced=force_memory_top,
            )
        predictions[sample.q_id] = [sample.candidates[idx].quote_id for idx in ranked_indices[:top_k]]
    return predictions


def reciprocal_rank_fusion(
    ranked_lists: Sequence[Sequence[int]],
    weights: Sequence[float],
    rrf_k: float = 60.0,
) -> List[int]:
    scores: Dict[int, float] = defaultdict(float)
    first_seen: Dict[int, int] = {}
    for list_idx, ranked in enumerate(ranked_lists):
        weight = weights[list_idx] if list_idx < len(weights) else 1.0
        for rank, candidate_idx in enumerate(ranked):
            scores[candidate_idx] += weight / (rrf_k + rank + 1)
            first_seen.setdefault(candidate_idx, len(first_seen))
    return sorted(scores.keys(), key=lambda idx: (-scores[idx], first_seen[idx]))


def quota_union_indices(
    ranked_lists: Sequence[Sequence[int]],
    quotas: Sequence[int],
    fallback_order: Sequence[int],
    top_k: int,
) -> List[int]:
    selected: List[int] = []
    selected_set = set()
    for list_idx, ranked in enumerate(ranked_lists):
        quota = quotas[list_idx] if list_idx < len(quotas) else 0
        if quota <= 0:
            continue
        added = 0
        for candidate_idx in ranked:
            if candidate_idx in selected_set:
                continue
            selected.append(candidate_idx)
            selected_set.add(candidate_idx)
            added += 1
            if added >= quota or len(selected) >= top_k:
                break
        if len(selected) >= top_k:
            break

    if len(selected) < top_k:
        for candidate_idx in fallback_order:
            if candidate_idx in selected_set:
                continue
            selected.append(candidate_idx)
            selected_set.add(candidate_idx)
            if len(selected) >= top_k:
                break
    return selected[:top_k]


def dual_reranker_memory_hybrid_retrieve_batch(
    samples: Sequence[Sample],
    memory_train_samples: Sequence[Sample],
    dense_model_name: str,
    device: str,
    batch_size: int,
    top_k: int,
    bm25_weight: float,
    dense_weight: float,
    memory_weight: float,
    similar_memory_weight: float,
    similar_memory_topn: int,
    primary_reranker_model_name: str,
    secondary_reranker_model_name: str,
    rerank_topk: int,
    primary_weight: float,
    secondary_weight: float,
    base_weight: float,
    rrf_k: float,
    cache_dir: Optional[Path] = None,
) -> Dict[str, List[str]]:
    memory = build_gold_memory(memory_train_samples)
    doc_question_index = build_doc_question_index(memory_train_samples)
    bm25_score_map = {sample.q_id: bm25_score_candidates(sample.question, sample.candidates) for sample in samples}
    dense_score_map = dense_score_batch(samples, model_name=dense_model_name, device=device, batch_size=batch_size, cache_dir=cache_dir)
    primary_reranker = HFReranker(primary_reranker_model_name, device=device)
    secondary_reranker = HFReranker(secondary_reranker_model_name, device=device)

    predictions: Dict[str, List[str]] = {}
    for sample in samples:
        bm25_scores = minmax_normalize(bm25_score_map[sample.q_id])
        dense_scores = minmax_normalize(dense_score_map[sample.q_id])
        memory_scores = minmax_normalize(memory_score_candidates(sample, sample.candidates, memory))
        similar_memory_scores = minmax_normalize(
            similar_gold_score_candidates(
                sample,
                sample.candidates,
                doc_question_index=doc_question_index,
                top_n=similar_memory_topn,
            )
        )
        fused_scores = (
            (bm25_weight * bm25_scores)
            + (dense_weight * dense_scores)
            + (memory_weight * memory_scores)
            + (similar_memory_weight * similar_memory_scores)
        )
        base_ranked = stable_rank_indices(fused_scores)
        primary_ranked = rerank_order(
            sample=sample,
            base_scores=fused_scores,
            reranker=primary_reranker,
            rerank_topk=rerank_topk,
            reranker_weight=0.70,
            batch_size=max(1, min(batch_size, 8)),
        )
        secondary_ranked = rerank_order(
            sample=sample,
            base_scores=fused_scores,
            reranker=secondary_reranker,
            rerank_topk=rerank_topk,
            reranker_weight=0.70,
            batch_size=max(1, min(batch_size, 8)),
        )
        ranked_indices = reciprocal_rank_fusion(
            [primary_ranked, secondary_ranked, base_ranked],
            weights=[primary_weight, secondary_weight, base_weight],
            rrf_k=rrf_k,
        )
        predictions[sample.q_id] = [sample.candidates[idx].quote_id for idx in ranked_indices[:top_k]]
    return predictions


# ============================================================================
# Doc-aware supervised ranker
# ============================================================================


def rank_percentile(scores: Sequence[float]) -> np.ndarray:
    if len(scores) == 0:
        return np.zeros(0, dtype=np.float32)
    ranked = stable_rank_indices(scores)
    n = len(ranked)
    if n <= 1:
        return np.ones(n, dtype=np.float32)
    percentiles = np.zeros(n, dtype=np.float32)
    for rank, idx in enumerate(ranked):
        percentiles[idx] = 1.0 - (rank / (n - 1))
    return percentiles


def comparative_expected_mixed(sample: Sample) -> bool:
    expected = set(expected_modality_key(sample))
    has_text = "text" in expected
    has_visual = any(raw_type in expected for raw_type in {"table", "figure", "chart", "image"})
    return sample.question_type == "Comparative" and has_text and has_visual


def build_order_scores(ranked_indices: Sequence[int], num_candidates: int) -> np.ndarray:
    if num_candidates <= 0:
        return np.zeros(0, dtype=np.float32)
    if num_candidates == 1:
        scores = np.zeros(1, dtype=np.float32)
        scores[ranked_indices[0]] = 1.0
        return scores
    scores = np.zeros(num_candidates, dtype=np.float32)
    denom = max(num_candidates - 1, 1)
    for rank, idx in enumerate(ranked_indices):
        scores[idx] = 1.0 - (rank / denom)
    return scores


def needs_comparative_coverage_adjustment(
    sample: Sample,
    ranked_indices: Sequence[int],
    top_k: int,
) -> bool:
    if not comparative_expected_mixed(sample):
        return False
    initial = ranked_indices[:top_k]
    modalities = [sample.candidates[idx].modality for idx in initial]
    if "text" not in modalities or "image" not in modalities:
        return True
    page_ids = [sample.candidates[idx].page_id for idx in initial if sample.candidates[idx].page_id is not None]
    if len(page_ids) >= 3 and len(set(page_ids)) <= 2:
        return True
    return False


def comparative_coverage_select_indices(
    sample: Sample,
    base_scores: Sequence[float],
    ranked_indices: Sequence[int],
    top_k: int,
    page_penalty: float,
    layout_penalty: float,
    modality_repeat_penalty: float,
    raw_type_repeat_penalty: float,
    missing_modality_bonus: float,
) -> List[int]:
    if not needs_comparative_coverage_adjustment(sample, ranked_indices, top_k=top_k):
        return list(ranked_indices[:top_k])

    expected = set(expected_modality_key(sample))
    base_arr = np.asarray(base_scores, dtype=np.float32)
    if base_arr.size == 0:
        return []

    order_scores = build_order_scores(ranked_indices, num_candidates=len(sample.candidates))
    combined_scores = (0.70 * minmax_normalize(base_arr)) + (0.30 * order_scores)

    selected: List[int] = []
    remaining = set(range(len(sample.candidates)))
    while remaining and len(selected) < top_k:
        selected_pages = {sample.candidates[idx].page_id for idx in selected if sample.candidates[idx].page_id is not None}
        selected_layouts = {sample.candidates[idx].layout_id for idx in selected if sample.candidates[idx].layout_id is not None}
        selected_modalities = Counter(sample.candidates[idx].modality for idx in selected)
        selected_raw_types = Counter(sample.candidates[idx].raw_type.lower() for idx in selected)

        best_idx = None
        best_score = -1e9
        for idx in remaining:
            candidate = sample.candidates[idx]
            adjusted = float(combined_scores[idx])
            if candidate.page_id is not None and candidate.page_id in selected_pages:
                adjusted -= page_penalty
            if candidate.layout_id is not None and candidate.layout_id in selected_layouts:
                adjusted -= layout_penalty
            adjusted -= modality_repeat_penalty * max(0, selected_modalities[candidate.modality] - 1)
            adjusted -= raw_type_repeat_penalty * selected_raw_types[candidate.raw_type.lower()]

            if "text" in expected and selected_modalities["text"] == 0 and candidate.modality == "text":
                adjusted += missing_modality_bonus
            if any(raw_type in expected for raw_type in {"table", "figure", "chart", "image"}):
                if selected_modalities["image"] == 0 and candidate.modality == "image":
                    adjusted += missing_modality_bonus

            if adjusted > best_score:
                best_score = adjusted
                best_idx = idx

        if best_idx is None:
            break
        selected.append(best_idx)
        remaining.remove(best_idx)

    return selected[:top_k]


def best_group_score_candidates(
    candidates: Sequence[EvidenceCandidate],
    scores: Sequence[float],
    group_attr: str,
) -> np.ndarray:
    group_best: Dict[Any, float] = {}
    for idx, candidate in enumerate(candidates):
        group_key = getattr(candidate, group_attr)
        if group_key is None:
            continue
        group_best[group_key] = max(group_best.get(group_key, -1e9), float(scores[idx]))
    return np.asarray(
        [group_best.get(getattr(candidate, group_attr), 0.0) for candidate in candidates],
        dtype=np.float32,
    )


def select_doc_question_neighbors(
    sample: Sample,
    doc_question_index: Dict[str, List[Sample]],
    top_n: int,
    exclude_qid: Optional[str] = None,
) -> List[Tuple[float, Sample]]:
    if top_n <= 0:
        return []

    doc_samples = [
        train_sample
        for train_sample in doc_question_index.get(sample.doc_name, [])
        if exclude_qid is None or train_sample.q_id != exclude_qid
    ]
    if not doc_samples:
        return []

    question_scores = bm25_score_texts(sample.question, [train_sample.question for train_sample in doc_samples])
    overlap_scores = np.asarray(
        [question_token_similarity(sample.question, train_sample.question) for train_sample in doc_samples],
        dtype=np.float32,
    )
    fused_scores = (0.70 * minmax_normalize(question_scores)) + (0.30 * minmax_normalize(overlap_scores))

    neighbors = [
        (float(fused_scores[idx]), doc_samples[idx])
        for idx in stable_rank_indices(fused_scores)
        if float(fused_scores[idx]) > 0.0
    ]
    return neighbors[:top_n]


def overlap_confidence_bucket(score: float) -> str:
    if score <= 0.0:
        return "none"
    if score < 0.20:
        return "low"
    if score < 0.40:
        return "mid"
    return "high"


def best_neighbor_score_by_qid(
    samples: Sequence[Sample],
    reference_samples: Sequence[Sample],
    top_n: int,
    exclude_self_from_reference: bool = False,
) -> Dict[str, float]:
    doc_question_index = build_doc_question_index(reference_samples)
    scores: Dict[str, float] = {}
    for sample in samples:
        neighbors = select_doc_question_neighbors(
            sample,
            doc_question_index=doc_question_index,
            top_n=top_n,
            exclude_qid=sample.q_id if exclude_self_from_reference else None,
        )
        scores[sample.q_id] = float(neighbors[0][0]) if neighbors else 0.0
    return scores


def candidate_neighbor_vote_scores(
    sample: Sample,
    candidates: Sequence[EvidenceCandidate],
    neighbors: Sequence[Tuple[float, Sample]],
) -> Dict[str, np.ndarray]:
    exact_scores = np.zeros(len(candidates), dtype=np.float32)
    page_scores = np.zeros(len(candidates), dtype=np.float32)
    layout_scores = np.zeros(len(candidates), dtype=np.float32)
    page_type_scores = np.zeros(len(candidates), dtype=np.float32)
    raw_type_scores = np.zeros(len(candidates), dtype=np.float32)

    candidate_contents = [content_key(candidate.content) for candidate in candidates]
    candidate_raw_types = [candidate.raw_type.lower() for candidate in candidates]

    for similarity, neighbor_sample in neighbors:
        if similarity <= 0.0:
            continue
        gold_ids = set(neighbor_sample.gold_quotes)
        for gold_candidate in neighbor_sample.candidates:
            if gold_candidate.quote_id not in gold_ids:
                continue
            gold_content = content_key(gold_candidate.content)
            gold_raw_type = gold_candidate.raw_type.lower()
            for idx, candidate in enumerate(candidates):
                if gold_content and candidate_contents[idx] == gold_content:
                    exact_scores[idx] += similarity
                if candidate.page_id is not None and candidate.page_id == gold_candidate.page_id:
                    page_scores[idx] += similarity
                    if candidate_raw_types[idx] == gold_raw_type:
                        page_type_scores[idx] += similarity
                if candidate.layout_id is not None and candidate.layout_id == gold_candidate.layout_id:
                    layout_scores[idx] += similarity
                if candidate_raw_types[idx] == gold_raw_type:
                    raw_type_scores[idx] += 0.20 * similarity

    return {
        "exact": exact_scores,
        "page": page_scores,
        "layout": layout_scores,
        "page_type": page_type_scores,
        "raw_type": raw_type_scores,
    }


def domain_balance_multipliers(
    samples: Sequence[Sample],
    strategy: str,
) -> Dict[str, float]:
    normalized = safe_strip(strategy).lower()
    if normalized in {"", "none"}:
        return {}

    domain_counts = Counter(sample.domain for sample in samples)
    if not domain_counts:
        return {}

    num_domains = len(domain_counts)
    total = sum(domain_counts.values())
    multipliers: Dict[str, float] = {}
    for domain, count in domain_counts.items():
        if count <= 0:
            multipliers[domain] = 1.0
            continue
        uniform_ratio = total / (num_domains * count)
        if normalized == "uniform":
            multipliers[domain] = float(uniform_ratio)
        elif normalized == "sqrt":
            multipliers[domain] = float(math.sqrt(uniform_ratio))
        else:
            raise ValueError(f"Unsupported ranker domain balance strategy: {strategy}")
    return multipliers


def extract_doc_aware_ranker_features(
    samples: Sequence[Sample],
    reference_samples: Sequence[Sample],
    dense_model_name: str,
    device: str,
    batch_size: int,
    bm25_weight: float,
    dense_weight: float,
    memory_weight: float,
    similar_memory_weight: float,
    type_weight: float,
    page_weight: float,
    layout_weight: float,
    similar_memory_topn: int,
    ranker_domain_balance: str = "none",
    exclude_self_from_reference: bool = False,
    cache_dir: Optional[Path] = None,
) -> Tuple[np.ndarray, np.ndarray, List[Dict[str, Any]], np.ndarray]:
    memory = build_gold_memory(reference_samples)
    doc_question_index = build_doc_question_index(reference_samples)
    doc_sample_counts = Counter(reference_sample.doc_name for reference_sample in reference_samples)
    domain_weight_map = domain_balance_multipliers(reference_samples, ranker_domain_balance)

    bm25_score_map = {sample.q_id: bm25_score_candidates(sample.question, sample.candidates) for sample in samples}
    dense_score_map = dense_score_batch(
        samples,
        model_name=dense_model_name,
        device=device,
        batch_size=batch_size,
        cache_dir=cache_dir,
    )

    feature_rows: List[List[float]] = []
    labels: List[int] = []
    spans: List[Dict[str, Any]] = []
    row_domain_weights: List[float] = []

    for sample in samples:
        bm25_raw = np.asarray(bm25_score_map[sample.q_id], dtype=np.float32)
        dense_raw = np.asarray(dense_score_map[sample.q_id], dtype=np.float32)
        bm25_norm = minmax_normalize(bm25_raw)
        dense_norm = minmax_normalize(dense_raw)
        bm25_rank = rank_percentile(bm25_raw.tolist())
        dense_rank = rank_percentile(dense_raw.tolist())

        subtract_memory = build_gold_memory([sample]) if exclude_self_from_reference else None
        memory_raw = memory_score_candidates(
            sample,
            sample.candidates,
            memory,
            subtract_memory=subtract_memory,
        )
        precise_memory_raw = precise_doc_memory_score_candidates(
            sample,
            sample.candidates,
            memory,
            subtract_memory=subtract_memory,
        )
        similar_memory_raw = similar_gold_score_candidates(
            sample,
            sample.candidates,
            doc_question_index=doc_question_index,
            top_n=similar_memory_topn,
            exclude_qid=sample.q_id if exclude_self_from_reference else None,
        )
        type_scores = expected_type_score_candidates(sample, sample.candidates)

        memory_norm = minmax_normalize(memory_raw)
        precise_memory_norm = minmax_normalize(precise_memory_raw)
        similar_memory_norm = minmax_normalize(similar_memory_raw)

        base_scores = (
            (bm25_weight * bm25_norm)
            + (dense_weight * dense_norm)
            + (memory_weight * memory_norm)
            + (similar_memory_weight * similar_memory_norm)
            + (type_weight * type_scores)
        )
        base_scores = apply_context_boost(
            sample,
            base_scores,
            page_weight=page_weight,
            layout_weight=layout_weight,
        )
        base_norm = minmax_normalize(base_scores)

        page_context_scores = minmax_normalize(
            best_group_score_candidates(sample.candidates, 0.50 * bm25_norm + 0.50 * dense_norm, "page_id")
        )
        layout_context_scores = minmax_normalize(
            best_group_score_candidates(sample.candidates, 0.50 * bm25_norm + 0.50 * dense_norm, "layout_id")
        )

        neighbors = select_doc_question_neighbors(
            sample,
            doc_question_index=doc_question_index,
            top_n=similar_memory_topn,
            exclude_qid=sample.q_id if exclude_self_from_reference else None,
        )
        neighbor_votes = candidate_neighbor_vote_scores(sample, sample.candidates, neighbors)
        neighbor_exact_norm = minmax_normalize(neighbor_votes["exact"])
        neighbor_page_norm = minmax_normalize(neighbor_votes["page"])
        neighbor_layout_norm = minmax_normalize(neighbor_votes["layout"])
        neighbor_page_type_norm = minmax_normalize(neighbor_votes["page_type"])
        neighbor_raw_type_norm = minmax_normalize(neighbor_votes["raw_type"])

        neighbor_scores = [score for score, _ in neighbors]
        best_neighbor_score = float(neighbor_scores[0]) if neighbor_scores else 0.0
        avg_neighbor_score = float(np.mean(neighbor_scores[:3])) if neighbor_scores else 0.0
        normalized_neighbor_count = float(len(neighbors) / max(similar_memory_topn, 1))
        has_doc_overlap = 1.0 if neighbors else 0.0

        semantic_scores = (
            (bm25_weight * bm25_norm)
            + (dense_weight * dense_norm)
            + (type_weight * type_scores)
        )
        overlap_scores = (
            (memory_weight * memory_norm)
            + (0.35 * precise_memory_norm)
            + (similar_memory_weight * similar_memory_norm)
            + (0.60 * neighbor_exact_norm)
            + (0.25 * neighbor_page_norm)
            + (0.20 * neighbor_page_type_norm)
            + (0.10 * neighbor_layout_norm)
            + (0.10 * page_context_scores)
        )
        overlap_confidence = max(
            best_neighbor_score,
            float(np.max(precise_memory_norm)) if precise_memory_norm.size else 0.0,
            float(np.max(neighbor_exact_norm)) if neighbor_exact_norm.size else 0.0,
            float(np.max(neighbor_page_type_norm)) if neighbor_page_type_norm.size else 0.0,
        )

        candidate_keys = [memory_keys_for_candidate(sample, candidate) for candidate in sample.candidates]
        gold_ids = set(sample.gold_quotes)
        start = len(feature_rows)
        for idx, candidate in enumerate(sample.candidates):
            keys = candidate_keys[idx]
            feature_rows.append(
                [
                    float(base_norm[idx]),
                    float(bm25_norm[idx]),
                    float(dense_norm[idx]),
                    float(memory_norm[idx]),
                    float(precise_memory_norm[idx]),
                    float(similar_memory_norm[idx]),
                    float(type_scores[idx]),
                    float(page_context_scores[idx]),
                    float(layout_context_scores[idx]),
                    float(neighbor_exact_norm[idx]),
                    float(neighbor_page_norm[idx]),
                    float(neighbor_layout_norm[idx]),
                    float(neighbor_page_type_norm[idx]),
                    float(neighbor_raw_type_norm[idx]),
                    float(memory_precision(memory, "doc_content", keys.get("doc_content"), subtract_memory)),
                    float(memory_precision(memory, "doc_page_layout_type", keys.get("doc_page_layout_type"), subtract_memory)),
                    float(memory_precision(memory, "doc_page_type", keys.get("doc_page_type"), subtract_memory)),
                    float(memory_precision(memory, "doc_layout_type", keys.get("doc_layout_type"), subtract_memory)),
                    float(memory_precision(memory, "global_content", keys.get("global_content"), subtract_memory)),
                    best_neighbor_score,
                    avg_neighbor_score,
                    normalized_neighbor_count,
                    has_doc_overlap,
                    float(candidate.modality == "text"),
                    float(candidate.modality == "image"),
                    float(candidate.raw_type.lower() == "table"),
                    float(candidate.raw_type.lower() == "figure"),
                    float(candidate.raw_type.lower() == "chart"),
                    float(candidate.raw_type.lower() == "image"),
                    float(candidate.page_id is not None),
                    float(candidate.layout_id is not None),
                    float(len(tokenize_for_bm25(candidate.content))),
                    float(bm25_rank[idx]),
                    float(dense_rank[idx]),
                ]
            )
            labels.append(1 if candidate.quote_id in gold_ids else 0)
            row_domain_weights.append(float(domain_weight_map.get(sample.domain, 1.0)))

        spans.append(
            {
                "sample": sample,
                "start": start,
                "end": len(feature_rows),
                "base_scores": base_scores.astype(np.float32),
                "semantic_scores": semantic_scores.astype(np.float32),
                "overlap_scores": overlap_scores.astype(np.float32),
                "doc_train_count": int(doc_sample_counts.get(sample.doc_name, 0)),
                "overlap_confidence": float(overlap_confidence),
            }
        )

    return (
        np.asarray(feature_rows, dtype=np.float32),
        np.asarray(labels, dtype=np.int32),
        spans,
        np.asarray(row_domain_weights, dtype=np.float32),
    )


def train_doc_aware_ranker(
    feature_matrix: np.ndarray,
    labels: np.ndarray,
    seed: int,
    max_iter: int,
    learning_rate: float,
    max_depth: int,
    min_samples_leaf: int,
    positive_weight: float,
    row_domain_weights: Optional[np.ndarray] = None,
):
    try:
        from sklearn.ensemble import HistGradientBoostingClassifier  # type: ignore
    except ImportError as exc:
        raise RuntimeError(
            "doc_aware_ranker requires scikit-learn in the active environment."
        ) from exc

    if feature_matrix.size == 0 or labels.size == 0:
        raise RuntimeError("No training rows available for doc_aware_ranker.")

    model = HistGradientBoostingClassifier(
        learning_rate=learning_rate,
        max_depth=max_depth,
        max_iter=max_iter,
        min_samples_leaf=min_samples_leaf,
        random_state=seed,
    )
    sample_weight = np.ones(labels.shape[0], dtype=np.float32)
    if row_domain_weights is not None and row_domain_weights.shape[0] == labels.shape[0]:
        sample_weight *= row_domain_weights
    sample_weight[labels > 0] = positive_weight
    if row_domain_weights is not None and row_domain_weights.shape[0] == labels.shape[0]:
        sample_weight[labels > 0] *= row_domain_weights[labels > 0]
    model.fit(feature_matrix, labels, sample_weight=sample_weight)
    return model


def doc_aware_ranker_retrieve_batch(
    samples: Sequence[Sample],
    memory_train_samples: Sequence[Sample],
    dense_model_name: str,
    device: str,
    batch_size: int,
    top_k: int,
    bm25_weight: float,
    dense_weight: float,
    memory_weight: float,
    similar_memory_weight: float,
    type_weight: float,
    page_weight: float,
    layout_weight: float,
    similar_memory_topn: int,
    ranker_max_iter: int,
    ranker_learning_rate: float,
    ranker_max_depth: int,
    ranker_min_samples_leaf: int,
    ranker_positive_weight: float,
    ranker_model_weight: float,
    ranker_base_weight: float,
    ranker_domain_balance: str,
    seed: int,
    use_reranker: bool,
    reranker_model_name: str,
    rerank_topk: int,
    reranker_weight: float,
    use_comparative_coverage: bool,
    coverage_page_penalty: float,
    coverage_layout_penalty: float,
    coverage_modality_penalty: float,
    coverage_raw_type_penalty: float,
    coverage_missing_bonus: float,
    cache_dir: Optional[Path] = None,
) -> Dict[str, List[str]]:
    train_features, train_labels, _, train_domain_weights = extract_doc_aware_ranker_features(
        memory_train_samples,
        reference_samples=memory_train_samples,
        dense_model_name=dense_model_name,
        device=device,
        batch_size=batch_size,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
        memory_weight=memory_weight,
        similar_memory_weight=similar_memory_weight,
        type_weight=type_weight,
        page_weight=page_weight,
        layout_weight=layout_weight,
        similar_memory_topn=similar_memory_topn,
        ranker_domain_balance=ranker_domain_balance,
        exclude_self_from_reference=True,
        cache_dir=cache_dir,
    )
    model = train_doc_aware_ranker(
        train_features,
        train_labels,
        seed=seed,
        max_iter=ranker_max_iter,
        learning_rate=ranker_learning_rate,
        max_depth=ranker_max_depth,
        min_samples_leaf=ranker_min_samples_leaf,
        positive_weight=ranker_positive_weight,
        row_domain_weights=train_domain_weights,
    )

    feature_matrix, _, spans, _ = extract_doc_aware_ranker_features(
        samples,
        reference_samples=memory_train_samples,
        dense_model_name=dense_model_name,
        device=device,
        batch_size=batch_size,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
        memory_weight=memory_weight,
        similar_memory_weight=similar_memory_weight,
        type_weight=type_weight,
        page_weight=page_weight,
        layout_weight=layout_weight,
        similar_memory_topn=similar_memory_topn,
        ranker_domain_balance=ranker_domain_balance,
        exclude_self_from_reference=False,
        cache_dir=cache_dir,
    )

    if hasattr(model, "predict_proba"):
        model_scores_all = model.predict_proba(feature_matrix)[:, 1].astype(np.float32)
    else:  # pragma: no cover - sklearn should expose predict_proba here
        model_scores_all = model.predict(feature_matrix).astype(np.float32)

    reranker = HFReranker(reranker_model_name, device=device) if use_reranker else None
    predictions: Dict[str, List[str]] = {}
    for span in spans:
        sample = span["sample"]
        start = span["start"]
        end = span["end"]
        model_scores = minmax_normalize(model_scores_all[start:end])
        base_scores = minmax_normalize(span["base_scores"])
        fused_scores = (ranker_model_weight * model_scores) + (ranker_base_weight * base_scores)
        if reranker is not None:
            ranked_indices = rerank_order(
                sample=sample,
                base_scores=fused_scores,
                reranker=reranker,
                rerank_topk=rerank_topk,
                reranker_weight=reranker_weight,
                batch_size=max(1, min(batch_size, 8)),
            )
        else:
            ranked_indices = stable_rank_indices(fused_scores)
        if use_comparative_coverage:
            ranked_indices = comparative_coverage_select_indices(
                sample=sample,
                base_scores=fused_scores,
                ranked_indices=ranked_indices,
                top_k=top_k,
                page_penalty=coverage_page_penalty,
                layout_penalty=coverage_layout_penalty,
                modality_repeat_penalty=coverage_modality_penalty,
                raw_type_repeat_penalty=coverage_raw_type_penalty,
                missing_modality_bonus=coverage_missing_bonus,
            )
        predictions[sample.q_id] = [sample.candidates[idx].quote_id for idx in ranked_indices[:top_k]]
    return predictions


def routed_doc_aware_ranker_retrieve_batch(
    samples: Sequence[Sample],
    memory_train_samples: Sequence[Sample],
    dense_model_name: str,
    device: str,
    batch_size: int,
    top_k: int,
    bm25_weight: float,
    dense_weight: float,
    memory_weight: float,
    similar_memory_weight: float,
    type_weight: float,
    page_weight: float,
    layout_weight: float,
    similar_memory_topn: int,
    ranker_max_iter: int,
    ranker_learning_rate: float,
    ranker_max_depth: int,
    ranker_min_samples_leaf: int,
    ranker_positive_weight: float,
    ranker_model_weight: float,
    ranker_base_weight: float,
    route_low_doc_count: int,
    route_high_doc_count: int,
    route_low_confidence: float,
    route_high_confidence: float,
    route_low_final_quota: int,
    route_low_overlap_quota: int,
    route_low_semantic_quota: int,
    route_high_final_quota: int,
    route_high_overlap_quota: int,
    route_high_semantic_quota: int,
    ranker_domain_balance: str,
    seed: int,
    use_comparative_coverage: bool,
    coverage_page_penalty: float,
    coverage_layout_penalty: float,
    coverage_modality_penalty: float,
    coverage_raw_type_penalty: float,
    coverage_missing_bonus: float,
    cache_dir: Optional[Path] = None,
) -> Dict[str, List[str]]:
    train_features, train_labels, _, train_domain_weights = extract_doc_aware_ranker_features(
        memory_train_samples,
        reference_samples=memory_train_samples,
        dense_model_name=dense_model_name,
        device=device,
        batch_size=batch_size,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
        memory_weight=memory_weight,
        similar_memory_weight=similar_memory_weight,
        type_weight=type_weight,
        page_weight=page_weight,
        layout_weight=layout_weight,
        similar_memory_topn=similar_memory_topn,
        ranker_domain_balance=ranker_domain_balance,
        exclude_self_from_reference=True,
        cache_dir=cache_dir,
    )
    model = train_doc_aware_ranker(
        train_features,
        train_labels,
        seed=seed,
        max_iter=ranker_max_iter,
        learning_rate=ranker_learning_rate,
        max_depth=ranker_max_depth,
        min_samples_leaf=ranker_min_samples_leaf,
        positive_weight=ranker_positive_weight,
        row_domain_weights=train_domain_weights,
    )

    feature_matrix, _, spans, _ = extract_doc_aware_ranker_features(
        samples,
        reference_samples=memory_train_samples,
        dense_model_name=dense_model_name,
        device=device,
        batch_size=batch_size,
        bm25_weight=bm25_weight,
        dense_weight=dense_weight,
        memory_weight=memory_weight,
        similar_memory_weight=similar_memory_weight,
        type_weight=type_weight,
        page_weight=page_weight,
        layout_weight=layout_weight,
        similar_memory_topn=similar_memory_topn,
        ranker_domain_balance=ranker_domain_balance,
        exclude_self_from_reference=False,
        cache_dir=cache_dir,
    )

    if hasattr(model, "predict_proba"):
        model_scores_all = model.predict_proba(feature_matrix)[:, 1].astype(np.float32)
    else:  # pragma: no cover - sklearn should expose predict_proba here
        model_scores_all = model.predict(feature_matrix).astype(np.float32)

    predictions: Dict[str, List[str]] = {}
    for span in spans:
        sample = span["sample"]
        start = span["start"]
        end = span["end"]
        model_scores = minmax_normalize(model_scores_all[start:end])
        base_scores = minmax_normalize(span["base_scores"])
        final_scores = (ranker_model_weight * model_scores) + (ranker_base_weight * base_scores)
        final_order = stable_rank_indices(final_scores)
        semantic_order = stable_rank_indices(span["semantic_scores"])
        overlap_order = stable_rank_indices(span["overlap_scores"])

        doc_train_count = int(span["doc_train_count"])
        overlap_confidence = float(span["overlap_confidence"])
        if doc_train_count >= route_high_doc_count and overlap_confidence >= route_high_confidence:
            ranked_indices = quota_union_indices(
                [final_order, overlap_order, semantic_order],
                [route_high_final_quota, route_high_overlap_quota, route_high_semantic_quota],
                fallback_order=final_order,
                top_k=top_k,
            )
        elif doc_train_count >= route_low_doc_count and overlap_confidence >= route_low_confidence:
            ranked_indices = quota_union_indices(
                [final_order, overlap_order, semantic_order],
                [route_low_final_quota, route_low_overlap_quota, route_low_semantic_quota],
                fallback_order=final_order,
                top_k=top_k,
            )
        else:
            ranked_indices = final_order[:top_k]
        if use_comparative_coverage:
            ranked_indices = comparative_coverage_select_indices(
                sample=sample,
                base_scores=final_scores,
                ranked_indices=ranked_indices,
                top_k=top_k,
                page_penalty=coverage_page_penalty,
                layout_penalty=coverage_layout_penalty,
                modality_repeat_penalty=coverage_modality_penalty,
                raw_type_repeat_penalty=coverage_raw_type_penalty,
                missing_modality_bonus=coverage_missing_bonus,
            )
        predictions[sample.q_id] = [sample.candidates[idx].quote_id for idx in ranked_indices[:top_k]]
    return predictions


# ============================================================================
# LLM selection baseline
# ============================================================================


def truncate_candidates_for_llm(text: str, max_words: int) -> str:
    words = normalize_text(text).split()
    if len(words) <= max_words:
        return " ".join(words)
    return " ".join(words[:max_words])


def build_llm_prompt(sample: Sample, max_words_per_candidate: int) -> Tuple[str, str]:
    system_prompt = (
        "You are a retrieval assistant. Select the 5 most relevant evidence quote_ids "
        "for the question. Output only a JSON array of quote_ids ordered from most "
        "relevant to least relevant."
    )
    lines = [
        f"Question: {sample.question}",
        "Candidates:",
    ]
    for candidate in sample.candidates:
        truncated = truncate_candidates_for_llm(candidate_to_text(candidate), max_words_per_candidate)
        lines.append(f"- {candidate.quote_id}: {truncated}")
    lines.append("Return exactly one JSON array like [\"text3\", \"image2\", ...].")
    return system_prompt, "\n".join(lines)


def parse_llm_quote_ids(output_text: str, allowed_ids: Sequence[str]) -> List[str]:
    allowed_set = set(allowed_ids)
    parsed: List[str] = []
    try:
        candidate = json.loads(output_text)
        if isinstance(candidate, list):
            for item in candidate:
                item = safe_strip(item)
                if item in allowed_set and item not in parsed:
                    parsed.append(item)
    except json.JSONDecodeError:
        pass

    if parsed:
        return parsed

    pattern = re.compile(r"\b(?:text|image)\d+\b")
    for match in pattern.findall(output_text):
        if match in allowed_set and match not in parsed:
            parsed.append(match)
    return parsed


def llm_select_one(
    sample: Sample,
    selector: HFLLMSelector,
    top_k: int,
    max_words_per_candidate: int,
) -> List[str]:
    system_prompt, user_prompt = build_llm_prompt(sample, max_words_per_candidate=max_words_per_candidate)
    output = selector.generate(system_prompt, user_prompt)
    allowed_ids = [candidate.quote_id for candidate in sample.candidates]
    parsed = parse_llm_quote_ids(output, allowed_ids)
    if len(parsed) < top_k:
        fallback = bm25_retrieve_one(sample, top_k=len(sample.candidates))
        for quote_id in fallback:
            if quote_id not in parsed:
                parsed.append(quote_id)
            if len(parsed) >= top_k:
                break
    return parsed[:top_k]


def llm_retrieve_batch(
    samples: Sequence[Sample],
    model_name: str,
    device: str,
    top_k: int,
    max_words_per_candidate: int,
) -> Dict[str, List[str]]:
    selector = HFLLMSelector(model_name=model_name, device=device)
    return {
        sample.q_id: llm_select_one(
            sample=sample,
            selector=selector,
            top_k=top_k,
            max_words_per_candidate=max_words_per_candidate,
        )
        for sample in samples
    }


# ============================================================================
# Raw image retrieval baseline
# ============================================================================


def image_score_batch(
    samples: Sequence[Sample],
    model_name: str,
    device: str,
    batch_size: int,
    cache_dir: Optional[Path] = None,
) -> Dict[str, np.ndarray]:
    retriever = HFClipRetriever(model_name=model_name, device=device)
    unique_image_paths = sorted(
        {
            candidate.img_path
            for sample in samples
            for candidate in sample.candidates
            if candidate.modality == "image" and candidate.img_path
        }
    )

    image_feature_cache_path = None
    image_features_by_path: Dict[str, np.ndarray] = {}
    if cache_dir is not None:
        ensure_dir(cache_dir)
        cache_key = build_cache_key("clip-images", model_name, unique_image_paths)
        image_feature_cache_path = cache_dir / f"clip_image_{cache_key}.pkl"
        cached = maybe_load_cached_embeddings(image_feature_cache_path)
        if cached is not None:
            image_features_by_path = {path: np.asarray(vector, dtype=np.float32) for path, vector in cached.items()}

    if not image_features_by_path and unique_image_paths:
        image_vectors = retriever.encode_images(unique_image_paths, batch_size=max(1, min(batch_size, 8)))
        image_features_by_path = {path: image_vectors[idx] for idx, path in enumerate(unique_image_paths)}
        if image_feature_cache_path is not None:
            save_pickle(image_feature_cache_path, {path: vector.tolist() for path, vector in image_features_by_path.items()})

    query_vectors = retriever.encode_texts([sample.question for sample in samples], batch_size=batch_size)
    score_map: Dict[str, np.ndarray] = {}

    for sample_idx, sample in enumerate(samples):
        text_candidates = [candidate for candidate in sample.candidates if candidate.modality == "text"]
        image_candidates = [candidate for candidate in sample.candidates if candidate.modality == "image"]

        text_scores = bm25_score_candidates(sample.question, text_candidates) if text_candidates else np.zeros(0, dtype=np.float32)
        text_scores = minmax_normalize(text_scores) if text_candidates else text_scores

        image_scores: List[float] = []
        for candidate in image_candidates:
            if candidate.img_path and candidate.img_path in image_features_by_path:
                image_scores.append(float(np.dot(query_vectors[sample_idx], image_features_by_path[candidate.img_path])))
            else:
                image_scores.append(0.0)
        image_scores_arr = minmax_normalize(image_scores) if image_scores else np.zeros(0, dtype=np.float32)

        merged_scores = np.zeros(len(sample.candidates), dtype=np.float32)
        text_ptr = 0
        image_ptr = 0
        for idx, candidate in enumerate(sample.candidates):
            if candidate.modality == "text":
                merged_scores[idx] = text_scores[text_ptr]
                text_ptr += 1
            else:
                merged_scores[idx] = image_scores_arr[image_ptr]
                image_ptr += 1
        score_map[sample.q_id] = merged_scores
    return score_map


def image_retrieve_batch(
    samples: Sequence[Sample],
    model_name: str,
    device: str,
    batch_size: int,
    top_k: int,
    cache_dir: Optional[Path] = None,
) -> Dict[str, List[str]]:
    score_map = image_score_batch(samples, model_name=model_name, device=device, batch_size=batch_size, cache_dir=cache_dir)
    predictions: Dict[str, List[str]] = {}
    for sample in samples:
        ranked_indices = stable_rank_indices(score_map[sample.q_id])
        predictions[sample.q_id] = [sample.candidates[idx].quote_id for idx in ranked_indices[:top_k]]
    return predictions


# ============================================================================
# Reporting and orchestration
# ============================================================================


def summarize_stats(train_samples: Sequence[Sample], test_samples: Sequence[Sample]) -> Dict[str, Any]:
    def summarize_split(samples: Sequence[Sample], with_gold: bool) -> Dict[str, Any]:
        candidate_counts = [len(sample.candidates) for sample in samples]
        text_counts = [sum(1 for candidate in sample.candidates if candidate.modality == "text") for sample in samples]
        image_counts = [sum(1 for candidate in sample.candidates if candidate.modality == "image") for sample in samples]
        payload: Dict[str, Any] = {
            "num_samples": len(samples),
            "avg_candidates": round(float(np.mean(candidate_counts)), 3),
            "avg_text_candidates": round(float(np.mean(text_counts)), 3),
            "avg_image_candidates": round(float(np.mean(image_counts)), 3),
            "question_type_top": Counter(sample.question_type for sample in samples).most_common(10),
            "evidence_modality_top": Counter(
                modality
                for sample in samples
                for modality in sample.evidence_modality_type
            ).most_common(10),
        }
        if with_gold:
            payload["gold_len_dist"] = Counter(len(sample.gold_quotes) for sample in samples).most_common()
        return payload

    return {
        "train": summarize_split(train_samples, with_gold=True),
        "test": summarize_split(test_samples, with_gold=False),
    }


def format_eval_result(result: EvalResult) -> str:
    payload = {
        "method": result.method,
        "recall_at_5": round(result.recall_at_5, 6),
        "evaluated_samples": result.evaluated_samples,
        "skipped_samples": result.skipped_samples,
    }
    payload.update(result.extra)
    return json.dumps(payload, indent=2, ensure_ascii=False)


def analyze_modality_preference(
    samples: Sequence[Sample],
    predictions: Dict[str, List[str]],
    method_name: str,
) -> Dict[str, Any]:
    candidate_lookup = build_candidate_lookup(samples)
    overall_counter: Counter[str] = Counter()
    raw_type_counter: Counter[str] = Counter()
    avg_text = []
    avg_image = []
    by_expected_modality: Dict[str, List[Dict[str, int]]] = defaultdict(list)

    for sample in samples:
        text_hits = 0
        image_hits = 0
        quote_lookup = candidate_lookup[sample.q_id]
        for quote_id in predictions.get(sample.q_id, [])[:5]:
            candidate = quote_lookup.get(quote_id)
            if candidate is None:
                continue
            overall_counter[candidate.modality] += 1
            raw_type_counter[candidate.raw_type] += 1
            if candidate.modality == "text":
                text_hits += 1
            else:
                image_hits += 1
        avg_text.append(text_hits)
        avg_image.append(image_hits)
        expected_key = "|".join(sorted(sample.evidence_modality_type))
        by_expected_modality[expected_key].append({"text": text_hits, "image": image_hits})

    grouped_summary = {}
    for key, rows in by_expected_modality.items():
        grouped_summary[key] = {
            "avg_text_in_top5": round(float(np.mean([row["text"] for row in rows])), 4),
            "avg_image_in_top5": round(float(np.mean([row["image"] for row in rows])), 4),
            "num_samples": len(rows),
        }

    total_predictions = max(sum(overall_counter.values()), 1)
    return {
        "method": method_name,
        "overall_counts": dict(overall_counter),
        "overall_ratios": {
            key: round(value / total_predictions, 4) for key, value in overall_counter.items()
        },
        "raw_type_counts": dict(raw_type_counter),
        "avg_text_in_top5": round(float(np.mean(avg_text)), 4) if avg_text else 0.0,
        "avg_image_in_top5": round(float(np.mean(avg_image)), 4) if avg_image else 0.0,
        "by_expected_modality_type": grouped_summary,
    }


def retrieve_predictions(
    samples: Sequence[Sample],
    args: argparse.Namespace,
    method: str,
    memory_train_samples: Optional[Sequence[Sample]] = None,
) -> Dict[str, List[str]]:
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    if method == "bm25":
        return bm25_retrieve_batch(samples, top_k=args.top_k)
    if method == "dense":
        return dense_retrieve_batch(
            samples,
            model_name=args.dense_model,
            device=args.device,
            batch_size=args.batch_size,
            top_k=args.top_k,
            cache_dir=cache_dir,
        )
    if method in {"hybrid", "typed_hybrid"}:
        return hybrid_retrieve_batch(
            samples,
            dense_model_name=args.dense_model,
            device=args.device,
            batch_size=args.batch_size,
            top_k=args.top_k,
            bm25_weight=args.bm25_weight,
            dense_weight=args.dense_weight,
            type_weight=args.type_weight if method == "typed_hybrid" else 0.0,
            page_weight=args.page_weight,
            layout_weight=args.layout_weight,
            use_reranker=args.use_reranker,
            reranker_model_name=args.reranker_model,
            rerank_topk=args.rerank_topk,
            reranker_weight=args.reranker_weight,
            cache_dir=cache_dir,
        )
    if method == "memory":
        if memory_train_samples is None:
            raise ValueError("memory method requires memory_train_samples")
        return memory_retrieve_batch(
            samples,
            memory_train_samples=memory_train_samples,
            top_k=args.top_k,
            bm25_weight=args.bm25_weight,
            memory_weight=args.memory_weight,
        )
    if method in {"memory_hybrid", "typed_memory_hybrid"}:
        if memory_train_samples is None:
            raise ValueError("memory_hybrid method requires memory_train_samples")
        return memory_hybrid_retrieve_batch(
            samples,
            memory_train_samples=memory_train_samples,
            dense_model_name=args.dense_model,
            device=args.device,
            batch_size=args.batch_size,
            top_k=args.top_k,
            bm25_weight=args.bm25_weight,
            dense_weight=args.dense_weight,
            memory_weight=args.memory_weight,
            type_weight=args.type_weight if method == "typed_memory_hybrid" else 0.0,
            page_weight=args.page_weight,
            layout_weight=args.layout_weight,
            force_memory_top=args.force_memory_top,
            similar_memory_weight=args.similar_memory_weight,
            similar_memory_topn=args.similar_memory_topn,
            use_reranker=args.use_reranker,
            reranker_model_name=args.reranker_model,
            rerank_topk=args.rerank_topk,
            reranker_weight=args.reranker_weight,
            cache_dir=cache_dir,
        )
    if method == "dual_reranker":
        if memory_train_samples is None:
            raise ValueError("dual_reranker method requires memory_train_samples")
        return dual_reranker_memory_hybrid_retrieve_batch(
            samples,
            memory_train_samples=memory_train_samples,
            dense_model_name=args.dense_model,
            device=args.device,
            batch_size=args.batch_size,
            top_k=args.top_k,
            bm25_weight=args.bm25_weight,
            dense_weight=args.dense_weight,
            memory_weight=args.memory_weight,
            similar_memory_weight=args.similar_memory_weight,
            similar_memory_topn=args.similar_memory_topn,
            primary_reranker_model_name=args.reranker_model,
            secondary_reranker_model_name=args.second_reranker_model,
            rerank_topk=args.rerank_topk,
            primary_weight=args.dual_primary_weight,
            secondary_weight=args.dual_secondary_weight,
            base_weight=args.dual_base_weight,
            rrf_k=args.dual_rrf_k,
            cache_dir=cache_dir,
        )
    if method == "doc_aware_ranker":
        if memory_train_samples is None:
            raise ValueError("doc_aware_ranker method requires memory_train_samples")
        return doc_aware_ranker_retrieve_batch(
            samples,
            memory_train_samples=memory_train_samples,
            dense_model_name=args.dense_model,
            device=args.device,
            batch_size=args.batch_size,
            top_k=args.top_k,
            bm25_weight=args.bm25_weight,
            dense_weight=args.dense_weight,
            memory_weight=args.memory_weight,
            similar_memory_weight=args.similar_memory_weight,
            type_weight=args.type_weight,
            page_weight=args.page_weight,
            layout_weight=args.layout_weight,
            similar_memory_topn=args.similar_memory_topn,
            ranker_max_iter=args.ranker_max_iter,
            ranker_learning_rate=args.ranker_learning_rate,
            ranker_max_depth=args.ranker_max_depth,
            ranker_min_samples_leaf=args.ranker_min_samples_leaf,
            ranker_positive_weight=args.ranker_positive_weight,
            ranker_model_weight=args.ranker_model_weight,
            ranker_base_weight=args.ranker_base_weight,
            ranker_domain_balance=args.ranker_domain_balance,
            seed=args.seed,
            use_reranker=args.use_reranker,
            reranker_model_name=args.reranker_model,
            rerank_topk=args.rerank_topk,
            reranker_weight=args.reranker_weight,
            use_comparative_coverage=args.use_comparative_coverage,
            coverage_page_penalty=args.coverage_page_penalty,
            coverage_layout_penalty=args.coverage_layout_penalty,
            coverage_modality_penalty=args.coverage_modality_penalty,
            coverage_raw_type_penalty=args.coverage_raw_type_penalty,
            coverage_missing_bonus=args.coverage_missing_bonus,
            cache_dir=cache_dir,
        )
    if method == "routed_doc_aware":
        if memory_train_samples is None:
            raise ValueError("routed_doc_aware method requires memory_train_samples")
        return routed_doc_aware_ranker_retrieve_batch(
            samples,
            memory_train_samples=memory_train_samples,
            dense_model_name=args.dense_model,
            device=args.device,
            batch_size=args.batch_size,
            top_k=args.top_k,
            bm25_weight=args.bm25_weight,
            dense_weight=args.dense_weight,
            memory_weight=args.memory_weight,
            similar_memory_weight=args.similar_memory_weight,
            type_weight=args.type_weight,
            page_weight=args.page_weight,
            layout_weight=args.layout_weight,
            similar_memory_topn=args.similar_memory_topn,
            ranker_max_iter=args.ranker_max_iter,
            ranker_learning_rate=args.ranker_learning_rate,
            ranker_max_depth=args.ranker_max_depth,
            ranker_min_samples_leaf=args.ranker_min_samples_leaf,
            ranker_positive_weight=args.ranker_positive_weight,
            ranker_model_weight=args.ranker_model_weight,
            ranker_base_weight=args.ranker_base_weight,
            route_low_doc_count=args.route_low_doc_count,
            route_high_doc_count=args.route_high_doc_count,
            route_low_confidence=args.route_low_confidence,
            route_high_confidence=args.route_high_confidence,
            route_low_final_quota=args.route_low_final_quota,
            route_low_overlap_quota=args.route_low_overlap_quota,
            route_low_semantic_quota=args.route_low_semantic_quota,
            route_high_final_quota=args.route_high_final_quota,
            route_high_overlap_quota=args.route_high_overlap_quota,
            route_high_semantic_quota=args.route_high_semantic_quota,
            ranker_domain_balance=args.ranker_domain_balance,
            seed=args.seed,
            use_comparative_coverage=args.use_comparative_coverage,
            coverage_page_penalty=args.coverage_page_penalty,
            coverage_layout_penalty=args.coverage_layout_penalty,
            coverage_modality_penalty=args.coverage_modality_penalty,
            coverage_raw_type_penalty=args.coverage_raw_type_penalty,
            coverage_missing_bonus=args.coverage_missing_bonus,
            cache_dir=cache_dir,
        )
    if method == "llm":
        return llm_retrieve_batch(
            samples,
            model_name=args.llm_model,
            device=args.device,
            top_k=args.top_k,
            max_words_per_candidate=args.llm_max_words_per_candidate,
        )
    if method == "image":
        return image_retrieve_batch(
            samples,
            model_name=args.image_model,
            device=args.device,
            batch_size=args.batch_size,
            top_k=args.top_k,
            cache_dir=cache_dir,
        )
    raise ValueError(f"Unknown retrieval method: {method}")


def run_eval(train_samples: Sequence[Sample], args: argparse.Namespace) -> EvalResult:
    memory_train_samples, dev_samples = make_eval_split(
        train_samples,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        split_strategy=args.eval_split,
        mixed_overlap_ratio=args.mixed_overlap_ratio,
        holdout_domain=args.holdout_domain,
    )
    dev_samples = apply_shard(dev_samples, shard_id=args.shard_id, num_shards=args.num_shards)
    predictions = retrieve_predictions(
        dev_samples,
        args=args,
        method=args.method,
        memory_train_samples=memory_train_samples,
    )
    result = evaluate_predictions(dev_samples, predictions, method=args.method)
    result.extra["num_dev_docs"] = len({sample.doc_name for sample in dev_samples})
    result.extra["eval_split"] = args.eval_split
    return result


def run_ablation_suite(train_samples: Sequence[Sample], args: argparse.Namespace) -> Dict[str, Any]:
    memory_train_samples, dev_samples = make_eval_split(
        train_samples,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        split_strategy=args.eval_split,
        mixed_overlap_ratio=args.mixed_overlap_ratio,
        holdout_domain=args.holdout_domain,
    )
    methods = ["bm25", "dense", "hybrid", "typed_hybrid", "memory", "memory_hybrid", "typed_memory_hybrid", "dual_reranker", "doc_aware_ranker", "routed_doc_aware", "llm", "image"]
    results: Dict[str, Any] = {}
    for method in methods:
        try:
            predictions = retrieve_predictions(
                dev_samples,
                args=args,
                method=method,
                memory_train_samples=memory_train_samples,
            )
            eval_result = evaluate_predictions(dev_samples, predictions, method=method)
            results[method] = {
                "recall_at_5": round(eval_result.recall_at_5, 6),
                "evaluated_samples": eval_result.evaluated_samples,
                "skipped_samples": eval_result.skipped_samples,
            }
        except Exception as exc:  # pragma: no cover - optional dependency path
            results[method] = {"error": str(exc)}
    return results


def run_analysis(
    train_samples: Sequence[Sample],
    test_samples: Sequence[Sample],
    args: argparse.Namespace,
) -> Dict[str, Any]:
    memory_train_samples, dev_samples = make_eval_split(
        train_samples,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        split_strategy=args.eval_split,
        mixed_overlap_ratio=args.mixed_overlap_ratio,
        holdout_domain=args.holdout_domain,
    )
    predictions = retrieve_predictions(
        dev_samples,
        args=args,
        method=args.method,
        memory_train_samples=memory_train_samples,
    )
    dev_doc_counts = Counter(sample.doc_name for sample in memory_train_samples)
    full_train_doc_counts = Counter(sample.doc_name for sample in train_samples)
    dev_best_neighbor_scores = best_neighbor_score_by_qid(
        dev_samples,
        reference_samples=memory_train_samples,
        top_n=args.similar_memory_topn,
        exclude_self_from_reference=False,
    )
    analysis = analyze_modality_preference(dev_samples, predictions, method_name=args.method)
    analysis["eval"] = json.loads(format_eval_result(evaluate_predictions(dev_samples, predictions, method=args.method)))
    analysis["doc_frequency_buckets"] = {
        "dev_distribution": bucket_proportions(dev_samples, dev_doc_counts),
        "dev_bucket_eval": evaluate_predictions_by_doc_frequency(
            dev_samples,
            predictions,
            method=args.method,
            doc_counts=dev_doc_counts,
        ),
    }
    dev_overlap_buckets = {
        q_id: overlap_confidence_bucket(score)
        for q_id, score in dev_best_neighbor_scores.items()
    }
    analysis["overlap_confidence_buckets"] = {
        "dev_distribution": bucket_proportions_from_map(dev_overlap_buckets),
        "dev_bucket_eval": evaluate_predictions_by_bucket_map(
            dev_samples,
            predictions,
            method=args.method,
            bucket_by_qid=dev_overlap_buckets,
        ),
    }
    analysis["domain_buckets"] = {
        "dev_distribution": bucket_proportions_from_map(domain_by_qid(dev_samples)),
        "dev_bucket_eval": evaluate_predictions_by_bucket_map(
            dev_samples,
            predictions,
            method=args.method,
            bucket_by_qid=domain_by_qid(dev_samples),
        ),
    }
    analysis["question_type_buckets"] = {
        "dev_distribution": bucket_proportions_from_map(question_type_by_qid(dev_samples)),
        "dev_bucket_eval": evaluate_predictions_by_bucket_map(
            dev_samples,
            predictions,
            method=args.method,
            bucket_by_qid=question_type_by_qid(dev_samples),
        ),
    }
    analysis["gold_len_buckets"] = {
        "dev_distribution": bucket_proportions_from_map(gold_len_bucket_by_qid(dev_samples)),
        "dev_bucket_eval": evaluate_predictions_by_bucket_map(
            dev_samples,
            predictions,
            method=args.method,
            bucket_by_qid=gold_len_bucket_by_qid(dev_samples),
        ),
    }
    analysis["gold_modality_buckets"] = {
        "dev_distribution": bucket_proportions_from_map(gold_modality_bucket_by_qid(dev_samples)),
        "dev_bucket_eval": evaluate_predictions_by_bucket_map(
            dev_samples,
            predictions,
            method=args.method,
            bucket_by_qid=gold_modality_bucket_by_qid(dev_samples),
        ),
    }
    if test_samples:
        test_bucket_weights = bucket_proportions(test_samples, full_train_doc_counts)
        analysis["doc_frequency_buckets"]["test_distribution"] = test_bucket_weights
        analysis["doc_frequency_buckets"]["weighted_proxy_recall"] = weighted_bucket_recall(
            analysis["doc_frequency_buckets"]["dev_bucket_eval"],
            test_bucket_weights,
        )
        analysis["doc_frequency_buckets"]["strict_proxy_recall"] = strict_weighted_bucket_recall(
            analysis["doc_frequency_buckets"]["dev_bucket_eval"],
            test_bucket_weights,
        )
        analysis["doc_frequency_buckets"]["coverage"] = bucket_coverage(
            analysis["doc_frequency_buckets"]["dev_bucket_eval"],
            test_bucket_weights,
        )
        test_best_neighbor_scores = best_neighbor_score_by_qid(
            test_samples,
            reference_samples=train_samples,
            top_n=args.similar_memory_topn,
            exclude_self_from_reference=False,
        )
        test_overlap_buckets = {
            q_id: overlap_confidence_bucket(score)
            for q_id, score in test_best_neighbor_scores.items()
        }
        analysis["overlap_confidence_buckets"]["test_distribution"] = bucket_proportions_from_map(test_overlap_buckets)
        analysis["overlap_confidence_buckets"]["weighted_proxy_recall"] = weighted_bucket_recall(
            analysis["overlap_confidence_buckets"]["dev_bucket_eval"],
            analysis["overlap_confidence_buckets"]["test_distribution"],
        )
        analysis["overlap_confidence_buckets"]["strict_proxy_recall"] = strict_weighted_bucket_recall(
            analysis["overlap_confidence_buckets"]["dev_bucket_eval"],
            analysis["overlap_confidence_buckets"]["test_distribution"],
        )
        analysis["overlap_confidence_buckets"]["coverage"] = bucket_coverage(
            analysis["overlap_confidence_buckets"]["dev_bucket_eval"],
            analysis["overlap_confidence_buckets"]["test_distribution"],
        )
        test_domains = bucket_proportions_from_map(domain_by_qid(test_samples))
        analysis["domain_buckets"]["test_distribution"] = test_domains
        analysis["domain_buckets"]["weighted_proxy_recall"] = weighted_bucket_recall(
            analysis["domain_buckets"]["dev_bucket_eval"],
            test_domains,
        )
        analysis["domain_buckets"]["strict_proxy_recall"] = strict_weighted_bucket_recall(
            analysis["domain_buckets"]["dev_bucket_eval"],
            test_domains,
        )
        analysis["domain_buckets"]["coverage"] = bucket_coverage(
            analysis["domain_buckets"]["dev_bucket_eval"],
            test_domains,
        )
    return analysis


def parse_float_grid(raw: str) -> List[float]:
    values = []
    for item in raw.split(","):
        item = item.strip()
        if item:
            values.append(float(item))
    return values


def run_weight_sweep(train_samples: Sequence[Sample], args: argparse.Namespace) -> Dict[str, Any]:
    memory_train_samples, dev_samples = make_eval_split(
        train_samples,
        dev_ratio=args.dev_ratio,
        seed=args.seed,
        split_strategy=args.eval_split,
        mixed_overlap_ratio=args.mixed_overlap_ratio,
        holdout_domain=args.holdout_domain,
    )
    dev_samples = apply_shard(dev_samples, shard_id=args.shard_id, num_shards=args.num_shards)
    cache_dir = Path(args.cache_dir) if args.cache_dir else None
    memory = build_gold_memory(memory_train_samples)
    bm25_score_map = {sample.q_id: bm25_score_candidates(sample.question, sample.candidates) for sample in dev_samples}
    dense_score_map = dense_score_batch(
        dev_samples,
        model_name=args.dense_model,
        device=args.device,
        batch_size=args.batch_size,
        cache_dir=cache_dir,
    )
    memory_score_map = {
        sample.q_id: memory_score_candidates(sample, sample.candidates, memory)
        for sample in dev_samples
    }
    type_score_map = {
        sample.q_id: expected_type_score_candidates(sample, sample.candidates)
        for sample in dev_samples
    }

    results = []
    for bm25_weight in parse_float_grid(args.sweep_bm25_weights):
        for dense_weight in parse_float_grid(args.sweep_dense_weights):
            for memory_weight in parse_float_grid(args.sweep_memory_weights):
                for type_weight in parse_float_grid(args.sweep_type_weights):
                    predictions: Dict[str, List[str]] = {}
                    for sample in dev_samples:
                        scores = (
                            bm25_weight * minmax_normalize(bm25_score_map[sample.q_id])
                            + dense_weight * minmax_normalize(dense_score_map[sample.q_id])
                            + memory_weight * minmax_normalize(memory_score_map[sample.q_id])
                            + type_weight * minmax_normalize(type_score_map[sample.q_id])
                        )
                        scores = apply_context_boost(
                            sample,
                            scores,
                            page_weight=args.page_weight,
                            layout_weight=args.layout_weight,
                        )
                        ranked_indices = stable_rank_indices(scores)
                        predictions[sample.q_id] = [
                            sample.candidates[idx].quote_id for idx in ranked_indices[: args.top_k]
                        ]
                    eval_result = evaluate_predictions(dev_samples, predictions, method="sweep")
                    results.append(
                        {
                            "recall_at_5": round(eval_result.recall_at_5, 6),
                            "bm25_weight": bm25_weight,
                            "dense_weight": dense_weight,
                            "memory_weight": memory_weight,
                            "type_weight": type_weight,
                        }
                    )

    results.sort(key=lambda row: row["recall_at_5"], reverse=True)
    return {
        "eval_split": args.eval_split,
        "evaluated_samples": len(dev_samples),
        "num_dev_docs": len({sample.doc_name for sample in dev_samples}),
        "top_results": results[: args.sweep_top_n],
    }


# ============================================================================
# CLI
# ============================================================================


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="INLP HW3 multimodal retrieval pipeline")
    parser.add_argument("--mode", choices=["stats", "eval", "dump_eval", "ablate", "analyze", "sweep", "submit"], required=True)
    parser.add_argument(
        "--method",
        choices=["bm25", "dense", "hybrid", "typed_hybrid", "memory", "memory_hybrid", "typed_memory_hybrid", "dual_reranker", "doc_aware_ranker", "routed_doc_aware", "llm", "image"],
        default="hybrid",
    )
    parser.add_argument("--train-path", default="train.jsonl")
    parser.add_argument("--test-path", default="test.jsonl")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--cache-dir", default=".cache/hw3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.10)
    parser.add_argument("--eval-split", choices=["doc", "row", "mixed", "domain"], default="doc")
    parser.add_argument("--mixed-overlap-ratio", type=float, default=0.74)
    parser.add_argument("--holdout-domain", default="")
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)

    parser.add_argument("--dense-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--bm25-weight", type=float, default=0.35)
    parser.add_argument("--dense-weight", type=float, default=0.65)
    parser.add_argument("--memory-weight", type=float, default=0.55)
    parser.add_argument("--type-weight", type=float, default=0.35)
    parser.add_argument("--page-weight", type=float, default=0.0)
    parser.add_argument("--layout-weight", type=float, default=0.0)
    parser.add_argument("--force-memory-top", type=int, default=0)
    parser.add_argument("--similar-memory-weight", type=float, default=0.0)
    parser.add_argument("--similar-memory-topn", type=int, default=3)
    parser.add_argument("--sweep-bm25-weights", default="0.25,0.35,0.45")
    parser.add_argument("--sweep-dense-weights", default="0.55,0.65,0.75")
    parser.add_argument("--sweep-memory-weights", default="0.25,0.4,0.55,0.7,0.9")
    parser.add_argument("--sweep-type-weights", default="0.0,0.05,0.1,0.15,0.25")
    parser.add_argument("--sweep-top-n", type=int, default=20)

    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-large")
    parser.add_argument("--second-reranker-model", default="BAAI/bge-reranker-large")
    parser.add_argument("--rerank-topk", type=int, default=10)
    parser.add_argument("--reranker-weight", type=float, default=0.50)
    parser.add_argument("--dual-primary-weight", type=float, default=1.15)
    parser.add_argument("--dual-secondary-weight", type=float, default=1.0)
    parser.add_argument("--dual-base-weight", type=float, default=0.45)
    parser.add_argument("--dual-rrf-k", type=float, default=60.0)
    parser.add_argument("--ranker-max-iter", type=int, default=220)
    parser.add_argument("--ranker-learning-rate", type=float, default=0.05)
    parser.add_argument("--ranker-max-depth", type=int, default=6)
    parser.add_argument("--ranker-min-samples-leaf", type=int, default=20)
    parser.add_argument("--ranker-positive-weight", type=float, default=4.0)
    parser.add_argument("--ranker-model-weight", type=float, default=0.75)
    parser.add_argument("--ranker-base-weight", type=float, default=0.25)
    parser.add_argument("--ranker-domain-balance", choices=["none", "uniform", "sqrt"], default="none")
    parser.add_argument("--use-comparative-coverage", action="store_true")
    parser.add_argument("--coverage-page-penalty", type=float, default=0.08)
    parser.add_argument("--coverage-layout-penalty", type=float, default=0.04)
    parser.add_argument("--coverage-modality-penalty", type=float, default=0.05)
    parser.add_argument("--coverage-raw-type-penalty", type=float, default=0.03)
    parser.add_argument("--coverage-missing-bonus", type=float, default=0.12)
    parser.add_argument("--route-low-doc-count", type=int, default=3)
    parser.add_argument("--route-high-doc-count", type=int, default=10)
    parser.add_argument("--route-low-confidence", type=float, default=0.20)
    parser.add_argument("--route-high-confidence", type=float, default=0.40)
    parser.add_argument("--route-low-final-quota", type=int, default=3)
    parser.add_argument("--route-low-overlap-quota", type=int, default=1)
    parser.add_argument("--route-low-semantic-quota", type=int, default=1)
    parser.add_argument("--route-high-final-quota", type=int, default=2)
    parser.add_argument("--route-high-overlap-quota", type=int, default=2)
    parser.add_argument("--route-high-semantic-quota", type=int, default=1)

    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--llm-max-words-per-candidate", type=int, default=320)

    parser.add_argument("--image-model", default="openai/clip-vit-base-patch32")

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root).resolve()
    train_path = Path(args.train_path).resolve()
    test_path = Path(args.test_path).resolve()

    if args.mode in {"stats", "eval", "ablate", "analyze", "sweep"} and not train_path.exists():
        raise FileNotFoundError(f"Training file not found: {train_path}")
    if args.mode == "submit" and not test_path.exists():
        raise FileNotFoundError(f"Test file not found: {test_path}")

    train_samples = load_samples(train_path, data_root) if train_path.exists() else []
    test_samples = load_samples(test_path, data_root) if test_path.exists() else []

    if args.mode == "stats":
        stats = summarize_stats(train_samples, test_samples)
        print(json.dumps(stats, indent=2, ensure_ascii=False))
        return 0

    if args.mode == "eval":
        result = run_eval(train_samples, args)
        print(format_eval_result(result))
        return 0

    if args.mode == "dump_eval":
        memory_train_samples, dev_samples = make_eval_split(
            train_samples,
            dev_ratio=args.dev_ratio,
            seed=args.seed,
            split_strategy=args.eval_split,
            mixed_overlap_ratio=args.mixed_overlap_ratio,
            holdout_domain=args.holdout_domain,
        )
        dev_samples = apply_shard(dev_samples, shard_id=args.shard_id, num_shards=args.num_shards)
        predictions = retrieve_predictions(
            dev_samples,
            args=args,
            method=args.method,
            memory_train_samples=memory_train_samples,
        )
        output_path = Path(args.output).resolve()
        write_submission(output_path, dev_samples, predictions)
        result = evaluate_predictions(dev_samples, predictions, method=args.method)
        print(
            json.dumps(
                {
                    "output": str(output_path),
                    "num_rows": len(dev_samples),
                    "method": args.method,
                    "recall_at_5": round(result.recall_at_5, 6),
                    "eval_split": args.eval_split,
                },
                ensure_ascii=False,
            )
        )
        return 0

    if args.mode == "ablate":
        results = run_ablation_suite(train_samples, args)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return 0

    if args.mode == "analyze":
        analysis = run_analysis(train_samples, test_samples, args)
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
        return 0

    if args.mode == "sweep":
        sweep = run_weight_sweep(train_samples, args)
        print(json.dumps(sweep, indent=2, ensure_ascii=False))
        return 0

    if args.mode == "submit":
        test_samples = apply_shard(test_samples, shard_id=args.shard_id, num_shards=args.num_shards)
        predictions = retrieve_predictions(
            test_samples,
            args=args,
            method=args.method,
            memory_train_samples=train_samples,
        )
        output_path = Path(args.output).resolve()
        write_submission(output_path, test_samples, predictions)
        print(json.dumps({"output": str(output_path), "num_rows": len(test_samples)}, ensure_ascii=False))
        return 0

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    sys.exit(main())
