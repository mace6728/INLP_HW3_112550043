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


def build_candidate_lookup(samples: Sequence[Sample]) -> Dict[str, Dict[str, EvidenceCandidate]]:
    lookup: Dict[str, Dict[str, EvidenceCandidate]] = {}
    for sample in samples:
        lookup[sample.q_id] = {candidate.quote_id: candidate for candidate in sample.candidates}
    return lookup


# ============================================================================
# BM25 baseline
# ============================================================================


def bm25_score_candidates(question: str, candidates: Sequence[EvidenceCandidate], k1: float = 1.5, b: float = 0.75) -> np.ndarray:
    query_tokens = tokenize_for_bm25(question)
    tokenized_docs = [tokenize_for_bm25(candidate_to_text(candidate)) for candidate in candidates]

    if not query_tokens or not tokenized_docs:
        return np.zeros(len(candidates), dtype=np.float32)

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
        fused_scores = (bm25_weight * bm25_scores) + (dense_weight * dense_scores)
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


def retrieve_predictions(samples: Sequence[Sample], args: argparse.Namespace, method: str) -> Dict[str, List[str]]:
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
    if method == "hybrid":
        return hybrid_retrieve_batch(
            samples,
            dense_model_name=args.dense_model,
            device=args.device,
            batch_size=args.batch_size,
            top_k=args.top_k,
            bm25_weight=args.bm25_weight,
            dense_weight=args.dense_weight,
            use_reranker=args.use_reranker,
            reranker_model_name=args.reranker_model,
            rerank_topk=args.rerank_topk,
            reranker_weight=args.reranker_weight,
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
    _, dev_samples = group_split_by_doc(train_samples, dev_ratio=args.dev_ratio, seed=args.seed)
    dev_samples = apply_shard(dev_samples, shard_id=args.shard_id, num_shards=args.num_shards)
    predictions = retrieve_predictions(dev_samples, args=args, method=args.method)
    result = evaluate_predictions(dev_samples, predictions, method=args.method)
    result.extra["num_dev_docs"] = len({sample.doc_name for sample in dev_samples})
    return result


def run_ablation_suite(train_samples: Sequence[Sample], args: argparse.Namespace) -> Dict[str, Any]:
    _, dev_samples = group_split_by_doc(train_samples, dev_ratio=args.dev_ratio, seed=args.seed)
    methods = ["bm25", "dense", "hybrid", "llm", "image"]
    results: Dict[str, Any] = {}
    for method in methods:
        try:
            predictions = retrieve_predictions(dev_samples, args=args, method=method)
            eval_result = evaluate_predictions(dev_samples, predictions, method=method)
            results[method] = {
                "recall_at_5": round(eval_result.recall_at_5, 6),
                "evaluated_samples": eval_result.evaluated_samples,
                "skipped_samples": eval_result.skipped_samples,
            }
        except Exception as exc:  # pragma: no cover - optional dependency path
            results[method] = {"error": str(exc)}
    return results


def run_analysis(train_samples: Sequence[Sample], args: argparse.Namespace) -> Dict[str, Any]:
    _, dev_samples = group_split_by_doc(train_samples, dev_ratio=args.dev_ratio, seed=args.seed)
    predictions = retrieve_predictions(dev_samples, args=args, method=args.method)
    analysis = analyze_modality_preference(dev_samples, predictions, method_name=args.method)
    analysis["eval"] = json.loads(format_eval_result(evaluate_predictions(dev_samples, predictions, method=args.method)))
    return analysis


# ============================================================================
# CLI
# ============================================================================


def parse_args(argv: Optional[Sequence[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="INLP HW3 multimodal retrieval pipeline")
    parser.add_argument("--mode", choices=["stats", "eval", "ablate", "analyze", "submit"], required=True)
    parser.add_argument("--method", choices=["bm25", "dense", "hybrid", "llm", "image"], default="hybrid")
    parser.add_argument("--train-path", default="train.jsonl")
    parser.add_argument("--test-path", default="test.jsonl")
    parser.add_argument("--data-root", default=".")
    parser.add_argument("--output", default="submission.csv")
    parser.add_argument("--cache-dir", default=".cache/hw3")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dev-ratio", type=float, default=0.10)
    parser.add_argument("--top-k", type=int, default=5)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-id", type=int, default=0)

    parser.add_argument("--dense-model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument("--bm25-weight", type=float, default=0.35)
    parser.add_argument("--dense-weight", type=float, default=0.65)

    parser.add_argument("--use-reranker", action="store_true")
    parser.add_argument("--reranker-model", default="BAAI/bge-reranker-large")
    parser.add_argument("--rerank-topk", type=int, default=10)
    parser.add_argument("--reranker-weight", type=float, default=0.50)

    parser.add_argument("--llm-model", default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--llm-max-words-per-candidate", type=int, default=320)

    parser.add_argument("--image-model", default="openai/clip-vit-base-patch32")

    return parser.parse_args(argv)


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = parse_args(argv)
    data_root = Path(args.data_root).resolve()
    train_path = Path(args.train_path).resolve()
    test_path = Path(args.test_path).resolve()

    if args.mode in {"stats", "eval", "ablate", "analyze"} and not train_path.exists():
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

    if args.mode == "ablate":
        results = run_ablation_suite(train_samples, args)
        print(json.dumps(results, indent=2, ensure_ascii=False))
        return 0

    if args.mode == "analyze":
        analysis = run_analysis(train_samples, args)
        print(json.dumps(analysis, indent=2, ensure_ascii=False))
        return 0

    if args.mode == "submit":
        test_samples = apply_shard(test_samples, shard_id=args.shard_id, num_shards=args.num_shards)
        predictions = retrieve_predictions(test_samples, args=args, method=args.method)
        output_path = Path(args.output).resolve()
        write_submission(output_path, test_samples, predictions)
        print(json.dumps({"output": str(output_path), "num_rows": len(test_samples)}, ensure_ascii=False))
        return 0

    raise ValueError(f"Unsupported mode: {args.mode}")


if __name__ == "__main__":
    sys.exit(main())
