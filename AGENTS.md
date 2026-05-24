# INLP HW3 Agent Guide

## Objective

This homework is about **multimodal Retrieval-Augmented Generation (RAG) retrieval**.

The system should:

- Take a `question` and its associated document sample.
- Retrieve the **top 5 most relevant evidence items** from the provided candidates.
- Rank evidence across **all modalities together**.
- Focus only on **retrieval**, not final answer generation.

The goal is to maximize overlap with the ground-truth evidence using **Recall@5**.

## What The Homework Requires

For each sample:

- Read the question.
- Consider both:
  - `text_quotes`
  - `img_quotes`
- Return one ranked list of **5 `quote_id`s total**.
- The 5 results should be sorted from most relevant to least relevant.

Important:

- This is **not** "5 text + 5 image".
- This is **one combined top-5 list**.
- Using raw images is optional because each image already has an `img_description`.

## Constraints

- Only **open-weight** models are allowed.
- Model size must be **80B parameters or below**.
- Any API used in the retrieval pipeline must also be backed by an open-weight model within the same limit.
- Closed-source APIs such as GPT, Claude, etc. are **not allowed**.

## Dataset In This Folder

Expected files:

- `train.jsonl`: labeled training/experiment data
- `test.jsonl`: unlabeled test data for Kaggle submission
- `sample_submission.csv`: submission format example
- `images/`: image assets referenced by `img_quotes`
- `INLP-HW3.pptx.pdf`: homework specification

Each sample contains fields such as:

- `q_id`: question id
- `doc_name`, `domain`
- `question`
- `evidence_modality_type`
- `text_quotes`: candidate text evidence
- `img_quotes`: candidate image/table/chart evidence with `img_path` and `img_description`

Training samples also include:

- `gold_quotes`: ground-truth supporting `quote_id`s
- `answer_short`, `answer_interleaved`: reference answers for development only

## Submission Requirements

### Kaggle

Submit a CSV with exactly 2 columns:

- `q_id`
- `gold_quotes`

Rules:

- One row per question in `test.jsonl`
- `q_id` must match exactly
- `gold_quotes` must contain up to 5 predicted `quote_id`s
- The ids must be separated by a **single space**

Example:

```csv
q_id,gold_quotes
0,text1 text2 image3 text4 image5
```

Metric:

- **Recall@5**

### E3

Submit:

- source code
- report

Expected archive format:

- `HW3_<student ID>.zip`

Containing:

- `HW3_<student ID>.py` or `HW3_<student ID>.ipynb`
- `HW3_<student ID>.pdf`

## Report Questions

The report must cover these 4 parts:

1. **Method Description**
   - overall retrieval pipeline
   - preprocessing
   - ranking
   - extra techniques

2. **Comparison of Retrieval Methods**
   - BM25
   - Dense Retriever
   - direct LLM selection
   - your final method

3. **Multimodal Embedding vs. Text-Description Retrieval**
   - direct image embedding
   - image description as text retrieval
   - which one works better and why

4. **Modality Preference Analysis**
   - whether the system prefers text or image evidence
   - why that preference appears, or why it stays balanced

## Recommended Work Plan

1. Build a strong text-based retriever using:
   - `text_quotes`
   - `img_description` from `img_quotes`

2. Implement and compare:
   - BM25 baseline
   - Dense retrieval baseline
   - Hybrid method
   - Optional reranking

3. Evaluate on `train.jsonl` with a local validation split using `gold_quotes`.

4. Generate predictions for `test.jsonl`.

5. Export a Kaggle submission CSV.

6. Record experiments carefully for the report.

## Non-Goals

- Do not generate final natural-language answers as the main task.
- Do not use closed-source LLM APIs.
- Do not retrieve separately per modality and then submit more than 5 total items.

## Practical Notes For Future Agents

- Prefer treating `img_description` as text first; it is the fastest valid baseline.
- Keep retrieval output as `quote_id`s, not evidence text.
- Ensure ranking is **within each sample's candidate pool**.
- Validate submission formatting before uploading.
- Keep experiment logs because the report requires result-based analysis, not just method descriptions.
