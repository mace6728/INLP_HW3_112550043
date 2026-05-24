# Report Questions

## Q1. Method Description (5%)

Clearly describe your overall retrieval pipeline and the main idea behind your method.

Explain:

- the retriever
- evidence preprocessing
- the ranking procedure
- any additional techniques you used

## Q2. Comparison of Retrieval Methods (5%)

Based on your experimental results, compare the performance of:

- BM25
- Dense Retriever
- direct LLM selection
- your own method

Analyze the strengths and weaknesses of all four methods, and explain what you think causes the performance differences.

## Q3. Multimodal Embedding vs. Text-Description Retrieval for Image Evidence (10%)

There are two common ways to retrieve image evidence:

1. Use a multimodal embedding model to encode the image directly into a vector, then compare it with the question embedding.
2. Convert each image into a text description, then retrieve it together with other text evidence using a text-only retriever. You may either use the `img_description` field already provided in the dataset, or generate your own description with a VLM or image-captioning model.

Briefly state which option you chose and why.

Compare your experimental results for these two approaches. Discuss which one performs better on this task and explain what you think causes the difference.

## Q4. Modality Preference Analysis (10%)

Examine whether your retriever shows a preference for a particular modality in its retrieval results.

For example, does it tend to retrieve more text evidence, or does it favor image evidence?

Based on your actual experimental results, describe this preference and analyze what you think causes it.

Even if your retriever shows no clear preference, you are still expected to explain what aspects of your method cause this balanced behavior.
