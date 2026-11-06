# TeleAgent-5G: RAG Evaluation Protocol

## 1. Objective

Evaluate whether retrieval-augmented generation improves the quality
and grounding of answers to questions about 5G networks and 3GPP
technical specifications.

This document defines the evaluation procedure. It does not contain
experimental results.

## 2. Configurations to compare

### A. Baseline: No RAG

- Use the language model without retrieved knowledge-base context.
- Record the generated answer and response latency.

### B. Dense RAG

- Retrieve context using semantic vector search.
- Prepare the model prompt using the retrieved context.
- Record retrieval output, generated answer, and latency.

### C. Hybrid RAG

- Use dense semantic retrieval and sparse BM25 retrieval.
- Combine results through Reciprocal Rank Fusion.
- Apply the implemented reranking and evidence expansion stages.
- Prepare the model prompt using compact evidence.
- Record retrieval output, generated answer, and latency.

Only configurations that can actually be executed should be included
in the final comparison. If a configuration requires additional
implementation, document that limitation rather than presenting it
as a completed experiment.

## 3. Test dataset

Prepare a fixed set of questions covering:

- 5G Core network functions.
- AMF, SMF, UPF, UDM, and PCF responsibilities.
- 5G registration and session-management procedures.
- NAS and other relevant protocols.
- Security procedures.
- 3GPP specification-based technical questions.

For every question, maintain:

- Unique question ID.
- Question text.
- Expected answer or reference facts.
- Relevant specification and section, where known.
- Expected supporting evidence.

Keep evaluation questions and reference answers outside the
knowledge-base ingestion directory.

## 4. Evaluation metrics

### Retrieval quality

- Precision@K: proportion of retrieved chunks judged relevant.
- Recall@K: proportion of known relevant chunks retrieved.
- MRR: reciprocal rank of the first relevant result.
- Evidence relevance: human assessment of whether retrieved passages
  support the question.

### Answer quality

- Correctness: agreement with the reference answer.
- Completeness: coverage of the important reference facts.
- Groundedness: whether claims are supported by retrieved evidence.
- Unsupported claims: factual claims without supporting evidence.

### Performance

- Retrieval latency.
- End-to-end response latency.
- Number of retrieved chunks and evidence blocks.
- Prompt context size, where measurable.

Report the number of test questions and the measurement procedure
alongside every metric.

## 5. Experimental procedure

1. Freeze the evaluation question set.
2. Run each supported configuration against the same questions.
3. Record the retrieved chunks and their source metadata.
4. Record generated answers.
5. Compare answers against the reference facts.
6. Assess whether answer claims are supported by retrieved evidence.
7. Measure latency using the same timing method.
8. Calculate aggregate metrics.
9. Preserve raw outputs and evaluation results.

Use the same model, generation settings, questions, and hardware
where possible. Record any differences that cannot be controlled.

## 6. Results reporting

The final results should include:

| Configuration | Questions | Retrieval metrics | Correctness | Groundedness | Latency |
|---|---:|---|---|---|---|
| No RAG | Pending | N/A | Pending | Pending | Pending |
| Dense RAG | Pending | Pending | Pending | Pending | Pending |
| Hybrid RAG | Pending | Pending | Pending | Pending | Pending |

Do not replace pending values with estimates or expected improvements.

## 7. Reproducibility requirements

Preserve:

- Evaluation question set.
- Reference answers and supporting sections.
- Configuration details.
- Raw retrieval outputs.
- Generated answers.
- Metric calculation procedure.
- Evaluation results.
- Date and environment details for each run.

## 8. Limitations

The existing AMF retrieval trace demonstrates retrieval and prompt
preparation, but it is not a benchmark.

No comparative performance improvement is claimed until the
configurations have been executed on the same evaluation set and
their results have been measured.

Human relevance and groundedness judgments should use documented
criteria so that results can be reviewed and reproduced.
