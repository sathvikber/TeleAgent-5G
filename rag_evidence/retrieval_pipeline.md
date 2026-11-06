# TeleAgent-5G: RAG Retrieval Pipeline

## 1. Overview

TeleAgent-5G uses a retrieval-augmented generation (RAG) pipeline to retrieve
relevant 5G and telecom knowledge before preparing context for the language
model.

The retrieval implementation combines semantic search and lexical search,
merges their results, expands relevant sections, and constructs compact
evidence for prompt preparation.

The implementation is located primarily in:

- `teleagent/retrieval/intelligent_retrieval_service.py`
- `teleagent/retrieval/dense_retriever.py`
- `teleagent/retrieval/sparse_retriever.py`
- `teleagent/retrieval/fusion.py`
- `teleagent/retrieval/reranker.py`
- `teleagent/retrieval/evidence_sufficiency_checker.py`

## 2. Retrieval stages

The implemented pipeline follows this sequence:

1. Query processing and query-type detection.
2. Dense semantic retrieval.
3. Sparse lexical retrieval using BM25.
4. Optional filtering by requested 3GPP specification.
5. Reciprocal Rank Fusion of retrieval results.
6. Score-based reranking.
7. Hierarchical expansion of retrieved sections.
8. Compact evidence selection and context construction.
9. Evidence sufficiency checking.
10. RAG prompt preparation when the evidence check passes.

## 3. Dense retrieval

Dense retrieval uses the existing embedding and FAISS vector-search
implementation.

The knowledge-base metadata identifies the embedding model as:

`BAAI/bge-small-en-v1.5`

The indexed vectors have:

- Vector dimension: 384
- Number of indexed vectors: 16,061
- FAISS index type: `IndexFlatIP`

The embedding implementation normalizes embeddings before indexing and
searching. Inner-product search over normalized vectors is used as a
cosine-similarity-style semantic search.

The `DenseRetriever` component wraps the existing retriever implementation.

## 4. Sparse retrieval

Sparse retrieval is implemented using `BM25Okapi` from the `rank_bm25`
package.

The current implementation:

- Converts chunk content to lowercase.
- Splits content using whitespace.
- Applies the same lowercase whitespace tokenization to queries.
- Uses BM25 to rank lexical matches.

No stemming or specialized telecom tokenizer was observed in this
implementation.

Sparse retrieval can therefore complement semantic retrieval by finding
literal term matches, although tokenization and terminology differences
can affect its results.

## 5. Result fusion

The `Fusion` component implements Reciprocal Rank Fusion (RRF).

The configured RRF constant is:

`k = 60`

Each result contributes a reciprocal-rank score based on its position in
the retrieval lists. Results are combined by chunk ID, avoiding duplicate
entries for the same chunk.

This combines the rankings from dense and sparse retrieval rather than
directly comparing their raw scores.

## 6. Reranking

The current `ReRanker` sorts retrieved results by their existing score in
descending order and returns the requested top results.

Important implementation distinction:

- The active implementation is score-based sorting.
- A cross-encoder or neural reranker is not currently demonstrated by
  this implementation.
- BGE reranker model names in comments are references to possible
  future implementations, not evidence that those models are active.

## 7. Hierarchical evidence expansion

After reranking, the intelligent retrieval service expands selected
results using the knowledge-base chunk structure.

The implementation can include:

- The retrieved anchor chunk.
- Continuation chunks from the same section.
- Child-section chunks.
- Nearby chunks for certain short-heading cases.

Expansion is bounded by a maximum result count and uses chunk and section
metadata to avoid duplicate results.

This stage is intended to recover useful surrounding context when an
initial retrieval result is a short heading or only part of a larger
technical section.

## 8. Compact evidence construction

The service selects and ranks expanded results to construct compact
evidence for the prompt.

The selection logic considers query terms and metadata such as:

- Section title.
- Chunk content.
- Network-function information.
- Chunk length.
- Short-heading characteristics.

The compact evidence builder can include canonical definitions,
entity hints, specification information, a primary section, and selected
evidence blocks.

The implementation limits the compact evidence collection to a maximum
of six blocks in the observed configuration.

## 9. Evidence sufficiency checking

The `EvidenceSufficiencyChecker` evaluates whether retrieved evidence
can support the detected query type.

The current implementation has different behavior for definition
queries and other query types.

For definition queries, it applies explicit definition-related checks.

For non-definition query types, the implementation currently accepts
non-empty evidence with the reason:

`No strict sufficiency gate is currently enabled for query type ...`

Therefore, this component should not be described as a universal
hallucination-prevention mechanism. Its checks are query-type dependent,
and strict sufficiency validation is not enabled for every query type.

## 10. Candidate limits in the default top_k=5 path

The service's configured maximum limits for a call using `top_k=5` are:

| Stage | Configured maximum |
|---|---:|
| Dense retrieval | 30 |
| Sparse retrieval | 30 |
| Fusion output | 20 |
| Reranked results | 5 |
| Expanded results | 10 |
| Compact evidence blocks | 6 |

These are configured maximums, not guaranteed result counts. Actual
counts depend on the query, available matches, filtering, and expansion.

## 11. Observed retrieval trace: AMF role query

Query:

`What is the role of the AMF in the 5G System?`

The recorded debug trace showed:

| Stage | Observed result count |
|---|---:|
| Dense retrieval | 10 |
| Sparse retrieval | 10 |
| Fusion | 10 |
| Reranking | 5 |
| Expanded context | 10 |
| Compact evidence | 6 blocks |

The compact evidence contained approximately 7,943 characters.

The trace also reported:

- Query type: `role`
- Extracted target: `the`
- Evidence sufficiency: `True`
- Prompt generated: `True`
- Should infer: `True`

### Observations from the trace

The trace demonstrates that the retrieval pipeline executed through
compact evidence construction and prompt preparation.

It also exposes areas for improvement:

1. The query-type detector classified the query as a role query, but
   extracted `the` as the target instead of `AMF`. This is a query-target
   extraction issue.
2. Dense retrieval included an AMF-related result from 3GPP TS 23.502,
   section 5.2.2, titled AMF Services. The top result was a heading-only
   chunk.
3. Sparse retrieval included some results that were not directly relevant
   to the AMF role query, demonstrating lexical retrieval noise.
4. Hierarchical expansion recovered surrounding AMF service information,
   including sections concerning general AMF services and
   `Namf_Communication`.
5. The sufficiency check returned true because non-empty evidence was
   accepted for this non-definition query type. This does not establish
   that every evidence block was relevant or sufficient.

### What this trace proves

The trace supports the following claims:

- Dense and sparse retrieval were executed.
- Their results were fused.
- Reranking and hierarchical expansion were executed.
- Compact evidence was constructed.
- The sufficiency checker returned true.
- A RAG prompt was prepared.

The trace does **not** by itself prove that the language model completed
inference or generated a final answer. Prompt preparation and final
model generation are separate stages.

## 12. Limitations and improvement opportunities

The inspected implementation and trace identify these improvement areas:

- Improve query-target extraction for role-based questions.
- Reduce irrelevant lexical matches through improved tokenization or
  telecom-aware query processing.
- Avoid allowing heading-only chunks to dominate retrieval results.
- Consider a true cross-encoder reranker if evaluation justifies it.
- Introduce stricter evidence sufficiency checks for non-definition
  query types.
- Evaluate whether hierarchical expansion improves answer relevance
  without introducing unrelated context.

These are observations and proposed improvements, not claims that the
changes have already been implemented.

## 13. Reproducibility notes

This report describes the inspected source implementation and the
recorded AMF retrieval trace.

Configured candidate limits should not be confused with actual result
counts. Likewise, successful prompt preparation should not be presented
as proof of successful final language-model inference.

The knowledge-base inventory is documented separately in
`knowledge_base_summary.json`, and representative metadata-only chunk
records are documented in `sample_chunks.jsonl`.
