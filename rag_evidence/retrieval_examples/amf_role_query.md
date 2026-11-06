# Retrieval Example: AMF Role Query

## Query

What is the role of the AMF in the 5G System?

## Recorded pipeline output

| Pipeline stage | Observed count |
|---|---:|
| Dense retrieval | 10 |
| Sparse retrieval | 10 |
| Fusion | 10 |
| Reranking | 5 |
| Expanded context | 10 |
| Compact evidence | 6 blocks |

Compact evidence length: approximately 7,943 characters.

## Query analysis

- Detected query type: `role`
- Extracted target: `the`
- Expected target from the query: `AMF`

The extracted target indicates a query-target extraction issue.

## Retrieval observations

### Dense retrieval

The top dense result had a score of approximately `0.790945`.

- Source: `23502-k30.docx`
- Specification: 3GPP TS 23.502
- Section: 5.2.2
- Title: AMF Services
- Observation: The retrieved chunk was heading-only.

The second dense result had a score of approximately `0.777759`.

- Source: `23501.docx`
- Specification: 3GPP TS 23.501
- Section: 6.3.5
- Title: AMF discovery and selection

### Sparse retrieval

The top sparse result had a score of approximately `38.546501`.

The result was from TS 33.501 and concerned the EAP framework, rather than directly answering the AMF role query.

This illustrates that lexical matches can introduce less relevant results.

### Fusion and reranking

The fused results included both AMF-related material and less relevant matches.

The reranker returned five results. The inspected implementation uses score-based sorting, not a neural cross-encoder.

### Hierarchical expansion

Expansion recovered surrounding AMF-related material, including:

- TS 23.502 section 5.2.2.1, General
- TS 23.502 section 5.2.2.2, Namf_Communication
- TS 23.502 section 5.2.2.2.1, General

The recovered context included information about AMF services, N1/NAS and N2 communication, messages, security context, and UE information management.

## Evidence sufficiency and prompt preparation

The recorded trace reported:

- Evidence sufficiency: `True`
- Should infer: `True`
- Direct response: `None`
- Prompt generated: `True`

The sufficiency result must be interpreted carefully. For this non-definition query type, the current checker accepts non-empty evidence without a strict relevance threshold.

## Interpretation

This trace demonstrates that the retrieval pipeline executed through
prompt preparation and produced compact evidence for an AMF-related query.

It also reveals three limitations:

1. The query target was extracted as `the` instead of `AMF`.
2. Sparse retrieval included an unrelated EAP-framework result.
3. A heading-only chunk appeared at the top of dense retrieval.

## Scope of this evidence

This is a summary of a recorded retrieval debug trace, not a newly
executed benchmark.

It demonstrates retrieval and prompt preparation. It does not establish
that the language model completed inference or generated a final answer.

Raw debug-log output is not included in this file.
