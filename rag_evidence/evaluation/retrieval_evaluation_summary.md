# Retrieval Evaluation Results

This run records retrieval outputs from the existing TeleAgent pipeline.

- Questions evaluated: 5
- Dense retrieval: existing FAISS retriever
- Sparse retrieval: existing BM25 retriever
- Fusion: existing Reciprocal Rank Fusion component
- Reranking: existing score-based reranker
- Expanded evidence: existing hierarchical expansion
- Ground-truth relevance labels: not yet assigned
- Precision, recall, and MRR: not calculated

## Q1: What is the role of the AMF in the 5G System?

- Processed query: `What is the role of the AMF in the 5G System?`
- Requested specification: `None`
- Retrieval time: 7.1154 seconds
- dense: 30
- sparse: 30
- fusion: 20
- reranked: 5
- expanded: 10
- compact_evidence_characters: 7943

### Top reranked chunks

1. `23502-k30.docx` | Section 5.2.2 AMF Services | score 0.016393
2. `33501-k30.docx` | Section 6.1.1.2 EAP framework | score 0.016393
3. `23501.docx` | Section 6.3.5 AMF discovery and selection | score 0.016129
4. `23502-k30.docx` | Section 4.11.0 General | score 0.016129
5. `29518-k10.docx` | Section 5 Services offered by the AMF | score 0.015873

## Q2: What is the role of the SMF in the 5G Core?

- Processed query: `What is the role of the SMF in the 5G Core?`
- Requested specification: `None`
- Retrieval time: 5.3066 seconds
- dense: 30
- sparse: 30
- fusion: 20
- reranked: 5
- expanded: 10
- compact_evidence_characters: 7255

### Top reranked chunks

1. `23501.docx` | Section 5.29.3 PDU Session management | score 0.027984
2. `23501.docx` | Section 5.8.2.13.0 General | score 0.023128
3. `23501.docx` | Section 5.34.6 Interaction between I-SMF and SMF for the support of traffic offload by UPF controlled by the I-SMF | score 0.016393
4. `33501-k30.docx` | Section 6.1.1.2 EAP framework | score 0.016393
5. `23502-k30.docx` | Section 5.2.8 SMF Services | score 0.016129

## Q3: What is the role of the UPF in the 5G Core?

- Processed query: `What is the role of the UPF in the 5G Core?`
- Requested specification: `None`
- Retrieval time: 9.4327 seconds
- dense: 30
- sparse: 30
- fusion: 20
- reranked: 5
- expanded: 10
- compact_evidence_characters: 9308

### Top reranked chunks

1. `23501.docx` | Section 6.3.3.3 Selection of an UPF for a particular PDU Session | score 0.025448
2. `23501.docx` | Section 5.13 Support for Edge Computing | score 0.016393
3. `33501-k30.docx` | Section 6.1.1.2 EAP framework | score 0.016393
4. `23502-k30.docx` | Section 5.2.26 UPF Services | score 0.016129
5. `None` | Section None None | score 0.016129

## Q4: Explain the UE registration procedure in 5G.

- Processed query: `Explain the UE registration procedure in 5G.`
- Requested specification: `None`
- Retrieval time: 11.9135 seconds
- dense: 30
- sparse: 30
- fusion: 20
- reranked: 5
- expanded: 10
- compact_evidence_characters: 8794

### Top reranked chunks

1. `24501-k10_3_Main-Body_s0505_s0506.docx` | Section 5.5.1.2.1 General | score 0.031319
2. `24501-k10_3_Main-Body_s0505_s0506.docx` | Section 5.5.1.2.2 Initial registration initiation | score 0.029199
3. `23502-k30.docx` | Section 4.11.2.3 EPS to 5GS Mobility | score 0.027242
4. `23501.docx` | Section 5.3.2.4 Support of a UE registered over both 3GPP and Non-3GPP access | score 0.025063
5. `24501-k10_3_Main-Body_s0505_s0506.docx` | Section 5.5.1.3.7 Abnormal cases in the UE | score 0.024548

## Q5: Explain the PDU session establishment procedure.

- Processed query: `Explain the PDU session establishment procedure.`
- Requested specification: `None`
- Retrieval time: 3.9109 seconds
- dense: 30
- sparse: 30
- fusion: 20
- reranked: 5
- expanded: 10
- compact_evidence_characters: 5300

### Top reranked chunks

1. `23502-k30.docx` | Section 4.3.2.1 General | score 0.029236
2. `None` | Section None None | score 0.024481
3. `23502-k30.docx` | Section 4.3.2 PDU Session Establishment | score 0.016393
4. `None` | Section None None | score 0.016393
5. `23502-k30.docx` | Section 4.23.5 PDU Session Management procedure | score 0.016129

