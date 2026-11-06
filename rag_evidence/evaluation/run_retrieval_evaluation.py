import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path("/home/st/teleagent")
KB_PATH = PROJECT_ROOT / "teleagent/data/knowledge_base"
OUTPUT_DIR = PROJECT_ROOT / "rag_evidence/evaluation"

sys.path.insert(0, str(PROJECT_ROOT))

from teleagent.rag.rag_service import RAGService
from teleagent.retrieval.intelligent_retrieval_service import (
    IntelligentRetrievalService,
)

QUESTIONS = [
    "What is the role of the AMF in the 5G System?",
    "What is the role of the SMF in the 5G Core?",
    "What is the role of the UPF in the 5G Core?",
    "Explain the UE registration procedure in 5G.",
    "Explain the PDU session establishment procedure.",
]

def result_record(result, rank):
    chunk = result.chunk
    metadata = chunk.metadata or {}

    return {
        "rank": rank,
        "chunk_id": chunk.id,
        "document_id": chunk.document_id,
        "score": float(result.score),
        "source": metadata.get("source"),
        "specification_number": metadata.get("specification_number"),
        "section_number": metadata.get("section_number"),
        "section_title": metadata.get("section_title"),
        "parent_section": metadata.get("parent_section"),
        "content_characters": len(chunk.content or ""),
        "content_preview": (chunk.content or "")[:500],
    }

def main():
    if not (KB_PATH / "info.json").exists():
        raise FileNotFoundError(f"Knowledge base not found: {KB_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading existing knowledge base...")
    rag = RAGService(str(KB_PATH))
    service = IntelligentRetrievalService(rag_service=rag)

    print("KB loaded:", rag.stats)

    all_runs = []

    for number, query in enumerate(QUESTIONS, start=1):
        print(f"\n[{number}/{len(QUESTIONS)}] {query}")
        started = time.perf_counter()

        processed_query = service._query_processor.process(query)
        requested_spec = service._extract_requested_specification(query)

        dense = service._dense_retriever.retrieve(
            processed_query, top_k=30
        )
        sparse = service._sparse_retriever.retrieve(
            processed_query, top_k=30
        )

        if requested_spec:
            filtered_dense = service._filter_by_specification(
                dense, requested_spec
            )
            filtered_sparse = service._filter_by_specification(
                sparse, requested_spec
            )
            if filtered_dense or filtered_sparse:
                dense = filtered_dense
                sparse = filtered_sparse

        fused = service._fusion.fuse(
            dense_results=dense,
            sparse_results=sparse,
            top_k=20,
        )

        reranked = service._reranker.rerank(
            query=processed_query,
            results=fused,
            top_k=5,
        )

        expanded = service._expand_results(
            reranked,
            requested_specification=requested_spec,
            max_results=10,
        )

        compact_evidence = service._build_compact_context(
            query=query,
            results=expanded,
            requested_specification=requested_spec,
        )

        elapsed = time.perf_counter() - started

        record = {
            "question_id": number,
            "query": query,
            "processed_query": processed_query,
            "requested_specification": requested_spec,
            "retrieval_time_seconds": round(elapsed, 4),
            "counts": {
                "dense": len(dense),
                "sparse": len(sparse),
                "fusion": len(fused),
                "reranked": len(reranked),
                "expanded": len(expanded),
                "compact_evidence_characters": len(compact_evidence),
            },
            "dense_results": [
                result_record(r, i)
                for i, r in enumerate(dense[:10], start=1)
            ],
            "sparse_results": [
                result_record(r, i)
                for i, r in enumerate(sparse[:10], start=1)
            ],
            "fusion_results": [
                result_record(r, i)
                for i, r in enumerate(fused[:10], start=1)
            ],
            "reranked_results": [
                result_record(r, i)
                for i, r in enumerate(reranked, start=1)
            ],
            "expanded_results": [
                result_record(r, i)
                for i, r in enumerate(expanded, start=1)
            ],
            "compact_evidence": compact_evidence,
        }

        all_runs.append(record)
        print("  Dense:", len(dense),
              "| Sparse:", len(sparse),
              "| Fused:", len(fused),
              "| Reranked:", len(reranked),
              "| Expanded:", len(expanded))
        print("  Time:", round(elapsed, 2), "seconds")

    output_file = OUTPUT_DIR / "retrieval_evaluation_results.json"
    with output_file.open("w", encoding="utf-8") as f:
        json.dump(all_runs, f, indent=2, ensure_ascii=False)

    summary_file = OUTPUT_DIR / "retrieval_evaluation_summary.md"
    with summary_file.open("w", encoding="utf-8") as f:
        f.write("# Retrieval Evaluation Results\n\n")
        f.write("This run records retrieval outputs from the existing TeleAgent pipeline.\n\n")
        f.write(f"- Questions evaluated: {len(all_runs)}\n")
        f.write("- Dense retrieval: existing FAISS retriever\n")
        f.write("- Sparse retrieval: existing BM25 retriever\n")
        f.write("- Fusion: existing Reciprocal Rank Fusion component\n")
        f.write("- Reranking: existing score-based reranker\n")
        f.write("- Expanded evidence: existing hierarchical expansion\n")
        f.write("- Ground-truth relevance labels: not yet assigned\n")
        f.write("- Precision, recall, and MRR: not calculated\n\n")

        for run in all_runs:
            f.write(f"## Q{run['question_id']}: {run['query']}\n\n")
            f.write(f"- Processed query: `{run['processed_query']}`\n")
            f.write(f"- Requested specification: `{run['requested_specification']}`\n")
            f.write(f"- Retrieval time: {run['retrieval_time_seconds']} seconds\n")
            for stage, count in run["counts"].items():
                f.write(f"- {stage}: {count}\n")
            f.write("\n### Top reranked chunks\n\n")
            for item in run["reranked_results"]:
                f.write(
                    f"{item['rank']}. `{item['source']}` | "
                    f"Section {item['section_number']} "
                    f"{item['section_title']} | "
                    f"score {item['score']:.6f}\n"
                )
            f.write("\n")

    print("\nEvaluation completed.")
    print("JSON:", output_file)
    print("Summary:", summary_file)

if __name__ == "__main__":
    main()
