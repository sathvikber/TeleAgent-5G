"""
Manual integration test for telecom-aware ingestion + FAISS.

This test verifies:

Document
    ->
DataChunker
    ->
Telecom-aware Chunk metadata
    ->
FAISS
    ->
Retriever
    ->
SearchResult
    ->
metadata preserved

The real SentenceTransformer model is deliberately NOT used.
A tiny deterministic embedding generator is used so that this
test does not require network access or model downloads.
"""

from __future__ import annotations

import shutil
import tempfile

from teleagent.rag.chunker import DataChunker
from teleagent.rag.faiss_vector_store import FaissVectorStore
from teleagent.rag.retriever import Retriever
from teleagent.rag.schemas import (
    Chunk,
    Document,
    Embedding,
)


# ============================================================
# Fake embedding generator
# ============================================================

class FakeEmbeddingGenerator:
    """
    Tiny deterministic embedding generator used only for testing.

    Authentication-related text is mapped near:
        [1, 0, 0]

    Session-related text is mapped near:
        [0, 1, 0]

    Other text is mapped near:
        [0, 0, 1]
    """

    @property
    def embedding_dimension(self) -> int:
        return 3

    def embed_chunk(
        self,
        chunk: Chunk,
    ) -> Embedding:

        text = chunk.content.lower()

        if (
            "authentication" in text
            or "ausf" in text
        ):
            vector = [1.0, 0.0, 0.0]

        elif (
            "session" in text
            or "smf" in text
            or "upf" in text
        ):
            vector = [0.0, 1.0, 0.0]

        else:
            vector = [0.0, 0.0, 1.0]

        return Embedding(
            chunk_id=chunk.id,
            vector=vector,
        )

    def embed_chunks(
        self,
        chunks: list[Chunk],
    ) -> list[Embedding]:

        return [
            self.embed_chunk(chunk)
            for chunk in chunks
        ]


# ============================================================
# Telecom document
# ============================================================

telecom_text = """
3GPP TS 23.502 V18.3.0
Release 18

4 Registration Procedures

This clause describes UE registration.

4.1 General

The UE sends NAS signalling to the AMF during Registration.

4.2 Initial Registration

The UE sends an NGAP message through the gNB to the AMF.

4.2.1 Authentication

The AMF communicates with the AUSF and UDM during authentication.

4.3 Session Management

The SMF controls the UPF using PFCP during PDU Session Establishment.
"""


document = Document(
    id="doc_001",
    source="TS_23.502.pdf",
    content=telecom_text,
    metadata={
        "category": "telecom_test",
    },
)


# ============================================================
# 1. Telecom-aware chunking
# ============================================================

print()
print("=" * 80)
print("STEP 1 - TELECOM-AWARE CHUNKING")
print("=" * 80)


chunker = DataChunker(
    chunk_size=1000,
    chunk_overlap=200,
)


chunks = chunker.chunk_document(
    document
)


print("Chunks generated:", len(chunks))


assert len(chunks) > 0


for chunk in chunks:

    print(
        chunk.id,
        "| section:",
        chunk.metadata.get("section_number"),
    )


# ============================================================
# Find authentication chunk
# ============================================================

authentication_chunk = None


for chunk in chunks:

    if (
        chunk.metadata.get("section_number")
        == "4.2.1"
    ):
        authentication_chunk = chunk
        break


assert authentication_chunk is not None


print()
print(
    "Authentication chunk:",
    authentication_chunk.id,
)


# ============================================================
# 2. Generate test embeddings
# ============================================================

print()
print("=" * 80)
print("STEP 2 - GENERATE TEST EMBEDDINGS")
print("=" * 80)


embedding_generator = FakeEmbeddingGenerator()


embeddings = embedding_generator.embed_chunks(
    chunks
)


print(
    "Embeddings generated:",
    len(embeddings),
)


assert len(embeddings) == len(chunks)


# ============================================================
# 3. Store chunks and embeddings in FAISS
# ============================================================

print()
print("=" * 80)
print("STEP 3 - ADD TO FAISS")
print("=" * 80)


vector_store = FaissVectorStore(
    dimension=embedding_generator.embedding_dimension
)


vector_store.add(
    chunks,
    embeddings,
)


print(
    "FAISS size:",
    vector_store.size,
)


assert vector_store.size == len(chunks)


# ============================================================
# 4. Verify original metadata is inside FAISS chunk store
# ============================================================

print()
print("=" * 80)
print("STEP 4 - VERIFY STORED METADATA")
print("=" * 80)


stored_auth_chunk = vector_store._chunk_store[
    authentication_chunk.id
]


print(
    "Chunk ID       :",
    stored_auth_chunk.id,
)

print(
    "Specification  :",
    stored_auth_chunk.metadata.get(
        "specification_number"
    ),
)

print(
    "Section        :",
    stored_auth_chunk.metadata.get(
        "section_number"
    ),
)

print(
    "Network Funcs  :",
    stored_auth_chunk.metadata.get(
        "network_functions"
    ),
)

print(
    "Procedures     :",
    stored_auth_chunk.metadata.get(
        "procedures"
    ),
)


assert (
    stored_auth_chunk.metadata[
        "specification_number"
    ]
    == "23.502"
)

assert (
    stored_auth_chunk.metadata[
        "section_number"
    ]
    == "4.2.1"
)

assert (
    stored_auth_chunk.metadata[
        "parent_section"
    ]
    == "4.2"
)

assert (
    "AMF"
    in stored_auth_chunk.metadata[
        "network_functions"
    ]
)

assert (
    "AUSF"
    in stored_auth_chunk.metadata[
        "network_functions"
    ]
)

assert (
    "UDM"
    in stored_auth_chunk.metadata[
        "network_functions"
    ]
)

assert (
    "Authentication"
    in stored_auth_chunk.metadata[
        "procedures"
    ]
)


# ============================================================
# 5. Retrieve authentication context
# ============================================================

print()
print("=" * 80)
print("STEP 5 - RETRIEVER TEST")
print("=" * 80)


retriever = Retriever(
    embedding_generator=embedding_generator,
    vector_store=vector_store,
)


context = retriever.retrieve(
    "Explain authentication using AUSF",
    top_k=3,
)


print(
    "Results returned:",
    len(context.results),
)


for result in context.results:

    print("-" * 80)

    print(
        "Score:",
        result.score,
    )

    print(
        "Chunk:",
        result.chunk.id,
    )

    print(
        "Section:",
        result.chunk.metadata.get(
            "section_number"
        ),
    )

    print(
        "Title:",
        result.chunk.metadata.get(
            "section_title"
        ),
    )

    print(
        "Network Functions:",
        result.chunk.metadata.get(
            "network_functions"
        ),
    )

    print(
        "Procedures:",
        result.chunk.metadata.get(
            "procedures"
        ),
    )


assert len(context.results) > 0


top_result = context.results[0]


assert (
    top_result.chunk.metadata.get(
        "section_number"
    )
    == "4.2.1"
)

assert (
    "Authentication"
    in top_result.chunk.metadata.get(
        "procedures",
        [],
    )
)


# ============================================================
# 6. Test FAISS persistence
# ============================================================

print()
print("=" * 80)
print("STEP 6 - FAISS SAVE / LOAD TEST")
print("=" * 80)


temp_directory = tempfile.mkdtemp(
    prefix="teleagent_faiss_test_"
)


try:

    vector_store.save(
        temp_directory
    )

    loaded_store = FaissVectorStore.load(
        temp_directory
    )

    print(
        "Loaded FAISS size:",
        loaded_store.size,
    )

    assert (
        loaded_store.size
        == vector_store.size
    )

    loaded_auth_chunk = (
        loaded_store._chunk_store[
            authentication_chunk.id
        ]
    )

    print(
        "Loaded specification:",
        loaded_auth_chunk.metadata.get(
            "specification_number"
        ),
    )

    print(
        "Loaded section:",
        loaded_auth_chunk.metadata.get(
            "section_number"
        ),
    )

    print(
        "Loaded network functions:",
        loaded_auth_chunk.metadata.get(
            "network_functions"
        ),
    )

    assert (
        loaded_auth_chunk.metadata[
            "specification_number"
        ]
        == "23.502"
    )

    assert (
        loaded_auth_chunk.metadata[
            "section_number"
        ]
        == "4.2.1"
    )

    assert (
        loaded_auth_chunk.metadata[
            "parent_section"
        ]
        == "4.2"
    )

    assert (
        "AUSF"
        in loaded_auth_chunk.metadata[
            "network_functions"
        ]
    )

finally:

    shutil.rmtree(
        temp_directory,
        ignore_errors=True,
    )


# ============================================================
# Final result
# ============================================================

print()
print("=" * 80)
print(
    "ALL TELECOM FAISS INTEGRATION TESTS PASSED"
)
print("=" * 80)
