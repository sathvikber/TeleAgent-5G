from teleagent.rag.embedding_generator import (
    EmbeddingGenerator,
)
from teleagent.rag.schemas import Chunk


def test_load_model():

    generator = EmbeddingGenerator()

    assert generator.embedding_dimension > 0


def test_single_chunk():

    chunk = Chunk(
        id="chunk1",
        document_id="doc1",
        content="The AMF manages UE registration.",
        metadata={},
    )

    generator = EmbeddingGenerator()

    embedding = generator.embed_chunk(chunk)

    assert embedding.chunk_id == "chunk1"

    assert len(embedding.vector) == generator.embedding_dimension


def test_multiple_chunks():

    chunks = [
        Chunk(
            id="1",
            document_id="doc",
            content="AMF handles registration.",
            metadata={},
        ),
        Chunk(
            id="2",
            document_id="doc",
            content="SMF manages sessions.",
            metadata={},
        ),
    ]

    generator = EmbeddingGenerator()

    embeddings = generator.embed_chunks(chunks)

    assert len(embeddings) == 2

    assert embeddings[0].chunk_id == "1"

    assert embeddings[1].chunk_id == "2"


def test_empty_input():

    generator = EmbeddingGenerator()

    embeddings = generator.embed_chunks([])

    assert embeddings == []


def test_embedding_dimension():

    generator = EmbeddingGenerator()

    chunk = Chunk(
        id="chunk1",
        document_id="doc1",
        content="gNodeB communicates with the AMF.",
        metadata={},
    )

    embedding = generator.embed_chunk(chunk)

    assert len(embedding.vector) == generator.embedding_dimension


def test_normalized_embedding():

    import math

    generator = EmbeddingGenerator()

    chunk = Chunk(
        id="chunk1",
        document_id="doc1",
        content="5G Core Network",
        metadata={},
    )

    embedding = generator.embed_chunk(chunk)

    norm = math.sqrt(sum(v * v for v in embedding.vector))

    assert abs(norm - 1.0) < 1e-5
