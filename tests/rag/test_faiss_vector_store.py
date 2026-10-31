import tempfile

from teleagent.rag.embedding_generator import EmbeddingGenerator
from teleagent.rag.faiss_vector_store import FaissVectorStore
from teleagent.rag.schemas import Chunk


def create_chunks():

    return [
        Chunk(
            id="c1",
            document_id="doc",
            content="AMF handles registration",
            metadata={},
        ),
        Chunk(
            id="c2",
            document_id="doc",
            content="SMF manages sessions",
            metadata={},
        ),
        Chunk(
            id="c3",
            document_id="doc",
            content="UPF forwards packets",
            metadata={},
        ),
    ]


def test_create_store():

    store = FaissVectorStore(384)

    assert store.size == 0


def test_add_embeddings():

    chunks = create_chunks()

    generator = EmbeddingGenerator()

    embeddings = generator.embed_chunks(chunks)

    store = FaissVectorStore(
        generator.embedding_dimension
    )

    store.add(chunks, embeddings)

    assert store.size == 3


def test_search():

    chunks = create_chunks()

    generator = EmbeddingGenerator()

    embeddings = generator.embed_chunks(chunks)

    store = FaissVectorStore(
        generator.embedding_dimension
    )

    store.add(chunks, embeddings)

    query = generator.embed_chunk(
        Chunk(
            id="query",
            document_id="q",
            content="AMF registration",
            metadata={},
        )
    )

    results = store.search(
        query.vector,
        top_k=2,
    )

    assert len(results) > 0

    assert "AMF" in results[0].chunk.content


def test_save_load():

    chunks = create_chunks()

    generator = EmbeddingGenerator()

    embeddings = generator.embed_chunks(chunks)

    store = FaissVectorStore(
        generator.embedding_dimension
    )

    store.add(chunks, embeddings)

    with tempfile.TemporaryDirectory() as tmp:

        store.save(tmp)

        restored = FaissVectorStore.load(tmp)

        assert restored.size == store.size


def test_empty_search():

    store = FaissVectorStore(384)

    results = store.search([0.0] * 384)

    assert results == []


def test_mismatched_input():

    chunks = create_chunks()

    generator = EmbeddingGenerator()

    embeddings = generator.embed_chunks(chunks[:2])

    store = FaissVectorStore(
        generator.embedding_dimension
    )

    try:
        store.add(chunks, embeddings)
        assert False
    except ValueError:
        assert True
