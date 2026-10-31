from teleagent.rag.schemas import (
    Document,
    Chunk,
    Embedding,
    SearchResult,
    RetrievedContext,
    KnowledgeBaseStats,
)


def test_document_schema():

    doc = Document(
        id="doc1",
        source="sample.pdf",
        content="This is a telecom document.",
        metadata={"page": 1},
    )

    assert doc.id == "doc1"
    assert doc.source == "sample.pdf"
    assert doc.content.startswith("This")
    assert doc.metadata["page"] == 1


def test_chunk_schema():

    chunk = Chunk(
        id="chunk1",
        document_id="doc1",
        content="This is one chunk.",
        metadata={"page": 1},
    )

    assert chunk.id == "chunk1"
    assert chunk.document_id == "doc1"
    assert chunk.content == "This is one chunk."
    assert chunk.metadata["page"] == 1


def test_embedding_schema():

    emb = Embedding(
        chunk_id="chunk1",
        vector=[0.1, 0.2, 0.3],
    )

    assert emb.chunk_id == "chunk1"
    assert len(emb.vector) == 3
    assert emb.vector[0] == 0.1


def test_search_result_schema():

    chunk = Chunk(
        id="chunk1",
        document_id="doc1",
        content="Hello",
        metadata={},
    )

    result = SearchResult(
        chunk=chunk,
        score=0.92,
    )

    assert result.chunk.id == "chunk1"
    assert result.score == 0.92


def test_retrieved_context_schema():

    chunk = Chunk(
        id="chunk1",
        document_id="doc1",
        content="Example",
        metadata={},
    )

    result = SearchResult(
        chunk=chunk,
        score=0.95,
    )

    context = RetrievedContext(
        query="What is AMF?",
        results=[result],
    )

    assert context.query == "What is AMF?"
    assert len(context.results) == 1
    assert context.results[0].score == 0.95


def test_kb_stats_schema():

    stats = KnowledgeBaseStats(
        documents=10,
        chunks=125,
        embedding_dimension=384,
    )

    assert stats.documents == 10
    assert stats.chunks == 125
    assert stats.embedding_dimension == 384


def test_default_metadata():

    doc = Document(
        id="doc1",
        source="a.txt",
        content="Hello",
    )

    assert doc.metadata == {}


def test_chunk_default_metadata():

    chunk = Chunk(
        id="chunk1",
        document_id="doc1",
        content="Hello",
    )

    assert chunk.metadata == {}
