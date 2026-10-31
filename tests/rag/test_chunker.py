from teleagent.rag.chunker import DataChunker
from teleagent.rag.schemas import Document


def test_single_small_document():

    document = Document(
        id="doc1",
        source="sample.txt",
        content="Hello World",
        metadata={}
    )

    chunker = DataChunker()

    chunks = chunker.chunk_document(document)

    assert len(chunks) == 1

    assert chunks[0].content == "Hello World"

    assert chunks[0].document_id == "doc1"


def test_large_document():

    text = "A" * 2500

    document = Document(
        id="doc1",
        source="sample.txt",
        content=text,
        metadata={}
    )

    chunker = DataChunker(
        chunk_size=1000,
        chunk_overlap=200,
    )

    chunks = chunker.chunk_document(document)

    assert len(chunks) == 3

    assert len(chunks[0].content) == 1000
    assert len(chunks[1].content) == 1000
    assert len(chunks[2].content) == 900


def test_overlap():

    text = "A" * 1500

    document = Document(
        id="doc1",
        source="sample.txt",
        content=text,
        metadata={}
    )

    chunker = DataChunker(
        chunk_size=1000,
        chunk_overlap=100,
    )

    chunks = chunker.chunk_document(document)

    assert chunks[0].content[-100:] == chunks[1].content[:100]


def test_chunk_ids():

    text = "A" * 2500

    document = Document(
        id="mydoc",
        source="sample.txt",
        content=text,
        metadata={}
    )

    chunker = DataChunker()

    chunks = chunker.chunk_document(document)

    assert chunks[0].id == "mydoc_chunk_0"
    assert chunks[1].id == "mydoc_chunk_1"
    assert chunks[2].id == "mydoc_chunk_2"


def test_empty_document():

    document = Document(
        id="doc1",
        source="sample.txt",
        content="",
        metadata={}
    )

    chunker = DataChunker()

    chunks = chunker.chunk_document(document)

    assert chunks == []


def test_multiple_documents():

    docs = [
        Document(
            id="doc1",
            source="a.txt",
            content="Hello",
            metadata={}
        ),
        Document(
            id="doc2",
            source="b.txt",
            content="World",
            metadata={}
        ),
    ]

    chunker = DataChunker()

    chunks = chunker.chunk_documents(docs)

    assert len(chunks) == 2

    assert chunks[0].document_id == "doc1"
    assert chunks[1].document_id == "doc2"


def test_invalid_chunk_size():

    try:
        DataChunker(chunk_size=0)
        assert False
    except ValueError:
        assert True


def test_invalid_overlap():

    try:
        DataChunker(
            chunk_size=100,
            chunk_overlap=100,
        )
        assert False
    except ValueError:
        assert True
