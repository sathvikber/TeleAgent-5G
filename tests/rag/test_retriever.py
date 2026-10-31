from teleagent.rag.embedding_generator import EmbeddingGenerator
from teleagent.rag.faiss_vector_store import FaissVectorStore
from teleagent.rag.retriever import Retriever
from teleagent.rag.schemas import Chunk


def create_store():

    chunks = [

        Chunk(
            id="1",
            document_id="doc",
            content="AMF handles UE registration.",
            metadata={},
        ),

        Chunk(
            id="2",
            document_id="doc",
            content="SMF manages PDU sessions.",
            metadata={},
        ),

        Chunk(
            id="3",
            document_id="doc",
            content="UPF forwards user packets.",
            metadata={},
        ),
    ]

    generator = EmbeddingGenerator()

    embeddings = generator.embed_chunks(chunks)

    store = FaissVectorStore(
        generator.embedding_dimension
    )

    store.add(chunks, embeddings)

    return generator, store


def test_retrieve():

    generator, store = create_store()

    retriever = Retriever(
        generator,
        store,
    )

    context = retriever.retrieve(
        "What is the role of AMF?",
        top_k=2,
    )

    assert context.query == "What is the role of AMF?"

    assert len(context.results) == 2

    assert "AMF" in context.results[0].chunk.content


def test_top_k():

    generator, store = create_store()

    retriever = Retriever(
        generator,
        store,
    )

    context = retriever.retrieve(
        "5G Core",
        top_k=1,
    )

    assert len(context.results) == 1


def test_empty_store():

    generator = EmbeddingGenerator()

    store = FaissVectorStore(
        generator.embedding_dimension
    )

    retriever = Retriever(
        generator,
        store,
    )

    context = retriever.retrieve(
        "AMF"
    )

    assert context.results == []


def test_result_type():

    generator, store = create_store()

    retriever = Retriever(
        generator,
        store,
    )

    context = retriever.retrieve(
        "SMF"
    )

    assert hasattr(context, "results")

    assert hasattr(context, "query")


def test_scores_sorted():

    generator, store = create_store()

    retriever = Retriever(
        generator,
        store,
    )

    context = retriever.retrieve(
        "AMF",
        top_k=3,
    )

    scores = [
        r.score
        for r in context.results
    ]

    assert scores == sorted(
        scores,
        reverse=True,
    )
