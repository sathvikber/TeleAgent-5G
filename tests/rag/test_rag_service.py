import tempfile

from teleagent.rag.knowledge_base_builder import (
    KnowledgeBaseBuilder,
)
from teleagent.rag.rag_service import (
    RAGService,
)


def create_documents(directory):

    with open(
        f"{directory}/amf.txt",
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            """
AMF handles UE registration,
mobility management,
authentication,
handover,
and connection management.
"""
        )

    with open(
        f"{directory}/smf.txt",
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            """
SMF manages PDU sessions,
IP allocation,
and session establishment.
"""
        )


def test_retrieve():

    with tempfile.TemporaryDirectory() as docs:

        with tempfile.TemporaryDirectory() as kb:

            create_documents(docs)

            builder = KnowledgeBaseBuilder()

            builder.build(
                docs,
                kb,
            )

            rag = RAGService(kb)

            context = rag.retrieve(
                "What is AMF?"
            )

            assert len(context.results) > 0


def test_retrieve_text():

    with tempfile.TemporaryDirectory() as docs:

        with tempfile.TemporaryDirectory() as kb:

            create_documents(docs)

            builder = KnowledgeBaseBuilder()

            builder.build(
                docs,
                kb,
            )

            rag = RAGService(kb)

            text = rag.retrieve_text(
                "AMF"
            )

            assert "AMF" in text


def test_statistics():

    with tempfile.TemporaryDirectory() as docs:

        with tempfile.TemporaryDirectory() as kb:

            create_documents(docs)

            builder = KnowledgeBaseBuilder()

            builder.build(
                docs,
                kb,
            )

            rag = RAGService(kb)

            stats = rag.stats

            assert stats["documents"] == 2
            assert stats["vector_store"] == "FAISS"
            assert "embedding_dimension" in stats


def test_empty_query():

    with tempfile.TemporaryDirectory() as docs:

        with tempfile.TemporaryDirectory() as kb:

            create_documents(docs)

            builder = KnowledgeBaseBuilder()

            builder.build(
                docs,
                kb,
            )

            rag = RAGService(kb)

            context = rag.retrieve(
                "",
                top_k=2,
            )

            assert isinstance(
                context.results,
                list,
            )


def test_top_k():

    with tempfile.TemporaryDirectory() as docs:

        with tempfile.TemporaryDirectory() as kb:

            create_documents(docs)

            builder = KnowledgeBaseBuilder()

            builder.build(
                docs,
                kb,
            )

            rag = RAGService(kb)

            context = rag.retrieve(
                "5G",
                top_k=1,
            )

            assert len(context.results) <= 1
