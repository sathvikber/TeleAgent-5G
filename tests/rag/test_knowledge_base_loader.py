import json
import tempfile

from teleagent.rag.knowledge_base_builder import KnowledgeBaseBuilder
from teleagent.rag.knowledge_base_loader import KnowledgeBaseLoader


def create_documents(directory):

    with open(
        f"{directory}/doc1.txt",
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "AMF manages registration procedures."
        )

    with open(
        f"{directory}/doc2.txt",
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "UPF forwards user plane packets."
        )


def test_load():

    with tempfile.TemporaryDirectory() as docs:

        with tempfile.TemporaryDirectory() as kb:

            create_documents(docs)

            builder = KnowledgeBaseBuilder()

            builder.build(
                docs,
                kb,
            )

            loader = KnowledgeBaseLoader()

            store = loader.load(kb)

            assert store is not None


def test_metadata():

    with tempfile.TemporaryDirectory() as docs:

        with tempfile.TemporaryDirectory() as kb:

            create_documents(docs)

            builder = KnowledgeBaseBuilder()

            builder.build(
                docs,
                kb,
            )

            loader = KnowledgeBaseLoader()

            loader.load(kb)

            assert loader.document_count == 2

            assert loader.chunk_count > 0

            assert loader.embedding_dimension > 0


def test_missing_directory():

    loader = KnowledgeBaseLoader()

    try:

        loader.load("/tmp/does_not_exist")

        assert False

    except FileNotFoundError:

        assert True


def test_info_json():

    with tempfile.TemporaryDirectory() as docs:

        with tempfile.TemporaryDirectory() as kb:

            create_documents(docs)

            builder = KnowledgeBaseBuilder()

            builder.build(
                docs,
                kb,
            )

            with open(
                f"{kb}/info.json",
                "r",
                encoding="utf-8",
            ) as f:

                info = json.load(f)

            assert info["vector_store"] == "FAISS"

            loader = KnowledgeBaseLoader()

            loader.load(kb)

            assert loader.embedding_model == info["embedding_model"]
