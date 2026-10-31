import json
import tempfile
from pathlib import Path

from teleagent.rag.faiss_vector_store import FaissVectorStore
from teleagent.rag.knowledge_base_builder import (
    KnowledgeBaseBuilder,
)


def create_documents(directory):

    Path(directory, "amf.txt").write_text(
        "AMF handles UE registration."
    )

    Path(directory, "smf.txt").write_text(
        "SMF manages PDU sessions."
    )


def test_build():

    with tempfile.TemporaryDirectory() as docs:

        with tempfile.TemporaryDirectory() as kb:

            create_documents(docs)

            builder = KnowledgeBaseBuilder()

            store = builder.build(
                docs,
                kb,
            )

            assert store.size > 0


def test_files_created():

    with tempfile.TemporaryDirectory() as docs:

        with tempfile.TemporaryDirectory() as kb:

            create_documents(docs)

            builder = KnowledgeBaseBuilder()

            builder.build(
                docs,
                kb,
            )

            assert Path(
                kb,
                "index.faiss",
            ).exists()

            assert Path(
                kb,
                "metadata.pkl",
            ).exists()

            assert Path(
                kb,
                "info.json",
            ).exists()


def test_reload():

    with tempfile.TemporaryDirectory() as docs:

        with tempfile.TemporaryDirectory() as kb:

            create_documents(docs)

            builder = KnowledgeBaseBuilder()

            builder.build(
                docs,
                kb,
            )

            store = FaissVectorStore.load(
                kb
            )

            assert store.size > 0


def test_metadata():

    with tempfile.TemporaryDirectory() as docs:

        with tempfile.TemporaryDirectory() as kb:

            create_documents(docs)

            builder = KnowledgeBaseBuilder()

            builder.build(
                docs,
                kb,
            )

            with open(
                Path(
                    kb,
                    "info.json",
                )
            ) as f:

                metadata = json.load(f)

            assert metadata["documents"] == 2

            assert metadata["chunks"] > 0

            assert metadata["vector_store"] == "FAISS"
