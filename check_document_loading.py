from collections import Counter

from teleagent.rag.document_loader import DocumentLoader
from teleagent.rag.chunker import DataChunker


INPUT_DIR = "/workspace/teleagent/data/raw_documents"


print("=" * 80)
print("DOCUMENT LOADING DIAGNOSTIC")
print("=" * 80)

loader = DocumentLoader()

documents = loader.load_directory(
    INPUT_DIR
)

print("Documents returned by loader:", len(documents))
print()


# ------------------------------------------------------------
# Check duplicate document IDs
# ------------------------------------------------------------

id_counts = Counter(
    document.id
    for document in documents
)

duplicate_ids = {
    doc_id: count
    for doc_id, count in id_counts.items()
    if count > 1
}


print("=" * 80)
print("DOCUMENT ID CHECK")
print("=" * 80)

print("Unique document IDs :", len(id_counts))
print("Duplicate IDs       :", len(duplicate_ids))
print()

for doc_id, count in sorted(
    duplicate_ids.items()
):
    print(
        count,
        "x",
        doc_id,
    )


# ------------------------------------------------------------
# Check duplicate sources
# ------------------------------------------------------------

source_counts = Counter(
    document.source
    for document in documents
)

duplicate_sources = {
    source: count
    for source, count in source_counts.items()
    if count > 1
}


print()
print("=" * 80)
print("DOCUMENT SOURCE CHECK")
print("=" * 80)

print("Unique sources       :", len(source_counts))
print("Duplicate sources    :", len(duplicate_sources))
print()

for source, count in sorted(
    duplicate_sources.items()
):
    print(
        count,
        "x",
        source,
    )


# ------------------------------------------------------------
# Print every loaded document
# ------------------------------------------------------------

print()
print("=" * 80)
print("ALL LOADED DOCUMENTS")
print("=" * 80)

for index, document in enumerate(
    documents,
    start=1,
):

    print()
    print("Document :", index)
    print("ID       :", document.id)
    print("Source   :", document.source)
    print("Metadata :", document.metadata)
    print("Length   :", len(document.content))


# ------------------------------------------------------------
# Check chunk IDs before FAISS
# ------------------------------------------------------------

print()
print("=" * 80)
print("CHUNK ID DIAGNOSTIC")
print("=" * 80)

chunker = DataChunker()

chunks = chunker.chunk_documents(
    documents
)

chunk_id_counts = Counter(
    chunk.id
    for chunk in chunks
)

duplicate_chunk_ids = {
    chunk_id: count
    for chunk_id, count in chunk_id_counts.items()
    if count > 1
}


print("Chunks produced      :", len(chunks))
print("Unique chunk IDs     :", len(chunk_id_counts))
print(
    "Duplicate chunk IDs  :",
    len(duplicate_chunk_ids),
)

print()
print("First 30 duplicate chunk IDs:")
print("-" * 80)

for chunk_id, count in list(
    sorted(
        duplicate_chunk_ids.items()
    )
)[:30]:

    print(
        count,
        "x",
        chunk_id,
    )
