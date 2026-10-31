from collections import Counter
from hashlib import sha1

from teleagent.rag.document_loader import DocumentLoader
from teleagent.rag.chunker import DataChunker


INPUT_DIR = "/workspace/teleagent/data/raw_documents"


print("=" * 90)
print("FINAL TELEAGENT RAG INGESTION CHECK")
print("=" * 90)


# ------------------------------------------------------------
# Load documents
# ------------------------------------------------------------

loader = DocumentLoader()

documents = loader.load_directory(
    INPUT_DIR
)


print()
print("=" * 90)
print("DOCUMENTS")
print("=" * 90)

print("Documents loaded :", len(documents))
print(
    "Unique IDs       :",
    len({document.id for document in documents}),
)


# ------------------------------------------------------------
# Check exact duplicate documents
# ------------------------------------------------------------

document_hashes = {}

for document in documents:

    normalized = " ".join(
        document.content.split()
    )

    digest = sha1(
        normalized.encode(
            "utf-8",
            errors="ignore",
        )
    ).hexdigest()

    document_hashes.setdefault(
        digest,
        [],
    ).append(
        document.source
    )


duplicate_documents = {
    digest: sources
    for digest, sources in document_hashes.items()
    if len(sources) > 1
}


print(
    "Exact duplicate documents :",
    len(duplicate_documents),
)

for sources in duplicate_documents.values():

    print(
        "  DUPLICATE:",
        sources,
    )


# ------------------------------------------------------------
# Generate actual chunks used by TeleAgent
# ------------------------------------------------------------

chunker = DataChunker()

chunks = chunker.chunk_documents(
    documents
)


print()
print("=" * 90)
print("CHUNKS")
print("=" * 90)

chunk_ids = [
    chunk.id
    for chunk in chunks
]

unique_chunk_ids = set(
    chunk_ids
)


print(
    "Total chunks      :",
    len(chunks),
)

print(
    "Unique chunk IDs  :",
    len(unique_chunk_ids),
)

print(
    "Duplicate IDs     :",
    len(chunk_ids)
    - len(unique_chunk_ids),
)


# ------------------------------------------------------------
# Telecom-aware vs generic
# ------------------------------------------------------------

telecom_chunks = [
    chunk
    for chunk in chunks
    if chunk.metadata.get(
        "telecom_aware"
    ) is True
]

generic_chunks = [
    chunk
    for chunk in chunks
    if chunk.metadata.get(
        "telecom_aware"
    ) is not True
]


print()
print("=" * 90)
print("CHUNK TYPE")
print("=" * 90)

print(
    "Telecom-aware chunks :",
    len(telecom_chunks),
)

print(
    "Generic chunks       :",
    len(generic_chunks),
)


# ------------------------------------------------------------
# Specification distribution
# ------------------------------------------------------------

spec_counts = Counter(
    chunk.metadata.get(
        "specification_number"
    )
    for chunk in telecom_chunks
    if chunk.metadata.get(
        "specification_number"
    )
)


print()
print("=" * 90)
print("3GPP SPECIFICATION DISTRIBUTION")
print("=" * 90)

for spec, count in sorted(
    spec_counts.items()
):

    print(
        f"{spec:10s} : {count}"
    )


# ------------------------------------------------------------
# Source distribution
# ------------------------------------------------------------

source_counts = Counter(
    chunk.metadata.get(
        "source"
    )
    or chunk.metadata.get(
        "path"
    )
    or "UNKNOWN"
    for chunk in chunks
)


print()
print("=" * 90)
print("CHUNKS BY SOURCE")
print("=" * 90)

for source, count in sorted(
    source_counts.items()
):

    print(
        f"{count:6d} : {source}"
    )


# ------------------------------------------------------------
# Validate required specifications
# ------------------------------------------------------------

required_specs = {
    "23.501",
    "23.502",
    "24.501",
    "29.509",
    "29.518",
    "33.501",
}

detected_specs = set(
    spec_counts.keys()
)

missing_specs = (
    required_specs
    - detected_specs
)


print()
print("=" * 90)
print("REQUIRED SPECIFICATION CHECK")
print("=" * 90)

print(
    "Required :",
    sorted(required_specs),
)

print(
    "Detected :",
    sorted(detected_specs),
)

print(
    "Missing  :",
    sorted(missing_specs),
)


# ------------------------------------------------------------
# Check Y.3172 misclassification
# ------------------------------------------------------------

y3172_chunks = [
    chunk
    for chunk in chunks
    if "Y.3172"
    in (
        chunk.metadata.get(
            "source"
        )
        or ""
    )
]


print()
print("=" * 90)
print("Y.3172 CHECK")
print("=" * 90)

print(
    "Y.3172 chunks :",
    len(y3172_chunks),
)

wrong_y3172 = [
    chunk
    for chunk in y3172_chunks
    if chunk.metadata.get(
        "specification_number"
    )
]


print(
    "Y.3172 chunks incorrectly carrying "
    "3GPP specification metadata :",
    len(wrong_y3172),
)


# ------------------------------------------------------------
# Sample metadata
# ------------------------------------------------------------

print()
print("=" * 90)
print("SAMPLE TELECOM CHUNKS")
print("=" * 90)

for chunk in telecom_chunks[:10]:

    print()
    print(
        "ID      :",
        chunk.id,
    )

    print(
        "Source  :",
        chunk.metadata.get(
            "source"
        ),
    )

    print(
        "Spec    :",
        chunk.metadata.get(
            "specification_number"
        ),
    )

    print(
        "Section :",
        chunk.metadata.get(
            "section_number"
        ),
    )

    print(
        "Title   :",
        chunk.metadata.get(
            "section_title"
        ),
    )


# ------------------------------------------------------------
# Final result
# ------------------------------------------------------------

print()
print("=" * 90)
print("FINAL RESULT")
print("=" * 90)


problems = []


if len(chunk_ids) != len(unique_chunk_ids):

    problems.append(
        "duplicate chunk IDs detected"
    )


if missing_specs:

    problems.append(
        "required 3GPP specifications missing"
    )


if wrong_y3172:

    problems.append(
        "Y.3172 incorrectly classified as 3GPP"
    )


if duplicate_documents:

    problems.append(
        "exact duplicate source documents detected"
    )


if problems:

    print("STATUS: FAIL")

    for problem in problems:

        print(
            " -",
            problem,
        )

else:

    print("STATUS: PASS")
    print(
        "Knowledge-base ingestion is ready "
        "for FAISS rebuild."
    )
