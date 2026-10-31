from teleagent.rag.chunker import DataChunker
from teleagent.rag.schemas import Document


# =========================================================
# Telecom document
# =========================================================

telecom_text = """
3GPP TS 23.502 V18.3.0
Release 18

4 Registration Procedures

This clause describes UE registration.

4.1 General

The UE sends NAS signalling to the AMF during Registration.

4.2 Initial Registration

The UE sends an NGAP message through the gNB to the AMF.

4.2.1 Authentication

The AMF communicates with the AUSF and UDM during authentication.

4.3 Session Management

The SMF controls the UPF using PFCP during PDU Session Establishment.
"""


telecom_document = Document(
    id="doc_001",
    source="TS_23.502.pdf",
    content=telecom_text,
    metadata={
        "category": "telecom_test",
    },
)


# =========================================================
# DataChunker
# =========================================================

chunker = DataChunker(
    chunk_size=1000,
    chunk_overlap=200,
)


telecom_chunks = chunker.chunk_document(
    telecom_document
)


# =========================================================
# Print telecom chunks
# =========================================================

print()
print("=" * 80)
print("TELECOM-AWARE DATACHUNKER INTEGRATION TEST")
print("=" * 80)

print(
    "Number of telecom chunks:",
    len(telecom_chunks),
)

print()


for chunk in telecom_chunks:

    print("-" * 80)

    print("Chunk ID       :", chunk.id)
    print("Document ID    :", chunk.document_id)

    print(
        "Telecom Aware  :",
        chunk.metadata.get("telecom_aware"),
    )

    print(
        "Source         :",
        chunk.metadata.get("source"),
    )

    print(
        "Document Type  :",
        chunk.metadata.get("document_type"),
    )

    print(
        "Specification  :",
        chunk.metadata.get(
            "specification_number"
        ),
    )

    print(
        "Release        :",
        chunk.metadata.get("release"),
    )

    print(
        "Version        :",
        chunk.metadata.get("version"),
    )

    print(
        "Section        :",
        chunk.metadata.get("section_number"),
    )

    print(
        "Title          :",
        chunk.metadata.get("section_title"),
    )

    print(
        "Parent         :",
        chunk.metadata.get("parent_section"),
    )

    print(
        "Network Funcs  :",
        chunk.metadata.get("network_functions"),
    )

    print(
        "Protocols      :",
        chunk.metadata.get("protocols"),
    )

    print(
        "Procedures     :",
        chunk.metadata.get("procedures"),
    )

    print(
        "Original Meta  :",
        chunk.metadata.get("category"),
    )

    print("Content:")
    print(chunk.content)


print("-" * 80)


# =========================================================
# Locate authentication chunk
# =========================================================

authentication_chunk = None


for chunk in telecom_chunks:

    if (
        chunk.metadata.get("section_number")
        == "4.2.1"
    ):
        authentication_chunk = chunk
        break


# =========================================================
# Telecom assertions
# =========================================================

assert authentication_chunk is not None

assert (
    authentication_chunk.document_id
    == "doc_001"
)

assert (
    authentication_chunk.metadata[
        "telecom_aware"
    ]
    is True
)

assert (
    authentication_chunk.metadata[
        "specification_number"
    ]
    == "23.502"
)

assert (
    authentication_chunk.metadata["release"]
    == "18"
)

assert (
    authentication_chunk.metadata[
        "section_number"
    ]
    == "4.2.1"
)

assert (
    authentication_chunk.metadata[
        "parent_section"
    ]
    == "4.2"
)

assert (
    "AMF"
    in authentication_chunk.metadata[
        "network_functions"
    ]
)

assert (
    "AUSF"
    in authentication_chunk.metadata[
        "network_functions"
    ]
)

assert (
    "UDM"
    in authentication_chunk.metadata[
        "network_functions"
    ]
)

assert (
    "Authentication"
    in authentication_chunk.metadata[
        "procedures"
    ]
)

# Existing metadata must survive.

assert (
    authentication_chunk.metadata["category"]
    == "telecom_test"
)


# =========================================================
# Generic document backward-compatibility test
# =========================================================

generic_text = """
This is an ordinary document.

It does not contain a 3GPP specification and should continue
using the original DataChunker behavior.

The existing TeleAgent generic RAG ingestion path must remain
backward compatible.
"""


generic_document = Document(
    id="generic_001",
    source="ordinary_document.txt",
    content=generic_text,
    metadata={
        "category": "generic_test",
    },
)


generic_chunks = chunker.chunk_document(
    generic_document
)


print()
print("=" * 80)
print("GENERIC DOCUMENT BACKWARD COMPATIBILITY TEST")
print("=" * 80)


for chunk in generic_chunks:

    print("-" * 80)

    print("Chunk ID      :", chunk.id)
    print("Document ID   :", chunk.document_id)
    print("Metadata      :", chunk.metadata)

    print("Content:")
    print(chunk.content)


print("-" * 80)


# =========================================================
# Generic assertions
# =========================================================

assert len(generic_chunks) > 0

assert (
    generic_chunks[0].id
    == "generic_001_chunk_0"
)

assert (
    generic_chunks[0].document_id
    == "generic_001"
)

assert (
    generic_chunks[0].metadata[
        "category"
    ]
    == "generic_test"
)

assert (
    "telecom_aware"
    not in generic_chunks[0].metadata
)


# =========================================================
# Final result
# =========================================================

print()
print("=" * 80)
print(
    "ALL DATACHUNKER INTEGRATION TESTS PASSED"
)
print("=" * 80)
