from teleagent.rag.telecom_ingestion import TelecomIngestionProcessor
from teleagent.rag.telecom_chunker import TelecomSectionChunker


# ---------------------------------------------------------
# Test telecom document
# ---------------------------------------------------------

text = """
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


# ---------------------------------------------------------
# STEP 1
# Telecom-aware ingestion
# ---------------------------------------------------------

processor = TelecomIngestionProcessor()

sections = processor.process(
    text,
    source="TS_23.502.pdf",
)

print()
print("=" * 80)
print("TELECOM INGESTION TEST")
print("=" * 80)

print("Number of sections:", len(sections))
print()


for section in sections:

    print("-" * 80)

    print("Section       :", section.section_number)
    print("Title         :", section.section_title)
    print("Parent        :", section.parent_section)

    print("Source        :", section.source)
    print("Document Type :", section.document_type)
    print("Specification :", section.specification_number)
    print("Release       :", section.release)
    print("Version       :", section.version)

    print("Network Funcs :", section.network_functions)
    print("Protocols     :", section.protocols)
    print("Procedures    :", section.procedures)

    print("Text:")
    print(section.text)

print("-" * 80)


# ---------------------------------------------------------
# STEP 2
# Normal telecom-aware chunking
# ---------------------------------------------------------

chunker = TelecomSectionChunker(
    chunk_size=500,
    chunk_overlap=75,
)

chunks = chunker.chunk_sections(sections)

print()
print("=" * 80)
print("NORMAL CHUNKING TEST")
print("=" * 80)

print("Number of sections :", len(sections))
print("Number of chunks   :", len(chunks))
print()


for chunk in chunks:

    print("-" * 80)

    print("Chunk ID       :", chunk.chunk_id)
    print("Chunk Index    :", chunk.chunk_index)

    print("Section        :", chunk.section_number)
    print("Title          :", chunk.section_title)
    print("Parent         :", chunk.parent_section)

    print("Source         :", chunk.source)
    print("Document Type  :", chunk.document_type)
    print("Specification  :", chunk.specification_number)
    print("Release        :", chunk.release)
    print("Version        :", chunk.version)

    print("Network Funcs  :", chunk.network_functions)
    print("Protocols      :", chunk.protocols)
    print("Procedures     :", chunk.procedures)

    print("Text:")
    print(chunk.text)

print("-" * 80)


# ---------------------------------------------------------
# STEP 3
# Verify deterministic chunk IDs
# ---------------------------------------------------------

print()
print("=" * 80)
print("CHUNK ID TEST")
print("=" * 80)

for chunk in chunks:
    print(
        chunk.chunk_id,
        "| section:",
        chunk.section_number,
        "| title:",
        chunk.section_title,
    )


# ---------------------------------------------------------
# STEP 4
# Force splitting of a section
#
# We deliberately use a small chunk size to verify that
# one large logical section can become multiple chunks.
# ---------------------------------------------------------

small_chunker = TelecomSectionChunker(
    chunk_size=8,
    chunk_overlap=2,
)

# Use the last section:
# 4.3 Session Management
large_section_test = sections[-1]

test_chunks = small_chunker.chunk_section(
    large_section_test
)

print()
print("=" * 80)
print("LARGE SECTION SPLITTING TEST")
print("=" * 80)

print(
    "Original section:",
    large_section_test.section_number,
    large_section_test.section_title,
)

print(
    "Chunks generated:",
    len(test_chunks),
)

print()


for chunk in test_chunks:

    print("-" * 80)

    print("Chunk ID      :", chunk.chunk_id)
    print("Chunk Index   :", chunk.chunk_index)

    print("Section       :", chunk.section_number)
    print("Title         :", chunk.section_title)
    print("Parent        :", chunk.parent_section)

    print("Network Funcs :", chunk.network_functions)
    print("Protocols     :", chunk.protocols)
    print("Procedures    :", chunk.procedures)

    print("Text:")
    print(chunk.text)

print("-" * 80)


# ---------------------------------------------------------
# STEP 5
# Basic assertions
#
# These checks will immediately fail if important metadata
# was lost during chunking.
# ---------------------------------------------------------

print()
print("=" * 80)
print("ASSERTION TESTS")
print("=" * 80)


assert len(sections) > 0
assert len(chunks) > 0

# ---------------------------------------------------------
# Find authentication section.
# ---------------------------------------------------------

authentication_section = None

for section in sections:

    if section.section_number == "4.2.1":
        authentication_section = section
        break


assert authentication_section is not None

assert authentication_section.parent_section == "4.2"

assert authentication_section.specification_number == "23.502"

assert authentication_section.release == "18"

assert "AMF" in authentication_section.network_functions
assert "AUSF" in authentication_section.network_functions
assert "UDM" in authentication_section.network_functions

assert "Authentication" in authentication_section.procedures


# ---------------------------------------------------------
# Find the authentication chunk.
# ---------------------------------------------------------

authentication_chunk = None

for chunk in chunks:

    if chunk.section_number == "4.2.1":
        authentication_chunk = chunk
        break


assert authentication_chunk is not None

assert authentication_chunk.parent_section == "4.2"

assert authentication_chunk.specification_number == "23.502"

assert authentication_chunk.release == "18"

assert "AMF" in authentication_chunk.network_functions
assert "AUSF" in authentication_chunk.network_functions
assert "UDM" in authentication_chunk.network_functions

assert "Authentication" in authentication_chunk.procedures


# ---------------------------------------------------------
# Verify chunk IDs.
# ---------------------------------------------------------

assert (
    authentication_chunk.chunk_id
    == "ts_23_502_4_2_1_000"
)


# ---------------------------------------------------------
# Verify split chunks stay inside the same section.
# ---------------------------------------------------------

for chunk in test_chunks:

    assert chunk.section_number == "4.3"

    assert chunk.parent_section == "4"

    assert chunk.specification_number == "23.502"

    assert chunk.release == "18"

    assert "SMF" in chunk.network_functions
    assert "UPF" in chunk.network_functions

    assert "PFCP" in chunk.protocols


print()
print("ALL TELECOM INGESTION AND CHUNKING TESTS PASSED")
print("=" * 80)
