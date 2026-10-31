from teleagent.rag.document_loader import DocumentLoader
from teleagent.rag.telecom_metadata import TelecomMetadataExtractor
from teleagent.rag.telecom_parser import TelecomSectionParser


INPUT_DIR = "/workspace/teleagent/data/raw_documents"


loader = DocumentLoader()
extractor = TelecomMetadataExtractor()
parser = TelecomSectionParser()

documents = loader.load_directory(INPUT_DIR)


print("=" * 90)
print("TELECOM METADATA / SECTION PARSER DIAGNOSTIC")
print("=" * 90)
print()


for document in documents:

    metadata = extractor.extract(
        document.content,
        source=document.source,
    )

    sections = parser.parse(
        document.content
    )

    print("=" * 90)
    print("Source        :", document.source)
    print("Document Type :", metadata.document_type)
    print("Specification :", metadata.specification_number)
    print("Release       :", metadata.release)
    print("Version       :", metadata.version)
    print("Sections      :", len(sections))

    suspicious = []

    for section in sections:

        number = section.section_number

        if number is None:
            continue

        parts = number.split(".")

        # Suspicious large top-level numbers:
        # 100, 1354, 2000, etc.
        if (
            len(parts) == 1
            and parts[0].isdigit()
            and int(parts[0]) > 99
        ):
            suspicious.append(section)
            continue

        # Something like 23.501 is usually a specification
        # identifier rather than a clause number.
        if (
            metadata.specification_number
            and number
            == metadata.specification_number
        ):
            suspicious.append(section)
            continue

        # Telecom section components normally do not look like
        # 5.1234 or 23.501.
        if any(
            len(part) > 2
            for part in parts[1:]
        ):
            suspicious.append(section)

    print(
        "Suspicious sections:",
        len(suspicious),
    )

    if suspicious:

        print()
        print("Sample suspicious headings:")

        for section in suspicious[:15]:

            print(
                "  ",
                section.section_number,
                "|",
                section.section_title,
            )

    print()
