from pathlib import Path
import zipfile
import xml.etree.ElementTree as ET


DOCX = Path(
    "/workspace/teleagent/data/raw_documents/23501.docx"
)

W_NS = (
    "http://schemas.openxmlformats.org/"
    "wordprocessingml/2006/main"
)

NS = {"w": W_NS}


with zipfile.ZipFile(DOCX, "r") as archive:

    xml_data = archive.read(
        "word/document.xml"
    )


root = ET.fromstring(xml_data)


print("=" * 100)
print("DOCX PARAGRAPH / NUMBERING DIAGNOSTIC")
print("=" * 100)


count = 0

for paragraph in root.findall(".//w:p", NS):

    text_parts = []

    for node in paragraph.findall(
        ".//w:t",
        NS,
    ):
        if node.text:
            text_parts.append(node.text)

    text = "".join(text_parts).strip()

    if not text:
        continue

    p_pr = paragraph.find(
        "w:pPr",
        NS,
    )

    style = None
    num_id = None
    level = None

    if p_pr is not None:

        style_node = p_pr.find(
            "w:pStyle",
            NS,
        )

        if style_node is not None:
            style = style_node.get(
                f"{{{W_NS}}}val"
            )

        num_pr = p_pr.find(
            "w:numPr",
            NS,
        )

        if num_pr is not None:

            num_id_node = num_pr.find(
                "w:numId",
                NS,
            )

            level_node = num_pr.find(
                "w:ilvl",
                NS,
            )

            if num_id_node is not None:
                num_id = num_id_node.get(
                    f"{{{W_NS}}}val"
                )

            if level_node is not None:
                level = level_node.get(
                    f"{{{W_NS}}}val"
                )

    # Show paragraphs that look useful for heading analysis.
    if (
        style is not None
        or num_id is not None
    ):

        print("-" * 100)
        print("STYLE :", style)
        print("NUM ID:", num_id)
        print("LEVEL :", level)
        print("TEXT  :", text[:300])

        count += 1

        if count >= 100:
            break


print()
print("Paragraphs displayed:", count)
