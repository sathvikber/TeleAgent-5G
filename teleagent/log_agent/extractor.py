import os
import json

import pandas as pd
import fitz

from docx import Document


def extract_pdf(file_path):

    result = []

    document = fitz.open(file_path)

    for page_number, page in enumerate(
        document,
        start=1
    ):

        text = page.get_text()

        result.append({
            "page": page_number,
            "text": text
        })

    document.close()

    return result


def extract_docx(file_path):

    document = Document(file_path)

    paragraphs = []

    for paragraph in document.paragraphs:

        text = paragraph.text.strip()

        if text:

            paragraphs.append({
                "text": text
            })

    tables = []

    for table in document.tables:

        rows = []

        for row in table.rows:

            rows.append([
                cell.text.strip()
                for cell in row.cells
            ])

        tables.append(rows)

    return {
        "paragraphs": paragraphs,
        "tables": tables
    }


def extract_txt(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8",
        errors="ignore"
    ) as file:

        return file.read()


def extract_csv(file_path):

    dataframe = pd.read_csv(file_path)

    return {
        "columns": list(dataframe.columns),
        "records": dataframe.to_dict(
            orient="records"
        )
    }


def extract_json(file_path):

    with open(
        file_path,
        "r",
        encoding="utf-8"
    ) as file:

        return json.load(file)


def extract_xlsx(file_path):

    excel = pd.ExcelFile(file_path)

    sheets = {}

    for sheet_name in excel.sheet_names:

        dataframe = pd.read_excel(
            file_path,
            sheet_name=sheet_name
        )

        sheets[sheet_name] = {
            "columns": list(dataframe.columns),
            "records": dataframe.to_dict(
                orient="records"
            )
        }

    return sheets


def extract_file(file_path):

    extension = os.path.splitext(
        file_path
    )[1].lower()

    if extension == ".pdf":
        return "PDF", extract_pdf(file_path)

    if extension == ".docx":
        return "DOCX", extract_docx(file_path)

    if extension == ".txt":
        return "TXT", extract_txt(file_path)

    if extension == ".csv":
        return "CSV", extract_csv(file_path)

    if extension == ".json":
        return "JSON", extract_json(file_path)

    if extension in [".xlsx", ".xls"]:
        return "XLSX", extract_xlsx(file_path)

    raise ValueError(
        f"Unsupported file type: {extension}"
    )