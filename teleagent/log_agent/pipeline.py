import os

from .extractor import extract_file
from .normalizer import (
    normalize_txt,
    normalize_csv
)

from .formatter import (
    create_slm_context,
    create_prompt,
    save_json
)


class LogPipeline:

    def __init__(self, output_dir="output"):

        self.output_dir = output_dir

        os.makedirs(
            self.output_dir,
            exist_ok=True
        )


    def process_file(self, file_path):

        file_name = os.path.basename(
            file_path
        )

        file_type, extracted = extract_file(
            file_path
        )

        records = []

        if file_type == "TXT":

            records = normalize_txt(
                extracted,
                file_name
            )


        elif file_type == "CSV":

            records = normalize_csv(
                extracted,
                file_name
            )


        elif file_type == "PDF":

            print("\n========== PDF DEBUG ==========")

            print(
                "File:",
                file_name
            )

            print(
                "Pages extracted:",
                len(extracted)
            )

            total_lines = 0
            total_records = 0

            for page in extracted:

                page_number = page["page"]
                page_text = page["text"]

                lines = page_text.splitlines()

                print(
                    f"\nPage {page_number}:"
                )

                print(
                    "  Characters:",
                    len(page_text)
                )

                print(
                    "  Lines:",
                    len(lines)
                )

                # Show first few extracted lines
                for debug_line in lines[:5]:

                    print(
                        "  RAW:",
                        repr(debug_line)
                    )

                page_records = normalize_txt(
                    page_text,
                    file_name
                )

                print(
                    "  Parsed records:",
                    len(page_records)
                )

                for record in page_records:

                    if hasattr(
                        record,
                        "metadata"
                    ):

                        record.metadata[
                            "page"
                        ] = page_number

                records.extend(
                    page_records
                )

                total_lines += len(lines)

                total_records += len(
                    page_records
                )

            print(
                "\n---------- PDF TOTAL ----------"
            )

            print(
                "Total extracted lines:",
                total_lines
            )

            print(
                "Total parsed records:",
                total_records
            )

            print(
                "================================\n"
            )


        elif file_type == "DOCX":

            for paragraph in extracted[
                "paragraphs"
            ]:

                records.extend(
                    normalize_txt(
                        paragraph["text"],
                        file_name
                    )
                )

            for table in extracted[
                "tables"
            ]:

                for row in table:

                    row_text = " | ".join(
                        row
                    )

                    records.extend(
                        normalize_txt(
                            row_text,
                            file_name
                        )
                    )


        elif file_type == "JSON":

            if isinstance(
                extracted,
                list
            ):

                for item in extracted:

                    records.extend(
                        normalize_txt(
                            str(item),
                            file_name
                        )
                    )

            else:

                records.extend(
                    normalize_txt(
                        str(extracted),
                        file_name
                    )
                )


        elif file_type == "XLSX":

            for sheet_name, sheet_data in extracted.items():

                records.extend(
                    normalize_csv(
                        sheet_data,
                        f"{file_name}:{sheet_name}"
                    )
                )


        return records


    def process_files(self, file_paths):

        all_records = []

        source_files = []

        errors = []

        for file_path in file_paths:

            try:

                records = self.process_file(
                    file_path
                )

                all_records.extend(
                    records
                )

                source_files.append(
                    os.path.basename(
                        file_path
                    )
                )

            except Exception as exc:

                errors.append({
                    "file": file_path,
                    "error": str(exc)
                })


        context = create_slm_context(
            all_records,
            source_files
        )

        prompt = create_prompt(
            context
        )


        structured_path = os.path.join(
            self.output_dir,
            "structured_logs.json"
        )

        context_path = os.path.join(
            self.output_dir,
            "slm_context.json"
        )

        prompt_path = os.path.join(
            self.output_dir,
            "slm_prompt.txt"
        )


        save_json(
            [
                record.__dict__
                for record in all_records
            ],
            structured_path
        )


        save_json(
            context,
            context_path
        )


        with open(
            prompt_path,
            "w",
            encoding="utf-8"
        ) as file:

            file.write(
                prompt
            )


        return {
            "records": all_records,

            "source_files":
                source_files,

            "errors":
                errors,

            "context":
                context,

            "prompt":
                prompt,

            "structured_path":
                structured_path,

            "context_path":
                context_path,

            "prompt_path":
                prompt_path
        }