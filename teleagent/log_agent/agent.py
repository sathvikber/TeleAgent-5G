from .pipeline import LogPipeline
from .ui import select_log_files


class LogAgent:

    def __init__(
        self,
        output_dir="output"
    ):

        self.pipeline = LogPipeline(
            output_dir=output_dir
        )


    def run_from_files(
        self,
        file_paths
    ):

        return self.pipeline.process_files(
            file_paths
        )


    def run_with_file_picker(self):

        file_paths = select_log_files()

        if not file_paths:

            return {
                "status": "cancelled",
                "records": [],
                "source_files": []
            }

        result = self.run_from_files(
            file_paths
        )

        result["status"] = "success"

        return result