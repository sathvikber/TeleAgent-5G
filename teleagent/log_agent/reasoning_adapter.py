class LogReasoningAdapter:

    def __init__(
        self,
        reasoning_service
    ):

        self.reasoning_service = (
            reasoning_service
        )


    def analyze(self, log_result):

        prompt = log_result["prompt"]

        return self.reasoning_service.analyze(
            prompt
        )