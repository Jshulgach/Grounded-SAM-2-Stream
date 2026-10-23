"""Compatibility adapter. Prefer grounded_stream.phrases.OllamaPhrases."""

from grounded_stream.phrases import OllamaPhrases, model_exists


class LlamaWrapper(OllamaPhrases):
    def __init__(self, model_id="llama3.1"):
        super().__init__(model=model_id)

    model_exists = staticmethod(model_exists)

    def send(self, message):
        # Historical send returned detector text. Preserve that shape, with structured validation.
        return ". ".join(self.suggest(message)["targets"]) + "."
