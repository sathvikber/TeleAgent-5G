import re
import torch
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM


class BoundaryTranslator:
    MODEL_NAME = "facebook/nllb-200-distilled-600M"

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("Loading translation model...")

        self.tokenizer = AutoTokenizer.from_pretrained(
            self.MODEL_NAME,
            src_lang="hin_Deva",
        )

        self.model = AutoModelForSeq2SeqLM.from_pretrained(
            self.MODEL_NAME
        ).to(self.device)

        self.model.eval()

        # Avoid conflicting max_length and max_new_tokens settings.
        self.model.generation_config.max_length = 512
        self.model.generation_config.max_new_tokens = None

        print(f"Translation model loaded successfully on {self.device}!")

    @staticmethod
    def contains_hindi(text):
        """Detect Hindi written in Devanagari script."""
        return bool(re.search(r"[\u0900-\u097F]", text))

    def translate(self, text, source_language, target_language):
        """Translate text between supported NLLB languages."""

        if not isinstance(text, str) or not text.strip():
            return text if isinstance(text, str) else ""

        self.tokenizer.src_lang = source_language

        inputs = self.tokenizer(
            text.strip(),
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self.device)

        target_id = self.tokenizer.convert_tokens_to_ids(
            target_language
        )

        if target_id == self.tokenizer.unk_token_id:
            raise ValueError(
                f"Unsupported target language: {target_language}"
            )

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                forced_bos_token_id=target_id,
                max_new_tokens=256,
                num_beams=4,
                early_stopping=True,
            )

        translated_text = self.tokenizer.batch_decode(
            outputs,
            skip_special_tokens=True,
        )[0].strip()

        if not translated_text:
            raise RuntimeError(
                f"Translation returned empty text: "
                f"{source_language} -> {target_language}"
            )

        return translated_text

    def prepare_input(self, text):
        """
        Translate Hindi input into English.

        Returns:
            translated_text: Query sent to TeleAgent.
            reply_in_hindi: Whether to translate the answer into Hindi.
        """

        if not isinstance(text, str) or not text.strip():
            return text, False

        text = text.strip()

        if self.contains_hindi(text):
            print("\nTranslating Hindi input to English...")

            english_text = self.translate(
                text,
                source_language="hin_Deva",
                target_language="eng_Latn",
            )

            print(f"English query: {english_text}")

            return english_text, True

        # English input passes through unchanged.
        return text, False

    def translate_output(self, text, reply_in_hindi):
        """Translate the final TeleAgent response into Hindi if needed."""

        if not isinstance(text, str) or not text.strip():
            return text if isinstance(text, str) else ""

        if not reply_in_hindi:
            return text

        print("\nTranslating TeleAgent response to Hindi...")

        hindi_text = self.translate(
            text,
            source_language="eng_Latn",
            target_language="hin_Deva",
        )

        return hindi_text
