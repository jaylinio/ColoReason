import os
from collections.abc import Sequence
from typing import Any


DEFAULT_INSTRUCTION = (
    "Given two schema-constrained colorectal-cancer patient states, represent their clinical semantic equivalence."
)


class QwenEmbeddingScorer:
    """Lazy, frozen Qwen3-Embedding scorer used inside an ms-swift ORM."""

    def __init__(
        self,
        model_path: str,
        *,
        instruction: str = DEFAULT_INSTRUCTION,
        max_length: int = 8192,
        batch_size: int = 8,
    ):
        self.model_path = model_path
        self.instruction = instruction
        self.max_length = max_length
        self.batch_size = batch_size
        self._model: Any = None
        self._tokenizer: Any = None
        self._device: Any = None

    def _load(self) -> None:
        if self._model is not None:
            return
        import torch
        from transformers import AutoModel, AutoTokenizer

        device = os.environ.get("COLOREASON_EMBEDDING_DEVICE")
        if device is None and os.environ.get("LOCAL_RANK") is not None:
            device = f"cuda:{int(os.environ['LOCAL_RANK'])}"
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self._device = torch.device(device)
        dtype = torch.bfloat16 if self._device.type == "cuda" else torch.float32
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_path,
            padding_side="left",
            local_files_only=True,
        )
        self._model = AutoModel.from_pretrained(
            self.model_path,
            dtype=dtype,
            attn_implementation="sdpa",
            local_files_only=True,
        )
        self._model.to(self._device)
        self._model.eval()

    def _encode(self, texts: Sequence[str]) -> Any:
        import torch
        import torch.nn.functional as F

        batches = []
        for start in range(0, len(texts), self.batch_size):
            inputs = self._tokenizer(
                list(texts[start : start + self.batch_size]),
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self._device)
            with torch.inference_mode():
                hidden = self._model(**inputs).last_hidden_state
            left_padded = bool(inputs["attention_mask"][:, -1].all())
            if left_padded:
                embeddings = hidden[:, -1]
            else:
                sequence_lengths = inputs["attention_mask"].sum(dim=1) - 1
                batch_indices = torch.arange(hidden.shape[0], device=hidden.device)
                embeddings = hidden[batch_indices, sequence_lengths]
            batches.append(F.normalize(embeddings.float(), p=2, dim=1))
        return torch.cat(batches, dim=0)

    def similarity(self, predictions: Sequence[str], references: Sequence[str]) -> list[float]:
        if len(predictions) != len(references):
            raise ValueError("predictions and references must have the same length")
        if not predictions:
            return []
        self._load()

        query_texts = [f"Instruct: {self.instruction}\nQuery:{text}" for text in predictions]
        unique_references = list(dict.fromkeys(references))
        query_embeddings = self._encode(query_texts)
        reference_embeddings = self._encode(unique_references)
        reference_index = {text: index for index, text in enumerate(unique_references)}
        scores = [
            float((query_embeddings[i] * reference_embeddings[reference_index[reference]]).sum().item())
            for i, reference in enumerate(references)
        ]
        return [min(max(score, 0.0), 1.0) for score in scores]
