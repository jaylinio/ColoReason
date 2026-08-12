import os
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any

from fuzzywuzzy import fuzz

from .embedding import QwenEmbeddingScorer
from .json_utils import canonical_json, extract_json_object, parse_reference
from .schema import SchemaRegistry


@dataclass
class _ValidRow:
    index: int
    prediction: str
    reference: str
    fuzzy_reward: float


class ColoReasonCompositeReward:
    """Paper-aligned reward with non-compensable length and schema gates."""

    def __init__(self, args: Any = None, embedding_scorer: Any = None):
        del args
        project_root = os.environ.get("COLOREASON_RL_ROOT")
        if not project_root:
            project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        schema_dir = os.environ.get("COLOREASON_SCHEMA_DIR", os.path.join(project_root, "schemas"))
        embedding_model = os.environ.get(
            "COLOREASON_EMBEDDING_MODEL", os.path.join(project_root, "models", "Qwen3-Embedding-0.6B")
        )
        embedding_max_length = int(os.environ.get("COLOREASON_EMBEDDING_MAX_LENGTH", "8192"))
        embedding_batch_size = int(os.environ.get("COLOREASON_EMBEDDING_BATCH_SIZE", "8"))
        self.max_tokens = int(os.environ.get("COLOREASON_MAX_COMPLETION_TOKENS", "32768"))
        self.schemas = SchemaRegistry(schema_dir)
        self.embedding = embedding_scorer or QwenEmbeddingScorer(
            embedding_model,
            max_length=embedding_max_length,
            batch_size=embedding_batch_size,
        )

    @staticmethod
    def _column(kwargs: dict[str, Any], name: str, size: int, default: Any = None) -> list[Any]:
        value = kwargs.get(name)
        if value is None:
            return [default] * size
        if len(value) != size:
            raise ValueError(f"column {name!r} has length {len(value)}, expected {size}")
        return list(value)

    def __call__(self, completions: Sequence[str], solution: Sequence[Any], **kwargs: Any) -> list[float]:
        size = len(completions)
        if len(solution) != size:
            raise ValueError("solution and completions must have the same length")
        schema_ids = self._column(kwargs, "schema_id", size)
        token_ids = self._column(kwargs, "response_token_ids", size, [])
        truncated = self._column(kwargs, "is_truncated", size, False)
        finish_reasons = self._column(kwargs, "finish_reason", size)
        rewards = [-1.0] * size
        valid_rows: list[_ValidRow] = []

        for index, (completion, reference_value, schema_id) in enumerate(zip(completions, solution, schema_ids)):
            is_overlong = len(token_ids[index] or []) > self.max_tokens
            is_truncated = bool(truncated[index]) or finish_reasons[index] == "length"
            if is_overlong or is_truncated:
                continue
            try:
                prediction_obj = extract_json_object(completion)
                reference_obj = parse_reference(reference_value)
                if not self.schemas.validate(prediction_obj, schema_id):
                    continue
                prediction = canonical_json(prediction_obj)
                reference = canonical_json(reference_obj)
            except (TypeError, ValueError, FileNotFoundError):
                continue

            fuzzy_score = float(fuzz.ratio(prediction, reference))
            valid_rows.append(_ValidRow(index, prediction, reference, 2.0 * fuzzy_score / 100.0 - 1.0))

        if not valid_rows:
            return rewards

        embedding_scores = self.embedding.similarity(
            [row.prediction for row in valid_rows], [row.reference for row in valid_rows]
        )
        for row, similarity in zip(valid_rows, embedding_scores):
            embedding_reward = 2.0 * min(max(float(similarity), 0.0), 1.0) - 1.0
            rewards[row.index] = (1.0 + row.fuzzy_reward + embedding_reward) / 3.0
        return rewards
