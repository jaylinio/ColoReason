# -*- coding: utf-8 -*-
"""Evaluation utilities."""

import json
import os

from typing import Any, Dict, List
import torch
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer


def to_text(value: Any) -> str:
    """Run to text."""
    if value is None:
        return ""
    if isinstance(value, dict):
        try:
            return json.dumps(value, ensure_ascii=False, sort_keys=True)
        except Exception:
            return str(value)
    if isinstance(value, (list, tuple, set)):
        return " ".join(to_text(v) for v in value)
    return str(value)


class EmbeddingScorer:
    """Helper for EmbeddingScorer."""

    def __init__(
        self,
        model_name: str = "",
        use_query_prompt: bool = False,
    ):

        # self.model = SentenceTransformer(
        #     model_name,
        #     model_kwargs={"attn_implementation": "flash_attention_2", "device_map": "auto", "trust_remote_code": True},
        #     tokenizer_kwargs={"padding_side": "left"},
        # )
        if not model_name:
            raise ValueError("model_name must be configured")
        self.model = SentenceTransformer(model_name)
        self.use_query_prompt = use_query_prompt
        self.cache: Dict[str, torch.Tensor] = {}

        self.device = self.model.device

    def embed(self, text: str) -> torch.Tensor:
        """Run embed."""
        t = text.strip()
        if t in self.cache:
            return self.cache[t]

        if self.use_query_prompt:
            emb = self.model.encode(
                [t], prompt_name="query", convert_to_tensor=True, device=self.device
            )
        else:
            emb = self.model.encode([t], convert_to_tensor=True, device=self.device)
        self.cache[t] = emb
        return emb

    def score_0_100(self, v1: Any, v2: Any) -> float:
        """Run score 0 100."""
        s1 = to_text(v1)
        if s1 == "":
            return None
        s2 = to_text(v2)

        e1 = self.embed(s1)  # (1, dim) tensor on self.device
        e2 = self.embed(s2)  # (1, dim) tensor on self.device

        sim_mat = self.model.similarity(e1, e2)  # (1,1) tensor，[-1, 1]
        cos = float(sim_mat[0][0].item())
        score = (cos + 1.0) / 2.0 * 100.0
        return round(score, 4)


def compare_values_with_embedding(val1: Any, val2: Any, scorer: EmbeddingScorer) -> Any:
    """Run compare values with embedding."""
    if isinstance(val1, dict) and isinstance(val2, dict):
        nested_similarity: Dict[str, Any] = {}
        for key in val1:
            nested_similarity[key] = compare_values_with_embedding(
                val1[key], val2.get(key, ""), scorer
            )
        return nested_similarity
    else:
        return scorer.score_0_100(val1, val2)


def rescore_existing_result_jsonl(
    input_file: str,
    model_name: str = "",
    use_query_prompt: bool = False,
    output_dir: str = "",
    output_suffix: str = "_processed_emb.jsonl",
    show_progress: bool = True,
) -> str:
    """Run rescore existing result jsonl."""
    torch.set_grad_enabled(False)

    scorer = EmbeddingScorer(model_name=model_name, use_query_prompt=use_query_prompt)

    if torch.cuda.is_available():
        cuda_name = torch.cuda.get_device_name(0)
    else:
        cuda_name = "CPU (no CUDA)"
    print(
        f"[Device] SentenceTransformer on: {scorer.device} | torch.cuda.is_available={torch.cuda.is_available()} | {cuda_name}"
    )

    total_lines = None
    if show_progress:
        with open(input_file, "r", encoding="utf-8") as f:
            total_lines = sum(1 for _ in f)

    output_data: List[Dict[str, Any]] = []
    with open(input_file, "r", encoding="utf-8") as infile:
        iterator = infile
        if show_progress:
            iterator = tqdm(infile, total=total_lines, desc="Rescoring", unit="line")

        for line in iterator:
            if not line.strip():
                continue
            data = json.loads(line)
            response = data.get("response", "")
            answer = data.get("answer", "")

            similarity = {}

            if isinstance(response, dict) and isinstance(answer, dict):
                for key in response:
                    similarity[key] = compare_values_with_embedding(
                        response[key], answer.get(key, ""), scorer
                    )

            new_record = dict(data)
            new_record["similarity"] = similarity
            output_data.append(new_record)

    os.makedirs(output_dir, exist_ok=True)
    file_name = os.path.splitext(os.path.basename(input_file))[0] + output_suffix
    output_file = os.path.join(output_dir, file_name)

    with open(output_file, "w", encoding="utf-8") as outfile:
        for item in output_data:
            json.dump(item, outfile, ensure_ascii=False)
            outfile.write("\n")

    print(f"Processed file saved to: {output_file}")
    return output_file


if __name__ == "__main__":
    input_file = ""
    output_dir = ""
    os.makedirs(output_dir, exist_ok=True)
    rescore_existing_result_jsonl(
        input_file=input_file,
        model_name="",
        use_query_prompt=False,
        output_dir=output_dir,
        output_suffix="_emb.jsonl",
        show_progress=True,
    )
