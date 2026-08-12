import json
import os
import sys
import unittest
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
os.environ["COLOREASON_RL_ROOT"] = str(PROJECT_ROOT)

from coloreason_reward import ColoReasonCompositeReward
from coloreason_reward.json_utils import canonical_json, extract_json_object


class FakeEmbeddingScorer:
    def similarity(self, predictions, references):
        return [1.0 if prediction == reference else 0.5 for prediction, reference in zip(predictions, references)]


class RewardTest(unittest.TestCase):
    def setUp(self):
        self.reward = ColoReasonCompositeReward(embedding_scorer=FakeEmbeddingScorer())
        self.solution = {"diagnosis": "colorectal adenocarcinoma", "tnm": "T2N0M0"}

    def score(self, completion, **kwargs):
        defaults = {
            "schema_id": ["smoke_v1"],
            "response_token_ids": [[1, 2, 3]],
            "is_truncated": [False],
            "finish_reason": ["stop"],
        }
        defaults.update(kwargs)
        return self.reward([completion], [self.solution], **defaults)[0]

    def test_exact_json_scores_one(self):
        self.assertAlmostEqual(self.score(json.dumps(self.solution)), 1.0)

    def test_last_json_object_is_used(self):
        text = '<think>reasoning</think> {"bad": true}\n' + json.dumps(self.solution)
        self.assertEqual(extract_json_object(text), self.solution)
        self.assertAlmostEqual(self.score(text), 1.0)

    def test_nested_json_returns_top_level_object(self):
        value = {"diagnosis": "x", "evidence": {"source": "pathology"}}
        self.assertEqual(extract_json_object(json.dumps(value)), value)

    def test_invalid_json_is_hard_failure(self):
        self.assertEqual(self.score("not json"), -1.0)

    def test_schema_violation_is_hard_failure(self):
        self.assertEqual(self.score('{"diagnosis":"x","tnm":"T9N9M9"}'), -1.0)

    def test_truncation_is_hard_failure(self):
        self.assertEqual(self.score(json.dumps(self.solution), is_truncated=[True]), -1.0)

    def test_length_finish_reason_is_hard_failure(self):
        self.assertEqual(self.score(json.dumps(self.solution), finish_reason=["length"]), -1.0)

    def test_canonical_json_is_key_order_invariant(self):
        reverse = {"tnm": "T2N0M0", "diagnosis": "colorectal adenocarcinoma"}
        self.assertEqual(canonical_json(reverse), canonical_json(self.solution))


if __name__ == "__main__":
    unittest.main()
