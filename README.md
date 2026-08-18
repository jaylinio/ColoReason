# ColoReason

ColoReason is a research codebase for longitudinal colorectal-cancer EHR structuring, evaluation, and GRPO reward design.

This release contains source code, prompts, JSON schemas, a synthetic smoke-test fixture, and data-preparation utilities.
It intentionally excludes all clinical data, derived datasets, patient identifiers, model weights, checkpoints, Python
environments, compiled wheels, logs, caches, and vendored third-party source trees.

## Repository layout

- `prompts/`: upstream and downstream prompting templates.
- `evaluation/scripts/`: metric and failure-analysis scripts. They require externally supplied prediction and gold files.
- `preprocessed/scripts/`: utilities that transform private raw data into private derived artifacts.
- `train/RL/coloreason_rl/`: composite reward, schemas, GRPO launch scripts, and unit tests.

## Data availability

This repository does not ship any datasets or trained model artifacts. The colorectal EHR datasets and trained model
checkpoints used in this work are available by email request to the corresponding authors, subject to institutional
approval and an appropriate data-use agreement.

## Environment boundary

The repository does not pin or bundle a Python/CUDA environment. Install compatible versions of Python, PyTorch,
Transformers, `jsonschema`, `fuzzywuzzy`, ms-swift, vLLM, DeepSpeed, and FlashAttention in the execution environment.
Set `ROLLOUT_ENV`, `TRAIN_ENV`, `BASE_MODEL`, and adapter/model paths locally; none have repository defaults containing
machine-specific paths.

## Quick checks

```bash
python -m unittest discover -s train/RL/coloreason_rl/tests
```

The full GRPO pipeline additionally requires GPUs, external datasets, model checkpoints, and a compatible ms-swift/vLLM
installation. See `train/RL/coloreason_rl/README.md` for the runtime contract.

## License

Released under the MIT License. See [LICENSE](LICENSE).
