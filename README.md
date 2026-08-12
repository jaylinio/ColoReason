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

## Data boundary

No dataset is distributed with this repository. Clinical data must remain in an access-controlled environment and must be
passed to scripts through explicit command-line arguments or environment variables. Do not add `rawdata/`, `preprocessed/`
data artifacts, predictions, checkpoints, or patient-level examples to Git.

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
