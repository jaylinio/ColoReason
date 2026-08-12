# ColoReason GRPO reproduction

This directory contains the ColoReason project layer. The training scripts expect an externally installed
ms-swift/vLLM stack; no Python environment, third-party source tree, model weights, or datasets are included.

## Method contract

The recovered historical method defines a four-stage **knowledge-domain curriculum**, not one stage per reward:

1. TNM extraction and harmonization.
2. Pathology and anatomical findings.
3. Treatment and metastasis reasoning.
4. Symptoms, follow-up, comorbidity, longitudinal aggregation, and integrated reasoning.

Every stage uses the same composite reward. Invalid/truncated or schema-invalid generations receive `-1` before
semantic terms are evaluated. Valid outputs receive the equal-weight mean of format (`1`), normalized fuzzy
similarity, and normalized embedding similarity.

Later-stage dataset files are intentionally named `*_cumulative.jsonl`: knowledge domains are introduced
sequentially while previous domains remain represented. The real datasets and production JSON schemas are not yet
available and must be supplied separately before production training.

## Environment

External GRPO uses two processes and two environments:

- `ROLLOUT_ENV`: environment containing the `swift rollout` CLI and vLLM.
- `TRAIN_ENV`: environment containing the `swift rlhf` CLI, FlashAttention, and vLLM external-client modules.

They communicate through the rollout HTTP port and a separate NCCL weight-sync group port. This is not a static
OpenAI-compatible inference service: GRPO synchronizes the updated LoRA policy to the rollout engine after optimizer
updates. In ms-swift 4.5.0.dev0 the trainer-side external client therefore still imports vLLM's PyNccl components.
`swift-rl` was cloned from the working `swift` environment and adds a source-built FlashAttention 2.8.4 wheel while
preserving its Python 3.10, Torch 2.11+cu130, vLLM 0.26 and CXX11 ABI combination. The wheel was compiled only for
Blackwell `sm_120`; do not reuse it on A100/H100 without rebuilding for the target compute capability.
When the processes run on different hosts, both `VLLM_SERVER_PORT` and `VLLM_SERVER_GROUP_PORT` must be reachable;
on one host, keep `VLLM_SERVER_HOST=127.0.0.1`.

Run the environment audit before allocating GPUs:

```bash
scripts/check_env.sh
```

Once the trainer client dependency check passes, start the two processes in separate tmux sessions. Example for a
single host, with GPU 4 for rollout and GPUs 0-3 for training:

```bash
tmux new-session -d -s coloreason-rollout \
  'cd path/to/repo/train/RL/coloreason_rl && \
   ROLLOUT_GPU_INDEX=4 VLLM_SERVER_HOST=127.0.0.1 VLLM_SERVER_PORT=8000 \
   scripts/start_rollout_server.sh'

tmux new-session -d -s coloreason-train \
  'cd path/to/repo/train/RL/coloreason_rl && \
   TRAIN_CUDA_VISIBLE_DEVICES=0,1,2,3 NPROC_PER_NODE=4 \
   VLLM_SERVER_HOST=127.0.0.1 VLLM_SERVER_PORT=8000 \
   BASE_MODEL=/path/to/Qwen3-8B SFT_REFERENCE_ADAPTER=/path/to/sft-adapter \
   scripts/run_curriculum.sh'
```
