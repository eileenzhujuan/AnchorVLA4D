# AnchorVLA4D — Quick Start

This file explains how to run the example scripts under the `examples/qwenvla` directory and lists a few important notes.

## Overview
- The codebase is built on top of the MindSpeed-MM / MindSpeed framework. See [MindSpeedMM.md](MindSpeedMM.md) for environment and framework details.

## Quick Start (run example scripts from repository root)

You can run the example scripts directly from the repository root. Examples:

1. List available example scripts:

```bash
ls examples/qwenvla/*.sh
```

2. Run an example script from the repository root:

```bash
# Training / fine-tuning example
bash examples/qwenvla/qwen2_5_vl_cosinelr.sh

# Alternative variant
bash examples/qwenvla/qwen2_5_vl_fixlr.sh
```

Most scripts set environment variables at the top (for example `LOAD_PATH`, `SAVE_PATH`, `DATA_JSON`) and reference `model_*.json` and `data_*.json` configuration files. Update those paths to match your local layout or export environment variables before running the script.

Example: set the model load path before running (run from repository root):

```bash
export LOAD_PATH="/path/to/ckpt/mm_path/Qwen2.5-VL-7B-Instruct"
bash examples/qwenvla/qwen2_5_vl_cosinelr.sh
```


## Configuration notes
- Edit `data_*.json` and `model_*.json` to point to local model weights, datasets, and cache directories (fields such as `from_pretrained`, `dataset_dir`, `cache_dir`).
- Avoid using the same `cache_dir` across multiple machines to prevent write conflicts.


## Data preparation
1. Produce a JSON file with format like `examples/qwenvla/mllm_format_example.json`

3. Place the generated JSON under `./data/` (or a path you choose) and update the `dataset` field in the `data_*.json` config used by the examples. The `dataset` field can list multiple JSON files separated by commas.

Example `data_*.json` fragment:

```json
"basic_parameters": {
	"dataset_dir": "./data",
	"dataset": "./data/mllm_format_example.json",
	"cache_dir": "./data/cache_dir",
	"max_samples": null
}
```

Notes and variants:
- For pure text samples, omit the `image` key. The loader supports mixed-image and text-only entries.
- Some datasets use `user`/`assistant` role tags instead of `human`/`gpt`; the loader supports configurable role and content tags via `data_*.json` (`attr` fields such as `role_tag`, `content_tag`, `user_tag`, `assistant_tag`).
- For trajectory/action/state datasets (e.g. `lerobot`), entries may include additional keys such as `states` or `messages`. Check the example `data_lerobot_*.json` configs in `examples/qwenvla` for how to set `attr` mappings (`states`, `images`, `messages`).

Loading and pre-processing:
- The dataset config supports streaming, batching, and caching. Set `streaming` and `cache_dir` in `data_*.json` according to your environment.
- During development you can limit processed samples with `max_samples` to validate the pipeline quickly.

## Checkpoint conversion (Megatron-style ↔ HuggingFace-style)

This framework trains using Megatron-style checkpoint formats (pipeline & tensor parallel sharded checkpoints). After training you can convert weights between Megatron/MindSpeed-MM and HuggingFace formats using the `mm-convert` tool included in the MindSpeed toolkit.

General examples:

- Convert HuggingFace -> Megatron/MM format:

```bash
mm-convert Qwen2_5_VLConverter hf_to_mm \
	--cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct" \
	--cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-7B-Instruct" \
	--cfg.parallel_config.llm_pp_layers [[12,16]] \
	--cfg.parallel_config.vit_pp_layers [[32,0]] \
	--cfg.parallel_config.tp_size 1
```

- Convert Megatron/MM -> HuggingFace format:

```bash
mm-convert Qwen2_5_VLConverter mm_to_hf \
	--cfg.save_hf_dir "ckpt/mm_to_hf/Qwen2.5-VL-7B-Instruct" \
	--cfg.mm_dir "ckpt/mm_path/Qwen2.5-VL-7B-Instruct" \
	--cfg.hf_config.hf_dir "ckpt/hf_path/Qwen2.5-VL-7B-Instruct" \
	--cfg.parallel_config.llm_pp_layers [1,10,10,7] \
	--cfg.parallel_config.vit_pp_layers [32,0,0,0] \
	--cfg.parallel_config.tp_size 1
```

Notes:
- Ensure `llm_pp_layers`, `vit_pp_layers`, and `tp_size` match your model's `model_*.json` pipeline configuration.
- After conversion, update `LOAD_PATH` (or `model_name_or_path` in `data_*.json`) to point to the converted checkpoint directory when running examples or fine-tuning.
- Converting large models requires sufficient disk space and may be slow; perform conversions on a machine with adequate CPU/RAM.


## Licenses

- This repository retains upstream license files in `LICENSE` (NVIDIA, Huawei and other third parties). Do not remove or overwrite upstream license or NOTICE files.
- This project also ships its own Apache-2.0 license in `LICENSE-ANCHORVLA4D` (Copyright 2026 AnchorVLA4D) covering the new or modified parts contributed here.



