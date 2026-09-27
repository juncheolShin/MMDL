# Evaluation code sources

| Directory | Upstream | Commit | Files | License |
|---|---|---|---|---|
| `qwen3vl_mmmu/` | [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) `evaluation/mmmu/` | `96588727e44c78b25ba03ea03b8e12f7e64fd0da` | `input_utils.py` | Apache-2.0 (`qwen3vl_mmmu/LICENSE`) |
| `mmmu_official/` | [MMMU-Benchmark/MMMU](https://github.com/MMMU-Benchmark/MMMU) `mmmu/utils/` | `268471d0d488258990025331c7528359c324aa25` | `eval_utils.py`, `data_utils.py` | Apache-2.0 (`mmmu_official/LICENSE`) |

`input_utils.py` contains only `prepare_inputs_for_vllm`, extracted from the pinned Qwen `run_mmmu.py` without changing its function body. It applies the processor's chat template and prepares images with `qwen_vl_utils`. The multiprocessing environment setting is preserved.

The MMMU parser files are unmodified upstream copies used for reference scoring. The primary parser is `code/eval/answer_extraction.py`; the fixed prompts are in `code/eval/prompting.py` and documented in the submission report.

All evaluation questions, images, options, and ground-truth answers are loaded by `code/eval/mmmu_data.py` from `MMMU/MMMU` at revision `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`, validation split, one call per subject configuration.
