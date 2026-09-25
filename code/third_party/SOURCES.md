# Vendored evaluation code

These files are unmodified copies of upstream code. `code/eval/` imports them instead of reimplementing them,
so the prompt and parsers stay identical to what the upstream authors used.

| Directory | Upstream | Commit | Files | License |
|---|---|---|---|---|
| `qwen3vl_mmmu/` | [QwenLM/Qwen3-VL](https://github.com/QwenLM/Qwen3-VL) `evaluation/mmmu/` | `96588727e44c78b25ba03ea03b8e12f7e64fd0da` (the last change to `evaluation/mmmu` is `f8dca990`, 2025-11-25) | `run_mmmu.py`, `dataset_utils.py`, `eval_utils.py`, `common_utils.py` | Apache-2.0 (`qwen3vl_mmmu/LICENSE`) |
| `mmmu_official/` | [MMMU-Benchmark/MMMU](https://github.com/MMMU-Benchmark/MMMU) `mmmu/utils/` | `268471d0d488258990025331c7528359c324aa25` | `eval_utils.py`, `data_utils.py` | Apache-2.0 (`mmmu_official/LICENSE`) |

`data/mmmu_answer_dict_val.json` is `mmmu/answer_dict_val.json` from the same MMMU commit.

## Where each file is used

- `qwen3vl_mmmu/run_mmmu.py`: `build_mmmu_prompt` (the prompt function *p*) and `prepare_inputs_for_vllm`. It applies the chat template and uses `qwen_vl_utils` to resize images. The prompt shown on p.10 of the lecture comes from here.
- `qwen3vl_mmmu/dataset_utils.py`: loads the TSV, dumps the images and runs `MMMU_preproc`.
- `qwen3vl_mmmu/eval_utils.py`: the rule-based extractor `can_infer` (the VLMEvalKit rule stage). The GPT-judge fallback is not used here because it needs a DashScope or OpenAI API key.
- `mmmu_official/eval_utils.py`: the benchmark's official parser (`parse_multi_choice_response` and `parse_open_response`) and scorer (`evaluate` and `calculate_ins_level_acc`).

## Data file checksum

The Qwen3-VL repo pins the MD5 of `MMMU_DEV_VAL.tsv` as `521afc0f3bf341e6654327792781644d`. The file currently served at
`https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv` has MD5 `585e8ad75e73f75dcad265dfd0417d64`,
which is the value VLMEvalKit lists today in `vlmeval/dataset/image_mcq.py`. Qwen's loader re-downloads the file whenever the MD5 does not match,
so running their code also ends up using this file. `code/eval/mmmu_data.py` checks for the VLMEvalKit MD5.
(The server's TLS certificate had expired on 2026-09-24. The file was downloaded with `curl -k`, and its MD5 was then checked against the VLMEvalKit value.)

## One deliberate deviation from Qwen's loader

`pd.read_csv` with default settings reads the option text `None` as NaN. In MMMU val this affects one cell:
`validation_Geography_15` option D = "None", **which is also the ground-truth answer**. With Qwen's code, option D
disappears from the prompt and from the option list the parsers see, so the question cannot be answered correctly.
`code/eval/mmmu_data.py` puts back any option cell that is non-empty in the file, and `run_config.json` records
which cells were restored (`restored_option_cells`).
