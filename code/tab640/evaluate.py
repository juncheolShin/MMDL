"""Apply the archived presentation protocol to pinned test parquet snapshots."""
import argparse
from hashlib import sha256
import io
import json
import os
from pathlib import Path
import sys
import time
import tomllib

import pyarrow.parquet as pq
from PIL import Image
from transformers import AutoProcessor
from vllm import LLM, SamplingParams

ROOT = Path.cwd()
sys.path[:0] = [str(ROOT / "code/eval"), str(ROOT / "code")]
from mmmu_data import _parse_options, _normalize_question_type, IMAGE_COLUMNS
from prompting import build_mmmu_prompt
from third_party.qwen3vl_mmmu.input_utils import prepare_inputs_for_vllm

SNAP = ROOT / "data/hf"
SETS = {"pro10": SNAP / "MMMU_Pro/standard (10 options)", "pro4": SNAP / "MMMU_Pro/standard (4 options)",
        "vision": SNAP / "MMMU_Pro/vision", "test": SNAP / "MMMU"}
HASHES = {"MMMU": "98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68", "MMMU_Pro": "563f3e84bb3b90893083a1f039cfa13077f2302b"}
CFG = tomllib.loads((ROOT / "configs/evaluation.toml").read_text())["evaluation"]
CHUNK = 36


def request(row, setting, processor):
    opts = _parse_options(row["options"], row["id"])
    kind = _normalize_question_type(row.get("question_type"), opts, row["id"])
    if setting == "vision":
        cols = ["image"]
    else:
        cols = [k for k in IMAGE_COLUMNS if row.get(k) is not None]
    imgs = [Image.open(io.BytesIO(row[k]["bytes"])) for k in cols]
    img_hashes = [sha256(row[k]["bytes"]).hexdigest() for k in cols]
    pixels = CFG["max_pixels"]
    for attempt in range(8):
        line = {"id": row["id"], "question": row.get("question", ""), "question_type": kind, **opts}
        messages = build_mmmu_prompt(line, imgs, CFG["min_pixels"], pixels)
        if setting == "vision":
            messages[0]["content"][-1]["text"] = "Answer with the option letter only."
        req = prepare_inputs_for_vllm(messages, processor)
        # Each merged 2x2 group of 16x16 visual patches uses one language token.
        visual = req["multi_modal_data"].get("image", [])
        size = len(processor.tokenizer.encode(req["prompt"], add_special_tokens=False))
        size += sum((im.width // 32) * (im.height // 32) - 1 for im in visual)
        if size <= CFG["max_model_len"] - CFG["max_new_tokens"]:
            break
        if pixels <= CFG["min_pixels"]:
            raise ValueError(f"Context overflow at minimum resolution for {row['id']}")
        pixels = max(CFG["min_pixels"], pixels * 3 // 4)
    else:
        raise ValueError(f"Unresolved context overflow for {row['id']}")
    fingerprint = sha256(json.dumps([req["prompt"], img_hashes, pixels]).encode()).hexdigest()
    return req, {"id": row["id"], "question_type": kind, "options": opts, "gold": row["answer"],
                 "input_sha256": fingerprint, "max_pixels": pixels}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("model", type=Path, help="Base or verified merged checkpoint directory")
    parser.add_argument("output", type=Path, help="New directory for this model only")
    args = parser.parse_args()
    for name, rev in HASHES.items():
        assert json.loads((SNAP / name / "MMDL_SNAPSHOT.json").read_text())["revision"] == rev
    args.output.mkdir(parents=True, exist_ok=False)
    model = str(args.model.resolve())
    processor = AutoProcessor.from_pretrained(model, local_files_only=True)
    engine = LLM(model=model, tensor_parallel_size=1, gpu_memory_utilization=CFG["gpu_memory_utilization"],
                 trust_remote_code=True, max_model_len=CFG["max_model_len"], max_num_seqs=CFG["max_num_seqs"],
                 limit_mm_per_prompt={"image": CFG["max_images_per_prompt"]}, seed=CFG["seed"])
    sampling = SamplingParams(max_tokens=CFG["max_new_tokens"], stop_token_ids=[],
                              **{k:CFG[k] for k in ("temperature","top_k","top_p","repetition_penalty","presence_penalty")})
    for setting, folder in SETS.items():
        dest = args.output / setting
        dest.mkdir(parents=True, exist_ok=True)
        for index, file in enumerate(sorted(folder.rglob("test-*.parquet"))):
            for j, batch in enumerate(pq.ParquetFile(file).iter_batches(batch_size=CHUNK)):
                path = dest / f"{index:03d}-{j:03d}.json"
                inputs, metadata = zip(*(request(row,setting,processor) for row in batch.to_pylist()))
                started = time.monotonic()
                outputs = engine.generate(list(inputs), sampling, use_tqdm=False)
                seconds = time.monotonic() - started
                for m,res in zip(metadata,outputs,strict=True):
                    out, = res.outputs
                    m.update(response=out.text.split("</think>")[-1].strip(), n_output_tokens=len(out.token_ids),
                             n_prompt_tokens=len(res.prompt_token_ids), finish_reason=out.finish_reason)
                tmp = path.with_suffix(".tmp")
                tmp.write_text(json.dumps({"generation_s":seconds,"records":metadata},ensure_ascii=False))
                os.replace(tmp,path)
                print(args.model, setting, path.stem, len(metadata), round(seconds,1), flush=True)
    print("FINISHED", args.model, flush=True)


if __name__ == "__main__":
    main()
