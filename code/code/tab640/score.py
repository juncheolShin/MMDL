"""Score a complete frozen-order test/Pro run with the original official parsers."""
import argparse
from hashlib import sha256
import importlib.util
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT / "icot/scoring"), str(ROOT / "code")]
from score_mmmu import RecordingRandom, score_official


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("run", type=Path)
    args = parser.parse_args()
    official_dir = Path("data/hf/official")
    answer_file = official_dir / "answer_dict_test.json"
    pro_file = official_dir / "evaluate.py"
    assert sha256(answer_file.read_bytes()).hexdigest() == "45b639a937017d3d71f81f610df3a1471debe1887c5c1d24644fec1b33059d5c"
    assert sha256(pro_file.read_bytes()).hexdigest() == "4c825ed353774658a1a8bc9966e4c52c0c59227e4c5fe5f480b7549817027c2c"
    answers = json.loads(answer_file.read_bytes())
    spec = importlib.util.spec_from_file_location("mmmu_pro_official", pro_file)
    pro = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(pro)
    summary = {}
    pro_ids = None
    for name in ("test", "pro4", "pro10", "vision"):
        records = []
        for file in sorted((args.run / name).glob("*.json")):
            chunk = json.loads(file.read_bytes())
            records.extend(chunk["records"] if isinstance(chunk, dict) else chunk)
        ids = {r["id"] for r in records}
        assert len(records) == len(ids) == (10500 if name == "test" else 1730)
        if name == "test":
            assert ids == set(answers)
            gold = {r["id"]: answers[r["id"]] | {"options": r["options"]} for r in records}
            scored = score_official(records, gold)
            count = sum(r["correct"] for r in scored.values())
        else:
            if pro_ids is not None:
                assert pro_ids == ids
            pro_ids = ids
            pro.random = RecordingRandom(42)
            count = sum(pro.parse_multi_choice_response(r["response"], list(r["options"]), r["options"]) == r["gold"]
                        for r in records)
        tokens = sum(r["n_output_tokens"] if "n_output_tokens" in r else r["tokens"] for r in records)
        summary[name] = dict(correct=count, n=len(ids), accuracy=count/len(ids), mean_output_tokens=tokens/len(ids))
    summary["pro_overall"] = (summary["pro10"]["accuracy"] + summary["vision"]["accuracy"])/2
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
