"""Restore the locked training examples from the two official train archives."""
import argparse
import hashlib
import json
from pathlib import Path
from zipfile import ZipFile

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--geometry-zip", type=Path, required=True)
parser.add_argument("--chart-zip", type=Path, required=True)
args = parser.parse_args()
root = Path("training_data/icot")
root.mkdir(exist_ok=True, parents=True)
rows = {}
with ZipFile(args.geometry_zip) as geo, ZipFile(args.chart_zip) as charts:
    prefix = "ChartQA Dataset/train/"
    chart_items = json.loads(charts.read(prefix + "train_human.json"))
    for spec in json.loads(Path("icot/recipes/sources.json").read_text()):
        item_id = spec["id"]
        number = int(item_id.rsplit("_", 1)[1])
        if spec["source"] == "geometry3k":
            folder = f"train/{number}/"
            item = json.loads(geo.read(folder + "data.json"))
            assert item["data_type"] == "train" and int(item["id"]) == number
            question = item["problem_text"]
            options = dict(zip("ABCD", item["choices"], strict=True))
            gold = item["answer"]
            image = geo.read(folder + "img_diagram.png")
        else:
            assert spec["source"] == "chartqa"
            item = chart_items[number]
            assert item["imgname"] == spec["image_file"]
            question = item["query"]
            options, gold = spec["options"], spec["gold"]
            image = charts.read(prefix + "png/" + item["imgname"])
        assert hashlib.sha256(image).hexdigest() == spec["sha"], item_id
        payload = json.dumps([question, options, gold], ensure_ascii=False).encode()
        assert hashlib.sha256(payload).hexdigest() == spec["text_sha256"], item_id
        path = root / (item_id + ".png")
        path.write_bytes(image)
        rows[item_id] = dict(id=item_id, question=question, options=options, gold=gold, image=str(path))
(root / "rows.json").write_text(json.dumps(rows, ensure_ascii=False, indent=2))
print(f"Restored {len(rows)} distinct train examples with verified images.")
