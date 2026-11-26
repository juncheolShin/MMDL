"""Check the published adapters, evaluation code and paired item counts on CPU."""
import csv
import hashlib
import json
from pathlib import Path

root = Path(__file__).resolve().parent
evidence = root / "evidence"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def records(name, count):
    with (evidence / name).open(encoding="utf-8-sig", newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == len({row["id"] for row in rows}) == count
    return rows


for number in ("495", "498"):
    parent = root / "adapters" / number
    saved = json.loads((parent / "checkpoint.json").read_text())
    for name, expected in saved["files_sha256"].items():
        assert digest(parent / name) == expected, (number, name)
assert saved["manifest"]["parent_checkpoint_sha256"] == digest(root / "adapters/495/checkpoint.json")
protocol = json.loads((evidence / "evaluation_498.json").read_text())
assert digest(root.parent / "configs/evaluation.toml") == protocol["config_sha256"]
for filename, expected in (protocol["pipeline_sha256"] | protocol["data_and_vendored_sha256"]).items():
    if filename == "evaluation_data.json":
        continue
    published = root / "scoring" / Path(filename).name
    path = published if published.exists() else root.parent / filename
    assert digest(path) == expected, filename

mmmu = records("mmmu.csv", 900)
assert sum(row["type"] == "open" for row in mmmu) == 53
for name, expected in (("495", 495), ("498", 498), ("answer", 494)):
    assert sum(int(row[name]) for row in mmmu) == expected
changes = {row["id"]: "lost" if row["498"] == "1" else "recovered"
           for row in mmmu if row["498"] != row["answer"]}
assert changes == {row["id"]: row["direction"] for row in records("changes_vs_498.csv", 12)}
assert list(changes.values()).count("recovered") == 4
assert sum(int(row["498_tokens"]) for row in mmmu) == 3775
assert sum(int(row["answer_tokens"]) for row in mmmu) == 2609
external = records("external.csv", 141)
for group, size, expected in ((None, 141, [76, 81, 79]), ("recovery", 23, [0, 5, 3]),
                              ("preservation", 38, [38, 38, 38])):
    subset = [row for row in external if group is None or row["group"] == group]
    assert len(subset) == size
    assert [sum(int(row[name]) for row in subset) for name in ("498", "answer", "short")] == expected
assert {row["id"] for row in records("baseline_open_wrong_46.csv", 46)} <= {
    row["id"] for row in mmmu if row["type"] == "open"}
print("Verified both adapters and the frozen evaluation files.")
print("MMMU: 495 -> 498 -> 494 (answer-only continuation: +4 / -8).")
print("External validation (498 / answer / short): 76 / 81 / 79 of 141.")
