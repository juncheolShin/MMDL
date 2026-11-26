"""Restore the 640 selected TabMWP tables and both ordered training passes."""
import argparse
import csv
import hashlib
import json
import urllib.request
from decimal import Decimal
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

DATA = Path("training_data/tab640")
RECIPE = Path(__file__).with_name("recipe.tsv")
URL = "https://raw.githubusercontent.com/lupantech/PromptPG/5a1a52214521590b075f545b76a4f5ce666345e3/data/tabmwp/problems_train.json"
SOURCE_SHA = "63e57d9a212740f080d4e08657d7aeda3d9f95b26c74a1bdfd9c576846d9b828"
font = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', 24)
bold = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf', 24)

def render(r,path):
    grid = [line.split('|') for line in r['table'].splitlines()]
    grid = [[cell.strip() for cell in row] for row in grid]
    assert len(set(map(len,grid))) == 1
    head = all(not k.startswith('Column ') for k in r['table_for_pd'])
    title = r['table_title'] or ''
    widths = [max(110, int(max(bold.getlength(row[j]) for row in grid))+32) for j in range(len(grid[0]))]
    width = max(sum(widths)+30, int(bold.getlength(title))+32)
    offset = 50 if title else 16
    im = Image.new('RGB',(width,offset+len(grid)*42+16),'white')
    draw=ImageDraw.Draw(im)
    if title: draw.text((16,8),title,font=bold,fill='black')
    for i,row in enumerate(grid):
        x,y=15,offset+42*i
        for j,v in enumerate(row):
            draw.rectangle((x,y,x+widths[j],y+42),fill='#e8edf2' if head and i==0 else 'white',outline='#526070')
            draw.text((x+12,y+5),v,font=bold if head and i==0 else font,fill='black')
            x+=widths[j]
    im.save(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, help="Optional local copy of the pinned problems_train.json")
    args = parser.parse_args()
    raw = args.source.read_bytes() if args.source else urllib.request.urlopen(URL, timeout=120).read()
    if hashlib.sha256(raw).hexdigest() != SOURCE_SHA:
        raise ValueError("Unexpected upstream TabMWP train revision")
    problems = json.loads(raw)
    (DATA / "images").mkdir(parents=True, exist_ok=True)
    rendered = {}
    seen_tables = set()
    rows = []
    with RECIPE.open() as f:
        for r in csv.DictReader(f, delimiter="\t"):
            i = r["source_id"]
            original = problems[i]
            assert original["split"] == "train"
            image = DATA / "images" / (i + ".png")
            if i not in rendered:
                table = original["table"].strip().lower()
                assert table not in seen_tables
                seen_tables.add(table)
                render(original, image)
                with Image.open(image) as im:
                    rgb = im.convert("RGB")
                    pixels = hashlib.sha256(f"RGB:{rgb.width}:{rgb.height}:".encode() + rgb.tobytes()).hexdigest()
                if pixels != r["rgb_sha256"]:
                    raise ValueError(f"Rendering differs for table {i}; use matching Pillow/FreeType/DejaVu.")
                rendered[i] = hashlib.sha256(image.read_bytes()).hexdigest()
            options = r["options"].split("|")
            gold = r["gold"]
            mode = r["mode"]
            assert len(set(options)) == 10 and gold in "ABCDEFGHIJ" and mode in ("bridge", "mc")
            number = options["ABCDEFGHIJ".index(gold)]
            assert Decimal(number) == Decimal(original["answer"].replace(",", "").replace("$", ""))
            target = number + " → " + gold if mode == "bridge" else gold
            rows.append(dict(id=i+("/bridge" if mode=="bridge" else "/direct"), question=original["question"],
                             options=options, gold=gold, mode=mode, target=target,
                             image=str(image), image_sha256=rendered[i]))
    assert len(rendered) == 640 and len({r["id"] for r in rows}) == len(rows) == 1280
    assert [r["mode"] for r in rows] == ["bridge"]*640 + ["mc"]*640
    temp = DATA / "train.json.tmp"
    temp.write_text(json.dumps(rows, ensure_ascii=False))
    temp.replace(DATA / "train.json")
    print("Verified 640 source tables and 1,280 ordered training examples.")

if __name__ == "__main__":
    main()
