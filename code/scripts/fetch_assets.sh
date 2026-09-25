#!/usr/bin/env bash
# Download the MMMU val data and Qwen3-VL-4B-Instruct at pinned versions into <repo>/data and <repo>/models,
# verifying every file's SHA-256. Safe to re-run: files that already verify are skipped.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

fetch() {  # fetch <url> <dest> <sha256> [extra curl args]
  local url=$1 dest=$2 sum=$3
  shift 3
  if [ -f "$dest" ] && echo "$sum  $dest" | sha256sum -c --status; then
    echo "ok       $dest"
    return
  fi
  mkdir -p "$(dirname "$dest")"
  # explicit returns: set -e does not apply inside a function called from `a || b`
  curl -fL --retry 3 "$@" -o "$dest.part" "$url" || return 1
  mv "$dest.part" "$dest" || return 1
  echo "$sum  $dest" | sha256sum -c -
}

# VLMEvalKit's MMMU_DEV_VAL.tsv (MD5 585e8ad75e73f75dcad265dfd0417d64). Its server's TLS certificate had expired
# on 2026-09-24, so this falls back to -k; the SHA-256 check is what authenticates the file.
TSV_URL=https://opencompass.openxlab.space/utils/VLMEval/MMMU_DEV_VAL.tsv
TSV_SHA=cae0445a60c2a22f29cb9df43d2d97671baa8d5921df5d8667f4f3b1f643a478
fetch "$TSV_URL" "$ROOT/data/MMMU_DEV_VAL.tsv" "$TSV_SHA" 2>/dev/null \
  || fetch "$TSV_URL" "$ROOT/data/MMMU_DEV_VAL.tsv" "$TSV_SHA" -k

# Official MMMU val answers (MMMU-Benchmark/MMMU mmmu/answer_dict_val.json)
fetch https://raw.githubusercontent.com/MMMU-Benchmark/MMMU/268471d0d488258990025331c7528359c324aa25/mmmu/answer_dict_val.json \
  "$ROOT/data/mmmu_answer_dict_val.json" 76080f5597b8f4d29abba8551489c4b82e4a285b9d62b946fd67a1952e95502c

# Qwen/Qwen3-VL-4B-Instruct at a fixed revision
REV=ebb281ec70b05090aa6165b016eac8ec08e71b17
MODEL="$ROOT/models/Qwen3-VL-4B-Instruct"
while read -r sum file; do
  fetch "https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/resolve/$REV/$file" "$MODEL/$file" "$sum"
done <<'EOF'
6f8a6a55027e3da5160105556cda5dd69f6423f1c32645f6730d32de7773d0c4 chat_template.json
edac7703329133edfc53e46ac0081835144c99d7eebf28b71c732694d435224d config.json
8469742d1fce0de951c8909b26a2c0c0d8490837ce476efb114da9e0cefc4d44 generation_config.json
599bab54075088774b1733fde865d5bd747cbcc7a547c5bc12610e874e26f5e3 merges.txt
30a01a0556622645a3cce87b655bbbbbc1f170c196099f1b666c93202c3339a9 model-00001-of-00002.safetensors
046296a2a387efb43b0c997d5833c789604d168834f6e0d3064bf7bb13d002a6 model-00002-of-00002.safetensors
58a7841d7bff2548dd91577d216274a83cf1b500bc6a534b809d6c1b1707cf2b model.safetensors.index.json
27225450ac9c6529872ee1924fcb0962ff5634834f817040f444118116f4e516 preprocessor_config.json
a884e5e78f7d6f7bfe237f909dbc41a126542e259dc79d8ab33cc8980580ff79 README.md
c2da771801886ad9ae98181793ffd3dfb7f1af30f6f7c6a4e15d7dbba52e2399 tokenizer_config.json
a5d85b6dcc535e6b93115a9ef287e6132fdbf30270da6218194ba742261173c7 tokenizer.json
7768af27c1fafa9cc9011c1dc20067e03f8915e03b63504550e11d5066986d13 video_preprocessor_config.json
ca10d7e9fb3ed18575dd1e277a2579c16d108e32f27439684afa0e10b1440910 vocab.json
EOF
echo "$REV" > "$MODEL/REVISION"
echo "all assets verified"
