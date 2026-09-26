# MMDL: 2026-2 멀티모달딥러닝 10조

Qwen3-VL-4B-Instruct를 별도 학습 데이터로 LoRA 파인튜닝하고, 같은 조건에서 MMMU validation 성능을 비교하는 프로젝트입니다. 제출 보고서의 항목은 [`results/BASELINE_REPORT.md`](results/BASELINE_REPORT.md)에 맞췄습니다.

> **데이터 분리:** MMMU와 MMMU-Pro의 모든 split은 평가 전용입니다. 학습에 사용하지 않습니다. `code/train/train_lora.py`는 평가 데이터 폴더와 MMMU 이름의 입력을 거부합니다.

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `Dockerfile` | CUDA 12.8, Python 3.12, 고정 패키지, 모델·평가 데이터가 포함된 이미지 |
| `code/eval/` | 회의 합의안 프롬프트 기반 vLLM 추론, MMMU 채점, 검증 코드 |
| `code/train/` | Qwen 공식 멀티모달 trainer를 호출하는 LoRA 학습·병합 코드 |
| `code/scripts/` | 에셋 다운로드 및 한 명령 평가 실행 |
| `configs/rtx4090.toml` | 추론·LoRA 실험 파라미터의 단일 설정 파일 |
| `code/third_party/` | 변경하지 않은 공식 Qwen/MMMU 평가 코드와 출처 |
| `results/` | 실험 설정, 결과, 제출 보고서 양식 |

## 재현 환경

- **모델:** [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) @ `ebb281ec70b05090aa6165b016eac8ec08e71b17`, BF16
- **추론:** vLLM `0.11.2`, transformers `4.57.6`, RTX 4090 24 GB 한 장 기준
- **학습:** [Qwen3-VL 공식 멀티모달 trainer](https://github.com/QwenLM/Qwen3-VL/tree/96588727e44c78b25ba03ea03b8e12f7e64fd0da/qwen-vl-finetune) @ `96588727e44c78b25ba03ea03b8e12f7e64fd0da`, LoRA, BF16, SDPA. `code/train/patch_upstream.py`는 FlashAttention을 SDPA 경로에서 가져오지 않도록 import 한 곳만 지연시킵니다.
- **MMMU:** 평가용 `MMMU_DEV_VAL.tsv`와 공식 val 정답; [Hugging Face 전체 snapshot](https://huggingface.co/datasets/MMMU/MMMU/tree/98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68) @ `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`
- **MMMU-Pro:** [전체 snapshot](https://huggingface.co/datasets/MMMU/MMMU_Pro/tree/563f3e84bb3b90893083a1f039cfa13077f2302b) @ `563f3e84bb3b90893083a1f039cfa13077f2302b` (standard 4/10 options, vision)

Docker 빌드 중 모델 약 8.9 GB, MMMU snapshot 약 3.7 GB, MMMU-Pro snapshot 약 3.0 GB와 평가용 TSV를 **이미지 안에 다운로드**합니다. 빌드가 완료된 이미지는 해당 에셋을 다시 받지 않고 오프라인으로 실행할 수 있습니다. 이미지 저장 공간과 빌드 중 임시 공간을 넉넉히 확보하세요. Docker 사용 호스트에는 NVIDIA 드라이버와 NVIDIA Container Toolkit이 필요합니다.

## Docker 빌드와 MMMU baseline

저장소 루트에서 실행합니다. 빌드 자체에는 GPU가 필요하지 않지만, 추론과 학습에는 CUDA GPU가 필요합니다.

```bash
docker build --progress=plain -t mmdl:qwen3vl .

# 여섯 문항 스모크 테스트
docker run --rm --gpus all --shm-size=16g \
  -v "$PWD/results:/opt/mmdl/results" \
  -e RUN_NAME=smoke mmdl:qwen3vl \
  bash code/scripts/run_mmmu_eval.sh --ids validation_Art_8 validation_Geography_15 validation_Music_21 validation_Psychology_17 validation_Math_15 validation_Basic_Medical_Science_10

# MMMU validation 900문항 및 채점: 추론과 채점을 한 명령으로 실행
docker run --rm --gpus all --shm-size=16g \
  -v "$PWD/results:/opt/mmdl/results" \
  -e RUN_NAME=base_greedy mmdl:qwen3vl bash code/scripts/run_mmmu_eval.sh
```

`MODEL_PATH`, `DATA_ROOT`, `OUT_ROOT`, `RUN_NAME` 환경변수로 경로를 바꿀 수 있고, 추가 인자는 `infer_mmmu.py`에 전달됩니다. `DATA_ROOT`에는 `MMMU_DEV_VAL.tsv`와 `mmmu_answer_dict_val.json`이 있어야 합니다. 이미지 안의 기본 경로는 각각 `/opt/mmdl/models/Qwen3-VL-4B-Instruct`, `/opt/mmdl/data`입니다. 외부 모델·데이터 폴더를 쓰려면 각각 별도 볼륨으로 연결하세요.

추론·학습 숫자 설정은 [`configs/rtx4090.toml`](configs/rtx4090.toml)에서 관리합니다. 기본값은 **greedy, seed 0, 최대 512 생성 토큰, 이미지당 262,144~2,097,152픽셀, max_model_len 8192, max_num_seqs 2, GPU 메모리 사용 목표 0.85**입니다. 다중 이미지 문항에서 메모리가 부족하면 설정 파일을 복사한 뒤 `max_num_seqs = 1`로 바꾸고 새 run 이름을 쓰세요. Qwen 샘플링 조건을 비교하려면 복사본의 `temperature = 0.7`을 비롯한 생성 설정을 조정합니다.

이미 빌드한 이미지에서 설정만 바꿀 때는 수정한 TOML을 볼륨으로 연결하고 `CONFIG_PATH`를 지정합니다. 학습 명령에는 `--config`로 같은 경로를 줍니다. 결과 파일에는 **사용한 값과 설정 파일 SHA-256**이 남습니다.

```bash
cp configs/rtx4090.toml configs/rtx4090_seq1.toml
# rtx4090_seq1.toml의 max_num_seqs를 1로 수정
docker run --rm --gpus all --shm-size=16g \
  -v "$PWD/configs:/workspace/configs:ro" \
  -v "$PWD/results:/opt/mmdl/results" \
  -e CONFIG_PATH=/workspace/configs/rtx4090_seq1.toml -e RUN_NAME=base_seq1 \
  mmdl:qwen3vl bash code/scripts/run_mmmu_eval.sh
```

`results/<run-name>/`에는 설정 `run_config.json`, 원문 출력 `predictions.jsonl`, `scores.json`, 문항별 `per_sample.csv`, 파서 차이 `mc_disagreements.csv`가 생깁니다. 원문에는 문제 내용이 들어 있으므로 Git에서 제외됩니다. 실제 GPU 시간·VRAM과 결과 수치는 실행 후 제출 보고서에 기록하세요.

## 별도 데이터로 LoRA 학습

학습 데이터는 아직 확정되지 않았습니다. 아래 형식의 **MMMU와 무관한** JSONL과 실제 이미지 파일을 준비합니다. 각 줄은 Qwen 공식 학습 형식입니다. 이미지가 두 장이면 `<image>` 표기도 두 개여야 합니다.

```json
{"image":"images/example.jpg","conversations":[{"from":"human","value":"<image>\nWhat is shown?"},{"from":"gpt","value":"A chart."}],"source":"my_training_set"}
```

```bash
# 호스트: training_data/train.jsonl 및 training_data/images/...
mkdir -p checkpoints
docker run --rm --gpus all --shm-size=16g \
  -v "$PWD/training_data:/workspace/train:ro" \
  -v "$PWD/checkpoints:/workspace/checkpoints" \
  mmdl:qwen3vl python code/train/train_lora.py \
  --annotation /workspace/train/train.jsonl \
  --image-root /workspace/train \
  --output-dir /workspace/checkpoints/lora_run
```

학습은 Qwen 공식 trainer의 LoRA 경로를 사용합니다. 시작값은 배치 1, 누적 8, 1 epoch, 학습률 `1e-4`, 이미지당 65,536~524,288픽셀, 최대 길이 2048입니다. 학습의 역전파 메모리가 추론보다 크므로 학습 해상도 기본값을 낮게 두었습니다. GPU 학습 스모크 테스트에는 설정 파일 복사본의 `max_steps = 1`을 사용합니다. `--validate-only`는 데이터 형식과 이미지 경로를 GPU 없이 점검합니다.

vLLM 평가에는 LoRA를 병합한 모델을 사용합니다.

```bash
docker run --rm -v "$PWD/checkpoints:/workspace/checkpoints" \
  mmdl:qwen3vl python code/train/merge_lora.py \
  --base-model /opt/mmdl/models/Qwen3-VL-4B-Instruct \
  --adapter /workspace/checkpoints/lora_run \
  --output /workspace/checkpoints/merged

docker run --rm --gpus all --shm-size=16g \
  -v "$PWD/checkpoints:/workspace/checkpoints:ro" \
  -v "$PWD/results:/opt/mmdl/results" \
  -e MODEL_PATH=/workspace/checkpoints/merged -e RUN_NAME=lora_greedy \
  mmdl:qwen3vl bash code/scripts/run_mmmu_eval.sh
```

병합 단계는 CPU 메모리에 기본 모델을 올리므로 충분한 호스트 RAM이 필요합니다. `checkpoints/`, `training_data/`, `data/`, `models/`는 Git에 올리지 않습니다.

## 포함된 MMMU-Pro 데이터 사용

이미지에는 MMMU-Pro의 세 구성 모두가 `/opt/mmdl/data/hf/MMMU_Pro`에 들어갑니다. 예를 들어 다음 명령은 인터넷 없이 standard 10 options의 test split을 읽습니다.

```bash
docker run --rm mmdl:qwen3vl python -c 'from datasets import load_dataset; d=load_dataset("parquet", data_files={"test":"/opt/mmdl/data/hf/MMMU_Pro/standard (10 options)/test-*.parquet"}, split="test"); print(len(d), d.column_names)'
```

원본 MMMU snapshot은 `/opt/mmdl/data/hf/MMMU`에 있습니다. 현재 자동 추론·채점 파이프라인은 **제출 템플릿의 MMMU validation**을 대상으로 합니다. MMMU-Pro의 세 구성은 이미지에 포함되고 로드 가능하며, MMMU-Pro 점수는 별도 평가 프로토콜을 정한 뒤 기록합니다.

## 평가와 제출 기록

프롬프트는 [2026-09-25 회의 메모](https://app.notion.com/p/MMDL-10-3e2877ea237c80e280a8e48b14a49159)의 상단 합의안을 `code/eval/prompting.py`에 구현했습니다. 이미지를 원래 순서대로 텍스트 앞에 넣습니다. 객관식은 `Question: ...`, `Options:`와 실제 존재하는 선택지를 보여준 뒤 `Answer with the option letter only.`를 붙입니다. 주관식은 원문 질문 뒤에 `Final answer: <your short answer>` 형식을 요구합니다. [Qwen 공식 MMMU 코드](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/run_mmmu.py)의 vLLM 입력 준비 함수는 유지합니다. 주 지표는 단일 글자와 최종 답 선언을 모두 읽는 structured 파서이고, [MMMU 공식 파서](https://github.com/MMMU-Benchmark/MMMU/tree/268471d0d488258990025331c7528359c324aa25/mmmu/utils)는 참고 지표입니다. 실패한 추출은 주 지표에서 오답으로 처리하며, 공식 파서의 무작위 추측 횟수는 따로 기록합니다.

기존 CPU 검증은 `python code/eval/tests/test_configuration.py`, `python code/eval/tests/test_answer_extraction.py`, `python code/eval/tests/test_prompting.py`, `python code/eval/tests/check_official_parity.py`로 반복할 수 있습니다. 프롬프트 점검은 `python code/eval/check_prompts.py --model models/Qwen3-VL-4B-Instruct`입니다. 새 실험은 `results/README.md`의 표에 추가하고 제출 보고서의 실측 칸을 채우세요. 과제의 `assignment/`는 공식 명세를 받으면 추가합니다.

협업은 브랜치와 PR을 사용하며 최종 결과물은 `main`에 병합합니다.
