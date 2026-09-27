# MMDL: 2026-2 멀티모달딥러닝 10조

Qwen3-VL-4B-Instruct를 별도 학습 데이터로 LoRA 파인튜닝하고, 같은 조건에서 MMMU validation 성능을 비교하는 프로젝트입니다. 제출 보고서의 항목은 [`reports/mmmu_baseline.md`](reports/mmmu_baseline.md)에 맞췄습니다.

> **데이터 분리:** MMMU와 MMMU-Pro의 모든 split은 평가 전용입니다. 학습에 사용하지 않습니다. `code/train/train_lora.py`는 평가 데이터 폴더와 MMMU 이름의 입력을 거부합니다.

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `Dockerfile` | CUDA 12.8, Python 3.12, 고정 패키지, 모델·평가 데이터가 포함된 이미지 |
| `code/eval/` | 보고서의 고정 프롬프트 기반 vLLM 추론, MMMU 채점, 검증 코드 |
| `code/eval/mmmu_data.py` | 지정 revision의 30개 과목 로딩·문항 수 확인·이미지 저장 |
| `code/train/` | Qwen 공식 멀티모달 trainer를 호출하는 LoRA 학습·병합 코드 |
| `code/scripts/` | 에셋 다운로드 및 한 명령 평가 실행 |
| `configs/evaluation.toml` | 제출 기준 512토큰 설정 |
| `configs/rtx4090.toml` | 2048토큰 추가 실험·LoRA 설정 |
| `code/third_party/` | 공식 이미지 입력 처리 함수·참조 파서와 출처 |
| `results/` | 실험 설정·원문 응답·채점 결과 (Git 제외) |
| `reports/` | 제출 보고서·요약 증거·과제 원본 양식 |

## 재현 환경

- **모델:** [`Qwen/Qwen3-VL-4B-Instruct`](https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct) @ `ebb281ec70b05090aa6165b016eac8ec08e71b17`, BF16
- **추론:** vLLM `0.11.2`, transformers `4.57.6`, RTX 4090 24 GB 한 장 기준
- **학습:** [Qwen3-VL 공식 멀티모달 trainer](https://github.com/QwenLM/Qwen3-VL/tree/96588727e44c78b25ba03ea03b8e12f7e64fd0da/qwen-vl-finetune) @ `96588727e44c78b25ba03ea03b8e12f7e64fd0da`, LoRA, BF16, SDPA. `code/train/patch_upstream.py`는 FlashAttention을 SDPA 경로에서 가져오지 않도록 import 한 곳만 지연시킵니다.
- **MMMU:** 30개 과목 config를 개별 로드한 validation 900문항; [Hugging Face 고정 revision](https://huggingface.co/datasets/MMMU/MMMU/tree/98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68) @ `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`
- **MMMU-Pro:** [전체 snapshot](https://huggingface.co/datasets/MMMU/MMMU_Pro/tree/563f3e84bb3b90893083a1f039cfa13077f2302b) @ `563f3e84bb3b90893083a1f039cfa13077f2302b` (standard 4/10 options, vision)

Docker 빌드 중 모델 약 8.9 GB, MMMU validation Parquet 약 341 MB와 MMMU-Pro snapshot 약 3.0 GB를 받습니다. MMMU는 평가와 동일한 `load_mmmu()`로 30개 config를 개별 로드합니다. 데이터와 이미지는 `/opt/mmdl/data/`, 데이터 출처 기록은 `/opt/mmdl/data/dataset_manifest.json`에 저장합니다. 새 이미지에서는 이 캐시를 재사용합니다. HF 메타데이터 조회에는 네트워크가 필요하며, 모델은 오프라인으로 읽습니다.

캐시가 없는 이미지에서는 첫 실행 때 지정 데이터를 받습니다. `DATA_PATH`는 컨테이너 안의 실제 데이터 폴더이며 기본값은 `/opt/mmdl/data`입니다. Docker 사용 호스트에는 NVIDIA 드라이버와 NVIDIA Container Toolkit이 필요합니다.

## Docker 빌드와 MMMU baseline

호스트의 프로젝트 루트에서 이미지를 한 번 빌드합니다. 모델·평가 데이터와 실행 환경이 이미지에 포함됩니다.

```bash
docker build --progress=plain -t ssu:MMDL .
```

이후 baseline 평가는 다음 명령으로 실행합니다.

```bash
bash code/scripts/run_mmmu_eval.sh
```

스크립트가 `mmdl-work` 컨테이너의 상태를 확인해 처리합니다.

| 상태 | 처리 |
|---|---|
| 컨테이너 없음 | `ssu:MMDL` 이미지로 계속 사용할 컨테이너 생성 |
| 중지됨 | `docker start` 후 평가 |
| 일시 정지됨 | `docker unpause` 후 평가 |
| 실행 중 | `docker exec`로 추론·채점 실행 |
| 이미지 없음 / Docker 연결 불가 | 이미지 빌드 또는 Docker 실행·접근 권한 확인 안내 |

컴퓨터를 재시작한 뒤에도 Docker가 실행 중이면 같은 명령을 쓰면 됩니다. 중지한 컨테이너는 스크립트가 시작합니다. 재부팅 전에 실행하던 학습·평가 작업이 자동으로 재개되는 것은 아닙니다.

컨테이너를 새로 만들 때는 호스트의 `code/`, `configs/`, `training_data/`, `checkpoints/`, `results/`를 연결합니다. 코드와 설정을 양쪽에서 수정할 수 있고, 학습 자료·체크포인트·결과도 호스트에 보관됩니다. 이미 있던 컨테이너를 지정하면 기존 폴더 연결을 그대로 사용합니다. 평가는 끝나도 컨테이너는 유지됩니다.

기본 모델·데이터·평가 설정을 그대로 쓰면 경로를 입력할 필요가 없습니다. 결과는 `results/evaluation_<실행 시각>/`에 저장합니다. 평가 경로는 CLI 인자나 환경변수로 지정합니다. 같은 항목을 둘 다 지정하면 CLI 인자가 우선합니다. 컨테이너 설정은 환경변수로 지정합니다.

| CLI 인자 | 환경변수 | 기본값 | 용도 |
|---|---|---|---|
| — | `CONTAINER_NAME` | `mmdl-work` | 재사용할 컨테이너 |
| — | `DOCKER_IMAGE` | `ssu:MMDL` | 컨테이너가 없을 때 사용할 이미지 |
| — | `CONTAINER_ROOT` | `/opt/mmdl` | 컨테이너 내부 프로젝트 위치 |
| `--model` | `MODEL_PATH` | `/opt/mmdl/models/Qwen3-VL-4B-Instruct` | 기본 모델 또는 병합한 체크포인트 |
| `--data-path` | `DATA_PATH` | `/opt/mmdl/data` | 평가 데이터 저장 위치 |
| `--config` | `CONFIG_PATH` | `/opt/mmdl/configs/evaluation.toml` | 평가 설정 파일 |
| `--run-name` | `RUN_NAME` | 실행 시각을 붙인 이름 | 결과 폴더 이름 |
| `--out-root` | `OUT_ROOT` | `/opt/mmdl/results` | 결과를 저장할 상위 폴더 |
| `--overwrite` | — | 사용 안 함 | 지정한 결과 폴더의 기존 응답 덮어쓰기 |

표의 경로는 기본 프로젝트 위치 기준이며 모두 컨테이너 내부 경로입니다. `CONTAINER_ROOT`를 바꾸면 생략한 경로도 그 아래를 기준으로 정합니다. 기준 설정은 greedy, seed 0, 최대 512 생성 토큰, 이미지당 262144~2097152픽셀입니다. 파인튜닝 전후 비교에는 같은 설정과 고정된 프롬프트를 사용합니다. `configs/rtx4090.toml`은 2048토큰 추가 실험 설정이므로 혼용하지 않습니다.

예를 들어 학습 후 병합한 모델은 호스트에서 다음과 같이 평가합니다.

```bash
bash code/scripts/run_mmmu_eval.sh \
  --model "<병합한 모델 체크포인트 경로>" \
  --run-name "<새 결과 폴더 이름>"
```

추론과 채점은 같은 `--out-root`와 `--run-name`을 사용합니다. 결과 폴더에 기존 응답이 있으면 기본적으로 종료하며, `--overwrite`를 추가한 경우에만 덮어씁니다. 새 이름을 쓰거나 `--run-name`과 `RUN_NAME`을 모두 생략하면 이전 결과를 보존할 수 있습니다. `--out-root`를 바꿀 때는 컨테이너 내부 경로를 지정합니다. 호스트에서도 결과를 보려면 호스트 폴더와 연결된 경로를 사용하세요.

`--model PATH`와 `--model=PATH` 형식을 모두 지원하며, 전체 사용법은 `bash code/scripts/run_mmmu_eval.sh --help`로 확인합니다. 생성 설정은 `--config`로 지정한 TOML에서 읽습니다.

## 별도 데이터로 LoRA 학습

학습 데이터는 아직 확정되지 않았습니다. 아래 형식의 **MMMU와 무관한** JSONL과 실제 이미지 파일을 준비합니다. 각 줄은 Qwen 공식 학습 형식입니다. 이미지가 두 장이면 `<image>` 표기도 두 개여야 합니다.

```json
{"image":"images/example.jpg","conversations":[{"from":"human","value":"<image>\nWhat is shown?"},{"from":"gpt","value":"A chart."}],"source":"my_training_set"}
```

호스트에서 다음 명령으로 컨테이너 셸에 접속합니다. 셸에서 나와도 컨테이너는 계속 실행됩니다.

```bash
docker exec -it --workdir /opt/mmdl mmdl-work bash
```

컨테이너 내부에서 각자 준비한 학습 데이터로 실행합니다.

```bash
python code/train/train_lora.py \
  --model /opt/mmdl/models/Qwen3-VL-4B-Instruct \
  --annotation /opt/mmdl/training_data/train.jsonl \
  --image-root /opt/mmdl/training_data \
  --output-dir /opt/mmdl/checkpoints/lora
```

학습은 Qwen 공식 trainer의 LoRA 경로를 사용합니다. 시작값은 배치 1, 누적 8, 1 epoch, 학습률 `1e-4`, 이미지당 65,536~524,288픽셀, 최대 길이 2048입니다. 학습의 역전파 메모리가 추론보다 크므로 학습 해상도 기본값을 낮게 두었습니다. GPU 학습 스모크 테스트에는 설정 파일 복사본의 `max_steps = 1`을 사용합니다. `--validate-only`는 데이터 형식과 이미지 경로를 GPU 없이 점검합니다.

같은 컨테이너에서 LoRA를 기본 모델과 병합하고 vLLM으로 평가합니다. 컨테이너 안에서 평가 스크립트를 호출하면 추론과 채점을 바로 실행합니다.

```bash
python code/train/merge_lora.py \
  --base-model /opt/mmdl/models/Qwen3-VL-4B-Instruct \
  --adapter /opt/mmdl/checkpoints/lora \
  --output /opt/mmdl/checkpoints/merged

bash code/scripts/run_mmmu_eval.sh \
  --model "<병합한 모델 체크포인트 경로>" \
  --data-path "<MMMU 데이터 경로>" \
  --config "<평가 설정 파일 경로>" \
  --run-name "<새 결과 폴더 이름>"
```

평가 설정은 baseline과 같고 평가 대상 모델 경로와 결과 저장 이름만 바꿉니다.

병합 단계는 CPU 메모리에 기본 모델을 올리므로 충분한 호스트 RAM이 필요합니다. `checkpoints/`, `training_data/`, `data/`, `models/`는 Git에 올리지 않습니다.

## 포함된 MMMU-Pro 데이터 사용

이미지에는 MMMU-Pro의 세 구성 모두가 `/opt/mmdl/data/hf/MMMU_Pro`에 들어갑니다. 예를 들어 다음 명령은 인터넷 없이 standard 10 options의 test split을 읽습니다.

```bash
docker exec mmdl-work python -c 'from datasets import load_dataset; d=load_dataset("parquet", data_files={"test":"/opt/mmdl/data/hf/MMMU_Pro/standard (10 options)/test-*.parquet"}, split="test"); print(len(d), d.column_names)'
```

MMMU 평가는 `MMMU/MMMU`의 고정 revision에서 validation 30개 config를 개별 로드합니다. 과목당 30개·총 900개·ID 중복 없음을 확인하며, 채점은 추론할 때 저장한 `evaluation_data.json`의 정답과 선택지를 사용합니다. 데이터 소스 선택이나 부분 평가 옵션은 없습니다. MMMU-Pro 평가는 별도 프로토콜을 정한 뒤 수행합니다.

## 평가와 제출 기록

보고서의 객관식·주관식 프롬프트를 `code/eval/prompting.py`에 고정했습니다. 이미지를 원래 순서대로 텍스트 앞에 넣고, 객관식에는 `Answer with the option letter only.`를 사용합니다. 주관식에는 한 줄 `Answer: <your short answer>`를 요구하고 풀이 출력을 금지합니다. 프롬프트 선택 인자는 없습니다.

주 지표는 자체 파서이며 [MMMU 공개 파서](https://github.com/MMMU-Benchmark/MMMU/tree/268471d0d488258990025331c7528359c324aa25/mmmu/utils)는 참고 지표입니다. 추출 실패는 주 지표에서 오답으로 처리합니다. 이미지 입력 준비 함수의 출처는 `code/third_party/SOURCES.md`에 기록했습니다.

기존 CPU 검증은 `python code/eval/tests/test_configuration.py`, `python code/eval/tests/test_answer_extraction.py`, `python code/eval/tests/test_prompting.py`로 반복할 수 있습니다. 프롬프트 점검은 `python code/eval/check_prompts.py --model models/Qwen3-VL-4B-Instruct`입니다. 새 실험은 `results/README.md`의 표에 추가하고 제출 보고서의 실측 칸을 채우세요. 과제 원본 양식은 `reports/references/`에 보존했습니다.

협업은 브랜치와 PR을 사용하며 최종 결과물은 `main`에 병합합니다.
