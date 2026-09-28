# MMDL: 2026-2 멀티모달딥러닝 10조

Qwen3-VL-4B-Instruct를 별도 학습 데이터로 LoRA 파인튜닝하고, 같은 조건에서 MMMU validation 성능을 비교하는 프로젝트입니다. 첫 번째 과제 보고서(MMMU val baseline 평가)는 [`assignment/assignment1.md`](assignment/assignment1.md)입니다.

> **데이터 분리:** MMMU와 MMMU-Pro의 모든 split은 평가 전용입니다. 학습에 사용하지 않습니다. `code/train/train_lora.py`는 평가 데이터 폴더와 MMMU 이름의 입력을 거부합니다.

## 저장소 구성

| 경로 | 내용 |
|---|---|
| `assignment/` | 과제 보고서 (`assignment1.md`: MMMU val baseline 평가) |
| `Dockerfile` | CUDA 12.8, Python 3.12, 고정 패키지, 모델·평가 데이터가 포함된 이미지 |
| `code/eval/` | 보고서의 고정 프롬프트 기반 vLLM 추론, MMMU 채점, 검증 코드 |
| `code/eval/mmmu_data.py` | 지정 revision의 30개 과목 로딩·문항 수 확인·이미지 저장 |
| `code/train/` | Qwen 공식 멀티모달 trainer를 호출하는 LoRA 학습·병합 코드 |
| `code/scripts/` | 에셋 다운로드 및 한 명령 평가 실행 |
| `configs/evaluation.toml` | 제출 기준 512토큰 설정 |
| `configs/rtx4090.toml` | LoRA 학습 설정 |
| `code/third_party/` | 공식 이미지 입력 처리 함수·참조 파서와 출처 |
| `results/` | 실험 설정·원문 응답·채점 결과 (Git 제외) |

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

표의 경로는 기본 프로젝트 위치 기준이며 모두 컨테이너 내부 경로입니다. `CONTAINER_ROOT`를 바꾸면 생략한 경로도 그 아래를 기준으로 정합니다. 기준 설정은 greedy, seed 0, 최대 512 생성 토큰, 이미지당 262144~2097152픽셀입니다. 파인튜닝 전후 비교에는 같은 설정과 고정된 프롬프트를 사용합니다. `configs/rtx4090.toml`은 LoRA 학습 설정이며 평가에는 사용하지 않습니다.

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

이미지 입력 준비 함수의 출처는 `code/third_party/SOURCES.md`에 기록했습니다.

### 채점: 자체 파서 (주 지표)

점수는 자체 파서로 계산하고, [MMMU 공개 파서](https://github.com/MMMU-Benchmark/MMMU/tree/268471d0d488258990025331c7528359c324aa25/mmmu/utils)의 결과는 참고 지표로 함께 기록합니다. 답 추출은 `code/eval/answer_extraction.py`, 정답 비교와 집계는 `code/eval/score_mmmu.py`가 맡습니다. 자체 파서는 표준 라이브러리만 쓰고, 채점 전체가 GPU 없이 CPU에서 돌아갑니다.

**원칙**
- 응답에서 **모델이 답으로 제시한 부분**을 찾아 정규화한 뒤 정답과 비교합니다(exact match). 굵은 글씨·LaTeX 표기는 먼저 정리합니다.
- 정답은 비교 방식(숫자·기호·텍스트)을 정할 때만 보고, 답을 찾는 데는 쓰지 않습니다. 다른 모델에게 정오 판단을 맡기지도 않습니다.
- **추측하지 않습니다.** 답을 찾지 못하면 오답이며 분모 900에 포함합니다. 공개 파서는 이때 선택지를 무작위로 고르므로, 두 점수가 다를 수 있습니다.

**객관식**: 아래 순서로 찾고, 앞 단계에서 찾으면 멈춥니다. 추출한 글자가 정답 글자와 같으면 정답입니다.
1. 답 선언: `Answer: C`, `The correct answer is C`, `C is correct`, `\boxed{C}`, `I would choose C`. 여러 번 답하면 마지막 답을 씁니다. 선언 바로 뒤에 글자 하나만 있으면 `Answer: c`처럼 소문자도 읽습니다.
2. 첫머리 선택지: `C. …`로 시작하는 응답(선택지 목록을 옮겨 적은 경우는 제외)이나 글자 하나뿐인 응답.
3. 체크 표시(✅ 등)나 `correct`가 붙은 선택지 줄. 그다음 응답 전체, 마지막 문단 순서로 선택지 내용이 하나만 언급됐는지 봅니다.
4. 문장 속에 따로 쓰인 B–H 중 유일한 글자.

**주관식**: 정답의 형식으로 비교 방식을 정합니다. 답은 마지막 `\boxed{…}`, 마지막 `Answer:` 뒤 순서로 찾고, 없으면 숫자·기호형은 끝 문장부터, 텍스트형은 첫 문장과 마지막 문장에서 찾습니다.

| 정답 형식 | 예 | 비교 규칙 |
|---|---|---|
| 숫자 | `Answer: 1/2` → 0.5 | 분수·지수·쉼표·`%`·million 등을 수치로 바꾸고, 정답과 함께 소수 둘째 자리까지 반올림해 비교 |
| 한 글자 기호 | `Answer: c` → C | 대문자로 바꿔 정답 기호와 비교 |
| 텍스트 | `Answer: Paris` → paris | 대소문자·공백을 정리한 뒤 정답 구문이 단어 단위로 들어 있는지 확인 |

정답이 여러 개로 주어진 문항(예: `['24/7', '3.429']`)은 그중 하나와 일치하면 정답입니다.

**출력 한도(512토큰)에 걸려 잘린 응답**: 모델이 명시한 답(답 선언, `\boxed{}`, 첫머리 선택지)만 인정합니다. 객관식 3·4단계와 주관식의 문장 추정은 적용하지 않습니다. 풀이 도중에 끊긴 문장에서 답을 지어내지 않기 위해서입니다.

**결과 파일** (`results/<run-name>/`)
- `scores.json`: 전체·객관식·주관식·분야별·과목별 정확도, 추출 실패 건수, 규칙별 추출 건수, 공개 파서 점수와 무작위 선택 횟수, 출력 길이·한도 도달 건수
- `per_sample.csv`: 문항별 정답, 추출한 답, 적용 규칙, 정오, 응답 앞부분
- `mc_disagreements.csv`: 두 파서가 다르게 읽은 객관식 문항과 응답 원문 (파서 점검용)

**다시 채점하기**: 추론 결과의 `predictions.jsonl`과 `evaluation_data.json`만 있으면 CPU로 다시 채점할 수 있습니다. 채점 규칙을 바꾸면 비교할 모든 실행을 같은 규칙으로 다시 채점합니다.

```bash
CUDA_VISIBLE_DEVICES= python code/eval/score_mmmu.py results/<run-name>
```

기준 실행 `results/evaluation_20260928_173159`의 점수는 자체 파서 **51.44% (463/900)**, 공개 파서 51.56% (464/900)입니다(`results/README.md`). 파서의 응답 형식별 동작은 `code/eval/tests/test_answer_extraction.py`로 확인합니다.

### 검증과 기록

기존 CPU 검증은 `python code/eval/tests/test_configuration.py`, `python code/eval/tests/test_answer_extraction.py`, `python code/eval/tests/test_prompting.py`로 반복할 수 있습니다. 프롬프트 점검은 `python code/eval/check_prompts.py --model models/Qwen3-VL-4B-Instruct`입니다. 새 실험은 `results/README.md`의 표에 추가하고 제출 보고서의 실측 칸을 채우세요.

협업은 브랜치와 PR을 사용하며 최종 결과물은 `main`에 병합합니다.
