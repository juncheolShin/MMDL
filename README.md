# MMDL: 2026-2 멀티모달딥러닝 10조

Qwen3-VL-4B를 fine-tuning해서 MMMU 성능을 높이는 팀 프로젝트입니다.

- **Base model**: Qwen3-VL-4B (`Qwen/Qwen3-VL-4B-Instruct`)
- **Validation**: MMMU val / **Final evaluation**: MMMU test + MMMU-Pro
- **학습 데이터**: MMMU와 MMMU-Pro를 제외한 모든 소스

> ⚠️ MMMU·MMMU-Pro 데이터는 어떤 split이든 **학습에 사용 금지**입니다. `data/`는 평가 전용이며 git에 올리지 않습니다.

## 저장소 구조

```
MMDL/
├── assignment/          과제 (assignment1.md, …)
├── code/
│   ├── eval/            평가 코드: 추론, 채점, 답 추출기, 프롬프트 점검, 테스트
│   ├── third_party/     Qwen3-VL·MMMU 공식 평가 코드 (커밋 고정, 수정 없음)
│   ├── scripts/         fetch_assets.sh: 데이터·모델을 고정 버전으로 받고 SHA-256 검증
│   └── requirements-eval.txt
├── results/             실험 설정과 결과 (results/README.md)
└── README.md
```

`data/`와 `models/`는 `code/scripts/fetch_assets.sh`가 저장소 루트에 만들며, git에는 올리지 않습니다.

| 양식의 필수 항목 | 위치 | 상태 |
|---|---|---|
| Assignment | `assignment/` | 과제 명세를 받으면 작성 |
| Evaluation code | `code/eval/` | 작성 완료. CPU 검증까지 마쳤고 GPU 실행은 아직 |
| Fine-tuning code | `code/` | 아직 없음 |
| Experimental settings and results | `results/` | 평가 설정과 파이프라인 검증 기록. baseline 결과는 아직 |
| README.md with instructions | `README.md` | 이 문서 |

## 재현 방법

모든 명령은 저장소 루트에서 실행합니다.

### 1. 환경

```bash
conda create -n mmdl python=3.12 -y
conda activate mmdl
pip install -r code/requirements-eval.txt
```

vLLM 0.11.2를 사용합니다(torch 2.9.0+cu128이 함께 설치됨). 이 설정은 RTX 4090(24 GB)을 기준으로 잡았습니다.

### 2. 데이터와 모델

```bash
bash code/scripts/fetch_assets.sh     # data/ (MMMU val TSV 81 MB, 공식 정답), models/ (8.9 GB)
```

파일 15개를 모두 SHA-256으로 검증합니다. 이미 받은 파일은 건너뜁니다.

### 3. GPU 없이 하는 점검

```bash
python code/eval/tests/test_answer_extraction.py                         # 응답 형식별 추출기 테스트
python code/eval/tests/check_official_parity.py                          # MMMU 공식 스크립트와 대조
CUDA_VISIBLE_DEVICES= python code/eval/check_prompts.py --model models/Qwen3-VL-4B-Instruct --show validation_Art_8
```

### 4. 평가 (GPU)

```bash
# 스모크 테스트: 슬라이드 예시, 선택지 복원 문항, 최장 프롬프트, 이미지 5장, 주관식(숫자·라벨)
CUDA_VISIBLE_DEVICES=0 python code/eval/infer_mmmu.py --model models/Qwen3-VL-4B-Instruct --run-name smoke --decoding greedy \
  --ids validation_Art_8 validation_Geography_15 validation_Music_21 validation_Psychology_17 validation_Math_15 validation_Basic_Medical_Science_10
python code/eval/score_mmmu.py results/smoke

# MMMU val 전체
CUDA_VISIBLE_DEVICES=0 python code/eval/infer_mmmu.py --model models/Qwen3-VL-4B-Instruct --run-name base_greedy --decoding greedy
python code/eval/score_mmmu.py results/base_greedy
```

Fine-tuned 체크포인트는 `--model`에 체크포인트 디렉터리를 주면 됩니다(LoRA는 병합한 뒤). `--run-name`만 바꾸면 같은 설정으로 평가됩니다.

### 5. 결과 기록

`results/<run-name>/`에 `run_config.json`, `scores.json`, `per_sample.csv`, `mc_disagreements.csv`가 생깁니다. 이 파일들을 커밋하고, `results/README.md`의 실험 기록 표에 한 줄을 추가합니다. `predictions.jsonl`은 MMMU 질문이 들어 있어서 `.gitignore`로 제외됩니다.

실행 뒤에는 `scores.json`의 `structured.unparsed`와 `method_counts`, 그리고 `mc_disagreements.csv`로 답 추출이 제대로 되었는지 확인합니다.

## 평가 프로토콜 요약

$$r_n = f_\theta\big(p(q_n, i_n)\big), \qquad s = \frac{1}{N}\sum_n g\big(r_{\text{gt},n},\ \hat r_n\big)$$

- **p**: Qwen3-VL 공식 MMMU 프롬프트(강의 p.10과 동일)에 모델의 ChatML chat template을 적용합니다.
- **g**: 응답의 최종 답 선언(`\boxed{}`, "The correct answer is …", "Answer: …" 등)에서 답을 읽고, 정규화한 뒤 exact match로 비교합니다. 무작위로 찍지 않으며, 답을 못 찾으면 오답으로 처리하고 건수를 따로 셉니다.
  - 주관식은 정답 형식(숫자, 라벨 문자, 텍스트)에 맞춰 비교합니다.
  - MMMU 공식 파서 결과도 참고 지표로 함께 기록합니다.
- 세부 설정과 검증 기록은 [`results/README.md`](results/README.md), 공식 코드 출처와 수정 사항은 [`code/third_party/SOURCES.md`](code/third_party/SOURCES.md)에 있습니다.

## 협업 규칙

- 커밋과 pull request는 **각자 본인 GitHub 계정**으로 합니다.
- 작업은 브랜치에서 하고, PR로 기본 브랜치(`main`)에 merge합니다. **최종 결과물은 `main`에 있어야 합니다.**
- 새 실험은 `results/`에 run 폴더와 실험 기록 표 한 줄을 함께 커밋해서, 저장소만으로 재현할 수 있게 합니다.
