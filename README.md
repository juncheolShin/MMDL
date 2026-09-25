# 2026-2 멀티모달딥러닝 10조: MMMU 평가 파이프라인

Qwen3-VL-4B-Instruct(및 fine-tuned 체크포인트)를 **MMMU val** 900문항으로 평가합니다.

$$r_n = f_\theta\big(p(q_n, i_n)\big), \qquad s = \frac{1}{N}\sum_n g\big(r_{\text{gt},n},\ \hat r_n\big),\quad g=\text{정규화 후 exact match}$$

> ⚠️ MMMU·MMMU-Pro 데이터는 어떤 split이든 **학습에 사용 금지**입니다. `data/`는 평가 전용이며 git에 올리지 않습니다.

## 구성

| 경로 | 내용 |
|---|---|
| `eval/infer_mmmu.py` | vLLM 추론 → `runs/<run>/predictions.jsonl`, `run_config.json`(설정·모델/데이터 해시·패키지 버전) |
| `eval/score_mmmu.py` | 채점 → `scores.json`, `per_sample.csv`, `mc_disagreements.csv` |
| `eval/answer_extraction.py` | 응답 → 정규화된 답 r̂ 추출기 (주 지표) |
| `eval/check_prompts.py` | 모델 없이(CPU) 프롬프트 렌더링·토큰 길이 점검 |
| `eval/mmmu_data.py` | 데이터 로더 (Qwen 로더 + pandas `"None"` 결측 처리 수정) |
| `eval/tests/` | 응답 형식별 추출기 테스트 |
| `third_party/` | Qwen3-VL 공식 MMMU 평가 코드, MMMU 공식 파서 (커밋 고정, 수정 없음; `SOURCES.md` 참고) |
| `scripts/fetch_assets.sh` | 데이터·모델을 고정 버전으로 받고 SHA-256 검증 |

## 설치

```bash
conda create -n mmdl python=3.12 -y
conda activate mmdl
pip install -r requirements-eval.txt
bash scripts/fetch_assets.sh        # data/ (TSV 81 MB, 정답), models/ (8.9 GB)
python eval/tests/test_answer_extraction.py
```

## 평가 절차

```bash
cd eval
# 0) 프롬프트 점검 (GPU 불필요)
CUDA_VISIBLE_DEVICES= python check_prompts.py --model ../models/Qwen3-VL-4B-Instruct --show validation_Art_8

# 1) 스모크 테스트: 슬라이드 예시, 선택지 복원 문항, 최장 프롬프트, 5장 이미지, 주관식(숫자·라벨)
CUDA_VISIBLE_DEVICES=0 python infer_mmmu.py --model ../models/Qwen3-VL-4B-Instruct --run-name smoke --decoding greedy \
  --ids validation_Art_8 validation_Geography_15 validation_Music_21 validation_Psychology_17 validation_Math_15 validation_Basic_Medical_Science_10
python score_mmmu.py ../runs/smoke

# 2) 전체 val
CUDA_VISIBLE_DEVICES=0 python infer_mmmu.py --model ../models/Qwen3-VL-4B-Instruct --run-name base_greedy --decoding greedy
CUDA_VISIBLE_DEVICES=0 python infer_mmmu.py --model ../models/Qwen3-VL-4B-Instruct --run-name base_qwen_official --decoding qwen_official
python score_mmmu.py ../runs/base_greedy
python score_mmmu.py ../runs/base_qwen_official
```

Fine-tuned 체크포인트는 `--model`에 (LoRA라면 병합한) 체크포인트 디렉토리를 주면 같은 조건으로 평가됩니다.

## 평가 프로토콜 (베이스라인과 모든 체크포인트에 동일 적용)

- **프롬프트 p**: Qwen3-VL 공식 `evaluation/mmmu`의 `build_mmmu_prompt`를 그대로 사용합니다. 모델의 chat template(ChatML)으로 감싸고, 이미지를 텍스트 앞에 둡니다. 강의 p.10의 프롬프트와 동일합니다.
  ```
  Question: <image 1>: A Conversation with the Sea stands in which British seaside town?
  Options:
  A. Herne Bay, Kent
  ...
  Please select the correct answer from the options above.
  ```
  이미지는 Qwen 설정대로 1.0M–4.0M 픽셀로 조정합니다(val 이미지의 90%가 확대됨). 프롬프트 길이는 1,015–5,627 토큰입니다.
- **디코딩**: `greedy`(temperature 0, 비교용 기본)와 `qwen_official`(T=0.7, top-p 0.8, top-k 20, presence 1.5; Qwen 보고 수치 재현용) 두 가지입니다. 둘의 차이는 temperature 하나뿐입니다. 공통으로 `max_new_tokens` 32768, `max_model_len` 40960을 씁니다(Qwen 원본 128000은 24 GB GPU에 맞지 않음).
- **추출·채점 g (주 지표: `structured`)**: 응답의 최종 답 선언(`\boxed{}`, "The correct answer is …", "Answer: …", "Option C is correct" 등)을 찾아 그 위치에서 답을 읽습니다. 선언이 여러 번이면 마지막 것을 씁니다. 답을 못 찾으면 **무작위로 찍지 않고 오답**으로 처리하며, 그 건수를 기록합니다.
  - 객관식: 선택지 문자 일치
  - 주관식 숫자: 분수·쉼표·`million`·`×10^n`·`%`를 해석한 뒤 소수 둘째 자리 반올림으로 비교(MMMU 공식 규칙)
  - 주관식 라벨(정답이 문자 하나): "arrow labeled C", "Step b" 같은 표현에서 문자를 읽음
  - 주관식 텍스트: 답 문장에 정답 구절이 포함되는지 확인
- **참고 지표 (`official`)**: MMMU 공식 파서입니다. 짧은 "문자만" 응답을 전제로 만들어져서, 긴 설명형 응답에서는 관사 "A"를 선택지로 읽거나, 언급된 다른 선택지에 매칭하거나, 무작위로 찍습니다. 문헌 수치와 비교할 때만 씁니다.

### 실행 후 반드시 확인할 것
- `scores.json`의 `structured.unparsed`와 `method_counts`: 추출기가 어떤 규칙으로 답을 읽었는지
- `mc_disagreements.csv`: 두 추출기가 다르게 읽은 문항과 원문. 추출기가 실제 출력 형식을 제대로 읽는지 여기서 감사합니다.
- `generation.finish_reason`의 `length`(출력 한도에서 잘림) 건수

## 검증 기록
- 공식 파서 재현: MMMU 저장소의 LLaVA-1.5 응답 900개에서 `main_parse_and_eval.py`와 **900/900 판정 일치**
- 짧은 응답에서 `structured`와 공식 파서가 같은 답을 읽음: 객관식 844/845 일치. 나머지 1건은 선택지 목록을 따라 쓴 무응답으로, 공식 파서는 D로 처리했고 `structured`는 unparsed로 처리함
- 데이터 수정: pandas가 `validation_Geography_15`의 선택지 D `"None"`(정답)을 결측으로 읽어, 원본 Qwen 코드에서는 이 문항을 맞힐 수 없었음 → 복원하고 `run_config.json`에 기록
