# MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명**: 2026-2 멀티모달딥러닝 10조
- **팀원**: _(기입)_
- **작성일**: _(기입)_
- **재현 커맨드**: `docker run --rm --gpus all --shm-size=16g -v "$PWD/results:/opt/mmdl/results" -e RUN_NAME=base_greedy mmdl:qwen3vl bash code/scripts/run_mmmu_eval.sh`

---

## 1. 환경 / 재현성

| 항목 | 값 |
|---|---|
| 모델 checkpoint | `Qwen/Qwen3-VL-4B-Instruct` (ebb281ec70b05090aa6165b016eac8ec08e71b17) |
| 추론 백엔드 | vLLM 0.11.2, BF16 |
| 사용 GPU | RTX 4090 (24 GB) 예정; 실행 후 실제 GPU 기입 |
| 실측 peak VRAM | _(GB)_ |
| 총 소요 시간 | _(900문제 기준)_ |
| 의존성 | [`Dockerfile`](../Dockerfile), [`configs/rtx4090.toml`](../configs/rtx4090.toml), [`code/requirements-eval.txt`](../code/requirements-eval.txt), [`code/requirements-train.txt`](../code/requirements-train.txt) |
| 실행 커맨드 | `MODEL_PATH=/path/to/model DATA_ROOT=/path/to/data CONFIG_PATH=/path/to/rtx4090.toml RUN_NAME=base_greedy bash code/scripts/run_mmmu_eval.sh` (`DATA_ROOT`에 `MMMU_DEV_VAL.tsv`, `mmmu_answer_dict_val.json` 필요) |

## 2. 프롬프트

**실제 모델에 들어간 프롬프트 전문** (변수 부분은 `{}`로 표시):

```
객관식 (이미지는 텍스트 앞에 원래 순서대로 배치):
Question: {question}
Options:
A. {option_A}
B. {option_B}
C. {option_C}
D. {option_D}
E. {option_E}
F. {option_F}
G. {option_G}
H. {option_H}
I. {option_I}
Answer with the option letter only.

주관식 (이미지는 텍스트 앞에 원래 순서대로 배치):
{question}

Solve the question using the image(s) when relevant. You may reason briefly. End with exactly one line in this format: Final answer: <your short answer>.
```

- **출처**: [2026-09-25 회의 메모](https://app.notion.com/p/MMDL-10-3e2877ea237c80e280a8e48b14a49159) 상단 합의안. 실제 존재하는 선택지만 출력하고 모델의 chat template을 적용합니다. 시스템 프롬프트와 few-shot 예시는 없습니다.
- **선택 이유**: 객관식은 파서가 가장 확실하게 읽는 한 글자 답을 요청하고, 주관식은 마지막 `Final answer:` 선언을 파싱할 수 있게 합니다. 이미지 면적은 4090 메모리에 맞춰 제한합니다.

## 3. 생성(Decoding) 설정

### 3.1 Sampling recipe

| 파라미터 | 값 |
|---|---|
| `do_sample` | false (greedy) |
| `temperature` | 0 |
| `top_p` | 0.8 (greedy에서는 영향 없음) |
| `top_k` | 20 (greedy에서는 영향 없음) |
| `repetition_penalty` | 1.0 |
| `presence_penalty` | 1.5 |
| `seed` | 0 |

- **출처**: [Qwen 공식 MMMU 평가 코드](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/run_mmmu.py)의 top-p/top-k/penalty를 유지하고, baseline과 파인튜닝 전후 비교의 재현성을 위해 temperature만 0으로 설정했습니다. `VLLM_ENABLE_V1_MULTIPROCESSING=0`을 사용합니다.

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---|
| `max_new_tokens` | 512 (`max_model_len=8192`) |
| 이미지 해상도 처리 (`min_pixels`/`max_pixels` 등) | 이미지당 262,144 / 2,097,152픽셀 (`256*32*32` / `2048*32*32`) |

**선택 근거**: RTX 4090 한 장(24 GB)에서 이미지·KV cache를 함께 올리기 위해 `max_num_seqs=2`, `gpu_memory_utilization=0.85`로 시작합니다. 512토큰은 답 뒤의 짧은 설명을 허용하며, 결과의 `finish_reason`으로 잘림을 확인합니다. 다중 이미지 문항에서 OOM이면 배치를 1로 낮춘 새 run으로 기록합니다.

## 4. 채점(파싱) 방식

- 사용한 파서/로직: [`code/eval/answer_extraction.py`](../code/eval/answer_extraction.py)의 structured 파서를 주 지표로 사용하고, [MMMU 공식 파서](https://github.com/MMMU-Benchmark/MMMU/tree/268471d0d488258990025331c7528359c324aa25/mmmu/utils)를 참고 지표로 함께 기록합니다.
- 동작 방식 요약: 객관식은 마지막 답 선언과 선택지 내용을 확인하고, 주관식은 답 형식에 맞춰 정규화합니다. 추출 실패는 주 지표에서 오답으로 처리합니다. 공식 파서의 무작위 추측 횟수는 별도로 기록하며, `predictions.jsonl`에 원문을 남겨 재채점합니다.

## 5. 결과

| No. | Subject | Data Num | Acc |
|---|---|---|---|
| 1 | Accounting | 30 | |
| 2 | Agriculture | 30 | |
| 3 | Architecture_and_Engineering | 30 | |
| 4 | Art | 30 | |
| 5 | Art_Theory | 30 | |
| 6 | Basic_Medical_Science | 30 | |
| 7 | Biology | 30 | |
| 8 | Chemistry | 30 | |
| 9 | Clinical_Medicine | 30 | |
| 10 | Computer_Science | 30 | |
| 11 | Design | 30 | |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | |
| 13 | Economics | 30 | |
| 14 | Electronics | 30 | |
| 15 | Energy_and_Power | 30 | |
| 16 | Finance | 30 | |
| 17 | Geography | 30 | |
| 18 | History | 30 | |
| 19 | Literature | 30 | |
| 20 | Manage | 30 | |
| 21 | Marketing | 30 | |
| 22 | Materials | 30 | |
| 23 | Math | 30 | |
| 24 | Mechanical_Engineering | 30 | |
| 25 | Music | 30 | |
| 26 | Pharmacy | 30 | |
| 27 | Physics | 30 | |
| 28 | Psychology | 30 | |
| 29 | Public_Health | 30 | |
| 30 | Sociology | 30 | |
| | **Overall (macro avg)** | **900** | |

계산식: `Overall = mean(30개 과목 accuracy)` _(다른 방식을 썼다면 명시)_

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---|
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| 우리 재현 결과 | |
| 차이 (Δ) | |

## 7. 격차 분석

_(1000 char 이내로 작성 - Official 성능과 차이가 발생하는지, 그렇다면 그 이유를 서술. 길게 쓴다고 credit이 느는 게
아니라, 근거의 질이 핵심입니다. 레포트는 짧을수록 좋습니다.)_


## 8. 기타 특이사항 / 한계 (Optional)

_(재현 중 겪은 문제, 시간 관계상 못 해본 것, 다음에 시도해보고 싶은 것 등. 자유롭게)_
