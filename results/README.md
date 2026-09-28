# results: 실험 설정과 결과

## 폴더 규칙

| 경로 | 만드는 코드 | 내용 |
|---|---|---|
| `<run-name>/run_config.json` | `code/eval/infer_mmmu.py` | 실험 설정: 모델·체크포인트 해시, 데이터·공식 코드 해시, 실제 파라미터, TOML 설정 해시, 패키지 버전, seed |
| `<run-name>/scores.json` | `code/eval/score_mmmu.py` | 정확도(전체, 객관식, 주관식, 분야별, 과목별), unparsed 건수, 추출 규칙별 건수 |
| `<run-name>/per_sample.csv` | `code/eval/score_mmmu.py` | 문항별 정답, 추출된 답, 추출 규칙, 정오 (error analysis용) |
| `<run-name>/mc_disagreements.csv` | `code/eval/score_mmmu.py` | 두 추출기가 다르게 읽은 객관식 문항과 응답 원문 (추출기 감사용) |
| `<run-name>/predictions.jsonl` | `code/eval/infer_mmmu.py` | 응답 원문과 프롬프트. **MMMU 질문이 들어 있으므로 git에서 제외** |

## 기준 평가 설정 (`configs/evaluation.toml`)

데이터 버전과 출력 토큰 한도는 모든 평가에서 고정합니다. 성능 향상은 이 조건 안에서 찾습니다.

| 항목 | 값 |
|---|---|
| Base model | `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| 평가 데이터 | MMMU val 900문항 (객관식 847, 주관식 53). Hugging Face Hub `MMMU/MMMU` @ `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68` 하나만 씁니다(`code/eval/mmmu_data.py`). 이미지는 원본을 손실 없이 PNG로 저장해 입력합니다 |
| 프롬프트 p | `code/eval/prompting.py`: 객관식은 `Question/Options` 뒤 `Answer with the option letter only.`, 주관식은 `Answer: <your short answer>` 한 줄 형식과 풀이 생략 지시. 모델 chat template을 적용하고 이미지를 텍스트 앞에 둠. 이미지는 262,144–2,097,152픽셀로 조정 |
| 디코딩 | greedy (temperature 0, seed 0). top-p 0.8, top-k 20, presence penalty 1.5도 전달하지만 greedy에서는 영향이 없음 |
| 생성 길이 | `max_new_tokens` 512(고정), `max_model_len` 8192, vLLM seed 0, `max_num_seqs` 2, GPU 메모리 사용 목표 0.85 |
| 채점 g | **structured**(주 지표): 최종 답 선언을 읽고, 무작위 추측 없이 정규화 후 exact match. 출력 한도에 걸려 잘린 응답은 모델이 명시한 답(`\boxed{}`, `Answer: C` 같은 답 선언, 첫머리의 선택지 글자)만 인정하고, 없으면 오답입니다. **official**(참고): MMMU 공식 파서 |

Fine-tuned 체크포인트도 위 설정 그대로 평가합니다(`MODEL_PATH`만 바꿈). 파라미터를 바꾸는 실험은 TOML을 복사해 이름을 바꾸고, 해당 설정의 baseline도 다시 측정합니다. 각 run의 설정 값과 파일 해시는 `run_config.json`에 남습니다.

## 실험 기록 (MMMU val)

강의 p.20의 원칙대로 baseline에서 시작하고, 한 번에 하나만 바꾸고, 행마다 run 폴더를 남깁니다.

| Run | 모델 / 체크포인트 | 학습 데이터 | 디코딩 | structured Acc. | 객관식 Acc. | official Acc. | unparsed | 비고 |
|---|---|---|---|---:|---:|---:|---:|---|
| `evaluation_20260928_173159` | Qwen3-VL-4B-Instruct | 없음 | greedy | **51.44%** (463/900) | 53.84% (456/847) | 51.56% (464/900) | 객관식 57, 주관식 0 | **baseline.** 2026-09-28 RTX 3090 실행, 생성 343.7초. 512토큰 한도 도달 59문항 중 57문항은 답을 명시하지 못해 오답, 2문항은 명시한 답으로 채점. run 폴더의 `scores.json`·`per_sample.csv`는 현재 채점기로 다시 채점한 결과(실행 당시 채점기로는 467/900) |

## 입력 점검 (기준 실행, GPU 없이 수행)

- 현재 코드(`mmmu_data.py`, `prompting.py`, `infer_mmmu.py`, `evaluation.toml`)의 파일 해시가 기준 실행의 `run_config.json` 기록과 같습니다.
- 현재 로더로 900문항의 입력을 다시 만들면 문제·선택지 문구, 이미지 목록, 정답, chat template 적용 결과, 입력 토큰 수(이미지 포함)가 기준 실행의 `predictions.jsonl`과 모두 일치합니다(2026-09-28 확인).
- 가장 긴 문항도 입력과 출력을 합쳐 4,142토큰으로 `max_model_len` 8,192 안입니다.

| 문항당 이미지 수 | 문항 수 | 입력 토큰 중앙값 | 최대 |
|---|---:|---:|---:|
| 1 | 857 | 417 | 2,231 |
| 2 | 24 | 891 | 4,140 |
| 3 | 5 | 905 | 1,088 |
| 4 | 8 | 1,178 | 2,503 |
| 5 | 6 | 1,458 | 1,690 |
