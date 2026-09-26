# results: 실험 설정과 결과

## 폴더 규칙

| 경로 | 만드는 코드 | 내용 |
|---|---|---|
| `<run-name>/run_config.json` | `code/eval/infer_mmmu.py` | 실험 설정: 모델·체크포인트 해시, 데이터·공식 코드 해시, 실제 파라미터, TOML 설정 해시, 패키지 버전, seed |
| `<run-name>/scores.json` | `code/eval/score_mmmu.py` | 정확도(전체, 객관식, 주관식, 분야별, 과목별), unparsed 건수, 추출 규칙별 건수 |
| `<run-name>/per_sample.csv` | `code/eval/score_mmmu.py` | 문항별 정답, 추출된 답, 추출 규칙, 정오 (error analysis용) |
| `<run-name>/mc_disagreements.csv` | `code/eval/score_mmmu.py` | 두 추출기가 다르게 읽은 객관식 문항과 응답 원문 (추출기 감사용) |
| `<run-name>/predictions.jsonl` | `code/eval/infer_mmmu.py` | 응답 원문과 프롬프트. **MMMU 질문이 들어 있으므로 git에서 제외** |
| `prompt_check/prompt_tokens.csv` | `code/eval/check_prompts.py` | 문항별 이미지 크기와 프롬프트 토큰 수 |
| `pipeline_validation/official_parity.txt` | `code/eval/tests/check_official_parity.py` | MMMU 공식 스크립트와의 대조 결과 |

## 기준 평가 설정 (`configs/rtx4090.toml`)

| 항목 | 값 |
|---|---|
| Base model | `Qwen/Qwen3-VL-4B-Instruct` @ `ebb281ec70b05090aa6165b016eac8ec08e71b17` |
| 평가 데이터 | MMMU val 900문항 (객관식 847, 주관식 53). VLMEvalKit `MMMU_DEV_VAL.tsv`(MD5 `585e8ad7…`), 정답은 `MMMU-Benchmark/MMMU@268471d`의 `answer_dict_val.json` |
| 프롬프트 p | 2026-09-25 회의 합의안: 객관식은 `Question/Options` 뒤 한 글자 답 지시, 주관식은 마지막 `Final answer:` 줄 지시. 모델 chat template을 적용하고 이미지를 텍스트 앞에 둠. 이미지는 262,144–2,097,152픽셀로 조정 |
| 디코딩 | `greedy`(temperature 0, 실험 비교용) / `qwen_official`(T=0.7, top-p 0.8, top-k 20, presence 1.5; 별도 비교용). 기본은 greedy |
| 생성 길이 | `max_new_tokens` 512, `max_model_len` 8192, vLLM seed 0, `max_num_seqs` 2, GPU 메모리 사용 목표 0.85 |
| 채점 g | **structured**(주 지표): 최종 답 선언을 읽고, 무작위 추측 없이 정규화 후 exact match. **official**(참고): MMMU 공식 파서 |

Fine-tuned 체크포인트도 위 설정 그대로 평가합니다(`MODEL_PATH`만 바꿈). 파라미터를 바꾸는 실험은 TOML을 복사해 이름을 바꾸고, 해당 설정의 baseline도 다시 측정합니다. 각 run의 설정 값과 파일 해시는 `run_config.json`에 남습니다.

## 실험 기록 (MMMU val)

강의 p.20의 원칙대로 baseline에서 시작하고, 한 번에 하나만 바꾸고, 행마다 run 폴더를 남깁니다.

| Run | 모델 / 체크포인트 | 학습 데이터 | 디코딩 | structured Acc. | 객관식 Acc. | official Acc. | unparsed | 비고 |
|---|---|---|---|---:|---:|---:|---:|---|
| `base_greedy` | Qwen3-VL-4B-Instruct | 없음 | greedy | | | | | 미실행 (GPU 대기) |
| `base_qwen_official` | Qwen3-VL-4B-Instruct | 없음 | qwen_official | | | | | 미실행 (GPU 대기) |

## 파이프라인 검증 (GPU 없이 수행)

**공식 파서 재현** (`pipeline_validation/official_parity.txt`): MMMU 저장소에 들어 있는 LLaVA-1.5-13B val 응답 900개를 사용했습니다.
- `score_mmmu.py`의 official 판정이 MMMU 공식 스크립트와 900/900 일치합니다(정확도 0.3667로 동일).
- 짧은 응답에서 structured와 official이 같은 선택지를 읽었습니다(844/845). 나머지 1건은 선택지 목록을 따라 쓴 무응답입니다. official은 D로 읽었고 structured는 unparsed로 처리했습니다.
- 긴 설명형 응답에서의 정확성은 실제 Qwen3-VL 출력으로 `mc_disagreements.csv`를 감사해서 확인해야 합니다.

**이전 프롬프트 검증 기록** (`prompt_check/prompt_tokens.csv`, Qwen 원본 프롬프트·해상도 설정에서 측정)
- `validation_Art_8`을 렌더링한 결과가 강의 p.10 query와 글자 단위로 일치했습니다. 현재 합의안 프롬프트의 결과는 아닙니다.
- 이미지 토큰 수는 900문항 모두 (W/32)×(H/32)와 일치합니다. 이미지 982장 중 884장이 1.0M 픽셀까지 확대됩니다(원본 중앙값 약 0.2M 픽셀).

| 문항당 이미지 수 | 문항 수 | 프롬프트 토큰 중앙값 | 최대 |
|---|---:|---:|---:|
| 1 | 857 | 1,118 | 4,031 |
| 2 | 24 | 2,188 | 5,627 |
| 3 | 5 | 3,143 | 3,173 |
| 4 | 8 | 4,130 | 4,214 |
| 5 | 6 | 5,150 | 5,184 |

새 프롬프트·이미지 면적 설정과 생성 한도(512)에서는 `check_prompts.py`로 토큰 길이를 다시 측정해야 합니다. 위 표는 기존 원본 설정의 기록이며 새 설정의 실측 결과로 간주하지 않습니다.

**데이터 수정**: pandas가 `validation_Geography_15`의 선택지 D `"None"`(정답)을 결측값으로 읽습니다. 그래서 원본 Qwen 코드에서는 이 문항이 프롬프트에서 정답 선택지를 잃습니다. 로더에서 원래 값으로 복원하고 `run_config.json`에 기록합니다(`code/third_party/SOURCES.md`).
