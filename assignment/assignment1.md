# MMMU-val Baseline Evaluation Report — Qwen3-VL-4B-Instruct

- **팀명**: 도비
- **팀원**: 신준철, 이민규, 김연서, 이진우
- **작성일**: 2026-09-28
- **재현 커맨드**: `docker build -t ssu:MMDL . && bash code/scripts/run_mmmu_eval.sh` (인자 없이 실행하면 3절의 기준 설정으로 추론과 채점을 수행한다. 경로 지정은 1절 참조)

---

## 1. 환경 / 재현성

| 항목 | 값 |
|---|---|
| 모델 checkpoint | `Qwen/Qwen3-VL-4B-Instruct` (ebb281ec70b05090aa6165b016eac8ec08e71b17) |
| 평가 데이터 | Hugging Face `MMMU/MMMU` @ `98e6ac0cb9b7b2cd2c991b85a50762edc4aedc68`, validation 900문항 (객관식 847, 주관식 53) |
| 기준 실행 | `results/evaluation_20260928_173159` (2026-09-28). run 폴더는 Git에 올리지 않고, 설정과 점수 요약은 [results/README.md](../results/README.md)에 기록 |
| 추론 백엔드 | vLLM 0.11.2 |
| 사용 GPU | RTX 3090, 24GB |
| 실측 peak VRAM | 따로 측정하지 않음. vLLM이 `gpu_memory_utilization=0.85`에 따라 GPU 메모리 23.6 GiB 중 약 20.0 GiB를 예약 |
| 총 소요 시간 | 440.2초(약 7분 20초) / 900문항 로드·입력 준비·모델 로드·생성. 이 중 생성 343.7초. 채점은 CPU에서 별도로 수행 |
| 의존성 | [평가 의존성](../code/requirements-eval.txt), [학습·다운로드 의존성](../code/requirements-train.txt), [Dockerfile](../Dockerfile) |
| 실행 커맨드 | 아래 bash 명령. 모델·데이터·설정·결과 경로를 CLI 인자로 지정한다. |

호스트의 프로젝트 루트에서 처음 한 번 이미지를 빌드한다. 모델과 평가 데이터는 이미지에 포함된다.

```bash
docker build -t "<Docker 이미지 이름>" .
```

다음 명령으로 추론과 채점을 실행한다. `<...>`는 실제 값으로 바꾸며, 모델·데이터·설정 파일은 모두 **컨테이너 내부 경로**를 지정한다.

```bash
DOCKER_IMAGE="<Docker 이미지 이름>" \
CONTAINER_NAME="<컨테이너 이름>" \
bash code/scripts/run_mmmu_eval.sh \
  --model "<모델 체크포인트 경로>" \
  --data-path "<MMMU 데이터 경로>" \
  --config "<평가 설정 파일 경로>" \
  --out-root "<결과 저장 상위 경로>" \
  --run-name "<결과 폴더 이름>"
```



`<평가 설정 파일 경로>`에는 3절의 생성 설정을 담은 TOML 파일을 지정한다. 데이터와 이미지는 `--data-path`에 저장한다.   
인자를 생략하면 기본값으로 평가한다. 기본값은 이미지 `ssu:MMDL`, 컨테이너 `mmdl-work`, 모델 `/opt/mmdl/models/Qwen3-VL-4B-Instruct`, 데이터 `/opt/mmdl/data`, 설정 `/opt/mmdl/configs/evaluation.toml`(3절의 기준 설정), 결과 폴더 `results/evaluation_<실행 시각>`(호스트의 `results/`에 연결)이다. 자세한 default 값은 `code/scripts/run_mmmu_eval.sh` 에서 확인 가능하다.

파인튜닝 작업은 호스트에서 다음 명령으로 같은 컨테이너에 접속해 진행한다.

```bash
docker exec -it --workdir /opt/mmdl "<컨테이너 이름>" bash
```
학습 후에는 평가 명령의 `--model`을 병합한 체크포인트 경로로 바꾸고 결과 폴더 이름을 새로 지정한다. 

## 2. 프롬프트

**실제 모델에 들어간 프롬프트 전문** (변수 부분은 `{}`로 표시):

객관식:

```text
Question: {question}
Options:
A. {option_A}
B. {option_B}
...
Answer with the option letter only.
```

선택지는 문항에 실제 있는 것만 한 줄씩 넣는다(문항에 따라 2–9개, A–I).

주관식:

```text
{question}

Solve the question using the image(s) when relevant.Answer with exactly one line in this format: Answer: <your short answer>.You must not show reasoning.
```

두 유형 모두 모델의 chat template을 적용하고, 문항의 이미지를 원래 순서대로 텍스트 앞에 둔다. 문제 속 `<image 1>` 같은 표시는 그대로 두며, 시스템 프롬프트와 few-shot 예시는 없다. 모델이 실제로 받은 입력은 다음 형태다(`<|image_pad|>`는 이미지 토큰으로 채워진다).

```text
<|im_start|>user
<|vision_start|><|image_pad|><|vision_end|>Question: {question}
Options:
A. {option_A}
...
Answer with the option letter only.<|im_end|>
<|im_start|>assistant
```

- 출처: 객관식의 `Question:`·`Options:` 구성은 [Qwen 공식 MMMU 추론 코드](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/run_mmmu.py)와 같다. 출력 지시문은 [MMMU 공식 저장소의 유형별 프롬프트 설정](https://github.com/MMMU-Benchmark/MMMU/blob/268471d0d488258990025331c7528359c324aa25/mmmu/configs/llava1.5.yaml)을 참고해 수정했다. 객관식에는 선택지 글자만 출력하도록 요구하고, 주관식에는 Answer: <정답> 형식과 풀이 생략 지시를 추가했다.

- 선택 이유: 문항 유형에 맞춰 응답 형식을 고정함으로써 풀이와 부가 설명이 정답 추출에 개입할 여지를 줄이고자 했다. 또한 한정된 생성 예산에 맞춰 풀이 과정을 생략하고 정답만을 출력하게 하여 정답 잘림의 가능성을 줄이고자 하였다. 이 프롬프트를 파인튜닝 이후에도 그대로 사용해, 동일한 입력 형식과 채점 규칙 아래에서 성능 변화를 비교한다. 



## 3. 생성(Decoding) 설정

### 3.1 Sampling recipe

| 파라미터 | 값 |
|---|---|
| `do_sample` | vLLM에는 전달하지 않음. temperature=0으로 greedy decoding 사용 |
| `temperature` | 0.0 |
| `top_p` | 0.8 |
| `top_k` | 20 |
| `repetition_penalty` | 1.0 |
| `presence_penalty` | 1.5 |
| `seed` | 0 |

- **출처**: [Qwen MMMU Instruct 실행 스크립트](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/infer_instruct.sh)와 [추론 코드](https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/run_mmmu.py). 공식 설정은 temperature 0.7, top_p 0.8, top_k 20, repetition_penalty 1.0, presence_penalty 1.5, seed 42다. 이번 평가는 파인튜닝 전후 문항별 비교에서 샘플링에 따른 변동을 줄이려고 greedy와 seed 0을 선택했다. 문항당 한 번 생성하며, top_p·top_k는 전달값이지만 greedy에서는 확률 샘플링을 하지 않는다.

### 3.2 생성 예산 / 이미지 해상도

| 파라미터 | 값 |
|---|---|
| `max_new_tokens` | 512 |
| 이미지 해상도 처리 (`min_pixels`/`max_pixels` 등) | min_pixels=262144 (256×32×32), max_pixels=2097152 (2048×32×32). qwen-vl-utils로 종횡비를 유지해 처리 |

**선택 근거** (본인이 사용한 인프라 제약과 어떻게 연결되는지 — 속도/VRAM/응답 잘림 등 trade-off): 짧은 정답을 요구해도 풀이가 먼저 나올 수 있어 최대 512토큰을 허용했다. 이미지 면적 상한은 2048토큰에 해당하는 2097152 픽셀로 두고 동시 요청을 2개로 제한해, 24GB GPU에서 다중 이미지 입력과 KV cache의 메모리 부담을 줄이고자 했다.

기준 실행에서 평균 입력은 630.86토큰, 평균 출력은 39.59토큰이었고 생성은 RTX 3090에서 343.7초가 걸렸다. 가장 긴 문항도 입력과 출력을 합쳐 4,142토큰으로 context 한도 8,192 안이었다. 반면 900문항 중 59문항(6.6%)은 풀이를 쓰다가 512토큰 한도에 도달했다. 512토큰이 모든 문항에 충분하지는 않다는 뜻이다. 한도를 늘리면 잘림은 줄 수 있지만, 풀이가 언제 끝날지 알 수 없어 잘림을 없앨 수는 없고 체크포인트마다 반복할 평가 시간은 늘어난다. 그래서 512토큰을 파인튜닝 전후 비교의 고정 조건으로 두었다(7·8절).

## 4. 채점(파싱) 방식

- **사용한 파서/로직**: [answer_extraction.py](../code/eval/answer_extraction.py)에서 모델 응답의 답을 추출하고, [score_mmmu.py](../code/eval/score_mmmu.py)에서 정답과 비교한다.

  채점 규칙의 참고 기준은 [MMMU 공식 평가 코드의 `parse_multi_choice_response`, `parse_open_response`, `eval_open`](https://github.com/MMMU-Benchmark/MMMU/blob/268471d0d488258990025331c7528359c324aa25/mmmu/utils/eval_utils.py)이다. 객관식의 선택지 글자 비교와 숫자 답의 소수 둘째 자리 반올림 비교는 공식 구현과 같다. 다만 답을 찾는 과정은 별도로 구현했다. `Answer: C`처럼 답을 명시한 표현을 우선하며, 굵은 글씨나 수식 표기로 감싼 답도 읽도록 했다. 또한 공식 객관식 파서는 답을 찾지 못하면 선택지를 무작위로 고르지만, 이 평가에서는 오답으로 처리한다. 공식 파서로도 별도 채점해 비교하며, **5절의 제출 점수에는 자체 파서의 결과를 사용했다.**

- **동작 방식 요약**: 응답에서 **모델이 답으로 제시한 부분을 찾은 뒤, 그 값을 정답과 비교**한다. 풀이 전체를 다른 모델에게 보내 정오를 판단하게 하지는 않는다.

  **① 객관식 — 선택지 글자 하나를 찾아 비교한다.**

  먼저 굵은 글씨·수식 기호 같은 표시를 정리한다. 예를 들어 `Final answer: **C**`에서는 `C`를 읽는다. 답을 찾는 순서는 다음과 같으며, 앞 단계에서 답을 찾으면 뒤 단계는 적용하지 않는다.

  1. `Answer: C`, `C is correct`, `\boxed{C}`처럼 답을 명시한 표현을 찾는다. 여러 번 답을 제시했다면 가장 마지막 답을 사용한다. 답 표시 바로 뒤에 글자 하나만 있으면 `Answer: c`처럼 소문자도 읽는다.
  2. 명시한 답이 없으면 `C`나 `C. 설명…`처럼 응답 첫머리에 적힌 선택지 글자를 답으로 본다. 한 줄에 글자 하나만 쓴 응답은 소문자도 읽는다. 선택지를 단순히 나열한 경우는 제외한다. 이번 평가에서 객관식 847문항 중 778문항이 이 단계에서 채점됐다.
  3. 선택지 옆에 체크 표시나 `correct`가 있는지 확인한다. 다음으로 응답 전체, 마지막 문단 순서로 선택지 내용이 하나만 언급됐는지 살핀다.
  4. 그래도 찾지 못하면 문장 속에 독립적으로 등장한 B–H 중 유일한 글자를 사용한다. 이 보조 규칙만 B–H로 제한하며, 앞 단계에서는 해당 문항의 실제 선택지 범위를 사용한다.

  추출한 글자가 정답 글자와 같으면 정답이다. 단, 512토큰 한도에 걸려 잘린 응답에는 1·2단계만 적용한다. 풀이 도중 끊긴 문장에서 선택지 내용이나 글자를 답으로 읽지 않기 위해서다.

  **② 주관식 — 숫자, 그림 속 기호, 일반 텍스트를 구분해 비교한다.**

  데이터셋 정답의 **형식**을 보고 숫자형, 한 글자 기호형, 텍스트형 중 적용할 규칙을 정한다. 정답 정보는 추론 후 채점에만 사용하며 모델 입력에는 넣지 않는다. 응답에서는 마지막 `\boxed{…}` 안의 답, 마지막 `Answer:` 뒤의 답 순서로 확인한다. 둘 다 없으면 숫자·기호형은 끝 문장부터 답을 찾고, 텍스트형은 첫 문장과 마지막 문장을 확인한다. 잘린 응답은 `\boxed{…}`와 `Answer:` 뒤의 답만 인정한다. 정답이 여러 개로 주어진 문항(예: `['24/7', '3.429']`)은 그중 하나와 일치하면 정답이다.

  아래는 비교 규칙을 설명하기 위한 예시다.

  | 답 유형 | 모델 응답 → 비교할 값 | 정답 판정 규칙 |
  |---|---|---|
  | 숫자 | `Answer: 1/2` → `0.5` | 분수·지수·쉼표·million 등의 표기를 수치로 바꾼 뒤, 정답과 각각 소수 둘째 자리까지 반올림해 비교한다. `%`가 있으면 표시된 숫자와 이를 100으로 나눈 값 모두를 비교 후보로 둔다. |
  | 그림 속 기호 | `Answer: c` → `C` | 추출한 글자를 대문자로 바꿔 정답 기호와 비교한다. |
  | 텍스트 | `Answer: Paris` → `paris` | 대소문자와 공백을 정리한 뒤 정답 구문이 포함됐는지 확인한다. 다른 단어의 일부가 우연히 일치하는 경우는 제외한다. |

  **③ 답을 찾지 못한 문항도 평가에 포함한다.**

  추출 실패는 오답으로 처리하고 전체 900문항의 분모에 포함한다. 이번 평가에서는 객관식 57문항이 이에 해당했으며, 모두 512토큰 한도에서 풀이가 잘려 답을 명시하지 못한 응답이다. 주관식은 없었다. 참고로 MMMU 공식 파서로 채점하면 51.56%(464/900)이며, 이때 답을 찾지 못한 객관식 59문항은 공식 규칙대로 선택지를 무작위로 골랐다. 무작위 선택은 공식 코드의 seed 42를 그대로 써서 다시 채점해도 같은 결과가 나온다. 문항별 응답 원문, 추출한 답, 적용한 규칙을 저장해 채점 결과를 다시 확인할 수 있도록 했다.

## 5. 결과

| No. | Subject | Data Num | Acc |
|---|---|---|---|
| 1 | Accounting | 30 | 43.33% (13/30) |
| 2 | Agriculture | 30 | 46.67% (14/30) |
| 3 | Architecture_and_Engineering | 30 | 20.00% (6/30) |
| 4 | Art | 30 | 66.67% (20/30) |
| 5 | Art_Theory | 30 | 80.00% (24/30) |
| 6 | Basic_Medical_Science | 30 | 66.67% (20/30) |
| 7 | Biology | 30 | 56.67% (17/30) |
| 8 | Chemistry | 30 | 30.00% (9/30) |
| 9 | Clinical_Medicine | 30 | 63.33% (19/30) |
| 10 | Computer_Science | 30 | 50.00% (15/30) |
| 11 | Design | 30 | 83.33% (25/30) |
| 12 | Diagnostics_and_Laboratory_Medicine | 30 | 33.33% (10/30) |
| 13 | Economics | 30 | 50.00% (15/30) |
| 14 | Electronics | 30 | 40.00% (12/30) |
| 15 | Energy_and_Power | 30 | 50.00% (15/30) |
| 16 | Finance | 30 | 33.33% (10/30) |
| 17 | Geography | 30 | 50.00% (15/30) |
| 18 | History | 30 | 70.00% (21/30) |
| 19 | Literature | 30 | 86.67% (26/30) |
| 20 | Manage | 30 | 40.00% (12/30) |
| 21 | Marketing | 30 | 60.00% (18/30) |
| 22 | Materials | 30 | 20.00% (6/30) |
| 23 | Math | 30 | 43.33% (13/30) |
| 24 | Mechanical_Engineering | 30 | 43.33% (13/30) |
| 25 | Music | 30 | 36.67% (11/30) |
| 26 | Pharmacy | 30 | 66.67% (20/30) |
| 27 | Physics | 30 | 36.67% (11/30) |
| 28 | Psychology | 30 | 73.33% (22/30) |
| 29 | Public_Health | 30 | 43.33% (13/30) |
| 30 | Sociology | 30 | 60.00% (18/30) |
| | **Overall (macro avg)** | **900** | **51.44% (463/900)** |

계산식: `Overall = mean(30개 과목 accuracy)` 과목당 30문항이므로 463/900×100=51.44%와 같다. 표시한 반올림값이 아닌 원래 정답 수로 계산했다. 객관식 456/847, 주관식 7/53이다.

## 6. 공식 수치와의 비교

| | Overall (MMMU val) |
|---|---|
| 공식 (Qwen3-VL Technical Report) | 67.4 |
| 우리 재현 결과 | 51.44 |
| 차이 (Δ) | −15.96%p (우리 결과 − 공식) |

공식 수치 출처: [Qwen3-VL Technical Report, Table 4](https://arxiv.org/pdf/2511.21631v2#page=18).

## 7. 격차 분석

본 평가의 51.44%(463/900)는 공식 67.4%보다 15.96%p 낮다. 오답 437개 중 334개는 정상 종료한 객관식 응답이 선택지를 골랐지만 틀린 경우이고(그중 333개는 풀이 없이 곧바로 답한 응답), 57개는 풀이가 512토큰에서 잘려 답을 명시하지 못한 객관식, 46개는 주관식 오답이다. 따라서 격차는 답 추출 실패만으로 설명되지 않는다. **가장 의심되는 원인은 즉답 요구가 계산·추론 문항에 불리하게 작용했을 가능성이다.** [공식 보고서][official-report] · 오답 수치는 본 평가 응답을 자체 집계했다.

사용한 Instruct 모델의 현재 실행에는 풀이를 별도로 생성하고 숨기는 단계가 없다. 모델은 생성한 중간 계산을 다음 답변 생성에 활용할 수 있으므로, 정답만 요구하는 지시는 이 기회를 제한한다. 따라서 “내부에서 충분히 풀고 답만 출력할 것”이라는 가정은 보장되지 않는다. [모델 응답 템플릿][instruct-template] · [CoT 연구][cot-paper]

같은 실행의 응답에도 이를 뒷받침하는 정황이 있다. 객관식 847문항 중 곧바로 답한 778문항(출력 16토큰 미만)의 정답률은 57.2%(445/778)였지만, 지시와 달리 풀이를 쓴 뒤 512토큰 안에 끝낸 10문항은 9문항이 정답이었다. 풀이를 쓰다 한도에 걸린 나머지 59문항(정답 2)은 건축공학 13, 재료 12, 수학 7, 기계 7, 회계 6문항 등 계산형 과목에 몰렸다. 다만 풀이 여부는 모델이 문항에 따라 스스로 정했으므로 이 차이를 풀이의 효과로 단정할 수는 없다. Qwen 공식 평가 코드가 temperature 0.7과 최대 32,768토큰의 출력 예산을 쓴다는 점도 차이의 원인일 수 있다. [Qwen 실행 설정][qwen-recipe]

## 8. 기타 특이사항 / 한계 (Optional)

- **출력 한도 512토큰에서 59문항(6.6%)이 잘렸다.** 그중 57문항은 답을 명시하기 전에 끊겨 오답으로 처리했다. 여러 체크포인트를 반복 평가할 시간 예산에 맞춰 한도를 512로 고정했으며, 파인튜닝 전후 비교에도 같은 한도를 적용한다.
- **다른 조건과의 비교는 하지 않았다.** 이미지 해상도 상·하한, temperature(Qwen 공식 설정은 0.7), 프롬프트 형식을 바꾼 결과는 측정하지 않아 각 설정의 영향을 수치로 분리할 수 없다. [Qwen 실행 설정][qwen-recipe]
- **다음 실험에서는 한 번에 한 조건만 바꿀 계획이다.** 정확도뿐 아니라 출력 한도 도달 문항 수와 생성 시간도 함께 기록하고, 파인튜닝 전후 비교에는 이 보고서의 평가 조건을 동일하게 적용한다.

[official-report]: https://arxiv.org/pdf/2511.21631v2#page=18
[qwen-recipe]: https://github.com/QwenLM/Qwen3-VL/blob/96588727e44c78b25ba03ea03b8e12f7e64fd0da/evaluation/mmmu/infer_instruct.sh
[instruct-template]: https://huggingface.co/Qwen/Qwen3-VL-4B-Instruct/blob/ebb281ec70b05090aa6165b016eac8ec08e71b17/chat_template.json
[cot-paper]: https://arxiv.org/abs/2201.11903
