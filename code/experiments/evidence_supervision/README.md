# 증거 지도 실험 · 재현 안내

정답만 가르친 지도와 증거·풀이·정답을 가르친 지도를 같은 데이터·같은 step·같은 홀드아웃에서 비교한 실험이다. **증거 지도가 정답만 지도를 이긴 측정은 없었다.** 결과와 해석은 [실험 보고서](REPORT.md), 데이터를 어떻게 만들고 무엇으로 검증했는지는 [데이터 구축](DATA.md), arm이 건드리는 모듈이 무엇인지는 [모델 구조](MODEL.md)에 있다.

아래 명령은 **저장소 최상위에서 `cd code` 후** 실행한다. 결과만 확인하려면 1절, 저장된 예측을 다시 채점하려면 2절, 데이터부터 처음 만들려면 3절을 쓴다.

> **채점 기준.** 모든 수치는 구조화 채점(`scores.json`의 `structured`)이다. MMMU 공식 채점기는 파싱 실패 시 보기를 무작위로 고르므로 출력 완결성이 다른 모델 간 비교에 쓸 수 없다([REPORT 5.2](REPORT.md#52-공식-채점기는-파싱-실패-시-무작위로-찍는다)).

## 1. 저장된 증거로 결과 확인

Python 3.12 표준 라이브러리만 쓴다. GPU·모델·데이터 다운로드가 전혀 필요 없고, 새로 클론한 저장소에서 바로 돈다.

```bash
python experiments/evidence_supervision/make_evidence.py
```

보고서의 두 가지 핵심 주장을 추적된 CSV에서 다시 계산해 출력한다.

- **공식 채점기가 못 푸는 모델에 보너스를 준다** — `t_stop`은 147문항을 무작위로 찍어 공식 점수가 64문항 부풀려졌다. [`evidence/official_vs_structured.csv`](evidence/official_vs_structured.csv)
- **끝맺기 이득은 증거 내용이 아니라 출력 구조에서 온다** — 기준선이 답을 못 낸 42문항(전부 `finish_reason=length`)에서 증거 arm은 31건을 회수했고, 내용을 전부 뺀 구조 대조군도 30~32건으로 같았다. [`evidence/truncated_42.csv`](evidence/truncated_42.csv)

진단 근거인 기준선 주관식 오답 46건 전수는 [`evidence/baseline_open_wrong_46.csv`](evidence/baseline_open_wrong_46.csv)다 — 46건 중 45건이 실제 오답이고 출력 토큰 중앙값이 8이다. **그림을 읽는 문제가 아니라 다단계 계산 문제**라는 것이 여기서 나온다.

로컬에 `results/` 트리가 있으면 테이블을 다시 만들 수 있다. `results/`는 Git에 올리지 않으므로 이 단계는 실험을 돌린 사람만 쓴다.

```bash
python experiments/evidence_supervision/make_evidence.py --build
```

## 2. 저장된 예측 다시 채점

추출 규칙을 바꾸면 **비교할 모든 실행을 같은 규칙으로** 다시 채점해야 한다. GPU 없이 CPU에서 돈다.

```bash
# MMMU 한 실행 재채점
CUDA_VISIBLE_DEVICES= python code/eval/score_mmmu.py results/<run-name>

# 도메인 내 홀드아웃 전 실행 재채점 (현재 추출기 기준)
python code/eval/rescore_predictions.py --write

# 공식 vs 구조화 채점 차이 분해
python experiments/evidence_supervision/official_gap.py

# 파싱 실패로 남아 있는 여지 산정
python experiments/evidence_supervision/parser_headroom.py

# 팀 ICoT 채점기로 교차 채점 (그들 프로토콜은 추론이 동일, 채점기만 다름)
python experiments/evidence_supervision/cross_score_icot.py
```

교차 채점은 **그들 채점기가 run 디렉터리의 `scores.json`·`per_sample.csv`를 덮어쓰기 때문에** 입력만 임시 디렉터리로 복사해 거기서 돌린다. 우리 런은 `evaluation_data.json`을 저장하지 않았으므로 `predictions.jsonl`의 `answer`와 프롬프트의 `Options:` 블록에서 복원하고, 선택지 글자 연속성·정답 범위·900행 유일성을 검증한 뒤에만 채점한다. 결과는 [REPORT 5.4~5.6](REPORT.md#54-채점기는-하나이고-버전-이력이-있다) — 채점기 버전 이력, 교차 채점 결과, ICoT와의 비교가 유효한 이유.

arm 간 쌍대 비교(McNemar)는 `--run LABEL=PATH`를 실행 수만큼 반복한다.

```bash
python code/eval/compare_arms.py \
  --run "a_cqa=results/holdout-a_cqa" \
  --run "b_cqa=results/holdout-b_cqa"
```

## 3. 데이터부터 처음 재현

### 3.1 기준선 채점 기록

층화와 쌍대 비교가 모두 이 기록을 기준으로 삼으므로 먼저 만든다. 후보 풀 20,222문항 전체에 기준선을 돌리며 약 9분 걸린다. **평가는 vLLM 0.11.2로 고정**이다 — 층화 기준이 평가와 다른 엔진에서 나오면 기준 자체가 흔들린다.

```bash
bash code/scripts/fetch_assets.sh
python code/data/fetch_sources.py
python code/eval/infer_indomain.py --run-name phase0
```

### 3.2 증거 생성과 타깃 렌더링

**분할을 먼저, 선택을 나중에** 한다. 이미지 해시로 9:1을 가른 뒤 오답 우선 선택은 학습 쪽 안에서만 한다 — 순서를 뒤집으면 홀드아웃이 쉬운 쪽으로 편향된다([DATA.md](DATA.md#분할을-먼저-선택을-나중에)).

```bash
python code/data/split_holdout.py
python code/data/synth_evidence_ai2d.py     --prefer-wrong results/phase0/indomain_split.json
python code/data/synth_evidence_chartqa.py  --prefer-wrong results/phase0/indomain_split.json
python code/data/synth_evidence_scienceqa.py

python code/data/render_targets.py --style answer_only --out training_data/sft/arm_a
python code/data/render_targets.py --style ekra        --out training_data/sft/arm_b

# 대조군 둘: 구조만 (내용 없음) / 한 문장 문구만
python code/data/make_diagnostic_sets.py --source training_data/sft/arm_b
python code/data/make_format_only.py --source training_data/sft/arm_a --out training_data/sft/a_cqa_stop

python code/data/make_indomain_holdout.py --sft training_data/sft/arm_a
```

증거 품질을 사람이 확인하려면 로컬 감사 UI를 띄운다. 이미지·모델 입력·학습 타깃을 한 화면에 보여주고 `a`/`b`/`c` 키로 판정한다.

```bash
python code/audit/build_queues.py
python code/audit/serve_audit.py          # http://127.0.0.1:8765/
```

### 3.3 학습

GPU 잡은 공유 클러스터에 제출한다. `--node`와 `--gpus`는 **필수이며 기본값이 없다** — 노드를 주지 않으면 제출이 거부된다.

```bash
ARM=a_1gpu_lr2e5 SEED=0 SFT_SET=arm_a bash code/cluster/submit_job.sh \
    --template lora --node <노드 이름> --gpus 1 --name yeunsuh-a-cqa
```

arm 설정은 `configs/arms/*.toml`이고 `ARM=` 이름으로 고른다. 글로벌 배치는 32로 고정이며, GPU 개수와 누적 step의 곱이 32에서 벗어나면 `code/train/launch.sh`가 학습을 **시작하지 않고 거부**한다 — GPU 개수가 바뀌어도 optimizer step 예산이 같아야 비교가 성립하기 때문이다.

학습 대상이 정의와 일치하는지(위치 일치·grad≠0·파라미터 수)는 제출 전에 GPU 없이 점검할 수 있다.

```bash
python code/train/verify_arms.py
```

> **학습률 주의.** `configs/arms/`의 기본 `learning_rate`는 `1e-4`이고, 그 값은 오답 우선 층화와 **겹칠 때만** 도메인 내 −4.51pp 회귀를 만든다. 비교 실험에는 `*_lr2e5.toml` 변종을 쓴다. `train_loss`는 0.207 vs 0.2035로 거의 같아서 **손실만 보면 알 수 없다**([REPORT 9절](REPORT.md#9-설계-교훈)).

### 3.4 병합과 평가

병합은 기본 모델을 CPU 메모리에 올리므로 호스트 RAM이 충분해야 한다.

```bash
python code/train/merge_lora.py \
  --base-model models/Qwen3-VL-4B-Instruct \
  --adapter checkpoints/<run> --output checkpoints/<run>-merged

# 도메인 내 홀드아웃
python code/eval/infer_indomain.py \
  --model checkpoints/<run>-merged \
  --candidates training_data/candidates_holdout/holdout.jsonl \
  --run-name holdout-<run>

# MMMU 900문항
bash code/scripts/run_mmmu_eval.sh --model checkpoints/<run>-merged --run-name mmmu-<run>
```

## 산출물 경로

| 내용 | 경로 |
|---|---|
| 검증용 증거 (추적됨) | `experiments/evidence_supervision/evidence/` |
| 기준선 채점 기록 | `results/phase0/indomain_split.json` (20,222행) |
| 학습 데이터 | `training_data/sft/<arm>/` (Git 제외) |
| 도메인 내 홀드아웃 | `training_data/candidates_holdout/` (Git 제외) |
| 감사 판정 원본 | `training_data/evidence/audit_queues/*.labels.json` (Git 제외) |
| 평가 결과 | `results/<run-name>/` (Git 제외) |
| arm 설정 | `configs/arms/*.toml` |

## 운영 규칙

- GPU 잡은 **노드와 GPU 개수를 명시적으로 받지 않으면 제출하지 않는다.** 공유 클러스터이기 때문이다.
- 잡 이름에 사용자 이름을 포함하고, **끝나면 삭제한다.** 지우지 않으면 공유 GPU가 계속 점유되고 네임스페이스 잡 쿼터에 걸린다.
- API 키는 Kubernetes Secret으로만 주입한다(`code/cluster/setup_secrets.sh`). 저장소·argv·렌더된 매니페스트에 넣지 않는다. 렌더된 `*.yaml`은 `.gitignore`가 제외한다.
- 평가는 **vLLM 0.11.2로 고정**한다. 모델 리비전도 고정이다.
- 유료 배치는 **동기 호출 1건 검증 → 소규모 파일럿 → 전량** 순서를 지킨다.
- 추출 규칙을 바꾸면 비교할 **모든** 실행을 다시 채점한다.
