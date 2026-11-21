# ICoT 실험 결과와 재현 방법

발표 모델은 **498/900 (55.33%)**을 기록한 어댑터다. 발표 baseline **463/900 (51.44%)**보다 35문항, 3.89%p 높았다. 같은 비교 환경에서 다시 실행한 T0는 469/900이었다. 이 T0와의 차이는 29문항, 3.22%p다. 주 지표는 MMMU validation 900문항에 대한 프로젝트 채점기 `structured`의 정확도다.

## 데이터와 모델을 바꾼 이유

ScienceQA·ChartQA·AI2D에서 E/K/R/A(시각 근거·지식·추론·정답) 해설을 갖춘 문항을 480개에서 2,000개로 늘렸다. 이 단계의 고정 조건 비교에서 Full-CoT는 481개에서 483개로 2문항 더 맞혔다. 2,000문항 모델의 평균 출력은 102.43토큰이었다. 발표에서는 같은 단답 프롬프트를 써야 했기 때문에, 도식 판독과 다단계 추론이 필요한 외부 문항을 골라 자연어 풀이를 학습하고 출력을 줄였을 때의 정확도를 확인했다.

1. Geometry3K·GeoQA·M³CoT science의 train에서 객관식을 32개씩 골랐다. 같은 base에서 자연 풀이 Full-CoT를 8스텝 학습한 뒤, 풀이 앞부분을 점진적으로 제거하는 ICoT-SI를 적용했다. 3 GPU, GPU당 배치 1, 누적 4로 업데이트당 12문항이었다. SI 16스텝과 24스텝을 비교해 16스텝 체크포인트를 채택했다. [학습 설정](recipes/natural_si.toml) · [제거 스케줄과 체크포인트 검증](evidence/si_schedule.json)
2. SI16이 단답으로 틀렸던 외부 문항과 기존 정답을 보존할 문항을 짝지어 추가 학습했다. 발표 조건에서 SI16은 482개, 첫 회복·보존 학습은 492개, 확대한 모델은 495개를 맞혔다.
3. 495에서 외부 학습 문항 12개(회복 6·보존 6)로 6스텝 학습했다. 정답 토큰 손실과 교사 모델의 선택지 확률 분포를 함께 사용했다. 두 번째 에포크에는 선택지와 교사 확률의 순서를 함께 바꾸어 전체 두 에포크에서 정답 위치 A–D가 각 6회가 되도록 했다. 495 대비 9개를 회복하고 6개를 놓쳐 498개가 됐다. 학습률은 1e-5, 업데이트당 4문항, 언어 모델 q/k/v/o LoRA의 학습 파라미터는 5,898,240개였다. [문항과 교사 분포](recipes/498.json) · [6스텝 기록](evidence/training_498.json)

## 498을 유지한 이유

498에서 새 외부 문항 12개를 두 후보에 똑같이 사용했다. 회복 6개와 보존 6개였고, 보존 문항은 기존 모델들의 답이 갈린 3개와 모두 맞혔지만 498의 정답 마진이 작았던 3개였다. 두 후보 모두 6스텝 학습했다. 짧은 풀이 후보는 첫 에포크의 회복 문항 6개에만 풀이를 붙였다. 외부 검증 141개는 학습 전에 분리했다. [후보별 학습 타깃 및 순서](recipes/paired.json)

| 모델 | 외부 전체 /141 | 회복군 /23 | 보존군 /38 | MMMU /900 | 평균 출력 토큰 | 생성 시간/문항 |
|---|---:|---:|---:|---:|---:|---:|
| 498 | 76 | 0 | 38 | **498 (55.33%)** | 4.19 | 0.155초 |
| 단답 추가 학습 | **81** | **5** | 38 | 494 (54.89%) | 2.90 | 0.148초 |
| 첫 에포크에 짧은 풀이 | 79 | 3 | 38 | 미실행 | — | — |

외부 전체 정답 수가 가장 높은 단답 후보만 MMMU에서 평가했다. 498이 틀린 4문항을 회복했지만 기존 정답 8문항을 놓쳤다. 두 모델의 차이에 대한 양측 exact McNemar p값은 0.388이었다. 발표 모델을 선택하는 과정에서 같은 MMMU validation을 여러 번 사용했으므로 55.33%는 이 검증 세트에서 관측한 값이다. 시간은 배치 생성 구간의 총 경과 시간을 900으로 나눈 수치다. [외부 141문항](evidence/external.csv) · [MMMU 900문항](evidence/mmmu.csv) · [답이 달라져 정오가 바뀐 12문항](evidence/changes_vs_498.csv)

팀에서 찾던 **baseline 주관식 오답 46개**는 2026-09-28 보관 실행의 주관식 53문항 중 오답이다. 문항 ID, 과목, 채점기가 추출한 답, 기준 정답을 담았다. [46문항 목록](evidence/baseline_open_wrong_46.csv)

## 재현

저장소 루트에서 실행한다. 재현 범위는 **495에서 498까지의 학습, 498의 발표 조건 재평가, 마지막 두 후보의 학습, 보관된 후보 비교 결과 집계**다. 이전 Full-CoT와 SI16 단계는 학습 설정과 토큰 제거 코드, 실행 당시 검증 기록으로 남겼다. [토큰 제거 구현](../code/train/icot_si.py)

**저장된 결과 검산**에는 GPU나 추가 패키지가 필요 없다. 두 어댑터의 SHA-256, 발표 추론·채점 코드 해시, MMMU 정오 변화와 외부 검증 점수를 확인한다.

```bash
python3 icot/check_results.py
```

**498 재평가**에는 저장소의 Docker 환경 또는 같은 패키지 버전과 24GB급 GPU 한 장을 사용한다. 실험 환경은 RTX 3090, Python 3.12.3, torch 2.9.0, vLLM 0.11.2, transformers 4.57.6이었다. 데이터 경로는 컨테이너 기준이다. 모델 다운로드가 끝난 환경에서는 첫 명령을 생략할 수 있다. 병합된 모델 디렉터리와 결과 디렉터리에는 새 경로를 사용한다.

```bash
bash code/scripts/fetch_assets.sh
mkdir -p results/icot_export
python code/train/export_mixed_checkpoint.py \
  --checkpoint icot/adapters/498 --output models/icot_498_merged \
  --preserve-inference-assets --verification results/icot_export/498.json
CUDA_VISIBLE_DEVICES=0 DATA_PATH=/opt/mmdl/data \
  bash icot/evaluate.sh models/icot_498_merged results/icot_498_eval
```

평가는 MMMU 고정 revision의 30과목, 객관식 847개와 주관식 53개를 읽는다. 발표 단답 프롬프트, greedy, 출력 상한 512토큰, 동시 시퀀스 2개로 생성한다. 실행 디렉터리의 `mmmu/scores.json`에서 `structured.overall.correct`를 확인한다. 실험 당시 채점기 두 파일은 `icot/scoring/`에 고정했다. [원 실행의 환경과 모델·코드 해시](evidence/evaluation_498.json) · [평가 설정](../configs/evaluation.toml)

**마지막 학습을 다시 실행**하려면 Geometry3K의 공식 train ZIP과 ChartQA의 지정 ZIP을 준비한다. 복원 스크립트가 공식 질문, 선택지, 이미지와 저장된 학습 타깃을 연결한다. 두 실험을 합쳐 고유 학습 문항은 24개다. [Geometry3K train ZIP](https://lupantech.github.io/inter-gps/geometry3k/train.zip) · [ChartQA 고정 revision ZIP](https://huggingface.co/datasets/ahmed-masry/ChartQA/resolve/af8b6f5c08c95085271561c2a3f9d15f2b5a9031/ChartQA%20Dataset.zip)

```bash
python icot/prepare_training.py --geometry-zip /path/to/train.zip --chart-zip /path/to/chartqa.zip
CUDA_VISIBLE_DEVICES=0 python icot/train_498.py
CUDA_VISIBLE_DEVICES=0 python icot/train_paired.py answer
CUDA_VISIBLE_DEVICES=0 python icot/train_paired.py short
```

첫 학습은 보관된 495에서 시작해 `checkpoints/icot_498`에 저장한다. 뒤의 두 학습은 각각 보관된 498에서 독립적으로 시작해 `checkpoints/icot_paired_answer`, `checkpoints/icot_paired_short`에 저장한다. 새 어댑터를 평가할 때는 위 병합 명령의 체크포인트와 출력 경로를 해당 후보에 맞게 지정한다.
