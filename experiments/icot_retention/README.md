# ICoT 후속 실험 · 전문 시각 문항과 선택적 증류

기존 ICoT의 MMMU 정답 수를 늘리기 위해 외부 객관식 18,956문항으로 vision encoder·merger·언어 attention LoRA를 함께 학습했다. 증류 강도 네 가지를 비교한 결과, 외부 검증에서 미리 선정한 λ=0.5 모델은 MMMU에서 **488/900**을 기록했다. 이 환경에서 재평가한 기존 ICoT는 **500/900**이었고, 학생 모델 중 MMMU 최고 점수는 λ=0.25의 **494/900**이었다. 훈련 데이터, 학습 손실, 후보 선택 기준과 문항별 변화는 [실험 보고서](REPORT.md)에 정리했다.

실행할 브랜치는 `experiment/icot-specialist-retention`이다. 처음 내려받을 때 `GIT_LFS_SKIP_SMUDGE=1 git clone -b experiment/icot-specialist-retention https://github.com/juncheolShin/MMDL.git`을 사용하면 필요한 모델만 2절에서 받을 수 있다.

아래 명령은 저장소 루트에서 실행한다. 결과만 확인하려면 1절, 배포한 모델을 직접 평가하려면 2절, 같은 데이터로 처음부터 학습하려면 3절을 사용한다.

## 1. 저장된 예측으로 결과 확인

Python 3.12에서 실행한다. GPU, 원자료 다운로드, 모델 다운로드 없이 작동한다.

```bash
python experiments/icot_retention/evaluation.py
```

이 명령은 외부 검증 문항별 예측으로 각 강도의 체크포인트와 대표 모델을 다시 고른다. 이어서 교사와 학생 4개의 MMMU 응답 총 4,500개를 팀 발표 때 사용한 채점기로 다시 채점한다. 출력의 `gained`는 교사의 오답을 학생이 새로 맞힌 수, `lost`는 교사의 정답을 학생이 놓친 수다. [선정 기록](evidence/selection.json) · [최종 점수](evidence/summary.json)

## 2. 배포한 학생 모델 복원 및 평가

기준 환경은 Linux, Python 3.12, CUDA 12.8, RTX 4090 24 GB다. MMMU 평가에는 GPU 한 장, 전체 학습에는 네 장을 사용한다. 배포 모델을 받으려면 Git LFS가 필요하다. Ubuntu에서는 `sudo apt-get install git-lfs`로 설치할 수 있다. 교사 모델이 이미 준비돼 있으면 복원 자체는 Python과 NumPy만 사용한다.

```bash
EX=experiments/icot_retention
WORK="$PWD/results/icot_retention"
mkdir -p "$WORK"
python -m pip install -r code/requirements-eval.txt -r code/requirements-train.txt \
  'Pillow==12.3.0' 'pyarrow==26.0.1' 'safetensors==0.8.0'
git lfs install --local
```

먼저 Qwen의 지정된 리비전과 저장소에 있는 팀 ICoT 어댑터를 병합해 **교사 모델**을 만든다. `--preserve-inference-assets`는 이 실험에 사용한 토크나이저와 설정 파일을 그대로 복원하기 위해 필요하다.

```bash
bash code/scripts/fetch_assets.sh
python code/train/export_mixed_checkpoint.py \
  --base models/Qwen3-VL-4B-Instruct --checkpoint icot/adapters/498 \
  --preserve-inference-assets --output "$WORK/teacher" \
  --verification "$WORK/teacher-export.json"
```

각 강도는 독립된 Git LFS 파일 하나다. 네 파일을 모두 받으면 **415.5 MB**, 대표 모델 λ=0.5만 받으면 **104.0 MB**다. 시각 가중치는 교사와 달라진 BF16 값을 압축했다. 언어 attention은 학습한 LoRA 행렬과, 병합 시 반올림 차이가 난 위치의 정확한 BF16 값을 저장했다. 나머지 가중치와 토크나이저는 교사 모델에서 복사한다. 복원한 모델은 각각 약 8.9 GB이며, 복원 명령이 평가 당시 파일의 SHA-256과 일치하는지 검사한다. [모델별 파일 크기와 SHA-256](weights/manifest.json) · [압축 도구](compact_weights.py)

| 증류 강도 | 다운로드 (MB, 1 MB = 10⁶ 바이트) |
|---|---:|
| 0 | 104.6 |
| 0.1 | 104.0 |
| 0.25 | 102.8 |
| 0.5 | 104.0 |

```bash
STRENGTH=0.5
git lfs pull --include="$EX/weights/lambda-$STRENGTH.zip" --exclude=""
python "$EX/model.py" --base "$WORK/teacher" --strength "$STRENGTH" \
  --output "$WORK/student-$STRENGTH"
CUDA_VISIBLE_DEVICES=0 DATA_PATH="$PWD/data" HF_HOME="$PWD/data" \
  bash icot/evaluate.sh "$WORK/student-$STRENGTH" "$WORK/mmmu-$STRENGTH"
```

결과는 `$WORK/mmmu-$STRENGTH/mmmu/scores.json`의 `structured.overall.correct`에 저장된다. 교사를 비교하려면 같은 명령에서 모델을 `"$WORK/teacher"`로, 출력 디렉터리를 `"$WORK/mmmu-teacher"`로 바꾼다. MMMU는 고정된 리비전의 메타데이터를 조회하므로 최초 실행 시 Hugging Face에 접속할 수 있어야 한다. 매 평가에 새 출력 경로를 사용한다.

## 3. 같은 데이터로 네 조건 다시 학습

2절에서 만든 교사 모델과 Python 환경을 사용한다. 공개 원자료 준비에 캐시 약 4.4 GiB와 재구성한 데이터 약 1.8 GiB가 필요하며, 전체 학습에는 네 모델과 중간 체크포인트를 저장할 추가 공간이 필요하다. 이미 받은 원자료의 경로를 지정하는 방법과 검증 절차는 [데이터 준비 안내](DATA.md)에 있다.

```bash
python "$EX/prepare_data.py" --download --output "$WORK/data"
CUDA_VISIBLE_DEVICES=0,1,2,3 python "$EX/reproduce.py" \
  --teacher "$WORK/teacher" --data "$WORK/data" \
  --mmmu-data "$PWD/data" --run "$WORK/sweep"
```

진행 로그는 `$WORK/sweep/attempts/<단계>-*/process.log`에 저장한다. 중단된 경우 같은 인자와 환경으로 다시 실행하면 완료한 단계의 출력 해시를 확인하고, 완료하지 못한 단계부터 새 디렉터리에서 다시 실행한다.

실행 호스트에 메일 전송용 `sendmail`이 설정되어 있으면 `reproduce.py` 명령에 `--notify-email 수신주소`를 추가해 단계 실패나 중단 신호를 메일로 받을 수 있다. 호스트 전원이 꺼지거나 실행기가 강제 종료(SIGKILL)되는 상황까지 감시하려면 호스트 외부의 작업 스케줄러가 필요하다.

1. 고정한 교사가 외부 훈련 18,956문항과 A-OKVQA·M³CoT 검증 1,044문항을 푼다. 선택지별 확률과 교사의 정오를 문항마다 저장한다.
2. 네 학생이 각각 같은 교사 가중치에서 한 에포크(2,370스텝)를 학습한다. 모든 문항에서 실제 정답을 학습하고, 교사가 맞힌 문항에만 해당 강도의 증류 손실을 더한다. 매 학습 실행은 GPU 네 장을 함께 쓴다. [학습 설정](recipe.toml) · [손실과 체크포인트 선택](training/trainer.py)
3. 각 강도에서 A-OKVQA·M³CoT 검증으로 체크포인트를 고른다. EMMA 검증 320문항을 추가로 평가해 두 검증셋 정확도의 평균으로 대표 강도를 확정한다.
4. 대표 모델을 확정한 뒤 교사와 네 학생을 MMMU **공식 validation 900문항 전체**에서 평가한다. 재실행 결과는 아래 명령으로 다시 집계할 수 있다.

```bash
python "$EX/evaluation.py" --evidence "$WORK/sweep/evidence"
```

손실 계산, 데이터 재구성, 모델 복원, 평가와 실행 순서를 점검하는 테스트는 학습용 Python 환경에서 실행한다.

```bash
python -m unittest discover -s experiments/icot_retention/tests -v
```

## 4. 다른 증류 강도 한 가지 실험

같은 교사와 데이터를 유지하고 증류 강도만 바꾸려면 먼저 교사 예측을 한 번 저장한다. 다음은 새 조건 λ=0.2의 실행 예시다. 여러 강도를 시험할 때 첫 번째 명령의 교사 예측을 재사용하고 두 번째 명령의 강도와 출력 경로만 바꾼다.

```bash
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --standalone --nproc-per-node=4 "$EX/training/teacher.py" \
  --model "$WORK/teacher" --data-dir "$WORK/data/public" --output-dir "$WORK/teacher-scores"
STRENGTH=0.2
torchrun --standalone --nproc-per-node=4 "$EX/training/train.py" \
  --model "$WORK/teacher" --data-dir "$WORK/data/public" --teacher-dir "$WORK/teacher-scores" \
  --distill-weight "$STRENGTH" --output-dir "$WORK/new-student-$STRENGTH"
torchrun --standalone --nproc-per-node=4 "$EX/training/teacher.py" \
  --model "$WORK/new-student-$STRENGTH" --data-dir "$WORK/data/emma" \
  --expert-output "$WORK/emma-$STRENGTH.json"
```

선택된 공개 검증 체크포인트는 학생 모델 디렉터리의 `export.json`에, EMMA의 정답 수는 `emma-$STRENGTH.json`의 `summary.correct`에 저장된다.
