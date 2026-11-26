# ICoT 최종 모델 재현 · MMMU test와 MMMU-Pro

Qwen3-VL-4B-Instruct를 T0로 두고, 자연 풀이의 점진 제거와 회복·보존 학습으로 만든 498 모델에 외부 TabMWP 표 640문항을 추가 학습했다. **498**은 MMMU validation 정답 수이고, **640**은 마지막 학습의 고유 문항 수다. [495에서 498을 재현하는 방법과 앞 단계 실험 기록](../code/icot/README.md)

| 평가 | T0 | 498 | 최종 640 |
|---|---:|---:|---:|
| MMMU test 정답 / 10,500 | 5,090 | 5,230 | **5,305** |
| MMMU test 정확도 | 48.48% | 49.81% | **50.52%** |
| MMMU test 평균 출력 토큰 | 50.64 | 4.19 | **2.34** |
| Pro Standard 10지 / 1,730 | 674 | 633 | 664 |
| Pro Vision / 1,730 | 567 | 537 | 547 |
| **Pro Overall** | **35.87%** | **33.82%** | **35.00%** |
| Pro Standard 4지 / 1,730 (별도) | 840 | 879 | 880 |

Overall은 같은 1,730문항을 Standard 10지와 Vision으로 평가한 정확도의 산술평균이다. 640은 T0보다 test에서 215개 더 맞혔고 Pro 두 형식의 합계에서는 30개 덜 맞혔다. MMMU validation에서는 498 → 488/900이었다. 모델 설계 시 test·Pro 오류 유형을 참고했고, 추가 학습 예시는 TabMWP 공식 train에서만 가져왔다. [공식 채점 기준 최종 수치](../results/tab640/final.json)

## 저장한 640 모델로 최종 점수 재현

저장소 루트의 Linux 셸에서 시작한다. 첫 줄에서 이동하는 `code/`가 이후 모든 호스트 명령의 작업 디렉터리다. 아래는 기존 프로젝트 Dockerfile로 고정한 실행 환경이다. 원 실험은 RTX 3090 24GB, BF16, vLLM 0.11.2, Transformers 4.57.6이었다. 첫 빌드와 평가 데이터 준비에는 인터넷이 필요하다. 병합된 모델은 약 9GB이며 추론에는 24GB급 GPU 한 장을 사용한다.

```bash
cd code
docker build -t mmdl-tab640 .
mkdir -p results/reproduction
docker run --rm -it --gpus all --shm-size=16g \
  -v "$PWD/results/reproduction:/opt/mmdl/results/reproduction" mmdl-tab640 bash
```

컨테이너의 기본 작업 디렉터리 `/opt/mmdl`에서 실행한다. 결과 디렉터리는 각 실행에 새 이름을 사용한다. 데이터 스크립트는 지정한 Hugging Face revision의 test Parquet와 MMMU 공식 Git revision의 정답 파일·Pro 파서를 다운로드한다. 병합기는 원본 base 및 어댑터 해시와 병합된 텐서를 검증한다.

```bash
python code/tab640/fetch_eval.py
python code/train/export_mixed_checkpoint.py --checkpoint icot/adapters/640 \
  --output models/icot640 --preserve-inference-assets \
  --verification results/reproduction/export640.json
CUDA_VISIBLE_DEVICES=0 python code/tab640/evaluate.py models/icot640 results/reproduction/640
python code/tab640/score.py results/reproduction/640 > results/reproduction/640-scores.json
cat results/reproduction/640-scores.json
```

T0는 같은 명령에서 모델 경로로 `models/Qwen3-VL-4B-Instruct`를 바로 사용한다. 498은 `icot/adapters/498`을 별도 경로에 병합한 뒤 동일하게 평가한다. 실행별 생성 파일을 공식 파서로 채점하며 고정 seed 42로 대체 선택을 재현한다. 기존 GPU 세 장에서 독립적으로 처리한 36문항 단위 묶음을 이 진입점에서는 한 GPU가 차례로 처리한다. 프롬프트·생성 설정·채점 순서를 유지한다. 장비, 라이브러리 버전, GPU 커널에 따라 일부 답이 달라질 수 있다. [평가 설정](../code/configs/evaluation.toml) · [추론](../code/code/tab640/evaluate.py) · [채점](../code/code/tab640/score.py)

## 498에서 640 추가 학습 재현

1. **학습 데이터 복원.** 공식 train에서 498이 숫자 단답은 맞히고 객관식 보기는 틀린 외부 표를 골랐다. 범위 204개, 평균 203개, 중앙값 203개, 잔돈 30개다. 최종 명세에는 원본 ID·보기 순서·정답·픽셀 해시를 기록했다. 아래 호스트 명령은 원본을 다운로드해 640개 표를 렌더링하고, 두 학습 단계의 예시 1,280개를 실제 순서대로 복원한다. 원래 렌더러와 맞추기 위해 Ubuntu 22.04의 Pillow 및 DejaVu 글꼴을 사용한다.

```bash
docker run --rm -v "$PWD:/repo" -w /repo ubuntu:22.04 bash -ec \
  'apt-get update -qq && apt-get install -y -qq python3-pil fonts-dejavu-core ca-certificates && python3 code/tab640/prepare.py'
```

2. **세 장의 GPU에서 학습.** 첫 160스텝은 보기 순서를 바꿔 계산값과 문자를 함께 지도하고, 다음 160스텝은 원래 보기에서 문자만 지도한다. base를 고정하고 498의 언어 attention LoRA 5,898,240개 파라미터를 갱신한다. 유효 배치 4, 학습률 1e−5, AdamW, seed 42를 사용한다. 호스트의 새 학습 출력은 `checkpoints/tab640_retrained`에 남는다.

```bash
mkdir -p checkpoints
docker run --rm --gpus all --shm-size=16g \
  -v "$PWD/training_data/tab640:/opt/mmdl/training_data/tab640:ro" \
  -v "$PWD/checkpoints:/opt/mmdl/checkpoints" \
  -e OMP_NUM_THREADS=2 -e OPENBLAS_NUM_THREADS=2 -e MKL_NUM_THREADS=2 \
  -e TOKENIZERS_PARALLELISM=false -e CUDA_VISIBLE_DEVICES=0,1,2 mmdl-tab640 \
  torchrun --standalone --nproc_per_node=3 code/tab640/train.py
```

재학습본을 평가할 때는 첫 번째 컨테이너 실행 명령에 `-v "$PWD/checkpoints:/opt/mmdl/checkpoints:ro"`를 추가한다. 병합 명령의 체크포인트를 `checkpoints/tab640_retrained`로 바꾸고 병합·평가 출력에는 새 경로를 지정한다. [학습 루프](../code/code/tab640/train.py) · [최종 학습 순서](../code/code/tab640/recipe.tsv) · [제출한 640 체크포인트의 SHA-256](../code/icot/adapters/640/checkpoint.json)
