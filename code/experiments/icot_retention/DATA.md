# 실험 데이터 재현

실험에 사용한 문항의 순서와 원자료 위치는 [데이터 목록](data_recipe/recipe.json)에 고정했다. 준비 프로그램은 이 목록에 지정된 버전에서 질문·보기·정답과 이미지를 읽어 실제 학습 프롬프트를 복원한다. 아래 명령은 저장소 최상위에서 `cd code` 후 프로젝트의 Python 환경에서 실행한다.

## 공개 원자료로 만들기

```bash
python experiments/icot_retention/prepare_data.py --download --output _workspace/retention/data
```

Hugging Face 캐시에 원자료 약 4.4GiB와 출력에 약 1.8GiB가 필요하다. 출력 경로는 새 디렉터리 또는 빈 디렉터리를 지정한다. 기존 데이터가 있으면 문항과 이미지의 무결성을 다시 확인한다.

이미 다운로드한 자료를 사용하는 경우 아래 예처럼 원자료 디렉터리를 지정한다. `--source-root`는 여러 번 사용할 수 있다. 상위 디렉터리 아래에 카탈로그의 파일 키와 같은 상대 경로를 배치하거나, 각 데이터셋의 공식 Hugging Face 파일 구조를 유지한 디렉터리를 개별 지정한다.

```bash
python experiments/icot_retention/prepare_data.py \
  --source-root /path/to/public-sources --source-root /path/to/specialist-sources \
  --output _workspace/retention/data
python experiments/icot_retention/prepare_data.py --verify-only --output _workspace/retention/data
```

## 분할과 검증

| 출력 | 문항 수 | 용도 |
|---|---:|---|
| `public/train.jsonl` | 18,956 | A-OKVQA 8,983, M³CoT 6,973, SciVQA 1,500, PMC-VQA v2 1,500문항으로 교사 판정과 공동학습 |
| `public/dev.jsonl` | 1,044 | 공식 훈련 자료에서 보류한 A-OKVQA 660, M³CoT 384문항으로 체크포인트·증류 강도 선택 |
| `emma/dev.jsonl` | 320 | EMMA 공식 공개 test에서 따로 고른 수학·물리·화학·코딩 각 80문항으로 증류 강도 선택 |

각 문항의 이미지 경로는 해당 분할이 있는 `public/` 또는 `emma/`를 기준으로 한다. 각 디렉터리의 `data_manifest.json`에는 JSONL의 SHA-256과 문항 수가 있다. 원자료 파일, 각 이미지의 원본 바이트와 디코딩된 RGB 픽셀을 검증한 뒤 세 JSONL의 SHA-256이 실험에 사용한 파일과 같은지 확인한다. 임시 디렉터리에서 모든 분할을 완성한 후 출력 디렉터리에 함께 반영한다. MMMU 공식 validation 900문항은 모델 최종 평가 단계에서 저장소의 기존 MMMU 평가기를 통해 읽는다.

세 분할 사이에 같은 문항 ID나 같은 이미지 픽셀은 없다. 보기와 부가 설명을 제외하고 질문 문구를 정규화해 비교하면, 공개 검증 1,044문항 중 300문항은 훈련에 등장한 질문을 다른 이미지에 적용한다. 예를 들어 “그림에서 무엇을 추론할 수 있는가?”라는 질문이 서로 다른 두 이미지에 쓰인다.

| 원자료 | 이 실험에서의 선택 |
|---|---|
| [A-OKVQA](https://huggingface.co/datasets/HuggingFaceM4/A-OKVQA), [M³CoT](https://huggingface.co/datasets/LightChen2333/M3CoT) | 단일 이미지와 정답이 있는 객관식. M³CoT의 원래 문항 맥락을 포함하고 풀이 과정은 학습에 사용하지 않음 |
| [SciVQA](https://huggingface.co/datasets/katebor/SciVQA) | 과학 논문 그림의 원래 시각형 비이진 네 보기 문항 중 단일 정답이 있고 미제공 본문을 명시적으로 요구하지 않는 문항 |
| [PMC-VQA v2](https://huggingface.co/datasets/RadGenome/PMC-VQA) | 비복합 그림 v2의 원래 네 보기 문항. 정답 글자 A–D 각 375개를 유지 |
| [EMMA](https://huggingface.co/datasets/EMMAModal/EMMA) | 단일 이미지 객관식 중 출처가 MMMU인 문항과 확인 가능한 문항·이미지 중복을 제외한 외부 검증 부분집합 |
