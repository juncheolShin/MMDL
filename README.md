# MMDL · 멀티모달딥러닝 10조 도비

Qwen3-VL-4B-Instruct의 단답 정확도와 출력 길이를 MMMU와 MMMU-Pro에서 평가했다. MMMU test 10,500문항에서 베이스라인은 5,090개, ICoT 498 모델은 5,230개, 표 640문항 추가 학습 모델은 5,305개를 맞혔다. MMMU-Pro Overall은 각각 35.87%, 33.82%, 35.00%였다.

| 제출 항목 | 내용 |
|---|---|
| [assignment/](assignment/) | [Baseline 과제 보고서](assignment/assignment1.md) · [ICoT 최종 모델 보고서와 재현 명령](assignment/icot640.md) |
| [code/](code/) | [실행 환경과 전체 안내](code/README.md) · [평가 코드](code/code/eval/) · [파인튜닝 코드](code/code/train/) · [표 640문항 학습·평가 코드](code/code/tab640/) |
| [results/](results/) | [실험 설정과 결과](results/README.md) · [T0·498·640 최종 점수 JSON](results/tab640/final.json) |

## 재현

실행 명령의 작업 디렉터리는 최상위의 `code/`다. 다음은 저장소를 복제한 위치에서 GPU 없이 평가 프롬프트를 점검하고 증거 지도 실험의 저장된 결과를 집계하는 예다.

```bash
cd code
python3 code/eval/tests/test_prompting.py
python3 experiments/evidence_supervision/make_evidence.py
```

최종 640 모델을 직접 평가하거나 498에서 다시 학습하는 절차는 [최종 모델 재현 안내](assignment/icot640.md)에 있다. Docker 빌드에는 `code/`를 컨텍스트로 사용한다. 다른 팀 실험은 [전문 시각 문항·선택적 증류](code/experiments/icot_retention/README.md)와 [증거 지도 실험](code/experiments/evidence_supervision/README.md)에서 재현할 수 있다.
