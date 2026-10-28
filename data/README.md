# 실험 데이터 (로컬)

`scripts/download_experiment_data.py`로 받은 파일을 둡니다.

| 경로 | 출처 | 설명 |
|------|------|------|
| `math500/test.jsonl` | [HuggingFaceH4/MATH-500](https://huggingface.co/datasets/HuggingFaceH4/MATH-500) | 500문항, `problem` / `answer` 등 |
| `openmath_instruct/train_<N>.jsonl` | [nvidia/OpenMathInstruct-1](https://huggingface.co/datasets/nvidia/OpenMathInstruct-1) | **스트리밍 상위 N행만** (기본 스크립트 N=100000; 전체 1.8M+ 미포함) |
| `stage2/math_train_prompts_<N>.jsonl` | [EleutherAI/hendrycks_math](https://huggingface.co/datasets/EleutherAI/hendrycks_math) | Stage2 PPO용 프롬프트 (기본 N=5000) |

벤치 실행 예:

```powershell
python scripts/run_math500.py --data data/math500/test.jsonl --limit 50
```

전체 OpenMath를 받으려면 HF에서 별도 `snapshot_download`·디스크 여유가 필요합니다 (수십 GB급 가능).
