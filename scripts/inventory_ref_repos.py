#!/usr/bin/env python3
"""
scripts/inventory_ref_repos.py - Inventory LocusLab DEQ reference repositories
Directive: EXE-260928-40 Task 1 (원장 #376)
"""

import os
import subprocess

REF_DIR = "/home/shoon/ref"
REPOS = ["deq", "mdeq", "torchdeq"]

KEYWORDS = [
    "broyden", "anderson", "line_search", "f_thres", "b_thres", "stop_mode",
    "lowest", "power_method", "jac_loss_estimate", "weight_norm",
    "VariationalHidDropout", "backward", "damping", "beta", "phantom"
]

inventory_items = []

# Core curated mappings of key functions to inspect
curated_targets = [
    ("deq", "lib/solvers.py", "broyden", "L-Broyden准뉴턴 고정점 솔버", "④ 준뉴턴 마무리", "사용 권고 (정본 L-Broyden 기준)"),
    ("deq", "lib/solvers.py", "anderson", "Anderson 가속 고정점 솔버", "④ 준뉴턴 마무리", "사용 권고 (정본 Anderson 기준)"),
    ("deq", "lib/solvers.py", "line_search", "Broyden용 backtracking 선형탐색", "④ 준뉴턴 마무리", "참고 권고 (발산 방지 안전망)"),
    ("deq", "lib/solvers.py", "broyden (lowest)", "최저 잔차 x 기억 반환 (lowest)", "①/④ 잔차 안정성", "적극 사용 권고 (진동 시 최저점 확보)"),
    ("deq", "lib/jacobian.py", "power_method", "JVP 기반 거듭제곱법 스펙트럼 반경 추정", "스펙트럼 진단", "적극 사용 권고 (수렴성 엄밀 진단)"),
    ("deq", "lib/jacobian.py", "jac_loss_estimate", "Hutchinson 트레이스 기반 야코비안 노름 정규화", "③ f 평활화", "참고 권고 (S8 학습 정규화)"),
    ("deq", "lib/optimizations.py", "VariationalHidDropout", "반복 간 일관 드롭아웃 마스크", "③ f 평활화", "참고 권고 (DEQ 드롭아웃 안정화)"),
    ("deq", "lib/layer_utils.py", "list2vec / vec2list", "임의 텐서 구조 1차원 평탄화/복원", "④ 텐서 형상 변환", "적극 사용 권고 (64x2560 평탄화)"),
    ("deq", "DEQ-Sequence/models/deq_transformer.py", "Backward hook (deq_func)", "음함수 미분 역방향 고정점 역전파", "⑤ 음함수 미분 학습", "참고 권고 (S8 디코더 학습 설계)"),
    ("mdeq", "lib/modules/deq_module.py", "DEQModuleList", "다중 해상도 DEQ 순전파/역전파 모듈", "⑤ 음함수 미분 학습", "참고 권고"),
    ("torchdeq", "torchdeq/solver/broyden.py", "broyden", "모듈화된 L-Broyden 솔버 (PyTorch 패키지)", "④ 준뉴턴 마무리", "참고 권고 (코드 리팩토링용)"),
    ("torchdeq", "torchdeq/solver/anderson.py", "anderson", "모듈화된 Anderson 솔버", "④ 준뉴턴 마무리", "참고 권고"),
    ("torchdeq", "torchdeq/grad.py", "phantom_grad", "Phantom Gradients (음함수 미분 대안 근사)", "⑤ 음함수 미분 학습", "참고 권고 (안정적 1-step 근사)"),
    ("torchdeq", "torchdeq/loss/jacobian.py", "jacobian_loss", "Jacobian 정규화 손실 모듈", "③ f 평활화", "참고 권고"),
]

table_rows = []
for repo, rel_path, func_name, role, roadmap, rec in curated_targets:
    full_path = os.path.join(REF_DIR, repo, rel_path)
    line_num = "N/A"
    if os.path.exists(full_path):
        # find line number
        try:
            cmd = f"grep -n -m 1 '{func_name}' {full_path}"
            out = subprocess.check_output(cmd, shell=True, text=True).strip()
            if out:
                line_num = out.split(":")[0]
        except Exception:
            pass
    table_rows.append({
        "repo": repo,
        "location": f"{rel_path}:{line_num}",
        "func": func_name,
        "role": role,
        "roadmap": roadmap,
        "rec": rec
    })

print("=" * 120)
print(f"{'저장소':<8} | {'파일:행':<35} | {'함수/메커니즘':<20} | {'역할 (하는 일)':<35} | {'로드맵 연결':<15} | {'권고'}")
print("=" * 120)
for r in table_rows:
    print(f"{r['repo']:<8} | {r['location']:<35} | {r['func']:<20} | {r['role']:<35} | {r['roadmap']:<15} | {r['rec']}")
print("=" * 120)
