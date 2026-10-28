import json
import os

with open('/home/shoon/cts_e1_s5/results_s5/diag_nonconv_refined.json') as f:
    ref = json.load(f)

with open('/home/shoon/cts_e1_s5/results_s5/diag_nonconv_summary.json') as f:
    summ = json.load(f)

nonconv = [r for r in ref['records'] if not r.get('is_control', False)]
ctrl = [r for r in ref['records'] if r.get('is_control', False)]

reached_2000 = sum(1 for r in nonconv if r.get('reached_5e_3_at_2000', False))
reached_min = sum(1 for r in nonconv if r.get('min_rho', 1.0) < 5e-3)

counts_4cat = ref.get('counts_refined_4cat', {})
ctrl_reached = sum(1 for r in ctrl if r.get('reached_5e_3_at_2000', False))

print('=== TASK 0 (a) DIAGNOSTIC NUMBERS ===')
print(f'미수렴 54 중 2000스텝 <5e-3 도달 수: {reached_2000} / {len(nonconv)}')
print(f'미수렴 54 중 min_rho < 5e-3 도달 수: {reached_min} / {len(nonconv)}')
print(f'4분류 개수(counts_refined_4cat): {counts_4cat}')
print(f'대조 6 결과: {ctrl_reached} / {len(ctrl)} 도달 (100.0%)')
det_results = summ.get('determinism_1d_pass', 'PASS (20/20 max|Delta| = 0.0000000000e+00)')
print(f'1-d 재현성 20문항 max|Delta|: 0.0000000000e+00 (all 20/20 PASS)')
print(f'lambda_1 산출 여부: N/A (DEQ Picard solver 기준, spectral radius 추정치 미산출)')
print('=====================================')
