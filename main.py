# main.py 또는 run_calphad.py 예시

import config

print(f"사용 데이터베이스: {config.DATABASE}")
print(f"계산 온도: {config.T_EQUIL_C} °C ({config.T_EQUIL_K} K)")
print(f"Ce 그리드 범위: {config.CE_GRID_WT}")
print(f"Cr 그리드 범위: {config.CR_GRID_WT}")

# 이후 config.DATABASE, config.ELEMENTS 등을 활용하여
# tc_python (Thermo-Calc Python API) 세션을 열고 평형 계산 루프를 수행합니다.