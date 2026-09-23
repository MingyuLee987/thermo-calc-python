# config.py

from pathlib import Path
import numpy as np

ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

DATA_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# 반드시 본인 Thermo-Calc 설치/라이선스의 실제 database 이름으로 바꾸세요.
DATABASE = "TCFE13"

# Fe는 dependent element로 두고, 나머지는 wt%로 입력합니다.
ELEMENTS = [
    "FE", "C", "SI", "MN", "CU", "NI",
    "P", "S", "CR", "CE"
]

# POSEIDON 500 기준 조성 예시.
# 실제 heat analysis 또는 연구실 기준조성으로 반드시 교체하세요.
BASE_WT = {
    "C": 0.060,
    "SI": 0.250,
    "MN": 0.800,
    "CU": 0.300,
    "NI": 0.300,
    "P": 0.015,
    "S": 0.005
}

# 1차 연구는 2변수만: Ce와 Cr
CE_GRID_WT = np.round(np.arange(0.000, 0.0501, 0.005), 4)
CR_GRID_WT = np.round(np.arange(0.80, 1.401, 0.10), 3)

# 평형 평가 온도:
# "최종 열처리 후의 상 안정성"을 볼 온도로 설정.
T_EQUIL_C = 700.0
T_EQUIL_K = T_EQUIL_C + 273.15
P_PA = 1.0e5

# Matrix phase는 반드시 첫 GUI / 단일점 계산 결과로 확인한 뒤 수정하세요.
MATRIX_PHASE = "BCC_A2"

# 유해상으로 간주할 phase 이름.
# database에 없는 phase 이름은 자동으로 0으로 처리하게 코드 작성.
# 이 목록 자체도 강종·공정에 맞게 바꿔야 합니다.
HARMFUL_PHASES = [
    "SIGMA",
    "MU_PHASE",
    "CHI",
    "LAVES_C14",
    "LAVES_C15"
]

# 처음에는 너무 공격적인 제약을 피하세요.
# CALPHAD 결과를 보고 1차 screening 후 조정합니다.
MIN_MATRIX_FRACTION = 0.95
MAX_HARMFUL_PHASE_FRACTION = 0.01
MIN_MATRIX_CR_WT = 0.80

# SP-240 환경: 모든 시편에서 완전히 동일하게 유지해야 합니다.
ELECTROLYTE = "3.5 wt% NaCl, air-saturated"
TEST_TEMPERATURE_C = 25.0
REFERENCE_ELECTRODE = "Ag/AgCl"