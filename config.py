# config.py (수정본)

from pathlib import Path
import numpy as np

# ---------------------------------------------------------------------
# 프로젝트 폴더
# ---------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data"
OUTPUT_DIR = ROOT / "output"

DATA_DIR.mkdir(parents=True, exist_ok=True)
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------
# Thermo-Calc database & elements
# ---------------------------------------------------------------------
DATABASE = "TCFE13"
ELEMENTS = [
    "FE",
    "C",
    "SI",
    "MN",
    "CU",
    "NI",
    "P",
    "S",
    "CR",
    "CE",
]

BASE_WT = {
    "C": 0.050,
    "SI": 0.750,
    "MN": 0.600,
    "CU": 0.350,
    "NI": 0.200,
    "P": 0.015,
    "S": 0.005,
}

# 1차 active-learning 설계공간 (Cr_wt 그리드에 0.8 추가 권장)
CE_GRID_WT = np.round(np.arange(0.000, 0.1, 0.001), 4)
CR_GRID_WT = np.round(
    np.arange(0.8, 1.4, 0.001), 3
)  # 0.8 wt% Cr 조건을 포함하도록 범위 확장

# ---------------------------------------------------------------------
# [수정] 실제 열처리 조건 반영 (Section 2.2 논문 내용)
# 1) 열간압연 온도: 1000 °C
# 2) 용체화 열처리 온도: 900 °C (수냉 전 오스테나이트화)
# ---------------------------------------------------------------------
T_HOT_ROLLING_C = 1000.0  # 열간 압연 온도
T_EQUIL_C = 900.0  # 논문의 용체화 열처리(Solution treatment) 온도
T_EQUIL_K = T_EQUIL_C + 273.15

P_PA = 1.0e5

# ---------------------------------------------------------------------
# [수정] Matrix phase: 900 °C 열처리 시 고온 기지상은 Austenite (FCC_A1)
# ---------------------------------------------------------------------
MATRIX_PHASE = "FCC_A1"

TRACKED_PHASES = [
    "FCC_A1",  # 900 °C 고온 기지상
    "BCC_A2",  # 저온 변태상 (Bainite / Ferrite)
    "M7C3_D101",
    "MS_B1",
    "CE2S3",
    "CE3S4_D73",
    "CE2C3_D5C",
]

HARMFUL_PHASES = [
    "SIGMA",
    "MU_PHASE",
    "CHI",
    "LAVES_C14",
    "LAVES_C15",
]

# Feasibility constraints (900 °C Austenite 기지 기준)
MIN_MATRIX_FRACTION = 0.95
MAX_HARMFUL_PHASE_FRACTION = 0.01
MIN_MATRIX_CR_WT = 0.80

# ---------------------------------------------------------------------
# 전기화학 시험 환경 (동일)
# ---------------------------------------------------------------------
ELECTROLYTE = "3.5 wt% NaCl, air-saturated"
TEST_TEMPERATURE_C = 25.0
REFERENCE_ELECTRODE = "Ag/AgCl"