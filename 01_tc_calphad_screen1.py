# 01_tc_calphad_screen.py
#
# Thermo-Calc CALPHAD screening for POSEIDON 500 + Ce + Cr
# 900 °C 용체화 열처리 평형 상태 계산 (Matrix: FCC_A1 Austenite)
#
# 수정 사항:
# 1. 미석출 상 질의로 인한 'ERROR IN QVFIND : NO PHASE NAMED' 방지 (Stable phase 캐싱)
# 2. composition_of_phase_as_weight_fraction 인자 순서 수정 (phase, component)
#
# Output:
# - data/calphad_equilibrium.csv
# - data/calphad_feasible_pool.csv

import os
from pathlib import Path

import numpy as np
import pandas as pd
from tc_python import TCPython, ThermodynamicQuantity

from config import (
    BASE_WT,
    CE_GRID_WT,
    CR_GRID_WT,
    DATA_DIR,
    DATABASE,
    ELEMENTS,
    HARMFUL_PHASES,
    MATRIX_PHASE,
    MAX_HARMFUL_PHASE_FRACTION,
    MIN_MATRIX_CR_WT,
    MIN_MATRIX_FRACTION,
    P_PA,
    T_EQUIL_C,
    T_EQUIL_K,
    TRACKED_PHASES,
)

# ---------------------------------------------------------------------
# 출력 파일 경로
# ---------------------------------------------------------------------
OUT_RAW = DATA_DIR / "calphad_equilibrium.csv"
OUT_POOL = DATA_DIR / "calphad_feasible_pool.csv"


# ---------------------------------------------------------------------
# Thermo-Calc cache 디렉터리 설정
# ---------------------------------------------------------------------
def get_tc_cache_folder() -> Path:
    local_appdata = os.environ.get("LOCALAPPDATA")
    if not local_appdata:
        cache_folder = DATA_DIR / "tc_cache"
    else:
        cache_folder = Path(local_appdata) / "ThermoCalcCache" / "poseidon500"

    cache_folder.mkdir(parents=True, exist_ok=True)
    return cache_folder


# ---------------------------------------------------------------------
# 조성 생성
# ---------------------------------------------------------------------
def make_composition(ce_wt: float, cr_wt: float) -> dict:
    composition = dict(BASE_WT)
    composition["CE"] = float(ce_wt)
    composition["CR"] = float(cr_wt)

    total_non_fe = sum(composition.values())
    if total_non_fe >= 100.0:
        raise ValueError(
            f"비-Fe 원소 총합이 100 wt% 이상입니다: {total_non_fe:.4f} wt%"
        )

    return composition


# ---------------------------------------------------------------------
# Stable phase instance 이름에서 suffix 제거 (#1, #2 제거)
# ---------------------------------------------------------------------
def base_phase_name(phase_instance_name: str) -> str:
    return str(phase_instance_name).split("#")[0]


# ---------------------------------------------------------------------
# 안전한 TC 결과 질의
# ---------------------------------------------------------------------
def safe_get_value(result, quantity, default=np.nan) -> float:
    try:
        val = result.get_value_of(quantity)
        if val is None:
            return default
        return float(val)
    except Exception:
        return default


# ---------------------------------------------------------------------
# 단일 평형 계산 수행
# ---------------------------------------------------------------------
def run_one_equilibrium(system, composition: dict) -> dict:
    calc = (
        system.with_single_equilibrium_calculation()
        .set_condition(ThermodynamicQuantity.temperature(), T_EQUIL_K)
        .set_condition(ThermodynamicQuantity.pressure(), P_PA)
    )

    # 질량 분율 (wt% / 100.0)
    for element, wt_percent in composition.items():
        calc = calc.set_condition(
            ThermodynamicQuantity.mass_fraction_of_a_component(element),
            float(wt_percent) / 100.0,
        )

    result = calc.calculate()
    stable_phase_instances = list(result.get_stable_phases())

    record = {
        "stable_phases": ";".join(stable_phase_instances),
        "n_stable_phases": len(stable_phase_instances),
    }

    # 1. 실제 관찰된 Stable Phase 인스턴스별 분율 추출 및 base phase 맵 생성
    stable_phase_map = {}
    phase_fraction_sum = 0.0

    for phase_inst in stable_phase_instances:
        base_name = base_phase_name(phase_inst)
        mole_fraction = safe_get_value(
            result,
            ThermodynamicQuantity.mole_fraction_of_a_phase(base_name),
            default=np.nan,
        )
        record[f"phasefrac_{phase_inst}"] = mole_fraction

        if np.isfinite(mole_fraction):
            phase_fraction_sum += mole_fraction
            stable_phase_map[base_name] = mole_fraction

    record["phase_fraction_sum"] = phase_fraction_sum

    # 2. TRACKED_PHASES 및 HARMFUL_PHASES 분율 매핑
    # 안정상 목록에 없는 상은 엔진에 질의하지 않고 바로 0.0 부여 (QVFIND 에러 방지)
    all_monitor_phases = set(TRACKED_PHASES) | set(HARMFUL_PHASES)
    for phase_base in all_monitor_phases:
        record[f"phasefrac_base_{phase_base}"] = stable_phase_map.get(
            phase_base, 0.0
        )

    # 3. 고온 기지상(Matrix Phase) 분율
    record["matrix_phase_fraction"] = stable_phase_map.get(
        MATRIX_PHASE, np.nan
    )

    # 4. 고온 기지상 내 고용 원소 함량 (Cr, Mn, Cu wt%)
    # 기지상이 실제 생성되었을 때만 질의하며 인자 순서는 (phase, component)
    if MATRIX_PHASE in stable_phase_map:
        for elem in ["CR", "MN", "CU"]:
            w_frac = safe_get_value(
                result,
                ThermodynamicQuantity.composition_of_phase_as_weight_fraction(
                    MATRIX_PHASE, elem
                ),
                default=np.nan,
            )
            col_name = f"matrix_{elem.title()}_wt"
            record[col_name] = (
                (100.0 * w_frac) if np.isfinite(w_frac) else np.nan
            )
    else:
        for elem in ["CR", "MN", "CU"]:
            record[f"matrix_{elem.title()}_wt"] = np.nan

    return record


# ---------------------------------------------------------------------
# Harmful Phase 분율 합산
# ---------------------------------------------------------------------
def calc_harmful_phase_sum(row: pd.Series) -> float:
    harmful_vals = []
    for phase_name in HARMFUL_PHASES:
        col = f"phasefrac_base_{phase_name}"
        if col in row.index:
            val = row[col]
            if pd.notna(val) and np.isfinite(float(val)):
                harmful_vals.append(float(val))
    return float(np.sum(harmful_vals)) if harmful_vals else 0.0


# ---------------------------------------------------------------------
# Main 실행 루프
# ---------------------------------------------------------------------
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    tc_cache = get_tc_cache_folder()

    print("=" * 75)
    print("Thermo-Calc CALPHAD Screening (POSEIDON 500 + Ce + Cr)")
    print("=" * 75)
    print(f"Database                : {DATABASE}")
    print(f"Target Matrix Phase     : {MATRIX_PHASE}")
    print(f"Solution Heat Treatment : {T_EQUIL_C:.1f} °C (900 °C Quenching)")
    print(f"Cache Directory         : {tc_cache.resolve()}")
    print(f"Raw Output CSV          : {OUT_RAW.resolve()}")
    print(f"Feasible Pool CSV       : {OUT_POOL.resolve()}")
    print("=" * 75)

    records = []

    try:
        with TCPython() as start:
            start.set_cache_folder(str(tc_cache))

            print("\n[1/3] Thermo-Calc system 빌드 중...")
            system = start.select_database_and_elements(
                DATABASE, ELEMENTS
            ).get_system()
            print("[완료] System 생성 성공")

            total_jobs = len(CE_GRID_WT) * len(CR_GRID_WT)
            job_idx = 0

            print("\n[2/3] Ce-Cr 설계 공간 평형 계산 진행...")

            for ce_wt in CE_GRID_WT:
                for cr_wt in CR_GRID_WT:
                    job_idx += 1
                    composition = make_composition(ce_wt, cr_wt)

                    record = {
                        "Ce_wt": float(ce_wt),
                        "Cr_wt": float(cr_wt),
                        "T_equil_C": float(T_EQUIL_C),
                        **composition,
                        "calculation_status": "started",
                        "calculation_error": "",
                    }

                    try:
                        calc_record = run_one_equilibrium(system, composition)
                        record.update(calc_record)
                        record["calculation_status"] = "ok"
                    except Exception as exc:
                        record["calculation_status"] = "failed"
                        record["calculation_error"] = repr(exc)

                    records.append(record)

                    matrix_frac = record.get("matrix_phase_fraction", np.nan)
                    matrix_cr = record.get("matrix_Cr_wt", np.nan)

                    print(
                        f"[{job_idx:02d}/{total_jobs}] "
                        f"Ce={ce_wt:.4f} wt%, Cr={cr_wt:.3f} wt% | "
                        f"Status: {record['calculation_status']} | "
                        f"Matrix({MATRIX_PHASE}): {matrix_frac:.4f} | "
                        f"Matrix Cr: {matrix_cr:.3f} wt%"
                    )

    except Exception as exc:
        print(f"\n[치명적 오류 발생]: {repr(exc)}")
        raise

    # -------------------------------------------------------------
    # 데이터 후처리 및 Feasibility 판정
    # -------------------------------------------------------------
    df = pd.DataFrame(records)
    if df.empty:
        raise RuntimeError("계산 결과 레코드가 비어 있습니다.")

    df.columns = df.columns.astype(str).str.strip()
    ok_mask = df["calculation_status"].eq("ok")

    numeric_cols = [
        "Ce_wt",
        "Cr_wt",
        "matrix_phase_fraction",
        "matrix_Cr_wt",
        "matrix_Mn_wt",
        "matrix_Cu_wt",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")

    # 유해상 합산
    df["harmful_phase_fraction"] = 0.0
    for idx in df.index[ok_mask]:
        df.loc[idx, "harmful_phase_fraction"] = calc_harmful_phase_sum(
            df.loc[idx]
        )

    # config 기준 Feasibility 판정
    df["calphad_feasible"] = (
        ok_mask
        & np.isfinite(df["matrix_phase_fraction"])
        & np.isfinite(df["matrix_Cr_wt"])
        & (df["matrix_phase_fraction"] >= MIN_MATRIX_FRACTION)
        & (df["harmful_phase_fraction"] <= MAX_HARMFUL_PHASE_FRACTION)
        & (df["matrix_Cr_wt"] >= MIN_MATRIX_CR_WT)
    )

    # CSV 저장
    df.to_csv(OUT_RAW, index=False, encoding="utf-8-sig")

    pool = df[df["calphad_feasible"]].copy()
    pool.to_csv(OUT_POOL, index=False, encoding="utf-8-sig")

    # -------------------------------------------------------------
    # 결과 요약 출력
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("[3/3] 평형 스크리닝 요약")
    print("=" * 75)
    print(f"총 그리드 포인트 수      : {len(df)}")
    print(f"계산 성공 건수            : {int(ok_mask.sum())}")
    print(f"계산 실패 건수            : {int((~ok_mask).sum())}")
    print(f"Feasible 선정 후보 수     : {len(pool)}")

    if not pool.empty:
        print("\n[Feasible 후보 요약 통계]")
        print(
            pool[
                [
                    "Ce_wt",
                    "Cr_wt",
                    "matrix_phase_fraction",
                    "matrix_Cr_wt",
                    "harmful_phase_fraction",
                ]
            ]
            .describe()
            .to_string()
        )

    print(f"\n저장된 파일:\n1. {OUT_RAW.resolve()}\n2. {OUT_POOL.resolve()}")


if __name__ == "__main__":
    main()