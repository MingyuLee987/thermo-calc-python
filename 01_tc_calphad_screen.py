# 01_tc_calphad_screen.py
#
# Thermo-Calc CALPHAD screening for POSEIDON 500 + Ce + Cr
#
# Thermo-Calc 2026a TC-Python API 대응 코드
#
# 핵심 API:
# - ThermodynamicQuantity.mole_fraction_of_a_phase(phase)
# - ThermodynamicQuantity.composition_of_phase_as_weight_fraction(
#       component,
#       phase
#   )
#
# Output:
# - data/calphad_equilibrium.csv
# - data/calphad_feasible_pool.csv

from pathlib import Path
import os

import numpy as np
import pandas as pd

from tc_python import TCPython, ThermodynamicQuantity

from config import (
    DATA_DIR,
    DATABASE,
    ELEMENTS,
    BASE_WT,
    CE_GRID_WT,
    CR_GRID_WT,
    T_EQUIL_K,
    P_PA,
    T_EQUIL_C,
    MATRIX_PHASE,
    TRACKED_PHASES,
    HARMFUL_PHASES,
    MIN_MATRIX_FRACTION,
    MAX_HARMFUL_PHASE_FRACTION,
    MIN_MATRIX_CR_WT,
)


# ---------------------------------------------------------------------
# 출력 파일 경로
# ---------------------------------------------------------------------
OUT_RAW = DATA_DIR / "calphad_equilibrium.csv"
OUT_POOL = DATA_DIR / "calphad_feasible_pool.csv"


# ---------------------------------------------------------------------
# Thermo-Calc cache
#
# OneDrive 폴더가 아닌 Windows LOCALAPPDATA에 cache 저장.
# ---------------------------------------------------------------------
def get_tc_cache_folder():
    local_appdata = os.environ.get("LOCALAPPDATA")

    if not local_appdata:
        raise RuntimeError(
            "LOCALAPPDATA 환경변수를 찾지 못했습니다."
        )

    cache_folder = (
        Path(local_appdata)
        / "ThermoCalcCache"
        / "poseidon500"
    )

    cache_folder.mkdir(parents=True, exist_ok=True)

    return cache_folder


# ---------------------------------------------------------------------
# 조성 생성
# ---------------------------------------------------------------------
def make_composition(ce_wt, cr_wt):
    composition = dict(BASE_WT)

    composition["CE"] = float(ce_wt)
    composition["CR"] = float(cr_wt)

    total_non_fe = sum(composition.values())

    if total_non_fe >= 100.0:
        raise ValueError(
            f"비-Fe 원소 총합이 100 wt% 이상입니다: "
            f"{total_non_fe:.4f} wt%"
        )

    return composition


# ---------------------------------------------------------------------
# Stable phase instance 이름에서 suffix 제거
#
# BCC_A2#1 -> BCC_A2
# CE2S3#1 -> CE2S3
# ---------------------------------------------------------------------
def base_phase_name(phase_instance_name):
    return str(phase_instance_name).split("#")[0]


# ---------------------------------------------------------------------
# 안전한 result value 추출
# ---------------------------------------------------------------------
def safe_get_value(result, quantity, default=np.nan):
    try:
        value = result.get_value_of(quantity)

        if value is None:
            return default

        return float(value)

    except Exception:
        return default


# ---------------------------------------------------------------------
# Thermo-Calc 2026a API quantity builders
# ---------------------------------------------------------------------
def q_mole_fraction_of_phase(phase_name):
    return ThermodynamicQuantity.mole_fraction_of_a_phase(
        phase_name
    )


def q_weight_fraction_of_component_in_phase(component, phase_name):
    return (
        ThermodynamicQuantity
        .composition_of_phase_as_weight_fraction(
            component,
            phase_name
        )
    )


# ---------------------------------------------------------------------
# 단일 equilibrium calculation
# ---------------------------------------------------------------------
def run_one_equilibrium(system, composition):
    calc = (
        system
        .with_single_equilibrium_calculation()
        .set_condition(
            ThermodynamicQuantity.temperature(),
            T_EQUIL_K,
        )
        .set_condition(
            ThermodynamicQuantity.pressure(),
            P_PA,
        )
    )

    # wt% -> mass fraction으로 입력
    for element, wt_percent in composition.items():
        calc = calc.set_condition(
            ThermodynamicQuantity.mass_fraction_of_a_component(
                element
            ),
            float(wt_percent) / 100.0,
        )

    result = calc.calculate()

    stable_phase_instances = list(
        result.get_stable_phases()
    )

    record = {
        "stable_phases": ";".join(stable_phase_instances),
        "n_stable_phases": len(stable_phase_instances),
    }

    # -------------------------------------------------------------
    # 1. 실제 stable phase별 mole fraction
    # -------------------------------------------------------------
    phase_fraction_sum = 0.0

    for phase_instance in stable_phase_instances:
        phase_base = base_phase_name(phase_instance)

        mole_fraction = safe_get_value(
            result,
            q_mole_fraction_of_phase(phase_base),
            default=np.nan,
        )

        # 실제 stable phase instance 이름을 CSV column에 유지
        # 예: phasefrac_BCC_A2#1
        record[
            f"phasefrac_{phase_instance}"
        ] = mole_fraction

        if np.isfinite(mole_fraction):
            phase_fraction_sum += mole_fraction

    record["phase_fraction_sum"] = phase_fraction_sum

    # -------------------------------------------------------------
    # 2. 추적상별 mole fraction
    #
    # 모든 row에 동일한 열을 만들기 위해 base phase 기준으로 저장.
    # -------------------------------------------------------------
    for phase_base in TRACKED_PHASES:
        record[
            f"phasefrac_base_{phase_base}"
        ] = safe_get_value(
            result,
            q_mole_fraction_of_phase(phase_base),
            default=0.0,
        )

    # -------------------------------------------------------------
    # 3. Matrix phase fraction
    #
    # BCC_A2의 mole fraction을 matrix phase fraction으로 사용.
    # -------------------------------------------------------------
    record["matrix_phase_fraction"] = safe_get_value(
        result,
        q_mole_fraction_of_phase(MATRIX_PHASE),
        default=np.nan,
    )

    # -------------------------------------------------------------
    # 4. Matrix 내부 Cr, Mn, Cu wt%
    #
    # Thermo-Calc output은 weight fraction이므로 100을 곱해 wt% 변환.
    # -------------------------------------------------------------
    for element in ["CR", "MN", "CU"]:
        weight_fraction = safe_get_value(
            result,
            q_weight_fraction_of_component_in_phase(
                element,
                MATRIX_PHASE,
            ),
            default=np.nan,
        )

        column_name = f"matrix_{element.title()}_wt"

        if np.isfinite(weight_fraction):
            record[column_name] = 100.0 * weight_fraction
        else:
            record[column_name] = np.nan

    # 후속 코드에 쓰는 통일 열 이름
    record["matrix_Cr_wt"] = record.pop(
        "matrix_Cr_wt"
    )
    record["matrix_Mn_wt"] = record.pop(
        "matrix_Mn_wt"
    )
    record["matrix_Cu_wt"] = record.pop(
        "matrix_Cu_wt"
    )

    return record


# ---------------------------------------------------------------------
# Harmful phase mole fraction 합산
# ---------------------------------------------------------------------
def harmful_phase_sum(row):
    values = []

    for phase_name in HARMFUL_PHASES:
        col = f"phasefrac_base_{phase_name}"

        if col in row.index:
            value = row[col]

            if pd.notna(value) and np.isfinite(
                float(value)
            ):
                values.append(float(value))

    return float(np.sum(values)) if values else 0.0


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tc_cache = get_tc_cache_folder()

    print("=" * 75)
    print("Thermo-Calc CALPHAD Screening")
    print("=" * 75)
    print(f"Database                : {DATABASE}")
    print(f"Matrix phase            : {MATRIX_PHASE}")
    print(f"Equilibrium temperature : {T_EQUIL_C:.1f} °C")
    print(f"Cache folder            : {tc_cache.resolve()}")
    print(f"Raw result CSV          : {OUT_RAW.resolve()}")
    print(f"Candidate pool CSV      : {OUT_POOL.resolve()}")
    print("=" * 75)

    records = []

    try:
        with TCPython() as start:
            start.set_cache_folder(str(tc_cache))

            print("\n[1/3] Thermo-Calc system 생성 중...")

            system = (
                start
                .select_database_and_elements(
                    DATABASE,
                    ELEMENTS,
                )
                .get_system()
            )

            print("[완료] Thermo-Calc system 생성 성공")

            total_jobs = (
                len(CE_GRID_WT)
                * len(CR_GRID_WT)
            )

            job_number = 0

            print("\n[2/3] Ce-Cr grid equilibrium calculation 시작")

            for ce_wt in CE_GRID_WT:
                for cr_wt in CR_GRID_WT:
                    job_number += 1

                    composition = make_composition(
                        ce_wt,
                        cr_wt,
                    )

                    record = {
                        "Ce_wt": float(ce_wt),
                        "Cr_wt": float(cr_wt),
                        "T_equil_C": float(T_EQUIL_C),
                        **composition,
                        "calculation_status": "started",
                        "calculation_error": "",
                    }

                    try:
                        calc_record = run_one_equilibrium(
                            system,
                            composition,
                        )

                        record.update(calc_record)

                        record["calculation_status"] = "ok"

                    except Exception as exc:
                        record["calculation_status"] = "failed"
                        record["calculation_error"] = repr(
                            exc
                        )

                    records.append(record)

                    matrix_fraction = record.get(
                        "matrix_phase_fraction",
                        np.nan,
                    )

                    matrix_cr = record.get(
                        "matrix_Cr_wt",
                        np.nan,
                    )

                    print(
                        f"[{job_number:02d}/{total_jobs}] "
                        f"Ce={ce_wt:.4f} wt%, "
                        f"Cr={cr_wt:.3f} wt%, "
                        f"status={record['calculation_status']}, "
                        f"matrix fraction={matrix_fraction:.6f}, "
                        f"matrix Cr={matrix_cr:.6f} wt%"
                    )

    except Exception as exc:
        print("\n[치명적 오류]")
        print(repr(exc))
        raise

    # -------------------------------------------------------------
    # DataFrame 및 numeric 처리
    # -------------------------------------------------------------
    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError(
            "Thermo-Calc calculation record가 생성되지 않았습니다."
        )

    df.columns = (
        df.columns
        .astype(str)
        .str.strip()
    )

    required_columns = [
        "Ce_wt",
        "Cr_wt",
        "calculation_status",
        "matrix_phase_fraction",
        "matrix_Cr_wt",
    ]

    missing_columns = [
        col
        for col in required_columns
        if col not in df.columns
    ]

    if missing_columns:
        raise RuntimeError(
            f"필수 열이 없습니다: {missing_columns}\n"
            f"현재 열: {df.columns.tolist()}"
        )

    ok = df["calculation_status"].eq("ok")

    numeric_columns = [
        "Ce_wt",
        "Cr_wt",
        "matrix_phase_fraction",
        "matrix_Cr_wt",
        "matrix_Mn_wt",
        "matrix_Cu_wt",
    ]

    for col in numeric_columns:
        if col in df.columns:
            df[col] = pd.to_numeric(
                df[col],
                errors="coerce",
            )

    # -------------------------------------------------------------
    # Harmful phase fraction
    # -------------------------------------------------------------
    df["harmful_phase_fraction"] = 0.0

    for idx in df.index[ok]:
        df.loc[
            idx,
            "harmful_phase_fraction",
        ] = harmful_phase_sum(df.loc[idx])

    # -------------------------------------------------------------
    # Feasibility
    # -------------------------------------------------------------
    df["calphad_feasible"] = (
        ok
        & np.isfinite(
            df["matrix_phase_fraction"]
        )
        & np.isfinite(
            df["matrix_Cr_wt"]
        )
        & (
            df["matrix_phase_fraction"]
            >= MIN_MATRIX_FRACTION
        )
        & (
            df["harmful_phase_fraction"]
            <= MAX_HARMFUL_PHASE_FRACTION
        )
        & (
            df["matrix_Cr_wt"]
            >= MIN_MATRIX_CR_WT
        )
    )

    # -------------------------------------------------------------
    # CSV output
    # -------------------------------------------------------------
    df.to_csv(
        OUT_RAW,
        index=False,
        encoding="utf-8-sig",
    )

    pool = df[
        df["calphad_feasible"]
    ].copy()

    pool.to_csv(
        OUT_POOL,
        index=False,
        encoding="utf-8-sig",
    )

    # -------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------
    print("\n" + "=" * 75)
    print("[3/3] Calculation Summary")
    print("=" * 75)

    print(f"Total compositions       : {len(df)}")
    print(f"Successful calculations  : {int(ok.sum())}")
    print(f"Failed calculations      : {int((~ok).sum())}")
    print(
        "Finite matrix Cr values : "
        f"{int(np.isfinite(df['matrix_Cr_wt']).sum())}"
    )
    print(f"Feasible compositions    : {len(pool)}")

    print("\nMatrix descriptor statistics:")
    print(
        df.loc[
            ok,
            [
                "matrix_phase_fraction",
                "matrix_Cr_wt",
                "matrix_Mn_wt",
                "matrix_Cu_wt",
            ],
        ]
        .describe()
        .to_string()
    )

    print("\nFirst 10 results:")
    print(
        df[
            [
                "Ce_wt",
                "Cr_wt",
                "stable_phases",
                "matrix_phase_fraction",
                "matrix_Cr_wt",
                "matrix_Mn_wt",
                "matrix_Cu_wt",
                "calphad_feasible",
            ]
        ]
        .head(10)
        .to_string(index=False)
    )

    print("\nSaved files:")
    print(OUT_RAW.resolve())
    print(OUT_POOL.resolve())


if __name__ == "__main__":
    main()