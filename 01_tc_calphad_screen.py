# 01_tc_calphad_screen.py
#
# 목적:
# 1) Ce-Cr 조성 격자를 생성
# 2) Thermo-Calc equilibrium calculation 수행
# 3) 안정상, matrix phase fraction, matrix 내부 Cr/Mn/Cu 함량 저장
# 4) CALPHAD feasibility 조건을 만족하는 후보만 별도 CSV로 저장
#
# 중요:
# - Thermo-Calc cache는 OneDrive/바탕화면 밖의 LOCALAPPDATA에 저장한다.
# - 기존 data/tc_cache는 사용하지 않는다.
# - matrix phase 이름(BCC_A2 등)은 계산 후 stable_phases 결과를 확인하여 config.py에서 수정한다.

from pathlib import Path
import os
import shutil

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
    HARMFUL_PHASES,
    MIN_MATRIX_FRACTION,
    MAX_HARMFUL_PHASE_FRACTION,
    MIN_MATRIX_CR_WT,
)


# ---------------------------------------------------------------------
# 출력 CSV 경로
# ---------------------------------------------------------------------
OUT_RAW = DATA_DIR / "calphad_equilibrium.csv"
OUT_POOL = DATA_DIR / "calphad_feasible_pool.csv"


# ---------------------------------------------------------------------
# Thermo-Calc cache 경로
#
# OneDrive 바탕화면의 프로젝트 내부에 cache를 만들면
# OneDrive 동기화/파일 잠금 때문에 System.poly3 access denied가 날 수 있다.
#
# 아래 경로 예:
# C:\Users\user\AppData\Local\ThermoCalcCache\poseidon500
# ---------------------------------------------------------------------
def get_tc_cache_folder():
    local_app_data = os.environ.get("LOCALAPPDATA")

    if not local_app_data:
        raise RuntimeError(
            "Windows 환경변수 LOCALAPPDATA를 찾지 못했습니다. "
            "Thermo-Calc cache 경로를 직접 지정해야 합니다."
        )

    cache_folder = (
        Path(local_app_data)
        / "ThermoCalcCache"
        / "poseidon500"
    )

    cache_folder.mkdir(parents=True, exist_ok=True)

    return cache_folder


# ---------------------------------------------------------------------
# 조성 생성
#
# BASE_WT의 원소는 그대로 사용하고,
# Ce와 Cr만 반복적으로 변화시킨다.
#
# Fe는 Thermo-Calc에서 balance 원소가 되므로 명시적으로 넣지 않는다.
# ---------------------------------------------------------------------
def make_composition(ce_wt, cr_wt):
    composition = dict(BASE_WT)

    composition["CE"] = float(ce_wt)
    composition["CR"] = float(cr_wt)

    total_non_fe = sum(composition.values())

    if total_non_fe >= 100.0:
        raise ValueError(
            f"Fe balance 계산 불가: 비-Fe 원소 총합 = {total_non_fe:.4f} wt%"
        )

    return composition


# ---------------------------------------------------------------------
# Thermo-Calc quantity helper
# ---------------------------------------------------------------------
def q_phase_fraction(phase_name):
    return ThermodynamicQuantity.phase_fraction(phase_name)


def q_mass_fraction_component_in_phase(component, phase_name):
    return ThermodynamicQuantity.mass_fraction_of_component_in_phase(
        component,
        phase_name
    )


# ---------------------------------------------------------------------
# 단일 조성 equilibrium calculation
# ---------------------------------------------------------------------
def run_one_equilibrium(system, composition):
    """
    Parameters
    ----------
    system:
        Thermo-Calc System object
    composition:
        예: {
            "C": 0.06,
            "SI": 0.25,
            ...
            "CE": 0.02,
            "CR": 1.10
        }
        단위: wt%

    Returns
    -------
    dict
        stable phase, 각 상의 phase fraction,
        matrix phase fraction, matrix 내 Cr/Mn/Cu wt%
    """

    calc = (
        system
        .with_single_equilibrium_calculation()
        .set_condition(
            ThermodynamicQuantity.temperature(),
            T_EQUIL_K
        )
        .set_condition(
            ThermodynamicQuantity.pressure(),
            P_PA
        )
    )

    # Composition은 wt% -> mass fraction으로 변환하여 입력.
    for element, wt_percent in composition.items():
        calc = calc.set_condition(
            ThermodynamicQuantity.mass_fraction_of_a_component(element),
            float(wt_percent) / 100.0
        )

    result = calc.calculate()

    stable_phases = list(result.get_stable_phases())

    record = {
        "stable_phases": ";".join(stable_phases),
        "n_stable_phases": len(stable_phases),
    }

    phase_fraction_sum = 0.0

    # 실제 안정상의 분율 저장.
    # 상 종류가 조성마다 달라도 각 행에 phasefrac_상명 형태로 저장된다.
    for phase_name in stable_phases:
        try:
            phase_fraction = result.get_value_of(
                q_phase_fraction(phase_name)
            )
        except Exception:
            phase_fraction = np.nan

        record[f"phasefrac_{phase_name}"] = phase_fraction

        if np.isfinite(phase_fraction):
            phase_fraction_sum += phase_fraction

    record["phase_fraction_sum"] = phase_fraction_sum

    # Config에서 설정한 matrix phase의 분율.
    # 예: BCC_A2
    try:
        record["matrix_phase_fraction"] = result.get_value_of(
            q_phase_fraction(MATRIX_PHASE)
        )
    except Exception:
        record["matrix_phase_fraction"] = 0.0

    # Matrix phase 내부의 Cr/Mn/Cu 조성.
    # 계산 불가 시 NaN으로 저장.
    for element in ["CR", "MN", "CU"]:
        column_name = f"matrix_{element.title()}_wt"

        try:
            mass_fraction = result.get_value_of(
                q_mass_fraction_component_in_phase(
                    element,
                    MATRIX_PHASE
                )
            )

            record[column_name] = 100.0 * mass_fraction

        except Exception:
            record[column_name] = np.nan

    # column 이름을 후속 active-learning 코드와 통일.
    record["matrix_Cr_wt"] = record.pop("matrix_Cr_wt")
    record["matrix_Mn_wt"] = record.pop("matrix_Mn_wt")
    record["matrix_Cu_wt"] = record.pop("matrix_Cu_wt")

    return record


# ---------------------------------------------------------------------
# 유해상 분율 합산
#
# HARMFUL_PHASES 목록에 있는 상만 합산.
# 실제 database에서 phase 이름이 다르면 config.py를 수정한다.
# ---------------------------------------------------------------------
def harmful_phase_sum(row):
    phase_fractions = []

    for phase_name in HARMFUL_PHASES:
        column_name = f"phasefrac_{phase_name}"

        if column_name in row.index:
            value = row[column_name]

            if pd.notna(value):
                phase_fractions.append(float(value))

    if not phase_fractions:
        return 0.0

    return float(np.sum(phase_fractions))


# ---------------------------------------------------------------------
# 기존 OneDrive cache를 사용하지 않는다는 확인용 함수.
#
# 삭제까지 자동으로 하고 싶으면 delete_old_project_cache=True로 바꾸면 되지만,
# 처음에는 False로 두는 편이 안전하다.
# ---------------------------------------------------------------------
def handle_old_project_cache(delete_old_project_cache=False):
    old_cache = DATA_DIR / "tc_cache"

    if not old_cache.exists():
        return

    print("\n[알림] 기존 OneDrive 프로젝트 cache 폴더가 존재합니다.")
    print(f"기존 cache: {old_cache.resolve()}")
    print("이 코드는 해당 폴더를 사용하지 않습니다.")

    if delete_old_project_cache:
        try:
            shutil.rmtree(old_cache)
            print("[완료] 기존 프로젝트 cache를 삭제했습니다.")
        except Exception as exc:
            print("[경고] 기존 cache 삭제 실패:")
            print(repr(exc))
            print("Thermo-Calc/VS Code/Python을 종료한 뒤 수동 삭제하세요.")


# ---------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------
def main():
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    handle_old_project_cache(
        delete_old_project_cache=False
    )

    tc_cache_folder = get_tc_cache_folder()

    print("=" * 70)
    print("Thermo-Calc CALPHAD screening started")
    print("=" * 70)
    print(f"Database          : {DATABASE}")
    print(f"Elements          : {ELEMENTS}")
    print(f"Equilibrium T     : {T_EQUIL_C:.1f} °C")
    print(f"Matrix phase      : {MATRIX_PHASE}")
    print(f"Output CSV        : {OUT_RAW.resolve()}")
    print(f"Candidate pool    : {OUT_POOL.resolve()}")
    print(f"TC cache folder   : {tc_cache_folder.resolve()}")
    print("=" * 70)

    records = []

    # -------------------------------------------------------------
    # Thermo-Calc 연결 및 System 생성
    # 여기서 실패하면 조성 loop로 들어가지 못하므로,
    # 에러를 명확히 출력한 뒤 종료한다.
    # -------------------------------------------------------------
    try:
        with TCPython() as start:
            start.set_cache_folder(str(tc_cache_folder))

            print("\n[1/3] Thermo-Calc system 생성 중...")

            system = (
                start
                .select_database_and_elements(
                    DATABASE,
                    ELEMENTS
                )
                .get_system()
            )

            print("[완료] Thermo-Calc system 생성 성공")

            total_jobs = len(CE_GRID_WT) * len(CR_GRID_WT)
            current_job = 0

            # -----------------------------------------------------
            # Ce-Cr 조성 격자 반복
            # -----------------------------------------------------
            for ce_wt in CE_GRID_WT:
                for cr_wt in CR_GRID_WT:
                    current_job += 1

                    composition = make_composition(
                        ce_wt,
                        cr_wt
                    )

                    # calculation_status와 calculation_error를
                    # try 이전부터 넣어 둔다.
                    # 그러면 모든 calculation이 실패해도 CSV schema가 유지된다.
                    record = {
                        "Ce_wt": float(ce_wt),
                        "Cr_wt": float(cr_wt),
                        "T_equil_C": float(T_EQUIL_C),
                        **composition,
                        "calculation_status": "started",
                        "calculation_error": "",
                    }

                    try:
                        result_record = run_one_equilibrium(
                            system,
                            composition
                        )

                        record.update(result_record)
                        record["calculation_status"] = "ok"

                    except Exception as exc:
                        record["calculation_status"] = "failed"
                        record["calculation_error"] = repr(exc)

                        # 실패해도 다음 조성으로 넘어간다.
                        # 그래야 어떤 조성이 왜 실패했는지 CSV에 남는다.
                        print(
                            f"[실패] {current_job}/{total_jobs} | "
                            f"Ce={ce_wt:.4f} wt%, "
                            f"Cr={cr_wt:.3f} wt%"
                        )
                        print(f"       Error: {repr(exc)}")

                    records.append(record)

                    if record["calculation_status"] == "ok":
                        print(
                            f"[성공] {current_job}/{total_jobs} | "
                            f"Ce={ce_wt:.4f} wt%, "
                            f"Cr={cr_wt:.3f} wt%"
                        )

    except Exception as exc:
        print("\n" + "=" * 70)
        print("[치명적 오류] Thermo-Calc system 생성 단계에서 중단되었습니다.")
        print("=" * 70)
        print(repr(exc))
        print("\n다음을 확인하세요:")
        print("1) TC cache가 OneDrive가 아닌 LOCALAPPDATA 경로인지")
        print("2) Thermo-Calc GUI / Python / Java process가 남아 있지 않은지")
        print("3) DATABASE 이름과 ELEMENTS가 실제 설치 DB에서 지원되는지")
        print("4) LOCALAPPDATA cache 폴더에 쓰기 권한이 있는지")
        print("5) cache 문제 분리를 위해 start.set_cache_folder(...) 줄을")
        print("   임시로 주석 처리한 뒤 재실행해 보는지")
        raise

    # -------------------------------------------------------------
    # DataFrame 생성
    # -------------------------------------------------------------
    df = pd.DataFrame(records)

    if df.empty:
        raise RuntimeError(
            "\nCALPHAD 결과가 한 행도 생성되지 않았습니다.\n"
            "Thermo-Calc system 생성 또는 조성 loop 이전에 코드가 중단되었을 수 있습니다."
        )

    # 혹시 모를 열 이름 공백 제거.
    df.columns = df.columns.astype(str).str.strip()

    required_columns = [
        "Ce_wt",
        "Cr_wt",
        "calculation_status",
        "calculation_error",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        print("\n생성된 DataFrame 열:")
        print(df.columns.tolist())

        raise RuntimeError(
            f"필수 열이 생성되지 않았습니다: {missing_columns}"
        )

    # -------------------------------------------------------------
    # Calculation status 확인
    # -------------------------------------------------------------
    ok = df["calculation_status"].eq("ok")

    print("\n" + "=" * 70)
    print("DataFrame columns")
    print("=" * 70)
    print(df.columns.tolist())

    print("\n" + "=" * 70)
    print("Calculation status counts")
    print("=" * 70)
    print(df["calculation_status"].value_counts(dropna=False))

    # -------------------------------------------------------------
    # 유해상 분율 계산
    # -------------------------------------------------------------
    df["harmful_phase_fraction"] = 0.0

    for idx in df.index[ok]:
        df.loc[idx, "harmful_phase_fraction"] = harmful_phase_sum(
            df.loc[idx]
        )

    # 계산 실패한 조성은 아래 입력값들이 없을 수 있으므로 안전하게 기본값 처리.
    for column in [
        "matrix_phase_fraction",
        "matrix_Cr_wt",
        "matrix_Mn_wt",
        "matrix_Cu_wt",
    ]:
        if column not in df.columns:
            df[column] = np.nan

    # -------------------------------------------------------------
    # CALPHAD feasible 판정
    #
    # 아래 3개를 모두 만족하는 조성만 True:
    # 1) matrix phase fraction 이상
    # 2) harmful phase fraction 이하
    # 3) matrix 내부 Cr 함량 이상
    # -------------------------------------------------------------
    df["calphad_feasible"] = (
        ok
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
    # CSV 저장
    # -------------------------------------------------------------
    df.to_csv(
        OUT_RAW,
        index=False,
        encoding="utf-8-sig"
    )

    pool = df[df["calphad_feasible"]].copy()

    pool.to_csv(
        OUT_POOL,
        index=False,
        encoding="utf-8-sig"
    )

    # -------------------------------------------------------------
    # 콘솔 요약
    # -------------------------------------------------------------
    print("\n" + "=" * 70)
    print("Calculation summary")
    print("=" * 70)
    print(f"All compositions       : {len(df)}")
    print(f"Successful calculations: {int(ok.sum())}")
    print(f"Failed calculations    : {int((~ok).sum())}")
    print(f"Feasible compositions  : {len(pool)}")
    print(f"Raw output CSV         : {OUT_RAW.resolve()}")
    print(f"Candidate pool CSV     : {OUT_POOL.resolve()}")

    print("\n" + "=" * 70)
    print("Stable phase examples")
    print("=" * 70)

    if ok.sum() > 0 and "stable_phases" in df.columns:
        print(
            df.loc[
                ok,
                [
                    "Ce_wt",
                    "Cr_wt",
                    "stable_phases",
                    "matrix_phase_fraction",
                    "matrix_Cr_wt",
                    "harmful_phase_fraction",
                    "calphad_feasible",
                ]
            ]
            .head(10)
            .to_string(index=False)
        )
    else:
        print("성공한 equilibrium calculation이 없습니다.")
        print("calphad_equilibrium.csv의 calculation_error 열을 확인하세요.")

    print("\n완료했습니다.")


if __name__ == "__main__":
    main()