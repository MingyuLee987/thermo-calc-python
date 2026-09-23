# 01_tc_calphad_screen.py

from tc_python import TCPython, ThermodynamicQuantity
import numpy as np
import pandas as pd

from config import (
    DATA_DIR, DATABASE, ELEMENTS, BASE_WT,
    CE_GRID_WT, CR_GRID_WT,
    T_EQUIL_K, P_PA, T_EQUIL_C,
    MATRIX_PHASE, HARMFUL_PHASES,
    MIN_MATRIX_FRACTION,
    MAX_HARMFUL_PHASE_FRACTION,
    MIN_MATRIX_CR_WT
)

OUT_RAW = DATA_DIR / "calphad_equilibrium.csv"
OUT_POOL = DATA_DIR / "calphad_feasible_pool.csv"


def make_composition(ce_wt, cr_wt):
    """Fe는 balance이며 나머지 원소만 wt%로 입력한다."""
    comp = dict(BASE_WT)
    comp["CE"] = float(ce_wt)
    comp["CR"] = float(cr_wt)
    return comp


def q_phase_fraction(phase):
    return ThermodynamicQuantity.phase_fraction(phase)


def q_mass_fraction_component_in_phase(component, phase):
    return ThermodynamicQuantity.mass_fraction_of_component_in_phase(
        component, phase
    )


def run_one_equilibrium(system, composition):
    """
    한 조성에서 equilibrium 수행.
    TC-Python 버전에 따라 quantity factory 문법이 다르면
    설치된 pyex_T_01 예제 문법을 우선 적용하세요.
    """
    calc = (
        system
        .with_single_equilibrium_calculation()
        .set_condition(ThermodynamicQuantity.temperature(), T_EQUIL_K)
        .set_condition(ThermodynamicQuantity.pressure(), P_PA)
    )

    for element, wt in composition.items():
        calc = calc.set_condition(
            ThermodynamicQuantity.mass_fraction_of_a_component(element),
            wt / 100.0
        )

    result = calc.calculate()
    stable_phases = list(result.get_stable_phases())

    record = {
        "stable_phases": ";".join(stable_phases)
    }

    phase_fraction_sum = 0.0

    for phase in stable_phases:
        try:
            frac = result.get_value_of(q_phase_fraction(phase))
        except Exception:
            frac = np.nan

        record[f"phasefrac_{phase}"] = frac

        if np.isfinite(frac):
            phase_fraction_sum += frac

    record["phase_fraction_sum"] = phase_fraction_sum

    try:
        record["matrix_phase_fraction"] = result.get_value_of(
            q_phase_fraction(MATRIX_PHASE)
        )
    except Exception:
        record["matrix_phase_fraction"] = 0.0

    try:
        record["matrix_Cr_wt"] = 100.0 * result.get_value_of(
            q_mass_fraction_component_in_phase("CR", MATRIX_PHASE)
        )
    except Exception:
        record["matrix_Cr_wt"] = np.nan

    try:
        record["matrix_Mn_wt"] = 100.0 * result.get_value_of(
            q_mass_fraction_component_in_phase("MN", MATRIX_PHASE)
        )
    except Exception:
        record["matrix_Mn_wt"] = np.nan

    try:
        record["matrix_Cu_wt"] = 100.0 * result.get_value_of(
            q_mass_fraction_component_in_phase("CU", MATRIX_PHASE)
        )
    except Exception:
        record["matrix_Cu_wt"] = np.nan

    return record


def harmful_phase_sum(row):
    values = []
    for phase in HARMFUL_PHASES:
        col = f"phasefrac_{phase}"
        if col in row.index and pd.notna(row[col]):
            values.append(float(row[col]))
    return float(np.sum(values)) if values else 0.0


def main():
    records = []

    with TCPython() as start:
        # 반복 실행 시 동일 계산의 cache 활용.
        start.set_cache_folder(str(DATA_DIR / "tc_cache"))

        system = (
            start
            .select_database_and_elements(DATABASE, ELEMENTS)
            .get_system()
        )

        for ce_wt in CE_GRID_WT:
            for cr_wt in CR_GRID_WT:
                composition = make_composition(ce_wt, cr_wt)

                record = {
                    "Ce_wt": ce_wt,
                    "Cr_wt": cr_wt,
                    "T_equil_C": T_EQUIL_C,
                    **composition
                }

                try:
                    result_record = run_one_equilibrium(system, composition)
                    record.update(result_record)
                    record["calculation_status"] = "ok"

                except Exception as exc:
                    record["calculation_status"] = "failed"
                    record["calculation_error"] = repr(exc)

                records.append(record)
                print(
                    f"Completed: Ce={ce_wt:.4f} wt%, "
                    f"Cr={cr_wt:.3f} wt%"
                )

    df = pd.DataFrame(records)

    ok = df["calculation_status"].eq("ok")
    df["harmful_phase_fraction"] = 0.0

    for idx in df.index[ok]:
        df.loc[idx, "harmful_phase_fraction"] = harmful_phase_sum(
            df.loc[idx]
        )

    df["calphad_feasible"] = (
        ok
        & (df["matrix_phase_fraction"] >= MIN_MATRIX_FRACTION)
        & (df["harmful_phase_fraction"] <= MAX_HARMFUL_PHASE_FRACTION)
        & (df["matrix_Cr_wt"] >= MIN_MATRIX_CR_WT)
    )

    df.to_csv(OUT_RAW, index=False, encoding="utf-8-sig")

    pool = df[df["calphad_feasible"]].copy()
    pool.to_csv(OUT_POOL, index=False, encoding="utf-8-sig")

    print("\n=== Calculation summary ===")
    print(f"All compositions       : {len(df)}")
    print(f"Successful calculations: {ok.sum()}")
    print(f"Feasible compositions  : {len(pool)}")
    print(f"Raw output             : {OUT_RAW}")
    print(f"Candidate pool         : {OUT_POOL}")

    print("\n=== Stable phase examples ===")
    print(
        df.loc[ok, ["Ce_wt", "Cr_wt", "stable_phases"]]
        .head(10)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()