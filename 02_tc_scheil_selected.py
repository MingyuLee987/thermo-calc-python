# 02_tc_scheil_selected.py

from tc_python import TCPython, ScheilQuantity
import numpy as np
import pandas as pd

from config import (
    DATA_DIR, DATABASE, ELEMENTS,
    BASE_WT
)

POOL_FILE = DATA_DIR / "calphad_feasible_pool.csv"
OUT_FILE = DATA_DIR / "scheil_descriptors.csv"

# 처음에는 대표 후보만 선택.
# 이후 active learning에서 선정된 next_candidates.csv를 읽도록 바꿔도 됩니다.
SELECTED = [
    {"Ce_wt": 0.000, "Cr_wt": 0.80},
    {"Ce_wt": 0.010, "Cr_wt": 1.00},
    {"Ce_wt": 0.020, "Cr_wt": 1.10},
    {"Ce_wt": 0.030, "Cr_wt": 1.20},
    {"Ce_wt": 0.040, "Cr_wt": 1.30},
]


def make_composition(ce_wt, cr_wt):
    comp = dict(BASE_WT)
    comp["CE"] = float(ce_wt)
    comp["CR"] = float(cr_wt)
    return comp


def run_scheil(system, composition):
    calc = system.with_scheil_calculation()

    for element, wt in composition.items():
        calc = calc.set_composition(element, float(wt))

    result = calc.calculate()

    temperature_k, solid_fraction = result.get_values_of(
        ScheilQuantity.temperature(),
        ScheilQuantity.mole_fraction_of_all_solid_phases()
    )

    temperature_k = np.asarray(temperature_k, dtype=float)
    solid_fraction = np.asarray(solid_fraction, dtype=float)

    liquid_fraction = 1.0 - solid_fraction

    liquidus_k = float(np.max(temperature_k))
    solidus_k = float(np.min(temperature_k))

    # 99% solid가 되는 온도를 operational solidus-like descriptor로 사용.
    idx_99 = np.where(solid_fraction >= 0.99)[0]
    t_99solid_k = (
        float(temperature_k[idx_99[0]])
        if len(idx_99) > 0
        else np.nan
    )

    # 응고 말기 지표: fs >= 0.95 부근의 온도폭.
    idx_95 = np.where(solid_fraction >= 0.95)[0]
    t_95solid_k = (
        float(temperature_k[idx_95[0]])
        if len(idx_95) > 0
        else np.nan
    )

    return {
        "liquidus_C": liquidus_k - 273.15,
        "scheil_end_C": solidus_k - 273.15,
        "T_95solid_C": t_95solid_k - 273.15
        if np.isfinite(t_95solid_k) else np.nan,
        "T_99solid_C": t_99solid_k - 273.15
        if np.isfinite(t_99solid_k) else np.nan,
        "freezing_range_C": liquidus_k - solidus_k,
        "n_scheil_steps": len(temperature_k)
    }


def main():
    records = []

    with TCPython() as start:
        start.set_cache_folder(str(DATA_DIR / "tc_cache"))

        system = (
            start
            .select_database_and_elements(DATABASE, ELEMENTS)
            .get_system()
        )

        for item in SELECTED:
            ce_wt = item["Ce_wt"]
            cr_wt = item["Cr_wt"]
            composition = make_composition(ce_wt, cr_wt)

            record = {
                "Ce_wt": ce_wt,
                "Cr_wt": cr_wt,
                **composition
            }

            try:
                record.update(run_scheil(system, composition))
                record["scheil_status"] = "ok"
            except Exception as exc:
                record["scheil_status"] = "failed"
                record["scheil_error"] = repr(exc)

            records.append(record)
            print(f"Scheil complete: Ce={ce_wt}, Cr={cr_wt}")

    df = pd.DataFrame(records)
    df.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()