# 02_tc_scheil_selected.py

import os
from pathlib import Path
import numpy as np
import pandas as pd
from tc_python import ScheilQuantity, TCPython

from config import BASE_WT, DATA_DIR, DATABASE, ELEMENTS, OUTPUT_DIR

POOL_FILE = DATA_DIR / "calphad_feasible_pool.csv"
OUT_FILE = DATA_DIR / "scheil_descriptors.csv"


def get_tc_cache_folder() -> Path:
    """OneDrive 파일 점유 충돌 방지를 위해 로컬 AppData 경로 사용"""
    local_appdata = os.environ.get("LOCALAPPDATA")
    if not local_appdata:
        cache_folder = (
            Path.home() / "AppData" / "Local" / "ThermoCalcCache" / "scheil"
        )
    else:
        cache_folder = (
            Path(local_appdata) / "ThermoCalcCache" / "poseidon500_scheil"
        )

    cache_folder.mkdir(parents=True, exist_ok=True)
    return cache_folder


def get_target_candidates():
    """후보 목록 로드: feasible_pool이 있으면 우선 읽고, 없으면 기본 대표 후보 사용"""
    if POOL_FILE.exists():
        df_pool = pd.read_csv(POOL_FILE)
        # 예시: Ce 간격 0.01 wt% 단위로 대표 샘플링
        df_sampled = (
            df_pool.sort_values(by=["Cr_wt", "Ce_wt"])
            .groupby(np.round(df_pool["Ce_wt"], 3))
            .first()
            .reset_index(drop=True)
        )
        return df_sampled[["Ce_wt", "Cr_wt"]].to_dict(orient="records")

    # 기본 설정 5종 후보
    return [
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
        ScheilQuantity.mole_fraction_of_all_solid_phases(),
    )

    temperature_k = np.asarray(temperature_k, dtype=float)
    solid_fraction = np.asarray(solid_fraction, dtype=float)

    liquidus_k = float(np.max(temperature_k))
    solidus_k = float(np.min(temperature_k))

    # 1. Operational Solidus (99% 응고 완료 시점)
    idx_99 = np.where(solid_fraction >= 0.99)[0]
    t_99solid_k = (
        float(temperature_k[idx_99[0]]) if len(idx_99) > 0 else np.nan
    )

    # 2. 응고 균열 취약 구간 (Hot tearing susceptibility: fs 0.90 ~ 0.99 구간의 온도차)
    idx_90 = np.where(solid_fraction >= 0.90)[0]
    t_90solid_k = (
        float(temperature_k[idx_90[0]]) if len(idx_90) > 0 else np.nan
    )

    vulnerable_temp_range = (
        t_90solid_k - t_99solid_k
        if (np.isfinite(t_90solid_k) and np.isfinite(t_99solid_k))
        else np.nan
    )

    return {
        "liquidus_C": liquidus_k - 273.15,
        "scheil_end_C": solidus_k - 273.15,
        "T_90solid_C": (
            t_90solid_k - 273.15 if np.isfinite(t_90solid_k) else np.nan
        ),
        "T_99solid_C": (
            t_99solid_k - 273.15 if np.isfinite(t_99solid_k) else np.nan
        ),
        "freezing_range_C": liquidus_k - solidus_k,
        "crack_susceptibility_dT": vulnerable_temp_range,  # 0.90 <= fs <= 0.99 구간 (좁을수록 주조 균열 저항성 우수)
        "n_scheil_steps": len(temperature_k),
    }


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    tc_cache = get_tc_cache_folder()
    candidates = get_target_candidates()
    records = []

    print(
        f"Scheil 응고 시뮬레이션 시작 (대상: 총 {len(candidates)}개 조성)..."
    )
    print(f"캐시 디렉터리: {tc_cache.resolve()}")

    with TCPython() as start:
        start.set_cache_folder(str(tc_cache))

        system = start.select_database_and_elements(
            DATABASE, ELEMENTS
        ).get_system()

        for idx, item in enumerate(candidates, 1):
            ce_wt = item["Ce_wt"]
            cr_wt = item["Cr_wt"]
            composition = make_composition(ce_wt, cr_wt)

            record = {
                "sample_id": f"S{idx:02d}",
                "Ce_wt": ce_wt,
                "Cr_wt": cr_wt,
                **composition,
            }

            try:
                scheil_data = run_scheil(system, composition)
                record.update(scheil_data)
                record["scheil_status"] = "ok"
                print(
                    f"[{idx}/{len(candidates)}] Scheil 완료: Ce={ce_wt:.4f}, Cr={cr_wt:.2f} "
                    f"(Liquidus: {scheil_data['liquidus_C']:.1f}°C, T99: {scheil_data['T_99solid_C']:.1f}°C)"
                )
            except Exception as exc:
                record["scheil_status"] = "failed"
                record["scheil_error"] = repr(exc)
                print(f"[{idx}/{len(candidates)}] Scheil 실패: {exc}")

            records.append(record)

    df_out = pd.DataFrame(records)
    df_out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")
    print(f"\n저장 완료: {OUT_FILE.resolve()}")

    # 계산 성공 데이터 요약 출력
    success_mask = df_out["scheil_status"].eq("ok")
    if success_mask.any():
        print(
            df_out.loc[
                success_mask,
                [
                    "sample_id",
                    "Ce_wt",
                    "Cr_wt",
                    "liquidus_C",
                    "T_99solid_C",
                    "crack_susceptibility_dT",
                ],
            ].to_string(index=False)
        )


if __name__ == "__main__":
    main()