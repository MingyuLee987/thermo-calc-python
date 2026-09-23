# 04_active_learning.py

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from sklearn.preprocessing import StandardScaler
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import (
    ConstantKernel,
    Matern,
    WhiteKernel
)

from config import DATA_DIR, OUTPUT_DIR


POOL_FILE = DATA_DIR / "calphad_feasible_pool.csv"
SP240_FILE = DATA_DIR / "sp240_results.csv"
NEXT_FILE = DATA_DIR / "next_candidates.csv"


def minmax(series):
    x = np.asarray(series, dtype=float)

    if np.all(~np.isfinite(x)):
        return np.zeros_like(x)

    x_min = np.nanmin(x)
    x_max = np.nanmax(x)

    if np.isclose(x_min, x_max):
        return np.zeros_like(x)

    return (x - x_min) / (x_max - x_min)


def same_composition(a, b, tol=1e-8):
    return (
        np.isclose(a["Ce_wt"], b["Ce_wt"], atol=tol)
        and np.isclose(a["Cr_wt"], b["Cr_wt"], atol=tol)
    )


def get_log_icorr_noise(exp):
    """
    log10(icorr)의 오차 전파.
    sigma_log10(i) ≈ sigma_i / (i ln(10))
    """
    i_mean = exp["icorr_mean_A_cm2"].to_numpy(dtype=float)
    i_sd = exp["icorr_sd_A_cm2"].to_numpy(dtype=float)

    sigma_log = i_sd / (i_mean * np.log(10.0))

    # 0 또는 비정상값 방지용 최소 관측오차.
    return np.clip(sigma_log, 0.03, 1.00)


def calphad_boundary_score(pool):
    """
    CALPHAD matrix Cr의 local gradient를 경계점 지표로 사용.
    이후에는 phase fraction gradient, Scheil segregation gradient를
    추가하는 방식으로 확장할 수 있음.
    """
    x = pool[["Ce_wt", "Cr_wt"]].to_numpy(dtype=float)
    q = pool["matrix_Cr_wt"].to_numpy(dtype=float)

    score = np.zeros(len(pool))

    for i in range(len(pool)):
        distance = np.sqrt(((x - x[i]) ** 2).sum(axis=1))
        valid = distance > 0

        if valid.sum() == 0:
            continue

        j_candidates = np.where(valid)[0]
        j = j_candidates[np.argmin(distance[valid])]

        score[i] = abs(q[i] - q[j]) / distance[j]

    return score


def select_first_distinct(sorted_df, used_pairs):
    for _, row in sorted_df.iterrows():
        pair = (round(row["Ce_wt"], 8), round(row["Cr_wt"], 8))
        if pair not in used_pairs:
            used_pairs.add(pair)
            return row.copy()

    raise RuntimeError("No distinct candidate remains.")


def main():
    pool = pd.read_csv(POOL_FILE)
    exp = pd.read_csv(SP240_FILE)

    # CALPHAD을 통과한 후보만 사용.
    pool = pool[pool["calphad_feasible"]].copy()

    key_cols = ["Ce_wt", "Cr_wt"]

    measured_set = {
        (round(r.Ce_wt, 8), round(r.Cr_wt, 8))
        for r in exp[key_cols].itertuples(index=False)
    }

    pool["already_measured"] = [
        (round(row.Ce_wt, 8), round(row.Cr_wt, 8))
        in measured_set
        for row in pool[key_cols].itertuples(index=False)
    ]

    unmeasured = pool.loc[~pool["already_measured"]].copy()

    if len(exp) < 6:
        raise ValueError(
            "GPR 기반 active learning 전 최소 6개 이상의 서로 다른 "
            "조성 실험점을 확보하세요."
        )

    if len(unmeasured) < 3:
        raise ValueError(
            "미측정 CALPHAD-feasible 후보가 3개 미만입니다."
        )

    # -------------------------------------------------------
    # 초기에는 Ce와 Cr만 GPR 입력값으로 사용.
    # 데이터가 더 쌓인 뒤 matrix_Cr_wt, freezing_range_C 등 추가.
    # -------------------------------------------------------
    features = ["Ce_wt", "Cr_wt"]

    x_train_raw = exp[features].to_numpy(dtype=float)
    x_pool_raw = unmeasured[features].to_numpy(dtype=float)

    x_scaler = StandardScaler()
    x_train = x_scaler.fit_transform(x_train_raw)
    x_pool = x_scaler.transform(x_pool_raw)

    y_log_icorr = np.log10(
        exp["icorr_mean_A_cm2"].to_numpy(dtype=float)
    )
    y_epit = exp["Epit_mean_V_AgAgCl"].to_numpy(dtype=float)

    noise_log_icorr = get_log_icorr_noise(exp)

    if "Epit_sd_V" in exp.columns:
        noise_epit = exp["Epit_sd_V"].to_numpy(dtype=float)
        noise_epit = np.clip(noise_epit, 0.005, 0.10)
    else:
        noise_epit = np.full(len(exp), 0.020)

    kernel_i = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=np.ones(len(features)),
            length_scale_bounds=(1e-2, 1e3),
            nu=2.5
        )
        + WhiteKernel(
            noise_level=1e-4,
            noise_level_bounds=(1e-8, 1.0)
        )
    )

    kernel_p = (
        ConstantKernel(1.0, (1e-3, 1e3))
        * Matern(
            length_scale=np.ones(len(features)),
            length_scale_bounds=(1e-2, 1e3),
            nu=2.5
        )
        + WhiteKernel(
            noise_level=1e-4,
            noise_level_bounds=(1e-8, 1.0)
        )
    )

    gp_icorr = GaussianProcessRegressor(
        kernel=kernel_i,
        alpha=noise_log_icorr ** 2,
        normalize_y=True,
        n_restarts_optimizer=30,
        random_state=42
    )

    gp_epit = GaussianProcessRegressor(
        kernel=kernel_p,
        alpha=noise_epit ** 2,
        normalize_y=True,
        n_restarts_optimizer=30,
        random_state=42
    )

    gp_icorr.fit(x_train, y_log_icorr)
    gp_epit.fit(x_train, y_epit)

    mu_i, std_i = gp_icorr.predict(x_pool, return_std=True)
    mu_p, std_p = gp_epit.predict(x_pool, return_std=True)

    unmeasured["pred_log10_icorr"] = mu_i
    unmeasured["pred_icorr_A_cm2"] = 10.0 ** mu_i
    unmeasured["std_log10_icorr"] = std_i

    unmeasured["pred_Epit_V_AgAgCl"] = mu_p
    unmeasured["std_Epit_V"] = std_p

    # CALPHAD 경계점 지표.
    # phase fraction 변화, Scheil 결과가 있으면 추후 결합 가능.
    unmeasured["calphad_boundary_score"] = calphad_boundary_score(
        unmeasured
    )

    # uncertainty는 두 target의 예측 표준편차를 같은 범위로 정규화해 합침.
    unmeasured["uncertainty_score"] = (
        0.5 * minmax(unmeasured["std_log10_icorr"])
        + 0.5 * minmax(unmeasured["std_Epit_V"])
    )

    # C 후보: 모델이 잘 모르는 영역을 우선하되,
    # CALPHAD 변화가 큰 경계영역에 약간의 가중치를 부여.
    unmeasured["exploration_score"] = (
        0.70 * unmeasured["uncertainty_score"]
        + 0.30 * minmax(unmeasured["calphad_boundary_score"])
    )

    used_pairs = set()

    # A: 예상 icorr 최소.
    A = select_first_distinct(
        unmeasured.sort_values(
            "pred_log10_icorr",
            ascending=True
        ),
        used_pairs
    )
    A["selection_type"] = "A_low_predicted_icorr"

    # B: 예상 Epit 최대.
    B = select_first_distinct(
        unmeasured.sort_values(
            "pred_Epit_V_AgAgCl",
            ascending=False
        ),
        used_pairs
    )
    B["selection_type"] = "B_high_predicted_Epit"

    # C: uncertainty + CALPHAD 경계.
    C = select_first_distinct(
        unmeasured.sort_values(
            "exploration_score",
            ascending=False
        ),
        used_pairs
    )
    C["selection_type"] = (
        "C_high_uncertainty_near_CALPHAD_transition"
    )

    next_df = pd.DataFrame([A, B, C])

    display_columns = [
        "selection_type",
        "Ce_wt",
        "Cr_wt",
        "pred_log10_icorr",
        "pred_icorr_A_cm2",
        "std_log10_icorr",
        "pred_Epit_V_AgAgCl",
        "std_Epit_V",
        "matrix_phase_fraction",
        "matrix_Cr_wt",
        "harmful_phase_fraction",
        "calphad_boundary_score",
        "uncertainty_score",
        "exploration_score"
    ]

    next_df = next_df[
        [c for c in display_columns if c in next_df.columns]
    ]

    next_df.to_csv(
        NEXT_FILE,
        index=False,
        encoding="utf-8-sig"
    )

    print("\n=== Updated GPR kernels ===")
    print("log10(icorr):", gp_icorr.kernel_)
    print("Epit:", gp_epit.kernel_)

    print("\n=== Next experimental candidates ===")
    print(next_df.to_string(index=False))

    # 전체 후보에 대한 prediction map 저장.
    create_prediction_maps(
        pool,
        exp,
        unmeasured,
        next_df
    )


def create_prediction_maps(pool, exp, unmeasured, next_df):
    """실험점, 추천점, CALPHAD 허용영역을 같이 보여주는 그림."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5.5))

    plots = [
        (
            "pred_log10_icorr",
            r"Predicted $\log_{10}(i_{\mathrm{corr}}$ [A/cm$^2$])",
            "viridis_r"
        ),
        (
            "pred_Epit_V_AgAgCl",
            r"Predicted $E_{\mathrm{pit}}$ (V vs Ag/AgCl)",
            "plasma"
        )
    ]

    for ax, (column, label, cmap) in zip(axes, plots):
        triangulation = tri.Triangulation(
            unmeasured["Cr_wt"],
            unmeasured["Ce_wt"]
        )

        contour = ax.tricontourf(
            triangulation,
            unmeasured[column],
            levels=20,
            cmap=cmap
        )

        cbar = fig.colorbar(contour, ax=ax)
        cbar.set_label(label)

        ax.scatter(
            exp["Cr_wt"],
            exp["Ce_wt"],
            s=60,
            marker="o",
            color="white",
            edgecolor="black",
            label="Measured SP-240"
        )

        for _, row in next_df.iterrows():
            ax.scatter(
                row["Cr_wt"],
                row["Ce_wt"],
                s=160,
                marker="*",
                color="red",
                edgecolor="black",
                zorder=5
            )
            ax.annotate(
                row["selection_type"][0],
                (row["Cr_wt"], row["Ce_wt"]),
                xytext=(5, 5),
                textcoords="offset points",
                fontweight="bold"
            )

        ax.set_xlabel("Cr (wt%)")
        ax.set_ylabel("Ce (wt%)")
        ax.set_title(label)
        ax.legend(loc="best")

    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / "active_learning_prediction_maps.png",
        dpi=300
    )
    plt.show()


if __name__ == "__main__":
    main()