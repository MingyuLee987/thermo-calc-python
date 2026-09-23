# 01b_plot_calphad_map.py
#
# CALPHAD equilibrium 결과를 Ce-Cr 조성 평면에 표시한다.
#
# 그림 1:
# - 색상: matrix Cr wt%
# - 흰 원: CALPHAD feasible 조성
# - 빨간 X: CALPHAD excluded 조성
#
# 그림 2:
# - 색상: matrix phase fraction
# - 흰 원: CALPHAD feasible 조성
# - 빨간 X: CALPHAD excluded 조성

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from config import DATA_DIR, OUTPUT_DIR, MATRIX_PHASE


CSV_FILE = DATA_DIR / "calphad_equilibrium.csv"


def prepare_plot_data(df, z_column):
    required_columns = [
        "Ce_wt",
        "Cr_wt",
        z_column,
        "calculation_status",
        "calphad_feasible",
    ]

    missing = [
        col for col in required_columns
        if col not in df.columns
    ]

    if missing:
        raise ValueError(
            f"CSV에 필요한 열이 없습니다: {missing}"
        )

    df = df[
        df["calculation_status"]
        .astype(str)
        .str.strip()
        .eq("ok")
    ].copy()

    for col in ["Ce_wt", "Cr_wt", z_column]:
        df[col] = pd.to_numeric(
            df[col],
            errors="coerce"
        )

    finite_mask = (
        np.isfinite(df["Ce_wt"])
        & np.isfinite(df["Cr_wt"])
        & np.isfinite(df[z_column])
    )

    return df.loc[finite_mask].copy()


def draw_map(ax, df_plot, z_column, title, colorbar_label, cmap):
    triangulation = tri.Triangulation(
        df_plot["Cr_wt"].to_numpy(),
        df_plot["Ce_wt"].to_numpy(),
    )

    contour = ax.tricontourf(
        triangulation,
        df_plot[z_column].to_numpy(),
        levels=20,
        cmap=cmap,
    )

    feasible = df_plot[
        df_plot["calphad_feasible"]
        .astype(bool)
    ]

    excluded = df_plot[
        ~df_plot["calphad_feasible"]
        .astype(bool)
    ]

    ax.scatter(
        feasible["Cr_wt"],
        feasible["Ce_wt"],
        marker="o",
        s=70,
        facecolors="none",
        edgecolors="white",
        linewidths=1.5,
        label="CALPHAD feasible",
        zorder=3,
    )

    ax.scatter(
        excluded["Cr_wt"],
        excluded["Ce_wt"],
        marker="x",
        s=40,
        color="red",
        linewidths=1.0,
        label="CALPHAD excluded",
        zorder=3,
    )

    ax.set_xlabel("Cr (wt%)")
    ax.set_ylabel("Ce (wt%)")
    ax.set_title(title)
    ax.grid(alpha=0.25, linestyle="--")
    ax.legend(loc="best")

    return contour


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"파일을 찾지 못했습니다:\n{CSV_FILE.resolve()}\n\n"
            "먼저 python 01_tc_calphad_screen.py를 실행하세요."
        )

    df = pd.read_csv(CSV_FILE)
    df.columns = df.columns.astype(str).str.strip()

    print("=" * 70)
    print("Loaded CALPHAD CSV")
    print("=" * 70)
    print(CSV_FILE.resolve())

    print("\nCSV columns:")
    print(df.columns.tolist())

    matrix_cr_data = prepare_plot_data(
        df,
        "matrix_Cr_wt"
    )

    matrix_fraction_data = prepare_plot_data(
        df,
        "matrix_phase_fraction"
    )

    print("\nData summary:")
    print(f"matrix Cr plot points       : {len(matrix_cr_data)}")
    print(f"matrix phase plot points    : {len(matrix_fraction_data)}")

    if len(matrix_cr_data) < 3:
        raise ValueError(
            "matrix_Cr_wt가 유한한 데이터가 3개 미만입니다.\n"
            "01_tc_calphad_screen.py를 새 코드로 교체한 뒤 다시 실행하세요."
        )

    if len(matrix_fraction_data) < 3:
        raise ValueError(
            "matrix_phase_fraction이 유한한 데이터가 3개 미만입니다.\n"
            "01_tc_calphad_screen.py의 Thermo-Calc quantity extraction을 확인하세요."
        )

    # -------------------------------------------------------------
    # Figure 1: Matrix Cr map
    # -------------------------------------------------------------
    fig1, ax1 = plt.subplots(figsize=(10, 7))

    contour1 = draw_map(
        ax1,
        matrix_cr_data,
        "matrix_Cr_wt",
        f"CALPHAD Matrix Cr Map: {MATRIX_PHASE}",
        "Matrix Cr (wt%)",
        "viridis",
    )

    cbar1 = fig1.colorbar(contour1, ax=ax1)
    cbar1.set_label("Matrix Cr (wt%)")

    fig1.tight_layout()

    out1 = OUTPUT_DIR / "calphad_map_matrix_Cr.png"

    fig1.savefig(
        out1,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {out1.resolve()}")

    # -------------------------------------------------------------
    # Figure 2: Matrix phase fraction map
    # -------------------------------------------------------------
    fig2, ax2 = plt.subplots(figsize=(10, 7))

    contour2 = draw_map(
        ax2,
        matrix_fraction_data,
        "matrix_phase_fraction",
        f"CALPHAD Matrix Phase Fraction Map: {MATRIX_PHASE}",
        f"NP({MATRIX_PHASE})",
        "plasma",
    )

    cbar2 = fig2.colorbar(contour2, ax=ax2)
    cbar2.set_label(f"NP({MATRIX_PHASE})")

    fig2.tight_layout()

    out2 = OUTPUT_DIR / "calphad_map_matrix_phase_fraction.png"

    fig2.savefig(
        out2,
        dpi=300,
        bbox_inches="tight"
    )

    print(f"Saved: {out2.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()