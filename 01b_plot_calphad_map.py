# 01b_plot_calphad_map_3d.py
#
# CALPHAD equilibrium 결과를 3D 곡면 및 산점도로 Ce-Cr-Z 공간에 표시한다.
#
# 그림 1: Z축 = matrix Cr wt% (3D Surface + Feasible/Excluded Scatter)
# 그림 2: Z축 = matrix phase fraction (3D Surface + Feasible/Excluded Scatter)

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (3D 투영 활성화)

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

    missing = [col for col in required_columns if col not in df.columns]
    if missing:
        raise ValueError(f"CSV에 필요한 열이 없습니다: {missing}")

    df = df[
        df["calculation_status"].astype(str).str.strip().eq("ok")
    ].copy()

    for col in ["Ce_wt", "Cr_wt", z_column]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    finite_mask = (
        np.isfinite(df["Ce_wt"])
        & np.isfinite(df["Cr_wt"])
        & np.isfinite(df[z_column])
    )

    return df.loc[finite_mask].copy()


def draw_3d_map(ax, df_plot, z_column, title, z_label, cmap):
    x = df_plot["Cr_wt"].to_numpy()
    y = df_plot["Ce_wt"].to_numpy()
    z = df_plot[z_column].to_numpy()

    triangulation = tri.Triangulation(x, y)

    # 1. 3D 삼각망 곡면(Surface) 생성
    surf = ax.plot_trisurf(
        triangulation,
        z,
        cmap=cmap,
        edgecolor="none",
        alpha=0.85,
        antialiased=True,
    )

    # 2. Feasible / Excluded 포인트 분류
    feasible = df_plot[df_plot["calphad_feasible"].astype(bool)]
    excluded = df_plot[~df_plot["calphad_feasible"].astype(bool)]

    # 3. 데이터 포인트 3D Scatter 표시 (곡면 상단에 명확히 보이도록 zorder 설정)
    if not feasible.empty:
        ax.scatter(
            feasible["Cr_wt"],
            feasible["Ce_wt"],
            feasible[z_column],
            marker="o",
            s=50,
            facecolors="white",
            edgecolors="black",
            linewidths=1.2,
            label="CALPHAD feasible",
            depthshade=False,
        )

    if not excluded.empty:
        ax.scatter(
            excluded["Cr_wt"],
            excluded["Ce_wt"],
            excluded[z_column],
            marker="x",
            s=45,
            color="red",
            linewidths=1.5,
            label="CALPHAD excluded",
            depthshade=False,
        )

    # 축 라벨 및 타이틀 설정
    ax.set_xlabel("Cr (wt%)", labelpad=10)
    ax.set_ylabel("Ce (wt%)", labelpad=10)
    ax.set_zlabel(z_label, labelpad=10)
    ax.set_title(title, pad=20, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")

    # 뷰 앵글 기본 세팅 (상하 25도, 좌우 회전 -60도)
    ax.view_init(elev=25, azim=-60)

    return surf


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    if not CSV_FILE.exists():
        raise FileNotFoundError(
            f"파일을 찾지 못했습니다:\n{CSV_FILE.resolve()}\n\n"
            "먼저 python 01_tc_calphad_screen.py를 실행하세요."
        )

    df = pd.read_csv(CSV_FILE)
    df.columns = df.columns.astype(str).str.strip()

    matrix_cr_data = prepare_plot_data(df, "matrix_Cr_wt")
    matrix_fraction_data = prepare_plot_data(df, "matrix_phase_fraction")

    if len(matrix_cr_data) < 3 or len(matrix_fraction_data) < 3:
        raise ValueError("표면을 생성하기 위한 데이터 포인트(유한값)가 3개 미만입니다.")

    # -------------------------------------------------------------
    # Figure 1: 3D Matrix Cr Map
    # -------------------------------------------------------------
    fig1 = plt.figure(figsize=(11, 8))
    ax1 = fig1.add_subplot(111, projection="3d")

    surf1 = draw_3d_map(
        ax1,
        matrix_cr_data,
        "matrix_Cr_wt",
        f"CALPHAD Matrix Cr 3D Surface: {MATRIX_PHASE}",
        "Matrix Cr (wt%)",
        "viridis",
    )

    cbar1 = fig1.colorbar(surf1, ax=ax1, shrink=0.6, aspect=15, pad=0.1)
    cbar1.set_label("Matrix Cr (wt%)")

    fig1.tight_layout()
    out1 = OUTPUT_DIR / "calphad_3d_matrix_Cr.png"
    fig1.savefig(out1, dpi=300, bbox_inches="tight")
    print(f"Saved: {out1.resolve()}")

    # -------------------------------------------------------------
    # Figure 2: 3D Matrix Phase Fraction Map
    # -------------------------------------------------------------
    fig2 = plt.figure(figsize=(11, 8))
    ax2 = fig2.add_subplot(111, projection="3d")

    surf2 = draw_3d_map(
        ax2,
        matrix_fraction_data,
        "matrix_phase_fraction",
        f"CALPHAD Matrix Phase Fraction 3D Surface: {MATRIX_PHASE}",
        f"NP({MATRIX_PHASE})",
        "plasma",
    )

    cbar2 = fig2.colorbar(surf2, ax=ax2, shrink=0.6, aspect=15, pad=0.1)
    cbar2.set_label(f"NP({MATRIX_PHASE})")

    fig2.tight_layout()
    out2 = OUTPUT_DIR / "calphad_3d_matrix_phase_fraction.png"
    fig2.savefig(out2, dpi=300, bbox_inches="tight")
    print(f"Saved: {out2.resolve()}")

    plt.show()


if __name__ == "__main__":
    main()