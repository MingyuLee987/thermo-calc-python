# 01b_plot_calphad_map_3d.py
#
# CALPHAD 평형 계산 결과 시각화
# 1. 3D 정격 그리드 곡면 플롯 (plot_surface 사용으로 메쉬 왜곡 방지)
# 2. 2D 컬러맵 (상전이 및 Feasible 경계 직관적 파악)

from pathlib import Path
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
import numpy as np
import pandas as pd

from config import DATA_DIR, MATRIX_PHASE, MIN_MATRIX_CR_WT, OUTPUT_DIR

CSV_FILE = DATA_DIR / "calphad_equilibrium.csv"


def load_and_clean_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"파일을 찾을 수 없습니다: {csv_path.resolve()}")

    df = pd.read_csv(csv_path)
    df.columns = df.columns.astype(str).str.strip()

    # 계산 성공 케이스만 필터링
    df = df[df["calculation_status"].astype(str).str.strip().eq("ok")].copy()

    # 숫자형 변환 및 결측치(상 미석출로 인한 NaN)를 0.0으로 대체
    numeric_cols = [
        "Ce_wt",
        "Cr_wt",
        "phasefrac_base_MS_B1",
        "phasefrac_base_M7C3_D101",
        "matrix_Cr_wt",
        "phasefrac_base_CE2C3_D5C",
    ]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0)

    df["calphad_feasible"] = df["calphad_feasible"].astype(bool)
    return df


def plot_grid_3d(
    df: pd.DataFrame,
    z_col: str,
    title: str,
    z_label: str,
    cmap: str,
    output_filename: str,
):
    """정격 그리드(Grid)를 구성하여 면 왜곡 없는 3D Surface 생성"""
    # Ce_wt (행), Cr_wt (열)로 Pivot
    pivot_z = df.pivot(index="Ce_wt", columns="Cr_wt", values=z_col)
    pivot_feas = df.pivot(
        index="Ce_wt", columns="Cr_wt", values="calphad_feasible"
    )

    cr_vals = pivot_z.columns.to_numpy()
    ce_vals = pivot_z.index.to_numpy()
    X, Y = np.meshgrid(cr_vals, ce_vals)
    Z = pivot_z.to_numpy()

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 정규 직교 그리드 곡면 (plot_trisurf 대신 plot_surface 사용)
    surf = ax.plot_surface(
        X,
        Y,
        Z,
        cmap=cmap,
        edgecolor="gray",
        linewidth=0.3,
        alpha=0.85,
        antialiased=True,
    )

    # Feasible(흰색 원) / Excluded(붉은색 X) 포인트 표시
    feasible_pts = df[df["calphad_feasible"]]
    excluded_pts = df[~df["calphad_feasible"]]

    if not feasible_pts.empty:
        ax.scatter(
            feasible_pts["Cr_wt"],
            feasible_pts["Ce_wt"],
            feasible_pts[z_col],
            color="white",
            edgecolors="black",
            s=40,
            linewidth=1.0,
            label="CALPHAD feasible",
            depthshade=False,
        )

    if not excluded_pts.empty:
        ax.scatter(
            excluded_pts["Cr_wt"],
            excluded_pts["Ce_wt"],
            excluded_pts[z_col],
            color="red",
            marker="x",
            s=45,
            linewidth=1.5,
            label="CALPHAD excluded",
            depthshade=False,
        )

    ax.set_xlabel("Cr (wt%)", labelpad=10)
    ax.set_ylabel("Ce (wt%)", labelpad=10)
    ax.set_zlabel(z_label, labelpad=10)
    ax.set_title(title, pad=20, fontsize=12, fontweight="bold")
    ax.legend(loc="upper left")
    ax.view_init(elev=25, azim=-60)

    cbar = fig.colorbar(surf, ax=ax, shrink=0.55, aspect=15, pad=0.1)
    cbar.set_label(z_label)

    fig.tight_layout()
    out_path = OUTPUT_DIR / output_filename
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path.resolve()}")
    plt.close(fig)


def plot_summary_2d(df: pd.DataFrame):
    """상전이 단차를 명확하게 관찰할 수 있는 2D 히트맵 요약도"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    configs = [
        (
            "phasefrac_base_MS_B1",
            "MnS Fraction (Pitting Site)",
            "coolwarm",
            axes[0],
        ),
        (
            "phasefrac_base_M7C3_D101",
            "M7C3 Carbide Fraction",
            "magma",
            axes[1],
        ),
        (
            "matrix_Cr_wt",
            f"Matrix Cr (wt%) [Min: {MIN_MATRIX_CR_WT}]",
            "viridis",
            axes[2],
        ),
    ]

    for z_col, title, cmap, ax in configs:
        pivot = df.pivot(index="Ce_wt", columns="Cr_wt", values=z_col)
        cr_vals = pivot.columns.to_numpy()
        ce_vals = pivot.index.to_numpy()

        c = ax.pcolormesh(
            cr_vals, ce_vals, pivot.to_numpy(), cmap=cmap, shading="auto"
        )
        fig.colorbar(c, ax=ax)

        # Excluded 포인트 위치 표시
        ex = df[~df["calphad_feasible"]]
        if not ex.empty:
            ax.scatter(
                ex["Cr_wt"],
                ex["Ce_wt"],
                color="red",
                marker="x",
                s=30,
                label="Excluded",
            )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Cr (wt%)")
        ax.set_ylabel("Ce (wt%)")
        if ax == axes[0]:
            ax.legend(loc="upper right")

    fig.tight_layout()
    out_path = OUTPUT_DIR / "corrosion_summary_2d_maps.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path.resolve()}")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_clean_data(CSV_FILE)

    # 1. 3D MnS 황화물 분율 맵 (직교 그리드 곡면)
    plot_grid_3d(
        df,
        z_col="phasefrac_base_MS_B1",
        title="Corrosion Initiation Site: MnS (MS_B1) Fraction",
        z_label="MnS Phase Fraction",
        cmap="coolwarm",
        output_filename="corrosion_3d_MnS_fraction.png",
    )

    # 2. 3D M7C3 탄화물 분율 맵 (직교 그리드 곡면)
    plot_grid_3d(
        df,
        z_col="phasefrac_base_M7C3_D101",
        title="Corrosion Depletion Factor: M7C3 Carbide Fraction",
        z_label="M7C3 Fraction",
        cmap="magma",
        output_filename="corrosion_3d_M7C3_fraction.png",
    )

    # 3. 3D Matrix 고용 Cr wt% 맵 (직교 그리드 곡면)
    plot_grid_3d(
        df,
        z_col="matrix_Cr_wt",
        title=f"Passivity Maintenance: Matrix Cr wt% ({MATRIX_PHASE})",
        z_label="Matrix Cr (wt%)",
        cmap="viridis",
        output_filename="corrosion_3d_matrix_Cr.png",
    )

    # 4. 상전이 단차 및 경계를 한눈에 보는 2D 종합 요약도
    plot_summary_2d(df)


if __name__ == "__main__":
    main()