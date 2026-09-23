# 01b_plot_calphad_map_3d.py
#
# CALPHAD 평형 계산 결과 시각화 (논문 공정: 900 °C 용체화 열처리, Matrix: FCC_A1)
# 1. 3D 정격 그리드 곡면 플롯 (plot_surface)
# 2. 2D 컬러맵 종합 요약 (개재물 소멸 및 Feasible 경계 관찰)

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

    # 계산 성공 데이터 필터링
    df = df[df["calculation_status"].astype(str).str.strip().eq("ok")].copy()

    # 숫자형 변환 및 미석출 상(결측치) 0.0 처리
    numeric_cols = [
        "Ce_wt",
        "Cr_wt",
        "matrix_phase_fraction",
        "matrix_Cr_wt",
        "phasefrac_base_MS_B1",
        "phasefrac_base_M7C3_D101",
        "phasefrac_base_CE2S3",
        "phasefrac_base_CE3S4_D73",
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
    """정규 직교 그리드(plot_surface)를 사용해 면 왜곡 없는 3D 곡면 생성"""
    pivot_z = df.pivot(index="Ce_wt", columns="Cr_wt", values=z_col).sort_index(
        ascending=True
    )
    pivot_z = pivot_z.reindex(sorted(pivot_z.columns), axis=1)

    cr_vals = pivot_z.columns.to_numpy(dtype=float)
    ce_vals = pivot_z.index.to_numpy(dtype=float)
    X, Y = np.meshgrid(cr_vals, ce_vals)
    Z = pivot_z.to_numpy(dtype=float)

    fig = plt.figure(figsize=(11, 8))
    ax = fig.add_subplot(111, projection="3d")

    # 직교 그리드 곡면 생성
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
    """900 °C 열처리 조건의 부식 관련 핵심 지표 2D 히트맵 요약"""
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))

    configs = [
        (
            "phasefrac_base_MS_B1",
            "MnS Fraction (Pitting Site)",
            "coolwarm",
            axes[0],
        ),
        (
            "matrix_phase_fraction",
            f"Matrix Fraction ({MATRIX_PHASE})",
            "plasma",
            axes[1],
        ),
        (
            "matrix_Cr_wt",
            f"Matrix Cr wt% (Min >= {MIN_MATRIX_CR_WT}%)",
            "viridis",
            axes[2],
        ),
    ]

    for z_col, title, cmap, ax in configs:
        pivot = df.pivot(
            index="Ce_wt", columns="Cr_wt", values=z_col
        ).sort_index(ascending=True)
        pivot = pivot.reindex(sorted(pivot.columns), axis=1)

        cr_vals = pivot.columns.to_numpy(dtype=float)
        ce_vals = pivot.index.to_numpy(dtype=float)

        c = ax.pcolormesh(
            cr_vals, ce_vals, pivot.to_numpy(dtype=float), cmap=cmap, shading="auto"
        )
        fig.colorbar(c, ax=ax)

        # Excluded 포인트 오버레이
        ex = df[~df["calphad_feasible"]]
        if not ex.empty:
            ax.scatter(
                ex["Cr_wt"],
                ex["Ce_wt"],
                color="red",
                marker="x",
                s=35,
                linewidth=1.2,
                label="Excluded",
            )

        ax.set_title(title, fontsize=11, fontweight="bold")
        ax.set_xlabel("Cr (wt%)")
        ax.set_ylabel("Ce (wt%)")
        if ax == axes[0] and not ex.empty:
            ax.legend(loc="upper right")

    fig.tight_layout()
    out_path = OUTPUT_DIR / "corrosion_summary_2d_maps.png"
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    print(f"Saved: {out_path.resolve()}")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_and_clean_data(CSV_FILE)

    # 1. MnS 황화물 분율 (공식 기점 개재물 - 낮을수록 내식성 우수)
    plot_grid_3d(
        df,
        z_col="phasefrac_base_MS_B1",
        title="Corrosion Initiation Site: MnS (MS_B1) Phase Fraction",
        z_label="MnS Phase Fraction",
        cmap="coolwarm",
        output_filename="corrosion_3d_MnS_fraction.png",
    )

    # 2. 900 °C Austenite (FCC_A1) 기지상 분율
    plot_grid_3d(
        df,
        z_col="matrix_phase_fraction",
        title=f"High-Temp Matrix Phase Fraction: {MATRIX_PHASE} (900 °C Sol.)",
        z_label=f"NP({MATRIX_PHASE})",
        cmap="plasma",
        output_filename="corrosion_3d_matrix_phase_fraction.png",
    )

    # 3. Austenite 기지 내 고용 Cr 함량 (수냉 후 베이나이트의 내식 피막 기초 원소)
    plot_grid_3d(
        df,
        z_col="matrix_Cr_wt",
        title=f"Solid-Solution Cr in Matrix: {MATRIX_PHASE} (900 °C Sol.)",
        z_label="Matrix Cr (wt%)",
        cmap="viridis",
        output_filename="corrosion_3d_matrix_Cr.png",
    )

    # 4. 2D 종합 요약 맵 생성
    plot_summary_2d(df)


if __name__ == "__main__":
    main()