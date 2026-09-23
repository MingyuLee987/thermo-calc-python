# 02b_plot_scheil_descriptors.py
#
# Scheil 응고 해석 결과(scheil_descriptors.csv) 시각화 스크립트
# - Ce 첨가량에 따른 액상선/고상선, 응고 온도폭, 응고 균열 민감도(Hot Tearing) 분석
# Output: output/scheil_solidification_analysis.png

from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from config import DATA_DIR, OUTPUT_DIR

CSV_FILE = DATA_DIR / "scheil_descriptors.csv"
OUT_IMG = OUTPUT_DIR / "scheil_solidification_analysis.png"


def load_data(csv_path: Path) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(
            f"파일을 찾을 수 없습니다: {csv_path.resolve()}\n"
            "먼저 02_tc_scheil_selected.py를 실행하세요."
        )

    df = pd.read_csv(csv_path)
    df.columns = df.columns.astype(str).str.strip()

    # 계산 성공 데이터 필터링
    df = df[df.get("scheil_status", pd.Series(["ok"] * len(df))).eq("ok")].copy()

    # 정렬 (Ce 첨가량 기준)
    df = df.sort_values(by="Ce_wt").reset_index(drop=True)
    return df


def plot_scheil_summary(df: pd.DataFrame):
    plt.rcParams["font.sans-serif"] = "DejaVu Sans"
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    ce = df["Ce_wt"].to_numpy()

    # -------------------------------------------------------------
    # Plot 1: 액상선 및 고상선 거동 (Liquidus & Solidus)
    # -------------------------------------------------------------
    ax1 = axes[0, 0]
    ax1.plot(ce, df["liquidus_C"], marker="o", color="#d62728", lw=2, label=r"Liquidus ($T_L$, Primary Ce Phase)")
    ax1.plot(ce, df["scheil_end_C"], marker="s", color="#1f77b4", lw=2, label=r"Solidus ($T_{\mathrm{solidus}}$, Scheil End)")
    if "T_90solid_C" in df.columns and "T_99solid_C" in df.columns:
        ax1.plot(ce, df["T_90solid_C"], "--", color="#ff7f0e", alpha=0.7, label=r"$T_{90\% \mathrm{solid}}$")
        ax1.plot(ce, df["T_99solid_C"], ":", color="#2ca02c", alpha=0.7, label=r"$T_{99\% \mathrm{solid}}$")

    ax1.set_title("Solidification Temperatures vs. Ce Content", fontsize=12, fontweight="bold")
    ax1.set_xlabel("Ce Content (wt%)", fontsize=11)
    ax1.set_ylabel("Temperature (°C)", fontsize=11)
    ax1.grid(True, linestyle="--", alpha=0.5)
    ax1.legend(loc="best", fontsize=9)

    # -------------------------------------------------------------
    # Plot 2: 전체 응고 온도폭 (Freezing Range: TL - TS)
    # -------------------------------------------------------------
    ax2 = axes[0, 1]
    ax2.plot(ce, df["freezing_range_C"], marker="^", color="#9467bd", lw=2)
    ax2.set_title("Total Freezing Range ($T_L - T_{\mathrm{end}}$)", fontsize=12, fontweight="bold")
    ax2.set_xlabel("Ce Content (wt%)", fontsize=11)
    ax2.set_ylabel(r"$\Delta T_{\mathrm{freeze}}$ (°C)", fontsize=11)
    ax2.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Plot 3: 응고 균열 민감도 (Hot Tearing Susceptibility: T90 - T99)
    # -------------------------------------------------------------
    ax3 = axes[1, 0]
    crack_dt = df["crack_susceptibility_dT"].to_numpy()
    bars = ax3.bar(ce, crack_dt, width=0.003, color="#e377c2", edgecolor="black", alpha=0.8)
    
    # 추세선
    ax3.plot(ce, crack_dt, color="#c51b7d", lw=1.5, marker="o", markersize=4)
    ax3.set_title(r"Hot Tearing Vulnerability ($\Delta T_{90\% - 99\%}$)", fontsize=12, fontweight="bold")
    ax3.set_xlabel("Ce Content (wt%)", fontsize=11)
    ax3.set_ylabel(r"$T_{90\%} - T_{99\%}$ (°C) [Lower is Safer]", fontsize=11)
    ax3.grid(True, linestyle="--", alpha=0.5)

    # -------------------------------------------------------------
    # Plot 4: 주조 공정 안전 마진 및 권장 영역 (Summary Map)
    # -------------------------------------------------------------
    ax4 = axes[1, 1]
    # Ce 첨가량에 따른 T99solid 저하 폭을 시각화
    t99 = df["T_99solid_C"] if "T_99solid_C" in df.columns else df["scheil_end_C"]
    ax4.plot(ce, t99, marker="D", color="#17becf", lw=2, label="Terminal Solidification Temp")
    
    # 0.01 ~ 0.03 wt% Ce 영역 강조 (일반적인 개재물 제어 최적 권장 구간)
    ax4.axvspan(0.010, 0.035, color="green", alpha=0.15, label="Target Safe Window (0.01~0.035 wt%)")
    ax4.axvline(0.040, color="red", linestyle="--", lw=1.5, label="High Segregation Risk (> 0.04 wt%)")

    ax4.set_title("Terminal Freezing & Recommended Ce Window", fontsize=12, fontweight="bold")
    ax4.set_xlabel("Ce Content (wt%)", fontsize=11)
    ax4.set_ylabel("Terminal Solid Temp (°C)", fontsize=11)
    ax4.grid(True, linestyle="--", alpha=0.5)
    ax4.legend(loc="lower left", fontsize=9)

    fig.tight_layout()
    fig.savefig(OUT_IMG, dpi=300, bbox_inches="tight")
    print(f"\n[완료] Scheil 분석 그래프가 저장되었습니다: {OUT_IMG.resolve()}")
    plt.close(fig)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = load_data(CSV_FILE)
    plot_scheil_summary(df)


if __name__ == "__main__":
    main()