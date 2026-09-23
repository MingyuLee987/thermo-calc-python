# 01b_plot_calphad_map.py

import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.tri as tri

from config import DATA_DIR, OUTPUT_DIR

df = pd.read_csv(DATA_DIR / "calphad_equilibrium.csv")
df = df[df["calculation_status"] == "ok"].copy()

x = df["Cr_wt"].to_numpy()
y = df["Ce_wt"].to_numpy()
z = df["matrix_Cr_wt"].to_numpy()

triangulation = tri.Triangulation(x, y)

fig, ax = plt.subplots(figsize=(9, 6))

contour = ax.tricontourf(
    triangulation,
    z,
    levels=20,
    cmap="viridis"
)

cbar = fig.colorbar(contour, ax=ax)
cbar.set_label("CALPHAD matrix Cr (wt%)")

bad = df[~df["calphad_feasible"]]
good = df[df["calphad_feasible"]]

ax.scatter(
    good["Cr_wt"],
    good["Ce_wt"],
    marker="o",
    facecolors="none",
    edgecolors="white",
    linewidths=1.2,
    label="CALPHAD feasible"
)

ax.scatter(
    bad["Cr_wt"],
    bad["Ce_wt"],
    marker="x",
    color="red",
    s=35,
    label="CALPHAD excluded"
)

ax.set_xlabel("Cr (wt%)")
ax.set_ylabel("Ce (wt%)")
ax.set_title("CALPHAD screening map: matrix Cr and feasible region")
ax.legend()

fig.tight_layout()
fig.savefig(
    OUTPUT_DIR / "calphad_map_matrix_Cr.png",
    dpi=300
)

plt.show()