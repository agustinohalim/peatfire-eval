"""
Gambar peluang peringkat untuk naskah JEPIN dari log enumerasi eksak percobaan 9 (bagian A).

Pakai:
    python gambar_jepin.py
Keluaran: Gambar_JEPIN/fig_peluang_peringkat (percobaan 9 A) dan fig_panjang_rekaman (percobaan 11 D)
"""
import os
import re
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

BASE = os.path.dirname(os.path.abspath(__file__))
# Log dicari di folder kerja (DL_FIRE_NASIONAL/oni_akhir) atau di logs/jepin pada repo publik.
LOGDIR = next(d for d in (os.path.join(BASE, "DL_FIRE_NASIONAL", "oni_akhir"), os.path.join(BASE, "logs", "jepin"))
              if os.path.exists(os.path.join(d, "percobaan9_jepin_tahun.log")))
LOG = os.path.join(LOGDIR, "percobaan9_jepin_tahun.log")
BIRU, TINTA = "#1F5FA8", "#1F2328"
plt.rcParams.update({"savefig.dpi": 300, "savefig.bbox": "tight", "font.family": "DejaVu Sans",
                     "font.size": 9, "xtick.color": "#5A6169", "ytick.color": "#5A6169",
                     "xtick.labelsize": 8, "ytick.labelsize": 8})

teks = open(LOG, encoding="utf-8").read().split("B. BUANG")[0]
baris = re.findall(r"^  (\S[\w ,]+?)\s+((?:\s*\d+\.\d){7})\s*$", teks, re.M)
nama = [b[0].strip() for b in baris]
M = np.array([[float(x) for x in b[1].split()] for b in baris])
assert M.shape == (7, 7), M.shape

cmap = LinearSegmentedColormap.from_list("biru", ["#FFFFFF", "#DCE7F5", "#8FB0DB", BIRU, "#0F3666"])
fig, ax = plt.subplots(figsize=(6.4, 3.4))
ax.imshow(M, cmap=cmap, vmin=0, vmax=100, aspect="auto")
for i in range(7):
    for k in range(7):
        if M[i, k] >= 0.5:
            ax.text(k, i, f"{M[i, k]:.0f}", ha="center", va="center", fontsize=8,
                    color="white" if M[i, k] > 55 else TINTA)
ax.set_xticks(range(7), [str(k) for k in range(1, 8)])
ax.set_yticks(range(7), nama)
ax.set_xlabel("Rank by AUC-PR (1 = best)")
for s in ax.spines.values():
    s.set_visible(False)
ax.tick_params(length=0)
out = os.path.join(BASE, "Gambar_JEPIN")
os.makedirs(out, exist_ok=True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(out, f"fig_peluang_peringkat.{ext}"))
print("tersimpan: Gambar_JEPIN/fig_peluang_peringkat.png dan .pdf")

# ---------------------------------------------------------------- panjang rekaman (percobaan 11 D)
LOG11 = os.path.join(LOGDIR, "percobaan11_jepin_v4.log")
t11 = open(LOG11, encoding="utf-8").read().split("D. PANJANG")[1]
k = np.array([[float(a), float(b), float(c)] for a, b, c in
              re.findall(r"^\s+(\d+) tahun .*?GB juara\s+([\d.]+)%; peringkat utuh sama\s+([\d.]+)%", t11, re.M)])
KELABU = "#6B7280"
fig, ax = plt.subplots(figsize=(5.4, 3.2))
ax.plot(k[:, 0], k[:, 1], "-o", color=BIRU, lw=2, ms=6)
ax.plot(k[:, 0], k[:, 2], "--o", color=KELABU, lw=2, ms=6)
ax.text(28, k[-1, 1] + 4, "gradient boosting ranked first", ha="right", color=BIRU, fontsize=8)
ax.text(11, 46, "full seven-model ranking\nreproduced", ha="left", color=KELABU, fontsize=8)
ax.axvline(7, color="#B8BCC4", lw=1, zorder=0)
ax.text(7.3, 3, "observed\nrecord", fontsize=7, color=KELABU, va="bottom")
ax.set_xticks([2, 4, 7, 10, 14, 21, 28])
ax.set_xlabel("Test record length (years drawn from the seven observed)")
ax.set_ylabel("Share of resamples (%)")
ax.set_ylim(0, 105)
for s in ("top", "right"):
    ax.spines[s].set_visible(False)
ax.grid(axis="y", color="#E7E9EC")
ax.set_axisbelow(True)
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(out, f"fig_panjang_rekaman.{ext}"))
print("tersimpan: Gambar_JEPIN/fig_panjang_rekaman.png dan .pdf")
