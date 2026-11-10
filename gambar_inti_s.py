"""
Gambar hasil utama naskah revisi (Fig. 3; fase I Hasil_Patokan_14, ditambah fase M, N dan O Hasil_Patokan_18-20;
baris VIIRS dengan ambang tahun latih dari fase S Hasil_Patokan_25 dan fase P Hasil_Patokan_22, penyimpangan 10):
Brier skill gradient boosting dan EF-L atas kalender mentah, GB atas EF-P, GB atas EF-L dan GB atas EF-LD (hanya
VIIRS dan prakiraan maju MODIS, satu-satunya desain MODIS tempat EF-LD dinilai), dengan selang dua tahap (garis
tebal) dan selang tahunan t (garis tipis), untuk VIIRS 2020-2025, prakiraan maju MODIS 2008-2019 (fase J) dan
empat desain MODIS LOYO 2003-2011, dua sasaran (per kab dulu); panel (c) bagian keunggulan GB atas kalender yang
hilang oleh offset periode dan oleh geser per tahun: VIIRS, MODIS maju (fase N) dan MODIS semua tahun lain.

Angka dibaca dari log, bukan diketik:
  DL_FIRE_NASIONAL/penuh/patokan_ambang_s_tabel_{kab,nas}.log (VIIRS fase S: BSS, ambang tahun latih; cp1252)
  DL_FIRE_NASIONAL/penuh/patokan_tingkat_p_{kab,nas}.log   (VIIRS fase P: bagian tingkat, ambang tahun latih)
  DL_FIRE_MODIS/patokan_modis_i_{kab,nas}.log               (MODIS fase I, GB rata-rata enam benih)
  DL_FIRE_MODIS/patokan_modis_j_{kab,nas}.log               (MODIS fase J: prakiraan maju R, penyangga jeda V6)
  DL_FIRE_MODIS/patokan_modis_n_{kab,nas}.log               (MODIS fase N: GB atas EF-LD, bagian tingkat maju)
  DL_FIRE_MODIS/patokan_modis_o_{kab,nas}.log               (MODIS fase O: GB + DC atas EF-LD, prakiraan maju)
Pakai:    python gambar_inti_s.py
Keluaran: Gambar_Patokan/gambar_rev6_inti.png (300 dpi) dan .pdf
"""
import os
import re

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

BIRU, MERAH, HIJAU, UNGU, JINGGA = "#1F5FA8", "#C1442A", "#2E7D4F", "#7B4FA0", "#B7791F"
COKLAT = "#6B3E12"
KELABU, KELABU_MUDA = "#6B7280", "#B8BCC4"
TINTA, TINTA_2 = "#1F2328", "#5A6169"
plt.rcParams.update({
    "figure.dpi": 120, "savefig.dpi": int(os.environ.get("GAMBAR_DPI", 300)), "savefig.bbox": "tight",
    "font.family": "DejaVu Sans", "font.size": 9,
    "axes.edgecolor": KELABU_MUDA, "axes.linewidth": 0.8, "axes.labelcolor": TINTA,
    "axes.titlesize": 10, "axes.titleweight": "bold", "axes.titlecolor": TINTA,
    "xtick.color": TINTA_2, "ytick.color": TINTA_2, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "grid.color": "#E7E9EC", "grid.linewidth": 0.7, "legend.frameon": False, "legend.fontsize": 8,
})
BASE = os.path.dirname(os.path.abspath(__file__))
ANG = r"([+-]\d\.\d+)"
SEL = rf"\[{ANG}, {ANG}\]"


def baca(p):
    return open(os.path.join(BASE, p), encoding="utf-8").read()


NAMA_S = {"xgb": "Gradient boosting", "ef1_tingkat": "EF-L"}
ATAS_S = {"klim_frek1": "kalender", "klim_frek": "kalender", "rujukan_peluang": "EF-P", "ef1_tingkat": "EF-L",
          "efld": "EF-LD"}


def bss_viirs(log, acu, kol):
    """baris rinci S2 log fase S: nama, atas, BSS, [dua tahap], [tahun t]."""
    blok = log.split("S2. Tabel 4")[1].split("Tabel 4 ringkas")[0]
    m = re.search(rf"^  {re.escape(NAMA_S[kol])}\s+{ATAS_S[acu]}\s+{ANG}\s+{SEL}\s+{SEL}", blok, re.M)
    return tuple(float(x) for x in m.groups())


def bss_modis(s, varian, nama):
    if varian in ("R", "V5", "V6"):
        blok = MJ[s].split(f"\nJ {varian}  ")[1].split("\nJ ")[0]
    else:
        blok = M[s].split(f"I {varian}  ")[1].split("\nI V")[0]
    m = re.search(rf"^\s+{nama}\s+{ANG} 2tahap {SEL} t95 {SEL}", blok, re.M)
    return tuple(float(x) for x in m.groups())


def bagian(teks):
    m = re.search(r"bagian periode (-?\d\.\d+) .*?bagian tahun (-?\d\.\d+) .*?total (-?\d\.\d+)", teks)
    return tuple(float(x) for x in m.groups())


V = {s: open(os.path.join(BASE, f"DL_FIRE_NASIONAL/penuh/patokan_ambang_s_tabel_{s}.log"), encoding="cp1252").read()
     for s in ("kab", "nas")}
VP = {s: baca(f"DL_FIRE_NASIONAL/penuh/patokan_tingkat_p_{s}.log") for s in ("kab", "nas")}
M = {s: baca(f"DL_FIRE_MODIS/patokan_modis_i_{s}.log") for s in ("kab", "nas")}
MJ = {s: baca(f"DL_FIRE_MODIS/patokan_modis_j_{s}.log") for s in ("kab", "nas")}
MN = {s: baca(f"DL_FIRE_MODIS/patokan_modis_n_{s}.log") for s in ("kab", "nas")}
MO = {s: baca(f"DL_FIRE_MODIS/patokan_modis_o_{s}.log") for s in ("kab", "nas")}


def gbdc_efld(s, var):
    if var != "R":
        return None
    m = re.search(rf"^\s+gb_dc/efld\s+\(a\) {ANG} {SEL} t {SEL}", MO[s].split("O2 GB dengan DC")[1], re.M)
    return tuple(float(x) for x in m.groups())


def gb_efld(s, var):
    if var is None:
        return bss_viirs(V[s], "efld", "xgb")
    elif var == "R":
        m = re.search(rf"^\s+gb/efld\s+\(a\) {ANG} {SEL} t {SEL}", MN[s].split("N1 rujukan")[1], re.M)
    else:
        return None
    return tuple(float(x) for x in m.groups())


def bagian_maju(s):
    return bagian(MN[s].split("N2 bagian")[1].split("atas kalender rata-rata rasio")[1].split("\n")[0])
REF = {"kab": "klim_frek1", "nas": "klim_frek"}
BARIS = ["VIIRS 2020–2025,\nchronological\n(all features)", "MODIS 2008–2019,\nforward",
         "MODIS 2003–2011,\nall other years", "MODIS, training-year\nthresholds, lag buffer",
         "MODIS, 12 random\ntraining years", "MODIS, trained\nbefore 2016"]
VAR = [None, "R", "V0", "V6", "V3", "V1"]
PERB = [("GB (VIIRS) or GB-FH (MODIS) over raw calendar", BIRU, ("xgb", REF), "gb atas kalender"),
        ("EF-L over raw calendar", HIJAU, ("ef1_tingkat", REF), "efl atas kalender"),
        ("GB or GB-FH over EF-P", UNGU, ("xgb", "rujukan_peluang"), "gb atas efp"),
        ("GB or GB-FH over EF-L", MERAH, ("xgb", "ef1_tingkat"), "gb atas efl"),
        ("GB or GB-FH over EF-LD", JINGGA, None, None),
        ("GB-FH + DC over EF-LD (forward, added later)", COKLAT, "dc", None)]

fig = plt.figure(figsize=(7.2, 9.8))
gs = fig.add_gridspec(2, 2, height_ratios=[4.8, 1.9], hspace=0.3, wspace=0.08)
for k, (s, judul) in enumerate((("kab", "(a) Per-district threshold"), ("nas", "(b) National threshold"))):
    ax = fig.add_subplot(gs[0, k])
    for r, (lab, var) in enumerate(zip(BARIS, VAR)):
        for j, (nm, w, spek, mn) in enumerate(PERB):
            if spek in (None, "dc"):
                v = gb_efld(s, var) if spek is None else gbdc_efld(s, var)
                if v is None:
                    continue
                x, a, b, c, d = v
            elif var is None:
                kol, acu = spek
                acu_ = REF[s] if acu == REF else acu
                x, a, b, c, d = bss_viirs(V[s], acu_, kol)
            else:
                x, a, b, c, d = bss_modis(s, var, mn)
            y = r + (j - 2.5) * 0.14
            ax.plot([c, d], [y, y], color=w, lw=0.9, solid_capstyle="butt")
            ax.plot([a, b], [y, y], color=w, lw=3.2, solid_capstyle="butt", alpha=0.85)
            mk = "D" if spek == "dc" else "o"
            ax.plot([x], [y], mk, ms=4.2, mfc="white", mec=w, mew=1.3, zorder=3,
                    label=nm if (k == 0 and (r == 0 or spek == "dc")) else None)
    ax.axvline(0, color=KELABU, lw=0.8)
    ax.set_ylim(len(BARIS) - 0.5, -0.5)
    ax.set_yticks(range(len(BARIS)))
    ax.set_yticklabels(BARIS if k == 0 else [])
    ax.set_xlabel("Brier skill score")
    ax.grid(axis="x")
    ax.set_title(judul, loc="left")

axc = fig.add_subplot(gs[1, :])
BAG = [("VIIRS, per district", bagian(VP["kab"].split("P1 Pemecahan")[1].split("\n")[1])),
       ("VIIRS, national", bagian(VP["nas"].split("P1 Pemecahan")[1].split("\n")[1])),
       ("MODIS forward, per district", bagian_maju("kab")),
       ("MODIS forward, national", bagian_maju("nas")),
       ("MODIS all other years, per district", bagian(M["kab"].split("I V0  ")[1].split("I4 Pemecahan")[1].split("\n")[1])),
       ("MODIS all other years, national", bagian(M["nas"].split("I V0  ")[1].split("I4 Pemecahan")[1].split("\n")[1]))]
for r, (lab, (bp, bt, tot)) in enumerate(BAG):
    axc.barh(r, bp, color=KELABU, height=0.62, label="Period offset" if r == 0 else None)
    axc.barh(r, bt, left=bp, color=BIRU, alpha=0.75, height=0.62, label="Level of each year" if r == 0 else None)
    axc.barh(r, 1 - bp - bt, left=bp + bt, color="white", edgecolor=KELABU_MUDA, height=0.62,
             label="Remainder" if r == 0 else None)
    axc.text(1.01, r, f"{bp:.2f} + {bt:.2f}", va="center", fontsize=8, color=TINTA_2)
axc.set_yticks(range(len(BAG)))
axc.set_yticklabels([b[0] for b in BAG])
axc.set_ylim(len(BAG) - 0.5, -0.5)
axc.set_xlim(0, 1)
axc.set_xlabel("Share of GB or GB-FH Brier skill over the raw calendar")
axc.set_title("(c) Where the gain over the raw calendar comes from", loc="left")
axc.legend(loc="lower center", bbox_to_anchor=(0.5, -0.48), ncol=3)
h, l = fig.axes[0].get_legend_handles_labels()
fig.legend(h, l, loc="upper center", bbox_to_anchor=(0.5, 0.995), ncol=2)
fig.subplots_adjust(top=0.9)
OUT = os.path.join(BASE, "Gambar_Patokan")
for ext in ("png", "pdf"):
    fig.savefig(os.path.join(OUT, f"gambar_rev6_inti.{ext}"))
print("tersimpan: gambar_rev6_inti.png dan .pdf")
