# peatfire-eval

**PeatFireBench**: a district-month benchmark for fire forecasting in Indonesia (498 districts and
cities, 2012–2025), built from VIIRS active fire detections with the national peat map, GFWED fire
weather, ENSO and IOD indices, CHIRPS rainfall, land cover, forest loss, terrain and population,
with twelve baselines, three splits and two severity targets; and the evaluation-protocol
experiments behind it.

This repository accompanies the manuscript *Evaluation choices change which fire forecasting
model a district would choose: PeatFireBench, a 498-district benchmark for Indonesia* (Halim,
Lukman, Andika, Macnamara and Erwin, in review). Everything needed to rebuild the panel from the
included tables and to reproduce every number, table and figure of that paper is in the section
**PeatFireBench** below.

Version history: 2.0.0 added PeatFireBench; 2.1.0 added the peat layer, fire weather and the FWI
baselines (the 2.0.x results are reproduced with `PATOKAN_PUTARAN=lengkap`); 2.1.1 adds the
analyses requested in review (district-relative FWI baselines, seed sensitivity, per-year
calibration, peat checks, peat and non-peat districts); 2.2.0 adds the West Kalimantan
ranking-stability analyses described below; 2.3.0 adds the decision value of the warnings and
the FWI System components; 2.4.0 adds the revision analyses (matched reference climatologies,
probability references and five registered analysis rounds, section **Revision analyses** below).
Versions 1.0.0–1.3.0 hold earlier
analyses of West Kalimantan and of the five Kalimantan provinces. Those manuscripts were withdrawn
and their questions folded into PeatFireBench; the code is kept below, unchanged, because its
Zenodo versions are cited.

## Requirements

- Python 3.12 with the packages in `requirements.txt`
- Node.js 20 or later, for `bangun_panel.js` and `bangun_panel_nasional.js` only — no third-party packages
- Google Earth Engine (Code Editor) only to regenerate the covariate tables, which are included

The paper's results were produced with Python 3.12, pandas 3.0.5, NumPy 2.5.2, scikit-learn
1.9.0, XGBoost 3.4.1 and matplotlib 3.11.1. The PeatFireBench baselines take about 30 minutes on
a laptop, the ablation about an hour, and the seed sensitivity about four hours; the archived
analyses take under ten minutes. No GPU is used.

```bash
pip install -r requirements.txt
```

## PeatFireBench (version 2.1.1)

### Files

| Path | Content |
|---|---|
| `DL_FIRE_NASIONAL/panel_nasional.csv` | Raw district-month VIIRS S-NPP detection counts, 502 GADM 4.1 level-2 units, 2012-01 to 2025-12 (from `bangun_panel_nasional.js`) |
| `DL_FIRE_NASIONAL/chirps_bulanan_kabupaten_2012_2025.csv` | Monthly CHIRPS rainfall per district (from `Skrip_CHIRPS_Earth_Engine.js`) |
| `DL_FIRE_NASIONAL/statis_kabupaten.csv`, `tutupan_tahunan.csv`, `hilang_hutan_tahunan_v2.csv` | Terrain, population, MODIS land cover and Hansen forest loss per district (from `Skrip_Statis_Earth_Engine.js`) |
| `data/oni.ascii.txt`, `data/dmi.had.long.data` | ENSO (ONI) and Indian Ocean Dipole (DMI) indices, NOAA |
| `DL_FIRE_NASIONAL/gambut/spatial-metrics-indonesia-peat_area_kabupaten.csv` | Peat area per current (BPS) district, Peta Lahan Gambut Indonesia 1:50.000, Kementerian Pertanian, December 2019, as published by [Trase](https://trase.earth/open-data/datasets/spatial-metrics-indonesia-peat-area) (release 18 September 2026, CC BY 4.0) |
| `DL_FIRE_NASIONAL/gfwed_komponen_MERRA2.CORRECTED.csv` | Monthly Drought Code, Duff Moisture Code, Fine Fuel Moisture Code, Buildup Index and FWI per district from the GFWED v2.0 monthly files of NASA (bias-corrected MERRA-2 precipitation; from `gabung_gfwed_komponen.py`, which needs the 168 monthly files and the GADM 4.1 level-2 polygons) |
| `DL_FIRE_NASIONAL/fwi_bulanan_kabupaten_2012_2025_v2.csv` | Monthly mean and maximum Fire Weather Index per district from GFWED (MERRA-2), with sea cells filled (from `Skrip_FWI_Earth_Engine.js`) |

### Build the benchmark panel (seconds)

```bash
python gabung_indeks_nasional.py DL_FIRE_NASIONAL/panel_nasional.csv data/oni.ascii.txt \
    data/dmi.had.long.data DL_FIRE_NASIONAL/panel_nasional_fitur.csv
python bersihkan_panel_nasional.py      # removes 4 water bodies, adds both severity targets
python gabung_chirps_nasional.py        # rainfall features
python gabung_lahan_nasional.py         # land and population features -> panel_nasional_fitur_lengkap.csv
python gabung_gambut_nasional.py        # peat fraction (BPS -> GADM crosswalk) -> panel_nasional_fitur_gambut.csv
python gabung_fwi_nasional.py           # lagged fire weather -> panel_nasional_fitur_penuh.csv
```

The resulting `panel_nasional_fitur_lengkap.csv` (77,688 rows, 2013–2025) is byte-identical to
the one used in the paper, and `panel_nasional_fitur_penuh.csv` matches it to floating-point
rounding (largest difference 1.4e-14, in `fwi_anom_lag1`). Columns: `gid` (GADM identifier), `provinsi`, `kabupaten`, `tahun`,
`bulan`, `titik_panas` (detections), `y_kabupaten` (per-district target, primary), `y_gabungan`
(pooled target), `ada_positif` (district has at least one per-district positive), and the
feature groups listed in `kelompok_fitur.py` (`api` fire history, `musim` season, `iklim`
climate indices, `hujan` rainfall, `lahan` land, `manusia` population, `gambut` peat, `cuaca`
fire weather). `fwi_kini`, the same-month FWI, is a reference column, not a forecast feature.

### Run the baselines and reproduce the paper

```bash
python patokan_prediksi.py                         # 10 baselines x splits S1/S2/S3 x 2 targets (~30 min)
python patokan_tabel_naskah.py                     # Tables 2, 3 and island transfer (S2)
python patokan_ablasi.py && python patokan_ablasi_analisis.py   # Table 4 (feature groups)
python patokan_provinsi.py                         # Section 6.7, provinces
python patokan_uji_panel.py                        # headroom-normalised contrast, permutation null
python patokan_uji_tahun.py                        # extreme-year contrast on the defined S2 years, lag baselines, jackknife
python patokan_kepekaan_ambang.py                  # thresholds from training years only
python gambar_patokan.py                           # Figures 1-5
# added in 2.1.1, from the third review round (Supplementary S1-S7 of the paper)
python cek_gambut_provinsi.py                      # S1: peat crosswalk against provincial totals
python patokan_fwi_adil.py                         # Table 2 and S3: FWI percentile, climatology + FWI
python patokan_ece.py                              # S4: calibration per year with intervals
python patokan_benih.py && python patokan_benih_analisis.py   # S5: five more seeds, seed-averaged provincial ranking (~1 h + ~3 h)
python patokan_gambut_bagi.py                      # S7: districts with and without peat
# added in 2.3.0 (logs in logs/peatfirebench/)
python patokan_nilai.py                            # Section 6.8 and Table 6: hit rate, false alarm ratio, relative economic value
python patokan_fwi_komponen.py                     # Table 5: FWI System components (~1 h)
```

### Revision analyses (versions 2.4.0 to 2.11.1)

The revised manuscript, *IndoFireBench: a district benchmark and reference forecasts for monthly fire
forecasting in Indonesia* (the benchmark is named PeatFireBench in this
repository and its earlier versions), compares every model with a
reference climatology computed from the training years only, on the same rows: the mean count
(MC, the reference of version 2.3.0), the share of training years in which the district exceeded
its threshold in that calendar month (EF-1m) or in a three-month window (EF-3m), a probability
climatology chosen per fold on training years (EF-P), and EF-1m adjusted with last month's national
detection count (EF-L). The analyses ran in fifteen rounds (phases A to O; phase F in version 2.5.0, phase G in 2.6.0, phase H in 2.7.0, phases I to L in 2.8.0, phase M in 2.9.0, phase N in 2.10.0, phase O in 2.11.0 and 2.11.1). For each round the design
and the decision rules were written and committed before any script ran; `registration/` holds
those records with their results (in Indonesian, `Hasil_Patokan_6.md` to `Hasil_Patokan_20.md`);
`Hasil_Patokan_21.md` registers the evaluation of the 2026 season, with two dated addenda, and
`RIWAYAT_COMMIT.md` lists the commits of every record with their time stamps.
Phase G tests EF-L on the MODIS years 2003-2011, which no earlier round had scored: the MODIS
standard-processing archive (`MODIS_SP`) is downloaded with the same box and filter, and a panel is
built with the same district polygons. Phase H repeats the MODIS test trained only on years before 2016
and with thresholds from training years, adds EF-P, the level-shift diagnostic and Brier skill as one
ratio, and applies the rule that a forecast beats the references only if it beats all three. Phase I
uses six gradient-boosting seeds on MODIS, controls the length of the training record, splits the
level-shift diagnostic into a period offset and the year-to-year level, and reports the minimum
detectable difference. Phase J adds the only MODIS forecast (test years 2008-2019, each trained only on
the years before it), cross-validation with a one-year buffer after the test year, and peat districts on
VIIRS. Phase K averages the forward forecast over the years in several ways (mean of yearly ratios, one
ratio over all test rows, median year) and repeats it with stricter event floors; phase L does the same
for the VIIRS probability results. Phase M adds two drought forms of EF-L on VIIRS (EF-D with the
district's Drought Code of the month before in place of the national count, EF-LD with both), repeats the
two-stage intervals with five bootstrap seeds, 10,000 draws and a moving-block year bootstrap, and lists the
ENSO and dipole phase of every test year. Phase N scores EF-D and EF-LD on the forward MODIS forecast,
with the GFWED Drought Code of December 2002 to December 2011 aggregated to districts in the same way as
for 2012-2025 (`DL_FIRE_NASIONAL/gfwed_komponen_MERRA2.CORRECTED_awal.csv`, from
`gabung_gfwed_komponen.py --unduh MERRA2.CORRECTED --awal` and `--presip MERRA2.CORRECTED --awal`), splits
the forward forecast's level-shift share into period and year, and gives sign and signed-rank tests over
the forward years for both targets. Phase O refits gradient boosting with the district Drought Code anomaly that
EF-LD uses, on the chronological VIIRS test years (`patokan_tingkat_o.py`, added in 2.11.1) and on the
forward MODIS forecast (`patokan_modis_o.py`), and scores it over the four references with five bootstrap
seeds, 10,000 draws and a moving-block year bootstrap; `gambar_inti_o.py` draws Figure 3 of the revised
manuscript with that series. The 2026 evaluation (`Hasil_Patokan_21.md`) was registered on 9 November 2026,
before standard-processing 2026 detections were released, and is to be run once with this code. The MODIS
data are not included; the scripts below rebuild them.
The commit identifiers in them refer to the authors' working repository; header paragraphs that
named internal review files were shortened, and nothing else was changed. Logs are in
`logs/peatfirebench/revision/`.

```bash
# phase A: frequency climatologies, out-of-fold training predictions (Hasil_Patokan_6)
python patokan_revisi_prediksi.py                  # adds klim_frek (EF-3m), klim_frek1 (EF-1m) and split Q (~30 min)
python patokan_revisi_oof.py                       # out-of-fold predictions on training years (~1 h)
python patokan_revisi_analisis.py                  # A1-A5
python patokan_revisi_pascahoc.py                  # post-hoc checks, not registered
# phase B (Hasil_Patokan_7)
python patokan_revisi_tabel2.py                    # Table 2 against the matched references
python patokan_revisi_gabung.py                    # years pooled against the matched references
python patokan_revisi_jeda2.py                     # forecasts without last month's data
python patokan_revisi_kalibrasi.py                 # recalibration from earlier years
python patokan_revisi_fwi.py                       # ISI and the FWI System components
python gadm_ke_csv_wkt.py                          # GADM 4.1 polygons as WKT (needs the GADM file)
python patokan_revisi_provinsi.py                  # provinces, scattered and contiguous nulls (several hours)
python patokan_revisi_provinsi_biaya.py            # provinces, cross-validated transfer cost
# phase C (Hasil_Patokan_8)
python patokan_revisi_tahun.py                     # year-resampled intervals; extreme-year contrast
python patokan_revisi_rujukan.py                   # reference chosen inside each fold
python patokan_revisi_siaga.py                     # alert rules against EF-1m
python patokan_revisi_ambang.py                    # thresholds from training years only (Supplementary S6)
python patokan_revisi_sederhana.py                 # local history on EF-1m, fire-history tree
python patokan_revisi_pasca2015.py                 # recalibration on years after 2015
# phase D (Hasil_Patokan_9)
python patokan_revisi_setara.py                    # equivalence, minimum detectable difference, Holm
python patokan_revisi_proper.py                    # Brier skill and log score
python patokan_revisi_proper2.py                   # smoothed probability references, post hoc
python patokan_revisi_fdrs.py                      # DC and FFMC baselines; fire versus weather lags (Supplementary S3)
python patokan_revisi_gambut.py                    # peat strata and provinces (Supplementary S7)
python patokan_revisi_siaga2.py                    # monthly quota against threshold rules
python patokan_revisi_rezim.py                     # fire per unit drought after 2015
python patokan_revisi_setel.py                     # gradient boosting tuned inside folds (~1 h)
# phase E (Hasil_Patokan_10): one target per run, for memory
python -u patokan_revisi_peluang.py y_kabupaten > DL_FIRE_NASIONAL/penuh/patokan_revisi_peluang_kab.log
python -u patokan_revisi_peluang.py y_gabungan  > DL_FIRE_NASIONAL/penuh/patokan_revisi_peluang_nas.log
# phase F (Hasil_Patokan_11): EF-L as a ranking and as alert rules, tuned quota, gradient boosting with the
# national count; reads the phase E outputs; one target per run, for memory
python -u patokan_revisi_tingkat.py y_kabupaten > DL_FIRE_NASIONAL/penuh/patokan_revisi_tingkat_kab.log
python -u patokan_revisi_tingkat.py y_gabungan  > DL_FIRE_NASIONAL/penuh/patokan_revisi_tingkat_nas.log
# phase G (Hasil_Patokan_12): minimum detectable Brier skill over EF-L; probability scores with
# training-year thresholds; EF-L and fire-count forecasts on the MODIS years 2003-2011
python patokan_revisi_daya.py > DL_FIRE_NASIONAL/penuh/patokan_revisi_daya.log
python -u patokan_revisi_ambang_peluang.py y_kabupaten > DL_FIRE_NASIONAL/penuh/patokan_revisi_ambang_peluang_kab.log
python -u patokan_revisi_ambang_peluang.py y_gabungan  > DL_FIRE_NASIONAL/penuh/patokan_revisi_ambang_peluang_nas.log
python unduh_firms_modis.py                        # MODIS_SP 2003-2025 into DL_FIRE_MODIS/ (needs FIRMS_MAP_KEY; ~1 h)
python modis_keyakinan.py                          # MODIS confidence (0-100) to the l/n/h classes
node bangun_panel_nasional.js DL_FIRE_SV-C2_792597/gadm41_IDN_2.json DL_FIRE_MODIS/panel_modis.csv DL_FIRE_MODIS/kelas
python -u patokan_modis.py y_kabupaten > DL_FIRE_MODIS/patokan_modis_kab.log
python -u patokan_modis.py y_gabungan  > DL_FIRE_MODIS/patokan_modis_nas.log
# phase H (Hasil_Patokan_13): MODIS variants trained before 2016 and with training-year thresholds
python -u patokan_modis_h.py y_kabupaten > DL_FIRE_MODIS/patokan_modis_h_kab.log
python -u patokan_modis_h.py y_gabungan  > DL_FIRE_MODIS/patokan_modis_h_nas.log
python gambar_patokan.py                           # Figures 1-5 of the revised manuscript (gambar_rev*)
# phase I (Hasil_Patokan_14): six seeds, training length, level-shift split, minimum detectable difference
python -u patokan_modis_i.py y_kabupaten > DL_FIRE_MODIS/patokan_modis_i_kab.log
python -u patokan_modis_i.py y_gabungan  > DL_FIRE_MODIS/patokan_modis_i_nas.log
python -u patokan_tingkat_i.py y_kabupaten > DL_FIRE_NASIONAL/penuh/patokan_tingkat_i_kab.log
python -u patokan_tingkat_i.py y_gabungan  > DL_FIRE_NASIONAL/penuh/patokan_tingkat_i_nas.log
# phase J (Hasil_Patokan_15): forward MODIS forecast 2008-2019, buffered cross-validation, peat districts
python -u patokan_modis_j.py y_kabupaten > DL_FIRE_MODIS/patokan_modis_j_kab.log
python -u patokan_modis_j.py y_gabungan  > DL_FIRE_MODIS/patokan_modis_j_nas.log
python -u patokan_tingkat_j.py y_kabupaten > DL_FIRE_NASIONAL/penuh/patokan_tingkat_j_kab.log
python -u patokan_tingkat_j.py y_gabungan  > DL_FIRE_NASIONAL/penuh/patokan_tingkat_j_nas.log
python gambar_inti_i.py                            # Figure 3 of the revised manuscript (gambar_rev6_inti)
# phase K (Hasil_Patokan_16): averaging the forward forecast over years; event floors 3 and 5
python -u patokan_modis_k.py y_kabupaten   > DL_FIRE_MODIS/patokan_modis_k_kab.log
python -u patokan_modis_k.py y_gabungan    > DL_FIRE_MODIS/patokan_modis_k_nas.log
python -u patokan_modis_k.py y_kabupaten 3 > DL_FIRE_MODIS/patokan_modis_k_kab_l3.log
python -u patokan_modis_k.py y_kabupaten 5 > DL_FIRE_MODIS/patokan_modis_k_kab_l5.log
# phase L (Hasil_Patokan_17): averaging the VIIRS probability results over years
python -u patokan_tingkat_l.py y_kabupaten > DL_FIRE_NASIONAL/penuh/patokan_tingkat_l_kab.log
python -u patokan_tingkat_l.py y_gabungan  > DL_FIRE_NASIONAL/penuh/patokan_tingkat_l_nas.log
# phase M (Hasil_Patokan_18): drought forms of EF-L, bootstrap Monte Carlo error, block bootstrap, ENSO phase
python -u patokan_tingkat_m.py y_kabupaten > DL_FIRE_NASIONAL/penuh/patokan_tingkat_m_kab.log
python -u patokan_tingkat_m.py y_gabungan  > DL_FIRE_NASIONAL/penuh/patokan_tingkat_m_nas.log
python -u patokan_modis_m.py y_gabungan    > DL_FIRE_MODIS/patokan_modis_m_nas.log
python -u patokan_enso_m.py > DL_FIRE_NASIONAL/penuh/patokan_enso_m.log    # reads the phase K and L logs
# phase N (Hasil_Patokan_19): drought forms and level share on the forward MODIS forecast
python -u patokan_modis_n.py y_gabungan    > DL_FIRE_MODIS/patokan_modis_n_nas.log
python -u patokan_modis_n.py y_kabupaten   > DL_FIRE_MODIS/patokan_modis_n_kab.log
# phase O (Hasil_Patokan_20): gradient boosting given the district Drought Code, VIIRS and forward MODIS
python -u patokan_tingkat_o.py y_kabupaten > DL_FIRE_NASIONAL/penuh/patokan_tingkat_o_kab.log
python -u patokan_tingkat_o.py y_gabungan  > DL_FIRE_NASIONAL/penuh/patokan_tingkat_o_nas.log
python -u patokan_modis_o.py y_gabungan    > DL_FIRE_MODIS/patokan_modis_o_nas.log
python -u patokan_modis_o.py y_kabupaten   > DL_FIRE_MODIS/patokan_modis_o_kab.log
python gambar_inti_o.py                            # Figure 3 of the revised manuscript, with phases M to O
```

The phase scripts read `patokan_revisi_prediksi.csv` and `patokan_revisi_oof.csv` from phase A,
and some read `patokan_benih.csv`, `patokan_fwi_adil.csv` and `patokan_fwi_komponen.csv` from the
2.1.1 and 2.3.0 scripts above, so phase A and those scripts run first.

Splits: **S1** chronological (test 2019–2025, train on earlier years); **S2** extreme year
withheld (2015, 2014, 2019); **S3** island transfer (train without the test island group).
All scripts read the run from `PATOKAN_PUTARAN` (`penuh`, the default, or `lengkap` for the
2.0.x results; see `kelompok_fitur.py`).
Score a new model by writing its predictions in the same layout as `patokan_prediksi_penuh.csv`
(one row per test district-month, one column per model) and passing it to
`patokan_tabel_naskah.py` with `PATOKAN_BERKAS=<file>`.

Raw FIRMS detections and GADM polygons are not redistributed: `unduh_firms_nasional.py`
downloads the detections (set `FIRMS_MAP_KEY` in the environment), and GADM 4.1 is free from
gadm.org. The Earth Engine scripts contain a placeholder project path to replace with your own.

## West Kalimantan ranking stability (version 2.2.0)

These scripts accompany the manuscript *How Stable Are Fire Prediction Model Rankings? Bootstrap
Evidence From West Kalimantan* (Halim, Lukman, Andika, Macnamara and Erwin, in review). They reuse
the West Kalimantan panel `data/panel_bulanan.csv`, the seven models and the rolling-origin folds
(2019–2025) of `percobaan.py`, with the ONI timing of version 1.3.0, and resample the test set by
test year. Logs of the runs reported in the manuscript are in `logs/jepin/`.

| Result | Script |
|---|---|
| Discarding non-fire months versus a natural-prevalence test set of the same size | `percobaan7_kontrol_ukuran.py` |
| Rank probabilities, record-length and balancing curves (district-month bootstrap), ten seeds, per-year skill, training-year threshold; Fig. 4 | `percobaan8_jepin.py` |
| Exact enumeration of all 1,716 resamples of the seven test years; leave-one-year-out; training-year threshold | `percobaan9_jepin_tahun.py` |
| Record length projected by resampling whole years | `percobaan10_proyeksi_tahun.py` |
| Paired difference interval, ROC-AUC versus AUC-PR stability, eleven seeds, discarding non-fire months with all severe months kept, record length 2–28 years | `percobaan11_jepin_v4.py` |
| Enumerated year-clustered intervals for Table II | `percobaan12_selang_eksak.py` |
| Exact sign and sign-flip tests on the seven per-year differences | `uji_tanda_jepin.py` |
| Figs. 2 and 3 (from the logs of percobaan 9 and 11) | `gambar_jepin.py` |

```bash
python percobaan7_kontrol_ukuran.py data/panel_bulanan.csv data/oni.ascii.txt
python percobaan8_jepin.py data/panel_bulanan.csv data/oni.ascii.txt
python percobaan9_jepin_tahun.py data/panel_bulanan.csv data/oni.ascii.txt
python percobaan10_proyeksi_tahun.py data/panel_bulanan.csv data/oni.ascii.txt
python percobaan11_jepin_v4.py data/panel_bulanan.csv data/oni.ascii.txt
python percobaan12_selang_eksak.py data/panel_bulanan.csv data/oni.ascii.txt
python uji_tanda_jepin.py      # reads logs/jepin/percobaan5_ulasan.log
python gambar_jepin.py         # reads logs/jepin/percobaan9_* and percobaan11_*; writes Gambar_JEPIN/
```

Fig. 1 of that manuscript is Fig. 1 of the archived analysis (`gambar.py`). The 1,716-resample
enumerations take a few minutes each; `percobaan11_jepin_v4.py`, which refits gradient boosting with
ten seeds, takes about ten minutes. All randomness uses fixed seeds, so the logs are reproduced
exactly.

## Archived: West Kalimantan and Kalimantan analyses (versions 1.x)

### What those analyses claimed, and which script produces it

| Claim | Script |
|---|---|
| A bounding box instead of administrative boundaries admits 50.5% of detections from outside the province and changes a reported ENSO-fire correlation | `bangun_panel.js` |
| ROC-AUC and AUC-PR rank identical predictions differently; balancing adds variance without reordering | `percobaan4_matriks2x2.py` |
| Calibration metrics with the calibrator fitted on out-of-fold predictions (Table 7) | `percobaan.py` |
| Neither protocol separates the two leading models: bootstrap intervals, per-year intervals, out-of-fold calibration against in-sample calibration | `percobaan5_ulasan.py` |
| A per-district severity threshold changes the leading model and shrinks the metric effect (Section 5.6, Table 9) | `percobaan6_kabupaten.py` |
| Reversals persist at the 80th, 90th, and 95th percentile thresholds | `percobaan3_dmi_ambang.py` |
| The Indian Ocean Dipole carries no forecastable annual signal and degrades the monthly model | `percobaan3_dmi_ambang.py` |
| Figures 2 and 3 (West Kalimantan) | `gambar.py` |
| **Five provinces:** the 2x2 design across Kalimantan under pooled and per-district thresholds; per-province rankings | `percobaan_kalimantan.py` |
| **Five provinces:** learned models trained without the test province, and on the province alone | `percobaan_kalimantan_transfer.py`, `percobaan_kalimantan3.py` |
| **Five provinces:** province identity and district mean count as features (mechanism test) | `percobaan_kalimantan4.py` |
| **Five provinces:** all predictions saved once; bootstrap intervals for the mechanism test | `kalimantan_prediksi.py`, `kalimantan_selang.py` |
| **Five provinces:** Figures 1, 4 and 5 | `gambar_kalimantan.py` |
| Thresholds computed from training years only (sensitivity analysis in the Limitations) | `kalimantan_kepekaan_ambang.py`, `kalimantan_banding_kepekaan.py`, `kalbar_kepekaan_ambang.py` |
| Every reference in the manuscript resolves on Crossref or arXiv | `verifikasi_rujukan.py` |

### Data (archived analyses)

Three of the four inputs are included. The two that are not are excluded for licensing
reasons, not for convenience, and both are free to obtain.

| Input | Included | Source |
|---|---|---|
| Derived district-month panel, West Kalimantan, 2,464 rows | **yes** — `data/panel_bulanan.csv` | produced by `bangun_panel.js` |
| Derived district-month panel, five Kalimantan provinces, 9,240 rows (2012–2025) | **yes** — `data/panel_kalimantan.csv` | Kalimantan rows of the national panel produced by `bangun_panel_nasional.js` |
| Oceanic Niño Index | **yes** — `data/oni.ascii.txt` | [NOAA Climate Prediction Center](https://origin.cpc.ncep.noaa.gov/products/analysis_monitoring/ensostuff/ONI_v5.php), US Government work, public domain |
| Dipole Mode Index, HadISST-based | **yes** — `data/dmi.had.long.data` | [NOAA Physical Sciences Laboratory](https://psl.noaa.gov/gcos_wgsp/Timeseries/DMI/), US Government work, public domain |
| VIIRS S-NPP active fire detections, 2012-2026 | no | [NASA FIRMS](https://firms.modaps.eosdis.nasa.gov/download/) — FIRMS terms require retrieval from source |
| District boundaries, GADM 4.1 level 2, Indonesia | no | [GADM](https://gadm.org/download_country.html) — licence permits academic use but not redistribution |

### Rebuilding the panel from raw detections

The included panel is the output of this step, so it can be skipped unless you want to verify
the construction itself.

1. Request a VIIRS S-NPP 375 m Collection 2 archive download from NASA FIRMS covering
   longitude 108.8 to 114.3 and latitude −3.3 to 2.1, January 2012 to the present. Place the
   CSV files in a directory of your choice.
2. Download `gadm41_IDN_2.json` from GADM.
3. Run:

```bash
node bangun_panel.js gadm41_IDN_2.json data/panel_bulanan.csv fire_archive_*.csv
```

The filters are fixed and must not be changed if you intend to compare results with the
paper: `confidence != l`, `type == 0`, and the detection must fall inside a West Kalimantan
district polygon. On the archive used in the paper this reads 833,208 detections, drops 31,664
low-confidence and 785 non-vegetation records, discards 404,060 that fall outside the province,
and retains 396,699.

### Correction in version 1.3.0: ONI timing

Versions 1.0.0 to 1.2.0 assigned each three-month Oceanic Niño Index season (for example DJF) to
its central month. The lag-1 ONI value used to forecast month *t* therefore covered months *t*−2 to
*t*, including the target month, and was not available at forecast time. Since 1.3.0 each season is
assigned to its last month (DJF to February, NDJ to January of the following year), so that lag 1 ends
in month *t*−1 (`muat` in `percobaan.py`). Every result changes slightly; results in any document
built on 1.0.0–1.2.0 should be replaced by those from 1.3.0. The derived panels in `data/` are
unaffected, because ONI lags are computed when the panel is loaded.

### Reproducing the archived results

```bash
# Experiments 1 and 2: the two protocols, and the extreme-year holdout
python percobaan.py data/panel_bulanan.csv data/oni.ascii.txt

# Experiment 3: severity-threshold sensitivity, and the Indian Ocean Dipole
python percobaan3_dmi_ambang.py data/panel_bulanan.csv data/oni.ascii.txt \
    data/dmi.had.long.data

# Experiment 4: the 2x2 design separating test distribution from metric
python percobaan4_matriks2x2.py data/panel_bulanan.csv data/oni.ascii.txt

# Experiment 5: bootstrap and per-year intervals, calibration procedures
python percobaan5_ulasan.py data/panel_bulanan.csv data/oni.ascii.txt

# Experiment 6: within-district metrics and per-district severity threshold
python percobaan6_kabupaten.py data/panel_bulanan.csv data/oni.ascii.txt

# Figures 1 to 5 of the West Kalimantan analysis
python gambar.py data/panel_bulanan.csv data/oni.ascii.txt
```

### Five provinces (version 1.2.0)

```bash
# The 2x2 design and per-province rankings (writes data/panel_kalimantan.csv again, unchanged)
python percobaan_kalimantan.py data/panel_kalimantan.csv data/oni.ascii.txt

# Transfer between provinces, training scope, and the mechanism test
python percobaan_kalimantan_transfer.py data/panel_kalimantan.csv data/oni.ascii.txt
python percobaan_kalimantan3.py data/panel_kalimantan.csv data/oni.ascii.txt
python percobaan_kalimantan4.py data/panel_kalimantan.csv data/oni.ascii.txt

# All predictions once (about 15 minutes), bootstrap intervals, and figures
python kalimantan_prediksi.py data/panel_kalimantan.csv data/oni.ascii.txt
python kalimantan_selang.py data/prediksi_kalimantan.csv
python gambar_kalimantan.py data/prediksi_kalimantan.csv data/panel_kalimantan.csv gadm41_IDN_2.json

# Decision-level analyses and robustness checks (version 1.3.0): regret of choosing a model from
# another province, warning-budget hit rates, prevalence matching, province-identity decomposition,
# training scope, fair baselines, district bootstrap of the regret, temporal split
python kalimantan_keputusan.py data/prediksi_kalimantan.csv
python kalimantan_uji_tajam.py data/panel_kalimantan.csv data/oni.ascii.txt

# Sensitivity: severity thresholds computed from each fold's training years only
python kalimantan_kepekaan_ambang.py data/panel_kalimantan.csv data/oni.ascii.txt
python kalimantan_banding_kepekaan.py data
python kalbar_kepekaan_ambang.py data/panel_bulanan.csv data/oni.ascii.txt
```

To rebuild the national panel from raw detections, request a VIIRS S-NPP 375 m Collection 2
archive for Indonesia from NASA FIRMS and run
`node bangun_panel_nasional.js gadm41_IDN_2.json panel_nasional.csv <directory of CSV files>`;
the Kalimantan panel is its rows whose province starts with "Kalimantan", 2012–2025.

Every experiment seeds its random number generator explicitly, so the balancing draws are
reproducible. Since version 1.1.0 isotonic calibration is fitted on out-of-fold predictions
within the training years (`kalibrator_oof` in `percobaan.py`), never on test data or on
in-sample training scores; version 1.0.0 fitted it on in-sample training scores.

## Reference checking

`verifikasi_rujukan.py` reads the manuscript's reference list, resolves each entry against
Crossref or the arXiv API, and compares title, journal, and year. It exits non-zero if any
entry fails, so it can be used as a pre-submission gate. Three citation errors were found this
way during preparation: two incorrect DOIs and one incorrect arXiv identifier.

## Notes on scope

This is benchmark and analysis code. It is not a fire warning system and should not be used as
one; the models are poorly calibrated in withheld extreme years. Hotspot counts are a proxy for fire activity, not a
measurement of burned area.

The code comments are in Indonesian; identifiers, filenames, and this README are in English.

## Licence

MIT — see `LICENSE`. The included NOAA index files are US Government works in the public
domain. The PeatFireBench panel and covariate tables are released under CC BY 4.0, because they
derive from CC BY 4.0 sources (Hansen Global Forest Change, WorldPop, Trase); please cite the
sources listed in the paper's dataset card. The West Kalimantan and Kalimantan panels of versions
1.x remain under CC0.

## Citation

See `CITATION.cff`, or cite the Zenodo record for the version you used.
