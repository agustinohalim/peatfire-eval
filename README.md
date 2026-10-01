# peatfire-eval

**Version 2.0.0 adds PeatFireBench**, a district-month benchmark for fire forecasting in all of
Indonesia (498 districts and cities, 2012–2025). See the section *PeatFireBench* below; the
West Kalimantan and Kalimantan analyses of earlier versions are kept unchanged.

Panel construction and evaluation-protocol experiments for district-month fire severity
forecasting in Indonesian Kalimantan: West Kalimantan on its own, and all five provinces.

This repository accompanies the manuscript *Evaluation choices decide fire-prediction model
rankings and whether they transfer between provinces: evidence from Indonesian Kalimantan*.
Versions 1.0.0 and 1.1.0 covered the West Kalimantan analysis only, then prepared as a separate
manuscript; version 1.2.0 adds the five-province analysis with which it was merged. The
repository contains everything needed to rebuild the derived panels from raw satellite
detections and to reproduce every number, table, and figure in the paper.

## What the paper claims, and which script produces it

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

## Requirements

- Python 3.12 with the packages in `requirements.txt`
- Node.js 20 or later, for `bangun_panel.js` only — it uses no third-party packages

The full pipeline completes in under ten minutes on a laptop. No GPU is used.

```bash
pip install -r requirements.txt
```

## Data

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

## PeatFireBench (version 2.0.0)

### Files

| Path | Content |
|---|---|
| `DL_FIRE_NASIONAL/panel_nasional.csv` | Raw district-month VIIRS S-NPP detection counts, 502 GADM 4.1 level-2 units, 2012-01 to 2025-12 (from `bangun_panel_nasional.js`) |
| `DL_FIRE_NASIONAL/chirps_bulanan_kabupaten_2012_2025.csv` | Monthly CHIRPS rainfall per district (from `Skrip_CHIRPS_Earth_Engine.js`) |
| `DL_FIRE_NASIONAL/statis_kabupaten.csv`, `tutupan_tahunan.csv`, `hilang_hutan_tahunan_v2.csv` | Terrain, population, MODIS land cover and Hansen forest loss per district (from `Skrip_Statis_Earth_Engine.js`) |
| `data/oni.ascii.txt`, `data/dmi.had.long.data` | ENSO (ONI) and Indian Ocean Dipole (DMI) indices, NOAA |

### Build the benchmark panel (seconds)

```bash
python gabung_indeks_nasional.py DL_FIRE_NASIONAL/panel_nasional.csv data/oni.ascii.txt \
    data/dmi.had.long.data DL_FIRE_NASIONAL/panel_nasional_fitur.csv
python bersihkan_panel_nasional.py      # removes 4 water bodies, adds both severity targets
python gabung_chirps_nasional.py        # rainfall features
python gabung_lahan_nasional.py         # land and population features -> panel_nasional_fitur_lengkap.csv
```

The resulting `panel_nasional_fitur_lengkap.csv` (77,688 rows, 2013–2025) is byte-identical to
the one used in the paper. Columns: `gid` (GADM identifier), `provinsi`, `kabupaten`, `tahun`,
`bulan`, `titik_panas` (detections), `y_kabupaten` (per-district target, primary), `y_gabungan`
(pooled target), `ada_positif` (district has at least one per-district positive), and the
feature groups listed in `kelompok_fitur.py` (`api` fire history, `musim` season, `iklim`
climate indices, `hujan` rainfall, `lahan` land, `manusia` population).

### Run the baselines and reproduce the paper

```bash
PATOKAN_FITUR=lengkap python patokan_prediksi.py   # 8 baselines x splits S1/S2/S3 x 2 targets (~15 min)
python patokan_tabel_naskah.py                     # Tables 2, 3, 5
python patokan_ablasi.py && python patokan_ablasi_analisis.py   # Table 4 (feature groups)
python patokan_provinsi.py                         # Section 6.7, provinces
python patokan_uji_panel.py                        # headroom-normalised contrast, permutation null
python patokan_kepekaan_ambang.py                  # thresholds from training years only
python gambar_patokan.py                           # Figures 1-5
```

Splits: **S1** chronological (test 2019–2025, train on earlier years); **S2** extreme year
withheld (2015, 2014, 2019); **S3** island transfer (train without the test island group).
Score a new model by writing its predictions in the same layout as `patokan_prediksi_lengkap.csv`
(one row per test district-month, one column per model) and passing it to
`patokan_tabel_naskah.py` with `PATOKAN_BERKAS=<file>`.

Raw FIRMS detections and GADM polygons are not redistributed: `unduh_firms_nasional.py`
downloads the detections (set `FIRMS_MAP_KEY` in the environment), and GADM 4.1 is free from
gadm.org. The Earth Engine scripts contain a placeholder project path to replace with your own.

## Correction in version 1.3.0: ONI timing

Versions 1.0.0 to 1.2.0 assigned each three-month Oceanic Niño Index season (for example DJF) to
its central month. The lag-1 ONI value used to forecast month *t* therefore covered months *t*−2 to
*t*, including the target month, and was not available at forecast time. Since 1.3.0 each season is
assigned to its last month (DJF to February, NDJ to January of the following year), so that lag 1 ends
in month *t*−1 (`muat` in `percobaan.py`). Every result changes slightly; results in any document
built on 1.0.0–1.2.0 should be replaced by those from 1.3.0. The derived panels in `data/` are
unaffected, because ONI lags are computed when the panel is loaded.

## Reproducing the results

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

This is analysis code for five provinces and thirteen complete years. It is not a fire warning
system and should not be used as one. Hotspot counts are a proxy for fire activity, not a
measurement of burned area.

The code comments are in Indonesian; identifiers, filenames, and this README are in English.

## Licence

MIT — see `LICENSE`. The included NOAA index files are US Government works in the public
domain. The derived panel is released under CC0.

## Citation

See `CITATION.cff`, or cite the Zenodo record for the version you used.
