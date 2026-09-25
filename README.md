# peatfire-eval

Panel construction and evaluation-protocol experiments for district-month fire severity
forecasting in West Kalimantan, Indonesia.

This repository accompanies the manuscript *Evaluation protocol choices determine model
rankings in fire hotspot prediction: evidence from West Kalimantan, Indonesia*. It contains
everything needed to rebuild the derived panel from raw satellite detections and to reproduce
every number, table, and figure in the paper.

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
| All five figures | `gambar.py` |
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
| Derived district-month panel, 2,464 rows | **yes** — `data/panel_bulanan.csv` | produced by `bangun_panel.js` |
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

# Figures 1 to 5
python gambar.py data/panel_bulanan.csv data/oni.ascii.txt
```

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

This is analysis code for one province and thirteen complete years. It is not a fire warning
system and should not be used as one. Hotspot counts are a proxy for fire activity, not a
measurement of burned area.

The code comments are in Indonesian; identifiers, filenames, and this README are in English.

## Licence

MIT — see `LICENSE`. The included NOAA index files are US Government works in the public
domain. The derived panel is released under CC0.

## Citation

See `CITATION.cff`, or cite the Zenodo record for the version you used.
