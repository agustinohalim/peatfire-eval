# Hasil Tolok Ukur Nasional 7 — fase B revisi PeatFireBench (provinsi, kalibrasi ulang, jeda dua bulan, FWI System)

**6 November 2026.** Lanjutan `Hasil_Patokan_6.md` (fase A), revisi PeatFireBench.

## Rancangan fase B (ditulis 6 Nov 2026, SEBELUM skrip mana pun dijalankan)

**Rujukan klimatologi** untuk semua bagian di bawah, ditetapkan dari fase A: yang terkuat per sasaran —
`klim_frek1` (frekuensi bulan kalender itu saja) untuk per kabupaten, `klim_frek` (jendela tiga bulan)
untuk nasional. Disebut `rujukan` di bawah. Pilihan ini sendiri berasal dari data uji (pasca-hoc fase A);
itu dicatat di naskah.

### B1. Provinsi (Temuan 5) — `patokan_revisi_provinsi.py`

S1 2020–2025, prediksi dilatih nasional. Random forest, gradient boosting, MLP dirata-rata atas enam
benih (42 dari `patokan_revisi_prediksi.csv`, 1–5 dari `patokan_benih.csv`). Model yang diperingkat:
`rujukan`, persistence, `log_klim_jeda`, logistik (FWI), rf, xgb, mlp. Provinsi dinilai bila ≥ 40
positif 2020–2025 untuk sasaran itu (sama dengan naskah).

- Statistik D: bagian pasangan provinsi yang juaranya (AUC-PR dalam-tahun tertinggi) berbeda.
- Statistik H: simpangan baku antarprovinsi dari keunggulan dalam-tahun rf − `rujukan` (interaksi
  model × provinsi langsung, bukan lewat juara).
- Dua nol, masing-masing 200 permutasi, provinsi palsu berukuran sama (banyak kabupaten) dan harus juga
  ≥ 40 positif (ditumbuhkan ulang sampai 50 kali bila tidak): (a) acak tidak bersebelahan (seperti naskah);
  (b) **bersebelahan**: wilayah ditumbuhkan dari kabupaten benih acak lewat graf ketetanggaan GADM 4.1
  level 2 (bertetangga = berbagi ≥ 2 titik batas, dibulatkan 10⁻⁴ derajat), tetangga ditambahkan acak.
  P = bagian permutasi dengan statistik ≥ teramati.
- Biaya transfer tervalidasi silang: juara dipilih pada tahun genap (2020, 2022, 2024), dinilai pada
  tahun ganjil (2021, 2023, 2025), dan sebaliknya, dirata-rata. Penyesalan memakai juara provinsi lain
  (rerata pasangan berurutan) dan penyesalan memakai juara nasional (dipilih pada paruh yang sama),
  dengan selang 95 % dari 200 resampel kabupaten di dalam tiap provinsi (seluruh prosedur diulang).
- **R6.** Temuan 5 tetap di badan naskah bila, untuk sasaran itu, D teramati > persentil 95 nol
  bersebelahan **dan** batas bawah selang penyesalan juara provinsi lain (tervalidasi silang) > 0.
  Lolos pada dua sasaran: temuan utama. Satu sasaran: dilaporkan dengan batasan itu. Tidak ada:
  dipindah ke suplemen dengan satu kalimat di badan naskah.

### B2. Kalibrasi ulang — `patokan_revisi_kalibrasi.py`

Gradient boosting, sasaran per kabupaten. Untuk tiap lipatan luar (S1 2020–2025, S2 2014, 2015, 2019),
pemeta kalibrasi dilatih pada prakiraan luar-lipatan tahun latih (`patokan_revisi_oof.csv`, semua tahun
dalam digabung) lalu diterapkan pada tahun uji: (i) Platt (logistik pada logit p), (ii) isotonik,
(iii) Platt + ONI dan DMI jeda satu bulan (logistik pada logit p, `oni_lag1`, `dmi_lag1`). Dilaporkan
per kelompok tahun (biasa, ekstrem), rata-rata tahun: ECE 10 bin lebar sama, Brier, nilai ekonomi
relatif "siaga bila p > a" pada a ∈ {0,05; 0,1; 0,2; 0,3; 0,5}, dan selisihnya dari amplop.

- **R7.** Kalimat "kalibrasi ulang pada tahun lampau memperbaiki peluang" boleh ditulis bila, untuk
  varian terbaik, ECE rata-rata per tahun turun di **kedua** kelompok dan selisih ke amplop mengecil
  pada ≥ 3 dari 5 a di kedua kelompok. Bila tidak, naskah menulis bahwa kalibrasi dari tahun lampau
  tidak cukup (salah kalibrasi bergeser antartahun) — dan itu dilaporkan sebagai hasil.

### B3. Jeda dua bulan — `patokan_revisi_jeda2.py`

Random forest dan gradient boosting dilatih ulang (parameter sama, benih 42) tanpa satu pun fitur jeda
satu bulan: `tp_lag1`, `oni_lag1`, `dmi_lag1`, `hujan_lag1`, `hujan_anom_lag1`, `fwi_lag1`,
`fwi_maks_lag1`, `fwi_anom_lag1` dibuang; fitur jeda ≥ 2, musim, dan statis tetap. S1 2019–2025 dan S2,
kedua sasaran. Dilaporkan: AUC-PR dalam-tahun S1 2020–2025, keunggulan atas `rujukan` dan rugi
terhadap model jeda satu bulan [95 %, 1.000], S2 per tahun atas `rujukan` [95 %, 500]. Tanpa aturan
keputusan; naskah menulis apa adanya berapa yang tersisa bila data bulan lalu belum tersedia.

### B4. FWI System — `patokan_revisi_fwi.py`

- ISI diambil dari berkas GFWED bulanan yang sama (`MERRA2.CORRECTED_ISI`, agregasi
  `gabung_gfwed_komponen.py` apa adanya) → `DL_FIRE_NASIONAL/gfwed_isi_MERRA2.CORRECTED.csv`, lalu
  `klim_isi` dengan rancangan `patokan_fwi_komponen.py` yang sama (persentil dan anomali kabupaten,
  ditambah log klimatologi dan musim; pembanding `klim_saja`). Dilaporkan seperti Hasil_Patokan_5 bagian 2.
- Tampilan gabungan antartahun untuk kelima indeks dan ISI: AUC-PR S1 2020–2025 dihitung atas semua
  tahun sekaligus, selisih terhadap `klim_saja` [95 %, bootstrap kabupaten 1.000], kedua sasaran.
  Untuk lima indeks lama dipakai `patokan_fwi_komponen.csv` apa adanya.
- Juga: rerata DC bulan lalu September 2015 di Kalimantan Tengah dan Sumatera Selatan lawan rerata
  September 2016–2025 di provinsi yang sama (apakah DC reanalisis menangkap kemarau 2015).
- Tanpa aturan keputusan; kalimat FWI di naskah mengikuti hasil, dipersempit ke "dalam-tahun, jeda satu
  bulan, rerata bulanan reanalisis" bila DC tetap tidak menambah.

### B5. Temuan 2 terhadap rujukan terkuat — `patokan_revisi_gabung.py`

Ditambahkan 6 Nov 2026 sesudah B1–B4 mulai berjalan, sebelum skrip ini dijalankan (hasil B1–B4 yang
sudah terbaca saat itu: kalibrasi B2 dan daftar provinsi B1; tidak menyangkut ukuran gabungan). Alasan:
angka Temuan 2 di naskah terkirim (+0,311 gabungan atas klimatologi rata-rata) memakai rujukan lemah.
S1 2020–2025, kedua sasaran: AUC-PR dalam-tahun dan gabungan untuk semua model; keunggulan gabungan atas
`rujukan` [95 %, bootstrap kabupaten 1.000] dan keunggulan dalam-tahun per tahun. Tanpa aturan
keputusan; Temuan 2 ditulis dengan angka ini apa adanya, termasuk bila keunggulan gabungan tidak lagi
melebihi semua tahun tunggal.

### B6. Tabel 2 lengkap — `patokan_revisi_tabel2.py`

Ditambahkan 6 Nov 2026 sesudah B2–B5 terbaca, sebelum skrip ini dijalankan. Alasan: A1 tidak memuat
selisih atas rujukan untuk *seasonal naive*, *ratio scaling*, FWI persentil kabupaten, dan klimatologi +
FWI. Semua baseline, S1 2020–2025, kedua sasaran: AUC-PR dalam-tahun, keunggulan atas `rujukan`
[95 %, 1.000, tarikan sama untuk semua tahun], AUC-PR gabungan. Pelaporan saja, tanpa aturan keputusan.

Selang: bootstrap kabupaten, seed 2026, seperti fase A.

## Hasil (dijalankan 6 Nov 2026; komit rancangan `a85b34e`, B5 dan B6 ditambahkan dan dikomit sebelum dijalankan)

Log: `DL_FIRE_NASIONAL/penuh/patokan_revisi_{provinsi,kalibrasi,jeda2,fwi,gabung,tabel2}.log`. Cek: FWI hasil
ekstraksi ISI sama persis dengan berkas komponen (selisih maks 0); `klim_saja` dihitung ulang sama dengan
berkas lama (1,1 × 10⁻¹⁶). Penyimpangan dari rancangan: tidak ada. Graf ketetanggaan: 502 satuan GADM
(termasuk empat badan air yang tidak ada di panel), 945 sisi, 29 tanpa tetangga (pulau).

### B6. Tabel 2 terhadap rujukan terkuat

S1 2020–2025. AUC-PR dalam-tahun, keunggulan atas rujukan [95 %, 1.000], AUC-PR gabungan.

| Model | Per kab (411; 1.190 positif; 4,02 %) | Unggul atas `klim_frek1` | Gabungan | Nasional (498; 2.197; 6,13 %) | Unggul atas `klim_frek` | Gabungan |
|---|---|---|---|---|---|---|
| Klimatologi rata-rata | 0,125 | −0,119 [−0,146, −0,098] | 0,116 | 0,479 | −0,209 [−0,240, −0,169] | 0,462 |
| `klim_frek` (jendela) | 0,193 | −0,052 [−0,073, −0,035] | 0,147 | **0,688** | — | 0,632 |
| `klim_frek1` (bulan itu) | **0,245** | — | 0,190 | 0,665 | −0,023 [−0,031, −0,012] | 0,613 |
| `log_klim_jeda` | 0,201 | −0,044 [−0,070, −0,021] | 0,249 | 0,674 | −0,014 [−0,035, +0,007] | 0,688 |
| Persistence | 0,142 | −0,103 [−0,133, −0,077] | 0,189 | 0,551 | −0,137 [−0,164, −0,108] | 0,574 |
| Seasonal naive | 0,113 | −0,132 [−0,156, −0,112] | 0,095 | 0,551 | −0,138 [−0,159, −0,116] | 0,449 |
| Ratio scaling | 0,115 | −0,130 [−0,159, −0,106] | 0,132 | 0,486 | −0,203 [−0,231, −0,168] | 0,508 |
| FWI skala nasional | 0,124 | −0,121 [−0,149, −0,095] | 0,163 | 0,198 | −0,490 [−0,545, −0,419] | 0,189 |
| FWI persentil kabupaten | 0,133 | −0,112 [−0,143, −0,084] | 0,196 | 0,140 | −0,548 [−0,592, −0,495] | 0,168 |
| Logistik (FWI) | 0,156 | −0,089 [−0,122, −0,061] | 0,236 | 0,212 | −0,476 [−0,518, −0,423] | 0,209 |
| Klimatologi + FWI | 0,181 | −0,064 [−0,090, −0,038] | 0,196 | 0,462 | −0,226 [−0,253, −0,194] | 0,454 |
| Logistik (ONI, DMI) | 0,124 | −0,121 [−0,151, −0,101] | 0,276 | 0,149 | −0,539 [−0,584, −0,490] | 0,198 |
| Random forest | 0,288 | +0,043 [+0,016, +0,071] | 0,433 | 0,738 | +0,050 [+0,032, +0,067] | 0,727 |
| Gradient boosting | 0,276 | +0,031 [−0,002, +0,066] | 0,427 | 0,726 | +0,038 [+0,021, +0,054] | 0,705 |
| MLP | 0,257 | +0,012 [−0,021, +0,042] | 0,265 | 0,714 | +0,026 [+0,007, +0,044] | 0,660 |

Angka model yang dipelajari sama dengan Hasil_Patokan_6 (A1 dan pasca-hoc). Rasio keunggulan gradient
boosting atas klimatologi rata-rata lawan atas rujukan terkuat: per kabupaten +0,151 / +0,031 = 4,9;
nasional +0,247 / +0,038 = 6,5.

### B5. Temuan 2 terhadap rujukan terkuat

Keunggulan gabungan antartahun atas rujukan [95 %, 1.000] dan keunggulan dalam-tahun per tahun 2020–2025:

| | Gabungan | Dalam-tahun (rata-rata) | Per tahun 2020 / 21 / 22 / 23 / 24 / 25 |
|---|---|---|---|
| GB, per kab | +0,237 [+0,202, +0,273] | +0,031 | +0,034 / +0,024 / +0,115 / +0,088 / −0,059 / −0,015 |
| RF, per kab | +0,243 [+0,211, +0,274] | +0,043 | +0,094 / +0,035 / +0,047 / +0,114 / −0,035 / +0,004 |
| MLP, per kab | +0,075 [+0,052, +0,097] | +0,012 | −0,040 / +0,010 / +0,096 / +0,031 / −0,033 / +0,009 |
| Logistik (ONI, DMI), per kab | +0,085 [+0,052, +0,122] | −0,121 | −0,141 / −0,036 / −0,030 / −0,176 / −0,179 / −0,165 |
| `log_klim_jeda`, per kab | +0,058 [+0,032, +0,085] | −0,044 | −0,063 / −0,009 / −0,011 / +0,001 / −0,130 / −0,051 |
| GB, nasional | +0,073 [+0,054, +0,094] | +0,038 | +0,070 / +0,046 / +0,081 / +0,013 / −0,020 / +0,039 |
| RF, nasional | +0,095 [+0,074, +0,120] | +0,050 | +0,087 / +0,065 / +0,080 / +0,032 / −0,002 / +0,038 |

Per kabupaten, keunggulan gabungan gradient boosting (+0,237) melebihi setiap tahun tunggal (terbesar
+0,115, 2022), sama seperti di naskah terkirim. Model logistik ONI/DMI, yang kalah dari rujukan di
setiap tahun, unggul +0,085 bila tahun digabung. Nasional, keunggulan gabungan (+0,073) dua kali
rata-rata dalam-tahun tetapi tidak melebihi tahun terbaik (+0,081).

### B3. Jeda dua bulan

| | AUC-PR (jeda 1) | Atas rujukan [95 %] | Rugi lawan jeda 1 [95 %] |
|---|---|---|---|
| RF, per kab | 0,251 (0,288) | +0,006 [−0,021, +0,036] | +0,037 [+0,017, +0,056] |
| GB, per kab | 0,215 (0,276) | −0,030 [−0,056, +0,000] | +0,061 [+0,041, +0,083] |
| RF, nasional | 0,678 (0,738) | −0,010 [−0,025, +0,003] | +0,060 [+0,049, +0,072] |
| GB, nasional | 0,663 (0,726) | −0,025 [−0,040, −0,011] | +0,063 [+0,052, +0,076] |

S2, GB jeda 2 atas rujukan: per kabupaten 2015 +0,047 [+0,022, +0,070] (jeda 1 +0,102), 2014 −0,145
[−0,175, −0,112] (−0,046), 2019 −0,012 [−0,050, +0,027] (+0,071); nasional +0,005, −0,052 [−0,074,
−0,031], +0,009 (jeda 1 +0,048, +0,009, +0,033). **Bacaan:** tanpa data bulan lalu, model yang dipelajari
tidak lebih baik daripada klimatologi yang sepadan; seluruh keunggulannya datang dari informasi jeda
satu bulan.

### B2. Kalibrasi ulang (R7)

Gradient boosting, per kabupaten, rata-rata tahun.

| Kelompok | Varian | ECE | Brier | p rata-rata | Prevalensi | Selisih ke amplop, a = 0,05 / 0,1 / 0,2 / 0,3 / 0,5 |
|---|---|---|---|---|---|---|
| Biasa | mentah | 0,023 | 0,0297 | 0,054 | 0,040 | 0,077 / 0,125 / 0,117 / 0,128 / 0,090 |
| Biasa | Platt | 0,031 | 0,0305 | 0,065 | 0,040 | 0,139 / 0,196 / 0,169 / 0,162 / 0,101 |
| Biasa | isotonik | 0,031 | 0,0305 | 0,064 | 0,040 | 0,183 / 0,229 / 0,140 / 0,133 / 0,088 |
| Biasa | Platt + ONI/DMI | 0,027 | 0,0296 | 0,064 | 0,040 | 0,112 / 0,155 / 0,151 / 0,135 / 0,088 |
| Ekstrem | mentah | 0,060 | 0,0931 | 0,127 | 0,187 | 0,174 / 0,088 / 0,059 / 0,071 / 0,098 |
| Ekstrem | Platt | 0,042 | 0,0885 | 0,150 | 0,187 | 0,106 / 0,054 / 0,026 / 0,039 / 0,068 |
| Ekstrem | isotonik | 0,046 | 0,0901 | 0,147 | 0,187 | 0,029 / 0,045 / 0,045 / 0,067 / 0,064 |
| Ekstrem | Platt + ONI/DMI | 0,039 | 0,0886 | 0,154 | 0,187 | 0,093 / 0,052 / 0,030 / 0,045 / 0,059 |

Nilai "siaga bila p > a" (mentah): biasa +0,380 / +0,219 / +0,077 / −0,001 / −0,021; ekstrem +0,285 /
+0,468 / +0,579 / +0,487 / +0,293. ECE per tahun, mentah: biasa 0,040 / 0,019 / 0,007 / 0,027 / 0,021 /
0,024; ekstrem (2014, 2015, 2019) 0,102 / 0,058 / 0,021.

**R7 — tidak lolos.** Di tahun ekstrem semua varian menurunkan ECE dan mempersempit selisih ke amplop pada
kelima a (Platt + ONI/DMI: ECE 0,060 → 0,039). Di tahun biasa semua varian **menaikkan** ECE (0,023 →
0,027–0,031) dan melebarkan selisih ke amplop. Pemeta yang dipelajari dari tahun lampau, yang campuran
tahun tenang dan tahun ekstrem, menaikkan peluang di semua tahun: menolong tahun ekstrem, merugikan
tahun biasa. Naskah menulis bahwa kalibrasi ulang dari tahun lampau tidak cukup karena salah kalibrasi
bergeser antartahun.

### B4. FWI System

Selisih terhadap `klim_saja` (logistik pada log klimatologi rata-rata dan musim; AUC-PR dalam-tahun
0,150 per kab, 0,488 nasional; gabungan 0,125 dan 0,457), S1 2020–2025:

| Komponen | Per kab, dalam-tahun | Per kab, gabungan [95 %] | Nasional, dalam-tahun | Nasional, gabungan [95 %] |
|---|---|---|---|---|
| DC | −0,005 | +0,034 [+0,027, +0,043] | −0,018 | +0,001 [−0,017, +0,016] |
| DMC | +0,020 | +0,053 [+0,044, +0,064] | +0,002 | +0,021 [+0,007, +0,033] |
| FFMC | +0,034 | +0,102 [+0,089, +0,118] | +0,031 | +0,072 [+0,061, +0,086] |
| BUI | +0,018 | +0,054 [+0,045, +0,065] | +0,002 | +0,023 [+0,009, +0,035] |
| FWI | +0,031 | +0,070 [+0,061, +0,083] | −0,021 | +0,004 [−0,014, +0,019] |
| Kelimanya | +0,027 | +0,068 [+0,057, +0,081] | −0,019 | +0,012 [−0,006, +0,028] |
| **ISI (baru)** | **+0,044 [+0,032, +0,061]** | +0,103 [+0,091, +0,120] | +0,017 [+0,003, +0,029] | +0,054 [+0,042, +0,067] |

(Dalam-tahun lima komponen lama dari Hasil_Patokan_5; selang di sana.) ISI di tahun ekstrem (S2), per
kab: 2015 +0,034 [+0,011, +0,060], 2014 −0,035 [−0,067, −0,003], 2019 −0,083 [−0,108, −0,056]; nasional
−0,017, −0,037, −0,077.

DC bulan lalu, rerata kabupaten, September: Kalimantan Tengah 2015 = 367, 2016–2025 rerata 89, maks 279
(2019); Sumatera Selatan 413, 131, maks 348 (2019). **Bacaan:** DC reanalisis menangkap kemarau 2015.
Ia menambah bila tahun digabung (+0,034 per kab), yaitu ia membedakan tahun, tetapi tidak membedakan
kabupaten di dalam satu tahun. Komponen cepat (FFMC, dan ISI yang dibangun darinya bersama angin)
yang menambah dalam-tahun; ISI yang terbesar per kabupaten. Di tahun ekstrem yang disisihkan semua
komponen kecuali FFMC (dan ISI pada 2015) menurunkan skor, seperti di Hasil_Patokan_5.

### B1. Provinsi (R6)

**Penyimpangan (6 Nov 2026, ditulis sebelum biaya nasional dijalankan).** `patokan_revisi_provinsi.py`
dihentikan sesudah ±1,5 jam CPU pada nol bersebelahan sasaran nasional. Uji terpisah (skrip sekali pakai di
scratchpad, tidak masuk repo): wilayah bersebelahan acak untuk 19 provinsi palsu berukuran 38…7 kabupaten
yang masing-masing ≥ 40 positif nasional lolos 0 dari 30 percobaan (besar dulu, 50 coba per wilayah), 0
dari 20 dengan 500 coba, dan 0 dari 20 bila benih diambil dari kabupaten yang pernah positif (besar dulu
maupun urutan acak); kegagalan terjadi pada wilayah ke-7 sampai ke-13, ketika kabupaten berapi yang
tersisa sudah terpecah. Nol bersebelahan nasional karenanya **tidak dapat dihitung** dengan rancangan
terdaftar, dan R6 nasional tidak dapat dinilai persis seperti tertulis. Sisa B1 nasional (biaya transfer
tervalidasi silang) dijalankan dengan fungsi yang sama oleh `patokan_revisi_provinsi_biaya.py`; RNG
dimulai ulang dari seed 2026. Nol bersebelahan per kabupaten selesai sebelum penghentian dan dipakai apa
adanya.

| | Per kabupaten (`klim_frek1`, 11 provinsi) | Nasional (`klim_frek`, 19 provinsi) |
|---|---|---|
| Juara | `klim_frek1` 7, RF 2, GB 1, MLP 1 | `klim_frek` 10, RF 3, `log_klim_jeda` 3, MLP 2, GB 1 |
| D teramati | 0,600 | 0,696 |
| Nol acak: D median [p95], P | 0,745 [0,818], 0,965 | 0,784 [0,825], 0,935 |
| Nol bersebelahan: D median [p95], P | 0,782 [0,855], 0,995 | tidak dapat dihitung (penyimpangan) |
| H teramati | 0,109 | 0,061 |
| Nol acak: H median [p95], P | 0,077 [0,111], 0,060 | 0,048 [0,065], 0,125 |
| Nol bersebelahan: H median [p95], P | 0,101 [0,145], 0,345 | — |
| Biaya juara provinsi lain [95 %] | +0,008 [−0,015, +0,049] | +0,003 [−0,015, +0,039] |
| Biaya juara nasional [95 %] | +0,017 [−0,017, +0,071] | +0,002 [−0,016, +0,040] |

RF − rujukan per provinsi, nasional: terbesar Riau +0,064, Sumatera Selatan +0,051, Kalimantan Tengah
+0,041 (juara GB, RF, RF); terkecil Kalimantan Selatan −0,210, Bangka Belitung −0,093. Per kabupaten:
Papua +0,115, Jawa Timur +0,106, Jawa Barat +0,066; terkecil Nusa Tenggara Timur −0,196, Sulawesi
Tenggara −0,184.

**R6 — tidak lolos pada kedua sasaran.** Per kabupaten: D di bawah median nol bersebelahan (dan nol acak),
selang biaya transfer mencakup nol. Nasional: nol bersebelahan tidak dapat dihitung, tetapi D teramati di
bawah median nol acak dan selang biaya transfer mencakup nol, jadi syarat kedua R6 gagal apa pun hasil
nol bersebelahan. Temuan 5 dipindah ke suplemen dengan satu kalimat di badan naskah. Dengan rujukan yang
sepadan, provinsi justru **lebih sering sepakat** daripada pengelompokan acak, karena klimatologi juara
di sebagian besar provinsi.

## Putusan fase B dan akibat untuk naskah

- **B1/R6:** Temuan 5 gugur → suplemen S5, satu kalimat di §6.6.
- **B2/R7:** kalibrasi ulang dari tahun lampau tidak cukup → dilaporkan sebagai hasil (§6.5, Tabel 6).
- **B3:** seluruh keunggulan model dari data bulan lalu → §6.3, abstrak.
- **B4:** DC menangkap kemarau 2015, menambah hanya bila tahun digabung; ISI komponen terbaik dalam-tahun
  per kabupaten → §3.5, §6.3, Tabel 4.
- **B5:** Temuan 2 bertahan dan lebih tajam (+0,237 gabungan lawan +0,031 dalam-tahun) → §6.4.
- **B6:** Tabel 2.
