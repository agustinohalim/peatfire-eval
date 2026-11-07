# Hasil Tolok Ukur Nasional 6 — klimatologi frekuensi lampau dan uji ulang temuan (revisi PeatFireBench, fase A)

**6 November 2026.** Revisi PeatFireBench, fase A. Pemicu: telaah internal pra-kirim atas naskah versi
2.3.0 (catatan telaah tidak dirilis).

## Rancangan fase A (ditulis 6 Nov 2026, SEBELUM skrip mana pun dijalankan)

Alasan: dua peran (metode, *devil's advocate*) secara independen menilai rujukan klimatologi
(rata-rata hitungan titik panas per kabupaten dan bulan kalender) tidak cocok dengan sasaran per
kabupaten (melampaui p90 kabupaten sendiri). Bila rujukan lemah, keunggulan model yang dipelajari,
Temuan 1, dan nilai keputusan Tabel 6 bisa terlalu besar. Selain itu: selang Temuan 1 menahan tahun
tetap; tahun tenang dan tahun ekstrem dilatih dengan rezim berbeda; ambang nilai keputusan dipilih
dengan melihat hasil; aturan 10 % per tahun memakai bulan yang belum diprakirakan; tidak ada
baseline sederhana "klimatologi + sejarah api terbaru".

Data dan model tidak berubah: `panel_nasional_fitur_penuh.csv`, putaran penuh, parameter model
sama dengan `patokan_prediksi.py`, seed 42. Prediksi S1/S2 model yang dipelajari diambil dari
`patokan_prediksi_penuh.csv` apa adanya; hanya baseline baru yang dihitung di sana.

**Skrip** (ditulis bersama rancangan ini, dijalankan sesudah komit):

1. `patokan_revisi_prediksi.py` → `DL_FIRE_NASIONAL/patokan_revisi_prediksi.csv`
   - Baseline baru, dari tahun latih saja, untuk S1 (2019–2025), S2 (2015, 2014, 2019), dan skema
     baru Q:
     - `klim_frek` (rujukan utama): bagian kabupaten-bulan latih yang positif dalam jendela tiga
       bulan kalender (m−1, m, m+1, melingkar) di kabupaten itu. Seri dipecah dengan
       10⁻⁶ × k/(1+k), k = klimatologi rata-rata hitungan. Ini peluang, jadi juga dipakai sebagai
       rujukan Brier.
     - `klim_frek1`: sama, tanpa jendela (bulan kalender itu saja). Kepekaan.
     - `log_klim_jeda`: regresi logistik terskala pada log1p(klimatologi), log1p(tp_lag1..3),
       bulan_sin, bulan_cos. Untuk baris latih, klimatologi dihitung tanpa tahun baris itu sendiri.
   - Skema Q (tahun tenang disisihkan, seperti S2): 2020, 2021, 2022, 2023, 2024, 2025 masing-masing
     disisihkan, latih pada semua tahun lain; rf, xgb, mlp dilatih ulang dengan parameter sama, plus
     klimatologi, persistence, dan tiga baseline baru.
   - Cek reproduksi: xgb S2 2015 per kabupaten dilatih ulang dan dibandingkan dengan berkas lama
     (selisih maksimum dicetak; harus < 10⁻⁶ atau dijelaskan).
2. `patokan_revisi_oof.py` → `DL_FIRE_NASIONAL/patokan_revisi_oof.csv`. Sasaran per kabupaten
   saja. Untuk tiap lipatan luar (S1 2020–2025, S2 2014, 2015, 2019) dan tiap tahun latih v:
   xgb dilatih pada tahun latih tanpa v, memprakirakan v; `klim_frek`, `klimatologi`, dan
   `log_klim_jeda` dari tahun latih tanpa v. Ini prakiraan luar-lipatan pada tahun latih, dipakai
   untuk memilih ambang tanpa melihat tahun uji.
3. `patokan_revisi_analisis.py` → `DL_FIRE_NASIONAL/penuh/patokan_revisi_analisis.log`.

**Analisis yang dilaporkan** (semua yang di bawah dilaporkan, apa pun hasilnya):

- **A1. Tabel 2 dan 3 ulang.** AUC-PR dalam-tahun S1 2020–2025 dan keunggulan atas `klim_frek`
  [95 %, bootstrap kabupaten 1.000] untuk semua model plus `log_klim_jeda`, kedua sasaran; S2 per
  tahun terhadap `klim_frek`. Klimatologi rata-rata dan `klim_frek1` sebagai baris tambahan.
- **A2. Temuan 1 ulang.** Kontras ekstrem (S2 2015, 2014, 2019) − tenang (S1 2020, 2021, 2022,
  2024, 2025) dari keunggulan gradient boosting, random forest, MLP atas `klim_frek`, 411
  kabupaten, kedua sasaran, dan selisih antar-sasaran, dengan lima ukuran:
  m1 selisih AUC-PR mentah; m2 bagian ruang sisa (AP_m − AP_R)/(1 − AP_R); m3 AUC-PR
  ternormalisasi prevalensi, (AP_m − AP_R)/(1 − π); m4 selisih ROC-AUC; m5 Brier skill terhadap
  `klim_frek`, 1 − BS_m/BS_R (model berpeluang saja). Dua selang: (i) bootstrap kabupaten 300
  ulangan; (ii) dua tahap 1.000 ulangan — tahun ekstrem diambil ulang dari tiga, tahun tenang dari
  lima, lalu kabupaten. Juga kontras per tahun, dan lengan Q: tahun tenang dari skema Q (rezim
  latih sama dengan S2). Diulang terhadap klimatologi rata-rata sebagai pembanding versi terkirim.
- **A3. Baseline sederhana.** `log_klim_jeda` dalam Tabel 2/3 (A1); selisih gradient boosting −
  `log_klim_jeda` dalam-tahun S1 2020–2025 [95 %].
- **A4. Nilai keputusan yang dapat dicapai** (per kabupaten). Untuk tiap lipatan luar dan tiap
  a ∈ {0,05; 0,1; 0,2; 0,3; 0,5}: ambang yang memaksimalkan nilai Richardson pada prakiraan
  luar-lipatan tahun latih, lalu diterapkan pada tahun uji. Gradient boosting, `klim_frek`,
  klimatologi rata-rata, `log_klim_jeda`. Selisih gradient boosting − `klim_frek` per a dengan
  bootstrap kabupaten 500. Amplop lama dicetak berdampingan.
- **A5. Aturan siaga kausal.** (a) ambang = skor yang menyiagakan 10 % kabupaten-bulan prakiraan
  luar-lipatan tahun latih; (b) 10 % kabupaten tertinggi tiap bulan (hanya bulan yang sama). Hit
  rate, FAR, dan bagian siaga yang terwujud per kelompok tahun, untuk model yang sama dengan A4.

**Aturan baca (dikunci sekarang):**

- **R1 (kritis 2).** Bila batas bawah selang keunggulan gradient boosting atas `klim_frek`
  (dalam-tahun, S1 2020–2025, per kabupaten) > 0, klaim "model yang dipelajari mengalahkan
  klimatologi yang sepadan" bertahan, dengan besar yang baru. Bila tidak, klaim itu dicabut dan
  naskah dibangun ulang, bukan direvisi.
- **R2 (Temuan 1).** *Bertahan* bila, terhadap `klim_frek`, selisih antar-sasaran gradient
  boosting > 0 dengan selang dua tahap di atas nol pada m1, dan sedikitnya tiga dari m2–m5
  bertanda sama dengan selang kabupaten di atas nol. *Terbatas* bila selang kabupaten di atas nol
  tetapi selang dua tahap mencakup nol: naskah menulisnya sebagai pengamatan pada tiga tahun
  bernama, bukan pola umum. *Gugur* bila tanda berbalik atau selang kabupaten mencakup nol pada m1.
- **R3 (rancu rezim latih).** Bila lengan Q mempertahankan tanda kontras kedua sasaran, rancu
  "tahun tenang dilatih kronologis" terjawab; bila tidak, dilaporkan sebagai batasan Temuan 1.
- **R4 (baseline sederhana).** Bila selang gradient boosting − `log_klim_jeda` mencakup nol, naskah
  menyatakan bahwa regresi logistik pada sejarah lokal menyamai *ensemble* pohon, dan klaim model
  dipersempit ke "sejarah lokal".
- **R5 (nilai keputusan).** "Lebih bernilai daripada klimatologi pada setiap rasio biaya/rugi"
  hanya boleh ditulis bila batas bawah selang gradient boosting − `klim_frek` dengan ambang dari
  tahun latih > 0 pada kelima a. Selain itu, naskah menyebut a yang lolos saja.

Batas yang diketahui sebelum dijalankan: ambang p90 kabupaten (`ambang_kabupaten`) di panel
dihitung dari semua tahun (temuan minor metode); tidak diubah di fase A. Fase B (nol spasial
provinsi, jeda dua bulan, kepekaan ambang, kalibrasi ulang, ISI dan DC gabungan) menyusul sesudah
fase A dibaca.

## Hasil (dijalankan 6 Nov 2026, komit rancangan `802c396`)

Log: `DL_FIRE_NASIONAL/penuh/patokan_revisi_prediksi.log`, `patokan_revisi_oof.log`,
`patokan_revisi_analisis.log` (peringatan numpy/pandas dibuang dari log sesudah dijalankan; isi angka
tidak disentuh). Cek: klimatologi rata-rata dihitung ulang sama dengan berkas lama (selisih maks
4,6 × 10⁻¹³); xgb S2 2015 dilatih ulang sama dengan berkas lama (selisih maks 3,0 × 10⁻⁸). Penyimpangan dari
rancangan: tidak ada. Selang dua tahap untuk rujukan klimatologi rata-rata memakai 300 ulangan (sesuai
skrip, `n=300`), bukan 1.000; rujukan itu hanya pembanding.

### A1. Tabel 2 dan 3 terhadap `klim_frek`

S1 2020–2025, AUC-PR dalam-tahun dan keunggulan atas `klim_frek` [95 %, bootstrap kabupaten 1.000].

| Model | Per kabupaten (411; prev 4,02 %) | Unggul atas `klim_frek` | Nasional (498; prev 6,13 %) | Unggul atas `klim_frek` |
|---|---|---|---|---|
| Random forest | 0,288 | +0,095 [+0,075, +0,122] | 0,738 | +0,050 [+0,032, +0,067] |
| Gradient boosting | 0,276 | +0,083 [+0,060, +0,115] | 0,726 | +0,038 [+0,021, +0,054] |
| MLP | 0,257 | +0,064 [+0,039, +0,091] | 0,714 | +0,026 [+0,007, +0,044] |
| `klim_frek1` (tanpa jendela) | 0,245 | +0,052 [+0,035, +0,073] | 0,665 | −0,023 [−0,031, −0,012] |
| `log_klim_jeda` | 0,201 | +0,008 [−0,007, +0,026] | 0,674 | −0,014 [−0,035, +0,007] |
| **`klim_frek`** (rujukan) | **0,193** | — | **0,688** | — |
| Logistik (FWI) | 0,156 | −0,037 [−0,056, −0,019] | 0,212 | −0,476 [−0,518, −0,423] |
| Persistence | 0,142 | −0,051 [−0,068, −0,034] | 0,551 | −0,137 [−0,164, −0,108] |
| Klimatologi rata-rata (rujukan naskah terkirim) | 0,125 | −0,068 [−0,083, −0,055] | 0,479 | −0,209 [−0,240, −0,169] |
| FWI (bulan lalu) | 0,124 | −0,069 [−0,085, −0,048] | 0,198 | −0,490 [−0,545, −0,419] |
| Logistik (ONI, DMI) | 0,124 | −0,069 [−0,085, −0,059] | 0,149 | −0,539 [−0,584, −0,490] |

Keunggulan gradient boosting atas klimatologi turun dari +0,151 menjadi +0,083 per kabupaten dan dari
+0,247 menjadi +0,038 nasional. Klimatologi rata-rata hitungan memang rujukan lemah: nasional 0,479
lawan 0,688 untuk frekuensi lampau.

S2, keunggulan atas `klim_frek` [95 %, 500]:

| Tahun | Per kab: GB | RF | MLP | Nasional: GB | RF | MLP |
|---|---|---|---|---|---|---|
| 2015 | +0,112 [+0,087, +0,135] | +0,099 | +0,090 | +0,048 [+0,032, +0,065] | +0,049 | +0,047 |
| 2014 | −0,004 [−0,039, +0,032] | −0,031 | −0,009 | +0,009 [−0,005, +0,026] | +0,011 | +0,003 |
| 2019 | +0,076 [+0,039, +0,107] | +0,062 | +0,021 | +0,033 [+0,016, +0,053] | +0,028 | +0,021 |

### A2. Temuan 1 terhadap `klim_frek`

Selisih antar-sasaran (per kabupaten − nasional) dari kontras ekstrem − tenang, gradient boosting,
411 kabupaten:

| Ukuran | Tenang S1 (seperti naskah) | Selang kab / dua tahap | Tenang Q (rezim latih sama) | Selang kab / dua tahap |
|---|---|---|---|---|
| m1 AUC-PR mentah | −0,010 | [−0,056, +0,024] / [−0,083, +0,049] | −0,024 | [−0,071, +0,018] / [−0,101, +0,045] |
| m2 ruang sisa | +0,028 | [−0,064, +0,101] / [−0,129, +0,154] | +0,006 | [−0,081, +0,084] / [−0,136, +0,140] |
| m3 AUC-PR ternormalisasi | −0,002 | [−0,051, +0,034] / [−0,084, +0,064] | −0,016 | [−0,065, +0,028] / [−0,100, +0,061] |
| m4 ROC-AUC | −0,065 | [−0,084, −0,044] / [−0,125, −0,017] | −0,062 | [−0,079, −0,041] / [−0,124, −0,011] |
| m5 Brier skill | −0,153 | [−0,198, −0,112] / [−0,296, −0,025] | −0,126 | [−0,174, −0,087] / [−0,274, +0,001] |

Pembanding, rujukan klimatologi rata-rata (versi terkirim), m1: +0,293 [+0,219, +0,340], dua tahap
[+0,157, +0,423]; m2 +0,549 — angka naskah terulang. RF dan MLP searah dengan gradient boosting pada
kedua rujukan (log).

Per tahun, gradient boosting atas `klim_frek`, m1: per kabupaten 2015 +0,112, 2014 −0,004, 2019 +0,076;
tenang 2020 +0,129, 2021 +0,049, 2022 +0,138, 2024 +0,012, 2025 +0,095. Nasional 2015 +0,048, 2014
+0,009, 2019 +0,033; tenang +0,070, +0,046, +0,081, −0,020, +0,039. Dengan rujukan yang sepadan,
keunggulan model tidak lebih kecil di tahun ekstrem untuk sasaran nasional, dan tidak lebih besar untuk
sasaran per kabupaten.

### A3. Baseline sederhana

Gradient boosting − `log_klim_jeda`, dalam-tahun S1 2020–2025: per kabupaten +0,075 [+0,054, +0,104],
nasional +0,052 [+0,032, +0,073]. Random forest: +0,087 [+0,068, +0,111] dan +0,064 [+0,043, +0,085].
`log_klim_jeda` sendiri tidak lebih baik daripada `klim_frek` (+0,008 [−0,007, +0,026] per kabupaten).

### A4. Nilai keputusan dengan ambang dari tahun latih

Nilai ekonomi relatif rata-rata tahun, a = 0,05 / 0,1 / 0,2 / 0,3 / 0,5; amplop dalam kurung.

| | Gradient boosting | `klim_frek` | Klimatologi rata-rata | `log_klim_jeda` |
|---|---|---|---|---|
| Biasa | +0,283 / +0,224 / +0,130 / +0,081 / +0,009 (+0,457 / +0,344 / +0,194 / +0,127 / +0,069) | +0,003 / −0,412 / −0,223 / 0 / 0 (+0,336 / +0,236 / +0,132 / +0,085 / +0,048) | −0,312 / −0,597 / −0,211 / 0 / 0 | +0,132 / −0,045 / −0,134 / −0,153 / −0,008 |
| Ekstrem | +0,367 / +0,461 / +0,568 / +0,442 / +0,287 (+0,459 / +0,555 / +0,638 / +0,558 / +0,391) | +0,061 / −0,143 / −0,057 / 0 / 0 (+0,412 / +0,531 / +0,617 / +0,519 / +0,317) | −0,148 / −0,284 / −0,057 / 0 / 0 | +0,140 / +0,276 / +0,337 / +0,134 / +0,001 |

Selisih gradient boosting − `klim_frek`, dapat dicapai [95 %, 500]: biasa +0,280 [+0,217, +0,401],
+0,635 [+0,495, +0,891], +0,353 [+0,289, +0,437], +0,081 [+0,049, +0,107], +0,009 [+0,002, +0,016];
ekstrem +0,306 [+0,224, +0,391], +0,604 [+0,533, +0,670], +0,624 [+0,594, +0,655], +0,442 [+0,422, +0,465],
+0,287 [+0,261, +0,313].

Bacaan: selisih yang besar sebagian besar berasal dari kegagalan klimatologi, bukan dari kehebatan
model. Ambang klimatologi yang dipilih di tahun latih tidak berpindah ke tahun uji (nilai negatif sampai
−0,41 di tahun biasa): klimatologi diskret, banyak seri, dan prevalensi berbeda antartahun. Pada amplop,
selisih gradient boosting − `klim_frek` di tahun biasa hanya +0,121 / +0,108 / +0,062 / +0,042 / +0,021,
dan di tahun ekstrem +0,047 / +0,024 / +0,021 / +0,039 / +0,074 (dihitung dari baris di atas). Naskah
harus melaporkan keduanya dan menyebut sebab selisih yang dapat dicapai.

### A5. Aturan siaga kausal

| | Aturan | GB | `klim_frek` | Klim rata-rata | `log_klim_jeda` | RF | MLP |
|---|---|---|---|---|---|---|---|
| Biasa | (a) ambang 10 % luar-lipatan: hit / FAR / siaga | 0,365 / 0,658 / 5,1 % | 0,434 / 0,808 / 10,7 % | 0,265 / 0,869 / 10,1 % | 0,334 / 0,790 / 6,5 % | — | — |
| Biasa | (b) 10 % tiap bulan: hit / FAR | 0,405 / 0,868 | 0,282 / 0,894 | 0,184 / 0,928 | 0,314 / 0,892 | 0,435 / 0,858 | 0,413 / 0,868 |
| Ekstrem | (a) | 0,732 / 0,379 / 22,1 % | 0,426 / 0,261 / 10,6 % | 0,349 / 0,363 / 10,1 % | 0,602 / 0,352 / 17,2 % | — | — |
| Ekstrem | (b) | 0,244 / 0,551 | 0,219 / 0,593 | 0,202 / 0,626 | 0,213 / 0,607 | 0,241 / 0,558 | 0,239 / 0,558 |

Dengan ambang tetap dari tahun latih (a), bagian siaga gradient boosting tidak stabil: 0–13 % di tahun
biasa (per tahun 4, 1, 0, 13, 6, 6 %) dan 15–29 % di tahun ekstrem; peluangnya bergeser antartahun.
Dengan kuota bulanan (b), gradient boosting menyiagakan 41 % bulan parah di tahun biasa lawan 28 % untuk
`klim_frek` dan 18 % untuk klimatologi rata-rata; di tahun ekstrem 24 % lawan 22 %. Angka naskah
terkirim "57 % lawan 26 %" (10 % per tahun, melihat seluruh tahun) tidak boleh dipakai lagi.

### Putusan menurut aturan baca

- **R1 — bertahan, lebih kecil.** Batas bawah keunggulan gradient boosting atas `klim_frek` +0,060 per
  kabupaten (dan +0,021 nasional). Klaim "model yang dipelajari mengalahkan klimatologi yang sepadan"
  boleh ditulis dengan +0,083 dan +0,038, bukan +0,151 dan +0,247.
- **R2 — Temuan 1 GUGUR.** Selisih antar-sasaran m1 −0,010, selang kabupaten [−0,056, +0,024] mencakup
  nol; tanda berbalik dari +0,293; m4 dan m5 bertanda berlawanan dengan klaim lama. Pembalikan di tahun
  ekstrem adalah buatan rujukan klimatologi rata-rata hitungan.
- **R3 —** lengan Q searah (m1 −0,024 [−0,071, +0,018]); Temuan 1 tidak pulih dengan rezim latih yang sama.
- **R4 — model pohon bertahan.** Gradient boosting dan random forest jelas di atas `log_klim_jeda`.
- **R5 — lolos pada kelima a** untuk nilai yang dapat dicapai, tetapi selisihnya didorong kegagalan
  ambang klimatologi; lihat bacaan A4. Kalimat naskah harus menyebut itu.

Akibat untuk naskah: judul dan tesis lama ("pilihan evaluasi mengubah model mana yang dipilih kabupaten")
dan Temuan 1 tidak dapat dipertahankan. Yang tersisa dan teruji: (1) model pohon mengalahkan
klimatologi frekuensi lampau dan regresi logistik sejarah lokal dengan selisih sedang; (2) **pilihan
rujukan klimatologi sendiri mengubah besar keunggulan 1,8 kali (per kabupaten, +0,151 lawan +0,083)
sampai 6,5 kali (nasional, +0,247 lawan +0,038) dan menciptakan pola tahun ekstrem yang
tidak ada** — ini pelajaran evaluasi yang dapat dibela, tetapi termasuk efek verifikasi yang dikenal
(pilihan prakiraan rujukan); (3) nilai keputusan yang dapat dicapai dan kegagalan ambang tetap.
Keputusan bingkai naskah diserahkan kepada penulis sebelum `Naskah_Artikel2_Revisi.md` ditulis.

## Cek pasca-hoc (TIDAK didaftarkan; ditulis sesudah hasil di atas dibaca)

Alasan: `klim_frek1` (frekuensi bulan kalender itu saja) ternyata rujukan terkuat untuk sasaran per
kabupaten (0,245 lawan 0,193 untuk `klim_frek` terdaftar), sedangkan `klim_frek` terkuat nasional.
Skrip `patokan_revisi_pascahoc.py`, log `DL_FIRE_NASIONAL/penuh/patokan_revisi_pascahoc.log`.

Keunggulan atas `klim_frek1`, dalam-tahun S1 2020–2025 [95 %, 1.000]:

| Model | Per kabupaten | Nasional |
|---|---|---|
| Random forest | +0,043 [+0,016, +0,071] | +0,073 [+0,054, +0,088] |
| Gradient boosting | **+0,031 [−0,002, +0,066]** | +0,061 [+0,042, +0,076] |
| MLP | +0,012 [−0,021, +0,042] | +0,049 [+0,028, +0,066] |
| `klim_frek` | −0,052 [−0,073, −0,035] | +0,023 [+0,012, +0,031] |
| `log_klim_jeda` | −0,044 [−0,070, −0,021] | +0,009 [−0,010, +0,026] |

S2, gradient boosting atas `klim_frek1`: per kabupaten 2015 +0,102 [+0,080, +0,122], 2014 −0,046
[−0,077, −0,011], 2019 +0,071 [+0,036, +0,105]; nasional +0,038, −0,008, +0,034.

Temuan 1 atas `klim_frek1`, gradient boosting, selisih antar-sasaran: m1 +0,070 [+0,022, +0,109]
(per kabupaten +0,022, nasional −0,048); m4 −0,050 [−0,068, −0,028].

Bacaan:

- **Terhadap rujukan klimatologi terkuat untuk tiap sasaran, keunggulan model kecil.** Per kabupaten,
  gradient boosting +0,031 dengan selang yang menyentuh nol; hanya random forest yang jelas di atas
  (+0,043). Nasional +0,038 (atas `klim_frek`, rujukan terkuat di sana). R1 terdaftar tetap lolos,
  tetapi naskah tidak boleh memilih rujukan yang lebih lemah bila rujukan yang lebih kuat sudah diketahui:
  rujukan untuk tiap sasaran harus yang terkuat, dan angka yang dilaporkan +0,03–0,04 (gradient boosting)
  atau +0,04–0,05 (random forest).
- **Temuan 1 bergantung pada rujukan.** Atas klimatologi rata-rata: +0,293; atas `klim_frek`: −0,010
  (nol); atas `klim_frek1`: +0,070 — dan ROC-AUC berlawanan tanda pada ketiganya. Pola yang tandanya
  ditentukan pilihan rujukan bukan temuan tentang tahun ekstrem; ini tetap gugur.
- 2014 tetap tahun ketika model tidak menambah apa pun atas klimatologi (per kabupaten bahkan lebih buruk
  atas `klim_frek1`).
