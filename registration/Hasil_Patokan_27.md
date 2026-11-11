# Hasil Patokan 27 — fase U: rujukan iklim musiman-wilayah, agregasi, klaster provinsi, multiplisitas

**11 November 2026.** Lanjutan `Hasil_Patokan_26.md` untuk `Naskah_Artikel2_Revisi.md`. Pemicu: telaah internal pra-kirim ketiga puluh satu
(catatan telaah tidak dirilis).

Temuan panel yang dijawab dengan analisis:

- **EF-E terlalu sederhana** (metode M1, domain D1). EF-E memakai satu koefisien ONI dan satu DMI untuk semua bulan
  dan semua kabupaten. Pengaruh ENSO/IOD atas kebakaran Indonesia terikat musim kemarau dan wilayah gambut. Jadi
  "model pohon mengalahkan EF-E" bisa jadi hanya menunjukkan cara EF-E dibuat.
- **Agregasi** (metode M2). Vonis dan bagian tingkat bergantung pada rata-rata rasio tahunan; satu rasio gabungan
  tidak pernah diberi selang dan vonis.
- **Klaster provinsi** (metode M6). Tabel 4 (ambang tahun latih) dan sel S20 belum diuji dengan provinsi diambil
  ulang.
- **Multiplisitas** (metode M4, devil's advocate X7). Klaim "unggul" di bagian peluang tidak dikoreksi untuk
  banyaknya perbandingan.

**Rancangan ini ditulis dan dikomit sebelum skrip fase U ditulis atau dijalankan.** Belum ada angka fase U yang
dilihat. Yang sudah diketahui: semua sel Tabel 4, Tabel 5, S20 dengan rata-rata rasio dan selangnya; satu rasio
gabungan sebagai angka titik (dicetak di log fase S, N, T tanpa selang dan vonis); koefisien EF-E per lipatan.

## Pertanyaan

- **U-Q1.** Apakah kalender yang disesuaikan dengan ONI dan DMI per musim dan per wilayah (EF-ES) menyerap lebih
  banyak keunggulan daripada EF-E, dan apakah ia terbedakan dari EF-L?
- **U-Q2.** Vonis mana di Tabel 4, Tabel 5 (maju) dan S20 yang bertahan bila BSS dihitung sebagai satu rasio
  gabungan, dan bila tahap pertama bootstrap mengambil provinsi, bukan kabupaten?
- **U-Q3.** Klaim "unggul" mana yang bertahan sesudah koreksi Holm dalam tiap keluarga perbandingan?

## Rujukan baru (satu bentuk masing-masing, tanpa varian)

Sel musim-wilayah k ∈ {kemarau, lain} × {Sumatra-Kalimantan, lain}:

- **kemarau** = bulan sasaran Juli–November; **lain** = Desember–Juni.
- **Sumatra-Kalimantan** = provinsi Aceh, SumateraUtara, SumateraBarat, Riau, KepulauanRiau, Jambi, Bengkulu,
  SumateraSelatan, BangkaBelitung, Lampung, KalimantanBarat, KalimantanTengah, KalimantanSelatan, KalimantanTimur,
  KalimantanUtara (kolom `provinsi` di `panel_nasional_bersih.csv`); **lain** = 19 provinsi sisanya.

Dengan c̃ dan logistik tanpa penalti (`C=1e6`) seperti EF-E:

- **EF-ES**: logit P = β₀ + β₁ logit(c̃_dm) + Σ_k 1[k] (γ_k ONI_m−1 + δ_k DMI_m−1): delapan koefisien indeks, tanpa
  intersep per sel. Kalender kabupaten sudah membawa musim dan tingkat wilayah.
- **EF-LES**: EF-ES ditambah suku nasional EF-L.

Jeda tetap satu bulan, seperti EF-E, supaya selisih EF-ES dan EF-E hanya karena musim dan wilayah. Indeks dipakai
sebagaimana terbit sekarang, sama dengan fase T. Ketersediaan waktu nyata (metode M8) tidak diuji dan disebut sebagai
batasan. Nilai hilang: skrip berhenti; tidak ada pengisian. Provinsi yang tidak terpetakan: skrip berhenti.

## Jalur, lipatan, model

Sama dengan fase T, tanpa perubahan:

1. **VIIRS S1 2020–2025, ambang tahun latih**, kedua sasaran. Model dari `patokan_ambang_s_<sasaran>.csv`: RF, GB,
   GB tersetel, MLP, GB + nasional, GB + nasional + DC; rujukan kalender, EF-P, EF-L, EF-LD. EF-L, EF-E dan EF-LE
   dihitung ulang dan harus sama dengan fase S/T (selisih < 1e-6).
2. **MODIS maju 2008–2019**, kedua sasaran. GB-FH (rata-rata enam benih), kalender, EF-P, EF-L, EF-LD dari
   `sse_modis_n_<nas|kab>.npz`; EF-L, EF-E dihitung ulang dan dicek. GB-FH + DC tidak ada di berkas itu dan tidak
   dilatih ulang.

## Skor dan selang

BSS dalam-tahun. Dua agregasi:

- **rata-rata rasio tahunan** (seperti selama ini);
- **satu rasio gabungan**: 1 − ΣSSE_a / ΣSSE_r atas tahun uji.

Tiga selang:

- **dua tahap kabupaten-tahun** (fase S/N, 1.000 ulangan, benih B.SEED), untuk kedua agregasi;
- **selang tahun t** (df 5 / 11), hanya untuk rata-rata rasio;
- **dua tahap provinsi-tahun**: tahap pertama mengambil provinsi dengan pengembalian (semua kabupatennya ikut),
  tahap kedua tahun; 1.000 ulangan, benih B.SEED + 1; untuk kedua agregasi.

Nilai p tahun: uji t satu sampel atas BSS tahunan (dua sisi). Holm di dalam tiap keluarga, yaitu tiap jalur ×
sasaran × rujukan. Keluarga VIIRS berisi enam model; MODIS satu (GB-FH), jadi Holm tidak mengubahnya.

## Sel

Untuk tiap jalur dan sasaran:

- **(a) Rujukan**: EF-ES atas kalender, atas EF-E, atas EF-L; EF-L atas EF-ES; EF-LES atas EF-L; EF-LES atas EF-ES.
- **(b) Model atas rujukan**: tiap model atas kalender, EF-P, EF-L, EF-LD, EF-E, EF-ES. Ini mengulang sel Tabel 4,
  Tabel 5 maju dan S20 dengan agregasi dan selang baru.
- **(c) Bagian tingkat**: tidak dihitung ulang. Fase P/Q/R sudah mencetak bagian rata-rata rasio dan satu rasio.
  Naskah melaporkan keduanya dari berkas hasil yang ada.

## Aturan keputusan (sebelum hasil)

- **R63 (rujukan iklim di teks).**
  - Bila EF-ES atas EF-E unggul per kabupaten (rata-rata rasio, kedua selang) di kedua jalur, EF-ES menjadi rujukan
    keadaan iklim di teks utama dan abstrak; EF-E pindah ke S20.
  - Selain itu, EF-E tetap di teks dan EF-ES masuk S20 dengan satu kalimat di §6.2.
  - Untuk rujukan yang dipakai di teks, perbandingannya dengan EF-L dinilai dengan aturan tiga keadaan R59. Abstrak
    menyebut hasilnya dengan kata-kata R59; untuk EF-E itu "tak terbedakan dari EF-L".
- **R64 (kata "beats" di abstrak, §1 dan §9).** Sebuah model boleh ditulis "beats" sebuah rujukan di tempat-tempat
  itu hanya bila tiga syarat terpenuhi:
  - lolos rata-rata rasio dengan kedua selang;
  - selang dua tahap satu rasio gabungan tanpa nol;
  - p tahun terkoreksi Holm < 0,05 di keluarganya.
  Sel yang hanya memenuhi syarat pertama ditulis di badan naskah sebagai "above it under the paper's rule, not robust
  to [agregasi | multiplisitas]" dan tidak masuk abstrak.
- **R65 (tanda tabel).**
  - Sel Tabel 4/5 yang vonis rata-rata rasionya berbeda dari vonis satu rasio gabungan (dua tahap) diberi ‡.
  - Sel yang vonisnya berubah dengan klaster provinsi diberi § (konvensi Tabel 4 fase R; diperluas ke ambang tahun
    latih).
  - Bila ada sel tebal yang kehilangan vonis di kedua cara, teks menyebutnya.
- **R66 (aturan rujukan tidak bertambah).** EF-ES dan EF-LES berada di luar aturan empat rujukan dan protokol 2026,
  sama dengan EF-E (R60).
- **R67 (multiplisitas di teks).** §4.3 menyebut jumlah keluarga dan perbandingan Holm. Klaim "per district no learned
  model is shown to beat EF-L" tidak berubah oleh Holm, karena Holm hanya mempersempit "unggul". Klaim yang berubah
  disebut.

## Putaran teks yang menyertai (sesudah fase U, apa pun hasilnya)

Tanpa analisis baru:

1. **Judul.** Judul memimpin dengan tolok ukur dan rujukan, bukan dengan bagian tingkat. Klausa syarat tahun uji
   pindah ke abstrak. Frasa "the fire level" tetap (R37d). Bentuk kerja: "IndoFireBench: a district benchmark for
   monthly fire forecasts in Indonesia and the reference forecasts that follow the fire level".
2. **Abstrak.**
   - Dibangun sebagai satu alur: tolok ukur → tangga rujukan → hubungan antara galat tingkat kalender dan skill yang
     terserap (kemiringan fase Q) sebagai besaran yang bisa dibawa ke tempat lain → vonis → uji 2026.
   - VIIRS, MODIS, ENSO dieja.
   - Paling panjang 240 kata menurut hitungan pemeriksa.
3. **Pendahuluan** paling panjang sekitar 650 kata. Dibuka dengan masalah statistik: skill atas klimatologi saat laju
   dasar berubah antartahun. Hukum dan lembaga Indonesia diringkas.
4. **Pustaka.** Ditambah tiga makalah model stokastik bahaya alam, tradisi model statistik kejadian kebakaran, dan tangga
   tolok ukur di prakiraan hidrologi. Hanya rujukan yang diverifikasi lewat Crossref dan dibaca abstraknya.
5. **Perbaikan dari panel.**
   - Analogi CLIPER dibalik: persistensi prediktan adalah EF-L (domain D6).
   - Kalimat "EF-L's level comes from fire counts, not from the climate state" diganti (domain D7).
   - "GB, fire history" vs "GB-FH" dan "raw calendar" diseragamkan (presentasi X5).
   - Model nasional + DC yang lolos atas EF-E disebut (metode M10).
6. **Log analisis.** §4.4 dan S18 ditambah baris fase T dan U, dan jumlah putaran diperbarui (metode M5). Rilis kode
   2.11.6 memuat skrip fase T dan U.

## Skrip

`patokan_iklim_u.py <sasaran>` (VIIRS) dan `patokan_iklim_u_modis.py <sasaran>` (MODIS), disalin dari skrip fase T
dengan blok salinan ditandai. Log ke `DL_FIRE_NASIONAL/penuh/patokan_iklim_u_{kab,nas}.log` dan
`DL_FIRE_MODIS/patokan_iklim_u_modis_{kab,nas}.log`.

## Hasil

Dijalankan 11 Nov 2026 sesudah komit 6eb44ef. Log: `DL_FIRE_NASIONAL/penuh/patokan_iklim_u_{kab,nas}.log`,
`DL_FIRE_MODIS/patokan_iklim_u_modis_{kab,nas}.log` (tidak masuk repo). Baris judul log VIIRS masih tertulis "fase T
VIIRS" karena teks cetak disalin dari skrip fase T; isinya fase U.

Pemeriksaan lolos:

- EF-L yang dihitung ulang sama dengan fase S (selisih maks 1e-16).
- SSE EF-L MODIS sama dengan fase N (selisih 0).
- ONI/DMI tidak ada yang hilang; semua kabupaten terpetakan ke provinsi.
- Cacah klaster: VIIRS per kabupaten 409 kab / 34 prov, nasional 498 / 34; MODIS per kabupaten 411 / 33, nasional
  498 / 34.

Koefisien EF-ES (VIIRS per kabupaten, lipatan 2020–2025):

- ONI di kemarau: Sumatra-Kalimantan +1,40 s.d. +1,70, wilayah lain +1,32 s.d. +1,50.
- ONI di luar kemarau: +0,15 s.d. +0,44.
- DMI di Sumatra-Kalimantan di luar kemarau: −2,51 s.d. −2,98.

Brier rata-rata tahun VIIRS:

| | kalender | EF-E | EF-ES | EF-L | EF-LE | EF-LES | EF-LD |
|---|---|---|---|---|---|---|---|
| per kabupaten | 0,04242 | 0,03191 | 0,03065 | 0,02900 | 0,02782 | 0,02762 | 0,02859 |
| nasional | 0,03615 | 0,03141 | 0,03024 | 0,02719 | 0,02672 | 0,02699 | 0,02635 |

### (a) Rujukan

Rata-rata rasio [dua tahap kab] [tahun t]; satu rasio [dua tahap kab]. **Tebal** = unggul dengan rata-rata rasio
(kedua selang). Vonis klaster provinsi sama dengan vonis rata-rata rasio di semua sel ini.

| Sel | VIIRS per kab | VIIRS nasional | MODIS maju per kab | MODIS maju nasional |
|---|---|---|---|---|
| EF-ES atas kalender | **+0,327 [+0,108, +0,545] [+0,012, +0,642]**; satu +0,277 | +0,179 [−0,006, +0,365] [−0,079, +0,436]; satu +0,164 | +0,073 [−0,044, +0,191] [−0,066, +0,211]; satu +0,060 | +0,014 [−0,058, +0,091] [−0,067, +0,096]; satu +0,022 |
| EF-ES atas EF-E | +0,053 [−0,042, +0,171] [−0,100, +0,207]; satu +0,040 | +0,045 [−0,073, +0,160] [−0,129, +0,219]; satu +0,037 | +0,014 [−0,064, +0,100] [−0,082, +0,111]; satu −0,008 | +0,025 [−0,035, +0,091] [−0,047, +0,097]; satu +0,028 |
| EF-L atas EF-ES | **+0,045 [+0,019, +0,078] [+0,015, +0,076]**; satu +0,054 [+0,015, +0,091] | **+0,083 [+0,040, +0,135] [+0,023, +0,142]**; satu +0,101 [+0,040, +0,161] | +0,064 [−0,002, +0,158] [−0,025, +0,152]; satu +0,058 [+0,005, +0,132] | +0,064 [−0,004, +0,152] [−0,022, +0,149]; satu +0,069 [+0,003, +0,161] |
| EF-LES atas EF-L | **+0,037 [+0,018, +0,064] [+0,012, +0,061]**; satu +0,047 [+0,016, +0,081] | +0,016 [−0,019, +0,048] [−0,026, +0,059]; satu +0,007 | +0,001 [−0,053, +0,038] [−0,047, +0,049]; satu +0,009 | +0,000 [−0,053, +0,044] [−0,053, +0,054]; satu +0,006 |
| EF-LES atas EF-ES | **+0,080** | **+0,099** | **+0,073** | **+0,073** |
| EF-L atas EF-E (ulang fase T) | +0,096 [+0,005, +0,216] [−0,052, +0,244]; satu +0,092 [−0,028, +0,263] | +0,127 [+0,034, +0,231] [−0,010, +0,265]; satu +0,134 [+0,026, +0,251] | +0,081 [−0,010, +0,196] [−0,034, +0,195]; satu +0,050 [−0,018, +0,150] | +0,090 [+0,015, +0,187] [−0,007, +0,187]; satu +0,096 [+0,019, +0,196] |

### (b) Model atas rujukan: vonis empat cara dan R64

Isi sel: rata-rata rasio, p tahun terkoreksi Holm untuk sel tebal, dan satu rasio [dua tahap kab] bila vonisnya
berbeda. Tanda: ‡ vonis satu rasio berbeda dari vonis rata-rata rasio; § vonis klaster provinsi berbeda. R64 = boleh
"beats" di abstrak, §1, §9.

VIIRS per kabupaten (Holm atas enam model per rujukan):

| Model | atas kalender | atas EF-P | atas EF-L | atas EF-LD | atas EF-E | atas EF-ES |
|---|---|---|---|---|---|---|
| RF | **+0,383** holm 0,052 | **+0,344** holm 0,035 R64 | +0,022 | +0,006 | **+0,123** holm 0,122 | +0,066; satu +0,101 [+0,006, +0,166] ‡ |
| GB | **+0,393** holm 0,052 | **+0,354** holm 0,035 R64 | +0,049 | +0,033 | **+0,146** holm 0,122 | +0,091; satu +0,116 [+0,010, +0,196] ‡ |
| GB tersetel | **+0,391** holm 0,052 | **+0,351** holm 0,035 R64 | +0,045 | +0,030 | **+0,146** holm 0,083 | +0,088 |
| MLP | **+0,289** holm 0,052 | +0,240; satu +0,197 [+0,054, +0,391] ‡ | −0,131 | −0,148; satu −0,114 [−0,343, −0,001] ‡ | −0,013 | −0,083 |
| GB + nasional | **+0,405** holm 0,052 | **+0,367** holm 0,035 R64 | +0,063 | +0,047 | **+0,156** holm 0,122 | **+0,105** holm 0,178 |
| GB + nasional + DC | **+0,404** holm 0,052 | **+0,367** holm 0,035 R64 | +0,064 | +0,048 | **+0,156** holm 0,122 | **+0,106** holm 0,173 |

VIIRS nasional:

| Model | atas kalender | atas EF-P | atas EF-L | atas EF-LD | atas EF-E | atas EF-ES |
|---|---|---|---|---|---|---|
| RF | **+0,293** holm 0,097 | **+0,296** holm 0,093 | +0,059; satu +0,047 [+0,002, +0,113] ‡ | +0,025 | **+0,182** holm 0,074 | **+0,139** holm 0,005 R64 |
| GB | **+0,272** holm 0,100 | **+0,275** holm 0,098 | +0,030 | −0,006 | +0,159; satu +0,149 [+0,021, +0,294] ‡ | +0,110 |
| GB tersetel | **+0,266** holm 0,100 | **+0,269** holm 0,098 | +0,016 | −0,020 | +0,146 | +0,097 |
| MLP | +0,224; satu +0,213 [+0,051, +0,414] ‡ | +0,227; satu +0,216 [+0,053, +0,418] ‡ | −0,047 | −0,085 | +0,094 | +0,038 |
| GB + nasional | **+0,293** holm 0,097 | **+0,296** holm 0,093 | +0,058 | +0,023 | **+0,179** holm 0,118 | **+0,137** holm 0,127 |
| GB + nasional + DC | **+0,297** holm 0,097 | **+0,300** holm 0,093 | +0,062 | +0,028 | **+0,183** holm 0,116 | **+0,141** holm 0,060 |

MODIS maju, GB-FH (keluarga satu anggota):

| Sasaran | atas kalender | atas EF-P | atas EF-L | atas EF-LD | atas EF-E | atas EF-ES |
|---|---|---|---|---|---|---|
| per kab | **+0,127** R64 | **+0,102** R64 | −0,062; satu +0,008 | −0,097; satu −0,024 | **+0,058** § R64 | +0,024; satu +0,066 [−0,004, +0,121] |
| nasional | **+0,137** R64 | **+0,145** R64 | **+0,050** R64 | +0,026 | **+0,141** R64 | **+0,115** R64 |

Klaster provinsi mengubah satu vonis rata-rata rasio: MODIS per kab GB-FH atas EF-E (prov [−0,011, +0,118]). Tidak ada
sel tebal Tabel 4 (ambang tahun latih) yang kehilangan vonis dengan klaster provinsi atau dengan satu rasio.

### Putusan

- **R63.** EF-ES tidak unggul atas EF-E di jalur mana pun.
  - EF-E tetap rujukan keadaan iklim di teks; EF-ES masuk S20 dengan satu kalimat di §6.2.
  - EF-E atas EF-L menurut R59: tak terbedakan, dan abstrak menulisnya dengan kata itu.
  - EF-ES atas EF-L (pelaporan, bukan aturan): di VIIRS EF-L unggul atas EF-ES untuk kedua sasaran; di MODIS tidak
    (titik +0,064, kedua selang menyentuh nol). Tambahan musim dan wilayah tidak mendekatkan rujukan iklim ke EF-L.
- **R64.** Kata "beats" di abstrak, §1 dan §9 hanya untuk:
  - VIIRS per kab: RF, GB, GB tersetel, GB + nasional, GB + nasional + DC atas EF-P;
  - VIIRS nasional: RF atas EF-ES;
  - MODIS maju per kab: GB-FH atas kalender, EF-P, EF-E;
  - MODIS maju nasional: GB-FH atas kalender, EF-P, EF-L, EF-E, EF-ES.

  Model VIIRS atas kalender lolos aturan naskah, tetapi tidak lolos Holm (per kab p terkoreksi 0,052; nasional
  0,097–0,100). Begitu pula model atas EF-E (per kab 0,083–0,122). Kalimat abstrak "the learned models beat each
  district's fire calendar" dan "per district, tree models beat it" diganti.
- **R65.** Tanda ‡ untuk sel Tabel 4 berikut:
  - MLP per kab atas EF-P dan EF-LD;
  - MLP nasional atas kalender dan EF-P;
  - RF nasional atas EF-L.

  Untuk S20: GB nasional atas EF-E (‡) dan GB-FH per kab atas EF-E (§). Tidak ada sel tebal yang kehilangan vonis.
- **R66.** EF-ES dan EF-LES di luar aturan. EF-LES atas EF-L unggul hanya di VIIRS per kab (+0,037), tidak di MODIS
  (+0,001); seperti R61 untuk EF-LE, ia hanya dilaporkan.
- **R67.** §4.3 menyebut keluarga Holm: 12 keluarga VIIRS (2 sasaran × 6 rujukan, enam model per keluarga), MODIS satu
  anggota. Klaim yang berubah: model belajar atas kalender VIIRS dan atas EF-E per kab tidak lagi ditulis "beat" di
  abstrak, §1 dan §9. Klaim "per district no learned model is shown to beat EF-L" tetap.
