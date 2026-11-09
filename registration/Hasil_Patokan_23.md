# Hasil Patokan 23 — fase Q revisi (bagian tingkat lawan selisih tingkat; bootstrap klaster provinsi)

**9 November 2026.** Lanjutan `Hasil_Patokan_22.md`, untuk `Naskah_Artikel2_Revisi.md`.
Pemicu: telaah internal pra-kirim kedua puluh tiga atas draf revisi (catatan telaah tidak dirilis).
Dua butir mayor butuh analisis: (1) metode X3 dan advokat setan X4 — naskah (§1, §9) menulis bahwa bagian tingkat
"depends on how far the test years' level departs from the training years'", tetapi hubungan itu tidak pernah
diuji; (2) metode X6 — selang dua tahap mengambil ulang kabupaten secara bebas, padahal kejadian berkelompok di
Kalimantan dan Sumatra, dan beberapa vonis nasional bertumpu pada batas bawah +0,000 sampai +0,013.

## Rancangan

**Q1 — tingkat yang diambil lawan galat tingkat kalender.** Untuk tiap tahun uji y, dua besaran:

- L_y = BSS_y(GB atas kalender, mentah) − BSS_y(GB atas kalender, sesudah geser per tahun), yaitu bagian
  keunggulan tahun itu yang hilang bila model dan kalender sama-sama digeser ke laju teramati tahun itu (geser
  per tahun definisi I4, tanpa offset periode lebih dulu; satuan BSS).
- d_y = |logit(rata-rata prakiraan kalender tahun y) − logit(laju teramati tahun y)|, galat tingkat kalender tahun
  itu. Kalender adalah frekuensi tahun latih, jadi d_y mengukur seberapa jauh tingkat tahun uji dari tahun latih
  pada kabupaten dan bulan yang sama. Tanda (logit kalender − logit teramati) dicatat sebagai deskriptif.

Rancangan: (i) VIIRS S1 2020–2025 dengan ambang tahun latih (data fase P / G2, GB benih 42), 6 tahun; (ii) MODIS
maju 2008–2019 (data fase O, GB-FH, SSE rata-rata enam benih, tiap benih digeser sendiri), 12 tahun. Kedua sasaran.
Statistik: kemiringan OLS L atas d dengan intersep per rancangan (gabungan 18 tahun per sasaran), selang bootstrap
tahun dalam rancangan (10.000 ulangan, benih 2026, persentil 2,5–97,5); Spearman ρ per rancangan dan gabungan
(deskriptif).

**Q2 — bootstrap klaster provinsi.** Tahap pertama dua tahap mengambil ulang 34 provinsi dengan pengembalian
(semua kabupaten provinsi terambil ikut, dengan bobot sebanyak provinsi itu terambil) alih-alih kabupaten; tahap
kedua tetap mengambil ulang tahun. Provinsi dari kolom `provinsi` `panel_nasional_bersih.csv`. 1.000 ulangan, benih
2026; untuk perbandingan yang batas bawahnya dalam ±0,01 dari nol, juga 10.000 ulangan. Statistik sama dengan
naskah (rata-rata rasio tahunan BSS). Selang tahun (t, df n−1) tidak berubah dan tidak dihitung ulang.

Perbandingan:
- VIIRS S1, ambang tahun latih (data fase P), kedua sasaran: GB dan RF atas kalender; EF-L atas kalender; GB dan RF
  atas EF-L.
- MODIS maju (data fase O), kedua sasaran: GB-FH atas kalender, EF-P, EF-L, EF-LD; GB-FH + DC atas kalender, EF-P,
  EF-L, EF-LD; EF-L atas kalender; EF-LD atas EF-L.

Skrip: `patokan_tingkat_q.py` (salinan `patokan_tingkat_p.py` tanpa perubahan bagian data, ditambah Q1 dan Q2 VIIRS)
dan `patokan_modis_q.py` (salinan bagian data dan fungsi `patokan_modis_o.py` tanpa perubahan, ditambah Q1 dan Q2
MODIS). Pemeriksaan: BSS dan batas dua tahap kabupaten harus sama dengan fase P dan fase O.

## Aturan keputusan

- **R50 (Q1).** Per kabupaten: bila kemiringan gabungan > 0 dan selang 95 % tidak memuat nol, kalimat §1/§9 bahwa
  bagian tingkat bergantung pada selisih tingkat latih–uji tetap, ditulis dengan kemiringan dan selangnya, dan
  "central measurement" §9 menjadi hubungan itu (bagian per rancangan sebagai contoh). Bila tidak, kalimat
  ketergantungan dihapus dari §1, §7 dan §9 dan diganti "by a share that differs between designs" dengan rentang
  bagian, tanpa penjelasan sebab. Nasional dilaporkan dengan aturan sama untuk kalimat nasional mana pun yang
  menyebut ketergantungan.
- **R51 (Q2).** Untuk tiap perbandingan yang kini "beats" (batas bawah dua tahap kabupaten > 0 dan selang tahun
  > 0): bila batas bawah dua tahap provinsi ≤ 0 (1.000 ulangan; bila dalam ±0,01 dari nol, 10.000 ulangan yang
  menentukan), naskah menulis di setiap tempat vonis itu disebut bahwa ia tidak bertahan bila provinsi diambil
  ulang, dan abstrak tidak menyatakannya sebagai kemenangan. Khusus: bila GB-FH atas EF-L nasional maju gagal,
  "nationally it beats EF-L" keluar dari abstrak, §1, judul §6.3 dan §9; bila GB-FH + DC atas EF-LD nasional maju
  gagal, lolos R46a tetap dicatat (aturannya kabupaten) tetapi disebut tidak kokoh atas ketergantungan spasial.
  Vonis nol tidak berubah kata; selang provinsi dilaporkan di suplemen (Tabel S16g baru) dan satu kalimat §6.
- **R52.** Bila satu vonis abstrak berubah karena R50 atau R51, panel 24 dijalankan; bila tidak, panel 24 tetap
  dijalankan sesudah putaran teks mayor panel 23 (bukan karena fase Q).

## Hasil (dijalankan 9 Nov 2026 sesudah komit c028252 dan 3fa2052)

Log: `DL_FIRE_NASIONAL/penuh/patokan_tingkat_q_{kab,nas}.log`, `DL_FIRE_MODIS/patokan_modis_q_{kab,nas}.log`.
Pemeriksaan: BSS VIIRS = fase P/G2 (GB +0,3929 / +0,2725); batas dua tahap kabupaten MODIS benih 2026 dalam rentang
lima benih fase O.

**Q1.** L dan d per tahun (GB atas kalender; VIIRS ambang tahun latih, MODIS maju GB-FH enam benih):

| Rancangan, sasaran | L per tahun | d per tahun | Spearman ρ |
|---|---|---|---|
| VIIRS 2020–2025, kab | +0,386 +0,580 +0,634 −0,023 +0,220 +0,306 | 1,33 1,85 2,12 0,30 0,98 1,22 | +1,00 |
| MODIS 2008–2019, kab | +0,186 −0,036 +0,280 +0,033 −0,024 +0,131 −0,007 −0,018 +0,331 +0,269 +0,065 −0,009 | 0,62 0,16 1,51 0,23 0,03 0,54 0,36 0,75 1,33 1,40 0,41 0,21 | +0,82 |
| VIIRS 2020–2025, nas | −0,031 +0,384 +0,474 −0,039 +0,039 +0,114 | 0,96 1,23 1,30 0,16 0,50 0,58 | +0,83 |
| MODIS 2008–2019, nas | +0,048 +0,018 +0,223 +0,000 +0,006 +0,038 +0,023 −0,000 +0,235 +0,327 +0,058 −0,010 | 0,36 0,15 0,98 0,01 0,09 0,25 0,24 0,36 0,97 0,98 0,39 0,06 | +0,86 |

Kemiringan gabungan L atas d (intersep per rancangan, 18 tahun): **per kabupaten +0,292 [+0,237, +0,341]**, **nasional
+0,335 [+0,199, +0,411]** (bootstrap tahun dalam rancangan, 10.000). Spearman gabungan +0,91 / +0,78. Tanda: semua
tahun dengan L besar adalah tahun yang kalendernya terlalu tinggi (tahun uji lebih sepi daripada tahun latih); pada
tahun yang lebih ramai daripada tahun latih (VIIRS 2023; MODIS 2009, 2012, 2014, 2015, 2019) L di sekitar nol atau
negatif, per kabupaten −0,036 sampai −0,007.

**Q2.** Batas dua tahap 95 %, kabupaten lawan klaster provinsi (1.000 ulangan, benih 2026; 10.000 bila batas
provinsi dalam ±0,01 dari nol):

| Perbandingan | Rata-rata | Kabupaten | Provinsi | Provinsi 10.000 | Tahun t |
|---|---|---|---|---|---|
| VIIRS kab: GB / RF / EF-L atas kalender | +0,393 / +0,384 / +0,362 | +0,212 / +0,203 / +0,161 | +0,214 / +0,200 / +0,144 | | +0,133 / +0,132 / +0,077 |
| VIIRS kab: GB / RF atas EF-L | +0,049 / +0,022 | −0,033 / −0,061 | −0,048 / −0,091 | | −0,042 / −0,058 |
| VIIRS nas: GB / RF / EF-L atas kalender | +0,273 / +0,293 / +0,256 | +0,118 / +0,148 / +0,123 | +0,114 / +0,147 / +0,113 | | +0,032 / +0,077 / +0,060 |
| VIIRS nas: GB / RF atas EF-L | +0,030 / +0,059 | −0,116 / +0,008 | −0,134 / −0,003 | RF −0,007 | −0,166 / −0,007 |
| MODIS kab: GB-FH atas kalender / EF-P / EF-L / EF-LD | +0,127 / +0,102 / −0,062 / −0,097 | +0,066 / +0,047 / −0,210 / −0,252 | +0,066 / +0,048 / −0,253 / −0,312 | | |
| MODIS kab: GB-FH + DC atas kalender / EF-P / EF-L / EF-LD | +0,151 / +0,128 / −0,016 / −0,048 | +0,078 / +0,060 / −0,125 / −0,150 | +0,077 / +0,058 / −0,161 / −0,183 | | |
| MODIS kab: EF-L atas kalender; EF-LD atas EF-L | +0,141; +0,030 | +0,038; +0,010 | +0,030; +0,007 | EF-LD +0,007 | +0,008; +0,007 |
| MODIS nas: GB-FH atas kalender / EF-P / EF-L / EF-LD | +0,137 / +0,145 / +0,050 / +0,026 | +0,080 / +0,077 / +0,008 / −0,014 | +0,070 / +0,072 / +0,002 / −0,021 | EF-L **+0,005** | +0,074 / +0,064 / +0,010 / −0,012 |
| MODIS nas: GB-FH + DC atas kalender / EF-P / EF-L / EF-LD | +0,146 / +0,154 / +0,061 / +0,038 | +0,086 / +0,082 / +0,018 / +0,003 | +0,074 / +0,078 / +0,010 / −0,004 | EF-L +0,013; EF-LD **−0,002** | +0,078 / +0,068 / +0,016 / +0,004 |
| MODIS nas: EF-L atas kalender; EF-LD atas EF-L | +0,087; +0,024 | +0,025; +0,003 | +0,022; −0,000 | EF-LD −0,000 | +0,010; −0,000 |

**R50.** Kemiringan > 0 dengan selang tanpa nol untuk kedua sasaran → kalimat ketergantungan §1/§9 tetap, ditulis
dengan kemiringan dan selang; "central measurement" §9 menjadi hubungan itu.
**R51.** Semua vonis "beats" bertahan, kecuali **GB-FH + DC atas EF-LD nasional maju**: batas provinsi −0,004 (10.000:
−0,002) → lolos R46a tetap dicatat, tetapi disebut tidak bertahan bila provinsi diambil ulang, dan abstrak tidak
menyatakannya sebagai kemenangan. GB-FH atas EF-L nasional maju bertahan tipis (provinsi 10.000 +0,005). RF atas EF-L
nasional VIIRS bukan vonis "beats" sebelumnya (selang tahun memuat nol) dan tetap tidak. EF-LD atas EF-L nasional maju
juga bukan "beats" sebelumnya.
**R52.** Abstrak berubah (lolos tipis GB-FH + DC keluar sebagai kemenangan) → panel 24 dijalankan.
