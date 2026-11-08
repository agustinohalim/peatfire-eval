# Hasil Tolok Ukur Nasional 15 — fase J revisi (prakiraan MODIS maju, penyangga jeda, beda V3 − V1, gambut)

**7 November 2026.** Lanjutan `Hasil_Patokan_14.md` (fase I), revisi PeatFireBench (dalam naskah revisi
bernama IndoFireBench). Pemicu: telaah internal pra-kirim kesebelas atas draf revisi (catatan telaah tidak
dirilis). Temuan yang perlu analisis: tidak ada desain MODIS yang merupakan prakiraan, semua leave-one-year-out
dengan tahun sesudah tahun uji di latih, dan desain utama melatih baris tahun u+1 yang jedanya memuat hitungan
tahun uji; "tahun-tahun belakangan, bukan panjang" bertumpu pada beda signifikansi, bukan beda yang diestimasi;
hasil pokok tidak ditunjukkan untuk kabupaten bergambut.

## Rancangan fase J (ditulis 7 Nov 2026, SEBELUM skrip fase J dijalankan)

Semua analisis pasca-hoc. Data, panel MODIS, F, fitur, parameter GB, enam benih {42, 1, 2, 3, 4, 5}, EF-P
(enam calon, Brier LOO tahun latih) dan EF-L (bentuk tetap, tiga koefisien) sama dengan fase I
(`patokan_modis_i.py`). Skrip baru `patokan_modis_j.py` (J1, J1b, J2; satu sasaran per jalan) dan
`patokan_tingkat_j.py` (J3). Dua tahap 1.000 ulangan, benih bootstrap B.SEED; t berpasangan dua sisi; MDD80.

**J1 Prakiraan maju MODIS (asal bergulir), desain R.** Tahun uji 2008–2019 (12 tahun, df 11). Tiap lipatan:
latih = semua tahun MODIS **sebelum** tahun uji (2003 … u−1, 5 sampai 16 tahun); ambang dan kabupaten yang
dinilai dari tahun latih saja (seperti V2). Tak ada tahun sesudah tahun uji di latih, ambang, EF-P, EF-L
atau fitur, jadi ini prakiraan sungguhan dan bebas kebocoran jeda. Tahun MODIS 2012–2019 belum pernah menjadi
tahun uji MODIS; tahun yang sama (sebagai kalender) adalah tahun latih VIIRS dan 2019 tahun uji VIIRS S2, dan
bentuk EF-L dipilih sesudah tahun uji VIIRS 2020–2025 terlihat. Tahun 2016–2019 pertama kali punya tahun
sesudah 2015 di latih.

**J1b Penyangga jeda (LOYO), desain V5 dan V6.** V5 = V0 tanpa tahun u+1 di latih (21 tahun; ambang 2003–2025);
V6 = V2 tanpa u+1 (ambang dan kabupaten dari tahun latih). Untuk u = 2011, u+1 = 2012. Membuang u+1 menghapus
setiap baris latih yang fitur jedanya (1, 2, 3, 12 bulan, keadaan bulan lalu, anomali nasional) memuat tahun
uji.

**J2 Beda V3 − V1 diestimasi.** Lipatan V1 (latih 2003–2015 tanpa u) dan V3 (12 tahun acak dari 22, lima
undian, sama persis dengan fase I) dijalankan ulang pada baris uji yang sama (ambang 2003–2025, label identik,
diperiksa). Per tahun: BSS GB atas kalender, EF-P, EF-L untuk tiap desain (GB rata-rata benih; V3 juga
rata-rata undian; rujukan V3 rata-rata undian), dan beda V3 − V1, dengan dua tahap bersama (satu bootstrap
untuk kedua desain) dan t95 (df 8). Juga BSS Brier GB V3 atas GB V1 langsung.

**J3 Kabupaten bergambut (VIIRS).** `patokan_tingkat_i.py` dibatasi ke kabupaten dengan gambut_fraksi > 0
(strata Suplemen S7; 125 kabupaten dinilai per kabupaten), prediksi tersimpan fase E, enam tahun biasa
2020–2025 (n = 6, df 5): sel Tabel 4 (BSS model atas kalender, EF-P, EF-L; dua tahap, t95, MDD80) dan
pemecahan offset periode / tingkat tahun untuk GB dan RF, geser dihitung di dalam strata. Dua sasaran.

**R38.** (a) **Prakiraan maju**: bila di R GB (rata-rata benih) mengalahkan EF-L di kedua selang untuk suatu
sasaran, naskah menulis bahwa keunggulan atas EF-L bertahan dalam prakiraan maju untuk sasaran itu dan
menjadikan R desain MODIS pertama di Tabel 5 dan Fig. 3; bila tidak, abstrak, §1, §6.3, §7 dan §9 menulis
bahwa keunggulan atas EF-L hanya ditunjukkan dalam validasi silang LOYO dan tidak ditunjukkan dalam prakiraan
maju (dengan titik dan MDD80). Vonis "ketiga rujukan" R dilaporkan apa adanya.
(b) **Penyangga**: bila GB atas EF-L lolos kedua selang di V5 dan V6 untuk sasaran yang lolos di V0/V2, naskah
menulis bahwa hasil tidak bergantung pada kebocoran jeda dan kalimat kebocoran di §6.3 dan Keterbatasan
diganti hasil ini; bila salah satu gagal, naskah menulis bahwa sebagian keunggulan LOYO dapat berasal dari
baris latih yang memuat tahun uji, dan V5/V6 masuk Tabel 5. Vonis "ketiga rujukan" per kabupaten V6
menggantikan V2 sebagai satu-satunya lolos bila V2 lolos dan V6 tidak.
(c) **Beda**: bila beda V3 − V1 dalam BSS GB atas EF-L (per kabupaten) dan atas EF-P (nasional) mengecualikan
nol di kedua selang, naskah menulis bahwa latih dengan tahun sesudah 2015 lebih baik daripada latih
sebelum 2016 dengan panjang yang sama, diestimasi; bila tidak, naskah menulis bahwa beda tidak dapat
dibedakan dari nol dan menghapus tafsir "tahun belakangan lebih penting daripada panjang", menyisakan
deskripsi dua desain.
(d) **Gambut**: bila di kabupaten bergambut GB atas EF-L per kabupaten berbeda tanda dari semua kabupaten
(+0,067) atau bagian tingkat total (periode + tahun) GB atas kalender < 0,5, hasil gambut masuk teks utama
§6.2 dengan satu kalimat dan Tabel 4 mendapat baris gambut; bila tidak, satu kalimat di §6.2 menyebut bahwa
hasil sama di kabupaten bergambut, angka di suplemen (S7).
(e) Semua angka naskah dari log fase J ini; tak ada kalimat baru tanpa baris log.

Waktu: fase I ±100 s per varian per sasaran; fase J kira-kira 12 menit per sasaran (MODIS) dan 2 menit (J3).

## Hasil (7 Nov 2026)

Log: `DL_FIRE_MODIS/patokan_modis_j_{kab,nas}.log` (J1, J1b, J2), `DL_FIRE_NASIONAL/penuh/patokan_tingkat_j_{kab,nas}.log`
(J3). Kedua jalan MODIS sampai "selesai". Pemeriksaan: V5 dan V6 nasional identik sampai digit terakhir karena
ambang nasional (kuantil 0,90 hitungan bulat) = 9 di setiap lipatan, dengan atau tanpa u+1 dan dengan ambang
2003–2025 (diperiksa terpisah; di V2 fase I lipatan 2010 berambang 10, sebab itu V0 ≠ V2 di fase I).

### J1 Prakiraan maju MODIS (R, uji 2008–2019, n = 12, df 11)

Latih 5 tahun (2008) sampai 16 tahun (2019). EF-P memilih EF-1m halus di semua lipatan per kabupaten. Koefisien
anomali nasional EF-L per kabupaten +0,60 sampai +0,83. Prevalensi per kabupaten: 2,1 % (2010), 2,3 % (2016),
2,1 % (2017), selainnya 4,8–15,9 %.

| BSS | Per kabupaten | Nasional |
|---|---|---|
| EF-L atas kalender | **+0,141** [+0,038, +0,257] t [+0,008, +0,275]; MDD80 0,187 | **+0,087** [+0,025, +0,166] t [+0,010, +0,165]; 0,108 |
| EF-P atas kalender | **+0,029** [+0,016, +0,047] t [+0,011, +0,048]; 0,026 | −0,016 [−0,050, +0,012] t [−0,049, +0,018]; 0,047 |
| GB atas kalender | **+0,127** [+0,066, +0,196] t [+0,053, +0,201]; 0,103 | **+0,137** [+0,080, +0,203] t [+0,074, +0,199]; 0,087 |
| GB atas EF-P | **+0,102** [+0,047, +0,163] t [+0,037, +0,168]; 0,092 | **+0,145** [+0,077, +0,227] t [+0,064, +0,226]; 0,113 |
| GB atas EF-L | −0,061 [−0,210, +0,049] t [−0,208, +0,085]; 0,204 | **+0,050** [+0,008, +0,088] t [+0,010, +0,090]; 0,056 |
| GB mengalahkan ketiganya | tidak (gagal EF-L); 0/6 benih | **ya; 6/6 benih** |

Rentang benih GB atas EF-L: per kabupaten −0,069 .. −0,055; nasional +0,044 .. +0,053. GB atas EF-L per
kabupaten per tahun 2008–2019: +0,035, −0,003, −0,468, +0,032, +0,059, +0,087, +0,113, +0,089, −0,178, −0,583,
+0,025, +0,054 (di atas nol 8 dari 12 tahun; tiga tahun di bawah −0,1 adalah tiga tahun tersepi 2010, 2016,
2017). Nasional per tahun: −0,011, −0,003, +0,084, +0,073, +0,067, +0,131, +0,065, +0,104, +0,006, −0,095,
+0,093, +0,081.

**R38a.** Nasional: GB mengalahkan EF-L, dan ketiga rujukan, di kedua selang dalam prakiraan maju → naskah
menulis bahwa keunggulan nasional bertahan dalam prakiraan maju; R menjadi desain MODIS pertama di Tabel 5 dan
Fig. 3. Per kabupaten: tidak → keunggulan atas EF-L per kabupaten hanya ditunjukkan dalam validasi silang LOYO,
tidak dalam prakiraan maju (−0,061, MDD80 0,204). Dalam prakiraan maju per kabupaten EF-L di atas kalender
mentah di kedua selang (+0,141), berbeda dengan LOYO (+0,059, tidak).

### J1b Penyangga jeda (V5 = V0 tanpa u+1; V6 = V2 tanpa u+1; n = 9)

| BSS | V5 per kab | V6 per kab | V5 = V6 nasional |
|---|---|---|---|
| EF-L atas kalender | +0,057 [−0,019, +0,162] t [−0,059, +0,172] | +0,053 [−0,022, +0,153] t [−0,060, +0,167] | **+0,098** [+0,042, +0,154] t [+0,030, +0,166] |
| GB atas kalender | +0,110 [+0,038, +0,212] t [−0,002, +0,222] | **+0,110** [+0,037, +0,210] t [+0,003, +0,217] | **+0,174** [+0,121, +0,228] t [+0,113, +0,236] |
| GB atas EF-P | **+0,110** [+0,039, +0,210] t [+0,001, +0,219] | **+0,110** [+0,038, +0,207] t [+0,005, +0,214] | **+0,146** [+0,092, +0,202] t [+0,080, +0,212] |
| GB atas EF-L | **+0,057** [+0,030, +0,085] t [+0,026, +0,089] | **+0,059** [+0,030, +0,089] t [+0,029, +0,091] | **+0,083** [+0,045, +0,122] t [+0,038, +0,127] |
| GB mengalahkan ketiganya | tidak (gagal kalender); 0/6 | **ya; 6/6 benih** | **ya; 6/6 benih** |

**R38b.** GB atas EF-L lolos kedua selang di V5 dan V6 untuk kedua sasaran → hasil LOYO atas EF-L tidak
bergantung pada kebocoran jeda; kalimat kebocoran di §6.3 dan Keterbatasan diganti hasil ini. V2 lolos ketiganya
per kabupaten dan V6 juga (6/6 benih, batas bawah t atas kalender +0,003) → klausul pengganti tidak terpicu.

### J2 Beda V3 − V1 (baris uji sama, n = 9, df 8)

| BSS GB | Per kab V3 | Per kab V1 | Per kab beda | Nasional V3 | Nasional V1 | Nasional beda |
|---|---|---|---|---|---|---|
| atas kalender | +0,105 | +0,067 | +0,037 [−0,016, +0,099] t [−0,031, +0,106] | +0,144 | +0,078 | **+0,066** [+0,023, +0,108] t [+0,017, +0,115] |
| atas EF-P | +0,113 | +0,057 | +0,055 [+0,003, +0,117] t [−0,011, +0,122] | +0,135 | +0,065 | **+0,070** [+0,021, +0,114] t [+0,014, +0,125] |
| atas EF-L | +0,062 | −0,024 | +0,085 [+0,017, +0,184] t [−0,021, +0,191]; MDD80 0,147 | +0,076 | +0,035 | **+0,041** [+0,016, +0,065] t [+0,014, +0,068] |
| Brier GB V3 atas GB V1 | | | +0,037 [−0,046, +0,161] t [−0,096, +0,170] | | | −0,009 [−0,066, +0,047] t [−0,084, +0,065] |

V1 nasional di sini: GB atas EF-L +0,035 dan atas EF-P +0,065 (t [−0,007, +0,137]), sama dengan fase I.

**R38c.** Rancangan menulis syarat "per kabupaten atas EF-L **dan** nasional atas EF-P". Nasional terpenuhi
(+0,070 atas EF-P, juga +0,041 atas EF-L dan +0,066 atas kalender, kedua selang); per kabupaten tidak (+0,085,
selang tahunan memuat nol). Dibaca per sasaran (tafsir, dicatat): nasional, latih dengan tahun sesudah 2015
memberi BSS GB atas rujukan yang lebih tinggi daripada latih sebelum 2016 dengan panjang sama, diestimasi;
per kabupaten beda tidak dapat dibedakan dari nol, dan tafsir "tahun belakangan lebih penting daripada panjang"
dihapus untuk per kabupaten. **Catatan penting:** Brier GB sendiri tidak berbeda antara V3 dan V1 (nasional
−0,009, per kab +0,037, keduanya memuat nol), jadi beda BSS berasal dari rujukan: kalender, EF-P dan EF-L yang
dilatih dengan 12 tahun acak (termasuk tahun sesudah 2015) menilai tahun 2003–2011 lebih buruk daripada yang
dilatih pada 2003–2015, sementara GB tidak memburuk. Naskah harus menulis ini, bukan "tahun belakangan membuat
GB lebih baik".

### J3 Kabupaten bergambut (VIIRS, gambut_fraksi > 0, n = 6)

Per kabupaten 125 kabupaten, 9.000 baris, prevalensi 3,73 %; nasional 131 kabupaten, 9.432 baris, 10,97 %.

| | Per kab (EF-1m) | Nasional (EF-3m) |
|---|---|---|
| GB atas kalender | +0,421 (kedua selang) | +0,243 (kedua selang) |
| RF atas kalender | +0,389 (kedua) | +0,259 (kedua) |
| EF-L atas kalender | +0,380 (kedua) | +0,215 (kedua) |
| GB atas EF-L | +0,066 [−0,023, +0,158] t [−0,047, +0,179] | +0,034 [−0,100, +0,149] t [−0,135, +0,202] |
| RF atas EF-L | +0,004 | +0,060 [+0,004, +0,127] t [−0,003, +0,123] |
| GB + hitungan nasional atas EF-L | +0,091 [−0,007, +0,187] | +0,071 [−0,011, +0,152] |
| Logistik lokal atas EF-L | **−0,224** [−0,436, −0,045] t [−0,437, −0,011] | +0,017 |
| Bagian periode / tahun / total, GB atas kalender | 0,69 [0,44, 0,91] / 0,10 [−0,02, 0,22] / 0,79 | 0,63 [0,07, 1,19] / 0,09 [−0,33, 0,50] / 0,73 |
| Bagian periode / tahun / total, RF atas kalender | 0,71 / 0,12 / 0,83 | 0,58 / 0,21 / 0,78 |
| Geser periode logit kalender / GB | −1,373 / −0,451 | −1,049 / −0,359 |

**R38d.** GB atas EF-L per kabupaten bertanda sama (+0,066 vs +0,067 semua kabupaten) dan bagian tingkat total
0,79 ≥ 0,5 → satu kalimat di §6.2, angka di Suplemen S7. Nasional EF-P tidak di atas EF-3m di kabupaten
bergambut (−0,007).

Pembulatan mengikuti program pembuat tabel (`f"{x:+.3f}"`), bukan pembulatan tangan.

### Bacaan

Prakiraan maju menguatkan bagian pertama hasil pokok per kabupaten: dalam 12 tahun MODIS yang diramalkan hanya
dari tahun sebelumnya, EF-L di atas kalender mentah di kedua selang dan GB tidak ditunjukkan di atas EF-L
(rata-rata −0,061, ditarik tiga tahun tersepi; di atas nol 8 dari 12 tahun). Keunggulan per kabupaten atas EF-L
(+0,06) hanya muncul dalam LOYO, dan bertahan dengan penyangga jeda. Nasional GB mengalahkan ketiga rujukan di
setiap desain dengan tahun sesudah 2015 di latih, dengan penyangga jeda, dan dalam prakiraan maju (+0,050 atas
EF-L, 6/6 benih). Beda V3 − V1 berasal dari rujukan yang memburuk bila dilatih lintas rezim, bukan dari GB yang
membaik. Kabupaten bergambut: hasil sama.

### Tabel dari program (kanonik untuk naskah)

Dibuat `tabel_j.py` dari log fase I dan J (dibaca dari log, bukan diketik). Tabel ketikan di atas dibulatkan dari
angka log empat desimal; beberapa batas selang di sana berbeda 0,001 dari tabel ini. Naskah memakai tabel ini.
Sel penuh: titik [dua tahap] [t95]; MDD80. Sel Tabel 5: titik (batas bawah terendah dari kedua selang).

**Tabel 5 naskah**

| Comparison | Forward, 2008–2019 | All other years | Training-year thresholds, lag buffer | 12 random years | Before 2016 |
|---|---|---|---|---|---|
| *Per-district target* | | | | | |
| EF-L over calendar | **+0.141 (+0.008)** | +0.059 (−0.057) | +0.053 (−0.059) | +0.048 (−0.069) | +0.073 (−0.062) |
| EF-P over calendar | **+0.029 (+0.011)** | +0.001 (−0.005) | +0.001 (−0.005) | −0.007 (−0.023) | **+0.012 (+0.001)** |
| Gradient boosting over calendar | **+0.127 (+0.053)** | +0.115 (−0.002) | **+0.110 (+0.003)** | +0.105 (−0.016) | +0.067 (−0.031) |
| Gradient boosting over EF-P | **+0.102 (+0.037)** | **+0.115 (+0.001)** | **+0.110 (+0.005)** | +0.113 (−0.002) | +0.057 (−0.033) |
| Gradient boosting over EF-L | −0.061 (−0.210) | **+0.061 (+0.031)** | **+0.059 (+0.029)** | **+0.061 (+0.032)** | −0.024 (−0.114) |
| Gradient boosting beats all three | no; fails EF-L; 0/6 seeds | no; fails calendar; 1/6 seeds | yes; 6/6 seeds | no; fails calendar and EF-P; 0/6 seeds | no; fails all three; 0/6 seeds |
| *National target* | | | | | |
| EF-L over calendar | **+0.087 (+0.010)** | **+0.101 (+0.033)** | **+0.098 (+0.030)** | **+0.071 (+0.009)** | +0.044 (−0.005) |
| EF-P over calendar | −0.016 (−0.050) | +0.033 (−0.002) | +0.031 (−0.002) | +0.010 (−0.014) | +0.010 (−0.031) |
| Gradient boosting over calendar | **+0.137 (+0.074)** | **+0.170 (+0.104)** | **+0.174 (+0.113)** | **+0.144 (+0.084)** | **+0.078 (+0.029)** |
| Gradient boosting over EF-P | **+0.145 (+0.064)** | **+0.140 (+0.067)** | **+0.146 (+0.080)** | **+0.135 (+0.073)** | +0.065 (−0.007) |
| Gradient boosting over EF-L | **+0.050 (+0.008)** | **+0.076 (+0.032)** | **+0.083 (+0.038)** | **+0.076 (+0.027)** | **+0.035 (+0.007)** |
| Gradient boosting beats all three | yes; 6/6 seeds | yes; 6/6 seeds | yes; 6/6 seeds | yes; 6/6 seeds | no; fails EF-P; 0/6 seeds |

**Tabel S16a suplemen**

| Comparison | Forward, 2008–2019 | All other years, lag buffer | Training-year thresholds, lag buffer |
|---|---|---|---|
| *Per-district target* | | | |
| EF-L over calendar | +0.141 [+0.038, +0.257] [+0.008, +0.275]; 0.186 | +0.057 [−0.019, +0.162] [−0.059, +0.172]; 0.160 | +0.053 [−0.022, +0.153] [−0.059, +0.167]; 0.157 |
| EF-P over calendar | +0.029 [+0.016, +0.047] [+0.011, +0.048]; 0.026 | +0.001 [−0.003, +0.007] [−0.005, +0.007]; 0.008 | +0.001 [−0.003, +0.006] [−0.005, +0.007]; 0.008 |
| Gradient boosting over calendar | +0.127 [+0.066, +0.196] [+0.053, +0.201]; 0.103 | +0.110 [+0.038, +0.212] [−0.002, +0.222]; 0.155 | +0.110 [+0.037, +0.210] [+0.003, +0.217]; 0.149 |
| Gradient boosting over EF-P | +0.102 [+0.047, +0.163] [+0.037, +0.168]; 0.092 | +0.110 [+0.038, +0.210] [+0.001, +0.219]; 0.151 | +0.110 [+0.038, +0.207] [+0.005, +0.214]; 0.145 |
| Gradient boosting over EF-L | −0.061 [−0.210, +0.049] [−0.208, +0.085]; 0.204 | +0.057 [+0.030, +0.085] [+0.026, +0.088]; 0.043 | +0.059 [+0.030, +0.088] [+0.029, +0.091]; 0.043 |
| Gradient boosting beats all three | no; fails EF-L; 0/6 seeds | no; fails calendar; 0/6 seeds | yes; 6/6 seeds |
| *National target* | | | |
| EF-L over calendar | +0.087 [+0.025, +0.166] [+0.010, +0.165]; 0.108 | +0.098 [+0.042, +0.154] [+0.030, +0.166]; 0.094 | +0.098 [+0.042, +0.154] [+0.030, +0.166]; 0.094 |
| EF-P over calendar | −0.016 [−0.050, +0.012] [−0.049, +0.018]; 0.046 | +0.031 [+0.003, +0.056] [−0.002, +0.065]; 0.046 | +0.031 [+0.003, +0.056] [−0.002, +0.065]; 0.046 |
| Gradient boosting over calendar | +0.137 [+0.080, +0.203] [+0.074, +0.199]; 0.087 | +0.174 [+0.120, +0.228] [+0.113, +0.236]; 0.085 | +0.174 [+0.120, +0.228] [+0.113, +0.236]; 0.085 |
| Gradient boosting over EF-P | +0.145 [+0.077, +0.227] [+0.064, +0.226]; 0.113 | +0.146 [+0.092, +0.202] [+0.080, +0.212]; 0.091 | +0.146 [+0.092, +0.202] [+0.080, +0.212]; 0.091 |
| Gradient boosting over EF-L | +0.050 [+0.008, +0.088] [+0.009, +0.090]; 0.056 | +0.083 [+0.045, +0.122] [+0.038, +0.127]; 0.061 | +0.083 [+0.045, +0.122] [+0.038, +0.127]; 0.061 |
| Gradient boosting beats all three | yes; 6/6 seeds | yes; 6/6 seeds | yes; 6/6 seeds |

**Tabel S16b suplemen**

| Brier skill | 12 random years | Before 2016 | Difference |
|---|---|---|---|
| *Per-district target* | | | |
| Gradient boosting over calendar | +0.105 [+0.026, +0.214] [−0.016, +0.226]; 0.168 | +0.067 [−0.002, +0.157] [−0.031, +0.165]; 0.136 | +0.037 [−0.016, +0.099] [−0.031, +0.106]; 0.095 |
| Gradient boosting over EF-P | +0.113 [+0.036, +0.218] [−0.002, +0.227]; 0.158 | +0.057 [−0.008, +0.141] [−0.033, +0.148]; 0.126 | +0.055 [+0.003, +0.117] [−0.011, +0.122]; 0.092 |
| Gradient boosting over EF-L | +0.061 [+0.036, +0.090] [+0.032, +0.091]; 0.040 | −0.024 [−0.110, +0.032] [−0.114, +0.066]; 0.124 | +0.085 [+0.017, +0.184] [−0.021, +0.191]; 0.147 |
| Brier score of gradient boosting, 12 random years over before 2016 | | | +0.037 [−0.046, +0.161] [−0.096, +0.170]; 0.184 |
| *National target* | | | |
| Gradient boosting over calendar | +0.144 [+0.092, +0.197] [+0.084, +0.204]; 0.083 | +0.078 [+0.040, +0.118] [+0.029, +0.127]; 0.067 | +0.066 [+0.023, +0.108] [+0.017, +0.115]; 0.068 |
| Gradient boosting over EF-P | +0.135 [+0.083, +0.187] [+0.073, +0.196]; 0.085 | +0.065 [+0.012, +0.120] [−0.007, +0.137]; 0.099 | +0.070 [+0.021, +0.114] [+0.014, +0.125]; 0.077 |
| Gradient boosting over EF-L | +0.076 [+0.034, +0.118] [+0.027, +0.125]; 0.068 | +0.035 [+0.011, +0.063] [+0.007, +0.063]; 0.038 | +0.041 [+0.016, +0.065] [+0.014, +0.068]; 0.037 |
| Brier score of gradient boosting, 12 random years over before 2016 | | | −0.009 [−0.066, +0.047] [−0.084, +0.065]; 0.104 |
