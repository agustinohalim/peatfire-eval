# Hasil Tolok Ukur Nasional 13 — fase H revisi (kepekaan jalur MODIS)

**7 November 2026.** Lanjutan `Hasil_Patokan_12.md` (fase G), revisi PeatFireBench (dalam naskah revisi
bernama IndoFireBench). Pemicu: telaah internal pra-kirim ketujuh atas draf revisi (catatan telaah tidak
dirilis). Temuan utamanya: di MODIS per kabupaten EF-L tidak di atas kalender mentah dan GB tidak terbukti di
atas kalender (selang tahunan [−0,003, +0,233]), jadi tiga rujukan tidak berurutan; jalur MODIS melatih dengan
tahun sesudah tahun uji dan mencampur rezim sebelum/sesudah 2015; ambang MODIS dari semua tahun termasuk tahun
uji; "87 % adalah tingkat tahun" mungkin akibat kalender pendek VIIRS; EF-P tidak dijalankan di MODIS; ENSO
tahun uji MODIS tidak dilaporkan; BSS rata-rata rasio didorong tahun sepi.

## Rancangan fase H (ditulis 7 Nov 2026, SEBELUM skrip fase H ditulis atau dijalankan)

Semua analisis pasca-hoc dan eksploratori; data MODIS dan panel sama dengan G3 (`DL_FIRE_MODIS/panel_modis.csv`),
tahun uji 2003–2011, sasaran dan F = 2 sama. Satu skrip `patokan_modis_h.py`, satu sasaran per jalan, tiga
varian lipatan:

- **V0 utama**: desain G3 tanpa perubahan (latih = semua tahun MODIS lain, ambang dari 2003–2025). Pemeriksaan:
  BSS GB atas EF-L harus sama dengan log G3 (+0,0612 per kabupaten, +0,0768 nasional).
- **V1 latih sebelum 2016**: latih = 2003–2015 tanpa tahun uji (12 tahun); tidak ada tahun sesudah 2015 dalam
  kalender, koefisien EF-L, anomali nasional, logistik lokal, maupun GB. Ambang seperti V0.
- **V2 ambang tahun latih**: latih seperti V0; ambang per kabupaten (maks(persentil 90, F)) dan ambang gabungan
  dihitung dari tahun latih lipatan saja, kabupaten yang dikecualikan (tanpa positif) juga dari tahun latih;
  keadaan lampau bulan lalu dihitung dari label ulang. Pola sama dengan `patokan_revisi_ambang.py`.

Di tiap varian: kalender (EF-1m per kabupaten, EF-3m nasional), **EF-P** (enam calon fase E: konstan, laju per
bulan, EF-1m, EF-3m, EF-1m dari tahun ≥ 2016 di tahun latih — bila tidak ada, laju per bulan —, EF-1m dihaluskan;
dipilih dengan Brier tinggalkan-satu-tahun di tahun latih), EF-L (bentuk tetap), logistik lokal, GB riwayat api
(fitur dan parameter G3). Metrik: BSS dalam-tahun rata-rata sembilan tahun, dua tahap (1.000) dan t (df 8);
AUC-PR GB − EF-L. Hanya V0: (i) diagnostik geser-tingkat fase E (tiap ramalan digeser di skala logit agar
rata-ratanya = laju teramati tahun itu) untuk GB atas kalender dan atas EF-P, dengan bagian yang hilang;
(ii) BSS rasio-jumlah (1 − ΣBrier model / ΣBrier rujukan atas semua baris uji) untuk perbandingan utama.
ENSO: ONI Agustus–Oktober 2003–2011 dari berkas NOAA di repo (dicatat di bawah, tanpa aturan).

**R36.** (a) **V1**: bila GB atas EF-L lolos (kedua selang) untuk suatu sasaran, naskah menulis bahwa
pembalikan MODIS tidak bergantung pada tahun latih sesudah tahun uji untuk sasaran itu; bila tidak lolos,
naskah menulis bahwa untuk sasaran itu pembalikan bergantung pada tahun latih sesudah 2015 dan bukan hasil
prakiraan, dan abstrak menyebutnya. (b) **V2**: sama, untuk ambang tahun latih. (c) **Urutan rujukan**:
protokol naskah ditetapkan SEKARANG, sebelum hasil: sebuah model dikatakan "mengalahkan rujukan" hanya bila
lolos atas ketiganya (kalender, EF-P, EF-L) di kedua selang; naskah melaporkan untuk tiap sasaran dan tiap
varian apakah GB memenuhinya, dan bila tidak, rujukan mana yang tidak dikalahkan. (d) **Bagian tingkat
(V0)**: bila bagian BSS GB atas kalender yang hilang oleh geser-tingkat per kabupaten < 0,5 (titik), naskah
membatasi "sebagian besar perolehan = tingkat tahun" ke panel VIIRS dengan kalender pendeknya, dan abstrak
menyebutnya; bila ≥ 0,5, naskah menulis bahwa itu juga berlaku di MODIS. (e) Rasio-jumlah: bila tanda titik
suatu perbandingan utama berbalik terhadap rata-rata rasio, dilaporkan di §6.9; tanpa aturan lain.

ONI Agustus–Oktober tahun uji MODIS (NOAA `oni.ascii.txt`): 2003 +0,23, 2004 +0,58, 2005 −0,11, 2006 +0,54,
2007 −1,05, 2008 −0,25, 2009 +0,66, 2010 −1,43, 2011 −0,76. Ambang ±0,5: El Niño 2004, 2006, 2009; La Niña
2007, 2010, 2011; netral 2003, 2005, 2008.

### Tidak dikerjakan di fase H

Model penuh (22) di MODIS, GB dengan banyak benih (dicatat sebagai keterbatasan), titik api awal vs lanjutan
(gambut membara), konsultasi lembaga, kalender bersyarat ENSO.

## Hasil fase H (dijalankan 7 Nov 2026 sesudah komit cf01a54 dan 25828ad)

Log: `DL_FIRE_NASIONAL/penuh/patokan_modis_h_{kab,nas}.log`. Pemeriksaan: V0 sama dengan G3 (GB atas EF-L
+0,0612 per kabupaten, +0,0768 nasional; EF-L atas kalender +0,0590 / +0,1007). EF-P memilih EF-1m dihaluskan
di semua 54 lipatan (dua sasaran × tiga varian × sembilan tahun).

BSS [dua tahap] [t95], sembilan tahun uji 2003–2011. Tebal = lolos kedua selang.

| Per kabupaten | V0 utama | V1 latih < 2016 | V2 ambang tahun latih |
|---|---|---|---|
| EF-L atas kalender | +0,059 [−0,018, +0,167] [−0,057, +0,175] | +0,073 [−0,011, +0,199] [−0,063, +0,208] | +0,058 [−0,018, +0,159] [−0,056, +0,172] |
| EF-P atas kalender | +0,001 | **+0,012** [+0,004, +0,022] [+0,001, +0,023] | +0,001 |
| GB atas kalender | +0,115 [+0,038, +0,223] [−0,003, +0,233] | +0,066 [−0,005, +0,155] [−0,034, +0,165] | **+0,114** [+0,036, +0,221] [+0,001, +0,227] |
| GB atas EF-P | +0,115 [+0,038, +0,222] [−0,000, +0,230] | +0,056 [−0,011, +0,140] [−0,036, +0,148] | **+0,114** [+0,038, +0,219] [+0,003, +0,225] |
| GB atas EF-L | **+0,061** [+0,034, +0,091] [+0,030, +0,093] | −0,025 [−0,109, +0,032] [−0,115, +0,065] | **+0,061** [+0,034, +0,090] [+0,032, +0,091] |
| Logistik lokal atas EF-L | −0,017 | −0,021 | −0,020 |
| AUC-PR GB − EF-L | **+0,058** | +0,018 [−0,003, +0,038] [−0,004, +0,040] | **+0,055** |
| GB mengalahkan ketiga rujukan | tidak (kalender, EF-P) | tidak (ketiganya) | ya (t kalender +0,001) |

| Nasional | V0 utama | V1 latih < 2016 | V2 ambang tahun latih |
|---|---|---|---|
| EF-L atas kalender | **+0,101** [+0,044, +0,157] [+0,033, +0,168] | +0,044 [+0,006, +0,085] [−0,005, +0,093] | **+0,102** [+0,045, +0,159] [+0,034, +0,170] |
| EF-P atas kalender | +0,033 [+0,004, +0,059] [−0,002, +0,068] | +0,010 | +0,033 |
| GB atas kalender | **+0,171** [+0,115, +0,228] [+0,106, +0,236] | **+0,075** [+0,034, +0,119] [+0,024, +0,127] | **+0,172** |
| GB atas EF-P | **+0,141** [+0,081, +0,204] [+0,068, +0,214] | +0,062 [+0,007, +0,124] [−0,013, +0,137] | **+0,143** |
| GB atas EF-L | **+0,077** [+0,041, +0,114] [+0,034, +0,120] | **+0,033** [+0,008, +0,061] [+0,004, +0,062] | **+0,077** [+0,041, +0,114] [+0,034, +0,120] |
| Logistik lokal atas EF-L | +0,022 | +0,013 | +0,023 |
| AUC-PR GB − EF-L | **+0,032** | +0,014 [+0,000, +0,029] [−0,003, +0,030] | **+0,032** |
| GB mengalahkan ketiga rujukan | ya | tidak (EF-P, t) | ya |

Geser-tingkat (V0, GB): per kabupaten atas kalender mentah +0,115 → tingkat-benar +0,037 (dua tahap [−0,020,
+0,092], t [−0,032, +0,106]), bagian hilang **0,68** [0,19, 1,36]; atas EF-P 0,70. Nasional atas kalender +0,171 →
+0,097 ([+0,052, +0,143], t [+0,046, +0,147]), bagian hilang **0,44** [0,21, 0,65]; atas EF-P 0,52.

Rasio-jumlah (V0) vs rata-rata rasio: per kabupaten EF-L/kalender +0,058 / +0,059, GB/kalender +0,108 / +0,115,
GB/EF-P +0,109 / +0,115, GB/EF-L +0,054 / +0,061, lokal/EF-L −0,016 / −0,017; nasional +0,108 / +0,101, +0,177 /
+0,171, +0,147 / +0,141, +0,077 / +0,077, +0,021 / +0,022. Tidak ada tanda berbalik.

**R36.** (a) **V1**: per kabupaten GB atas EF-L **tidak lolos** (−0,025): pembalikan per kabupaten bergantung
pada tahun latih sesudah 2015, bukan hasil prakiraan; naskah dan abstrak menyebutnya. Nasional **lolos**
(+0,033): pembalikan nasional tidak bergantung pada tahun sesudah tahun uji. (b) **V2**: kedua sasaran lolos
(+0,061 / +0,077): ambang dari semua tahun tidak menimbulkan pembalikan. (c) **Urutan**: protokol "mengalahkan
rujukan = lolos atas ketiganya". Per kabupaten V0: GB tidak mengalahkan rujukan (lolos atas EF-L saja; atas
kalender dan EF-P selang t memuat nol); V1: tidak satu pun; V2: ya, tipis (batas bawah t +0,001). Nasional V0
dan V2: ya; V1: tidak (EF-P, t −0,013). (d) Bagian tingkat per kabupaten di MODIS 0,68 ≥ 0,5: naskah menulis
bahwa sebagian besar perolehan GB atas kalender per kabupaten adalah tingkat tahun juga di MODIS (VIIRS 0,87);
nasional 0,44 (VIIRS ±2/3). (e) Tidak ada tanda berbalik.

**Bacaan.** Per kabupaten, di kedua panel, tidak ada ramalan yang terbukti di atas kalender mentah sebagai
peluang secara kokoh: di MODIS hanya V2 lolos tipis, V0 dan V1 tidak; EF-L tidak di atas kalender di varian
mana pun. Nasional, GB riwayat api mengalahkan kalender dan EF-L di semua varian, juga bila hanya dilatih
dengan tahun sebelum 2016. Satu benih, satu model; ENSO tahun uji MODIS: tiga El Niño (2004, 2006, 2009), tiga
La Niña (2007, 2010, 2011), tiga netral.

### Angka lengkap untuk Tabel 5 naskah (disalin dari log 7 Nov 2026)

Selang yang tidak dimuat tabel di atas, [dua tahap] [t95]:

- Per kabupaten V0: EF-P atas kalender +0,001 [−0,003, +0,006] [−0,005, +0,007]; logistik lokal atas EF-L −0,017
  [−0,050, +0,014] [−0,045, +0,012]; AUC-PR GB − EF-L +0,058 [+0,032, +0,085] [+0,029, +0,087]; AUC-PR GB − kalender
  +0,029 [+0,001, +0,060] [−0,004, +0,062].
- Per kabupaten V1: logistik lokal atas EF-L −0,021 [−0,060, +0,008] [−0,055, +0,013]; AUC-PR GB − kalender +0,008
  [−0,016, +0,032] [−0,019, +0,034].
- Nasional V0: logistik lokal atas EF-L +0,022 [−0,022, +0,068] [−0,028, +0,072]; AUC-PR GB − EF-L +0,032 [+0,015,
  +0,051] [+0,012, +0,052]; AUC-PR GB − kalender +0,030 [+0,012, +0,050] [+0,007, +0,053].
- Nasional V1: EF-P atas kalender +0,010 [−0,024, +0,042] [−0,031, +0,051]; logistik lokal atas EF-L +0,013 [−0,021,
  +0,047] [−0,025, +0,050]; AUC-PR GB − kalender +0,014 [−0,002, +0,032] [−0,006, +0,034].

BSS EF-L atas kalender per tahun (V0, 2003–2011): per kabupaten −0,018 −0,023 −0,029 +0,203 −0,068 −0,037 +0,074
+0,400 +0,029 (lima negatif; 2010 prevalensi terendah 2,33%); nasional +0,098 +0,039 +0,066 +0,264 +0,038 −0,035
+0,183 +0,126 +0,127 (delapan positif). Koefisien anomali nasional EF-L: V0 +0,871 s.d. +0,942 per kabupaten,
+0,897 s.d. +0,960 nasional; V1 +0,585 s.d. +0,787 dan +0,492 s.d. +0,696.
