# Hasil Tolok Ukur Nasional 12 — fase G revisi (daya uji Brier, ambang tahun latih untuk peluang, uji MODIS untuk EF-L)

**7 November 2026.** Lanjutan `Hasil_Patokan_11.md` (fase F), revisi PeatFireBench (dalam naskah revisi
bernama IndoFireBench). Pemicu: telaah internal pra-kirim keenam atas draf revisi (catatan telaah tidak
dirilis). Temuan utamanya: (1) EF-L tidak punya uji pada tahun yang tidak membentuknya (saran: panel MODIS
2003–2011); (2) "not shown to beat EF-L" tanpa selisih minimum yang terdeteksi untuk skor Brier;
(3) kebocoran ambang hanya diuji untuk urutan, padahal klaim yang tersisa tentang peluang.

## Rancangan fase G (ditulis 7 Nov 2026, SEBELUM skrip mana pun dijalankan)

Semua analisis pasca-hoc dan eksploratori kecuali G3 pada tahun 2003–2011, yang belum pernah dinilai oleh
putaran mana pun (panel VIIRS mulai 2012). Selang seperti fase F: dua tahap (1.000, tahun lalu kabupaten),
t atas tahun (df = banyak tahun − 1), kabupaten (1.000) bila disebut; seed 2026. "Lolos" = batas bawah dua
tahap dan t keduanya > 0.

### G1. Daya uji skor Brier atas EF-L (dari angka per tahun yang sudah ada)

Untuk BSS atas EF-L, tahun biasa S1 2020–2025, kedua sasaran, model GB, RF, GB tersetel, GB + nasional:
selisih minimum terdeteksi pada daya 80 % = (t₀,₉₇₅ + t₀,₈₀; df 5) × sd tahunan / √6, dan banyaknya tahun
yang dibutuhkan untuk mendeteksi BSS yang teramati (iterasi df). Sumber angka per tahun: log fase F
(`patokan_revisi_tingkat_{kab,nas}.log`, baris BSS "per tahun"). Skrip `patokan_revisi_daya.py`.

**R33.** Tanpa aturan keputusan; naskah menulis di §6.2 dan §8 bahwa enam tahun hanya mendeteksi BSS atas
EF-L di atas nilai itu, berdampingan dengan "not shown to beat".

### G2. Skor peluang dengan ambang dari tahun latih

Label diulang per lipatan dengan `label_ulang` dari `patokan_revisi_ambang.py` (persentil 90 kabupaten,
batas bawah 10, dan persentil 90 gabungan, dihitung dari tahun latih). RF dan GB dari
`patokan_prediksi_kepekaan_penuh.csv` (dilatih ulang pada label ulang, benih 42). EF-1m, EF-3m dan EF-L
dihitung ulang dari label ulang tahun latih (EF-L: fungsi fase E/F, anomali nasional tidak berubah). Baris
per kabupaten: kabupaten dengan positif di tahun latih lipatan itu. S1 2020–2025, kedua sasaran. Dilaporkan:
BSS RF dan GB atas kalender pasangan dan atas EF-L; BSS EF-L atas kalender. Skrip
`patokan_revisi_ambang_peluang.py`, satu sasaran per jalan.

**R34.** (a) Bila BSS GB dan RF atas kalender tetap lolos, naskah menulis bahwa perolehan peluang atas
kalender bertahan dengan ambang tahun latih. (b) Bila GB atau RF lolos atas EF-L, klaim "none shown to beat
EF-L" gugur untuk sasaran itu di bawah ambang tahun latih, dan abstrak menyebutnya. (c) Selain itu naskah
menulis bahwa klaim peluang tidak berubah dengan ambang tahun latih (dengan angka).

### G3. Uji EF-L pada tahun MODIS 2003–2011

**Data.** FIRMS `MODIS_SP` (Collection 6.1 standard processing, Terra dan Aqua), kotak unduhan sama
(95–141°E, 11°S–6°N), 1 Januari 2003 – 31 Desember 2025 (Aqua lengkap sejak 2003), jendela lima hari,
skrip `unduh_firms_modis.py` (salinan `unduh_firms_nasional.py`, sumber diganti), keluaran
`DL_FIRE_MODIS/` (tidak masuk repo). Saringan setara dengan panel VIIRS: `type = 0` dan keyakinan bukan
rendah; keyakinan MODIS berupa angka 0–100, dan kelas rendahnya < 30 (panduan pengguna FIRMS MODIS C6.1),
jadi `modis_keyakinan.py` menulis ulang CSV dengan `confidence` = `l` bila < 30, `n` bila 30–79, `h` bila
≥ 80, lalu `bangun_panel_nasional.js` dijalankan **tanpa perubahan** (GADM 4.1 sama) →
`DL_FIRE_MODIS/panel_modis.csv`.

**Sasaran.** Sama aturannya dengan panel VIIRS, dihitung dari MODIS 2003–2025: per kabupaten = jumlah bulan
di atas maks(persentil 90 kabupaten, F), nasional = di atas persentil 90 semua kabupaten-bulan. Batas bawah
F = bulat(10 × r), r = jumlah MODIS nasional ÷ jumlah VIIRS nasional 2012–2025 (dari kedua panel), minimal 1.
Kabupaten tanpa bulan positif dikecualikan dari metrik per kabupaten. Ambang memakai seluruh 2003–2025
(sama dengan panel utama; kebocoran yang sama dicatat).

**Lipatan.** Tiap tahun uji u ∈ 2003–2011 ditahan; semua tahun MODIS lain 2003–2025 adalah tahun latih
(seperti S2, termasuk tahun sesudah u). EF-1m, EF-3m, ef1_halus dan EF-L dihitung dengan fungsi fase E/F
**tanpa perubahan bentuk** (κ = 2, logistik pada logit ef1_halus dan anomali log1p jumlah MODIS nasional
bulan lalu terhadap rata-rata bulan kalender itu di tahun latih, tinggalkan-tahun-sendiri untuk baris
latih); hanya tiga koefisiennya yang dilatih ulang pada tahun latih MODIS. Model pembanding yang bisa
dibangun dari jumlah saja: (i) laju konstan tahun latih; (ii) logistik riwayat lokal pada logit EF-1m,
keadaan lampau bulan lalu dan log1p jumlah bulan lalu (bentuk `log_ef1_jeda`); (iii) GB riwayat api:
parameter baku fase F, fitur log1p jumlah jeda 1, 2, 3 dan 12 bulan, sin/cos bulan, EF-1m, keadaan lampau
bulan lalu, dan anomali nasional (sama dengan EF-L, supaya masukannya tidak ditahan).

**Metrik.** BSS dalam-tahun dan AUC-PR dalam-tahun, rata-rata atas sembilan tahun uji; dua tahap dan t (df 8);
per tahun dengan selang kabupaten. Skrip `patokan_modis.py`, satu sasaran per jalan.

**R35.** (a) Bila BSS EF-L atas kalender pasangan (EF-1m per kabupaten, EF-3m nasional) lolos, naskah menulis
bahwa penyesuaian tingkat bertahan pada sembilan tahun yang tidak membentuknya; bila tidak lolos, EF-L disebut
"candidate baseline" di judul dan abstrak. (b) Bila logistik riwayat lokal atau GB riwayat api lolos atas EF-L
dalam BSS untuk suatu sasaran, naskah menulis bahwa model yang dipelajari mengalahkan EF-L di luar sampel untuk
sasaran itu, dan EF-L tidak diusulkan sebagai baseline untuk sasaran itu. (c) Selain itu: "not shown to beat
EF-L on nine further years". (d) AUC-PR EF-L − kalender dilaporkan tanpa aturan.

**Penyimpangan yang sudah diketahui.** Sensor beda (MODIS 1 km, VIIRS 375 m): ambang dan jumlah tidak setara
antar panel; yang diuji adalah bentuk EF-L, bukan angkanya. Orbit Terra bergeser sejak 2022 (deteksi siang
lebih sore); tahun-tahun itu hanya masuk sebagai tahun latih.

**Penyesuaian sebelum G3 dijalankan (7 Nov, sesudah uji asap pada panel tiruan, sebelum data MODIS ada).**
EF-1m dan EF-3m di `patokan_modis.py` diberi pemecah seri protokol utama (+10⁻⁶·k/(1+k), k = rata-rata jumlah
tahun latih, seperti `R.klim_frek`; baris latih tanpa tahunnya sendiri), dan jeda kosong pada GB diisi 0 seperti
`xgb_sejarah`. EF-L dan ef1_halus tidak berubah. Pengaruh pemecah seri pada Brier ≤ 10⁻⁶; yang berubah hanya
urutan di antara seri untuk AUC-PR kalender.

### Tidak dikerjakan di fase G

Model penuh (22) pada MODIS (kovariat cuaca dan lahan perlu dibangun ulang untuk 2003–2011), jalur gambut
(benchmark diganti nama), konsultasi lembaga, kalender bersyarat ENSO.

## Hasil G1 dan G2 (dijalankan 7 Nov 2026 sesudah komit f44b1d2)

### G1. Daya uji BSS atas EF-L (`patokan_revisi_daya.log`)

| Model | Per kabupaten: rata-rata / sd / MDD80 / tahun dibutuhkan | Nasional |
|---|---|---|
| RF | +0,030 / 0,094 / 0,135 / 79 | +0,058 / 0,076 / 0,109 / 16 |
| GB | +0,067 / 0,109 / 0,155 / 24 | +0,037 / 0,169 / 0,240 / 164 |
| GB tersetel | +0,065 / 0,142 / 0,202 / 40 | +0,069 / 0,116 / 0,166 / 25 |
| GB + nasional | +0,091 / 0,108 / 0,153 / 14 | +0,066 / 0,111 / 0,158 / 25 |

**R33.** Enam tahun hanya mendeteksi BSS atas EF-L di atas 0,11–0,24 (daya 80 %); BSS yang teramati
(0,03–0,09) butuh 14 sampai 164 tahun uji. Naskah menulisnya di §6.2 dan §8.

### G2. Ambang dari tahun latih (`patokan_revisi_ambang_peluang_{kab,nas}.log`)

Pemeriksaan: label kepekaan sama dengan label dihitung ulang (1,0000); per kabupaten 29.376 baris
(kabupaten dengan positif di tahun latih), EF-L tanpa baris kosong. Prevalensi 3,75 % / 5,39 %.

| BSS [dua tahap] [t95] | Per kabupaten | Nasional |
|---|---|---|
| RF atas kalender | **+0,384** [+0,203, +0,564] [+0,132, +0,635] | **+0,293** [+0,148, +0,463] [+0,077, +0,510] |
| GB atas kalender | **+0,393** [+0,212, +0,580] [+0,133, +0,653] | **+0,273** [+0,118, +0,455] [+0,032, +0,513] |
| EF-L atas kalender | **+0,362** [+0,161, +0,559] [+0,077, +0,647] | **+0,256** [+0,123, +0,405] [+0,060, +0,451] |
| RF atas EF-L | +0,022 [−0,061, +0,078] [−0,058, +0,102] | +0,059 [+0,008, +0,121] [−0,007, +0,126] |
| GB atas EF-L | +0,049 [−0,033, +0,116] [−0,042, +0,139] | +0,030 [−0,116, +0,156] [−0,166, +0,225] |

AUC-PR rata-rata (label ulang): per kabupaten kalender 0,247, EF-L 0,227, RF 0,278, GB 0,276; nasional 0,694,
0,705, 0,748, 0,738.

**R34.** (a) terpenuhi: perolehan peluang RF dan GB atas kalender bertahan dengan ambang tahun latih (kedua
sasaran). (b) tidak terpenuhi: tidak ada yang lolos atas EF-L. (c) Naskah menulis bahwa klaim peluang tidak
berubah dengan ambang tahun latih: BSS GB atas EF-L +0,049 per kabupaten dan +0,030 nasional.

## Hasil G3 (dijalankan 7 Nov 2026 sesudah komit fd45c32)

**Data.** `MODIS_SP` 1 Jan 2003 – 31 Des 2025, 1.681 jendela, satu putus DNS di 2012-02-08 lalu dilanjutkan
(berkas yang ada dilewati), tanpa kegagalan tersisa. 1.553.413 titik; keyakinan rendah 115.821, bukan vegetasi
143.825, di luar kabupaten 98.928; panel 1.194.839 titik, 502 kabupaten, 2003-01 s.d. 2026-01 (dibatasi ke
498 kabupaten VIIRS dan 2003–2025 di skrip). Versi produk 6.03 untuk 2003–2022 dan 61.03 untuk 2023–2025
(tahun uji seluruhnya 6.03; 2023–2025 hanya tahun latih — penyimpangan tambahan, dicatat). Rasio MODIS/VIIRS
2012–2025 r = 0,195, batas bawah F = 2, ambang gabungan 9,0 titik. 419 kabupaten punya bulan positif;
prevalensi 6,35 % per kabupaten, 9,88 % nasional (lebih tinggi dari VIIRS karena seri pada jumlah kecil).
Koefisien anomali nasional EF-L +0,87 s.d. +0,96 di semua lipatan (VIIRS +0,71 s.d. +1,11).
Log: `DL_FIRE_NASIONAL/penuh/patokan_modis_{kab,nas}.log`.

| 9 tahun uji 2003–2011, df 8 [dua tahap] [t95] | Per kabupaten (kalender EF-1m) | Nasional (kalender EF-3m) |
|---|---|---|
| BSS EF-L atas kalender | +0,059 [−0,018, +0,167] [−0,057, +0,175] | **+0,101** [+0,044, +0,157] [+0,033, +0,168] |
| BSS konstan atas kalender | −0,304 [−0,436, −0,109] [−0,508, −0,100] | −1,008 |
| BSS ef1_halus atas kalender | +0,001 | +0,033 [+0,004, +0,059] [−0,002, +0,068] |
| BSS logistik lokal atas kalender | +0,046 [−0,030, +0,144] | **+0,123** [+0,076, +0,173] [+0,063, +0,183] |
| BSS GB riwayat api atas kalender | +0,115 [+0,038, +0,223] [−0,003, +0,233] | **+0,171** [+0,115, +0,228] [+0,106, +0,236] |
| BSS logistik lokal atas EF-L | −0,017 [−0,050, +0,014] [−0,045, +0,012] | +0,022 [−0,022, +0,068] [−0,028, +0,072] |
| BSS GB riwayat api atas EF-L | **+0,061** [+0,034, +0,091] [+0,030, +0,093] | **+0,077** [+0,041, +0,114] [+0,034, +0,120] |
| AUC-PR EF-L − kalender | −0,030 [−0,067, +0,007] [−0,073, +0,014] | −0,002 [−0,017, +0,011] |
| AUC-PR GB − kalender | +0,029 [+0,001, +0,060] [−0,004, +0,062] | **+0,030** [+0,012, +0,050] [+0,007, +0,053] |
| AUC-PR GB − EF-L | **+0,058** [+0,032, +0,085] [+0,029, +0,087] | **+0,032** [+0,015, +0,051] [+0,012, +0,052] |

Per tahun (selang kabupaten): GB atas EF-L positif di 8 dari 9 tahun untuk kedua sasaran (per kabupaten
hanya 2009 −0,007; nasional hanya 2003 −0,013). EF-L atas EF-1m per kabupaten negatif di 2003, 2004, 2005,
2007 dan 2008 (−0,018 s.d. −0,068) dan besar di 2006 (+0,203) dan 2010 (+0,400, tahun paling sepi);
nasional positif di 8 dari 9 tahun.

**R35.** (a) Per kabupaten **tidak terpenuhi**: BSS EF-L atas EF-1m +0,059, kedua selang memuat nol, jadi
EF-L disebut "candidate baseline" di judul dan abstrak. Nasional terpenuhi (+0,101): penyesuaian tingkat
bertahan pada sembilan tahun yang tidak membentuknya. (b) **Terpenuhi untuk kedua sasaran**: GB riwayat api
(jumlah titik panas jeda 1, 2, 3, 12, musim, EF-1m, keadaan bulan lalu, anomali nasional) lolos atas EF-L
dalam BSS per kabupaten (+0,061) dan nasional (+0,077). Menurut aturan, naskah menulis bahwa model yang
dipelajari mengalahkan EF-L di luar sampel untuk kedua sasaran, dan EF-L **tidak diusulkan sebagai baseline**
untuk sasaran mana pun. Logistik riwayat lokal tidak lolos atas EF-L. (c) tidak berlaku. (d) AUC-PR EF-L −
kalender −0,030 per kabupaten, −0,002 nasional.

**Batas tafsir.** Sensor lain (MODIS 1 km), panel latih 22 tahun (VIIRS 8–13), ambang dengan F = 2, dan GB
di sini hanya riwayat api + anomali nasional (tanpa cuaca, lahan, indeks iklim). Hasil ini sejalan dengan
VIIRS: GB + nasional atas EF-L per kabupaten +0,091 (dua tahap [+0,007, +0,180], t [−0,022, +0,203]) —
enam tahun tidak cukup, sembilan cukup. Yang gugur adalah usulan EF-L sebagai baseline; yang bertahan:
sebagian besar perolehan peluang atas kalender mentah pada VIIRS adalah tingkat tahun. Pada MODIS, dengan
kalender dari 22 tahun, penyesuaian tingkat per kabupaten jauh lebih kecil (+0,059, tak terbedakan; VIIRS +0,376).
