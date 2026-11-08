# Hasil Tolok Ukur Nasional 14 — fase I revisi (benih, panjang latih, geser periode, daya)

**7 November 2026.** Lanjutan `Hasil_Patokan_13.md` (fase H), revisi PeatFireBench (dalam naskah revisi
bernama IndoFireBench). Pemicu: telaah internal pra-kirim kesembilan atas draf revisi (catatan telaah tidak
dirilis). Temuan yang perlu analisis: vonis MODIS per kabupaten bertumpu pada batas bawah t +0,001 dan +0,003 dari
satu benih; varian latih sebelum 2016 juga memendekkan latih dari 22 ke 12 tahun; diagnostik geser-tingkat tidak
memisahkan offset periode dari tingkat tahun ke tahun; bukan-penolakan MODIS dibaca tanpa selisih terkecil yang
terdeteksi; homogenitas rekaman MODIS (koleksi 6.03 → 61.03 pada 2023, pergeseran orbit Terra/Aqua) tidak
diperiksa.

## Rancangan fase I (ditulis 7 Nov 2026, SEBELUM skrip fase I ditulis atau dijalankan)

Semua analisis pasca-hoc dan eksploratori. Data, panel, tahun uji, sasaran, F, fitur dan parameter GB sama
dengan fase H (`patokan_modis_h.py`); VIIRS memakai prediksi tersimpan fase E (`patokan_revisi_prediksi.csv`,
`patokan_revisi_peluang_<sasaran>.csv`), tanpa melatih ulang.

**I1 Benih (MODIS).** Skrip `patokan_modis_i.py`, satu sasaran per jalan. GB dilatih dengan enam benih
{42, 1, 2, 3, 4, 5} (benih yang sama dengan uji benih VIIRS fase B) di V0, V1, V2 fase H. Rujukan (kalender,
EF-P, EF-L) dan logistik lokal tidak bergantung benih. Per tahun uji, Brier GB = rata-rata Brier enam benih;
BSS dan AUC-PR GB dihitung dari rata-rata itu; dua tahap (1.000, benih bootstrap B.SEED) dengan rata-rata
benih di dalam tiap ulangan; t (df 8). Juga dicetak: BSS titik tiap benih (rentang benih) dan berapa benih dari
enam yang lolos R36c sendiri. Pemeriksaan: benih 42 sendiri = log fase H (GB atas EF-L V0 +0,0612 per
kabupaten, +0,0768 nasional).

**I2 Kontrol panjang latih (MODIS).** Varian baru **V3**: tiap lipatan, latih = 12 tahun diambil acak tanpa
pengembalian dari 22 tahun lain (panjang sama dengan V1), lima undian per lipatan (`default_rng(2026 + u)`),
ambang seperti V0, GB enam benih. Brier per tahun uji = rata-rata atas undian dan benih. Pembanding: V1 (12
tahun, semuanya ≤ 2015) dan V0 (22 tahun).

**I3 Tanpa tahun 2020–2025 (MODIS, homogenitas).** Varian baru **V4**: latih = tahun lain 2003–2019 (16
tahun; tanpa tiga tahun koleksi 61.03 dan tanpa tahun pergeseran orbit 2020–2022), ambang seperti V0, GB enam
benih. Deskriptif: deteksi MODIS bersaring (`type = 0`, keyakinan bukan rendah, berkas `DL_FIRE_MODIS/kelas/`,
sebelum gabung poligon) per tahun menurut satelit (Terra/Aqua) dan versi, dan bagian Terra.

**I4 Pemecahan geser-tingkat: periode lalu tahun.** Untuk GB (benih 42, angka yang dilaporkan naskah) atas
kalender dan atas EF-P: (i) BSS mentah; (ii) **offset periode**: tiap ramalan (model dan rujukan) digeser satu
konstanta di skala logit, sama untuk semua baris uji semua tahun uji, agar rata-rata gabungannya = laju
teramati gabungan (oracle); (iii) **geser per tahun** fase E (seperti sekarang). Bagian periode = 1 − BSS(ii)
/ BSS(i); bagian tahun = (BSS(ii) − BSS(iii)) / BSS(i); jumlahnya = bagian tingkat yang dilaporkan sekarang. Dua
tahap untuk kedua bagian. Dijalankan di VIIRS (S1, enam tahun biasa 2020–2025, skrip baru
`patokan_tingkat_i.py`, GB dan RF) dan MODIS V0 dan V1 (di `patokan_modis_i.py`).

**I5 Selisih terkecil yang terdeteksi.** Untuk tiap perbandingan BSS yang dicetak (MODIS V0–V4, n = 9, df 8;
VIIRS Tabel 4 naskah, n = 6, df 5): MDD80 = (t₀,₉₇₅;df + t₀,₈₀;df) × SD tahunan / √n, uji t berpasangan
dua sisi α = 0,05, daya 80 %. VIIRS: juga selang t95 untuk tiap sel Tabel 4 (permintaan telaah: tandai sel
yang vonis dua selangnya berbeda).

**R37.** (a) **Benih**: vonis R36c tiap varian dan sasaran di naskah dinyatakan pada BSS rata-rata benih.
Bila per kabupaten V2 rata-rata benih tidak lolos atas salah satu dari tiga rujukan, abstrak, judul §6.3, §7
dan §9 berhenti menulis "in one, narrowly, per district" dan menulis bahwa per kabupaten GB tidak mengalahkan
ketiga rujukan di desain mana pun. Bila lolos, kalimat tetap dan naskah menambahkan berapa benih dari enam yang
lolos sendiri. (b) **Panjang**: bila per kabupaten GB atas EF-L lolos kedua selang di V3, naskah menulis bahwa
kegagalan V1 bukan karena latih lebih pendek, melainkan karena tidak ada tahun sesudah 2015; bila tidak lolos,
naskah menulis bahwa panjang latih dan rezim tidak dapat dipisahkan di sini. Sama untuk nasional (GB atas
EF-P, rujukan yang gagal di V1). (c) **Homogenitas**: bila vonis R36c V4 sama dengan V0 untuk sasaran itu,
naskah menulis bahwa hasil tidak bertumpu pada tahun MODIS 2020–2025; bila berbeda, naskah menyebut rujukan
mana yang berubah. Deskriptif satelit dilaporkan tanpa aturan. (d) **Periode vs tahun**: bila bagian tahun
(titik) VIIRS per kabupaten GB atas kalender ≥ 0,5, judul dan abstrak tetap memakai "the year's fire level";
bila < 0,5, judul dan abstrak memakai "the fire level" dan naskah memisahkan offset periode dari tingkat tahun.
(e) **Daya**: tiap bukan-penolakan MODIS di naskah ditulis bersama MDD80-nya ("not shown; detectable only
above X"); kalimat "too weak a bar per district" dihapus kecuali MDD80 EF-L atas kalender per kabupaten V0 <
+0,059.

### Tidak dikerjakan di fase I

Panel MODIS Aqua saja (perlu bangun ulang panel lewat `bangun_panel_nasional.js`), seluruh rekaman dari satu
koleksi (perlu unduh ulang), model penuh (22) di MODIS, musim 2026 (tetap untuk evaluasi terdaftar).

## Hasil fase I (dijalankan 7 Nov 2026 sesudah komit e8c663f)

Log: `DL_FIRE_MODIS/patokan_modis_i_{kab,nas}.log`, `DL_FIRE_NASIONAL/penuh/patokan_tingkat_i_{kab,nas}.log`.
Pemeriksaan: benih 42 V0 = fase H (GB atas EF-L +0,0612 per kabupaten); VIIRS GB atas EF-1m +0,415, bagian
tingkat 0,87 dan 0,62 = Hasil_Patokan_10. Satu deviasi kecil dari rancangan: AUC-PR tidak dihitung untuk V3
(lima undian) agar bootstrap ringan; vonis R36c memakai BSS saja, seperti rancangan.

### I1–I3 MODIS, GB rata-rata enam benih (V3: dan lima undian), BSS [dua tahap] [t95], MDD80

Per kabupaten (kalender EF-1m):

| Varian | GB atas kalender | GB atas EF-P | GB atas EF-L | R36c (benih lolos sendiri) |
|---|---|---|---|---|
| V0 semua tahun lain | +0,115 [+0,038, +0,224] [−0,002, +0,232], MDD 0,162 | +0,115 [+0,039, +0,221] [+0,001, +0,229] | +0,061 [+0,035, +0,090] [+0,031, +0,091] | tidak, kalender (1/6) |
| V1 latih < 2016 | +0,067 [−0,002, +0,157] [−0,031, +0,165], MDD 0,136 | +0,057 [−0,008, +0,141] [−0,033, +0,148] | −0,024 [−0,110, +0,032] [−0,114, +0,066], MDD 0,124 | tidak, ketiganya (0/6) |
| V2 ambang tahun latih | +0,113 [+0,037, +0,219] [+0,001, +0,225] | +0,113 [+0,038, +0,218] [+0,004, +0,222] | +0,060 [+0,034, +0,088] [+0,031, +0,089] | **ya** (5/6) |
| V3 12 tahun acak | +0,105 [+0,026, +0,214] [−0,016, +0,226] | +0,113 [+0,036, +0,218] [−0,002, +0,227] | +0,061 [+0,036, +0,090] [+0,032, +0,091] | tidak, kalender dan EF-P (0/6) |
| V4 tanpa 2020–2025 | +0,116 [+0,019, +0,253] [−0,033, +0,265] | +0,113 [+0,018, +0,246] [−0,034, +0,259] | +0,064 [+0,028, +0,109] [+0,018, +0,111] | tidak, kalender dan EF-P (0/6) |

EF-L atas kalender per kabupaten: V0 +0,059 (t [−0,057, +0,175], MDD80 0,161), V1 +0,073 (MDD 0,188), V2
+0,058, V3 +0,048, V4 +0,063; tidak satu pun lolos. EF-P atas kalender: V0 +0,001 (MDD 0,008), V1 **+0,012**,
V3 −0,007, V4 +0,005. Logistik lokal atas EF-L: V0 −0,017 (MDD 0,039), V3 +0,005. Rentang benih GB atas EF-L
V0 +0,060 s.d. +0,062; V1 −0,027 s.d. −0,017.

Nasional (kalender EF-3m):

| Varian | GB atas kalender | GB atas EF-P | GB atas EF-L | R36c (benih) |
|---|---|---|---|---|
| V0 | +0,170 [+0,113, +0,228] [+0,104, +0,236] | +0,140 [+0,079, +0,203] [+0,067, +0,214] | +0,076 [+0,039, +0,115] [+0,032, +0,120] | **ya** (6/6) |
| V1 | +0,078 [+0,040, +0,118] [+0,029, +0,127] | +0,065 [+0,012, +0,120] [−0,007, +0,137], MDD 0,099 | +0,035 [+0,011, +0,063] [+0,008, +0,063] | tidak, EF-P (0/6) |
| V2 | +0,172 [+0,115, +0,229] [+0,105, +0,240] | +0,143 [+0,080, +0,207] [+0,066, +0,219] | +0,077 [+0,039, +0,116] [+0,032, +0,122] | **ya** (6/6) |
| V3 | +0,144 [+0,093, +0,197] [+0,084, +0,204] | +0,135 [+0,083, +0,187] [+0,073, +0,196] | +0,076 [+0,034, +0,118] [+0,027, +0,125] | **ya** (6/6) |
| V4 | +0,135 [+0,080, +0,194] [+0,064, +0,206] | +0,111 [+0,040, +0,181] [+0,018, +0,203] | +0,076 [+0,036, +0,115] [+0,031, +0,121] | **ya** (6/6) |

EF-L atas kalender nasional: V0 **+0,101** (MDD 0,094), V1 +0,044 (t [−0,005, +0,093]), V2 **+0,102**, V3
**+0,071**, V4 **+0,064**. EF-P atas kalender: V0 +0,033 (t [−0,002, +0,068]), V1 +0,010, V3 +0,010, V4 +0,022.
Logistik lokal atas EF-L: V0 +0,022 (MDD 0,070), V3 +0,036 [−0,002, +0,077].

Deskriptif satelit (I3): bagian deteksi Terra 0,23–0,30 di 2003–2011, 0,27–0,38 di 2012–2020, 0,24 (2022),
0,29 (2023, versi 61.03), 0,22 (2024), 0,19 (2025); Aqua 62–81 % deteksi tiap tahun. Versi 6.03 untuk 2003–2022,
61.03 untuk 2023–2025 (2026: enam titik, di luar panel).

### I4 Pemecahan geser-tingkat (GB benih 42; bagian periode / bagian tahun [dua tahap], total)

| Panel, sasaran | atas kalender | atas EF-P |
|---|---|---|
| VIIRS per kab. (GB) | 0,60 [0,29, 0,89] / 0,27 [0,09, 0,43], 0,87 | 0,62 [0,30, 0,89] / 0,23 [0,03, 0,38], 0,85 |
| VIIRS per kab. (RF) | 0,59 [0,38, 0,83] / 0,29 [0,13, 0,43], 0,88 | 0,62 / 0,24, 0,85 |
| VIIRS nasional (GB) | 0,55 [−0,01, 0,97] / 0,07 [−0,48, 0,42], 0,62 | 0,66 / 0,04, 0,70 |
| VIIRS nasional (RF) | 0,52 [0,09, 0,70] / 0,19 [0,03, 0,42], 0,71 | 0,63 / 0,16, 0,79 |
| MODIS V0 per kab. | 0,07 [−0,25, 0,80] / 0,61 [−0,02, 1,08], 0,68 | 0,09 / 0,62, 0,70 |
| MODIS V1 per kab. | 0,18 / 0,46, 0,64 | 0,21 / 0,65, 0,86 |
| MODIS V0 nasional | 0,23 [−0,02, 0,49] / 0,20 [−0,02, 0,46], 0,44 | 0,26 / 0,26, 0,52 |
| MODIS V1 nasional | −0,00 / 0,21, 0,21 | 0,02 / 0,28, 0,30 |

Geser periode VIIRS (logit): per kab. EF-1m −1,252, EF-P −1,142, GB −0,398, RF −0,607; nasional EF-3m −0,985,
GB −0,391. BSS sesudah offset periode VIIRS per kab. GB atas EF-1m +0,168 [+0,035, +0,309] (t [−0,015, +0,350]).

### I5 MDD80 VIIRS (Tabel 4, n = 6, df 5)

Atas EF-L: per kab. GB 0,155, RF 0,134, GB tersetel 0,202, MLP 0,392, logistik lokal 0,256, GB + hitungan
nasional 0,153; nasional 0,240, 0,109, 0,165, 0,541, 0,120, 0,158. Atas kalender: EF-L 0,398 / 0,232, EF-P
0,162 / 0,025. Sel yang vonis dua selangnya berbeda: per kab. logistik lokal atas EF-L (dua tahap
[−0,303, −0,005], t [−0,335, +0,042]), GB + hitungan nasional atas EF-L (dua tahap [+0,007, +0,180], t
[−0,022, +0,203]); nasional laju konstan atas EF-3m (dua tahap [−0,827, −0,076], t [−0,945, +0,039]).

### Putusan R37

(a) **Benih**: per kabupaten V2 rata-rata benih lolos atas ketiganya (batas bawah t +0,001 atas EF-1m, +0,004
atas EF-P; 5 dari 6 benih lolos sendiri): kalimat "in one, narrowly, per district" tetap, ditambah hitungan
benih. V0 per kabupaten tetap tidak (kalender, t −0,002; 1 dari 6 benih). Nasional V0 dan V2 ya (6/6), V1
tidak (EF-P). (b) **Panjang**: per kabupaten GB atas EF-L lolos di V3 (+0,061): kegagalan V1 bukan karena latih
lebih pendek, tetapi karena tidak ada tahun sesudah 2015. Nasional GB atas EF-P lolos di V3 (+0,135): sama.
(c) **Homogenitas**: vonis R36c V4 sama dengan V0 untuk kedua sasaran (per kabupaten tidak, nasional ya); hasil
tidak bertumpu pada tahun MODIS 2020–2025. Per kabupaten rujukan yang tidak dikalahkan bertambah: EF-P (t
−0,034). (d) **Periode vs tahun**: bagian tahun VIIRS per kabupaten GB atas kalender 0,27 < 0,5: judul dan
abstrak memakai "the fire level" dan naskah memisahkan offset periode (0,60) dari tingkat tahun. Di MODIS
(tinggalkan-satu-tahun) yang dominan justru tingkat tahun (0,61 per kabupaten). (e) **Daya**: MDD80 EF-L
atas kalender per kabupaten V0 0,161 > 0,059: kalimat "too weak a bar per district" dihapus; bukan-penolakan
MODIS ditulis dengan MDD80.

**Bacaan.** Nasional, GB riwayat api mengalahkan ketiga rujukan bila ada tahun sesudah 2015 di data latih, di
semua bentuk (semua tahun lain, 12 tahun acak, sampai 2019, ambang tahun latih), dan tidak bila dilatih hanya
dengan tahun sebelum 2016. Per kabupaten ia mengalahkan EF-L bila ada tahun sesudah 2015, tetapi kalender
mentah hanya dengan ambang tahun latih, tipis. Di VIIRS kronologis, sebagian besar "tingkat" adalah offset
periode (tahun uji lebih sepi daripada tahun latih), bukan tingkat tiap tahun.
