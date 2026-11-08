# Hasil Tolok Ukur Nasional 16 — fase K revisi (agregasi prakiraan maju MODIS, EF-L atas EF-P, Brier rujukan, lantai kejadian)

**7 November 2026.** Lanjutan `Hasil_Patokan_15.md` (fase J), revisi PeatFireBench (dalam naskah revisi
bernama IndoFireBench). Pemicu: telaah internal pra-kirim kedua belas atas draf revisi (catatan telaah tidak
dirilis). Temuan utamanya: vonis prakiraan maju per kabupaten (GB atas EF-L −0,061) adalah rata-rata rasio
tahunan, ditarik tiga tahun sepi (2010, 2016, 2017; GB di atas EF-L 8 dari 12 tahun); satu rasio atas semua baris
dan median belum dihitung untuk desain maju; sama untuk EF-L atas kalender (+0,141). Juga: EF-L atas EF-P dalam
desain maju tak dilaporkan; "beda V3 − V1 datang dari rujukan" disimpulkan tanpa Brier rujukan; lantai kejadian
MODIS per kabupaten 2 deteksi mungkin terlalu lemah.

## Rancangan fase K (ditulis 7 Nov 2026, SEBELUM skrip fase K dijalankan)

Semua analisis pasca-hoc dan dibuat sesudah angka fase J terlihat (per tahun GB atas EF-L per kabupaten sudah
tercatat di `Hasil_Patokan_15`). Data, panel MODIS, fitur, parameter GB, enam benih {42, 1, 2, 3, 4, 5}, EF-P
(enam calon, Brier LOO tahun latih), EF-L (bentuk tetap) dan undian V3 sama dengan fase J. Skrip baru
`patokan_modis_k.py` (satu sasaran dan satu lantai per jalan). Dua tahap 1.000 ulangan, benih bootstrap B.SEED;
t berpasangan dua sisi; MDD80. GB = rata-rata enam benih (jumlah galat kuadrat dirata-rata atas benih).

**K1 Agregasi prakiraan maju (desain R fase J, tahun uji 2008–2019, n = 12).** Lipatan R dijalankan ulang persis.
Untuk tiap pasangan: (a) rata-rata rasio tahunan (= fase J; harus mereproduksi angka `Hasil_Patokan_15`);
(b) **satu rasio** atas semua baris uji, 1 − Σ galat kuadrat model / Σ galat kuadrat rujukan, dengan selang dua
tahap (per ulangan, jumlah atas tahun dan kabupaten terambil); tak ada selang tingkat tahun untuk satu rasio;
(c) median rasio tahunan, banyak tahun di atas nol, uji tanda dua sisi; (d) rata-rata rasio tanpa **tiga tahun
uji berprevalensi terendah untuk sasaran itu** (dari log fase J: 2010, 2016, 2017 untuk kedua sasaran), dua tahap
dan t95, df 8; (e) deskriptif: rata-rata 2008–2015 dan 2016–2019 (tahun uji pertama dengan tahun sesudah 2015 di
latih). Dua sasaran.

**K2 EF-L atas EF-P** ditambahkan sebagai pasangan pada K1 (semua agregasi). Memisahkan penyesuaian tingkat
(anomali nasional) dari penghalusan kalender pada rekaman latih pendek.

**K3 Brier rujukan V3 atas V1.** Lipatan V1 (latih 2003–2015 tanpa u) dan V3 (12 tahun acak, lima undian)
dijalankan ulang tanpa GB pada baris uji yang sama (tahun uji 2003–2011, ambang 2003–2025, label identik,
diperiksa). BSS Brier rujukan V3 (rata-rata undian) atas V1 untuk kalender, EF-P, EF-L; dua tahap dan t95, df 8.
Dua sasaran.

**K4 Lantai kejadian per kabupaten.** K1 dan K2 diulang per kabupaten dengan lantai 3 dan 5 deteksi (bawaan:
bulat(10 r) = 2). Sasaran nasional tak memakai lantai.

**R39 (aturan keputusan, ditulis sebelum dijalankan).**
(a) **GB atas EF-L per kabupaten, maju.** Bila satu rasio, median dan rata-rata tanpa tiga tersepi semuanya
negatif (bertanda sama dengan rata-rata rasio), naskah menulis bahwa dalam prakiraan maju per kabupaten tak ada
model yang ditunjukkan mengalahkan EF-L di bawah ketiga agregasi. Bila salah satu positif tetapi selang dua
tahap satu rasio (dan, untuk tanpa-tersepi, kedua selang) memuat nol, naskah menulis bahwa per kabupaten GB dan
EF-L tak terbedakan dalam prakiraan maju dan bahwa tanda rata-rata bergantung pada tiga tahun tersepi;
abstrak tidak lagi menulis vonis berarah, cukup "not distinguishable". Bila satu rasio atau tanpa-tersepi lolos
(selang mengecualikan nol, positif), naskah menulis bahwa per kabupaten GB di atas EF-L dalam prakiraan maju
kecuali di tahun-tahun tersepi, dan abstrak serta §9 menyebut bahwa hasil per kabupaten bergantung pada
agregasi.
(b) **EF-L atas kalender per kabupaten, maju (+0,141).** Bila satu rasio lolos dua tahap, klaim "EF-L beats the
raw calendar" dalam prakiraan maju tetap, disebut di bawah kedua agregasi; bila tidak, naskah menulis bahwa
perolehan EF-L atas kalender maju bertumpu pada tahun-tahun tersepi.
(c) **EF-L atas EF-P, maju.** Bila lolos kedua selang (rata-rata rasio) untuk suatu sasaran, naskah menulis
bahwa perolehan EF-L atas kalender dalam prakiraan maju bukan sekadar penghalusan; bila tidak, naskah menulis
angka EF-L atas EF-P dan bahwa dalam prakiraan maju bagian perolehan EF-L yang berasal dari penghalusan dan dari
anomali nasional tak dapat dipisahkan.
(d) **Brier rujukan V3 atas V1.** Bila untuk sasaran nasional Brier EF-P dan EF-L V3 lebih buruk daripada V1
di kedua selang, kalimat "the references fit the earlier years worse" tetap dengan angka ini; bila tidak,
kalimat itu dihapus dan naskah hanya menulis bahwa keunggulan nasional V3 atas V1 tidak dapat dibagi antara
model dan rujukan.
(e) **Lantai.** Bila pada lantai 3 dan 5 vonis GB atas EF-L per kabupaten maju (rata-rata rasio dan satu
rasio) sama arah dan sama kelulusannya dengan lantai 2, satu kalimat di §6.3 menyebut bahwa hasil sama pada
lantai 3 dan 5; bila vonis berubah, perubahan itu ditulis di §6.3 dan masuk keterbatasan, dan abstrak tidak
memakai vonis per kabupaten maju tanpa menyebut lantai.
(f) K1(e) deskriptif; tanpa aturan. Interval tingkat tahun MODIS mengasumsikan tahun bebas (lipatan berbagi
tahun latih): ditulis sebagai keterbatasan tanpa analisis.

Teks yang tak bergantung pada fase K dan diperbaiki dalam putaran teks yang sama: "later years" dua makna, judul §6.3, tahun uji maju 2012–2019 bukan tak terlihat, kode
2.7.0 tak memuat fase I–K, "no model" = satu
model, MDD80 di luar §6.3, R38c dibaca per sasaran disebut di §6.3.

## Hasil

Log (gitignored): `DL_FIRE_MODIS/patokan_modis_k_{kab,nas,kab_l3,kab_l5}.log`, skrip pada 4cb69a1. Jalan pertama
(7 Nov 22:58) dihentikan sistem karena memori rendah di tengah K3 per kabupaten; keempat jalan
diulang satu per satu dari awal (8 Nov). Lipatan R mereproduksi fase J persis (lantai 2: prevalensi, EF-P,
rata-rata rasio GB atas EF-L −0,0615 per kabupaten, +0,0496 nasional). Tiga tahun tersepi menurut aturan
(prevalensi terendah per sasaran) = 2010, 2016, 2017 untuk kedua sasaran (per kabupaten 2,10 / 2,27 / 2,08 %,
nasional 3,93 / 4,00 / 3,95 %).

Angka di bawah dicetak program dengan empat desimal; di naskah dibulatkan ke tiga desimal, setengah ke atas,
kecuali GB atas EF-L per kabupaten (a), yang tetap −0,061 seperti fase J (pembulatan program dari nilai penuh).

### K1 + K2 Prakiraan maju 2008–2019, lantai 2 (n = 12)

Kolom: (a) rata-rata rasio [dua tahap] t [tingkat tahun]; (b) satu rasio [dua tahap]; (c) median, tahun > 0, p uji
tanda; (d) tanpa 2010/2016/2017 [dua tahap] t [..] (n = 9); (e) rata-rata 2008–2015 / 2016–2019. Tebal = lolos
(a, d: kedua selang; b: dua tahap).

**Per kabupaten**

| Pasangan | (a) rata-rata rasio | (b) satu rasio | (c) median | (d) tanpa tiga tersepi | (e) 08–15 / 16–19 |
|---|---|---|---|---|---|
| GB atas EF-L | −0,0615 [−0,2104, +0,0485] t [−0,2078, +0,0847] | +0,0081 [−0,0832, +0,0623] (tanda berbalik) | +0,0336; 8/12; 0,388 | **+0,0545** [+0,0259, +0,0808] t [+0,0265, +0,0826] | −0,0070 / −0,1706 |
| EF-L atas kalender | **+0,1414** [+0,0380, +0,2570] t [+0,0080, +0,2748] | **+0,1146** [+0,0267, +0,2210] | +0,0516; 8/12; 0,388 | +0,0347 [−0,0195, +0,0975] t [−0,0346, +0,1039] | +0,1017 / +0,2209 |
| EF-L atas EF-P | +0,1193 [+0,0215, +0,2313] t [−0,0082, +0,2468] | **+0,0912** [+0,0120, +0,1926] | +0,0336; 8/12; 0,388 | +0,0153 [−0,0328, +0,0682] t [−0,0433, +0,0739] | +0,0755 / +0,2069 |
| EF-P atas kalender | **+0,0294** [+0,0160, +0,0468] t [+0,0108, +0,0479] | **+0,0258** [+0,0127, +0,0439] | +0,0258; 11/12; 0,006 | **+0,0208** [+0,0080, +0,0385] t [+0,0015, +0,0400] | +0,0332 / +0,0217 |
| GB atas kalender | **+0,1270** [+0,0658, +0,1960] t [+0,0530, +0,2009] | **+0,1217** [+0,0629, +0,1885] | +0,0856; 10/12; 0,039 | **+0,0879** [+0,0329, +0,1470] t [+0,0216, +0,1542] | +0,1251 / +0,1307 |
| GB atas EF-P | **+0,1023** [+0,0469, +0,1630] t [+0,0366, +0,1679] | **+0,0985** [+0,0435, +0,1554] | +0,0642; 10/12; 0,039 | **+0,0693** [+0,0198, +0,1233] t [+0,0105, +0,1281] | +0,0972 / +0,1125 |

MDD80 (a) GB atas EF-L 0,2044; (d) 0,0389. Per tahun EF-L atas kalender: 2008 +0,226, 2009 −0,013, 2010 +0,543,
2011 +0,048, 2012 −0,071, 2013 +0,031, 2014 −0,052, 2015 +0,101, 2016 +0,415, 2017 +0,427, 2018 +0,055, 2019 −0,013.
Per tahun EF-L atas EF-P: +0,157, −0,039, +0,499, +0,022, −0,078, +0,004, −0,061, +0,099, +0,391, +0,404, +0,045, −0,013.

**Nasional**

| Pasangan | (a) rata-rata rasio | (b) satu rasio | (c) median | (d) tanpa tiga tersepi | (e) 08–15 / 16–19 |
|---|---|---|---|---|---|
| GB atas EF-L | **+0,0496** [+0,0081, +0,0880] t [+0,0095, +0,0897] | **+0,0553** [+0,0199, +0,0897] | +0,0699; 9/12; 0,146 | **+0,0667** [+0,0313, +0,0971] t [+0,0310, +0,1023] | +0,0638 / +0,0211 |
| EF-L atas kalender | **+0,0874** [+0,0245, +0,1656] t [+0,0103, +0,1645] | **+0,0894** [+0,0286, +0,1642] | +0,0756; 8/12; 0,388 | +0,0285 [−0,0130, +0,0677] t [−0,0183, +0,0753] | +0,0432 / +0,1757 |
| EF-L atas EF-P | +0,0951 [+0,0153, +0,1920] t [−0,0025, +0,1928] | **+0,1037** [+0,0228, +0,1999] | +0,0621; 7/12; 0,774 | +0,0173 [−0,0281, +0,0613] t [−0,0353, +0,0699] | +0,0449 / +0,1956 |
| EF-P atas kalender | −0,0158 [−0,0499, +0,0121] t [−0,0491, +0,0175] | −0,0160 [−0,0499, +0,0129] | +0,0076; 7/12; 0,774 | +0,0107 [−0,0083, +0,0286] t [−0,0069, +0,0283] | −0,0067 / −0,0340 |
| GB atas kalender | **+0,1366** [+0,0801, +0,2027] t [+0,0743, +0,1989] | **+0,1397** [+0,0818, +0,2052] | +0,1141; 11/12; 0,006 | **+0,0937** [+0,0477, +0,1383] t [+0,0439, +0,1435] | +0,1046 / +0,2006 |
| GB atas EF-P | **+0,1449** [+0,0772, +0,2270] t [+0,0639, +0,2259] | **+0,1532** [+0,0802, +0,2373] | +0,1124; 11/12; 0,006 | **+0,0835** [+0,0377, +0,1277] t [+0,0305, +0,1365] | +0,1061 / +0,2225 |

### K3 Brier rujukan V3 (12 tahun acak) atas V1 (latih < 2016), tahun uji 2003–2011 (n = 9)

Positif = rujukan V3 lebih baik. Dua tahap [..] t [..]; MDD80.

| Rujukan | Per kabupaten | Nasional |
|---|---|---|
| Kalender | −0,0053 [−0,0894, +0,1014] t [−0,1274, +0,1169]; 0,1692 | −0,0932 [−0,1673, −0,0060] t [−0,2006, +0,0143]; 0,1489 |
| EF-P | −0,0250 [−0,1129, +0,0846] t [−0,1510, +0,1011]; 0,1746 | −0,0967 [−0,1838, +0,0003] t [−0,2185, +0,0250]; 0,1687 |
| EF-L | −0,0304 [−0,0762, +0,0380] t [−0,1039, +0,0431]; 0,1018 | −0,0550 [−0,1000, +0,0020] t [−0,1235, +0,0134]; 0,0948 |

Nasional rujukan V3 lebih buruk di titik (kalender dan EF-L 8 dari 9 tahun, EF-P 7 dari 9), tetapi hanya kalender mengecualikan
nol di dua tahap, dan tak satu pun di tingkat tahun.

### Vonis R39 (K1–K3)

**R39a** (GB atas EF-L per kabupaten, maju): satu rasio +0,008 dan median +0,034 positif, rata-rata tanpa tiga
tersepi **lolos** kedua selang (+0,055) → cabang ketiga: naskah menulis bahwa per kabupaten GB di atas EF-L dalam
prakiraan maju kecuali di tiga tahun tersepi, dan abstrak serta §9 menyebut bahwa hasil per kabupaten bergantung
pada agregasi. Kalimat "no model is shown to beat EF-L" dihapus.
**R39b** (EF-L atas kalender per kabupaten, maju): satu rasio lolos dua tahap (+0,115) → klaim tetap, di bawah
rata-rata rasio dan satu rasio. Tanpa tiga tersepi tidak lolos (+0,035): perolehan EF-L atas kalender maju datang
dari tahun-tahun sepi (R39b tak mensyaratkan ini; dilaporkan karena memperjelas).
**R39c** (EF-L atas EF-P, maju): rata-rata rasio tidak lolos kedua selang untuk kedua sasaran (tingkat tahun
memuat nol: per kabupaten [−0,008, +0,247], nasional [−0,003, +0,193]) → naskah menulis angka EF-L atas EF-P dan
bahwa bagian perolehan EF-L dari penghalusan dan dari anomali nasional tak dapat dipisahkan dalam prakiraan maju.
**R39d** (Brier rujukan V3 atas V1, nasional): EF-P dan EF-L tidak lebih buruk di kedua selang → kalimat "the
references fit the earlier years worse" dihapus; naskah menulis bahwa keunggulan nasional V3 atas V1 tidak dapat
dibagi antara model dan rujukan.

### K4 Lantai kejadian per kabupaten 3 dan 5 (prakiraan maju, n = 12)

Tiga tahun tersepi tetap 2010, 2016, 2017 (lantai 3: 2,11 / 2,06 / 2,05 %; lantai 5: 1,73 / 1,81 / 1,72 %).

| Pasangan | Lantai | (a) rata-rata rasio | (b) satu rasio | (c) median | (d) tanpa tiga tersepi |
|---|---|---|---|---|---|
| GB atas EF-L | 3 | −0,0588 [−0,1998, +0,0487] t [−0,2032, +0,0857] | +0,0091 [−0,0865, +0,0638] | +0,0345; 8/12 | **+0,0550** [+0,0278, +0,0830] t [+0,0257, +0,0843] |
| GB atas EF-L | 5 | −0,0830 [−0,2723, +0,0586] t [−0,2636, +0,0975] | +0,0036 [−0,0995, +0,0660] | +0,0435; 8/12 | **+0,0556** [+0,0251, +0,0834] t [+0,0283, +0,0830] |
| EF-L atas kalender | 3 | **+0,1432** [+0,0379, +0,2776] t [+0,0073, +0,2792] | **+0,1165** [+0,0301, +0,2406] | +0,0528; 8/12 | +0,0342 [−0,0197, +0,0952] t [−0,0365, +0,1049] |
| EF-L atas kalender | 5 | **+0,1567** [+0,0404, +0,2820] t [+0,0105, +0,3029] | **+0,1250** [+0,0310, +0,2418] | +0,0619; 8/12 | +0,0394 [−0,0210, +0,1134] t [−0,0367, +0,1156] |
| EF-L atas EF-P | 3 | +0,1208 [+0,0173, +0,2472] t [−0,0092, +0,2508] | **+0,0927** [+0,0127, +0,2075] | +0,0339; 8/12 | +0,0144 [−0,0320, +0,0656] t [−0,0452, +0,0741] |
| EF-L atas EF-P | 5 | +0,1338 [+0,0219, +0,2533] t [−0,0066, +0,2742] | **+0,1001** [+0,0110, +0,2086] | +0,0429; 8/12 | +0,0190 [−0,0348, +0,0779] t [−0,0455, +0,0835] |

**R39e**: pada lantai 3 dan 5 GB atas EF-L per kabupaten maju sama arah dan sama kelulusan dengan lantai 2
(rata-rata rasio negatif tak lolos, satu rasio positif tak lolos; tanpa tiga tersepi lolos) → satu kalimat di
§6.3: hasil sama pada lantai 3 dan 5. Pola EF-L atas kalender dan atas EF-P juga sama.

### Bacaan (K1–K4)

Pola per kabupaten dalam prakiraan maju terbelah oleh tingkat api: di sembilan tahun biasa GB di atas EF-L
(+0,055, lolos) dan EF-L tidak terbedakan dari kalender (+0,035); di tiga tahun tersepi EF-L jauh di atas kalender
(+0,415 sampai +0,543) dan GB jauh di bawah EF-L (−0,18 sampai −0,58). Rata-rata rasio membiarkan tiga tahun itu
menentukan tanda; satu rasio menimbang tahun menurut Brier rujukannya dan memberi nol (+0,008). Jadi tak ada
satu vonis per kabupaten maju yang bebas agregasi: EF-L bernilai di tahun sepi, GB di tahun biasa. Nasional
semua agregasi sepakat (GB atas EF-L positif, lolos di a, b, d). EF-L atas EF-P juga terbelah: hampir seluruhnya
dari tahun sepi.
