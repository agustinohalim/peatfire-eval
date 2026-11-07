# Hasil Tolok Ukur Nasional 11 — fase F revisi PeatFireBench (EF-L sebagai urutan dan siaga; GB dengan tingkat nasional)

**7 November 2026.** Lanjutan `Hasil_Patokan_10.md` (fase E), revisi PeatFireBench. Pemicu: telaah
internal pra-kirim kelima atas draf revisi (catatan telaah tidak dirilis). Temuan utamanya: EF-L, yang
diusulkan sebagai baseline, belum dinilai dengan AUC-PR dan sebagai aturan siaga; kalimat abstrak
"ambang lebih baik daripada kuota" membandingkan kuota yang tidak disetel; EF-L memakai masukan (jumlah
titik panas nasional bulan lalu) yang tidak diberikan ke model yang dipelajari.

## Rancangan fase F (ditulis 7 Nov 2026, SEBELUM skrip mana pun dijalankan)

Semua analisis di sini **pasca-hoc dan eksploratori**: EF-L sendiri dirumuskan sesudah tahun uji dinilai
(fase E). Naskah melabelnya demikian di Lampiran A. Data, model, dan prediksi fase A–E tidak berubah kecuali
F3. Selang: kabupaten (1.000, tahun tetap), dua tahap (1.000, tahun lalu kabupaten), t atas enam tahun
(df 5); seed 2026; tahun biasa = S1 2020–2025, ekstrem = S2 2015, 2014, 2019. "Lolos" = batas bawah selang
dua tahap dan t keduanya > 0 (seperti R16, tanpa Holm). Skrip: `patokan_revisi_tingkat.py`, satu sasaran per
jalan (memori), log `DL_FIRE_NASIONAL/penuh/patokan_revisi_tingkat_{kab,nas}.log`.

### F1. EF-L sebagai urutan

AUC-PR dalam-tahun `ef1_tingkat` (EF-L) dan `ef1_halus`, dari `patokan_revisi_peluang_<sasaran>.csv`, pada
baris yang sama dengan Tabel 2 (per kabupaten: kabupaten `ada_positif`). Dilaporkan: EF-L − rujukan pasangan
(EF-1m per kabupaten, EF-3m nasional); RF, GB, GB tersetel, logistik EF-1m + jeda − EF-L; tahun biasa dengan
ketiga selang, tahun ekstrem per tahun dengan selang kabupaten.

**R29.** (a) Bila EF-L − rujukan pasangan punya kedua selang di bawah nol untuk suatu sasaran, naskah
menyebut dua baseline untuk sasaran itu: EF-L untuk peluang, rujukan pasangan untuk urutan. Bila tidak,
EF-L disebut satu baseline untuk keduanya dengan menyebut selisih AUC-PR-nya. (b) Model disebut lebih baik
daripada EF-L sebagai urutan hanya bila lolos; selain itu "not shown to rank better than EF-L".

### F2. EF-L sebagai aturan siaga (per kabupaten, seperti Tabel 9 dan 10)

Aturan untuk EF-L, GB, dan EF-1m: (i) siaga bila p > a; (ii) kuota 10 % kabupaten per bulan; (iii) ambang
tetap = kuantil 0,9 skor tahun latih; (iv) ambang yang memaksimalkan nilai rata-rata per tahun di tahun latih,
untuk tiap a; (v) ambang terbaik di tahun uji (batas atas, bukan aturan). Skor tahun latih: GB dan EF-1m dari
`patokan_revisi_oof.csv` (luar-lipatan, sama dengan fase C4/D7); EF-L dari nilai pas baris latih (kalender dan
anomali tinggalkan-tahun-sendiri, seperti saat EF-L dilatih; logistik tiga parameter, jadi dalam-sampel
untuk koefisiennya, dicatat sebagai batasan). Fungsi `kuota`, `nilai`, `ukur`, `ambang_oof` disalin dari
`patokan_revisi_siaga2.py` tanpa perubahan.

Tambahan (jawaban atas kritik kuota tak disetel): (vi) **kuota disetel** — bagian siaga q dipilih dari
{0,02, 0,04, …, 0,30} untuk tiap a dengan nilai rata-rata per tahun tertinggi di tahun latih (skor tahun latih
seperti iii–iv), lalu diterapkan sebagai kuota bulanan q di tahun uji.

**R30.** Selisih nilai ekonomi relatif GB − EF-L pada aturan (i) dan (iv) di a = 0,05, 0,1, 0,2, 0,3, 0,5,
tahun biasa, dua tahap dan t. GB disebut lebih bernilai daripada EF-L sebagai siaga untuk suatu aturan bila
lolos pada sedikitnya tiga dari lima a; selain itu "not distinguishable" dan abstrak tidak memuat klaim
siaga GB atas kalender.

**R31.** Bila kuota disetel (vi) untuk GB masih di bawah ambang yang memaksimalkan nilai (iv) dengan selang
dua tahap di bawah nol pada sedikitnya tiga dari lima a, naskah boleh menulis bahwa kuota kalah "even when
its share is tuned on training years". Bila tidak, kalimat kuota dihapus dari abstrak dan §6.8 menyebut
bahwa kekalahan kuota tergantung pada bagiannya yang tetap.

### F3. GB yang diberi tingkat api nasional

Gradient boosting dilatih ulang dengan fitur penuh + satu kolom `anomali_nasional` = log1p(jumlah titik panas
nasional bulan lalu) − rata-rata kolom itu di tahun latih untuk bulan kalender yang sama (baris latih:
tinggalkan-tahun-sendiri, sama dengan EF-L). Parameter baku (300 pohon, depth 3, lr 0,05, subsample dan
colsample 0,9, reg_lambda 1, benih 42), S1 2020–2025 dan S2, kedua sasaran. Keluaran
`DL_FIRE_NASIONAL/patokan_revisi_tingkat_gb_<sasaran>.csv` (tidak masuk repo).

**R32.** BSS dan selisih AUC-PR dalam-tahun `xgb_nas` atas EF-L dan atas `xgb`, tahun biasa. (a) Bila
`xgb_nas` lolos atas EF-L dalam BSS untuk suatu sasaran, tesis berubah untuk sasaran itu: model yang diberi
tingkat nasional mengalahkan EF-L sebagai peluang; judul, abstrak, simpulan ditulis ulang. (b) Bila tidak,
naskah menulis bahwa memberi GB jumlah titik panas nasional tidak membuatnya terbedakan dari EF-L (dengan
angka). (c) AUC-PR `xgb_nas` − EF-L dibaca dengan aturan R29b.

### F4. Tahun ekstrem lawan EF-L

BSS GB, RF atas EF-L di tiap tahun ekstrem dengan selang kabupaten (1.000), kedua sasaran (fase E hanya titik).
Tidak ada aturan; dilaporkan apa adanya di §6.2.

### Tidak dikerjakan di fase F (dicatat sekarang)

Data MODIS 2003–2011 untuk menambah tahun uji (perlu panel baru; tetap batasan), kalender bersyarat
ENSO/IOD sebagai rujukan lain (dicatat sebagai karya lanjutan), kebocoran ambang untuk skor peluang (tetap
batasan), konsultasi lembaga.

## Hasil fase F (dijalankan 7 Nov 2026 sesudah komit dbd74d7)

Log: `DL_FIRE_NASIONAL/penuh/patokan_revisi_tingkat_kab.log` dan `_nas.log`. Pemeriksaan: EF-L dihitung ulang
sama dengan `patokan_revisi_peluang_<sasaran>.csv` (selisih maks 1,1e-16), tidak ada baris kosong; BSS GB dan RF
atas EF-L sama dengan Tabel 4 fase E (per kabupaten GB +0,067 [−0,020, +0,152]; nasional GB +0,037
[−0,094, +0,152], RF +0,058 [−0,000, +0,127]). Penyimpangan: tidak ada. Catatan: kuota disetel memilih
q = 0,30, batas atas kisi, pada a = 0,05 di semua lipatan; kisi tidak diperlebar (rancangan tetap).

### F1. AUC-PR dalam-tahun, tahun biasa (selang kabupaten / dua tahap / t95)

AUC-PR rata-rata. Per kabupaten: EF-L 0,225, ef1_halus 0,233, EF-1m 0,245, RF 0,288, GB 0,276, GB tersetel
0,299, logistik EF-1m + jeda 0,246, GB + nasional 0,272. Nasional: EF-L 0,695, ef1_halus 0,690, EF-3m 0,688,
RF 0,738, GB 0,726, GB tersetel 0,733, logistik 0,713, GB + nasional 0,724.

| Selisih | Per kabupaten | Nasional |
|---|---|---|
| EF-L − rujukan | −0,020 kab [−0,035, −0,004] 2t [−0,050, +0,011] t [−0,052, +0,012] | +0,007 2t [−0,012, +0,025] t [−0,012, +0,025] |
| ef1_halus − rujukan | −0,012 2t [−0,025, +0,000] t [−0,025, +0,001] | +0,001 2t [−0,007, +0,010] |
| RF − EF-L | **+0,063** 2t [+0,020, +0,108] t [+0,021, +0,105] | **+0,044** 2t [+0,014, +0,080] t [+0,007, +0,080] |
| GB − EF-L | **+0,051** 2t [+0,003, +0,109] t [+0,007, +0,096] | +0,032 2t [−0,000, +0,070] t [−0,010, +0,074] |
| GB tersetel − EF-L | **+0,075** 2t [+0,012, +0,129] t [+0,019, +0,130] | **+0,039** 2t [+0,008, +0,077] t [+0,000, +0,077] |
| logistik EF-1m + jeda − EF-L | +0,022 2t [−0,028, +0,065] | +0,018 2t [−0,015, +0,059] |
| GB + nasional − EF-L | +0,047 2t [−0,006, +0,115] t [−0,012, +0,107] | **+0,030** 2t [+0,004, +0,064] t [+0,000, +0,059] |
| GB + nasional − GB | −0,004 2t [−0,032, +0,014] | −0,002 2t [−0,014, +0,009] |
| GB + nasional − rujukan | +0,027 2t [−0,047, +0,113] | **+0,036** 2t [+0,007, +0,070] t [+0,007, +0,066] |

Tebal = lolos (kedua batas bawah > 0). Tahun ekstrem (selang kabupaten), EF-L − rujukan: per kabupaten 2015
+0,027 [+0,009, +0,046], 2014 −0,176 [−0,206, −0,146], 2019 −0,025 [−0,053, +0,003]; nasional +0,009
[−0,002, +0,019], −0,058 [−0,075, −0,040], −0,023 [−0,039, −0,007]. GB, RF, GB + nasional di atas EF-L di
ketiga tahun ekstrem untuk kedua sasaran (selang kabupaten di atas nol; GB per kabupaten +0,074, +0,130,
+0,096; nasional +0,039, +0,067, +0,057).

**R29.** (a) EF-L − rujukan tidak punya kedua selang di bawah nol untuk sasaran mana pun (per kabupaten
−0,020, dua tahap [−0,050, +0,011]; nasional +0,007). Maka EF-L disebut **satu baseline**, dengan selisih
AUC-PR-nya disebut. (b) Sebagai urutan, model yang lolos atas EF-L: per kabupaten RF, GB, GB tersetel;
nasional RF, GB tersetel, GB + nasional. Lainnya "not shown to rank better than EF-L". Konsekuensi untuk
naskah: klaim "no model shown to beat EF-L" hanya berlaku untuk **peluang**; sebagai urutan beberapa model
mengalahkan EF-L walaupun tidak terbukti mengalahkan kalender mentah per kabupaten (EF-L sedikit di bawah
kalender sebagai urutan, tidak terbedakan).

### F3/F4. BSS dalam-tahun, tahun biasa

| BSS | Per kabupaten | Nasional |
|---|---|---|
| GB atas EF-L | +0,067 kab [+0,040, +0,096] 2t [−0,020, +0,152] t [−0,048, +0,181] | +0,037 2t [−0,094, +0,152] t [−0,140, +0,214] |
| RF atas EF-L | +0,030 2t [−0,055, +0,102] | +0,058 2t [−0,000, +0,127] t [−0,023, +0,138] |
| GB + nasional atas EF-L | +0,091 kab [+0,070, +0,116] 2t [+0,007, +0,180] t [−0,022, +0,203] | +0,065 2t [−0,016, +0,154] t [−0,051, +0,182] |
| GB + nasional atas GB | +0,024 2t [−0,034, +0,069] | +0,022 2t [−0,014, +0,065] |
| GB + nasional atas rujukan | **+0,428** 2t [+0,248, +0,622] | **+0,277** 2t [+0,156, +0,436] |
| EF-L atas rujukan | **+0,376** 2t [+0,170, +0,574] | **+0,232** 2t [+0,113, +0,363] |

**R32.** (a) tidak terpenuhi: GB + nasional tidak lolos atas EF-L dalam BSS untuk sasaran mana pun (per
kabupaten selang dua tahap di atas nol tetapi t95 tidak). (b) Naskah menulis bahwa memberi GB anomali titik
panas nasional menaikkan BSS-nya atas EF-L menjadi +0,091 per kabupaten dan +0,065 nasional, tetapi tidak
membuatnya terbedakan dari EF-L. (c) AUC-PR GB + nasional − EF-L: lolos nasional (+0,030), tidak per
kabupaten. Tesis tidak berubah.

F4, tahun ekstrem, BSS atas EF-L [selang kabupaten]:

| | 2015 | 2014 | 2019 |
|---|---|---|---|
| GB, per kabupaten | +0,145 [+0,112, +0,176] | −0,063 [−0,104, −0,024] | +0,197 [+0,160, +0,234] |
| RF, per kabupaten | −0,004 [−0,033, +0,027] | −0,106 [−0,150, −0,067] | +0,126 [+0,095, +0,157] |
| GB + nasional, per kabupaten | +0,191 [+0,159, +0,223] | +0,074 [+0,037, +0,111] | +0,200 [+0,160, +0,240] |
| EF-L atas EF-1m | +0,302 [+0,278, +0,327] | +0,031 [−0,006, +0,065] | +0,103 [+0,076, +0,127] |
| GB, nasional | +0,178 [+0,130, +0,218] | +0,078 [+0,033, +0,120] | +0,209 [+0,169, +0,254] |
| RF, nasional | +0,171 [+0,133, +0,208] | +0,068 [+0,027, +0,105] | +0,186 [+0,149, +0,225] |
| GB + nasional, nasional | +0,210 [+0,165, +0,252] | +0,096 [+0,053, +0,138] | +0,205 [+0,165, +0,250] |
| EF-L atas EF-3m | +0,311 [+0,279, +0,339] | +0,157 [+0,119, +0,193] | +0,132 [+0,099, +0,163] |

### F2. Siaga per kabupaten (nilai ekonomi relatif, a = 0,05 / 0,1 / 0,2 / 0,3 / 0,5; rata-rata tahun)

| Aturan | Model | Tahun biasa | Tahun ekstrem |
|---|---|---|---|
| (i) p > a | GB | +0,380 / +0,219 / +0,077 / −0,001 / −0,021 | +0,285 / +0,468 / +0,579 / +0,487 / +0,293 |
| | EF-L | +0,099 / −0,093 / −0,003 / +0,048 / +0,005 | +0,357 / +0,490 / +0,592 / +0,433 / +0,192 |
| | EF-1m | +0,031 / −0,668 / −1,364 / −1,692 / −1,655 | +0,119 / +0,245 / +0,440 / +0,314 / +0,080 |
| (iv) ambang nilai, tahun latih | GB | +0,283 / +0,224 / +0,130 / +0,081 / +0,009 | +0,367 / +0,461 / +0,568 / +0,442 / +0,287 |
| | EF-L | +0,228 / +0,164 / +0,093 / +0,017 / +0,003 | +0,238 / +0,464 / +0,584 / +0,433 / +0,153 |
| | EF-1m | +0,011 / −0,140 / −0,307 / −0,048 / +0,000 | −0,477 / −0,051 / +0,042 / +0,028 / +0,000 |
| (iii) ambang kuantil 0,9 | GB | +0,241 / +0,263 / +0,159 / +0,011 / −0,462 | −0,254 / +0,352 / +0,602 / +0,538 / +0,280 |
| | EF-L | +0,048 / +0,122 / +0,088 / +0,027 / −0,169 | −0,155 / +0,375 / +0,586 / +0,488 / +0,115 |
| (ii) kuota 10 % | GB | +0,004 / −0,047 / −0,556 / −1,242 / −3,439 | −2,413 / −0,652 / +0,119 / +0,114 / −0,060 |
| | EF-L | −0,061 / −0,112 / −0,629 / −1,326 / −3,556 | −2,480 / −0,686 / +0,098 / +0,090 / −0,093 |
| (vi) kuota disetel | GB | +0,009 / −0,341 / −0,113 / −0,161 / −0,557 | −1,222 / −0,652 / −0,008 / +0,038 / +0,011 |
| | EF-L | −0,143 / −0,415 / −0,094 / −0,187 / −0,593 | −1,269 / −0,775 / −0,012 / +0,033 / +0,003 |
| (v) terbaik (batas) | GB | +0,457 / +0,344 / +0,194 / +0,126 / +0,066 | +0,459 / +0,555 / +0,638 / +0,558 / +0,391 |
| | EF-L | +0,375 / +0,262 / +0,142 / +0,106 / +0,058 | +0,398 / +0,535 / +0,611 / +0,497 / +0,257 |

Kuota 10 %, tahun biasa: hit GB 0,405, EF-L 0,347, EF-1m 0,350; hit GB − EF-L +0,058 2t [−0,050, +0,174].
Bagian kuota disetel (tahun biasa): GB 0,30 di a = 0,05, 0,12–0,30 di a = 0,1, 0,02–0,08 di a ≥ 0,2.

**R30.** GB − EF-L, tahun biasa: aturan (i) +0,280 / +0,312 / +0,080 / −0,049 / −0,027, lolos dua tahap di
a = 0,05 dan 0,1 tetapi t95 tidak (a = 0,05: 2t [+0,053, +0,617], t [−0,091, +0,651]); aturan (iv)
+0,055 / +0,060 / +0,037 / +0,063 / +0,006, tidak lolos di a mana pun. Tidak ada aturan yang lolos di ≥ 3
dari 5 a: **GB tidak terbedakan dari EF-L sebagai siaga**. Tahun ekstrem: GB di bawah EF-L pada 2014 di
kedua aturan, di atasnya pada 2015 dan 2019 di sebagian besar a.

**R31.** Kuota disetel − ambang nilai (iv), GB: −0,274 / −0,565 / −0,244 / −0,242 / −0,566, dua tahap di
bawah nol di kelima a (t95 di bawah nol di a = 0,1, 0,2, 0,3, 0,5). EF-L searah (dua tahap di bawah nol di
empat dari lima a). Syarat terpenuhi: naskah boleh menulis bahwa kuota kalah "even when its share is tuned
on training years". Kuota 10 % − kuota disetel, GB: −0,005 / +0,294 / −0,443 / −1,081 / −2,882.

### Ringkas untuk naskah

1. EF-L tetap satu baseline (R29a), tetapi klaim "not shown to beat" dibatasi ke **peluang di tahun biasa**.
   Sebagai urutan EF-L tidak lebih baik daripada kalender (−0,020 per kabupaten, +0,007 nasional), dan RF
   serta GB tersetel (juga GB per kabupaten) mengurutkan lebih baik daripada EF-L di kedua selang.
2. GB yang diberi titik panas nasional tetap tidak terbedakan dari EF-L sebagai peluang (+0,091, +0,065).
3. Sebagai siaga, GB tidak terbedakan dari EF-L; keduanya jauh di atas EF-1m pada aturan (i).
4. Kalimat abstrak kuota dipertahankan dengan "even when its share is tuned on training years".
5. Tahun ekstrem: GB di atas EF-L di 2015 dan 2019 (kedua sasaran) dan di 2014 nasional, di bawahnya pada
   2014 per kabupaten; EF-L sebagai urutan di bawah kalender pada 2014.
