/**
 * Curah hujan bulanan CHIRPS per kabupaten — Google Earth Engine
 *
 * Menutup satu baris ⬜ pada `Artikel_2_Rencana.md` bagian C: curah hujan
 * CHIRPS untuk 502 kabupaten, 2012-2025, supaya dapat digabungkan ke
 * `panel_nasional_fitur.csv`.
 *
 * ==========================================================================
 * MENGAPA LEWAT EARTH ENGINE, BUKAN DIUNDUH SENDIRI
 * ==========================================================================
 *
 * Titik panas FIRMS berupa titik, jadi bisa diunduh sebagai CSV dan diolah
 * dengan Python biasa — itu yang dikerjakan `unduh_firms_nasional.py`.
 *
 * CHIRPS berupa raster 0,05 derajat. Mengambil rata-rata per kabupaten berarti
 * memotong raster dengan poligon, dan itu perlu GDAL atau rasterio yang tidak
 * terpasang di mesin ini. Memasangnya bisa, tetapi Earth Engine mengerjakan
 * pemotongannya di sisi server dan mengembalikan langsung tabel yang
 * diperlukan. Untuk satu peubah, itu jalan yang paling pendek.
 *
 * ==========================================================================
 * PRASYARAT — kerjakan berurutan, jangan dilompati
 * ==========================================================================
 *
 * 1. Akun Earth Engine noncommercial. Caranya ada di kepala
 *    `Skrip_Uji_FIRMS_Earth_Engine.js`, termasuk syarat Google Cloud project
 *    yang wajib sejak pertengahan 2025.
 *
 * 2. **Unggah batas kabupaten sebagai aset tabel Earth Engine.** Unggahan tabel
 *    hanya menerima shapefile atau CSV, bukan GeoJSON (dicek 25 September 2026),
 *    jadi jalankan dulu `python gadm_ke_csv_wkt.py`, lalu unggah
 *    `gadm41_IDN_2_wkt.csv`: Assets → New → CSV file → isi "Geometry column"
 *    dengan `WKT`. Setelah tugas di tab Tasks selesai, salin Asset ID-nya ke
 *    `ASSET_GADM` di bawah.
 *
 *    **Jangan memakai `FAO/GAUL/2015/level2` sebagai jalan pintas.** Batas GAUL
 *    berbeda dari GADM, dan pengenal kabupatennya juga berbeda, sehingga
 *    hasilnya tidak dapat digabungkan dengan panel yang sudah dibangun. Seluruh
 *    gunanya justru terletak pada GID_2 yang sama.
 *
 * 3. Jalankan, lalu ambil CSV-nya di Google Drive. Ekspor Earth Engine masuk
 *    ke tab Tasks, dan harus ditekan Run di sana — tidak jalan sendiri.
 *
 * ==========================================================================
 * YANG HARUS DIKETAHUI SEBELUM MEMAKAI HASILNYA
 * ==========================================================================
 *
 * **CHIRPS 0,05 derajat kira-kira 5,5 km.** Beberapa kabupaten kota lebih kecil
 * daripada satu sel. Untuk kabupaten seperti itu, `reduceRegions` dengan
 * reducer rata-rata dapat mengembalikan **null**, bukan nol. Null berarti
 * "tidak terukur", dan **tidak boleh diisi nol** — nol berarti tidak hujan,
 * dan itu pernyataan yang berbeda. Tangani sebagai data hilang saat menggabung.
 *
 * Cara memeriksanya sudah disiapkan: bagian D di bawah mencetak jumlah
 * kabupaten yang null pada satu bulan contoh, sebelum ekspor dijalankan.
 *
 * **CHIRPS cakupannya 50°S sampai 50°N, mulai 1981.** Indonesia tercakup
 * seluruhnya, dan rentang 2012-2025 aman.
 *
 * **Satuannya milimeter per bulan**, hasil penjumlahan curah hujan harian.
 *
 * ==========================================================================
 * SESUDAHNYA
 * ==========================================================================
 *
 * CSV keluarannya berkolom `gid`, `bulan`, `hujan_mm`. Gabungkan ke
 * `panel_nasional_fitur.csv` pada kunci `gid` + `bulan`, lalu bangun jeda
 * 1 sampai 3 bulan seperti perlakuan ONI dan DMI. Jangan memasukkan curah
 * hujan bulan berjalan sebagai peubah penjelas untuk bulan yang sama — itu
 * kebocoran, karena hujan dan kebakaran terjadi serentak.
 */

// ==========================================================================
// Parameter
// ==========================================================================

// Ganti dengan Asset ID hasil unggahan Anda sendiri.
var ASSET_GADM = 'projects/ganti-dengan-project-anda/assets/gadm41_IDN_2_wkt';

var TAHUN_AWAL = 2012;
var TAHUN_AKHIR = 2025;

// CHIRPS 0,05 derajat. Jangan diperkecil — memperhalus skala tidak menambah
// informasi, hanya menambah waktu hitung.
var SKALA = 5566;

var NAMA_EKSPOR = 'chirps_bulanan_kabupaten_' + TAHUN_AWAL + '_' + TAHUN_AKHIR;

// ==========================================================================
// A. Muat kabupaten
// ==========================================================================

var kab = ee.FeatureCollection(ASSET_GADM);
print('kabupaten dimuat:', kab.size());
print('contoh properti:', kab.first());

// ==========================================================================
// B. CHIRPS harian, dijumlahkan menjadi bulanan
// ==========================================================================

var chirps = ee.ImageCollection('UCSB-CHG/CHIRPS/DAILY')
  .filterDate(TAHUN_AWAL + '-01-01', (TAHUN_AKHIR + 1) + '-01-01');

// Daftar bulan sebagai (tahun, bulan). Dibuat eksplisit supaya bulan tanpa
// data pun tetap muncul sebagai baris, bukan hilang diam-diam.
var bulanList = ee.List.sequence(TAHUN_AWAL, TAHUN_AKHIR)
  .map(function (th) {
    return ee.List.sequence(1, 12).map(function (bl) {
      return ee.Dictionary({ tahun: th, bulan: bl });
    });
  })
  .flatten();

function hujanBulanan(d) {
  d = ee.Dictionary(d);
  var th = ee.Number(d.get('tahun'));
  var bl = ee.Number(d.get('bulan'));
  var awal = ee.Date.fromYMD(th, bl, 1);
  var akhir = awal.advance(1, 'month');
  var label = awal.format('YYYY-MM');

  var total = chirps.filterDate(awal, akhir).sum().rename('hujan_mm');

  return total
    .reduceRegions({
      collection: kab.select(['GID_2']),
      reducer: ee.Reducer.mean(),
      scale: SKALA,
    })
    .map(function (f) {
      return ee.Feature(null, {
        gid: f.get('GID_2'),
        bulan: label,
        // Bisa null untuk kabupaten yang lebih kecil daripada satu sel CHIRPS.
        // Dibiarkan null; jangan diisi nol.
        hujan_mm: f.get('mean'),
      });
    });
}

var hasil = ee.FeatureCollection(bulanList.map(hujanBulanan)).flatten();

// ==========================================================================
// C. Periksa satu bulan sebelum mengekspor semuanya
// ==========================================================================

var contoh = ee.FeatureCollection(hujanBulanan(
  ee.Dictionary({ tahun: 2015, bulan: 9 })
));
print('--- contoh September 2015 ---');
print('baris:', contoh.size());
print('lima baris pertama:', contoh.limit(5));

// ==========================================================================
// D. Berapa kabupaten yang tidak terukur pada bulan contoh
// ==========================================================================
// Angka ini yang menentukan apakah curah hujan layak dipakai apa adanya, atau
// kabupaten kecil perlu diperlakukan khusus. Baca sebelum menekan Run di Tasks.

var kosong = contoh.filter(ee.Filter.notNull(['hujan_mm'])).size();
print('kabupaten TERUKUR pada contoh   :', kosong);
print('kabupaten TIDAK terukur (null)  :', contoh.size().subtract(kosong));
print('Kalau yang null banyak, jangan diisi nol. Catat sebagai data hilang,');
print('atau naikkan reducer ke mean atas buffer kecil di sekitar sentroid.');

// ==========================================================================
// E. Ekspor
// ==========================================================================
// Setelah Run ditekan di tab Tasks, keluarannya masuk ke Google Drive.
// 502 kabupaten x 168 bulan = 84.336 baris. Bila tugasnya gagal karena batas
// waktu, jalankan per tahun: ubah TAHUN_AWAL dan TAHUN_AKHIR, lalu gabungkan
// CSV-nya di Python.

Export.table.toDrive({
  collection: hasil,
  description: NAMA_EKSPOR,
  fileFormat: 'CSV',
  selectors: ['gid', 'bulan', 'hujan_mm'],
});

print('');
print('Ekspor disiapkan sebagai:', NAMA_EKSPOR);
print('Buka tab Tasks di sebelah kanan, lalu tekan Run.');
