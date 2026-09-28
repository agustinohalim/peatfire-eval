/*
 * bangun_panel_nasional.js
 *
 * Menyusun panel kabupaten-bulan untuk SELURUH Indonesia dari CSV FIRMS,
 * memakai batas administratif GADM 4.1 level 2. Langkah 2 dari tiga sesudah
 * unduhan, sesuai `Artikel_2_Rencana.md` bagian I.
 *
 * Masukan  : direktori berisi CSV FIRMS + batas GADM level 2
 * Keluaran : CSV panel lengkap, termasuk kombinasi bernilai nol
 *
 *   node bangun_panel_nasional.js <gadm41_IDN_2.json> <keluaran.csv> <direktori_csv>
 *
 * Direktori, bukan daftar berkas, karena 1.019 berkas melewati batas panjang
 * baris perintah Windows.
 *
 * HUBUNGANNYA DENGAN bangun_panel.js
 *
 * Berkas ini TIDAK menggantikannya. `bangun_panel.js` membangun panel Kalbar
 * yang dipakai Artikel 1, dan harus tetap utuh supaya hasil artikel itu dapat
 * diulang. Geometri titik-dalam-poligon di sini disalin apa adanya dari sana,
 * supaya metodenya terbukti sama, bukan sekadar diklaim sama.
 *
 * Tiga perbedaan, semuanya disengaja:
 *
 *   1. Tanpa saringan provinsi. `bangun_panel.js` menyaring Kalimantan Barat.
 *   2. Kunci panel memakai GID_2, bukan NAME_2. Secara nasional NAME_2 tidak
 *      unik — 'Banjar' dipakai dua kali. GID_2 unik untuk 502 kabupaten.
 *   3. Ada indeks kisi. Dengan 502 kabupaten dan 3,4 juta titik, penelusuran
 *      linear seperti versi Kalbar akan makan waktu berjam-jam.
 *
 * CATATAN DATA yang harus diketahui sebelum membaca keluarannya:
 *
 *   - NAME_1 pada berkas GADM ini spasinya hilang: 'KalimantanBarat', bukan
 *     'Kalimantan Barat'. Dibiarkan apa adanya, tidak "diperbaiki", karena
 *     itu nilai sumbernya dan mengubahnya memutus kesamaan dengan Artikel 1.
 *   - GADM 4.1 memuat 502 kabupaten, bukan ±514 seperti taksiran
 *     `Pertanyaan_Riset.md` bagian A. Batas GADM mendahului pemecahan
 *     administratif terbaru; itu sudah tercatat sebagai keterbatasan naskah
 *     Artikel 1 butir 9.
 *
 * Penyaringan sama persis dengan Artikel 1: `confidence != l`, `type == 0`,
 * dan titik harus berada di dalam poligon kabupaten. Kotak koordinat hanya
 * dipakai saat mengunduh, bukan sebagai batas wilayah.
 */

'use strict';

const fs = require('fs');
const path = require('path');

const [gadmPath, outPath, csvDir] = process.argv.slice(2);
if (!gadmPath || !outPath || !csvDir) {
  console.error('Pakai: node bangun_panel_nasional.js <gadm.json> <keluaran.csv> <direktori_csv>');
  process.exit(1);
}

// --- 1. Muat batas kabupaten, seluruh provinsi ------------------------------

const gadm = JSON.parse(fs.readFileSync(gadmPath, 'utf8'));
const kab = gadm.features.map((f) => {
  const polys = f.geometry.type === 'MultiPolygon'
    ? f.geometry.coordinates
    : [f.geometry.coordinates];
  let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
  polys.forEach((p) => p[0].forEach(([x, y]) => {
    if (x < minx) minx = x; if (x > maxx) maxx = x;
    if (y < miny) miny = y; if (y > maxy) maxy = y;
  }));
  return {
    gid: f.properties.GID_2,
    prov: f.properties.NAME_1,
    nama: f.properties.NAME_2,
    tipe: f.properties.TYPE_2,
    polys,
    bbox: [minx, miny, maxx, maxy],
  };
});

console.error('kabupaten dimuat: ' + kab.length);
console.error('provinsi        : ' + new Set(kab.map((k) => k.prov)).size);

// --- 2. Indeks kisi satu derajat -------------------------------------------
// Tiap sel menyimpan kabupaten yang bbox-nya menyentuhnya. Titik hanya
// diperiksa terhadap kabupaten dalam selnya sendiri.

const SEL = 1.0;
const kisi = new Map();
const kunciSel = (ix, iy) => ix + ':' + iy;

for (const k of kab) {
  const [minx, miny, maxx, maxy] = k.bbox;
  for (let ix = Math.floor(minx / SEL); ix <= Math.floor(maxx / SEL); ix++) {
    for (let iy = Math.floor(miny / SEL); iy <= Math.floor(maxy / SEL); iy++) {
      const s = kunciSel(ix, iy);
      if (!kisi.has(s)) kisi.set(s, []);
      kisi.get(s).push(k);
    }
  }
}
console.error('sel kisi terisi : ' + kisi.size);

// --- 3. Titik di dalam poligon, ray casting, sama dengan bangun_panel.js ----

function dalamCincin(x, y, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0], yi = ring[i][1], xj = ring[j][0], yj = ring[j][1];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function cariKabupaten(x, y) {
  const kandidat = kisi.get(kunciSel(Math.floor(x / SEL), Math.floor(y / SEL)));
  if (!kandidat) return null;
  for (const k of kandidat) {
    const b = k.bbox;
    if (x < b[0] || x > b[2] || y < b[1] || y > b[3]) continue;
    for (const poly of k.polys) {
      if (!dalamCincin(x, y, poly[0])) continue;
      let diLubang = false;
      for (let h = 1; h < poly.length; h++) if (dalamCincin(x, y, poly[h])) { diLubang = true; break; }
      if (!diLubang) return k.gid;
    }
  }
  return null;
}

// --- 4. Baca CSV, saring, hitung -------------------------------------------

const berkasCsv = fs.readdirSync(csvDir)
  .filter((n) => n.toLowerCase().endsWith('.csv'))
  .sort();
if (berkasCsv.length === 0) {
  console.error('tidak ada CSV di ' + csvDir);
  process.exit(1);
}
console.error('berkas CSV      : ' + berkasCsv.length);
console.error('');

const panel = new Map();   // "GID_2|YYYY-MM" -> jumlah
const bulanSet = new Set();
const stat = {
  total: 0, confRendah: 0, bukanVegetasi: 0,
  lolosSaring: 0, luarKabupaten: 0, masuk: 0,
};
const mulai = Date.now();

berkasCsv.forEach((nama, idx) => {
  const isi = fs.readFileSync(path.join(csvDir, nama), 'utf8').split('\n');
  const kolom = isi[0].split(',').map((s) => s.trim());
  const iLat = kolom.indexOf('latitude');
  const iLon = kolom.indexOf('longitude');
  const iTgl = kolom.indexOf('acq_date');
  const iConf = kolom.indexOf('confidence');
  const iType = kolom.indexOf('type');
  if (iLat < 0 || iLon < 0 || iTgl < 0 || iConf < 0) {
    console.error('lewati, kolom tidak lengkap: ' + nama);
    return;
  }

  for (let i = 1; i < isi.length; i++) {
    const b = isi[i];
    if (!b) continue;
    const f = b.split(',');
    stat.total++;
    if (f[iConf] === 'l') { stat.confRendah++; continue; }
    if (iType >= 0 && +f[iType] !== 0) { stat.bukanVegetasi++; continue; }
    stat.lolosSaring++;
    const gid = cariKabupaten(+f[iLon], +f[iLat]);
    if (!gid) { stat.luarKabupaten++; continue; }
    stat.masuk++;
    const bln = f[iTgl].slice(0, 7);
    bulanSet.add(bln);
    const kunci = gid + '|' + bln;
    panel.set(kunci, (panel.get(kunci) || 0) + 1);
  }

  if ((idx + 1) % 100 === 0 || idx + 1 === berkasCsv.length) {
    const detik = (Date.now() - mulai) / 1000;
    console.error(
      `[${idx + 1}/${berkasCsv.length}] ${nama}  masuk ${stat.masuk.toLocaleString()}  ${detik.toFixed(0)}s`
    );
  }
});

// --- 5. Tulis panel lengkap, termasuk kombinasi bernilai nol ---------------

const bulan = [...bulanSet].sort();
const baris = ['gid,provinsi,kabupaten,tipe,bulan,tahun,bulan_ke,titik_panas'];
for (const k of kab) {
  for (const bl of bulan) {
    const n = panel.get(k.gid + '|' + bl) || 0;
    baris.push([
      k.gid,
      JSON.stringify(k.prov),
      JSON.stringify(k.nama),
      JSON.stringify(k.tipe),
      bl,
      bl.slice(0, 4),
      +bl.slice(5, 7),
      n,
    ].join(','));
  }
}
fs.writeFileSync(outPath, baris.join('\n') + '\n');

console.error('');
console.error('titik dibaca            : ' + stat.total.toLocaleString());
console.error('  dibuang keyakinan l   : ' + stat.confRendah.toLocaleString());
console.error('  dibuang bukan vegetasi: ' + stat.bukanVegetasi.toLocaleString());
console.error('  lolos penyaringan     : ' + stat.lolosSaring.toLocaleString());
console.error('  di luar kabupaten     : ' + stat.luarKabupaten.toLocaleString());
console.error('  masuk panel           : ' + stat.masuk.toLocaleString());
console.error('');
console.error('kabupaten               : ' + kab.length);
console.error('bulan                   : ' + bulan.length + '  (' + bulan[0] + ' sampai ' + bulan[bulan.length - 1] + ')');
console.error('baris panel             : ' + (kab.length * bulan.length).toLocaleString());
console.error('waktu                   : ' + ((Date.now() - mulai) / 60000).toFixed(1) + ' menit');
console.error('');
console.error('Titik "di luar kabupaten" itu laut, Malaysia, Brunei, PNG, dan');
console.error('Timor-Leste yang berada di dalam kotak unduh. Dibuangnya memang benar.');
