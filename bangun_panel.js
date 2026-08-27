/**
 * Bangun panel bulanan titik panas per kabupaten, Kalimantan Barat.
 *
 * Masukan  : berkas CSV FIRMS (arsip dan NRT) + batas GADM level 2
 * Keluaran : panel_bulanan.csv  — satu baris per kabupaten per bulan
 *
 * Pakai:
 *   node bangun_panel.js <gadm41_IDN_2.json> <keluaran.csv> <berkas1.csv> [berkas2.csv ...]
 *
 * Penyaringan baku, sama dengan seluruh analisis lain di folder ini:
 *   - confidence != "l"   (buang keyakinan rendah)
 *   - type == 0           (hanya kebakaran vegetasi)
 *   - titik harus jatuh di dalam poligon kabupaten Kalimantan Barat
 *
 * Catatan: GADM 4.1 terbit 2022. Pemekaran wilayah setelah itu belum tercermin —
 * misalnya Kabupaten Mempawah masih bernama "Pontianak". Sebut ini di bagian Data
 * artikel.
 */

const fs = require('fs');

const [gadmPath, outPath, ...csvPaths] = process.argv.slice(2);
if (!gadmPath || !outPath || csvPaths.length === 0) {
  console.error('Pakai: node bangun_panel.js <gadm.json> <keluaran.csv> <fire1.csv> [fire2.csv ...]');
  process.exit(1);
}

// --- 1. Muat batas kabupaten, siapkan bbox agar pencarian cepat -------------

const gadm = JSON.parse(fs.readFileSync(gadmPath, 'utf8'));
const kab = gadm.features
  .filter((f) => /Kalimantan\s*Barat/i.test(f.properties.NAME_1 || ''))
  .map((f) => {
    // Normalkan MultiPolygon dan Polygon menjadi daftar cincin luar + lubang
    const polys = f.geometry.type === 'MultiPolygon'
      ? f.geometry.coordinates
      : [f.geometry.coordinates];
    let minx = Infinity, miny = Infinity, maxx = -Infinity, maxy = -Infinity;
    polys.forEach((p) => p[0].forEach(([x, y]) => {
      if (x < minx) minx = x; if (x > maxx) maxx = x;
      if (y < miny) miny = y; if (y > maxy) maxy = y;
    }));
    return { nama: f.properties.NAME_2, tipe: f.properties.TYPE_2, polys, bbox: [minx, miny, maxx, maxy] };
  });

console.error('kabupaten dimuat: ' + kab.length);

// --- 2. Titik di dalam poligon, metode ray casting --------------------------

function dalamCincin(x, y, ring) {
  let inside = false;
  for (let i = 0, j = ring.length - 1; i < ring.length; j = i++) {
    const xi = ring[i][0], yi = ring[i][1], xj = ring[j][0], yj = ring[j][1];
    if ((yi > y) !== (yj > y) && x < ((xj - xi) * (y - yi)) / (yj - yi) + xi) inside = !inside;
  }
  return inside;
}

function cariKabupaten(x, y) {
  for (const k of kab) {
    const b = k.bbox;
    if (x < b[0] || x > b[2] || y < b[1] || y > b[3]) continue;
    for (const poly of k.polys) {
      if (!dalamCincin(x, y, poly[0])) continue;
      let diLubang = false;
      for (let h = 1; h < poly.length; h++) if (dalamCincin(x, y, poly[h])) { diLubang = true; break; }
      if (!diLubang) return k.nama;
    }
  }
  return null;
}

// --- 3. Baca CSV, saring, hitung --------------------------------------------

const panel = new Map();   // "kabupaten|YYYY-MM" -> jumlah
const bulanSet = new Set();
let stat = { total: 0, lolosSaring: 0, luarKabupaten: 0, masuk: 0, confRendah: 0, bukanVegetasi: 0 };

for (const p of csvPaths) {
  const isi = fs.readFileSync(p, 'utf8').split('\n');
  const kolom = isi[0].split(',').map((s) => s.trim());
  const iLat = kolom.indexOf('latitude');
  const iLon = kolom.indexOf('longitude');
  const iTgl = kolom.indexOf('acq_date');
  const iConf = kolom.indexOf('confidence');
  const iType = kolom.indexOf('type');          // -1 pada berkas NRT
  if (iLat < 0 || iLon < 0 || iTgl < 0 || iConf < 0) {
    console.error('lewati, kolom tidak lengkap: ' + p);
    continue;
  }

  for (let i = 1; i < isi.length; i++) {
    const b = isi[i];
    if (!b) continue;
    const f = b.split(',');
    stat.total++;
    if (f[iConf] === 'l') { stat.confRendah++; continue; }
    if (iType >= 0 && +f[iType] !== 0) { stat.bukanVegetasi++; continue; }
    stat.lolosSaring++;
    const nama = cariKabupaten(+f[iLon], +f[iLat]);
    if (!nama) { stat.luarKabupaten++; continue; }
    stat.masuk++;
    const bln = f[iTgl].slice(0, 7);
    bulanSet.add(bln);
    const kunci = nama + '|' + bln;
    panel.set(kunci, (panel.get(kunci) || 0) + 1);
  }
  console.error('selesai: ' + p);
}

// --- 4. Tulis panel lengkap, termasuk kombinasi bernilai nol ----------------

const bulan = [...bulanSet].sort();
const baris = ['kabupaten,tipe,bulan,tahun,bulan_ke,titik_panas'];
for (const k of kab) {
  for (const bl of bulan) {
    const n = panel.get(k.nama + '|' + bl) || 0;
    baris.push([k.nama, k.tipe, bl, bl.slice(0, 4), +bl.slice(5, 7), n].join(','));
  }
}
fs.writeFileSync(outPath, baris.join('\n') + '\n');

console.error('');
console.error('titik dibaca            : ' + stat.total);
console.error('  dibuang keyakinan l   : ' + stat.confRendah);
console.error('  dibuang bukan vegetasi: ' + stat.bukanVegetasi);
console.error('  lolos penyaringan     : ' + stat.lolosSaring);
console.error('  di luar kabupaten     : ' + stat.luarKabupaten);
console.error('  masuk panel           : ' + stat.masuk);
console.error('');
console.error('panel: ' + kab.length + ' kabupaten x ' + bulan.length + ' bulan = ' + (baris.length - 1) + ' baris');
console.error('rentang: ' + bulan[0] + ' s/d ' + bulan[bulan.length - 1]);
console.error('ditulis: ' + outPath);
