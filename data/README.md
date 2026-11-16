# Data

Folder ini menampung seluruh dataset. **Isinya di-`.gitignore`** — hanya
`README.md` dan `.gitkeep` yang di-commit. Jangan menaruh data dummy di repo.

## Struktur

```
data/
├── raw/                      # Foto multi-bean asli dari kamera (belum dianotasi)
├── annotated/                # Ekspor dari Roboflow (format YOLO, 5 kelas)
│   ├── train/
│   │   ├── images/           # citra .jpg/.png
│   │   └── labels/           # anotasi .txt (YOLO)
│   ├── valid/{images,labels}/
│   └── test/{images,labels}/
├── crops_raw/                # Crop biji individual — MENTAH, tanpa augmentasi
│   ├── train/{good,black,chipped,dry_pods,insect_damage}/
│   ├── valid/{...}/
│   └── test/{...}/
└── crops/                    # DATASET CNN — siap latih (train sudah diaugmentasi)
    ├── train/{good,black,chipped,dry_pods,insect_damage}/
    ├── valid/{...}/          # salinan crops_raw, tanpa augmentasi
    ├── test/{...}/           # salinan crops_raw, tanpa augmentasi
    └── augmentation_manifest.json
```

### Kenapa `crops_raw` dan `crops` dipisah

`crops_raw` adalah artefak **mentah** hasil crop dari anotasi; `crops` adalah
**masukan pelatihan** hasil augmentasi offline. Pemisahan ini membuat augmentasi
bisa diulang dengan multiplier berbeda tanpa meng-crop ulang, dan membuat jelas
berkas mana yang asli dan mana yang buatan.

Struktur keduanya mengikuti konvensi PyTorch `ImageFolder`.

### Konvensi nama berkas

| Pola | Artinya |
| ---- | ------- |
| `<nama_foto>_bean_<idx:03d>.jpg` | crop mentah di `crops_raw/` |
| `<...>_orig.jpg` | salinan citra asli di `crops/train/` |
| `<...>_aug01.jpg`, `_aug02.jpg` | varian hasil augmentasi offline |
| nama apa adanya | `crops/valid/` dan `crops/test/` (disalin tanpa sufiks) |

**Tidak boleh ada berkas `_aug*` di `crops/valid/` atau `crops/test/`.** Metrik
evaluasi harus diukur pada citra asli.

## Format label YOLO

Tiap file `.txt` di `annotated/*/labels/` berisi satu baris per biji:

```
<class_id> <x_center> <y_center> <width> <height>
```

Seluruh koordinat ternormalisasi (0..1) terhadap dimensi gambar.

`class_id` **harus** mengikuti urutan `CLASS_NAMES` di
`configs/base_config.py`:

| `class_id` | Nama kelas | Bahasa Indonesia | Definisi |
| ---------- | ---------- | ---------------- | -------- |
| `0` | `good` | Baik | Biji utuh tanpa cacat |
| `1` | `black` | Hitam | Biji menghitam (fermentasi berlebih / busuk) |
| `2` | `chipped` | Serpihan | Biji pecah atau terkelupas |
| `3` | `dry_pods` | Polong Kering | Polong/kulit kering yang terbawa |
| `4` | `insect_damage` | Rusak Serangga | Biji berlubang / tergerek serangga |

> **Urutan kelas di Roboflow harus persis seperti tabel di atas.** Kalau
> tertukar, seluruh label menjadi salah tanpa memunculkan satu pun error —
> pipeline tetap berjalan dan metriknya tetap keluar.

## Strategi anotasi: penuh vs sebagian

Sebagian foto dianotasi **penuh** (semua biji berlabel, termasuk `good`),
sebagian lagi hanya bijinya yang **cacat**. Keduanya sah: biji `good` yang tidak
berlabel pada foto *partial* tidak ikut di-crop, jadi tidak ada label palsu yang
masuk dataset.

Konsekuensinya, foto *partial* tidak menyumbang sampel `good` sama sekali.
Karena itu kedua kelompok harus tersebar **proporsional** di train/valid/test —
skrip validasi melaporkan komposisi `full=N partial=M` per split.

## Alur pengisian

1. **`raw/`** — letakkan foto mentah hasil pemotretan.

2. **`annotated/`** — hasil anotasi & ekspor Roboflow (format YOLOv8).
   Pastikan: 5 kelas urut benar, Auto-Orient ON, Resize None, **Augmentation
   kosong**, split per foto.

3. **`crops_raw/`** — dihasilkan otomatis:

   ```bash
   python -m scripts.crop_dataset
   ```

   Skrip membaca tiap pasangan gambar-label, mengonversi bounding box ke piksel,
   memotongnya, lalu menyimpan ke folder per kelas. Gambar tanpa file label
   dilewati dengan peringatan; koordinat yang melewati tepi di-clip otomatis.

   Gunakan `--padding-ratio 0.05` bila ingin menyertakan sedikit konteks di
   sekitar tepi biji.

4. **`crops/`** — dihasilkan otomatis:

   ```bash
   python -m scripts.augment_dataset           # menambah
   python -m scripts.augment_dataset --clean   # setelah ganti versi dataset
   ```

   Split `train` diaugmentasi sesuai `AUGMENTATION_MULTIPLIERS`; `valid` dan
   `test` disalin apa adanya.

## Gotcha: ganti versi dataset wajib pakai `--clean`

Nama berkas Roboflow memuat hash versi (`IMG_..._jpg.rf.<hash>.jpg`), dan hash
itu **berubah tiap versi**. Crop versi baru karena itu bernama beda dari crop
versi lama, sehingga yang lama **tidak tertimpa** — keduanya menumpuk.

Bahayanya bukan sekadar dataset ganda: foto yang dulu ada di `test` bisa pindah
ke `train` di versi baru. Crop lamanya tetap di folder `test`, crop barunya
masuk `train` → **biji yang sama ada di train dan test sekaligus**. Kebocoran
diam-diam, dan akurasi naik semu tanpa ketahuan.

Urutan aman saat ganti versi: arsipkan `results/` → kosongkan `data/annotated/`
→ extract versi baru → ratakan → validasi → crop ulang →
`augment_dataset --clean` → latih ulang ketiganya. Pakai skill
`/import-dataset`.

Verifikasi bahwa tidak ada kebocoran:

```bash
python .claude/skills/check-dataset/validate.py --crops
```
