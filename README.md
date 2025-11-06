# ConverterChess — Gambar ke FEN

Ubah gambar papan catur menjadi string FEN menggunakan CNN (gaya OCR) yang mengklasifikasikan bidak pada setiap kotak. Termasuk skrip pelatihan dan prediksi, serta opsi debugging/diagnostik.

## Fitur Utama
- Deteksi papan dan perspektif warp dari foto bebas (OpenCV)
- Pemotongan 8×8 kotak dan klasifikasi batch
- Model dapat dilatih (ResNet18) pada dataset Anda (PyTorch + TorchVision)
- Perakitan FEN dengan kompresi kotak kosong yang benar
- Mode batch: prediksi satu gambar atau satu folder (rekursif) dan ekspor CSV
- Opsi orientasi FEN: as-is (default), flip vertikal/horizontal, rot-180, atau auto-orientation
- Opsi pra-proses: skip-warp, crop-percent, board-size; overlay dapat diproyeksikan kembali ke gambar asli
- Skrip bantu ramah Windows (.cmd)

## Struktur Proyek
- `src/converter/` — Kode inti (deteksi papan, dataset, model, pelatihan, prediksi)
- `data/` — Tempatkan data pelatihan/validasi/tes Anda
- `weights/` — Model tersimpan (.pt) dan metadata kelas
- `scripts/` — Pembungkus perintah untuk Windows CMD

## Instalasi
1) Siapkan Python 3.9–3.12
2) (Disarankan) Buat virtual environment dan instal dependensi:

```cmd
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```

Jika PyTorch gagal dipasang melalui pip, ikuti panduan resmi sesuai platform CUDA/CPU Anda:
https://pytorch.org/get-started/locally/

Catatan Windows: Jika muncul error `ImportError: No module named cv2`, pastikan paket OpenCV terpasang (requirements sudah menyertakan `opencv-python`). Anda bisa mengulang:

```cmd
pip install --force-reinstall opencv-python
```

## Menyiapkan Data
Kelas yang digunakan (13 kelas):
- empty
- white_pawn, white_knight, white_bishop, white_rook, white_queen, white_king
- black_pawn, black_knight, black_bishop, black_rook, black_queen, black_king

Letakkan gambar (PNG/JPG) pada struktur berikut:
- `data/train/<kelas>/...`
- `data/val/<kelas>/...` (validasi)
- `data/test/<kelas>/...` (opsional)

Tips: Gunakan `scripts\extract_squares.cmd` pada foto papan untuk memotong 64 kotak otomatis, lalu beri label secara manual untuk mempercepat pembuatan dataset.

## Melatih Model
Melatih ResNet18 pada gambar kotak catur:

```cmd
scripts\train.cmd --data-root data --epochs 10 --batch-size 64 --lr 0.0005 --img-size 224 --out-dir weights
```

Argumen umum:
- `--data-root` folder berisi `train/` dan `val/`
- `--epochs` jumlah epoch
- `--batch-size` ukuran batch
- `--img-size` ukuran gambar input (default 224)
- `--out-dir` lokasi simpan checkpoint dan mapping kelas

Keluaran:
- `weights/best.pt` — checkpoint terbaik menurut akurasi validasi
- `weights/classes.json` — daftar nama kelas sesuai urutan pelatihan

## Prediksi (Gambar -> FEN)
Deteksi papan, bagi 64 kotak, klasifikasikan tiap kotak, lalu keluarkan FEN.

Contoh satu baris (Windows CMD) — gambar tunggal:

```cmd
scripts\predict.cmd --image "path\to\board.jpg" --weights "weights\best.pt" --save-overlays --output-dir "results"
```

Contoh satu baris — satu folder (rekursif) dan simpan CSV:

```cmd
scripts\predict.cmd --input "path\to\folder_gambar" --recursive --weights "weights\best.pt" --save-overlays --output-dir "results"
```

Catatan penting:
- Orientasi FEN default adalah `as-is` (tidak diputar). Untuk memaksakan orientasi lain gunakan `--fen-orientation rot-180|flip-v|flip-h`, atau `--auto-orientation` untuk memilih otomatis.
- Notasi FEN standar: bidak PUTIH huruf BESAR, bidak hitam huruf kecil. Opsi non-standar tersedia via `--white-lowercase` bila dibutuhkan.

Opsi berguna lainnya:
- `--skip-warp` lewati perspektif warp (cocok untuk screenshot papan yang sudah sejajar)
- `--crop-percent 0.08` pangkas margin seragam 8% sebelum dipotong 8×8
- `--board-size 1024` atur resolusi papan ter-warp (default otomatis = 8×`img-size`)
- `--overlay-on-original` saat warp aktif, proyeksikan overlay kembali ke gambar asli
- `--save-overlays` simpan visualisasi anotasi prediksi
- `--debug-save-squares` simpan 64 kotak hasil potongan untuk inspeksi
- `--topk 3` tampilkan 3 prediksi teratas per kotak (mode satu gambar)

Keluaran default:
- File `*.fen.txt` per gambar (FEN pada satu baris)
- Jika `--output-dir` dipakai: `results/predictions.csv` berisi pasangan `image,fen`

### Daftar Parameter (Predict)
Argumen utama untuk `scripts\predict.cmd` (meneruskan ke `src.converter.predict`):

- `--image <file>`: Prediksi satu gambar.
- `--input <file|dir>`: Prediksi file tunggal atau folder gambar.
- `--output-dir <dir>`: Direktori keluaran (overlay/FEN/CSV).
- `--recursive`: Scan subfolder saat `--input` adalah folder.
- `--weights <file.pt>`: Checkpoint model (wajib).
- `--classes <classes.json>`: Urutan kelas; default mengikuti checkpoint. Gunakan `--force-classes` untuk memaksa.
- `--force-classes`: Paksa gunakan `--classes` meski checkpoint punya daftar kelas sendiri.
- `--flip`: Paksa rotasi 180° (menggantikan autoflip).
- `--skip-warp`: Lewati perspektif warp (gunakan gambar apa adanya).
- `--no-autoflip`: Matikan heuristik auto-rotate 180° berdasarkan petak gelap.
- `--crop-percent <0.0-0.45>`: Pangkas margin seragam sebelum split (mis. 0.08).
- `--board-size <int>`: Resolusi papan hasil warp. 0=auto (8×`img-size`).
- `--overlay-on-original`: Saat warp aktif, proyeksikan overlay ke gambar asli juga.
- `--save-overlays`: Simpan visualisasi hasil prediksi.
- `--debug-save-squares`: Simpan 64 potongan kotak untuk inspeksi/label.
- `--try-orientations`: Cetak FEN untuk beberapa orientasi (mode satu gambar).
- `--topk <int>`: Tampilkan Top-K prediksi per kotak (mode satu gambar).
- `--img-size <int>`: Ukuran input model (default 224).
- `--white-lowercase`: Notasi non-standar (PUTIH huruf kecil, hitam besar) untuk overlay/FEN.
- `--fen-orientation {as-is,flip-v,flip-h,rot-180}`: Orientasi FEN akhir (default: as-is).
- `--auto-orientation`: Coba semua orientasi, pilih terbaik dengan heuristik (tidak memengaruhi overlay).
- `--prefer-empty`: Pasca-proses: condong memilih `empty` saat keyakinan rendah.
- `--empty-threshold <float>`: Ambang prob maksimum kelas teratas untuk memicu prefer-empty.
- `--empty-min <float>`: Ambang minimum prob `empty` agar dipilih.

## Tips & Troubleshooting
- Jika hasil tidak sesuai arah papan, coba `--auto-orientation` atau uji beberapa orientasi dengan `--try-orientations`.
- Jika overlay terlihat mengecil atau buram, naikkan `--board-size` atau aktifkan `--skip-warp` untuk screenshot.
- Pastikan urutan kelas konsisten dengan checkpoint. Secara default, urutan kelas diambil dari checkpoint `best.pt`. Anda dapat memaksa `--classes weights\classes.json` dengan `--force-classes` jika perlu.

## Author
N4co

## Lisensi
Proyek ini disediakan apa adanya (as-is). Anda sepenuhnya memiliki data dan model yang Anda latih.

---

## Daftar Parameter (Train)
Argumen untuk `scripts\train.cmd` (meneruskan ke `src.converter.train`):

- `--data-root <dir>`: Akar data berisi `train/` dan `val/` (default: `data`).
- `--epochs <int>`: Jumlah epoch pelatihan (default: 10).
- `--batch-size <int>`: Ukuran batch (default: 64).
- `--lr <float>`: Learning rate (default: 0.0005).
- `--img-size <int>`: Ukuran input model (default: 224).
- `--out-dir <dir>`: Direktori simpan model/kelas (default: `weights`).
- `--num-workers <int>`: Worker DataLoader (default: 2).

Keluaran: `best.pt` dan `classes.json` pada `--out-dir`.

## Labeler (GUI Pelabelan)
Jalankan langsung file Python:

```cmd
python scripts\labeler.py --mode squares --input_dir "debug_squares" --output_root "data" --val-split 0.1
```

atau mode papan utuh yang sudah top-down:

```cmd
python scripts\labeler.py --mode board --board_img "path\to\board.jpg" --output_root "data" --val-split 0.1
```

Argumen untuk `scripts/labeler.py`:

- `--mode {squares,board}`: squares=label gambar kotak; board=pecah 8×8 dan label per kotak.
- `--input_dir <dir>`: Direktori gambar (mode squares).
- `--board_img <file>`: Gambar papan top-down (mode board).
- `--output_root <dir>`: Akar dataset keluaran (default: `data`).
- `--val-split <float>`: Probabilitas kirim sampel ke split `val` (default: 0.1).
- `--rows <int>`: Jumlah baris grid (default: 8).
- `--cols <int>`: Jumlah kolom grid (default: 8).

Hotkeys ringkas: 0/e=empty; p/r/n/b/q/k=PUTIH; P/R/N/B/Q/K=hitam; SPACE=skip; u=undo; [ ]=prev/next (squares); g=grid (board); ESC=keluar.
