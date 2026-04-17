# Project Statistic AI

Proyek ini berisi pipeline klasifikasi untuk dataset **UCI Lower Limb EMG (SEMG_DB1)**. Alur utamanya mencakup loading data, preprocessing sinyal EMG, ekstraksi fitur, pelatihan beberapa model, evaluasi repeated hold-out, lalu penyimpanan log dan visualisasi hasil.

## Struktur Folder

```text
.
|-- configs/
|   `-- uci_baseline.yaml
|-- src/
|   |-- load_uci.py
|   |-- preprocessing.py
|   |-- feature_extraction.py
|   |-- models.py
|   |-- evaluation.py
|   `-- visualization.py
|-- main_uci.py
|-- logs/
`-- figures/
```

## Dependensi

Pastikan Python sudah terpasang, lalu install library yang dipakai proyek ini:

```bash
pip install numpy pandas scipy scikit-learn matplotlib pyyaml tqdm torch
```

## Konfigurasi

Pengaturan utama ada di file `configs/uci_baseline.yaml`, misalnya:

- `data_path`: lokasi dataset UCI Lower Limb EMG
- `window`: ukuran window dalam milidetik
- `overlap`: overlap antar-window
- `models`: daftar model yang akan diuji
- `n_repeats`: jumlah pengulangan evaluasi
- `use_gpu`: gunakan GPU jika tersedia

Sebelum menjalankan program, pastikan `data_path` mengarah ke folder dataset yang berisi `N_TXT` dan `A_TXT`.

## Menjalankan Proyek

Jalankan script utama:

```bash
python main_uci.py
```

Jika diperlukan, sesuaikan path konfigurasi default di `main_uci.py` agar mengarah ke `configs/uci_baseline.yaml`.

## Output

Hasil eksekusi akan disimpan ke:

- `logs/` untuk file JSON hasil evaluasi
- `figures/` untuk grafik bar dan confusion matrix

Folder output tersebut sudah dimasukkan ke `.gitignore`, sehingga file hasil run tidak ikut ter-commit.

## Git Ignore

File `.gitignore` pada proyek ini sudah mengabaikan:

- cache Python seperti `__pycache__/`
- folder output seperti `logs/` dan `figures/`
- virtual environment
- file lokal `check_file.py`
