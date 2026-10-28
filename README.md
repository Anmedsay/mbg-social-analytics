# Kekhawatiran Publik terhadap Keamanan Pangan Program Makan Bergizi Gratis (MBG) di YouTube dan X

Repositori *data collection pipeline*  Tugas Kelompok 2 Analitika Media Sosial.

| Anggota | NPM |
|---|---|
| Erland Hafizh Aristovi | 2306165686 |
| Jasmine Rakhaila | 2306165774 |
| Leonita Cecilia | 2306165710 |
| Yeshua Marco Gracia Manurung | 2306210286 |

## Pertanyaan penelitian

1. **RQ1:** Apa saja jenis permasalahan keamanan dan kebersihan makanan MBG yang muncul dalam diskusi media sosial? *(topic modeling)*
2. **RQ2:** Bagaimana jenis permasalahan tersebut bervariasi antarwilayah di Indonesia? *(topic modeling + location-based analysis)*
3. **RQ3:** Apa saja kekhawatiran atau keluhan publik yang muncul? *(topic modeling + sentiment analysis)*

## Hasil pengumpulan data

| Platform | Metode | Dokumen |
|---|---|---|
| YouTube | YouTube Data API v3, 28 channel (27 media nasional + Badan Gizi Nasional) | 17.663 video, 1.067.690 comment dan reply |
| X | *Web scraping* (Playwright) dengan 6 kueri kata kunci keamanan pangan dan 5 hashtag | 24.535 post, reply, dan quote |
| **Gabungan** | | **1.109.888 dokumen** |

Periode: Januari 2025 sampai September 2026.

## Alur pipeline

```
 YouTube Data API v3                           Halaman pencarian X (Playwright)
        │                                                   │
        ▼                                                   ▼
 youtube/TK2_YouTube_Pipeline.ipynb              x/TK2_X_Scraping_Pipeline.ipynb
   cari video → validasi → comment                 kueri → scroll → deduplikasi
   → bersihkan → tandai spam → lokasi               → bersihkan → lokasi → anonimisasi
        │                                                   │
        ▼                                                   ▼
 data/interim/youtube/youtube_unified.csv          data/interim/x_unified.csv
        └──────────────────────┬────────────────────────────┘
                               ▼
              merge/TK2_Merge_YouTube_X.ipynb
       standardisasi skema → gabung → 8 validasi → flag spam/relevansi
                               ▼
           data/processed/raw_merged/mbg_merged.csv
                               ▼
         rq_preprocessing/TK2_RQ_Preprocessing.ipynb
        penanda kelayakan dan teks khusus RQ1, RQ2, RQ3
                               ▼
         data/processed/mbg_processed/mbg_processed.csv
```

## Struktur repositori

```
├── youtube/
│   ├── TK2_YouTube_Pipeline.ipynb      pengumpulan + pembersihan data YouTube
│   ├── youtube_channel_registry.json   bukti verifikasi ID 28 channel
│   └── figures/                        grafik EDA YouTube
├── x/
│   └── TK2_X_Scraping_Pipeline.ipynb   scraping + pembersihan data X
├── merge/
│   ├── TK2_Merge_YouTube_X.ipynb       penggabungan YouTube dan X
│   └── figures/                        grafik EDA gabungan
├── rq_preprocessing/
│   └── TK2_RQ_Preprocessing.ipynb      prapemrosesan per RQ
├── data/
│   ├── external/                       kamus wilayah (kabupaten/kota)
│   ├── interim/                        data bersih per platform
│   └── processed/                      data gabungan, kamus data, log
└── requirements.txt
```

## Cara menjalankan

**1. Persiapan**

```bash
python -m venv .venv
.venv\Scripts\activate          # Linux/macOS: source .venv/bin/activate
pip install -r requirements.txt
```

Untuk pipeline YouTube, buat file `.env` di folder utama berisi API key YouTube Data API v3.

```
YT_API_KEY=isi_api_key_anda
```

**2. Jalankan notebook secara berurutan**

| # | Notebook | Catatan |
|---|---|---|
| 1 | `youtube/TK2_YouTube_Pipeline.ipynb` | Set `COLLECT_DATA = True`, lalu **Run All**. Kuota API harian tidak cukup untuk sekali jalan, jadi ulangi Run All setiap hari sampai muncul `STATUS: LENGKAP`; progres disimpan otomatis. `COLLECT_DATA = False` menjalankan ulang pembersihan dan EDA tanpa API. |
| 2 | `x/TK2_X_Scraping_Pipeline.ipynb` | Notebook memasang Playwright sendiri. Login pertama dilakukan manual di jendela browser; sesi disimpan di `.auth/` (tidak di-commit). Run All; pencarian yang sudah selesai dilewati saat dijalankan ulang. |
| 3 | `merge/TK2_Merge_YouTube_X.ipynb` | Run All. Tanpa API, selesai dalam beberapa menit. Semua validasi harus PASS. |
| 4 | `rq_preprocessing/TK2_RQ_Preprocessing.ipynb` | Run All. Butuh RAM cukup besar; tutup kernel notebook lain terlebih dahulu. |

Notebook juga bisa dijalankan dari folder utama repositori atau dari Google Colab (pindah ke folder repositori dengan `%cd` sebelum Run All, simpan API key sebagai Colab Secret `YT_API_KEY`).

## Lokasi data

| Data | Lokasi |
|---|---|
| Kode, grafik, data bersih berukuran kecil, metadata run, kamus data, log | Repositori ini |
| Empat dataset besar (> 100 MB, melebihi batas GitHub) | Google Drive, lihat tabel di bawah |

**Dataset di Google Drive.** Setelah diunduh, taruh setiap file di path yang tertulis, lalu notebook berikutnya bisa langsung dijalankan tanpa mengumpulkan ulang.

| File | Ukuran | Taruh di path | Dihasilkan oleh | Unduh |
|---|---|---|---|---|
| `youtube_unified.csv` | 566 MB | `data/interim/youtube/` | Pipeline YouTube | [Google Drive](https://drive.google.com/file/d/1THhdzuU5J-G-iZ30oT2EC9YX2NsaIIPv/view?usp=drive_link) |
| `comments_clean.csv` | 413 MB | `data/interim/youtube/` | Pipeline YouTube | [Google Drive](https://drive.google.com/file/d/1XodX0gF5Ys4rbaT5-kLIDA2Z-g0fnQqL/view?usp=drive_link) |
| `mbg_merged.csv` | 610 MB | `data/processed/raw_merged/` | Pipeline merge | [Google Drive](https://drive.google.com/file/d/1DJgUZKJmES_pHripBJpS1yk5ZkgiDEzR/view?usp=drive_link) |
| `mbg_processed.csv` | 887 MB | `data/processed/mbg_processed/` | Prapemrosesan per RQ | [Google Drive](https://drive.google.com/file/d/1VfyK_6EUDLwVF7HN9QUXvdKqZqjkzGYR/view?usp=drive_link) |