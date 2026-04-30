# 🧠 UTS Machine Learning — Deep Learning vs Classical ML

> **Nama:** Balqis Eka Nurfadisyah
> **NIM:** 1202220223
> **Mata Kuliah:** Pengantar Deep Learning


---

## 📋 Deskripsi Proyek

Proyek ini membandingkan pendekatan **Deep Learning** dengan **Classical Machine Learning** pada tiga tipe data yang berbeda: tabular, image, dan text. Tujuan utama bukan sekadar mencapai akurasi tertinggi, melainkan menunjukkan kemampuan analisis dalam memilih, membandingkan, dan menjustifikasi pendekatan yang digunakan.

---

## 🗂️ Struktur Repository

```
├── kasus1_titanic.ipynb          # Tabular — Binary Classification
├── kasus2_digit_recognizer.ipynb # Image — Multi-class Classification
├── kasus3_disaster_tweets.ipynb  # Text — NLP Binary Classification
├── laporan_uts.pdf               # Laporan lengkap (PDF)
└── README.md
```

---

## 📊 Ringkasan Tiga Kasus

### Kasus 1 — Titanic (Tabular Data)
**Dataset:** Kaggle Titanic · ~891 baris · Binary classification (Survived 0/1)

| Model | Val Accuracy | F1-Score |
|---|---|---|
| Random Forest (default) | 83.24% | 0.7727 |
| Random Forest (GridSearch) | 81.56% | 0.7402 |
| XGBoost (default) | 81.01% | 0.7536 |
| XGBoost (GridSearch) | 82.12% | 0.7681 |
| MLP (threshold 0.35) | 82.12% | 0.7778 |
| TabNet (threshold 0.55) | 83.80% | 0.7852 |
| FT-Transformer (threshold 0.30) | 82.68% | 0.7891 |
| **Ensemble RF+XGB+MLP** | **83.80%** | **0.7943** |

**Kaggle Score:** `0.76315`

**Insight utama:** Pada dataset kecil (~900 baris), MLP adalah pilihan DL paling justified — FT-Transformer hanya unggul 3 sampel dari 179, namun membutuhkan 26K parameter vs MLP yang hanya 929 parameter. RF Default mengalahkan RF GridSearch, membuktikan bahwa *extensive tuning tidak selalu bermanfaat di dataset kecil*.

---

### Kasus 2 — Digit Recognizer MNIST (Image Data)
**Dataset:** Kaggle Digit Recognizer · 42,000 gambar · 28×28 px · 10 kelas

| Model | Val Accuracy | Macro F1 |
|---|---|---|
| PCA (90% variance) + Random Forest | — | — |
| HOG + SVM (RBF, C=10) | — | — |
| **CNN + Data Augmentation** | **~98-99%** | **~0.98+** |

**Teknik yang digunakan:**
- **Classical:** PCA reduksi 784 → N dimensi (90% variance), HOG dengan `pixels_per_cell=(7,7)`, `cells_per_block=(2,2)`
- **Deep Learning:** CNN 2-layer Conv → MaxPool → FC → Dropout(0.25)
- **Data Augmentation:** RandomRotation(10°) + RandomAffine(translate=0.1)

**Arsitektur CNN:**
```
Input (1×28×28)
  → Conv2d(1→16, 3×3) → ReLU → MaxPool(2×2)   [→ 16×14×14]
  → Conv2d(16→32, 3×3) → ReLU → MaxPool(2×2)  [→ 32×7×7]
  → Flatten → FC(1568→128) → ReLU → Dropout(0.25)
  → FC(128→10) [output: 10 kelas]
```

**Pola misclassification:** HOG+SVM gagal pada digit dengan gaya penulisan tidak konvensional (terutama 9↔8, 3↔2 akibat kemiripan gradien lokal). CNN gagal pada kasus ambiguitas global shape (9→7, 1→2) — namun secara keseluruhan lebih robust.

---

### Kasus 3 — NLP Disaster Tweets (Text Data)
**Dataset:** Kaggle NLP Disaster Tweets · Binary classification (bencana nyata vs tidak)

*In progress — hasil akan diupdate setelah submission.*

**Rencana implementasi:**
- **Classical:** TF-IDF (unigram + bigram) + Logistic Regression / Naive Bayes / Linear SVM
- **Deep Learning:** Embedding + BiLSTM / DistilBERT fine-tuning
- **Metrik utama:** F1-Score (metrik resmi kompetisi)

---

## 🏆 Status Bonus

| Bonus | Status | Keterangan |
|---|---|---|
| Submit Kaggle Kasus 1 (+3) | ✅ | Score: 0.76315 |
| Submit Kaggle Kasus 2 (+3) | ✅ | MNIST CNN submission |
| Submit Kaggle Kasus 3 (+3) | 🔄 | In progress |
| Ensemble ML+DL (+5) | ✅ | RF+XGB+MLP Kasus 1 |
| Data Augmentation (+3) | ✅ | RandomRotation+Affine Kasus 2 |

---

## 🔬 Rubrik Penilaian

| Aspek | Bobot |
|---|---|
| Implementasi Metode Konvensional | 15% |
| Implementasi Deep Learning | 25% |
| Justifikasi Pemilihan Arsitektur | 15% |
| Kualitas Analisis Perbandingan | 20% |
| Kualitas Laporan | 15% |
| Reproducibility | 10% |

---

## ⚙️ Reproducibility

Semua eksperimen menggunakan **random seed = 42** yang dikunci secara konsisten:

```python
def set_seed(seed=42):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
```

**Train/Validation split:** 80:20, `stratify=y`, `random_state=42` — konsisten di semua model dalam satu kasus.

---

## 🛠️ Dependencies

```
numpy, pandas, matplotlib, seaborn
scikit-learn, xgboost
pytorch, torchvision
scikit-image (HOG)
pytorch-tabnet
transformers (HuggingFace) — Kasus 3
```

> Dijalankan di **Google Colab / Kaggle Notebook** dengan GPU acceleration.

---

## 📌 Catatan Teknis Penting

- **Perbandingan fair:** Data split yang sama digunakan untuk semua model dalam satu kasus
- **HOG tidak memerlukan normalisasi pixel** — HOG menghitung gradien relatif dan menerapkan L2-normalisasi internal per block, sehingga skala absolut piksel tidak berpengaruh
- **CNN train accuracy** dievaluasi menggunakan loader *tanpa* augmentasi untuk menghindari underestimation
- **SVM training accuracy** diestimasi dengan subset 5,000 sampel karena predict RBF kernel di 33K data sangat mahal secara komputasi
