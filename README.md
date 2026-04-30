# 🧠 UTS Pengantar Deep Learning
### Perbandingan Deep Learning vs Machine Learning Konvensional pada Tiga Kasus

> **Balqis Eka Nurfadisyah — 1202220223**  
> Program Studi S1 Sistem Informasi, Fakultas Rekayasa Industri  
> Universitas Telkom · SI-46-EDM · 2026

---

## 📋 Deskripsi Proyek

Studi empiris yang membandingkan performa **Deep Learning** vs **Machine Learning Konvensional** pada tiga jenis data berbeda menggunakan dataset dari platform Kaggle. Tujuan utama bukan sekadar mengejar skor tertinggi, melainkan memahami **pada kondisi apa deep learning unggul** dan kapan metode klasik masih lebih efektif.

Seluruh eksperimen menggunakan:
- `random_state = 42` (konsisten di semua model)
- Split train-validation **80:20** dengan stratifikasi label
- Pembagian split yang **identik** antara metode konvensional dan deep learning

---

## 📁 Struktur Repository

```
├── UTS_titanic.ipynb            # Kasus 1: Tabular Binary Classification
├── UTS_digit_recognizer.ipynb   # Kasus 2: Image Multi-Class Classification
├── UTS_nlp.ipynb                # Kasus 3: Text Binary Classification
├── titanic_ensemble_submission.csv
├── mnist_cnn_submission.csv
├── nlp_bilstm_submission.csv
└── README.md
```

---

## 🗂️ Ringkasan Tiga Kasus

### Kasus 1 — Titanic: Tabular Binary Classification

**Dataset:** 891 baris train, 8 fitur setelah preprocessing (Age, Sex, Pclass, Fare, Title, FamilySize, IsAlone, Embarked)

**Feature Engineering:** Ekstraksi Title dari Name, FamilySize, IsAlone, binning Age & Fare, imputasi berbasis grup

| Model | Val Accuracy | Val F1 | Train Time |
|---|---|---|---|
| Random Forest (Default) | 83.24% | 0.7727 | ~1 detik |
| Random Forest (GridSearch) | 81.56% | 0.7402 | ~52.9 detik |
| XGBoost (Default) | 81.01% | 0.7536 | ~1 detik |
| XGBoost (GridSearch) | 82.12% | 0.7681 | ~6.7 detik |
| **MLP** | 82.12% | 0.7778 | ~5 detik |
| **TabNet** | 83.80% | 0.7852 | ~32 detik |
| **FT-Transformer** | 82.68% | 0.7891 | ~17 detik |
| **Ensemble (RF+XGB+MLP)** | **83.80%** | **0.7943** | — |

> **Kesimpulan:** Deep Learning **TIDAK** mengungguli metode konvensional. Dataset terlalu kecil (~712 baris training). Random Forest default sudah cukup optimal.

---

### Kasus 2 — Digit Recognizer: Image Multi-Class Classification

**Dataset:** MNIST — 42.000 train, 28.000 test, citra grayscale 28×28 px, 10 kelas (digit 0–9)

**Preprocessing DL:** Reshape ke (28,28), augmentasi RandomRotation(10°) + RandomAffine(translate 0.1), normalisasi mean=0.5 std=0.5

| Model | Val Accuracy | Macro F1 | Train Time | Inf. Time |
|---|---|---|---|---|
| PCA + Random Forest | 94.71% | 0.9464 | 109.77 s | 0.141 s |
| HOG + SVM | 96.92% | 0.9691 | 285.62 s | 25.295 s |
| **CNN (Deep Learning)** | **98.88%** | **0.9887** | 176.81 s | 2.528 s |

**Arsitektur CNN:**
```
Input(1×28×28) → Conv2d(1→16) → ReLU → MaxPool →
Conv2d(16→32) → ReLU → MaxPool → Flatten →
Linear(1568→128) → ReLU → Dropout(0.25) → Linear(128→10)
```
Total parameter: **206.922**

> **Kesimpulan:** CNN **unggul signifikan** (~2–4 poin akurasi). Data dengan struktur spasial + sampel cukup → Deep Learning jelas lebih baik.

---

### Kasus 3 — NLP Disaster Tweets: Text Binary Classification

**Dataset:** 7.613 baris train, klasifikasi biner: bencana nyata (1) vs bukan (0)

**Augmentasi:** Synonym replacement (WordNet) pada kelas minoritas, split dilakukan *sebelum* augmentasi untuk mencegah data leakage

| Model | Val F1 | Val Accuracy | Train Time | Inf. Time |
|---|---|---|---|---|
| Logistic Regression (TF-IDF) | 0.7604 | 79.51% | 0.27 s | 0.0005 s |
| **Linear SVM (TF-IDF)** | **0.7661** | **80.43%** | 13.76 s | 0.5433 s |
| BiLSTM (Embedding) | 0.7245 | 77.08% | 9.31 s | 0.0358 s |

**Arsitektur BiLSTM:**
```
Input(B, 50) → Embedding(10002, 50) → BiLSTM(hidden=64, bidirectional) →
Concat hidden state (dim=128) → Dropout(0.5) → Linear(128→1)
```
Total parameter: **559.621**

> **Kesimpulan:** BiLSTM **KALAH** dari baseline TF-IDF. Dataset teks kecil dengan kata kunci diskriminatif → metode konvensional tetap lebih kompetitif.

---

## 📊 Ringkasan Lintas Kasus

| Kasus | Tipe Data | Ukuran Train | Konvensional Terbaik | DL Terbaik | Pemenang |
|---|---|---|---|---|---|
| Titanic | Tabular (8 fitur) | ~712 | RF 83.24% | TabNet 83.80% | **Setara** |
| Digit Recognizer | Image 28×28 | 33.600 | HOG+SVM 96.92% | CNN 98.88% | **Deep Learning** |
| NLP Disaster Tweets | Text (≤31 kata) | ~7.400 | LinearSVM F1=0.7661 | BiLSTM F1=0.7245 | **Konvensional** |

---

## 💡 Kesimpulan Utama

1. **Deep learning bukan jawaban universal.** Pada dataset kecil dan tabular, model berbasis tree masih setara atau lebih baik.
2. **Ukuran dataset adalah faktor penentu.** DL membutuhkan ribuan–puluhan ribu sampel agar regularisasi internal bekerja efektif.
3. **Struktur data menentukan arsitektur:** citra 2D → CNN, sekuensial → LSTM/Transformer, tabular heterogen → tree-based.
4. **Biaya komputasi nyata.** TabNet membutuhkan ~30× waktu lebih lama dari XGBoost default untuk akurasi yang sebanding.
5. **Ensemble lintas paradigma** (RF + XGB + MLP) bisa melampaui batas individual masing-masing model.

---

## 🔧 Cara Menjalankan

```bash
# Clone repository
git clone https://github.com/ghfkdkey/UTSDeepLearningBalqis.git
cd UTSDeepLearningBalqis

# Install dependencies
pip install torch scikit-learn xgboost pytorch-tabnet nltk scikit-image pandas matplotlib seaborn

# Jalankan notebook sesuai kasus
jupyter notebook UTS_titanic.ipynb
jupyter notebook UTS_digit_recognizer.ipynb
jupyter notebook UTS_nlp.ipynb
```

**Requirements:** Python 3, PyTorch, scikit-learn, XGBoost, NLTK, scikit-image, pytorch-tabnet  
**GPU:** Direkomendasikan untuk training CNN dan BiLSTM (CUDA)

---

## 📚 Referensi Utama

- Shwartz-Ziv & Armon (2022) — *Tabular Data: Deep Learning is Not All You Need*
- Arik & Pfister (2021) — *TabNet: Attentive Interpretable Tabular Learning*
- Gorishniy et al. (2021) — *Revisiting Deep Learning Models for Tabular Data*
- Dalal & Triggs (2005) — *Histograms of Oriented Gradients for Human Detection*
- Wei & Zou (2019) — *EDA: Easy Data Augmentation Techniques*

---

<div align="center">
  <sub>Laporan UTS Pengantar Deep Learning · Universitas Telkom · 2026</sub>
</div>
