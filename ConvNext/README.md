# Brain Tumor MRI — Cross-Dataset Generalization Pipeline

ConvNeXtBase tabanlı, Kaggle → Mendeley cross-domain MRI tümör sınıflandırma pipeline'ı.  
4 sınıf: `glioma`, `meningioma`, `notumor`, `pituitary`

> **Hiçbir şey indirmenize gerek yok.**  
> `python main.py` çalıştırıldığında model ve veri setleri otomatik indirilip hazırlanır.

---

## İçindekiler

1. [Proje Yapısı](#proje-yapısı)
2. [Kurulum](#kurulum)
3. [Çalıştırma](#çalıştırma)
4. [Çalıştırma Modları](#çalıştırma-modları)
5. [Eğitim (Opsiyonel)](#eğitim-opsiyonel)
6. [Çıktılar](#çıktılar)
7. [Mimari Detayları](#mimari-detayları)
8. [Sık Karşılaşılan Sorunlar](#sık-karşılaşılan-sorunlar)

---

## Proje Yapısı

```
mri_project/
├── config.py            # Drive ID'leri, yollar, hiperparametreler
├── download_data.py     # Otomatik indirme + ZIP çıkartma
├── preprocessing.py     # Görüntü ön işleme (robust crop, CLAHE, pad-resize)
├── data_pipeline.py     # Veri bölme, Mixup/CutMix augmentasyon, tf.data
├── model.py             # ConvNeXtBase mimarisi + CosineWarmup LR callback
├── train.py             # 2-stage eğitim akışı
├── evaluate.py          # Metrikler (ECE, AUC, TTA), özet tablo
├── visualize.py         # Grad-CAM, t-SNE domain shift, öğrenme eğrileri
├── main.py              # Ana giriş noktası (CLI)
├── requirements.txt
└── README.md
```

---

## Kurulum

### Gereksinimler

- Python **3.10+**
- CUDA uyumlu GPU önerilir (CPU ile de çalışır, ancak yavaştır)

### Adımlar

**1. Repoyu klonlayın:**

```bash
git clone https://github.com/<kullanici>/<repo>.git
cd mri_project
```

**2. Sanal ortam oluşturun (önerilen):**

```bash
python -m venv venv
source venv/bin/activate        # Linux / macOS
# venv\Scripts\activate         # Windows
```

**3. Bağımlılıkları kurun:**

```bash
pip install -r requirements.txt
```

Kurulum tamamlandı. Başka bir şey yapmanıza gerek yok.

---

## Çalıştırma

```bash
python main.py
```

İlk çalıştırmada otomatik olarak şunlar gerçekleşir:

```
=== Varlık Kontrolü & İndirme ===
[✓] Model zaten mevcut, atlanıyor.          ← varsa atlar
[↓] MRI Dataset (ZIP) indiriliyor…          ← yoksa indirir
[↓] MRI Dataset çıkartılıyor → data/        ← ZIP açılır, ZIP silinir
[↓] Mendeley Dataset (ZIP) indiriliyor…
[↓] Mendeley Dataset çıkartılıyor → data/
=== Tüm dosyalar hazır ===

Model yükleniyor...
Veri split ediliyor...  (train %85 / val %15 / id_test / ext_test)
Plain değerlendirme…
TTA değerlendirme…
Grad-CAM hesaplanıyor…
t-SNE hesaplanıyor…

✓ Tüm çıktılar kaydedildi: mri_cross_v3/artifacts/
```

İkinci çalıştırmada dosyalar zaten mevcut olduğundan tüm indirme adımları atlanır.

---

## Çalıştırma Modları

| Komut | Ne Yapar |
|-------|----------|
| `python main.py` | İndir (yoksa) → değerlendirme + görseller |
| `python main.py --train` | İndir → sıfırdan eğit → değerlendirme + görseller |
| `python main.py --skip-download` | İndirmeyi atla, mevcut dosyaları kullan |
| `python main.py --model-path /yol/model.keras` | Belirtilen modeli kullan |

---

## Eğitim (Opsiyonel)

Pretrained model yerine sıfırdan eğitmek isterseniz:

```bash
python main.py --train
```

**Stage 1 — Frozen Backbone (8 epoch)**

ConvNeXtBase dondurulur, sadece sınıflandırma başlığı eğitilir.  
Optimizer: `AdamW(lr=3e-4, weight_decay=1e-4)`, Label Smoothing: 0.10

**Stage 2 — Selective Fine-tune (25 epoch)**

Backbone'un son 50 katmanı açılır.  
Linear warmup (3 epoch) + cosine annealing LR uygulanır.  
Optimizer: `AdamW(lr=5e-5)`, EarlyStopping patience: 7

Her epoch sonunda Mendeley external test seti otomatik değerlendirilir.

---

## Çıktılar

Tüm çıktılar `mri_cross_v3/artifacts/` klasörüne kaydedilir:

| Dosya | İçerik |
|-------|--------|
| `final_summary.csv` | Plain ve TTA metrik özet tablosu |
| `ext_eval_stage2.csv` | Epoch başına Mendeley generalizasyon metrikleri |
| `train_log.csv` | Eğitim epoch logları (loss, accuracy, lr) |
| `gradcam_glioma.png` | Glioma örnek Grad-CAM |
| `gradcam_meningioma.png` | Meningioma örnek Grad-CAM |
| `gradcam_notumor.png` | No Tumor örnek Grad-CAM |
| `gradcam_pituitary.png` | Pituitary örnek Grad-CAM |
| `gradcam_grid_id_test.png` | Kaggle ID test Grad-CAM grid |
| `gradcam_grid_ext_test.png` | Mendeley external test Grad-CAM grid |
| `tsne_final.png` | Kaggle ID vs Mendeley Ext feature uzayı (t-SNE) |
| `learning_curves.png` | Accuracy / Loss / LR eğrileri |
| `ext_generalization_curve.png` | Epoch başına harici generalizasyon eğrisi |

### Değerlendirme Metrikleri

| Metrik | Açıklama |
|--------|----------|
| Accuracy | Doğruluk oranı |
| Macro F1 | Sınıf dengesizliğinden bağımsız F1 ortalaması |
| Weighted F1 | Örnek sayısına göre ağırlıklı F1 |
| Balanced Accuracy | Her sınıfın recall ortalaması |
| Macro AUC | OvR çok sınıflı ROC-AUC |
| ECE | Expected Calibration Error — kalibrasyon kalitesi |

---

## Mimari Detayları

### Model Yapısı

```
Input (224 × 224 × 3)
    └── ConvNeXtBase  (ImageNet pretrained, ~89M parametre)
        └── GlobalAveragePooling2D
            └── BatchNorm  →  Dropout(0.45)
                └── Dense(512, GELU)  +  L2(1e-4)
                    └── BatchNorm  →  Dropout(0.35)
                        └── Dense(128, GELU)  +  L2(1e-4)
                            └── Dropout(0.20)
                                └── Dense(4, Softmax)  +  L2(1e-4)
```

### Augmentasyon Stratejisi

| Teknik | Parametre | Amaç |
|--------|-----------|------|
| Mixup | α = 0.2 | Sınır bölgelerini yumuşatma |
| CutMix | α = 1.0 | Lokal özellik çeşitliliği |
| CLAHE | clipLimit=2.0 | Kaynak domain kontrast artırma |
| Channel Shift | ±15 | Domain shift simülasyonu |
| Rotation | ±25° | Geometrik çeşitlilik |
| Zoom / Shift | %15 / %12 | Ölçek ve konum değişimi |
| Horizontal Flip | — | Simetri genellemesi |

---

## Sık Karşılaşılan Sorunlar

**`gdown` indirme hatası / permission denied:**

```bash
pip install --upgrade gdown
```

Drive dosyalarının paylaşım izni **"Bağlantıya sahip olan herkes → Görüntüleyici"**
olarak ayarlı olmalıdır.

---

**ZIP çıkartıldıktan sonra veri bulunamıyor:**

ZIP içinde bazen iç içe klasör oluşur: `data/MRI dataset/MRI dataset/Training`  
`download_data.py` bunu otomatik düzeltir. Eğer hâlâ hata alıyorsanız terminalde kontrol edin:

```bash
ls data/
ls "data/MRI dataset/"
```

---

**TensorFlow GPU görünmüyor:**

```bash
pip install tensorflow[and-cuda]
```

GPU yoksa pipeline CPU ile çalışır, daha yavaştır.

---

**`imutils` bulunamıyor:**

```bash
pip install imutils
```

---

## Lisans

Bu proje akademik araştırma amaçlıdır.
