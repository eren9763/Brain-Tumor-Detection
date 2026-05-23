# Brain Tumor MRI — Baseline (EfficientNetB2) Reproduction & Domain Shift Analysis

Bu proje, kaynak kodu gizli olan referans bir beyin tümörü sınıflandırma makalesinin (s41598-023-50505-6) açık kaynaklı, uçtan uca ve birebir yeniden üretilmiş (reproduced) versiyonudur. Proje, makalenin iddia ettiği %97'lik iç doğrulama başarısını Figshare veri seti üzerinde doğrulamakla kalmaz; aynı zamanda modeli Kaggle dış veri setinde test ederek tıbbi yapay zeka modellerindeki **"Alan Kayması" (Domain Shift)** problemini ve veri zafiyetlerini (%79 doğruluk) şeffafça ortaya koyar.

---

## 📋 İçindekiler
1. [Proje Mimarisi (8 Modül)](#proje-mimarisi-8-modül)
2. [Kurulum](#kurulum)
3. [Çalıştırma Modları](#çalıştırma-modları)
4. [Bilimsel Metodoloji](#bilimsel-metodoloji)
5. [Test Sonuçları ve Alan Kayması (Domain Shift)](#test-sonuçları-ve-alan-kayması-domain-shift)

---

## 🏗️ Proje Mimarisi (8 Modül)
Proje, profesyonel yazılım mühendisliği prensiplerine uygun olarak "Sorumlulukların Ayrılığı" (Separation of Concerns) ilkesiyle 8 ana dosyaya bölünmüştür:

| Dosya | Görev |
| :--- | :--- |
| `config.py` | Sabitlerin (Seed, Drive ID, Hedef Resim Sayıları, Klasör Yolları) tutulduğu sinir merkezi. |
| `download_data.py` | Google Drive üzerinden Figshare/Kaggle verilerini ve eğitilmiş modeli otomatik indirir. |
| `preprocessing.py` | Makaleye has katı morfolojik filtreleri (Eşikleme, Aşındırma, Yayma) ve kırpmaları yapar. |
| `data_pipeline.py` | Sınıf dengesizliğini çözmek için (Glioma: 2795, Meningioma: 1290, vb.) diske fiziksel artırım (offline augmentation) yapar. |
| `model.py` | Hybrid EfficientNetB2 tabanını ve 3 sınıflı, %20 Dropout içeren "Custom Head" mimarisini kurar. |
| `train.py` | `ReduceLROnPlateau` eşliğinde 50 epoch'luk tek aşamalı eğitimi gerçekleştirir. |
| `evaluate.py` | İç doğrulama (Figshare) ve Dış doğrulama (Kaggle) testlerini yapıp, Confusion Matrix üretir. |
| `main.py` | Uçtan uca tüm boru hattını (pipeline) argümanlarla yöneten ana (entry) dosyadır. |

---

## ⚙️ Kurulum

**1. Repoyu klonlayın:**
```bash
git clone https://github.com/kullaniciadi/baseline-efficientnetb2-reproduction.git
cd baseline-efficientnetb2-reproduction
```

**2. Gerekli kütüphaneleri kurun:**
```bash
pip install tensorflow opencv-python-headless scikit-learn matplotlib seaborn imutils gdown
```

---

## 🚀 Çalıştırma Modları

Projenin kalbi olan `main.py`, hiçbir manuel işleme gerek kalmadan tam otomatik çalışacak şekilde tasarlanmıştır.

| Komut | Açıklama |
| :--- | :--- |
| `python main.py` | Veriyi/Modeli indirir, morfolojik kırpmaları yapar ve Figshare test setinde değerlendirir. |
| `python main.py --train` | Hazır model kullanmak yerine, modeli sıfırdan 50 epoch boyunca eğitir. |
| `python main.py --skip-prep` | Uzun süren offline augmentation ve kırpma (preprocessing) adımlarını atlar. |
| `python main.py --skip-download` | Google Drive indirmelerini atlar (Dosyalar zaten cihazınızdaysa kullanılır). |
| `python main.py --test-kaggle` | **(Kritik)** Modeli Kaggle veri setinde test ederek *Domain Shift* problemini kanıtlar. |

---

## 🔬 Bilimsel Metodoloji

*   **Morfolojik Ön İşleme (Skull-Stripping):** Tüm görüntüler doğrudan ağa verilmez. Önce Gri Tonlama, Gaussian Blur (5x5), Eşikleme (45, 255), 2 iterasyon Erosion (Aşındırma) ve 2 iterasyon Dilation (Yayma) işlemlerinden geçirilerek beyin dışı dokular (kafatası vb.) silinir.
*   **Çevrimdışı Veri Artırımı (Offline Augmentation):** Makalenin felsefesine uygun olarak, model eğitimi sırasında "class weight" kullanılmaz. Bunun yerine eğitimden önce eksik sınıflar (özellikle Meningioma) sentetik olarak çoğaltılıp diske kaydedilerek sayıları (Glioma: 2795, Meningioma: 1290, Pituitary: 1829) hedeflerine eşitlenir.
*   **Veri Sızıntısı Koruması:** Veriler %70 Eğitim ve %30 Test olarak kesin bir çizgiyle ayrılır. Ayrıca Eğitim verisinin içinden %10'luk kısım Validation (Doğrulama) için ayrılarak modelin test setini görmesi kesinlikle engellenir.

---

## 📊 Test Sonuçları ve Alan Kayması (Domain Shift)

Çalışmamız, sadece yüksek doğruluk oranlarını değil, tıbbi modellerin farklı hastane kalibrasyonlarına karşı ne kadar zayıf olabileceğini şeffafça sunmaktadır:

| Test Veri Seti | Genel Doğruluk (Accuracy) | Durum Özeti |
| :--- | :--- | :--- |
| **Figshare (İç Doğrulama)** | **%97.00** | Referans makalenin başarısı birebir kopyalanmış (reproduced) ve doğrulanmıştır. |
| **Kaggle (Dış Doğrulama)** | **%79.00** | Model yeni bir MR veri setine sokulduğunda iddia edilen %92 başarının aksine %79'a çakılarak **Alan Kayması (Domain Shift)** problemine karşı zayıf olduğunu ispatlamıştır. |

> *Akademik Not: Bu repodaki kodlar referans makalenin birebir üretilmiş (reproduced) halidir.*
