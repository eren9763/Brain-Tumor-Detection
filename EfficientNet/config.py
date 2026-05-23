"""
config.py — Baseline (EfficientNetB2) Projesi İçin Merkezi Konfigürasyon Dosyası

Bu modül:
1. Bilimsel tutarlılık (reproducibility) için rastgelelik tohumlarını (SEED) sabitler.
2. Sınıf isimleri, görüntü boyutları ve eğitim parametrelerini barındırır.
3. Çevrimdışı (offline) veri artırımının hedef sınırlamalarını tutar.
4. Google Drive indirme ID'lerini ve klasör yollarını merkezi olarak yönetir.
"""

import os
import random
import numpy as np
import tensorflow as tf

# =====================================================================
# 1. BİLİMSEL TUTARLILIK (REPRODUCIBILITY)
# =====================================================================
# Makaledeki deneylerin birebir aynı sonuçları vermesi için SEED = 42 olarak sabitlenmiştir.
SEED = 42
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)


# =====================================================================
# 2. SINIFLAR VE HİPERPARAMETRELER
# =====================================================================
# Baseline makalesine sadık kalındığı için 'notumor' sınıfı yoktur (3 Sınıf).
CLASSES = ['glioma', 'meningioma', 'pituitary']

# EfficientNetB2'nin makalede kullanılan standart giriş boyutu ve diğer hiperparametreler.
IMG_SIZE = (240, 240)
BATCH_SIZE = 32
EPOCHS = 50

# Çevrimdışı veri artırımı (Offline Augmentation) ile ulaşılacak makale hedef sayıları.
TARGET_COUNTS = {
    'glioma': 2795, 
    'meningioma': 1290, 
    'pituitary': 1829
}


# =====================================================================
# 3. GOOGLE DRIVE ID'LERİ (download_data.py için)
# =====================================================================
MODEL_GDRIVE_ID = '1ovAdw_5TC5jqUype2b27hTsDs-YyQgc9'  # .keras dosyası
FIGSHARE_ZIP_ID = '14sxMHDQycIleGVP9Kmw4IWWTNgAacZYR'  # Figshare MRI dataset ZIP dosyası
KAGGLE_ZIP_ID = '13f0-5M-PJBo8dy1c17vNLx--lKTOgNtt' # Kaggle veri seti ZIP dosyası


# =====================================================================
# 4. YEREL KLASÖR VE DOSYA YOLLARI
# =====================================================================
# Tüm verilerin ve modelin indirileceği / işleneceği ana çalışma dizini
BASE_PATH = os.path.join(os.getcwd(), 'baseline_workspace')

# ---- Kaynak (Ham) Veri Seti Yolları ----
FIGSHARE_SOURCE_DIR = os.path.join(BASE_PATH, 'Figshare Brain Tumor')
KAGGLE_SOURCE_TEST = os.path.join(BASE_PATH, 'Kaggle Dataset', 'Testing')

# ---- Ön İşleme Sonrası (Kırpılmış ve Bölünmüş) Veri Yolları ----
DATA_DIR = os.path.join(BASE_PATH, 'data')
TRAIN_DIR = os.path.join(DATA_DIR, 'train')
VAL_DIR = os.path.join(DATA_DIR, 'val')
TEST_DIR = os.path.join(DATA_DIR, 'test')

# ---- Çevrimdışı Veri Artırımı (Augmented) Klasörü ----
AUG_TRAIN_DIR = os.path.join(DATA_DIR, 'aug_train')

# ---- Kaggle Verisinin Baseline Standartlarında Kırpılmış Hali ----
KAGGLE_CROPPED_TEST = os.path.join(BASE_PATH, 'kaggle_cropped_test')

# ---- Model Kayıt Yolları ----
MODEL_SAVE_PATH = os.path.join(BASE_PATH, 'models')
FINAL_MODEL_NAME = 'Hybrid_EfficientNetB2_Baseline_v4_Final.keras' #
