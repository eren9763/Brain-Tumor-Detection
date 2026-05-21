"""
config.py — MRI Cross-Dataset Generalization Pipeline v3.0
"""
import os, random, warnings
import numpy as np
import tensorflow as tf

SEED = 42
os.environ['PYTHONHASHSEED']       = str(SEED)
os.environ['TF_DETERMINISTIC_OPS'] = '1'
random.seed(SEED)
np.random.seed(SEED)
tf.random.set_seed(SEED)
warnings.filterwarnings('ignore')

# ── Sınıflar & görüntü boyutu ────────────────────────────────────────────────
CLASSES    = ['glioma', 'meningioma', 'notumor', 'pituitary']
NUM_CLS    = 4
IMG_SIZE   = 224
BATCH_SIZE = 32

# ── Eğitim parametreleri ─────────────────────────────────────────────────────
EPOCHS_S1  = 8
EPOCHS_S2  = 25

# ── Google Drive File ID'leri ─────────────────────────────────────────────────
MODEL_GDRIVE_ID = '1Xiy2Nid8IprLJNxWbnGWHYLoIVciPNNi'  # .keras dosyası
MRI_ZIP_GDRIVE_ID  = '13f0-5M-PJBo8dy1c17vNLx--lKTOgNtt'  # MRI dataset ZIP
MENDELEY_ZIP_GDRIVE_ID = '1igmIhrv9je0QyJlW7ZxofDBMvjU782y6'  # Mendeley ZIP

# ── Yerel dizinler ───────────────────────────────────────────────────────────
BASE_PATH        = os.path.join(os.getcwd(), 'data')
KAGGLE_TRAIN_DIR = os.path.join(BASE_PATH, 'MRI dataset', 'Training')
KAGGLE_TEST_DIR  = os.path.join(BASE_PATH, 'MRI dataset', 'Testing')
MENDELEY_DIR     = os.path.join(BASE_PATH, 'Mendeley')

# ── Çalışma dizinleri ────────────────────────────────────────────────────────
WORK_DIR       = os.path.join(os.getcwd(), 'mri_cross_v3')
SRC_TRAIN_DIR  = os.path.join(WORK_DIR, 'src_train')
SRC_VAL_DIR    = os.path.join(WORK_DIR, 'src_val')
SRC_IDTEST_DIR = os.path.join(WORK_DIR, 'src_idtest')
EXT_TEST_DIR   = os.path.join(WORK_DIR, 'ext_test')
ART_DIR        = os.path.join(WORK_DIR, 'artifacts')
os.makedirs(ART_DIR, exist_ok=True)

# ── Model kayıt yolları ──────────────────────────────────────────────────────
CKPT1          = os.path.join(ART_DIR, 'best_stage1.keras')
CKPT2          = os.path.join(ART_DIR, 'best_final.keras')
TRAIN_LOG      = os.path.join(ART_DIR, 'train_log.csv')
MODEL_FILENAME  = 'convnext_weights.weights.h5'
SAVE_PATH      = os.path.join('models', MODEL_FILENAME)
