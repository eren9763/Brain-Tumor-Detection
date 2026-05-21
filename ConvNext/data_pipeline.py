"""
data_pipeline.py — Veri bölme, augmentasyon ve TF.data pipeline.

Dışa aktarılan fonksiyonlar / sınıflar:
  - prepare_split_dirs()   : ham veriyi train/val/id_test/ext_test olarak kopyalar
  - mixup()                : Mixup augmentasyonu (Zhang et al. 2018)
  - cutmix()               : CutMix augmentasyonu (Yun et al. 2019)
  - aug_fn()               : Her batch'e rastgele Mixup veya CutMix uygular
  - make_aug_dataset()     : tf.data.Dataset döngüsü (CutMix/Mixup dahil)
  - build_generators()     : Tüm split'ler için ImageDataGenerator döndürür
  - ExternalEvalCallback   : Her epoch sonunda Mendeley ext test'i değerlendirir
"""
import os, shutil, math, random
import numpy as np
import tensorflow as tf
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, balanced_accuracy_score
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import Callback

from config import (
    CLASSES, NUM_CLS, IMG_SIZE, BATCH_SIZE, SEED,
    KAGGLE_TRAIN_DIR, KAGGLE_TEST_DIR, MENDELEY_DIR,
    SRC_TRAIN_DIR, SRC_VAL_DIR, SRC_IDTEST_DIR, EXT_TEST_DIR, ART_DIR
)
from preprocessing import process_save


# ── Veri bölme ───────────────────────────────────────────────────────────────

def prepare_split_dirs() -> dict:
    """
    Ham Kaggle ve Mendeley verilerini işleyip split dizinlerine kopyalar.

    Returns
    -------
    counts : dict — {sınıf: {train, val, id_test, ext_test}} sayıları
    """
    for p in [SRC_TRAIN_DIR, SRC_VAL_DIR, SRC_IDTEST_DIR, EXT_TEST_DIR]:
        if os.path.exists(p):
            shutil.rmtree(p)
        for cls in CLASSES:
            os.makedirs(os.path.join(p, cls), exist_ok=True)

    counts = {}

    # Kaggle Training → %85 Train / %15 Val
    for cls in CLASSES:
        d = os.path.join(KAGGLE_TRAIN_DIR, cls)
        if not os.path.isdir(d):
            continue
        imgs = [f for f in os.listdir(d)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        tr, va = train_test_split(imgs, test_size=0.15, random_state=SEED)
        counts[cls] = {'train': len(tr), 'val': len(va)}
        for split, sdir in [(tr, SRC_TRAIN_DIR), (va, SRC_VAL_DIR)]:
            for f in split:
                process_save(os.path.join(d, f), os.path.join(sdir, cls, f))

    # Kaggle Testing → ID test (aynı domain)
    for cls in CLASSES:
        d = os.path.join(KAGGLE_TEST_DIR, cls)
        if not os.path.isdir(d):
            continue
        imgs = [f for f in os.listdir(d)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        counts[cls]['id_test'] = len(imgs)
        for f in imgs:
            process_save(os.path.join(d, f), os.path.join(SRC_IDTEST_DIR, cls, f))

    # Mendeley → External test (CLAHE kapalı — domain shift korunur)
    for cls in CLASSES:
        d = os.path.join(MENDELEY_DIR, cls)
        if not os.path.isdir(d):
            continue
        imgs = [f for f in os.listdir(d)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        counts[cls]['ext_test'] = len(imgs)
        for f in imgs:
            process_save(os.path.join(d, f),
                         os.path.join(EXT_TEST_DIR, cls, f),
                         apply_clahe=False)

    print('=== Veri Dağılımı ===')
    print(pd.DataFrame(counts).T.fillna(0).astype(int).to_string())
    return counts


# ── Augmentasyon (CutMix / Mixup) ────────────────────────────────────────────

def mixup(x: np.ndarray, y: np.ndarray, alpha: float = 0.2):
    """Zhang et al. (2018) Mixup — NumPy array üzerinde çalışır."""
    lam = np.random.beta(alpha, alpha) if alpha > 0 else 1.0
    idx = np.random.permutation(len(x))
    x_mix = (lam * x + (1 - lam) * x[idx]).astype(np.float32)
    y_mix = (lam * y + (1 - lam) * y[idx]).astype(np.float32)
    return x_mix, y_mix


def cutmix(x: np.ndarray, y: np.ndarray, alpha: float = 1.0):
    """Yun et al. (2019) CutMix — NumPy array üzerinde çalışır."""
    lam = np.random.beta(alpha, alpha)
    B, H, W = x.shape[0], x.shape[1], x.shape[2]
    cut_rat = np.sqrt(1.0 - lam)
    cut_w, cut_h = int(W * cut_rat), int(H * cut_rat)
    cx, cy = np.random.randint(W), np.random.randint(H)
    x1 = np.clip(cx - cut_w // 2, 0, W)
    y1 = np.clip(cy - cut_h // 2, 0, H)
    x2 = np.clip(cx + cut_w // 2, 0, W)
    y2 = np.clip(cy + cut_h // 2, 0, H)
    idx = np.random.permutation(B)
    x_mix = x.copy()
    x_mix[:, y1:y2, x1:x2] = x[idx, y1:y2, x1:x2]
    lam_adj = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
    y_mix = (lam_adj * y + (1 - lam_adj) * y[idx]).astype(np.float32)
    return x_mix.astype(np.float32), y_mix


def aug_fn(x: np.ndarray, y: np.ndarray, p_cutmix: float = 0.5):
    """Her batch'e rastgele Mixup veya CutMix uygular."""
    if np.random.rand() < p_cutmix:
        return cutmix(x, y)
    return mixup(x, y)


# ── TF.data pipeline ─────────────────────────────────────────────────────────

def make_aug_dataset(generator, n_cls: int = NUM_CLS) -> tf.data.Dataset:
    """
    ImageDataGenerator'ı CutMix/Mixup ile sarıp tf.data.Dataset döndürür.
    .prefetch ile GPU'yu bloklamadan besleme sağlar.
    """
    def gen():
        for xb, yb in generator:
            xm, ym = aug_fn(xb, yb)
            yield xm, ym

    sig = (
        tf.TensorSpec(shape=(None, IMG_SIZE, IMG_SIZE, 3), dtype=tf.float32),
        tf.TensorSpec(shape=(None, n_cls),                 dtype=tf.float32),
    )
    return tf.data.Dataset.from_generator(gen, output_signature=sig).prefetch(tf.data.AUTOTUNE)


# ── Generator fabrikası ───────────────────────────────────────────────────────

def build_generators():
    """
    Tüm split'ler için ImageDataGenerator ve sınıf ağırlıklarını döndürür.

    Returns
    -------
    train_gen_raw   : eğitim generator (CutMix/Mixup için ham)
    train_dataset   : augmentasyonlu tf.data.Dataset
    val_gen         : doğrulama generator
    idtest_gen      : ID test generator
    exttest_gen     : Mendeley ext test generator
    class_weights   : sınıf ağırlık dict'i
    classes_ordered : sınıf adları listesi (generator sırasına göre)
    """
    strong_aug = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=25,
        width_shift_range=0.12,
        height_shift_range=0.12,
        zoom_range=0.15,
        shear_range=0.10,
        brightness_range=(0.75, 1.25),
        channel_shift_range=15.0,  # domain shift simülasyonu
        horizontal_flip=True,
        fill_mode='reflect',
    )
    train_gen_raw = strong_aug.flow_from_directory(
        SRC_TRAIN_DIR,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=SEED,
    )
    train_dataset = make_aug_dataset(train_gen_raw)

    eval_gen = lambda d: ImageDataGenerator(rescale=1. / 255).flow_from_directory(
        d, target_size=(IMG_SIZE, IMG_SIZE), batch_size=BATCH_SIZE,
        class_mode='categorical', shuffle=False, seed=SEED,
    )
    val_gen     = eval_gen(SRC_VAL_DIR)
    idtest_gen  = eval_gen(SRC_IDTEST_DIR)
    exttest_gen = eval_gen(EXT_TEST_DIR)

    classes_ordered = list(dict(sorted(train_gen_raw.class_indices.items(),
                                        key=lambda x: x[1])).keys())
    print('Sınıf sırası:', classes_ordered)

    cw = compute_class_weight('balanced',
                               classes=np.unique(train_gen_raw.classes),
                               y=train_gen_raw.classes)
    class_weights = {i: float(w) for i, w in enumerate(cw)}
    print('Class weights:', class_weights)

    return (train_gen_raw, train_dataset,
            val_gen, idtest_gen, exttest_gen,
            class_weights, classes_ordered)


# ── Callback: external eval ───────────────────────────────────────────────────

class ExternalEvalCallback(Callback):
    """Her epoch sonunda Mendeley external test üzerinde metrik hesaplar."""

    def __init__(self, gen, csv_path: str, class_names: list):
        super().__init__()
        self.gen   = gen
        self.csv   = csv_path
        self.names = class_names
        self.rows  = []

    def on_epoch_end(self, epoch, logs=None):
        probs = self.model.predict(self.gen, verbose=0)
        preds = np.argmax(probs, axis=1)
        y_true = self.gen.classes
        f1  = f1_score(y_true, preds, average='macro',    zero_division=0)
        wf1 = f1_score(y_true, preds, average='weighted', zero_division=0)
        bac = balanced_accuracy_score(y_true, preds)
        acc = np.mean(preds == y_true)
        self.rows.append({
            'epoch':           epoch + 1,
            'ext_acc':         acc,
            'ext_macro_f1':    f1,
            'ext_weighted_f1': wf1,
            'ext_bal_acc':     bac,
        })
        pd.DataFrame(self.rows).to_csv(self.csv, index=False)
        print(f'ExtEval e{epoch+1}: acc={acc:.4f}  macro_f1={f1:.4f}  bal_acc={bac:.4f}')
