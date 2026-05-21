"""
evaluate.py — Model değerlendirme ve TTA tahmin fonksiyonları.

Dışa aktarılan:
  - expected_calibration_error()
  - compute_metrics()
  - tta_predict()
  - run_evaluation()   : tam özet tabloyu hesaplar ve CSV'e kaydeder
"""
import os
import numpy as np
import pandas as pd
import cv2
import tensorflow as tf
from sklearn.metrics import (classification_report, f1_score,
                              balanced_accuracy_score, roc_auc_score)
from sklearn.calibration import calibration_curve
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from config import ART_DIR, CLASSES
from config import SRC_IDTEST_DIR, EXT_TEST_DIR, IMG_SIZE

# ── ECE ──────────────────────────────────────────────────────────────────────

def expected_calibration_error(y_true: np.ndarray,
                                y_prob: np.ndarray,
                                n_bins: int = 10) -> float:
    """
    Beklenen Kalibrasyon Hatası (ECE).
    Çok sınıflı durumda maksimum güven üzerinden hesaplanır.
    """
    confidences = np.max(y_prob, axis=1) if y_prob.shape[1] > 2 else y_prob[:, 1]
    y_pred_labels = np.argmax(y_prob, axis=1)
    correctness   = (y_true == y_pred_labels).astype(int)

    bin_indices = np.digitize(confidences, np.linspace(0.0, 1.0, n_bins + 1)[1:-1])
    bin_counts  = np.bincount(bin_indices, minlength=n_bins)

    ece_val = 0.0
    for i in range(n_bins):
        if bin_counts[i] > 0:
            bin_conf = np.mean(confidences[bin_indices == i])
            bin_acc  = np.mean(correctness[bin_indices == i])
            ece_val += (bin_counts[i] / len(y_true)) * abs(bin_conf - bin_acc)
    return ece_val


# ── Metrik hesaplama ─────────────────────────────────────────────────────────

def compute_metrics(y_true: np.ndarray,
                    y_pred: np.ndarray,
                    y_prob: np.ndarray,
                    class_names: list) -> dict:
    """Kapsamlı sınıflandırma metrikleri döndürür."""
    report = classification_report(
        y_true, y_pred, target_names=class_names,
        output_dict=True, zero_division=0,
    )
    y_true_oh = tf.keras.utils.to_categorical(y_true, num_classes=len(class_names))
    try:
        macro_auc = roc_auc_score(y_true_oh, y_prob, average='macro', multi_class='ovr')
    except ValueError:
        macro_auc = 0.0

    return {
        'accuracy':     report['accuracy'],
        'macro avg':    report['macro avg'],
        'weighted avg': report['weighted avg'],
        'bal_acc':      balanced_accuracy_score(y_true, y_pred),
        'macro_auc':    macro_auc,
        'ece':          expected_calibration_error(y_true, y_prob),
    }


# ── TTA tahmin ───────────────────────────────────────────────────────────────

def tta_predict(model, generator, n_tta: int = 4):
    """
    Test Time Augmentation (TTA) ile robust tahmin.
    Flip + rotation + brightness varyantları ortalanır.

    Returns
    -------
    probs : (N, num_classes) float32 ortalama olasılıklar
    preds : (N,) argmax tahminler
    """
    tta_aug = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=12,
        width_shift_range=0.06,
        height_shift_range=0.06,
        zoom_range=0.06,
        horizontal_flip=True,
        brightness_range=(0.9, 1.1),
        fill_mode='nearest',
    )
    all_probs = []
    for path in generator.filepaths:
        img   = cv2.cvtColor(cv2.imread(path), cv2.COLOR_BGR2RGB).astype(np.float32)
        img_n = img / 255.0
        variants = [img_n, np.fliplr(img_n)]
        for _ in range(n_tta - 2):
            for b in tta_aug.flow((img[np.newaxis]).astype(np.uint8), batch_size=1):
                variants.append(b[0])
                break
        stack = np.stack(variants[:n_tta])
        all_probs.append(model.predict(stack, verbose=0).mean(axis=0))
    probs = np.array(all_probs)
    return probs, np.argmax(probs, axis=1)


# ── Tam değerlendirme akışı ───────────────────────────────────────────────────

def run_evaluation(model, idtest_gen, exttest_gen,
                   classes_ordered: list,
                   art_dir: str = ART_DIR, run_tta=False) -> pd.DataFrame:
    """
    Plain ve TTA tahminlerini çalıştırır, özet tabloyu CSV'e kaydeder.

    Returns
    -------
    summary : pandas DataFrame
    """
    def _load_images(base_dir, classes):
            X, y = [], []
            for idx, cls in enumerate(classes):
                cls_dir = os.path.join(base_dir, cls)
                if not os.path.isdir(cls_dir):
                    continue
                for fname in sorted(os.listdir(cls_dir)):
                    if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        continue
                    img = cv2.imread(os.path.join(cls_dir, fname))
                    if img is None:
                        continue
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                    X.append(img.astype(np.float32) / 255.0)
                    y.append(idx)
            return np.array(X), np.array(y)

    print('Görseller yükleniyor…')
    x_id,  y_id  = _load_images(SRC_IDTEST_DIR, classes_ordered)
    x_ext, y_ext = _load_images(EXT_TEST_DIR,   classes_ordered)
    print(f'ID: {x_id.shape}, EXT: {x_ext.shape}')

    id_plain_probs  = model.predict(x_id,  batch_size=64, verbose=1)
    id_plain_preds  = np.argmax(id_plain_probs, axis=1)
    r_id_plain      = compute_metrics(y_id, id_plain_preds, id_plain_probs, classes_ordered)

    ext_plain_probs = model.predict(x_ext, batch_size=64, verbose=1)
    ext_plain_preds = np.argmax(ext_plain_probs, axis=1)
    r_ext_plain     = compute_metrics(y_ext, ext_plain_preds, ext_plain_probs, classes_ordered)

    print('TTA değerlendirme…')
    id_tta_probs,  id_tta_preds  = tta_predict(model, idtest_gen)
    r_id_tta  = compute_metrics(idtest_gen.classes,  id_tta_preds,  id_tta_probs,  classes_ordered)

    ext_tta_probs, ext_tta_preds = tta_predict(model, exttest_gen)
    r_ext_tta = compute_metrics(exttest_gen.classes, ext_tta_preds, ext_tta_probs, classes_ordered)

    rows = [
    {'Setting':'Kaggle ID (plain)',
     'Accuracy': r_id_plain['accuracy'],
     'Macro F1': r_id_plain['macro avg']['f1-score'],
     'Weighted F1': r_id_plain['weighted avg']['f1-score'],
     'Balanced Acc': r_id_plain['bal_acc'],
     'Macro AUC': r_id_plain['macro_auc'],
     'ECE': r_id_plain['ece']},

    {'Setting':'Mendeley Ext (plain)',
     'Accuracy': r_ext_plain['accuracy'],
     'Macro F1': r_ext_plain['macro avg']['f1-score'],
     'Weighted F1': r_ext_plain['weighted avg']['f1-score'],
     'Balanced Acc': r_ext_plain['bal_acc'],
     'Macro AUC': r_ext_plain['macro_auc'],
     'ECE': r_ext_plain['ece']},
]

    if run_tta:
        print('TTA değerlendirme (bu adım uzun sürebilir)…')
        id_tta_probs,  id_tta_preds  = tta_predict(model, idtest_gen, n_tta=4)
        ext_tta_probs, ext_tta_preds = tta_predict(model, exttest_gen, n_tta=4)

        r_id_tta  = compute_metrics(idtest_gen.classes,  id_tta_preds,
                                 id_tta_probs,  classes_ordered)
        r_ext_tta = compute_metrics(exttest_gen.classes, ext_tta_preds,
                                 ext_tta_probs, classes_ordered)

        rows += [
            {'Setting':'Kaggle ID (TTA x4)',
            'Accuracy': r_id_tta['accuracy'],
            'Macro F1': r_id_tta['macro avg']['f1-score'],
            'Weighted F1': r_id_tta['weighted avg']['f1-score'],
            'Balanced Acc': r_id_tta['bal_acc'],
            'Macro AUC': r_id_tta['macro_auc'],
            'ECE': r_id_tta['ece']},

            {'Setting':'Mendeley Ext (TTA x4)',
            'Accuracy': r_ext_tta['accuracy'],
            'Macro F1': r_ext_tta['macro avg']['f1-score'],
            'Weighted F1': r_ext_tta['weighted avg']['f1-score'],
            'Balanced Acc': r_ext_tta['bal_acc'],
            'Macro AUC': r_ext_tta['macro_auc'],
            'ECE': r_ext_tta['ece']},
        ]

    summary = pd.DataFrame(rows)
    csv_path = os.path.join(art_dir, 'final_summary.csv')
    summary.to_csv(csv_path, index=False)

    print('\n' + '=' * 80)
    print('FINAL SUMMARY TABLE')
    print('=' * 80)
    print(summary.to_string(index=False))

    id_f1  = r_id_plain['macro avg']['f1-score']
    ext_f1 = r_ext_plain['macro avg']['f1-score']
    print(f'\nDomain Gap (Macro F1): {id_f1:.4f} → {ext_f1:.4f} (Δ = {id_f1 - ext_f1:.4f})')
    return summary
