"""
train.py — 2-Stage ConvNeXtBase eğitim akışı.

Kullanım:
    python train.py

Ortam değişkenleri (opsiyonel):
    MRI_BASE_PATH   : Ham veri kök dizini (varsayılan /data/brain_tumor)
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import optimizers, losses
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger,
)

from config import (
    SEED, BATCH_SIZE, EPOCHS_S1, EPOCHS_S2,
    CKPT1, CKPT2, TRAIN_LOG, ART_DIR, SAVE_PATH,
)
from data_pipeline import prepare_split_dirs, build_generators, ExternalEvalCallback
from model import build_convnext, CosineWarmup


def main():
    # ── Veri hazırlama ────────────────────────────────────────────────────────
    prepare_split_dirs()

    (train_gen_raw, train_dataset,
     val_gen, idtest_gen, exttest_gen,
     class_weights, classes_ordered) = build_generators()

    steps_per_epoch = math.ceil(train_gen_raw.n / BATCH_SIZE)

    # ── Stage 1 — Sadece head eğitimi ─────────────────────────────────────────
    print("\n[Stage 1] Frozen backbone — classification head warmup")
    model, backbone = build_convnext(freeze_backbone=True)
    model.compile(
        optimizer=optimizers.AdamW(learning_rate=3e-4, weight_decay=1e-4),
        loss=losses.CategoricalCrossentropy(label_smoothing=0.10),
        metrics=['accuracy'],
    )
    model.summary(line_length=80)

    csv_ext_s1 = os.path.join(ART_DIR, 'ext_eval_stage1.csv')
    cbs_s1 = [
        ModelCheckpoint(CKPT1, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=5, restore_best_weights=True),
        CSVLogger(TRAIN_LOG),
        ExternalEvalCallback(exttest_gen, csv_ext_s1, classes_ordered),
    ]
    model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        epochs=EPOCHS_S1,
        class_weight=class_weights,
        callbacks=cbs_s1,
        verbose=1,
    )
    model.load_weights(CKPT1)

    # ── Stage 2 — Selective fine-tune ─────────────────────────────────────────
    print("\n[Stage 2] Selective fine-tune — cosine warmup LR")
    backbone.trainable = True
    # Son 50 katmanı aç, geri kalanları dondur
    for layer in backbone.layers[:-50]:
        layer.trainable = False

    model.compile(
        optimizer=optimizers.AdamW(learning_rate=5e-5, weight_decay=1e-4),
        loss=losses.CategoricalCrossentropy(label_smoothing=0.10),
        metrics=['accuracy'],
    )

    csv_ext_s2 = os.path.join(ART_DIR, 'ext_eval_stage2.csv')
    cbs_s2 = [
        ModelCheckpoint(CKPT2, monitor='val_accuracy', save_best_only=True, verbose=1),
        EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True),
        CSVLogger(TRAIN_LOG, append=True),
        CosineWarmup(total_epochs=EPOCHS_S2, warmup_epochs=3,
                     base_lr=5e-5, min_lr=1e-7),
        ExternalEvalCallback(exttest_gen, csv_ext_s2, classes_ordered),
    ]
    model.fit(
        train_dataset,
        steps_per_epoch=steps_per_epoch,
        validation_data=val_gen,
        epochs=EPOCHS_S2,
        class_weight=class_weights,
        callbacks=cbs_s2,
        verbose=1,
    )
    model.load_weights(CKPT2)

    # ── Modeli kaydet ─────────────────────────────────────────────────────────
    os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
    model.save(SAVE_PATH)
    print(f"\nModel kaydedildi → {SAVE_PATH}")


if __name__ == '__main__':
    import math
    main()
