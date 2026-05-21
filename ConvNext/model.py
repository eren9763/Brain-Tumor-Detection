"""
model.py — ConvNeXtBase + Domain-Robust sınıflandırma başlığı ve LR scheduler.

Dışa aktarılan:
  - build_convnext()     : model ve backbone nesnelerini döndürür
  - CosineWarmup         : Linear warmup + cosine annealing LR callback
"""
import math
import tensorflow as tf
from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.applications import ConvNeXtBase
from tensorflow.keras.callbacks import Callback
import tensorflow.keras.backend as K

from config import IMG_SIZE, NUM_CLS


def build_convnext(num_classes: int = NUM_CLS,
                   freeze_backbone: bool = True,
                   unfreeze_last: int = 0):
    """
    ConvNeXtBase backbone + multi-dropout regularized classification head.

    Mimari katkı
    ─────────────
    - GELU activation  (ConvNeXt native, smoother gradient flow)
    - Dual BatchNorm + Dropout  (domain shift robustness)
    - L2 weight decay everywhere (overfitting önleme)
    - 2-stage training protokolü

    Parameters
    ----------
    num_classes      : çıkış sınıfı sayısı
    freeze_backbone  : True → Stage 1 (sadece head eğitilir)
    unfreeze_last    : Stage 2'de son kaç katman açılacak (0 = tümü açılır)

    Returns
    -------
    model    : derlenmiş Keras modeli
    backbone : ConvNeXtBase nesnesi (Stage 2 için gerekli)
    """
    inputs   = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name='input')
    backbone = ConvNeXtBase(
        include_top=False,
        weights='imagenet',
        input_tensor=inputs,
        pooling=None,
    )
    backbone.trainable = not freeze_backbone
    if not freeze_backbone and unfreeze_last > 0:
        for layer in backbone.layers[:-unfreeze_last]:
            layer.trainable = False

    x = backbone.output
    x = layers.GlobalAveragePooling2D(name='gap')(x)
    x = layers.BatchNormalization(momentum=0.99, epsilon=1e-5, name='bn1')(x)
    x = layers.Dropout(0.45, name='drop1')(x)
    x = layers.Dense(512, activation='gelu',
                     kernel_regularizer=regularizers.l2(1e-4),
                     name='dense1')(x)
    x = layers.BatchNormalization(momentum=0.99, epsilon=1e-5, name='bn2')(x)
    x = layers.Dropout(0.35, name='drop2')(x)
    x = layers.Dense(128, activation='gelu',
                     kernel_regularizer=regularizers.l2(1e-4),
                     name='dense2')(x)
    x = layers.Dropout(0.20, name='drop3')(x)
    out = layers.Dense(num_classes, activation='softmax',
                       kernel_regularizer=regularizers.l2(1e-4),
                       name='output')(x)

    return models.Model(inputs, out, name='ConvNeXtBase_CrossDomain'), backbone


class CosineWarmup(Callback):
    """
    Linear warmup + cosine annealing öğrenme hızı scheduler.
    Referans: Loshchilov & Hutter (2017) — SGDR.
    Keras 2 ve Keras 3 (TF ≥ 2.16) ile uyumludur.
    """

    def __init__(self, total_epochs: int, warmup_epochs: int,
                 base_lr: float, min_lr: float = 1e-7):
        super().__init__()
        self.total = total_epochs
        self.warm  = warmup_epochs
        self.base  = base_lr
        self.min   = min_lr

    def _compute_lr(self, epoch: int) -> float:
        if epoch < self.warm:
            return self.base * (epoch + 1) / self.warm
        t = (epoch - self.warm) / max(self.total - self.warm, 1)
        return self.min + 0.5 * (self.base - self.min) * (1 + math.cos(math.pi * t))

    def on_epoch_begin(self, epoch, logs=None):
        lr  = self._compute_lr(epoch)
        opt = self.model.optimizer
        try:                                # Keras 3 / TF >= 2.16
            opt.learning_rate.assign(lr)
        except AttributeError:
            try:                            # Keras 2 / TF 2.12–2.15
                K.set_value(opt.lr, lr)
            except Exception:
                opt.lr = lr

    def on_epoch_end(self, epoch, logs=None):
        if logs is not None:
            try:
                logs['lr'] = float(self.model.optimizer.learning_rate.numpy())
            except Exception:
                logs['lr'] = self._compute_lr(epoch)
