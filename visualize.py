"""
visualize.py — Grad-CAM görselleştirme ve t-SNE domain shift analizi.

Dışa aktarılan:
  - get_last_4d_layer_name()
  - compute_gradcam()
  - overlay_heatmap()
  - gradcam_on_path()
  - visualize_gradcam()    : Karanlık temalı tek görsel analiz
  - plot_gradcam_grid()    : Tüm sınıflar için grid
  - tsne_domain_shift()    : Kaggle vs Mendeley feature uzayı
  - plot_learning_curves() : Eğitim log'undan accuracy/loss/lr grafikleri
  - plot_ext_curve()       : Epoch başına harici generalizasyon eğrisi
"""
import os
import random
import math
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import tensorflow as tf
from sklearn.manifold import TSNE
from pathlib import Path

from config import SEED, IMG_SIZE, ART_DIR
from preprocessing import robust_crop, pad_resize


# ── Grad-CAM yardımcıları ─────────────────────────────────────────────────────

def get_last_4d_layer_name(model_cls) -> str | None:
    """Modelin son 4 boyutlu (batch, H, W, C) katman adını döndürür."""
    for layer in reversed(model_cls.layers):
        try:
            shape = layer.output.shape
            if len(shape) == 4:
                return layer.name
        except Exception:
            pass
    return None


def compute_gradcam(model_cls, img_tensor: np.ndarray, class_idx=None):
    """
    Grad-CAM ısı haritası hesaplar.

    Parameters
    ----------
    model_cls  : Keras modeli
    img_tensor : (1, H, W, 3) float32 [0, 1] tensör
    class_idx  : None → en yüksek olasılıklı sınıf

    Returns
    -------
    heatmap   : (H, W) float32 normalize ısı haritası
    class_idx : tahmin edilen sınıf indeksi
    probs     : (num_classes,) softmax çıktısı
    """
    last_conv_name = get_last_4d_layer_name(model_cls)
    if last_conv_name is None:
        raise ValueError("4D konvolüsyon katmanı bulunamadı.")

    last_conv_layer = model_cls.get_layer(last_conv_name)
    grad_model = tf.keras.Model(
        inputs=model_cls.inputs,
        outputs=[last_conv_layer.output, model_cls.output],
    )
    with tf.GradientTape() as tape:
        conv_out, preds = grad_model(img_tensor)
        if class_idx is None:
            class_idx = tf.argmax(preds[0])
        loss = preds[:, class_idx]

    grads  = tape.gradient(loss, conv_out)
    pooled = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_out = conv_out[0]
    heatmap  = conv_out @ pooled[..., tf.newaxis]
    heatmap  = tf.squeeze(heatmap)
    heatmap  = tf.maximum(heatmap, 0) / (tf.reduce_max(heatmap) + 1e-8)
    return heatmap.numpy(), int(class_idx), preds.numpy()[0]


def overlay_heatmap(img_rgb: np.ndarray, heatmap: np.ndarray,
                     alpha: float = 0.4) -> np.ndarray:
    """Grad-CAM ısı haritasını orijinal görüntüyle birleştirir."""
    heatmap = cv2.resize(heatmap, (img_rgb.shape[1], img_rgb.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    colored = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    colored = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
    overlay = (1 - alpha) * img_rgb + alpha * colored
    return np.uint8(np.clip(overlay, 0, 255))


def gradcam_on_path(model_cls, image_path: str,
                     true_label: str = None,
                     save_path: str = "gradcam.png",
                     classes_ordered: list = None):
    """Tek bir görüntü için Grad-CAM hesaplayıp kaydeder."""
    img     = cv2.imread(image_path)
    img     = robust_crop(img)
    img     = pad_resize(img, IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    x = img_rgb.astype(np.float32) / 255.0
    x = x[np.newaxis, ...]

    heatmap, pred_idx, probs = compute_gradcam(model_cls, x)
    pred_label = classes_ordered[pred_idx] if classes_ordered else str(pred_idx)
    overlay    = overlay_heatmap(img_rgb, heatmap)

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].imshow(img_rgb);     axes[0].set_title("Original")
    axes[1].imshow(heatmap, cmap="jet"); axes[1].set_title("Grad-CAM")
    axes[2].imshow(overlay);    axes[2].set_title(f"Pred: {pred_label} ({probs[pred_idx]*100:.1f}%)")

    if true_label is not None:
        fig.suptitle(f"True: {true_label}", fontsize=14)
    for ax in axes:
        ax.axis("off")
    plt.tight_layout()
    plt.savefig(save_path, dpi=180, bbox_inches="tight")
    plt.show()
    print(f"Grad-CAM kaydedildi: {save_path}")


# ── Karanlık temalı tek görüntü analizi ──────────────────────────────────────

def visualize_gradcam(image_path: str, true_label: str,
                       model, classes: list,
                       save_path: str = "gradcam_result.png"):
    """Karanlık arka planlı, güven barı içeren Grad-CAM görseli."""
    img_display, img_norm = _preprocess(image_path)
    img_tensor = img_norm[np.newaxis, ...]

    heatmap, pred_idx, probs = compute_gradcam(model, img_tensor)
    pred_label  = classes[pred_idx]
    confidence  = probs[pred_idx]
    overlay     = overlay_heatmap(img_display, heatmap, alpha=0.45)

    is_correct   = pred_label == true_label
    border_color = '#2ecc71' if is_correct else '#e74c3c'
    verdict      = '✓ DOĞRU' if is_correct else '✗ YANLIŞ'

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    fig.patch.set_facecolor('#1a1a2e')
    for ax, img, title in zip(axes,
                               [img_display, plt.cm.jet(heatmap)[:, :, :3], overlay],
                               ['Orijinal MRI', 'Grad-CAM Isı Haritası', 'Overlay']):
        ax.imshow(img)
        ax.set_title(title, color='white', fontsize=13, pad=10)
        ax.axis('off')
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_edgecolor('#444')
            spine.set_linewidth(1.5)

    fig.suptitle(
        f"Gerçek: {true_label.upper()}  |  Tahmin: {pred_label.upper()}  ({confidence*100:.1f}%)  {verdict}",
        fontsize=15, fontweight='bold', color=border_color, y=1.02,
    )

    ax_bar = fig.add_axes([0.35, -0.18, 0.30, 0.14])
    colors = ['#e74c3c' if c == pred_label else '#3498db' for c in classes]
    bars   = ax_bar.barh(classes, probs * 100, color=colors, height=0.55)
    ax_bar.set_xlim(0, 100)
    ax_bar.set_xlabel('Güven (%)', color='white', fontsize=9)
    ax_bar.tick_params(colors='white', labelsize=8)
    ax_bar.set_facecolor('#16213e')
    for spine_name in ['bottom', 'left']: ax_bar.spines[spine_name].set_color('#444')
    ax_bar.spines['top'].set_visible(False)
    ax_bar.spines['right'].set_visible(False)
    for bar, prob in zip(bars, probs):
        ax_bar.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                    f'{prob*100:.1f}%', va='center', color='white', fontsize=7)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight',
                facecolor='#1a1a2e', edgecolor='none')
    plt.show()
    print(f"Sonuç kaydedildi: {save_path}")
    print(f"Gerçek: {true_label} | Tahmin: {pred_label} | Güven: {confidence*100:.2f}%")
    return pred_label, confidence


def _preprocess(path: str):
    """İç yardımcı: Görsel yükle, crop et, normalize et."""
    img     = cv2.imread(str(path))
    img     = robust_crop(img)
    img     = pad_resize(img, IMG_SIZE)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    return img_rgb, img_rgb.astype(np.float32) / 255.0


# ── Grad-CAM grid ─────────────────────────────────────────────────────────────

def plot_gradcam_grid(model, generator, classes_ordered: list,
                       tag: str = 'gradcam_grid',
                       n_images_per_class: int = 3,
                       art_dir: str = ART_DIR):
    """Tüm sınıflar için orijinal + overlay grid çizer."""
    print(f'Grad-CAM grid oluşturuluyor: {tag}…')
    num_classes = len(classes_ordered)
    fig, axes = plt.subplots(
        num_classes, n_images_per_class * 2,
        figsize=(n_images_per_class * 4, num_classes * 2.5),
    )
    fig.suptitle(f'Grad-CAM Grid — {tag}', fontsize=18)

    selected = {c: [] for c in range(num_classes)}
    idx_to_class = {v: k for k, v in generator.class_indices.items()}
    pairs = list(zip(generator.filepaths, generator.classes))
    random.shuffle(pairs)

    for filepath, label_idx in pairs:
        if len(selected[label_idx]) < n_images_per_class:
            selected[label_idx].append((filepath, label_idx))
        if all(len(selected[c]) == n_images_per_class for c in range(num_classes)):
            break

    for cls_idx in range(num_classes):
        class_name = idx_to_class[cls_idx]
        for i, (image_path, _) in enumerate(selected[cls_idx][:n_images_per_class]):
            img     = cv2.imread(image_path)
            img     = robust_crop(img)
            img     = pad_resize(img, IMG_SIZE)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            x       = img_rgb.astype(np.float32) / 255.0
            heatmap, pred_idx, probs = compute_gradcam(model, x[np.newaxis])
            pred_label = classes_ordered[pred_idx]
            ov = overlay_heatmap(img_rgb, heatmap)

            axes[cls_idx, i * 2].imshow(img_rgb)
            axes[cls_idx, i * 2].set_title(f'True: {class_name}', fontsize=10)
            axes[cls_idx, i * 2].axis('off')
            axes[cls_idx, i * 2 + 1].imshow(ov)
            axes[cls_idx, i * 2 + 1].set_title(
                f'Pred: {pred_label} ({probs[pred_idx]*100:.1f}%)', fontsize=10)
            axes[cls_idx, i * 2 + 1].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_path = os.path.join(art_dir, f'gradcam_grid_{tag}.png')
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f'Grad-CAM grid kaydedildi: {out_path}')


# ── t-SNE domain shift ────────────────────────────────────────────────────────

def tsne_domain_shift(model, id_gen, ext_gen, classes_ordered: list,
                       tag: str = 'tsne',
                       n_samples: int = 500,
                       art_dir: str = ART_DIR):
    """Kaggle ID ve Mendeley Ext özellik uzaylarını 2D'de görselleştirir."""
    print(f't-SNE oluşturuluyor: {tag}…')
    feature_extractor = tf.keras.Model(
        inputs=model.inputs, outputs=model.get_layer('gap').output)

    def get_feats(gen):
        feats = feature_extractor.predict(gen, verbose=0)
        return feats, gen.classes

    id_f, id_l   = get_feats(id_gen)
    ext_f, ext_l = get_feats(ext_gen)

    for feats, labels, n in [(id_f, id_l, n_samples), (ext_f, ext_l, n_samples)]:
        if len(feats) > n:
            idx    = np.random.choice(len(feats), n, replace=False)
            feats  = feats[idx]
            labels = labels[idx]

    all_feats   = np.vstack((id_f[:n_samples], ext_f[:n_samples]))
    all_domains = (
        ['Kaggle ID']    * min(len(id_f),  n_samples) +
        ['Mendeley Ext'] * min(len(ext_f), n_samples)
    )

    tsne    = TSNE(n_components=2, random_state=SEED, perplexity=30.0, n_iter=1000)
    results = tsne.fit_transform(all_feats)

    plt.figure(figsize=(10, 8))
    scatter = plt.scatter(
        results[:, 0], results[:, 1],
        c=[0 if d == 'Kaggle ID' else 1 for d in all_domains],
        cmap='viridis', alpha=0.6, s=50,
    )
    plt.title(f't-SNE Domain Shift ({tag})', fontsize=16)
    plt.xlabel('t-SNE 1', fontsize=12)
    plt.ylabel('t-SNE 2', fontsize=12)
    plt.legend(handles=scatter.legend_elements()[0],
               labels=['Kaggle ID', 'Mendeley Ext'], title='Domain')
    plt.grid(True, linestyle='--', alpha=0.7)
    out_path = os.path.join(art_dir, f'tsne_{tag}.png')
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f't-SNE kaydedildi: {out_path}')


# ── Öğrenme eğrileri ──────────────────────────────────────────────────────────

def plot_learning_curves(log_csv: str, art_dir: str = ART_DIR):
    """Eğitim logından accuracy, loss ve LR eğrilerini çizer."""
    import pandas as pd
    log_df = pd.read_csv(log_csv)
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    axes[0].plot(log_df['accuracy'],     label='train')
    axes[0].plot(log_df['val_accuracy'], label='val')
    axes[0].set_title('Accuracy'); axes[0].legend()
    axes[1].plot(log_df['loss'],     label='train')
    axes[1].plot(log_df['val_loss'], label='val')
    axes[1].set_title('Loss'); axes[1].legend()
    if 'lr' in log_df.columns:
        axes[2].plot(log_df['lr'])
        axes[2].set_title('Learning Rate')
    plt.tight_layout()
    out_path = os.path.join(art_dir, 'learning_curves.png')
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f'Öğrenme eğrileri kaydedildi: {out_path}')


def plot_ext_curve(ext_csv: str, art_dir: str = ART_DIR):
    """Epoch başına harici generalizasyon eğrisi (F1 + Balanced Acc)."""
    import pandas as pd
    ext_df = pd.read_csv(ext_csv)
    plt.figure(figsize=(10, 4))
    plt.plot(ext_df['epoch'], ext_df['ext_macro_f1'],  marker='o', label='Ext Macro F1')
    plt.plot(ext_df['epoch'], ext_df['ext_bal_acc'],   marker='s', label='Ext Bal Acc')
    plt.xlabel('Epoch')
    plt.title('Mendeley External Generalization per Epoch')
    plt.legend(); plt.grid(True)
    out_path = os.path.join(art_dir, 'ext_generalization_curve.png')
    plt.savefig(out_path, dpi=200)
    plt.show()
    print(f'Ext eğrisi kaydedildi: {out_path}')
