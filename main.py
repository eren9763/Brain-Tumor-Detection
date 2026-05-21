"""
main.py — Eğitim + tam değerlendirme + görselleştirme boru hattı.

Kullanım:
    python main.py                   # Her şeyi indir, değerlendirme + görseller
    python main.py --train           # Sıfırdan eğit
    python main.py --model-path /path/to/model.keras
    python main.py --help
"""
import os
import argparse
import numpy as np
import tensorflow as tf

from config import (
    CLASSES, ART_DIR, SAVE_PATH, TRAIN_LOG, EXT_TEST_DIR,
)
from download_data import download_all
from data_pipeline import prepare_split_dirs, build_generators
from evaluate import run_evaluation
from visualize import (
    plot_learning_curves, plot_ext_curve,
    plot_gradcam_grid, tsne_domain_shift, visualize_gradcam,
)


os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL']  = '3'
os.environ['TF_NUM_INTRAOP_THREADS'] = '8'
os.environ['TF_NUM_INTEROP_THREADS'] = '2'
os.environ['OMP_NUM_THREADS']        = '8'

def parse_args():
    parser = argparse.ArgumentParser(
        description='Brain Tumor MRI — Cross-Dataset Generalization Pipeline')
    parser.add_argument('--train', action='store_true',
                        help='Modeli sıfırdan eğit (varsayılan: pretrained kullan)')
    parser.add_argument('--model-path', type=str, default=SAVE_PATH,
                        help=f'Model yolu (varsayılan: {SAVE_PATH})')
    parser.add_argument('--skip-download', action='store_true',
                        help='İndirme adımını atla (veriler zaten mevcutsa)')

    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--quick', action='store_true',
                  help='Hızlı mod: sadece plain eval + temel Grad-CAM')
    mode.add_argument('--full', action='store_true',
                  help='Tam mod: plain + TTA eval + tüm görseller')
    return parser.parse_args()


def main():
    args = parse_args()

    # ── 1. İndirme ────────────────────────────────────────────────────────────
    if not args.skip_download:
        download_all()

    # ── 2. Eğitim (opsiyonel) ─────────────────────────────────────────────────
    if args.train:
        from train import main as train_main
        train_main()
        model_path = SAVE_PATH
    else:
        model_path = args.model_path

    # ── 3. Model yükle ────────────────────────────────────────────────────────
    print(f'\nModel yükleniyor: {model_path}')
    from model import build_convnext

    model, _ = build_convnext(freeze_backbone=False)
    model.load_weights(model_path)

    # ── 4. Veri pipeline ──────────────────────────────────────────────────────
    prepare_split_dirs()
    _, _, val_gen, idtest_gen, exttest_gen, _, classes_ordered = (
        build_generators()
    )

    # ── 5. Değerlendirme ──────────────────────────────────────────────────────
    if args.quick:
        print("[ QUICK MOD ] Sadece plain değerlendirme çalışıyor…")
        run_evaluation(model, idtest_gen, exttest_gen, classes_ordered, run_tta=False)
    elif args.full:
        print("[ FULL MOD ] Plain + TTA değerlendirme çalışıyor…")
        run_evaluation(model, idtest_gen, exttest_gen, classes_ordered, run_tta=True)
    else:
        print("[ STANDART MOD ] Plain değerlendirme çalışıyor…")
        run_evaluation(model, idtest_gen, exttest_gen, classes_ordered, run_tta=False)

    # ── 6. Öğrenme eğrileri (eğitim yapıldıysa) ───────────────────────────────
    ext_csv = os.path.join(ART_DIR, 'ext_eval_stage2.csv')
    if os.path.exists(TRAIN_LOG):
        plot_learning_curves(TRAIN_LOG)
    if os.path.exists(ext_csv):
        plot_ext_curve(ext_csv)

    # ── 7. Grad-CAM ───────────────────────────────────────────────────────────
    if not args.quick:
        plot_gradcam_grid(model, idtest_gen,  classes_ordered, tag='id_test')
        plot_gradcam_grid(model, exttest_gen, classes_ordered, tag='ext_test')
        tsne_domain_shift(model, idtest_gen, exttest_gen, classes_ordered, tag='final')

    # ── 9. Sınıf başına örnek Grad-CAM ───────────────────────────────────────
    for cls in classes_ordered:
        cls_dir = os.path.join(EXT_TEST_DIR, cls)
        if not os.path.isdir(cls_dir):
            continue
        imgs = [f for f in os.listdir(cls_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if imgs:
            visualize_gradcam(
                image_path=os.path.join(cls_dir, imgs[0]),
                true_label=cls,
                model=model,
                classes=classes_ordered,
                save_path=os.path.join(ART_DIR, f'gradcam_{cls}.png'),
            )

    print('\n✓ Tüm çıktılar kaydedildi:', ART_DIR)


if __name__ == '__main__':
    main()
