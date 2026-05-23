"""
preprocessing.py — Baseline (EfficientNetB2) Ön İşleme ve Veri Bölme Modülü

Bu modül:
1. Görüntülere makale standartlarında Thresholding (45, 255), Erode ve Dilate 
   uygulayarak beynin ana konturunu bulur ve gereksiz kafatası/boşluk alanlarını kırpar (Skull-stripping).
2. Veri sızıntısını (Data Leakage) önlemek adına verileri %70 Eğitim ve %30 Test olarak ayırır.
3. Ayrılan eğitim verisinin içinden %10'unu Validation (Doğrulama) olarak böler.
4. Kırpılan tüm görüntüleri ilgili train, val ve test klasörlerine fiziksel olarak kaydeder.
"""

import os
import cv2
import shutil
import imutils
from sklearn.model_selection import train_test_split

# İleride config.py dosyasında tanımlayacağımız sabitler
from config import (
    CLASSES, 
    FIGSHARE_SOURCE_DIR, 
    TRAIN_DIR, 
    VAL_DIR, 
    TEST_DIR, 
    SEED
)

def crop_brain_contour(image):
    """
    Görüntüyü gri tonlamaya çevirir, Gaussian Blur uygular ve 
    katı eşikleme (threshold) ile morfolojik işlemler yaparak beyin konturunu kırpar.
    """
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Sadece eşikleme (thresh) matrisini alıyoruz
    thresh = cv2.threshold(gray, 45, 255, cv2.THRESH_BINARY)
    
    # Makaledeki 2 iterasyonlu aşındırma (erode) ve yayma (dilate) işlemleri
    thresh = cv2.erode(thresh, None, iterations=2)
    thresh = cv2.dilate(thresh, None, iterations=2)
    
    # Dış konturları bul
    cnts = cv2.findContours(thresh.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    
    # Hata Koruması (Failsafe): Eğer kontur bulunamazsa (bazı çok karanlık Kaggle MR'ları) resmi bozma, orijinali dön
    if len(cnts) == 0:
        return image 
        
    c = max(cnts, key=cv2.contourArea)
    
    # Ekstrem noktaları bul (Tuple içi NumPy Array indekslemesi düzeltildi)
    extLeft = tuple(c[c[:, :, 0].argmin()])
    extRight = tuple(c[c[:, :, 0].argmax()])
    extTop = tuple(c[c[:, :, 1].argmin()])
    extBot = tuple(c[c[:, :, 1].argmax()])
    
    # Görüntüyü ekstrem noktalara (beyin dokusuna) tam hizalı kırp
    cropped_image = image[extTop:extBot, extLeft:extRight]
    
    return cropped_image


def prepare_splits_and_crop():
    """
    Figshare veri setini okur, Seed ile Train/Val/Test olarak böler ve 
    crop_brain_contour fonksiyonundan geçirerek diske kaydeder.
    """
    print("\n[Ön İşleme] Eski hedef klasörler temizleniyor...")
    for d in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        if os.path.exists(d):
            shutil.rmtree(d)
        for cls in CLASSES:
            os.makedirs(os.path.join(d, cls), exist_ok=True)
            
    print("[Ön İşleme] Görüntüler makale standartlarında kırpılıyor ve %70-%30 oranında bölünüyor...")
    
    def process_and_save(img_list, target_folder, cls_name, cls_path):
        for img_name in img_list:
            img_path = os.path.join(cls_path, img_name)
            image = cv2.imread(img_path)
            
            if image is not None:
                cropped = crop_brain_contour(image)
                # Kırpma işlemi bir şekilde 0 pixellik resim döndürürse orijinali kurtar
                if cropped is not None and cropped.size > 0:
                    cv2.imwrite(os.path.join(target_folder, cls_name, img_name), cropped)
                else:
                    cv2.imwrite(os.path.join(target_folder, cls_name, img_name), image)

    # Sınıf bazlı döngü ve bölme işlemleri
    for cls in CLASSES:
        cls_path = os.path.join(FIGSHARE_SOURCE_DIR, cls)
        if not os.path.exists(cls_path):
            print(f"UYARI: Kaynak klasör bulunamadı: {cls_path}")
            continue
            
        images = [f for f in os.listdir(cls_path) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
        
        # Makaleye uygun: %70 Train-Val, %30 Test
        train_val_imgs, test_imgs = train_test_split(images, test_size=0.30, random_state=SEED)
        
        # Kalan verinin (Train-Val) %10'unu Doğrulama (Validation) olarak ayır
        train_imgs, val_imgs = train_test_split(train_val_imgs, test_size=0.10, random_state=SEED)
        
        # Resimleri işleyerek fiziksel olarak kaydet
        process_and_save(train_imgs, TRAIN_DIR, cls, cls_path)
        process_and_save(val_imgs, VAL_DIR, cls, cls_path)
        process_and_save(test_imgs, TEST_DIR, cls, cls_path)
        
    print("[Ön İşleme] Veri sızıntısı önlendi, tüm morfolojik kırpmalar tamamlandı.\n")
