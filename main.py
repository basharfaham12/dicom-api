
import os
import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance
from tqdm import tqdm


# المسارات
input_dir = '/content/drive/MyDrive/U-Net_final_CN82/images'
output_dir = '/content/drive/MyDrive/Final2/CN'
target_total = 400

# إنشاء مجلد الإخراج إذا لم يكن موجودًا
os.makedirs(output_dir, exist_ok=True)

# تحميل الصور الأصلية
image_paths = [os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(('.png', '.jpg'))]
original_count = len(image_paths)
augment_count = target_total - original_count

def center_crop_brain(img):
    """توسيط الدماغ عبر اكتشاف أكبر كتلة غير سوداء"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
        cropped = img[y:y+h, x:x+w]
        return cropped
    return img

def apply_clahe_and_soft_sharpen(img):
    """تطبيق CLAHE + Sharpen ناعم"""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    cl = clahe.apply(l)
    merged = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)

    # Sharpen ناعم عبر دمج الصورة مع نسخة مشوشة منها
    blurred = cv2.GaussianBlur(enhanced, (3, 3), sigmaX=1.0)
    sharpened = cv2.addWeighted(enhanced, 1.2, blurred, -0.2, 0)
    return sharpened

def preprocess_image(path):
    """تحميل، توسيط، تحسين، وتحويل إلى PIL"""
    img = cv2.imread(path)
    centered = center_crop_brain(img)
    enhanced = apply_clahe_and_soft_sharpen(centered)
    resized = cv2.resize(enhanced, (224, 224))
    return Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))

def augment_image(img):
    """تعزيز الصورة باستخدام PIL فقط"""
    ops = []

    # قلب أفقي أو عمودي
    if random.random() < 0.5:
        ops.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    if random.random() < 0.3:
        ops.append(img.transpose(Image.FLIP_TOP_BOTTOM))

    # تعديل الإضاءة والتباين
    enhancer_b = ImageEnhance.Brightness(img)
    enhancer_c = ImageEnhance.Contrast(img)
    img = enhancer_b.enhance(random.uniform(0.8, 1.2))
    img = enhancer_c.enhance(random.uniform(0.8, 1.2))

    # تدوير عشوائي
    angle = random.uniform(-15, 15)
    img = img.rotate(angle)

    # ترجمة بسيطة
    dx = random.randint(-10, 10)
    dy = random.randint(-10, 10)
    img = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))

    # إعادة ضبط الحجم بعد التحويلات
    img = img.resize((224, 224))

    # إضافة الصور المقلوبة إن وُجدت
    ops.append(img)
    return random.choice(ops)

# حفظ الصور الأصلية بعد المعالجة
for path in image_paths:
    img = preprocess_image(path)
    img.save(os.path.join(output_dir, os.path.basename(path)))

# توليد الصور المعززة
for i in tqdm(range(augment_count), desc="تعزيز الصور"):
    path = random.choice(image_paths)
    img = preprocess_image(path)
    augmented_img = augment_image(img)
    new_name = f"aug_{i}_{os.path.basename(path)}"
    augmented_img.save(os.path.join(output_dir, new_name))

print(f"✅ تم حفظ {target_total} صورة بحجم موحد (224×224) في: {output_dir}")
