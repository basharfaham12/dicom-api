# enhancer.py
import os
import cv2
from PIL import Image, ImageEnhance

def enhance_images(image_paths, output_dir):
    os.makedirs(output_dir, exist_ok=True)

    def center_crop_brain(img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            cropped = img[y:y+h, x:x+w]
            return cropped
        return img

    def apply_clahe_and_soft_sharpen(img):
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
        cl = clahe.apply(l)
        merged = cv2.merge((cl, a, b))
        enhanced = cv2.cvtColor(merged, cv2.COLOR_LAB2BGR)
        blurred = cv2.GaussianBlur(enhanced, (3, 3), sigmaX=1.0)
        sharpened = cv2.addWeighted(enhanced, 1.2, blurred, -0.2, 0)
        return sharpened

    for path in image_paths:
        img = cv2.imread(path)
        centered = center_crop_brain(img)
        enhanced = apply_clahe_and_soft_sharpen(centered)
        resized = cv2.resize(enhanced, (224, 224))
        pil_img = Image.fromarray(cv2.cvtColor(resized, cv2.COLOR_BGR2RGB))
        filename = os.path.basename(path)
        pil_img.save(os.path.join(output_dir, filename))
