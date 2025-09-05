import cv2
import numpy as np
import random
from PIL import Image, ImageEnhance

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

def augment_image(img):
    ops = []
    if random.random() < 0.5:
        ops.append(img.transpose(Image.FLIP_LEFT_RIGHT))
    if random.random() < 0.3:
        ops.append(img.transpose(Image.FLIP_TOP_BOTTOM))

    enhancer_b = ImageEnhance.Brightness(img)
    enhancer_c = ImageEnhance.Contrast(img)
    img = enhancer_b.enhance(random.uniform(0.8, 1.2))
    img = enhancer_c.enhance(random.uniform(0.8, 1.2))

    angle = random.uniform(-15, 15)
    img = img.rotate(angle)

    dx = random.randint(-10, 10)
    dy = random.randint(-10, 10)
    img = img.transform(img.size, Image.AFFINE, (1, 0, dx, 0, 1, dy))

    img = img.resize((224, 224))
    ops.append(img)
    return random.choice(ops)
