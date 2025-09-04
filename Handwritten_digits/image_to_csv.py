import os
from PIL import Image, ImageOps, ImageEnhance
import numpy as np
import csv

# -------- SETTINGS --------
input_folder = "Handwritten_digits"         #replace with input folder path
output_csv = "output.csv"
contrast_factor = 2.0                       # enhance letter contrast
lower_threshold = 50                        # pixels <= this become 0 (background)
upper_threshold = 200                       # pixels >= this become 255 (letters)

# -------- PROCESS ALL IMAGES --------
rows = []

for filename in os.listdir(input_folder):
    if filename.lower().endswith((".jpg", ".jpeg", ".png")):
        path = os.path.join(input_folder, filename)

        # Open grayscale
        img = Image.open(path).convert("L")

        # Invert colors (so letters are white)
        img = ImageOps.invert(img)

        # Enhance contrast
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(contrast_factor)

        # Resize to 28x28
        img_resized = img.resize((28, 28))

        # Convert to NumPy
        pixels = np.array(img_resized, dtype=np.float32)

        # Apply thresholds
        pixels[pixels <= lower_threshold] = 0                # background → black
        pixels[pixels >= upper_threshold] = 255              # letters → white
        mask = (pixels > lower_threshold) & (pixels < upper_threshold)
        pixels[mask] = (pixels[mask] - lower_threshold) * (255 / (upper_threshold - lower_threshold))  # scale grays

        # Convert to int
        pixels = np.round(pixels).astype(int)

        # Flatten and append
        rows.append(pixels.flatten().tolist())

# -------- SAVE TO CSV --------
with open(output_csv, "w", newline="") as f:
    writer = csv.writer(f)

    # Write header pixel no
    header = [f"pixel{i}" for i in range(28*28)]
    writer.writerow(header)
    
    writer.writerows(rows)

print(f"Processed {len(rows)} images with background cleaned, saved to {output_csv}")

