"""Generate synthetic 'background' / 'none' images for the negative class.

Creates diverse images that do NOT contain any of the target appliances:
  - Random noise patterns
  - Solid & gradient colors
  - Abstract textures
  - Random geometric shapes (simulating indoor scenes without appliances)

This gives the model a 'none of the above' option.

Usage:
    python train/generate_background.py
"""

import os
import numpy as np
from PIL import Image, ImageDraw, ImageFilter
import random

OUTPUT_DIR_TRAIN = "data/train/background"
OUTPUT_DIR_VAL = "data/val/background"
NUM_TRAIN = 150   # match the largest class
NUM_VAL = 20
IMG_SIZE = 320


def random_color():
    return tuple(random.randint(0, 255) for _ in range(3))


def make_noise_image():
    """Random noise."""
    arr = np.random.randint(0, 256, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    return Image.fromarray(arr)


def make_gradient_image():
    """Smooth gradient in random direction."""
    arr = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    c1 = np.array(random_color())
    c2 = np.array(random_color())
    
    for i in range(IMG_SIZE):
        t = i / IMG_SIZE
        color = (c1 * (1 - t) + c2 * t).astype(np.uint8)
        if random.random() > 0.5:
            arr[i, :] = color  # horizontal gradient
        else:
            arr[:, i] = color  # vertical gradient
    
    return Image.fromarray(arr)


def make_geometric_image():
    """Random geometric shapes (walls, floors, furniture silhouettes)."""
    img = Image.new('RGB', (IMG_SIZE, IMG_SIZE), random_color())
    draw = ImageDraw.Draw(img)
    
    num_shapes = random.randint(3, 12)
    for _ in range(num_shapes):
        shape = random.choice(['rect', 'ellipse', 'line', 'polygon'])
        color = random_color()
        
        if shape == 'rect':
            x1, y1 = random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE)
            x2, y2 = random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE)
            draw.rectangle([min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)], fill=color)
        elif shape == 'ellipse':
            x1, y1 = random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE)
            x2, y2 = random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE)
            draw.ellipse([min(x1,x2), min(y1,y2), max(x1,x2), max(y1,y2)], fill=color)
        elif shape == 'line':
            points = [(random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE)) 
                      for _ in range(random.randint(2, 5))]
            draw.line(points, fill=color, width=random.randint(1, 10))
        elif shape == 'polygon':
            points = [(random.randint(0, IMG_SIZE), random.randint(0, IMG_SIZE)) 
                      for _ in range(random.randint(3, 6))]
            draw.polygon(points, fill=color)
    
    # Sometimes add blur for more realism
    if random.random() > 0.5:
        img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(1, 5)))
    
    return img


def make_texture_image():
    """Procedural textures — wood-like, wall-like patterns."""
    base = np.random.randint(100, 200, (IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
    
    # Add stripe patterns
    stripe_w = random.randint(5, 30)
    for i in range(0, IMG_SIZE, stripe_w * 2):
        offset = random.randint(-20, 20)
        base[i:i+stripe_w] = np.clip(base[i:i+stripe_w].astype(int) + offset, 0, 255).astype(np.uint8)
    
    img = Image.fromarray(base)
    img = img.filter(ImageFilter.GaussianBlur(radius=random.uniform(0.5, 3)))
    return img


def make_solid_image():
    """Plain solid color or slight variation (walls, ceilings)."""
    base_color = random_color()
    arr = np.full((IMG_SIZE, IMG_SIZE, 3), base_color, dtype=np.uint8)
    noise = np.random.randint(-15, 15, (IMG_SIZE, IMG_SIZE, 3))
    arr = np.clip(arr.astype(int) + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(arr)


GENERATORS = [
    make_noise_image,
    make_gradient_image,
    make_geometric_image,
    make_texture_image,
    make_solid_image,
]


def generate_images(output_dir, count):
    os.makedirs(output_dir, exist_ok=True)
    
    for i in range(count):
        gen = random.choice(GENERATORS)
        img = gen()
        
        # Random transforms for more variety
        if random.random() > 0.5:
            img = img.rotate(random.randint(0, 360))
        if random.random() > 0.7:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        
        img.save(os.path.join(output_dir, f"bg_{i:04d}.jpg"), quality=90)
    
    print(f"  Generated {count} images → {output_dir}")


if __name__ == "__main__":
    print("Generating background/none images...")
    generate_images(OUTPUT_DIR_TRAIN, NUM_TRAIN)
    generate_images(OUTPUT_DIR_VAL, NUM_VAL)
    print("Done!")
