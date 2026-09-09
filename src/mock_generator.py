import requests
from PIL import Image, ImageDraw, ImageFont
import io
import random

def create_mock_image(text="TEST MATERIAL"):
    """
    Creates a simple image with text for testing OCR and Vision models.
    Returns: PIL Image object
    """
    width, height = 800, 600
    color = (random.randint(200, 255), random.randint(200, 255), random.randint(200, 255))
    img = Image.new('RGB', (width, height), color=color)
    d = ImageDraw.Draw(img)
    
    # Add some shapes
    for _ in range(5):
        x_coords = sorted([random.randint(0, width), random.randint(0, width)])
        y_coords = sorted([random.randint(0, height), random.randint(0, height)])
        fill = (random.randint(0, 200), random.randint(0, 200), random.randint(0, 200))
        d.rectangle([x_coords[0], y_coords[0], x_coords[1], y_coords[1]], outline=fill, width=3)

    # Add text
    try:
        # Try to use a default font
        font = ImageFont.load_default()
        # Scale isn't easily possible with load_default, but it's enough for OCR
    except:
        font = None

    d.text((width//2 - 50, height//2), text, fill=(0, 0, 0))
    d.text((10, 10), "AI MATERIAL LIBRARY TEST", fill=(255, 0, 0))
    
    return img

def image_to_bytes(img):
    """Convert PIL image to bytes for API upload"""
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='PNG')
    img_byte_arr = img_byte_arr.getvalue()
    return img_byte_arr
