import os
from PIL import Image, ImageDraw, ImageFilter
import io
import dashscope
from http import HTTPStatus
import json
import re
# from rembg import remove, new_session  # Moved inside functions to avoid crash on Py3.13 without onnxruntime
import streamlit as st
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

@st.cache_resource
def get_rembg_session():
    """
    Initialize and cache the IS-Net session.
    IS-Net (isnet-general-use) is generally better for complex structures than u2net.
    """
    try:
        from rembg import new_session
        print("🔄 Loading IS-Net model...")
        return new_session("isnet-general-use")
    except Exception as e:
        print(f"⚠️ Failed to load rembg session: {e}")
        return None

def remove_background(image: Image.Image) -> Image.Image:
    """
    Removes background from the input PIL Image using IS-Net (via rembg).
    If rembg fails, applies a soft oval mask as a fallback.
    """
    try:
        # Check for onnxruntime availability first to avoid hard crashes/exits
        import importlib.util
        if importlib.util.find_spec("onnxruntime") is None:
             print("⚠️ onnxruntime not found. Using fallback mask.")
             return _apply_fallback_mask(image)

        # Optimization: Resize if image is too large (e.g. > 1024px) to speed up processing
        # Reduced from 1024 to 768 to improve speed on CPU while maintaining IS-Net quality
        max_dim = 768
        w, h = image.size
        scale_factor = 1.0
        if w > max_dim or h > max_dim:
            if w > h:
                new_w = max_dim
                new_h = int(h * (max_dim / w))
            else:
                new_h = max_dim
                new_w = int(w * (max_dim / h))
            image = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        
        session = get_rembg_session()
        if session is None:
             print("⚠️ rembg session is None. Using fallback mask.")
             return _apply_fallback_mask(image)
        
        # Convert to RGBA if not already
        if image.mode != 'RGBA':
            image = image.convert('RGBA')
            
        # Rembg expects bytes or PIL image
        from rembg import remove
        output = remove(image, session=session)
        return output
    except Exception as e:
        print(f"⚠️ IS-Net failed: {e}. Trying u2net fallback...")
        try:
            # Fallback to standard u2net
            from rembg import new_session, remove
            session_u2net = new_session("u2net")
            output = remove(image, session=session_u2net)
            return output
        except Exception as e2:
             print(f"⚠️ u2net failed: {e2}. Using fallback mask.")
             return _apply_fallback_mask(image)

def _apply_fallback_mask(image: Image.Image) -> Image.Image:
    """Applies a soft oval mask to the image (fallback for background removal)."""
    if image.mode != 'RGBA':
        image = image.convert('RGBA')
    
    w, h = image.size
    mask = Image.new('L', (w, h), 0)
    draw = ImageDraw.Draw(mask)
    
    # Draw oval
    draw.ellipse((int(w*0.1), int(h*0.1), int(w*0.9), int(h*0.9)), fill=255)
    
    # Blur mask for soft edge
    mask = mask.filter(ImageFilter.GaussianBlur(radius=20))
    
    result = image.copy()
    result.putalpha(mask)
    return result

def detect_product_bbox(image: Image.Image) -> list:
    """
    Use Qwen-VL to detect the main product bounding box.
    Returns [ymin, xmin, ymax, xmax] (normalized 0-1000) or None.
    """
    api_key = os.getenv("DASHSCOPE_API_KEY")
    if not api_key:
        print("⚠️ No DASHSCOPE_API_KEY found.")
        return None

    # Save temp image for API
    temp_path = f"temp_vlm_detect_{os.getpid()}.png"
    
    try:
        # Resize for speed (Qwen doesn't need 4K)
        max_dim = 1024
        scale_img = image.copy()
        w, h = scale_img.size
        if w > max_dim or h > max_dim:
            scale_img.thumbnail((max_dim, max_dim))
        
        scale_img.save(temp_path)
        file_path = os.path.abspath(temp_path)
        file_url = f"file://{file_path}"

        messages = [
            {
                "role": "user",
                "content": [
                    {"image": file_url},
                    {"text": "Detect the main product object in this image. Ignore human hands or complex backgrounds. Return the bounding box of the single main product in [ymin, xmin, ymax, xmax] format (0-1000). Only return the numbers."}
                ]
            }
        ]
        
        # Use Qwen-VL-Max for best understanding
        response = dashscope.MultiModalConversation.call(
            model='qwen-vl-max',
            messages=messages,
            api_key=api_key
        )

        if response.status_code == HTTPStatus.OK:
            text = response.output.choices[0].message.content[0]['text']
            # Parse [ymin, xmin, ymax, xmax]
            # Try to extract 4 consecutive numbers
            numbers = re.findall(r'\d+', text)
            if len(numbers) >= 4:
                # Qwen typically returns [ymin, xmin, ymax, xmax]
                return [int(numbers[0]), int(numbers[1]), int(numbers[2]), int(numbers[3])]
            print(f"⚠️ VLM Output format unexpected: {text}")
        else:
            print(f"⚠️ VLM Error: {response.message}")
            
    except Exception as e:
        print(f"⚠️ VLM Exception: {e}")
    finally:
        if os.path.exists(temp_path):
            try:
                os.remove(temp_path)
            except: pass
            
    return None

def smart_extract_product(image: Image.Image) -> tuple[Image.Image, Image.Image]:
    """
    Intelligently extracts the main product:
    1. Uses VLM (Qwen-VL) to detect the bounding box of the 'main product'.
    2. Crops the image to the bounding box (with padding).
    3. Applies IS-Net (rembg) to the crop to remove the background.
    
    Returns:
        (extracted_image_rgba, mask_image)
    """
    w, h = image.size
    
    # Step 1: Detect
    print("🔍 Detecting product with Qwen-VL...")
    bbox = detect_product_bbox(image)
    
    crop_box = None
    cropped_img = image
    
    if bbox:
        print(f"✅ Product detected at: {bbox}")
        ymin, xmin, ymax, xmax = bbox
        
        # Convert 0-1000 to pixels
        x1 = int(xmin / 1000 * w)
        y1 = int(ymin / 1000 * h)
        x2 = int(xmax / 1000 * w)
        y2 = int(ymax / 1000 * h)
        
        # Add 10% padding
        pad_w = (x2 - x1) * 0.1
        pad_h = (y2 - y1) * 0.1
        
        x1 = max(0, int(x1 - pad_w))
        y1 = max(0, int(y1 - pad_h))
        x2 = min(w, int(x2 + pad_w))
        y2 = min(h, int(y2 + pad_h))
        
        # Ensure crop is valid
        if x2 > x1 and y2 > y1:
            crop_box = (x1, y1, x2, y2)
            cropped_img = image.crop(crop_box)
            print(f"✂️ Cropped to: {crop_box}")
    else:
        print("⚠️ No bbox detected, using full image.")
        
    # Step 2: Matting (IS-Net)
    print("✂️ Removing background with IS-Net...")
    matted_crop = remove_background(cropped_img)
    
    # Extract Mask
    mask = matted_crop.split()[-1]
    
    return matted_crop, mask

def composite_product(product_img: Image.Image, bg_img: Image.Image) -> Image.Image:
    """
    Composites the product onto the background.
    1. Resizes product to appropriate scale relative to background.
    2. Centers product horizontally.
    3. Places product on the 'ground' (bottom 15-20%).
    4. Adds a simple shadow.
    """
    bg_w, bg_h = bg_img.size
    
    # Convert to RGBA
    if bg_img.mode != 'RGBA':
        bg_img = bg_img.convert('RGBA')
    if product_img.mode != 'RGBA':
        product_img = product_img.convert('RGBA')

    # Target product size: ~70% of background height (max) or ~80% width
    # Maintain aspect ratio
    p_w, p_h = product_img.size
    if p_h == 0: return bg_img # Avoid div by zero
    aspect = p_w / p_h
    
    target_h = int(bg_h * 0.7)
    target_w = int(target_h * aspect)
    
    # Constrain width if too wide
    if target_w > bg_w * 0.8:
        target_w = int(bg_w * 0.8)
        target_h = int(target_w / aspect)
        
    product_resized = product_img.resize((target_w, target_h), Image.Resampling.LANCZOS)
    
    # Position: Center X, Bottom Y (with some margin)
    x = (bg_w - target_w) // 2
    y = int(bg_h * 0.85) - target_h # 15% from bottom
    
    # Create Shadow
    # Simple oval shadow
    shadow = Image.new('RGBA', (bg_w, bg_h), (0,0,0,0))
    shadow_draw = ImageDraw.Draw(shadow)
    
    # Shadow ellipse at the bottom of the product
    s_w = int(target_w * 0.8)
    s_h = int(target_w * 0.2)
    s_x = x + (target_w - s_w) // 2
    s_y = y + target_h - (s_h // 2)
    
    shadow_draw.ellipse((s_x, s_y, s_x + s_w, s_y + s_h), fill=(0, 0, 0, 100))
    shadow = shadow.filter(ImageFilter.GaussianBlur(20))
    
    # Composite: BG -> Shadow -> Product
    result = Image.alpha_composite(bg_img, shadow)
    result.alpha_composite(product_resized, (x, y))
    
    return result
