import requests
import re
from PIL import Image
from io import BytesIO

def get_main_image_from_item_url(url: str) -> Image.Image:
    """
    Extracts the first main image from a Taobao item URL.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.114 Safari/537.36",
        "Referer": "https://www.taobao.com/"
    }
    
    # Strategy 0: Direct Image URL
    if re.search(r'\.(jpg|jpeg|png|webp)$', url.split('?')[0], re.I) or 'img.alicdn.com' in url:
        print(f"🔗 Detected direct image URL: {url}")
        return _download_image(url)

    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        html = response.text
        
        # Strategy 1: Look for og:image meta tag
        # <meta property="og:image" content="https://img.alicdn.com/..." />
        og_image_match = re.search(r'<meta property="og:image" content="([^"]+)"', html)
        if og_image_match:
            img_url = og_image_match.group(1)
            return _download_image(img_url)
            
        # Strategy 2: Look for J_ImgBooth
        # <img id="J_ImgBooth" src="//img.alicdn.com/..." ... />
        img_booth_match = re.search(r'id="J_ImgBooth"[^>]*src="([^"]+)"', html)
        if img_booth_match:
            img_url = img_booth_match.group(1)
            if img_url.startswith('//'):
                img_url = 'https:' + img_url
            return _download_image(img_url)
            
        # Strategy 3: Look for general 400x400 or 800x800 image patterns if above fail
        # This is a fallback and might be less accurate
        general_match = re.search(r'//img\.alicdn\.com/imgextra/[^"]+\.jpg', html)
        if general_match:
            img_url = "https:" + general_match.group(0)
            return _download_image(img_url)

        print("⚠️ Could not find image in HTML. Response length:", len(html))
        return None

    except Exception as e:
        print(f"❌ Error fetching Taobao URL: {e}")
        return None

def _download_image(url: str) -> Image.Image:
    try:
        # Ensure URL is valid
        if not url.startswith('http'):
            if url.startswith('//'):
                url = 'https:' + url
            else:
                return None
                
        # Fix: Remove resize parameters if present to get full size
        # e.g., _.jpg_400x400.jpg -> _.jpg
        clean_url = re.sub(r'_\d+x\d+\.jpg.*$', '', url)
        clean_url = re.sub(r'_\.webp.*$', '', clean_url)

        resp = requests.get(clean_url, timeout=10)
        resp.raise_for_status()
        return Image.open(BytesIO(resp.content))
    except Exception as e:
        print(f"❌ Error downloading image: {e}")
        return None
