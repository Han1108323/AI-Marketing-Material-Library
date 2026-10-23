import os
import dashscope
import requests
import json
from http import HTTPStatus
from typing import Optional

class ImageGenerator:
    """
    Handles Text-to-Image generation using DashScope Qwen-Image (通义万相/Qwen).
    """
    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        dashscope.api_key = self.api_key
        # Use qwen-image-max for best quality
        self.model = "qwen-image-max" 
        # Default endpoint for Qwen-Image (Beijing Region)
        self.api_url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/multimodal-generation/generation"


    def generate(self, prompt: str, output_path: str = "generated_image.png") -> Optional[str]:
        """
        Generates an image from prompt using Qwen-Image-Max and saves it.
        Returns the path to the saved image or None if failed.
        """
        print(f"🎨 Generating image with {self.model} for: {prompt}")
        
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json"
        }

        # Qwen-Image API Payload
        data = {
            "model": self.model,
            "input": {
                "messages": [{
                    "role": "user",
                    "content": [{"text": prompt}]
                }]
            },
            "parameters": {
                "size": "1024*1024",
                "n": 1
            }
        }

        try:
            # Use direct HTTP call as SDK support for Qwen-Image might vary by version
            response = requests.post(self.api_url, headers=headers, json=data)
            
            if response.status_code == 200:
                res_json = response.json()
                
                # Try parsing Qwen-Image structure (output.choices[0].message.content[0].image)
                if "output" in res_json and "choices" in res_json["output"]:
                     choices = res_json["output"]["choices"]
                     if choices and len(choices) > 0:
                         content = choices[0].get("message", {}).get("content", [])
                         if content and len(content) > 0 and "image" in content[0]:
                             img_url = content[0]["image"]
                             print(f"   ⬇️ Downloading image from {img_url[:30]}...")
                             img_data = requests.get(img_url).content
                             with open(output_path, 'wb') as f:
                                 f.write(img_data)
                             print(f"✅ Image saved to {output_path}")
                             return output_path
                
                # Fallback check for Wanx structure (output.results[0].url) just in case
                if "output" in res_json and "results" in res_json["output"]:
                    result = res_json["output"]["results"][0]
                    if "url" in result:
                        img_url = result["url"]
                        print(f"   ⬇️ Downloading image from {img_url[:30]}...")
                        img_data = requests.get(img_url).content
                        with open(output_path, 'wb') as f:
                            f.write(img_data)
                        print(f"✅ Image saved to {output_path}")
                        return output_path

                if "code" in res_json:
                     print(f"❌ Qwen-Image API Error: {res_json.get('code')} - {res_json.get('message')}")
                     return None
                else:
                     print(f"❌ Unknown Response format: {res_json}")
                     return None
            else:
                print(f"❌ HTTP Request failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            print(f"❌ Generator Exception: {e}")
            return None

# Test block
if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    
    gen = ImageGenerator()
    # Simple test
    gen.generate("一只可爱的猫咪在草地上奔跑，卡通风格", "test_wanx.png")
