import os
import requests
import time
import json
import uuid
from typing import Dict, List, Optional
from PIL import Image, ImageStat
import io
import numpy as np
from dotenv import load_dotenv
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct
from huggingface_hub import InferenceClient
import dashscope
from http import HTTPStatus
import math

# Load environment variables
load_dotenv()

# Constants
TIMEOUT = 60
MAX_RETRIES = 3

class ProcessorBase:
    """Base class for all processors"""
    def __init__(self, name: str):
        self.name = name
        
    def _make_request(self, method: str, url: str, **kwargs) -> Optional[requests.Response]:
        pass

class OCRProcessor(ProcessorBase):
    """Handles OCR using OCR.space API"""
    def __init__(self):
        super().__init__("OCR")
        self.api_key = os.getenv("OCR_KEY") or os.getenv("OCRSPACE_API_KEY") or "helloworld"
        
    def process(self, image: Image.Image) -> Dict:
        """
        Returns {'text': '...', 'raw': ...}
        PRIMARY: OCR.space API (chs + isOverlayRequired=True → words bbox)
        FALLBACK (if helloworld quota / network error):
          1) PaddleOCR (if installed) → convert result to OCR.space ParsedResults format
          2) EasyOCR   (if installed) → same conversion
        This guarantees ocr_meta["raw"]["ParsedResults"][...]["TextOverlay"]["Lines"][...]["Words"]
        ALWAYS exists for downstream word-level tight bbox (history logic restored).
        """
        # Convert image to bytes
        img_byte_arr = io.BytesIO()
        image.save(img_byte_arr, format='PNG')
        img_bytes = img_byte_arr.getvalue()

        # ============================================================
        # LAYER 1: OCR.space API call with retries
        # ============================================================
        url = "https://api.ocr.space/parse/image"
        for attempt in range(MAX_RETRIES):
            try:
                payload = {
                    'apikey': self.api_key,
                    'language': 'chs',
                    'isOverlayRequired': True,
                }
                files = {'file': ('image.png', img_bytes, 'image/png')}
                response = requests.post(url, files=files, data=payload, timeout=TIMEOUT, verify=False)
                if response.status_code == 200:
                    result = response.json()
                    if result.get("IsErroredOnProcessing"):
                        continue
                    parsed_results = result.get("ParsedResults", [])
                    if parsed_results:
                        # Quick sanity: must have Lines/Words overlay for tight bboxes
                        has_overlay = bool(
                            parsed_results[0].get("TextOverlay", {}).get("Lines")
                        )
                        text = parsed_results[0].get("ParsedText", "").strip()
                        if has_overlay or text:
                            return {"text": text, "raw": result}
            except Exception as e:
                print(f"⚠️ OCR Attempt {attempt+1} Error: {e}")
                time.sleep(1)

        # ============================================================
        # LAYER 2: Local OCR fallback (PaddleOCR → EasyOCR → Mock)
        # ============================================================
        w_px, h_px = image.size
        lines_from_local = []
        full_text_parts = []

        # --- 2a: PaddleOCR (preferred, supports Chinese well) ---
        try:
            from paddleocr import PaddleOCR
            _ocr = PaddleOCR(use_angle_cls=True, lang="ch", show_log=False)
            res = _ocr.ocr(image, cls=True)
            if res and res[0]:
                # res[0] list of [ [poly_4pts, (text, score)] ]
                words_so_far = 0
                for entry in res[0]:
                    try:
                        poly, (txt, score) = entry
                        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
                        left = float(min(xs)); top = float(min(ys))
                        right_f = float(max(xs)); bot_f = float(max(ys))
                        width = max(1.0, right_f - left); height = max(1.0, bot_f - top)
                        # Each result becomes ONE LINE with ONE WORD (simplest valid overlay)
                        lines_from_local.append({
                            "LineText": txt,
                            "Words": [{
                                "WordText": txt,
                                "Left": left, "Top": top,
                                "Height": height, "Width": width,
                            }],
                            "MaxHeight": height, "MinTop": top,
                        })
                        full_text_parts.append(txt)
                    except Exception:
                        continue
        except Exception as _e_paddle:
            print(f"ℹ️  PaddleOCR fallback not available: {_e_paddle}")

        # --- 2b: EasyOCR (second choice, also Chinese-capable) ---
        if not lines_from_local:
            try:
                import easyocr
                self._easy_reader = getattr(self, "_easy_reader", None)
                if self._easy_reader is None:
                    self._easy_reader = easyocr.Reader(['ch_sim','en'], gpu=False, verbose=False)
                res2 = self._easy_reader.readtext(image)
                for entry in res2:
                    try:
                        poly, txt, score = entry
                        if not txt: continue
                        xs = [p[0] for p in poly]; ys = [p[1] for p in poly]
                        left = float(min(xs)); top = float(min(ys))
                        right_f = float(max(xs)); bot_f = float(max(ys))
                        width = max(1.0, right_f - left); height = max(1.0, bot_f - top)
                        lines_from_local.append({
                            "LineText": str(txt),
                            "Words": [{
                                "WordText": str(txt),
                                "Left": left, "Top": top,
                                "Height": height, "Width": width,
                            }],
                            "MaxHeight": height, "MinTop": top,
                        })
                        full_text_parts.append(str(txt))
                    except Exception:
                        continue
            except Exception as _e_easy:
                print(f"ℹ️  EasyOCR fallback not available: {_e_easy}")

        # --- Build OCR.space-compatible raw dict ---
        raw_compat = {
            "ParsedResults": [{
                "TextOverlay": {"Lines": lines_from_local},
                "FileParseExitCode": 1,
                "ParsedText": "\n".join(full_text_parts),
                "ErrorMessage": "",
                "ErrorDetails": "",
            }],
            "IsErroredOnProcessing": False,
            "ErrorMessage": None,
            "SearchablePDFURL": "",
            "ProcessingTimeInMilliseconds": "0",
        }
        return {"text": "\n".join(full_text_parts).strip(), "raw": raw_compat}

class VisionProcessor(ProcessorBase):
    """Generates visual description using Qwen-VL-Max"""
    def __init__(self):
        super().__init__("Vision")
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        dashscope.api_key = self.api_key
        self.model = "qwen-vl-max"

    def process(self, image: Image.Image) -> Dict:
        """
        Phase 1: Visual Perception (The Eye)
        Returns {
            "description": "...",
            "visual_style": "...",
            "color_tone": "...",
            "composition": "..."
        }
        """
        temp_path = f"temp_vision_{uuid.uuid4()}.png"
        
        # Optimization: Resize for Vision API (Max 1024px) to speed up upload/inference
        max_dim = 1024
        w, h = image.size
        scale_img = image.copy()
        if w > max_dim or h > max_dim:
            if w > h:
                new_w = max_dim
                new_h = int(h * (max_dim / w))
            else:
                new_h = max_dim
                new_w = int(w * (max_dim / h))
            scale_img = scale_img.resize((new_w, new_h), Image.Resampling.LANCZOS)
            
        scale_img.save(temp_path)
        file_path = os.path.abspath(temp_path)
        file_url = f"file://{file_path}"
        
        prompt = """
        你是‘视觉感知层’(The Eye)。请观察这张图片，完成以下视觉分析任务(思维链)：
        1. 先定风格：从 [赛博朋克, 极简主义, 商务专业, 温馨居家, 科技未来, 复古怀旧, 清新自然, 奢华高端, 活力运动, 艺术创意] 中选择最符合的一个。
        2. 再看色彩：分析主色调、饱和度(高/低)、对比度(强/弱)。
        3. 最后看构图：是中心构图、三分法、对角线构图还是留白？
        4. 视觉描述：用自然语言详细描述画面内容(主体、背景、动作等)，用于后续检索。

        请输出严格 JSON:
        {
            "visual_style": "风格",
            "color_tone": "色彩描述",
            "composition": "构图方式",
            "description": "详细的自然语言描述..."
        }
        """
        
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": file_url},
                    {"text": prompt}
                ]
            }
        ]
        
        default_res = {
            "visual_style": "Unknown",
            "color_tone": "Unknown",
            "composition": "Unknown",
            "description": "无法识别图片内容"
        }

        try:
            response = dashscope.MultiModalConversation.call(
                model=self.model,
                messages=messages
            )
            
            # Cleanup
            if os.path.exists(temp_path):
                os.remove(temp_path)
                
            if response.status_code == HTTPStatus.OK:
                content = response.output.choices[0].message.content[0]['text']
                # Parse JSON
                try:
                    start = content.find('{')
                    end = content.rfind('}') + 1
                    if start != -1 and end != -1:
                        json_str = content[start:end]
                        data = json.loads(json_str)
                        return data
                except:
                    pass
                # Fallback if no JSON found but text exists
                default_res["description"] = content
                return default_res
            else:
                return default_res
                
        except Exception as e:
            if os.path.exists(temp_path):
                os.remove(temp_path)
            print(f"⚠️ Vision Error: {e}")
            return default_res

class MultiAgentCTRModel(ProcessorBase):
    """
    Multi-Agent Scoring Model using Qwen-VL-Max to simulate 3 expert personas:
    1. Visual Designer (Stopping Power)
    2. Copy Strategist (Readability)
    3. Ad Optimizer (Click Desire)
    Returns a score out of 30.
    """
    def __init__(self):
        super().__init__("MultiAgentCTR")
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        dashscope.api_key = self.api_key
        self.model = "qwen-vl-max" 

    def predict(self, image_path: str, ocr_text: str, caption: str) -> Dict:
        """
        Returns {
            "predicted_score": int (0-30),
            "breakdown": {role: score},
            "reasoning": str
        }
        """
        prompt = f"""
你现在需要同时扮演三个营销专家角色，对这张广告素材进行严格评分。
总分30分，每个角色10分。

角色一：视觉设计师 (Visual Designer)
核心任务：停下来 (Stopping Power)
评估维度：色彩冲击力、构图清晰度、主体突出度。
指令：你不需要管文案写什么，你只看这张图在混杂的信息流里，能不能让人一眼看到。解决‘视而不见’的问题。

角色二：文案策略师 (Copy Strategist)
核心任务：读进去 (Readability)
评估维度：利益点是否直白（5折 > 优惠）、痛点是否精准、信息层级是否混乱。
指令：忽略画面美丑，只看文字信息。这句文案能打动30岁女性吗？解决‘不知所云’的问题。
参考文字内容：{ocr_text}

角色三：投放优化师 (Ad Optimizer)
核心任务：点下去 (Click Desire)
评估维度：点击诱导（CTA按钮明显吗？）、紧迫感营造（有限时标吗？）、商业气息（像不像正规广告？）。
指令：判断用户是否有点击欲望。解决‘叫好不叫座’的问题。

请输出 JSON 格式结果：
{{
    "visual_score": 整数(0-10),
    "copy_score": 整数(0-10),
    "ad_score": 整数(0-10),
    "total_score": 整数(0-30),
    "reasoning": "简短的一句话综合评价"
}}
        """
        messages = [
            {
                "role": "user",
                "content": [
                    {"image": image_path},
                    {"text": prompt}
                ]
            }
        ]
        
        try:
            response = dashscope.MultiModalConversation.call(
                model=self.model,
                messages=messages
            )
            
            if response.status_code == HTTPStatus.OK:
                content = response.output.choices[0].message.content[0]['text']
                # Parse JSON
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1 and end != -1:
                    json_str = content[start:end]
                    data = json.loads(json_str)
                    return {
                        "predicted_score": data.get("total_score", 0),
                        "breakdown": {
                            "visual": data.get("visual_score", 0),
                            "copy": data.get("copy_score", 0),
                            "ad": data.get("ad_score", 0)
                        },
                        "reasoning": data.get("reasoning", "")
                    }
        except Exception as e:
            print(f"⚠️ MultiAgentCTR Error: {e}")
            
        return {
            "predicted_score": 15, 
            "breakdown": {"visual": 5, "copy": 5, "ad": 5},
            "reasoning": "评分失败，使用默认值"
        }

class TagGenerator(ProcessorBase):
    """
    Phase 2: The Brain (Marketing Strategy)
    Uses Qwen-Turbo + OCR to analyze Scene, Audience, and Benefit.
    """
    def __init__(self):
        super().__init__("TagGen")
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        dashscope.api_key = self.api_key
        self.model = dashscope.Generation.Models.qwen_turbo

    def generate(self, ocr_text: str, visual_data: Dict) -> Dict:
        """
        Synthesizes Marketing Tags from Visual Data + OCR.
        """
        visual_desc = visual_data.get("description", "")
        visual_style = visual_data.get("visual_style", "")
        
        prompt = f"""
        你是‘营销策略层’(The Brain)。请基于视觉分析和OCR文案，深度拆解这张图的"高转化因子"，推导营销策略。
        
        输入信息:
        - 视觉风格: {visual_style}
        - 视觉描述: {visual_desc}
        - OCR文案: {ocr_text}
        
        任务(思维链):
        1. 识别品类：判断商品所属的垂直类目（如：口红、运动鞋、扫地机器人）。
        2. 场景与受众：分析使用场景(节日/日常/大促)和目标人群(Z世代/白领等)。
        3. 拆解背景：详细描述背景的视觉元素（颜色、材质、道具、光影）。例如："暖橙色渐变背景，带有科技感的线条装饰"。
        4. 拆解文案：提取核心营销文案。如果OCR有内容则优化提取；如果没有，请根据画面意图撰写一句高转化文案。
        5. 拆解文案样式：分析文案的字体风格（手写体/黑体）、颜色（红底白字/金字）、特效（发光/阴影）。
        6. 拆解文案位置：指出文案在画面中的布局位置（如：左上角、正下方、居中悬浮、右侧留白处）。
        7. 综合描述：生成一段50-100字的综合描述，融合视觉与营销信息。

        请输出严格 JSON:
        {{
          "product_category": "商品品类",
          "scene_type": "场景",
          "target_audience": ["受众1", "受众2"],
          "key_benefit": "核心利益点",
          "material_purpose": "用途",
          "background_desc": "背景的详细视觉描述",
          "marketing_copy": "核心营销文案",
          "copy_style": "文案的字体/颜色/特效描述",
          "copy_position": "文案的布局位置描述",
          "description": "融合视觉与营销的综合自然语言描述"
        }}
        """
        
        tags = {
            "product_category": "Unknown",
            "scene_type": "Unknown",
            "target_audience": [],
            "key_benefit": "Unknown",
            "material_purpose": "Unknown",
            "background_desc": "",
            "marketing_copy": "",
            "copy_style": "",
            "copy_position": "",
            "description": visual_desc # Fallback to visual description
        }

        for attempt in range(MAX_RETRIES):
            try:
                response = dashscope.Generation.call(
                    model=self.model,
                    prompt=prompt,
                    result_format='message'
                )
                
                if response.status_code == HTTPStatus.OK:
                    content = response.output.choices[0].message.content
                    try:
                        start = content.find('{')
                        end = content.rfind('}') + 1
                        if start != -1 and end != -1:
                            json_str = content[start:end]
                            data = json.loads(json_str)
                            tags.update(data)
                            return tags
                    except:
                        pass
                else:
                    print(f"⚠️ TagGen Attempt {attempt+1} Failed: {response.message}")
                    
            except Exception as e:
                print(f"⚠️ TagGen Attempt {attempt+1} Error: {e}")
                time.sleep(1)

        return tags

class EmbeddingGenerator(ProcessorBase):
    """Generates vector embeddings using DashScope"""
    def __init__(self):
        super().__init__("Embedding")
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        dashscope.api_key = self.api_key

    def generate(self, text: str) -> Optional[List[float]]:
        try:
            resp = dashscope.TextEmbedding.call(
                model=dashscope.TextEmbedding.Models.text_embedding_v1,
                input=text
            )
            if resp.status_code == HTTPStatus.OK:
                if hasattr(resp.output, 'embeddings'):
                    return resp.output.embeddings[0].embedding
                else:
                    return resp.output['embeddings'][0]['embedding']
            else:
                print(f"❌ Embedding Error: {resp.message}")
                return None
        except Exception as e:
            print(f"❌ Embedding Exception: {e}")
            return None

from db_client import get_qdrant_client

class VectorDB:
    """Manages Qdrant Interactions"""
    def __init__(self):
        # Use the unified client getter
        self.client = get_qdrant_client()
        self.collection_name = "material_library"
        self.vector_size = 1536 # DashScope v1 size

    def ensure_collection(self):
        try:
            self.client.get_collection(self.collection_name)
        except:
            print(f"Creating collection {self.collection_name}...")
            self.client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(size=self.vector_size, distance=Distance.COSINE)
            )

    def insert(self, vector: List[float], payload: Dict, id: str):
        self.client.upsert(
            collection_name=self.collection_name,
            points=[
                PointStruct(
                    id=id,
                    vector=vector,
                    payload=payload
                )
            ]
        )

class MaterialPipeline:
    """Orchestrates the full ingestion process"""
    def __init__(self, init_db=True):
        print("   Init OCR...")
        self.ocr = OCRProcessor()
        print("   Init Vision...")
        self.vision = VisionProcessor()
        print("   Init TagGen...")
        self.tag_gen = TagGenerator()
        print("   Init MultiAgentCTR...")
        self.ctr_model = MultiAgentCTRModel()
        print("   Init Embedder...")
        self.embedder = EmbeddingGenerator()
        
        self.db = None
        if init_db:
            print("   Init DB...")
            self.db = VectorDB()
            print("   Ensuring Collection...")
            self.db.ensure_collection()
            
        print("   Pipeline Init Done.")

    def process_image(self, image: Image.Image, filename: str = "unknown", user_id: str = None):
        print(f"🔄 Processing {filename}... (User: {user_id})")
        
        # Save temp file for VL model
        temp_dir = "AI素材案例"
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        image_path = os.path.abspath(os.path.join(temp_dir, filename))
        
        # Fix RGBA -> JPEG/WebP save errors (webp is often RGBA)
        save_image = image.copy()
        ext = os.path.splitext(filename)[1].lower()
        # webp/png preserve alpha; jpg/jpeg convert to RGB to avoid IOError
        rgba_or_palette = save_image.mode in ('RGBA', 'LA', 'P')
        if rgba_or_palette and ext in ('.jpg', '.jpeg', '.webp'):
            if save_image.mode == 'P':
                save_image = save_image.convert('RGBA')
            background = Image.new('RGB', save_image.size, (255, 255, 255))
            background.paste(save_image, mask=save_image.split()[-1] if 'A' in save_image.mode else None)
            save_image = background
            
        if not os.path.exists(image_path):
            save_image.save(image_path)
        file_url = f"file://{image_path}"
        
        # 1. OCR (Returns Dict now)
        ocr_data = self.ocr.process(image)
        ocr_text = ocr_data.get("text", "")
        print(f"   📝 OCR: {ocr_text[:50]}..." if ocr_text else "   📝 OCR: [No Text]")

        # 2. Vision (Phase 1)
        visual_data = self.vision.process(image)
        description = visual_data.get("description", "")
        print(f"   👁️  Vision: {description[:50]}...")

        # 3. Tag Generation (Phase 2)
        marketing_tags = self.tag_gen.generate(ocr_text, visual_data)
        
        # Merge Tags
        tags = {**visual_data, **marketing_tags}
        print(f"   🧠 Tags: {json.dumps(tags, ensure_ascii=False)[:100]}...")

        # 4. Multi-Agent Scoring
        print("   🤖 Multi-Agent Scoring...")
        ctr_result = self.ctr_model.predict(file_url, ocr_text, description)
        predicted_score = ctr_result["predicted_score"]
        print(f"   📊 Score: {predicted_score}/30")
        print(f"      Breakdown: {ctr_result['breakdown']}")
        
        # Enrich tags
        tags['predicted_score'] = predicted_score
        tags['score_breakdown'] = ctr_result['breakdown']
        tags['score_reasoning'] = ctr_result['reasoning']
        
        # Legacy/Compat (Map score 30 -> 0.05 CTR)
        ctr_val = (predicted_score / 30.0) * 0.05
        tags['predicted_ctr'] = round(ctr_val * 100, 2)
        tags['predicted_cvr'] = round(ctr_val * 0.3 * 100, 2)

        # 5. Generate Search Vector
        # Include description in embedding for better recall
        combined_text = f"{tags.get('description', '')} {tags.get('scene_type','')} {tags.get('key_benefit','')} {tags.get('visual_style','')} {tags.get('material_purpose','')} {ocr_text}"
        vector = self.embedder.generate(combined_text)

        # 6. Store
        if not vector:
            print("⚠️ Embedding failed (API Error?), using Zero Vector for storage.")
            # Use zero vector of size 1536
            vector = [0.0] * 1536
            
        if vector:
            # Use deterministic ID based on filename to support updates/re-indexing
            image_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, filename))
            
            # Use the merged description (Marketing + Visual) for the caption/display
            final_description = tags.get('description', description)
            
            payload = {
                "filename": filename,
                "user_id": user_id,
                "ocr_text": ocr_text,
                "ocr_meta": ocr_data, # Store full meta
                "caption": final_description, # Use merged description
                "tags": tags,
                "predicted_score": predicted_score,
                "score_breakdown": ctr_result['breakdown'],
                "score_reasoning": ctr_result['reasoning'],
                "historical_ctr": ctr_val, # Use mapped CTR for compatibility
                "validation_status": "pending",
                "timestamp": time.time()
            }
            
            if self.db:
                self.db.insert(vector, payload, image_id)
                print(f"✅ Indexed {filename} (ID: {image_id})")
            else:
                print(f"ℹ️ DB not initialized. Returning payload for {filename}.")
            
            return payload, vector
            
        return None, None
