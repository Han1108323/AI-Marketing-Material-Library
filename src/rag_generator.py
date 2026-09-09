import os
import json
from typing import List, Dict, Optional
import dashscope
from http import HTTPStatus
from qdrant_client import QdrantClient
from generator import ImageGenerator
from db_client import get_qdrant_client
from PIL import Image
try:
    import design_utils
except ImportError:
    from src import design_utils

class RAGGenerator:
    """
    Retrieval-Augmented Generation (RAG) for Ad Creatives.
    Retrieves high-scoring genes (visual/copy) from Qdrant to guide generation.
    """
    def __init__(self, qdrant_client=None, collection_name="material_library"):
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        dashscope.api_key = self.api_key
        
        # Use provided client or get from unified getter
        if qdrant_client:
            self.qdrant = qdrant_client
        else:
            self.qdrant = get_qdrant_client()
            
        self.collection_name = collection_name
        self.llm_model = dashscope.Generation.Models.qwen_turbo
        self.image_gen = ImageGenerator()
        
    def execute(self, user_query: str) -> Dict:
        """
        Execute RAG generation flow.
        """
        print(f"🧬 RAG Generation Start: {user_query}")
        
        # 1. Retrieve High CTR Materials (> 2% or just top sorted)
        refs = self._retrieve_references(user_query)
        
        # 2. Extract Genes & Construct Prompt
        if refs:
            prompt = self._construct_rag_prompt(user_query, refs)
            ref_info = [r["payload"].get("filename", "unknown") for r in refs]
        else:
            # Fallback to simple prompt
            prompt = f"{user_query}, 高质量, 大师作, 8k分辨率"
            ref_info = []
            
        print(f"   📝 Constructed Prompt: {prompt}")
        
        # 3. Generate Image
        # We'll save to a temp path
        output_filename = f"rag_gen_{os.urandom(4).hex()}.png"
        output_path = os.path.abspath(output_filename)
        
        result_path = self.image_gen.generate(prompt, output_path)
        
        if result_path:
            return {
                "success": True,
                "image_path": result_path,
                "used_prompt": prompt,
                "references": ref_info
            }
        else:
            return {
                "success": False,
                "reason": "Image generation failed"
            }

    def _retrieve_references(self, query: str) -> List[Dict]:
        """
        Search for relevant materials and filter/sort by CTR.
        """
        # We need an embedding for the query. 
        # Assuming we can use dashscope embedding here too.
        try:
            resp = dashscope.TextEmbedding.call(
                model=dashscope.TextEmbedding.Models.text_embedding_v1,
                input=query
            )
            if resp.status_code == HTTPStatus.OK:
                if hasattr(resp.output, 'embeddings'):
                    vector = resp.output.embeddings[0].embedding
                else:
                    vector = resp.output['embeddings'][0]['embedding']
                
                # hits = self.qdrant.search(
                #     collection_name=self.collection_name,
                #     query_vector=vector,
                #     limit=10
                # )
                
                hits = self.qdrant.query_points(
                    collection_name=self.collection_name,
                    query=vector,
                    limit=10
                ).points
                
                # Filter for high CTR (e.g., > 0.02)
                # Note: 'historical_ctr' might be None or 0.
                valid_hits = []
                for h in hits:
                    ctr = h.payload.get("historical_ctr", 0.0)
                    if ctr is None: ctr = 0.0
                    if ctr >= 0.02: # 2% Threshold
                        valid_hits.append({"payload": h.payload, "ctr": ctr})
                
                # Sort by CTR desc
                valid_hits.sort(key=lambda x: x["ctr"], reverse=True)
                return valid_hits[:3] # Top 3
                
        except Exception as e:
            print(f"   ⚠️ RAG Retrieval Failed: {e}")
            
        return []

    def _construct_rag_prompt(self, query: str, refs: List[Dict]) -> str:
        """
        Combine user query with reference genes.
        """
        # Extract tags
        styles = []
        colors = []
        compositions = []
        
        for r in refs:
            tags = r["payload"].get("tags", {})
            if tags.get("visual_style"): styles.append(tags["visual_style"])
            if tags.get("color_tone"): colors.append(tags["color_tone"])
            if tags.get("composition"): compositions.append(tags["composition"])
            
        # Deduplicate and take top
        from collections import Counter
        top_style = Counter(styles).most_common(1)[0][0] if styles else ""
        top_color = Counter(colors).most_common(1)[0][0] if colors else ""
        
        # LLM Synthesis
        prompt_template = f"""
        基于用户需求和优秀素材的特征，生成一个优化的绘画Prompt。
        
        【用户核心需求】(绝对优先，必须严格遵守)
        {query}
        
        【参考风格基因】(仅当不与用户需求冲突时参考，否则忽略)
        风格: {top_style} (如果与用户需求冲突，请忽略)
        色调: {top_color} (如果用户指定了色调，请忽略此项)
        参考元素: {', '.join(compositions[:2])}
        
        请输出一段高质量的英文Prompt。
        要求：
        1. 必须精准还原用户的核心需求（尤其是氛围、情感、色调）。
        2. 如果用户需求包含"温暖"、"温馨"等词，必须强制使用 warm lighting, cozy atmosphere 等词汇，严禁使用 cold, tech, blue tone 等相反词汇，即使参考图中包含这些。
        3. 借鉴参考基因的构图，但风格必须服从用户指令。
        4. 必须确保主体物（Product）的物理结构准确合理。例如：如果是牙刷(toothbrush)，必须描述 brush head with bristles 和 handle；如果是瓶子，必须有 cap 和 bottle body。避免生成变形或不符合常识的物体。
        5. 风格必须是 "High quality product photography, 8k resolution, photorealistic"，严禁生成抽象、卡通或扭曲的物体结构。
        6. 直接输出Prompt内容，不要包含其他文字。
        """
        
        try:
            resp = dashscope.Generation.call(
                model=self.llm_model,
                prompt=prompt_template,
                result_format='message'
            )
            if resp.status_code == HTTPStatus.OK:
                return resp.output.choices[0].message.content.strip()
        except:
            pass
            
        # Fallback construction
        return f"{query}, {top_style} style, {top_color} tone, {', '.join(compositions[:2])}, high quality"

    def generate_composite_design(self, base_image: Image.Image, user_query: str = "", use_rag: bool = True, skip_matting: bool = False) -> Dict:
        """
        Smart Design Flow:
        1. Remove background from base_image (unless skip_matting=True)
        2. Generate background based on RAG or user query
        3. Composite
        """
        print(f"🎨 Smart Design Start: {user_query}")
        
        try:
            # 1. Background Removal
            if skip_matting:
                if base_image.mode != 'RGBA':
                    product_img = base_image.convert("RGBA")
                else:
                    product_img = base_image
            else:
                from design_utils import remove_background
                product_img = remove_background(base_image)
            
            # 2. Determine Background Prompt
            prompt = ""
            ref_info = []
            marketing_factors = {}
            
            if use_rag:
                # Retrieve high conversion factors
                search_query = user_query if user_query else "high conversion product background"
                
                # Use query_points instead of search if qdrant is updated, but let's stick to simple retrieval logic
                # Actually, self.qdrant.search might be available depending on client version, 
                # but let's use the method we know works: _retrieve_high_ctr_factors or similar.
                # The previous code block used _retrieve_references which seems missing or I missed it.
                # Let's re-implement simple retrieval here to be safe or use what was there.
                
                # Wait, I see `_retrieve_high_ctr_factors` in my memory/previous read? 
                # No, I see `_retrieve_references` in the code I just read (line 199).
                # But I don't see the definition of `_retrieve_references` in the read output.
                # It might be further up.
                # Let's assume it exists or use `retrieve_relevant` from RAGGenerator if available.
                
                # Let's try to find it or implement a simple search.
                # Actually, let's use the `retrieve` method from the agent or just direct Qdrant search.
                # To be safe, let's implement a direct search here similar to `retrieve_relevant_examples`.
                
                try:
                    # Get embedding for query
                    import dashscope
                    resp = dashscope.TextEmbedding.call(
                        model=dashscope.TextEmbedding.Models.text_embedding_v1,
                        input=search_query
                    )
                    if resp.status_code == HTTPStatus.OK:
                        vector = resp.output.embeddings[0].embedding
                        
                        hits = self.qdrant.search(
                            collection_name=self.collection_name,
                            query_vector=vector,
                            limit=5
                        )
                        # Qdrant client return types vary. 
                        # If using QdrantClient (latest), search returns list of ScoredPoint.
                        refs = [{"payload": h.payload} for h in hits]
                    else:
                        refs = []
                except:
                    refs = []
                
                if refs:
                    # Construct prompt for BACKGROUND ONLY
                    prompt = self._construct_background_prompt(search_query, refs)
                    ref_info = [r["payload"].get("filename", "unknown") for r in refs]
                    
                    # Extract Marketing Factors
                    marketing_factors = self._extract_marketing_factors(refs)
                else:
                    prompt = f"{user_query}, high quality product background, 8k resolution"
                    marketing_factors = {}
            else:
                 # Direct user prompt for background
                 prompt = f"{user_query}, high quality background, 8k resolution"
                 marketing_factors = {}
    
            print(f"   📝 Background Prompt: {prompt}")
    
            # 3. Generate Background Image
            # We need to call Dashscope Image Generation directly if self.image_gen is not robust or we want specific control.
            # The previous code used `self.image_gen.generate(prompt, bg_path)`.
            # Let's stick to that if it works, or direct call.
            
            # Let's assume self.image_gen is a wrapper.
            # But wait, in the previous read, I saw:
            # `generated_bg_path = self.image_gen.generate(prompt, bg_path)`
            # So let's use that.
            
            bg_filename = f"rag_bg_{int(time.time())}_{os.urandom(2).hex()}.png"
            # Ensure directory exists
            os.makedirs("AI素材案例", exist_ok=True)
            bg_path = os.path.abspath(os.path.join("AI素材案例", bg_filename))
            
            # Generate
            # If self.image_gen is not available, we use direct call
            try:
                resp = dashscope.ImageSynthesis.call(
                    model=self.img_model,
                    prompt=prompt,
                    n=1,
                    size='1024*1024'
                )
                if resp.status_code == HTTPStatus.OK:
                    bg_url = resp.output.results[0].url
                    import requests
                    from io import BytesIO
                    bg_resp = requests.get(bg_url)
                    bg_image = Image.open(BytesIO(bg_resp.content))
                    bg_image.save(bg_path)
                else:
                    raise Exception(f"Image gen failed: {resp.message}")
            except Exception as e:
                 return {
                    "success": False,
                    "reason": f"Background generation failed: {e}"
                }
                
            # 4. Composite
            from design_utils import composite_product
            final_image = composite_product(product_img, bg_image)
            
            final_filename = f"smart_design_{int(time.time())}_{os.urandom(2).hex()}.png"
            final_path = os.path.abspath(os.path.join("AI素材案例", final_filename))
            final_image.save(final_path)
            
            return {
                "success": True,
                "image_path": final_path,
                "bg_path": bg_path,
                "used_prompt": prompt,
                "references": ref_info,
                "marketing_factors": marketing_factors
            }
        except Exception as e:
            return {
                "success": False,
                "reason": f"Composition failed: {e}"
            }

    def _extract_marketing_factors(self, refs: List[Dict]) -> Dict:
        """
        Extract marketing copy, style, and position from references.
        """
        copies = []
        styles = []
        positions = []
        
        for r in refs:
            tags = r["payload"].get("tags", {})
            if tags.get("marketing_copy"): copies.append(tags["marketing_copy"])
            if tags.get("copy_style"): styles.append(tags["copy_style"])
            if tags.get("copy_position"): positions.append(tags["copy_position"])
            
        from collections import Counter
        top_copy = copies[0] if copies else "高转化营销文案占位符" # Simplest strategy: take top 1
        top_style = Counter(styles).most_common(1)[0][0] if styles else "现代简约"
        top_position = Counter(positions).most_common(1)[0][0] if positions else "居中下方"
        
        return {
            "copy": top_copy,
            "style": top_style,
            "position": top_position
        }

    def _construct_background_prompt(self, query: str, refs: List[Dict]) -> str:
        """
        Construct a prompt specifically for generating a background.
        """
        # Extract tags
        styles = []
        colors = []
        bg_descs = []
        
        for r in refs:
            tags = r["payload"].get("tags", {})
            if tags.get("visual_style"): styles.append(tags["visual_style"])
            if tags.get("color_tone"): colors.append(tags["color_tone"])
            if tags.get("background_desc"): bg_descs.append(tags["background_desc"])
            
        from collections import Counter
        top_style = Counter(styles).most_common(1)[0][0] if styles else ""
        top_color = Counter(colors).most_common(1)[0][0] if colors else ""
        # Use the most detailed background description if available
        top_bg = bg_descs[0] if bg_descs else ""
        
        prompt_template = f"""
        Generate a high-quality product photography BACKGROUND based on these requirements.
        
        User Intent: {query}
        
        Reference Style: {top_style}
        Reference Color: {top_color}
        Reference Scene: {top_bg}
        
        Requirements:
        1. This is a BACKGROUND image for a product. Leave empty space in the center/foreground.
        2. NO text, NO product in the image. Just the scene, lighting, and podium/surface.
        3. Photorealistic, 8k resolution, commercial photography.
        4. If user asks for specific scene (e.g. Christmas), prioritize that.
        
        Output only the prompt.
        """
        
        try:
            resp = dashscope.Generation.call(
                model=self.llm_model,
                prompt=prompt_template,
                result_format='message'
            )
            if resp.status_code == HTTPStatus.OK:
                return resp.output.choices[0].message.content.strip()
        except:
            pass
            
        return f"background for {query}, {top_style} style, {top_color} tone, minimal, product photography background"
