import os
import json
import uuid
from typing import Dict, List, Any
from huggingface_hub import InferenceClient
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny
from dotenv import load_dotenv

load_dotenv()

import dashscope
from http import HTTPStatus
from analytics import aggregate_ctr_by_tag, derive_supply_refs
from fission import FissionAgent
from rag_generator import RAGGenerator
from db_client import get_qdrant_client
from dictionaries import COLORS, STYLES, PURPOSES, AUDIENCES, ABSTRACT_KEYWORDS, COMPLEX_KEYWORDS, SCENES

class MaterialAgent:
    """
    Intelligent Agent that routes user queries to the best retrieval strategy.
    Strategies:
    1. L1 (Attribute): Direct filter (Rule-based).
    2. L2 (Semantic): Vector search (Threshold 0.4).
    3. L3 (Complex): LLM Reasoning + Search.
    """
    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        dashscope.api_key = self.api_key
        
        self.qdrant = get_qdrant_client()
        self.collection_name = "material_library"
        self.llm_model = dashscope.Generation.Models.qwen_turbo
        self.embed_model = dashscope.TextEmbedding.Models.text_embedding_v1
        
        # Initialize Sub-Agents
        self.fission_agent = FissionAgent()
        # Pass self.qdrant to share the instance
        self.rag_generator = RAGGenerator(self.qdrant, self.collection_name)

    def get_materials_by_user(self, user_id: str, limit: int = 20, offset: Any = None) -> tuple[List[Dict], Any]:
        """Fetch recent materials for a specific user with pagination."""
        try:
            # Create filter
            scroll_filter = Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            )
            
            points, next_offset = self.qdrant.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=limit,
                offset=offset,
                with_payload=True,
                with_vectors=False
            )
            return [{"id": p.id, "payload": p.payload} for p in points], next_offset
        except Exception as e:
            print(f"❌ Error fetching user materials: {e}")
            return [], None

    def get_user_material_count(self, user_id: str) -> int:
        """Get total count of materials for a specific user."""
        try:
            count_filter = Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            )
            result = self.qdrant.count(
                collection_name=self.collection_name,
                count_filter=count_filter
            )
            return result.count
        except Exception as e:
            print(f"❌ Error counting user materials: {e}")
            return 0

    def _analyze_intent_rule_based(self, query: str) -> Dict:
        """
        Rule-based intent classification.
        Returns: {
            "level": "L1" | "L2" | "L3" | "L1_L2_HYBRID",
            "filters": {key: value}, # For L1
            "search_query": str
        }
        """
        query_lower = query.lower()
        filters = {}
        
        # 1. Define Rules (Imported from dictionaries.py)
        # COLORS, STYLES, PURPOSES, AUDIENCES are imported
        
        # 2. Extract Attributes (L1)
        found_l1 = False
        
        # Extract Color (Maps to 'color_tone')
        for c in COLORS:
            if c in query_lower:
                filters["color_tone"] = c 
                found_l1 = True
                break # Assume single color for now
        
        # Extract Purpose (Maps to 'material_purpose')
        for p in PURPOSES:
            if p in query_lower:
                filters["material_purpose"] = p 
                found_l1 = True
                break
        
        # Extract Style (Maps to 'visual_style')
        for s in STYLES:
            if s in query_lower:
                filters["visual_style"] = s
                found_l1 = True
                break

        # Extract Audience (Maps to 'target_audience')
        for a in AUDIENCES:
            if a in query_lower:
                filters["target_audience"] = a
                found_l1 = True
                break
        
        # Check Scene (L3 Trigger)
        found_scene = False
        for s in SCENES:
            if s in query_lower:
                found_scene = True
                break
                
        # 3. Check L3
        # L3 Conditions:
        # 1. Long query (> 20 chars)
        # 2. Contains complex keywords (适合, 针对...)
        # 3. Contains a Scene + (L1 Entity OR Abstract Style) -> Implies "Find X for Y"
        is_long = len(query) > 20
        has_complex = any(w in query_lower for w in COMPLEX_KEYWORDS)
        has_abstract = any(w in query_lower for w in ABSTRACT_KEYWORDS)
        
        is_multi_concept = found_scene and (found_l1 or has_abstract)
        
        if is_long or has_complex or is_multi_concept:
            # If we have filters but determined it's L3, we pass filters to L3 (optional)
            # But L3 typically rewrites query. 
            # We will just return L3 and let LLM handle the constraints.
            return {"level": "L3", "filters": filters, "search_query": query}
            
        # 4. Check L2
        is_short = len(query) < 15
        
        # 5. Determine Level
        if found_l1 and has_abstract:
            return {"level": "L1_L2_HYBRID", "filters": filters, "search_query": query}
        elif found_l1:
            return {"level": "L1", "filters": filters, "search_query": query}
        elif has_abstract and is_short:
             return {"level": "L2", "filters": {}, "search_query": query}
        else:
            # Default to L2 if unsure
            return {"level": "L2", "filters": {}, "search_query": query}

    def search(self, user_query: str, user_id: str = None) -> tuple[List[Dict], Dict]:
        intent = self._analyze_intent_rule_based(user_query)
        level = intent["level"]
        
        print(f"   🔍 Strategy: {level} | Filters: {intent.get('filters')}")
        
        results = []
        try:
            if level == "L1":
                results = self._search_l1_filter(intent["filters"], user_id=user_id)
            elif level == "L2":
                results = self._search_l2_vector(intent["search_query"], user_id=user_id)
            elif level == "L3":
                results = self._search_l3_complex(user_query, user_id=user_id)
                # Fallback
                if not results:
                     print("   ⚠️ L3 Empty -> Fallback to L2")
                     results = self._search_l2_vector(user_query, user_id=user_id)
                     
            elif level == "L1_L2_HYBRID":
                # Filter then Search (Cascade)
                # Qdrant supports filtering during search
                results = self._search_hybrid(user_query, intent["filters"], user_id)
        
        except Exception as e:
            print(f"   ❌ Search Error: {e}")
            results = []

        # API Error Fallback: If results empty and we suspect API failure, try Local Keyword Search
        if not results:
             # Check if we should try local keyword search
             # (If L2/L3 failed, it's likely embedding/LLM failure)
             print("   ⚠️ Strategies failed. Attempting Local Keyword Search Fallback...")
             results = self._search_fallback_keyword(user_query, user_id)
             if results:
                intent["level"] = f"{level} (API Error -> Local Fallback)"
                intent["error"] = "DashScope API Failure"

        # Global Fallback
        if not results:
            print("   ⚠️ All strategies failed -> Return fallback high-score items")
            results = self._get_fallback_items()

        # Rerank
        results = self._rerank_by_performance(results)
            
        return results, intent

    def _search_fallback_keyword(self, query: str, user_id: str = None) -> List[Dict]:
        """
        Fallback when Vector Search fails (e.g. API Error).
        Scans recent items and performs fuzzy text matching on Python side.
        """
        try:
            # 1. Fetch recent items (limit 100)
            # Create filter if user_id is present
            scroll_filter = None
            if user_id:
                scroll_filter = Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                )

            points, _ = self.qdrant.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=100,
                with_payload=True,
                with_vectors=False
            )
            
            hits = []
            query_lower = query.lower()
            
            for p in points:
                payload = p.payload
                # Double check user_id just in case
                if user_id and payload.get("user_id") != user_id:
                    continue

                score = 0
                
                # Check Filename
                filename = payload.get("filename", "").lower()
                if query_lower in filename:
                    score += 0.8
                    
                # Check Synonym Map for "Watch"
                if "手表" in query_lower and "watch" in filename:
                    score += 0.9
                    
                # Check Synonym Map for "Watch" (Filename fallback)
                if query_lower == "手表" and ("watch" in filename or "smart" in filename):
                    score += 0.9
                    
                # Check OCR Text
                ocr = payload.get("ocr_text", "").lower()
                if query_lower in ocr:
                    score += 0.5
                    
                # Check Tags
                tags = payload.get("tags", {})
                if isinstance(tags, str):
                    try: 
                        import json
                        tags = json.loads(tags)
                    except: 
                        tags = {}
                        
                # Search values in tags
                for v in tags.values():
                    if isinstance(v, str) and query_lower in v.lower():
                        score += 0.6
                    elif isinstance(v, list):
                        for item in v:
                             if isinstance(item, str) and query_lower in item.lower():
                                 score += 0.6
                                 
                if score > 0:
                    # Normalize score to resemble cosine similarity (0-1)
                    final_score = min(score, 0.99)
                    hits.append({"score": final_score, "payload": payload, "id": p.id})
            
            print(f"   🧹 Local Keyword Search found {len(hits)} matches.")
            # Sort by match score
            hits.sort(key=lambda x: x["score"], reverse=True)
            return hits[:10]
            
        except Exception as e:
            print(f"   ❌ Local Keyword Search Error: {e}")
            return []

    def _search_l1_filter(self, filters: Dict, user_id: str = None) -> List[Dict]:
        """L1: Pure Meta Filter"""
        print(f"   🔍 Executing L1 Filter: {filters}")
        
        must_conditions = []
        if user_id:
            must_conditions.append(
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            )

        for key, val in filters.items():
            # For 'color_tone', it's an array in DB (from new pipeline).
            # MatchAny works if field is list and we provide list, or MatchValue if we check containment.
            # Qdrant "match" checks equality. For list fields, it checks if value is IN list.
            # But we need to handle nested payload fields: "tags.color_tone"
            
            # Note: Payload structure is tags: { ... }
            path = f"tags.{key}"
            
            # Since 'color_tone' is list in JSON, Qdrant treats it as list.
            # MatchValue: "val" in [list] -> True.
            
            # Partial match is tricky in Qdrant filters (needs Full Text).
            # Let's assume Exact Match for now or use 'MatchValue'.
            # If user searches "红色", and tag is "红色", MatchValue works.
            # If user searches "海报", and tag is "活动Banner" (contains 'Banner'?), 
            # we might need to be careful. 
            # For simplicity, we use MatchValue.
            
            must_conditions.append(
                FieldCondition(key=path, match=MatchValue(value=val))
            )
            
        if not must_conditions:
            return []

        points, _ = self.qdrant.scroll(
            collection_name=self.collection_name,
            scroll_filter=Filter(must=must_conditions),
            limit=20,
            with_payload=True,
            with_vectors=False
        )
        
        return [{"score": 1.0, "payload": p.payload, "id": p.id} for p in points]

    def update_material_ctr(self, filename: str, ctr: float) -> bool:
        """Updates the 'real_ctr' field in the material's payload."""
        try:
            # Generate deterministic ID from filename
            image_id = str(uuid.uuid5(uuid.NAMESPACE_DNS, filename))
            
            # 1. Check if exists (Optional, set_payload might fail or create ghost if not careful, 
            # but Qdrant usually requires point to exist)
            
            # 2. Update payload
            # Qdrant set_payload overwrites specified keys, keeps others
            self.qdrant.set_payload(
                collection_name=self.collection_name,
                payload={"real_ctr": ctr},
                points=[image_id]
            )
            print(f"✅ Updated CTR for {filename}: {ctr}")
            return True
            
        except Exception as e:
            print(f"❌ Error updating CTR for {filename}: {e}")
            return False

    def _search_l2_vector(self, query: str, threshold: float = 0.42, user_id: str = None) -> List[Dict]:
        """L2: Vector Search with Threshold (Stricter to avoid bad matches)"""
        print(f"   🔍 Executing L2 Vector: '{query}' (Threshold: {threshold})")
        
        vector = self._get_embedding(query)
        if not vector: return []
        
        # Build filter
        query_filter = None
        if user_id:
            query_filter = Filter(
                must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
            )

        # Use query_points instead of search (deprecated/missing in newer client)
        try:
            hits = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=vector,
                query_filter=query_filter,
                limit=10,
                score_threshold=threshold
            ).points
        except Exception as e:
            print(f"   ⚠️ Vector Search Error: {e}")
            return []
        
        return [{"score": h.score, "payload": h.payload, "id": h.id} for h in hits]

    def _search_l3_complex(self, query: str, user_id: str = None) -> List[Dict]:
        """
        L3: The Reasoner (Complex Logic)
        Pipeline: 
        1. Query Rewrite (LLM): User Query -> Structured Constraints
        2. Query Expansion (LLM): Constraints -> Expanded Keywords
        3. Vector Search (L2): Expanded Keywords -> Candidates
        4. Rerank (LLM): Candidates + Original Query -> Final Top 5
        """
        print(f"   🧠 Executing L3 Reasoning Pipeline...")
        
        # Step 1 & 2: Rewrite and Expand
        expanded_query = query # Default
        try:
            prompt = f"""
            你是一个搜索专家。请分析用户的复杂需求，拆解出结构化条件，并扩展出更易检索的视觉关键词。
            
            用户需求: "{query}"
            
            请输出 JSON:
            {{
                "scene": "使用场景(如母亲节)",
                "audience": "目标人群(如白领)",
                "style": "视觉风格(如温情)",
                "expanded_keywords": "用于向量检索的扩展关键词组合(如: 康乃馨 拥抱 家 暖色调)"
            }}
            """
            resp = dashscope.Generation.call(
                model=self.llm_model,
                prompt=prompt,
                result_format='message'
            )
            if resp.status_code == HTTPStatus.OK:
                content = resp.output.choices[0].message.content
                import json
                start = content.find('{')
                end = content.rfind('}') + 1
                data = json.loads(content[start:end])
                expanded_query = data.get("expanded_keywords", query)
                print(f"   -> Rewrite: {data}")
                print(f"   -> Expanded Query: {expanded_query}")
        except Exception as e:
            print(f"   ⚠️ L3 Rewrite Failed: {e}")
            
        # Step 3: Vector Search (High Recall)
        # We fetch more candidates (e.g., 20) to give LLM enough room to rerank
        vector_results = self._search_l2_vector(expanded_query, threshold=0.15, user_id=user_id) 
        if not vector_results:
            return []
            
        # Step 4: LLM Rerank
        return self._rerank_by_llm(query, vector_results[:20])

    def _rerank_by_llm(self, user_query: str, candidates: List[Dict]) -> List[Dict]:
        """
        L3 Step 4: Rerank candidates based on logical relevance to user query.
        """
        print(f"   ⚖️ LLM Reranking {len(candidates)} candidates...")
        
        if len(candidates) == 0:
            return []
            
        # Prepare candidates for LLM
        candidate_descriptions = []
        for idx, item in enumerate(candidates):
            p = item['payload']
            # Use 'visual_desc' if available, otherwise construct from tags
            desc = f"ID:{idx} | Filename:{p.get('filename')} | Tags:{p.get('tags')}"
            candidate_descriptions.append(desc)
            
        prompt = f"""
        用户原始需求: "{user_query}"
        
        候选素材列表:
        {chr(10).join(candidate_descriptions)}
        
        任务:
        请根据用户需求，从候选列表中选出最符合逻辑的 Top 5 素材。
        请考虑场景匹配度、人群匹配度和风格匹配度。
        
        输出 JSON 列表 (按推荐顺序排列):
        [
            {{"id": 候选ID, "reason": "推荐理由"}},
            ...
        ]
        """
        
        try:
            resp = dashscope.Generation.call(
                model=self.llm_model,
                prompt=prompt,
                result_format='message'
            )
            if resp.status_code == HTTPStatus.OK:
                content = resp.output.choices[0].message.content
                import json
                start = content.find('[')
                end = content.rfind(']') + 1
                ranking = json.loads(content[start:end])
                
                reranked_results = []
                for rank_item in ranking:
                    original_idx = rank_item.get("id")
                    if 0 <= original_idx < len(candidates):
                        item = candidates[original_idx]
                        # Boost score to reflect LLM confidence
                        # We give a synthetic high score (0.9 descending)
                        item['score'] = 0.99 - (len(reranked_results) * 0.01)
                        item['rerank_reason'] = rank_item.get("reason")
                        reranked_results.append(item)
                        
                print(f"   ✅ LLM Selected {len(reranked_results)} items")
                return reranked_results
                
        except Exception as e:
            print(f"   ⚠️ L3 Rerank Failed: {e}")
            
        # Fallback: Return original top results
        return candidates[:5]

    def _search_hybrid(self, query: str, filters: Dict, user_id: str = None) -> List[Dict]:
        """Hybrid: Vector Search + Filter"""
        print(f"   🔍 Executing Hybrid: '{query}' + {filters}")
        
        vector = self._get_embedding(query)
        if not vector: return []
        
        must_conditions = []
        if user_id:
            must_conditions.append(
                FieldCondition(key="user_id", match=MatchValue(value=user_id))
            )

        for key, val in filters.items():
            path = f"tags.{key}"
            must_conditions.append(
                FieldCondition(key=path, match=MatchValue(value=val))
            )
            
        try:
            hits = self.qdrant.query_points(
                collection_name=self.collection_name,
                query=vector,
                query_filter=Filter(must=must_conditions),
                limit=10,
                score_threshold=0.35
            ).points
        except Exception as e:
            print(f"   ⚠️ Hybrid Search Error: {e}")
            return []
            
        return [{"score": h.score, "payload": h.payload, "id": h.id} for h in hits]

    def _get_embedding(self, text: str) -> List[float]:
        print(f"   🧬 Generating embedding for: '{text[:20]}...'")
        try:
            # Add timeout mechanism if possible, or rely on DashScope's internal timeout?
            # DashScope currently doesn't expose timeout in .call() easily, 
            # but we can wrap it or just hope it works. 
            # We will use a timer to debug.
            import time
            start_t = time.time()
            
            resp = dashscope.TextEmbedding.call(
                model=self.embed_model,
                input=text
            )
            
            duration = time.time() - start_t
            print(f"   ✅ Embedding generated in {duration:.2f}s")
            
            if resp.status_code == HTTPStatus.OK:
                if hasattr(resp.output, 'embeddings'):
                    return resp.output.embeddings[0].embedding
                else:
                    return resp.output['embeddings'][0]['embedding']
            else:
                print(f"   ❌ Embedding API Error: {resp.message}")
        except Exception as e:
            print(f"   ❌ Embedding Exception: {e}")
        return []

    def _get_fallback_items(self) -> List[Dict]:
        """Return recent high-score items"""
        try:
            # We can't sort by payload field easily without payload index, 
            # but we can scroll and sort in memory.
            points, _ = self.qdrant.scroll(
                collection_name=self.collection_name,
                limit=50,
                with_payload=True,
                with_vectors=False
            )
            # Fix: Set score to 0.05 to indicate these are NOT semantic matches
            items = [{"score": 0.05, "payload": p.payload, "id": p.id, "is_fallback": True} for p in points]
            # Sort by predicted_score descending
            items.sort(key=lambda x: x["payload"].get("predicted_score", 0), reverse=True)
            return items[:5]
        except:
            return []

    def list_materials(self, limit: int = 100, user_id: str = None) -> List[Dict]:
        """List materials for the Table View"""
        try:
            scroll_filter = None
            if user_id:
                scroll_filter = Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                )
            
            points, _ = self.qdrant.scroll(
                collection_name=self.collection_name,
                limit=limit,
                scroll_filter=scroll_filter,
                with_payload=True,
                with_vectors=False
            )
            return [{"id": p.id, "payload": p.payload} for p in points]
        except:
            return []


    def evaluate_supply(self, query: str, user_id: str = None) -> Dict:
        """
        Smart Supply Decision Matrix: Similarity x Predicted Score (0-30)
        """
        print(f"🧐 Evaluating supply for: {query} (User: {user_id})")
        
        # 1. Recall (Hybrid/L2)
        results, intent = self.search(query, user_id=user_id)
        top_candidates = results[:5]
        
        # 2. Decision Logic
        # Thresholds (Lowered for better recall)
        SIM_HIGH, SIM_MID = 0.65, 0.30 
        SCORE_HIGH, SCORE_MID = 24, 18 # 80%, 60% of 30
        
        action = "GENERATE"
        reason = "库存不足或表现不佳"
        suggestion = ""
        ref_material = None
        
        # Check top candidate
        if top_candidates:
            best = top_candidates[0]
            sim = best["score"]
            payload = best["payload"]
            
            # Get Score (Prefer predicted_score, fallback to historical_ctr mapped to 30)
            score = payload.get("predicted_score", 0)
            if score == 0:
                ctr = payload.get("historical_ctr", 0.0)
                if ctr > 0:
                    score = min(ctr * 600, 30)
                else:
                    # No score data available. 
                    # If similarity is high, we assume it's a valid material (e.g. newly uploaded).
                    # Assign a provisional 'safe' score to allow REUSE/FISSION logic to trigger.
                    score = 20.0 
            
            print(f"   🏆 Best Candidate: Sim={sim:.2f}, Score={score:.1f}/30")
            
            if sim > SIM_HIGH:
                # If L3 reasoning was used, we should be more conservative about "Reuse"
                # because complex queries often imply specific needs not fully met by stock images.
                is_l3 = intent.get("level") == "L3"
                
                if score >= SCORE_HIGH and not is_l3:
                    action = "REUSE"
                    reason = f"找到高度相似素材(相似度{sim:.2f})，且预测分优秀({score:.1f}/30)"
                    suggestion = "建议直接复用，无需重新设计"
                    ref_material = best
                elif score >= SCORE_HIGH and is_l3:
                    # L3 Case: High score but complex intent
                    action = "FISSION"
                    reason = f"找到高相关素材(L3推理)，但因需求复杂，建议基于此微调"
                    suggestion = "建议基于此素材进行微调(Fission)以完全匹配需求"
                    ref_material = best
                elif score >= SCORE_MID:
                    action = "FISSION" # Downgrade to Fission for safety
                    reason = f"相似度高({sim:.2f})，但预测分中等({score:.1f}/30)"
                    suggestion = "建议基于此素材进行微调优化"
                    ref_material = best
                else:
                    action = "GENERATE" # Avoid trap
                    reason = f"虽有相似素材，但预测分极差({score:.1f}/30)，需避坑"
                    suggestion = "参考其失败教训，重新设计"
            elif sim > SIM_MID:
                if score > SCORE_MID:
                    action = "FISSION"
                    reason = f"找到一定相关性素材({sim:.2f})，且有一定潜力({score:.1f}/30)"
                    suggestion = "建议参考此素材进行裂变优化"
                    ref_material = best
                else:
                    action = "GENERATE"
                    reason = f"相关素材表现一般({score:.1f}/30)"
                    suggestion = "重新设计"
            else:
                action = "GENERATE"
        else:
            action = "GENERATE"
            reason = "无相关素材"

        # 3. Construct Output
        decision = {
            "query": query,
            "intent": intent,
            "top_candidates": top_candidates,
            "action": action,
            "reason": reason,
            "suggestion": suggestion,
            "strategy": {}
        }
        
        if action == "REUSE":
            decision["recommended_item"] = ref_material
        elif action == "FISSION":
            decision["reference_material_id"] = ref_material["id"]
            # Generate hints based on tags
            tags = ref_material["payload"].get("tags", {})
            style = tags.get("visual_style", "该风格")
            comp = tags.get("composition", "该构图")
            decision["optimization_hints"] = [
                f"保持其{style}风格",
                f"参考其{comp}",
                "优化利益点文案"
            ]
        elif action == "GENERATE":
            # Generate Design Prompt
            # Get High Score (>24) and Low Score (<15) items
            high_score_items = []
            low_score_items = []
            
            for r in results:
                p = r["payload"]
                s = p.get("predicted_score", 0)
                if s == 0: s = p.get("historical_ctr", 0) * 600
                
                if s > SCORE_HIGH: high_score_items.append(r)
                elif s < 15: low_score_items.append(r)
            
            # Extract common tags (Mocking simple extraction)
            def get_top_tag(items, key):
                from collections import Counter
                tags = []
                for i in items:
                    t = i["payload"].get("tags", {}).get(key)
                    if t: tags.append(t)
                if tags: return Counter(tags).most_common(1)[0][0]
                return "未定"

            rec_style = get_top_tag(high_score_items, "visual_style")
            rec_color = get_top_tag(high_score_items, "color_tone")
            avoid_style = get_top_tag(low_score_items, "visual_style")
            
            decision["design_prompt"] = f"【推荐方向】风格:{rec_style} 色彩:{rec_color} (基于高分素材分析) 【避坑】避免:{avoid_style}"
            decision["reference_materials"] = [i["payload"].get("filename") for i in high_score_items[:3]]
            
            # Populate strategy for UI
            decision["strategy"] = {
                "positive_cues": [t for t in [rec_style, rec_color] if t != "未定"],
                "negative_cues": [t for t in [avoid_style] if t != "未定"],
                "advice": ""
            }

            # Smart Advice Construction
            advice_parts = []
            if rec_style != "未定":
                advice_parts.append(f"建议采用{rec_style}风格")
            if rec_color != "未定":
                advice_parts.append(f"搭配{rec_color}色调")
            
            if advice_parts:
                advice_str = "，".join(advice_parts) + "。"
            else:
                advice_str = "暂无高分参考，建议根据品牌调性自由探索。"

            if avoid_style != "未定":
                advice_str += f" 另外请避免使用{avoid_style}风格（历史表现不佳）。"
            
            decision["strategy"]["advice"] = f"基于库存分析，{advice_str}"

        return decision

    # ... (Keep existing analyze_assets and _rerank_by_performance) ...
    def analyze_assets(self, user_id: str = None) -> Dict:
        try:
            scroll_filter = None
            if user_id:
                scroll_filter = Filter(
                    must=[FieldCondition(key="user_id", match=MatchValue(value=user_id))]
                )
                
            points, _ = self.qdrant.scroll(
                collection_name=self.collection_name,
                scroll_filter=scroll_filter,
                limit=200,
                with_payload=True,
                with_vectors=False
            )
        except Exception as e:
            return {"error": str(e)}
        if not points:
            return {"error": "No data found"}
        tag_perf = aggregate_ctr_by_tag(points, field="visual_style")
        import pandas as pd
        df = pd.DataFrame({"ctr": [p.payload.get("historical_ctr", 0.0) for p in points]})
        global_avg_ctr = df["ctr"].mean() if not df.empty else 0.0
        insights = []
        if not tag_perf.empty and global_avg_ctr > 0:
            best = tag_perf.iloc[0]
            worst = tag_perf.iloc[-1]
            if best["avg_ctr"] > global_avg_ctr * 1.2:
                lift = (best["avg_ctr"] - global_avg_ctr) / global_avg_ctr
                insights.append(f"💡 策略发现：[自动标签: {best['tag']}] 的素材点击率比平均水平高 {lift:.0%} ({best['avg_ctr']:.2%} vs {global_avg_ctr:.2%})。")
            if worst["avg_ctr"] < global_avg_ctr * 0.8:
                drop = (global_avg_ctr - worst["avg_ctr"]) / global_avg_ctr
                insights.append(f"⚠️ 风险提示：[自动标签: {worst['tag']}] 表现不佳，低于平均水平 {drop:.0%}。")
        return {
            "sample_count": len(points),
            "tag_performance": tag_perf,
            "global_avg_ctr": global_avg_ctr,
            "insights": insights
        }

    def _rerank_by_performance(self, results: List[Dict]) -> List[Dict]:
        """
        Reranks results based on a fusion of Semantic Score and Predicted CTR.
        Formula: Final_Score = (Semantic_Score * 0.7) + (Normalized_CTR * 0.3)
        """
        if not results:
            return []
            
        reranked = []
        for item in results:
            payload = item['payload']
            # Parse tags if string
            tags = payload.get('tags', {})
            if isinstance(tags, str):
                try: 
                    import json
                    tags = json.loads(tags)
                except: 
                    tags = {}
            
            # Get CTR
            ctr = payload.get('historical_ctr', 0.0) or tags.get('predicted_ctr', 0.0)
            
            # Normalize CTR (Assume max expected CTR is 5.0%)
            normalized_ctr = min(ctr / 0.05, 1.0) # 0.05 = 5%
            
            # Original Semantic Score (Cosine Similarity, usually 0.0 - 1.0)
            semantic_score = item['score']
            
            # Fusion
            final_score = (semantic_score * 0.7) + (normalized_ctr * 0.3)
            
            item['score'] = final_score
            item['original_score'] = semantic_score # Keep for debug
            reranked.append(item)
            
        # Sort by new final score descending
        reranked.sort(key=lambda x: x['score'], reverse=True)
        return reranked

    def execute_decision(self, decision: Dict) -> Dict:
        """
        Execute the action determined by evaluate_supply.
        """
        action = decision.get("action")
        query = decision.get("query")
        
        print(f"🚀 Executing Action: {action}")
        
        if action == "FISSION":
            ref_id = decision.get("reference_material_id")
            if ref_id:
                # Retrieve material
                points = self.qdrant.retrieve(
                    collection_name=self.collection_name,
                    ids=[ref_id],
                    with_payload=True
                )
                if points:
                    material = points[0].payload
                    # For automated fission, we use the query as instruction.
                    return self.fission_agent.execute(material, query)
            
            return {"success": False, "reason": "Reference material not found or missing ID"}
            
        elif action == "GENERATE":
            return self.rag_generator.execute(query)
            
        elif action == "REUSE":
            return {
                "success": True, 
                "message": "Direct Reuse Recommended", 
                "material": decision.get("recommended_item")
            }
            
        return {"success": False, "reason": f"Unknown or Unsupported Action: {action}"}

# Test block
if __name__ == "__main__":
    agent = MaterialAgent()
    print("\n--- Test 1: Simple Semantic ---")
    results = agent.search("summer vibes")
    for r in results:
        print(f"   Found: {r['payload'].get('filename')} (Score: {r['score']:.2f})")
