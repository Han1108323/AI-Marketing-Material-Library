import os
import json
import dashscope
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from http import HTTPStatus
from dotenv import load_dotenv

load_dotenv()

@dataclass
class PersonaConfig:
    """Configuration for Persona weights"""
    junior_weight: float = 0.2
    senior_weight: float = 0.5
    growth_weight: float = 0.3

    @classmethod
    def default(cls):
        return cls(0.2, 0.5, 0.3)
    
    @classmethod
    def startup(cls):
        # Junior team scenario
        return cls(0.5, 0.3, 0.2)
        
    @classmethod
    def data_driven(cls):
        # ByteDance style
        return cls(0.15, 0.35, 0.5)

@dataclass
class Persona:
    key: str
    name: str
    role: str
    prompt_template: str

class PersonaAgent:
    """
    Simulates different user personas to evaluate marketing materials.
    Now supports multi-dimensional scoring and dynamic weights.
    """
    def __init__(self, config: PersonaConfig = None):
        self.api_key = os.getenv("DASHSCOPE_API_KEY")
        dashscope.api_key = self.api_key
        self.model = dashscope.Generation.Models.qwen_turbo
        self.config = config or PersonaConfig.default()
        
        self.personas = [
            Persona(
                key="junior",
                name="初级运营 (Junior)",
                role="Junior Operator",
                prompt_template="""
你是工作1年的初级运营小王，看图主要看"颜值"。
请对这张素材打分(0-10分):
1. 视觉冲击力(权重50%): 颜色/构图是否抓眼球?
2. 创意度(权重30%): 是否有新鲜感?
3. 信息清晰度(权重20%): 一眼能看懂吗?

素材信息:
- OCR文案: {ocr_text}
- 视觉描述: {vl_caption}
- 标签: {tags}

输出JSON:
{{
  "visual_appeal": <0-10>,
  "creativity": <0-10>,
  "clarity": <0-10>,
  "would_click": <true/false>,
  "reason": "评价理由..."
}}
"""
            ),
            Persona(
                key="senior",
                name="资深运营 (Senior)",
                role="Senior Operator",
                prompt_template="""
你是工作5年的资深运营李姐，看图主要看"转化逻辑"。
请对这张素材打分(0-10分):
1. 信息传达(权重40%): 卖点/优惠是否清晰?
2. 利益点突出(权重30%): 是否有促销信息?
3. 品牌调性(权重30%): 是否符合品牌?

素材信息:
- OCR文案: {ocr_text}
- 视觉描述: {vl_caption}
- 标签: {tags}

输出JSON:
{{
  "info_clarity": <0-10>,
  "benefit_prominence": <0-10>,
  "brand_fit": <0-10>,
  "would_click": <true/false>,
  "reason": "评价理由..."
}}
"""
            ),
            Persona(
                key="growth",
                name="增长PM (Growth)",
                role="Growth Product Manager",
                prompt_template="""
你是增长PM Alex，看图主要看"数据潜力"。
请对这张素材打分(0-10分):
1. 转化潜力(权重50%): 点击/购买可能性?
2. 测试价值(权重30%): 值得做AB测试吗?
3. 可追踪性(权重20%): 能归因到具体要素吗?

素材信息:
- OCR文案: {ocr_text}
- 视觉描述: {vl_caption}
- 标签: {tags}
- 预测CTR: {predicted_ctr}

输出JSON:
{{
  "conversion_potential": <0-10>,
  "test_value": <0-10>,
  "trackability": <0-10>,
  "would_click": <true/false>,
  "reason": "评价理由..."
}}
"""
            )
        ]

    def set_config(self, config_type: str):
        """Allow dynamic configuration switching"""
        if config_type == "startup":
            self.config = PersonaConfig.startup()
        elif config_type == "data_driven":
            self.config = PersonaConfig.data_driven()
        else:
            self.config = PersonaConfig.default()

    def evaluate_material(self, material_info: Dict) -> Dict:
        """
        Runs all personas against the material.
        material_info: {
            "ocr_text": str,
            "caption": str,
            "tags": dict,
            "predicted_ctr": float (optional)
        }
        """
        results = {}
        
        # Prepare context data
        context_data = {
            "ocr_text": material_info.get('ocr_text', '无'),
            "vl_caption": material_info.get('caption', '无'),
            "tags": json.dumps(material_info.get('tags', {}), ensure_ascii=False),
            "predicted_ctr": material_info.get('predicted_ctr', '未知')
        }
        
        # Run Personas
        weighted_score_sum = 0.0
        click_votes = 0
        
        for persona in self.personas:
            try:
                # Format prompt
                prompt = persona.prompt_template.format(**context_data)
                
                # Call LLM
                eval_res = self._run_llm(prompt)
                
                # Calculate internal weighted score for this persona (if detailed scores provided)
                # But here we just take the sub-scores returned by LLM and maybe calculate a total score?
                # The prompt asks for sub-scores. We should calculate a composite score for the persona.
                # Let's simplify: Ask LLM for sub-scores, calculate persona_total_score here.
                
                persona_score = self._calculate_persona_score(persona.key, eval_res)
                eval_res['total_score'] = persona_score
                
                results[persona.key] = eval_res
                
                # Aggregate for Global Score
                weight = getattr(self.config, f"{persona.key}_weight", 0.33)
                weighted_score_sum += persona_score * weight
                
                if eval_res.get('would_click', False):
                    click_votes += 1
                    
            except Exception as e:
                print(f"⚠️ Persona {persona.key} failed: {e}")
                results[persona.key] = {
                    "total_score": 0,
                    "would_click": False,
                    "reason": f"Error: {e}"
                }
                
        # Consistency Analysis
        consistency = self._analyze_consistency(results)
        
        results['summary'] = {
            "weighted_score": round(weighted_score_sum, 2),
            "click_consensus": f"{click_votes}/{len(self.personas)}",
            "consistency_analysis": consistency,
            "final_decision": "RECOMMEND" if weighted_score_sum > 7.0 and click_votes >= 2 else "REJECT"
        }
        
        return results

    def _calculate_persona_score(self, key: str, res: Dict) -> float:
        """Calculate weighted score based on persona specific dimensions"""
        score = 0.0
        try:
            if key == "junior":
                score = (res.get('visual_appeal', 0) * 0.5 + 
                         res.get('creativity', 0) * 0.3 + 
                         res.get('clarity', 0) * 0.2)
            elif key == "senior":
                score = (res.get('info_clarity', 0) * 0.4 + 
                         res.get('benefit_prominence', 0) * 0.3 + 
                         res.get('brand_fit', 0) * 0.3)
            elif key == "growth":
                score = (res.get('conversion_potential', 0) * 0.5 + 
                         res.get('test_value', 0) * 0.3 + 
                         res.get('trackability', 0) * 0.2)
        except:
            score = 5.0 # Fallback
        return round(score, 2)

    def _analyze_consistency(self, results: Dict) -> Dict:
        """Analyze if personas agree"""
        votes = []
        details = []
        
        for p in self.personas:
            res = results.get(p.key, {})
            click = res.get('would_click', False)
            votes.append(click)
            details.append(f"{p.name}: {'✅' if click else '❌'}")
            
        if all(votes):
            status = "高置信度推荐 ✅"
            desc = "所有角色一致看好"
        elif not any(votes):
            status = "一致否决 ❌"
            desc = "所有角色一致不看好"
        else:
            status = "有争议 ⚠️"
            # Who disagreed?
            junior = results.get('junior', {}).get('would_click')
            senior = results.get('senior', {}).get('would_click')
            if junior and not senior:
                desc = "初级运营看好颜值，但资深运营担心转化"
            elif not junior and senior:
                desc = "颜值一般，但转化逻辑清晰"
            else:
                desc = "角色意见分歧，建议AB测试"
                
        return {
            "status": status,
            "description": desc,
            "detail": " | ".join(details)
        }

    def _run_llm(self, prompt: str) -> Dict:
        response = dashscope.Generation.call(
            model=self.model,
            prompt=prompt,
            result_format='message'
        )
        
        if response.status_code == HTTPStatus.OK:
            content = response.output.choices[0].message.content
            # Clean JSON
            try:
                start = content.find('{')
                end = content.rfind('}') + 1
                if start == -1 or end == -1: raise ValueError("No JSON found")
                json_str = content[start:end]
                return json.loads(json_str)
            except Exception as e:
                print(f"JSON Parse Error: {content}")
                raise e
        else:
            raise Exception(f"API Error: {response.message}")

if __name__ == "__main__":
    agent = PersonaAgent()
    mock_data = {
        "ocr_text": "双11大促 全场5折起",
        "caption": "一张红色的海报，中间有大的金色字体写着50% OFF，背景有礼花。",
        "tags": {"visual_style": "促销, 扁平化"},
        "predicted_ctr": 0.025
    }
    print(json.dumps(agent.evaluate_material(mock_data), ensure_ascii=False, indent=2))
