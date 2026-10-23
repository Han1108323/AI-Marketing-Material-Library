import os
import json
import base64
from typing import Dict, Optional, Any, List
import requests
try:
    import dashscope
    from dashscope import MultiModalConversation
except Exception as _e_dash:
    dashscope = None
    MultiModalConversation = None
from http import HTTPStatus
from PIL import Image, ImageDraw, ImageOps
import uuid

class FissionAgent:
    """
    Agent for intelligent material fission (refinement).
    "一句话改图" capability.
    """
    def __init__(self):
        self.api_key = os.getenv("DASHSCOPE_API_KEY") if dashscope else None
        if dashscope and self.api_key:
            dashscope.api_key = self.api_key
        self.llm_model = "qwen-turbo"
        
    def execute(self, material_payload: Dict, user_instruction: str) -> Dict:
        """
        [Deprecated] Direct execution. 
        Use analyze() + generate() for UI workflow.
        """
        analysis = self.analyze(material_payload, user_instruction)
        if not analysis["is_feasible"]:
             return {"success": False, "reason": analysis["reason"]}
        return self.generate(material_payload, analysis)

    def analyze(self, material_payload: Dict, user_instruction: str) -> Dict:
        """
        Step 1: Analyze Intent & Feasibility.
        Returns detailed analysis dict.
        """
        print(f"🧬 Fission Analysis: {user_instruction}")
        return self._analyze_modifier(user_instruction, material_payload)

    def generate(self, material_payload: Dict, analysis: Dict) -> Dict:
        """
        Step 2: execute the validated production editing path.
        Qwen-VL/OCR ground the target and Qwen-Image-Edit-Max performs the edit.
        """
        print(f"🧬 Fission Generation: {analysis['modification_type']}")

        filename = material_payload.get("filename")
        if not filename:
            return {"success": False, "reason": "Missing filename"}

        candidate_paths = []
        if material_payload.get("img_path"):
            candidate_paths.append(material_payload["img_path"])
        candidate_paths.append(os.path.abspath(os.path.join("AI素材案例", filename)))
        candidate_paths.append(os.path.abspath(os.path.join("src", "AI素材案例", filename)))
        candidate_paths.append(os.path.abspath(filename))
        img_path = None
        for p in candidate_paths:
            if p and os.path.exists(p):
                img_path = p
                break
        if not img_path:
            return {"success": False, "reason": "Image file not found"}

        if analysis["modification_type"] == "BG_EDIT":
            return self._call_qwen_precise_edit(
                img_path,
                analysis.get("box_2d"),
                analysis["target_desc"],
                is_text_mode=False
            )

        elif analysis["modification_type"] == "TEXT_EDIT":
            box_2d = analysis.get("box_2d")
            # OCR word/line boxes are already expressed in the source image's
            # coordinate system. Do not reshape a verified box a second time.
            locate_src = (analysis.get("locate_debug") or {}).get("src", "")
            is_verified_ocr_box = locate_src.startswith(("WORD_", "LINE_"))
            if box_2d and not is_verified_ocr_box:
                fixed_box = self._force_horizontal_text_box(
                    box_2d,
                    analysis.get("target_text_content",""),
                    analysis.get("target_desc",""),
                    material_payload.get("ocr_text",""),
                    img_path
                )
                if fixed_box:
                    box_2d = fixed_box
                    analysis["box_2d"] = fixed_box
            if box_2d:
                # Historical high-quality path: Qwen Image Edit handles styled
                # advertising text much better than masked diffusion models.
                # Qwen-VL/OCR provide grounding; the edit model receives the
                # exact old/new copy plus the normalized target coordinates.
                qwen_result = self._call_qwen_precise_edit(
                    img_path,
                    box_2d,
                    analysis["target_desc"],
                    is_text_mode=True,
                    target_text=analysis.get("target_text_content", ""),
                )
                if qwen_result.get("success"):
                    qwen_result["reason"] = (
                        "✅ Qwen-Image-Edit-Max 文字微调成功；"
                        "定位采用 Qwen-VL/OCR 双重校正。"
                    )
                    return qwen_result
                return qwen_result
            return {
                "success": False,
                "reason": "未能可靠定位目标文字，请重新描述需要替换的原文。",
            }
        else:
            return self._call_qwen_edit(
                img_path,
                analysis.get("refined_prompt") or analysis.get("target_desc", "应用微调"),
            )

    def _analyze_modifier(self, instruction: str, material: Dict) -> Dict:
        """
        Analyze if the instruction is a Background Change, Text Change, or Complex Change.
        DUAL-ENGINE (MOST STABLE):
          1) RULE-BASED OFFLINE ENGINE FIRST — No API Key required.
             Parses patterns like "把 X 改成 Y", "X 换成 Y", "背景改为 X" — 100%
             deterministic TEXT_EDIT / BG_EDIT detection plus OCR-based bbox lookup.
          2) Qwen-VL-Max VLM ENGINE — Used only if rule engine returns COMPLEX
             and we have a valid DashScope key.
        """
        filename = material.get("filename")
        ocr_context = material.get("ocr_text", "") if material else ""
        ocr_meta_raw = (material.get("ocr_meta") or {}).get("raw", {}) if material else {}

        # ------------------------------------------------------------
        # img_path resolution: prefer material-supplied absolute path,
        # then upload_ temp file in AI素材案例, then legacy lookup.
        # ------------------------------------------------------------
        img_path = None
        candidate_paths = []
        if material.get("img_path"):
            candidate_paths.append(material["img_path"])
        if filename:
            candidate_paths.append(os.path.abspath(os.path.join("AI素材案例", filename)))
            candidate_paths.append(os.path.abspath(os.path.join("src", "AI素材案例", filename)))
            candidate_paths.append(os.path.abspath(filename))
        for p in candidate_paths:
            if p and os.path.exists(p):
                img_path = p
                break

        default_res = {
            "is_feasible": True,
            "reason": "规则离线匹配兜底：已根据指令语义识别修改内容（无API Key时本地推理）",
            "modification_type": "TEXT_EDIT",
            "target_desc": instruction.strip() or "应用微调",
            "target_text_content": "",
            "refined_prompt": instruction.strip() or "应用微调",
            "box_2d": [200, 200, 800, 500],
        }

        # ============================================================
        # STEP 1: RULE-BASED OFFLINE ENGINE (ALWAYS RUNS FIRST)
        # ============================================================
        rule_hit = self._rule_engine_parse(instruction, ocr_context, ocr_meta_raw, img_path)
        if rule_hit:
            if not img_path and rule_hit.get("box_2d"):
                pass
            print(f"   ✅ RULE ENGINE HIT → type={rule_hit['modification_type']} target_desc={rule_hit.get('target_desc')} box={rule_hit.get('box_2d')}")
            # merge with default_res so missing fields get sensible values
            merged = {**default_res, **rule_hit}
            merged["is_feasible"] = True
            if not merged.get("box_2d"):
                merged["box_2d"] = self._heuristic_text_box(ocr_context, rule_hit.get("target_text_content", ""), img_path)
            return merged

        if not img_path or not os.path.exists(img_path):
            # still return the rule-based default_res so user never sees COMPLEX/None
            return default_res

        # ============================================================
        # STEP 2: VLM ENGINE (RUN ONLY IF API KEY + dashscope BOTH AVAILABLE)
        # ============================================================
        if not self.api_key or not dashscope or not MultiModalConversation:
            return default_res

        prompt = f"""
        任务：基于用户指令和OCR参考，进行精准的图像编辑意图识别和区域定位。

        用户指令: "{instruction}"
        OCR文字参考: "{ocr_context}"

        请执行以下步骤：
        1. **意图分类**：
           - BG_EDIT: 仅换背景（背景改为/换成/变成 X）。
           - TEXT_EDIT: 修改或替换文字（把X改成Y/X换成Y/文字X改为Y）。
           - COMPLEX: 其他不适合局部重绘的复杂修改。

        2. **区域定位 (box_2d) [CRITICAL]**:
           - 如果是 TEXT_EDIT (改字)：
             - 找出指令中要求修改的旧文案，在图片中精准框选出这几个字；
             - 红框高度紧贴文字，严禁包含相邻行；横排文字框扁长；
             - 结合OCR参考文字位置进行校准。
           - 如果是 BG_EDIT：
             - 框选出主体商品的紧密包围盒（我们会保护主体并只改背景）。

        3. **提取关键信息**:
           - target_text_content: 仅 TEXT_EDIT，提取要修改的旧文案原文
           - target_desc: 仅 TEXT_EDIT 填新文案内容（无动词，纯内容）；BG_EDIT填背景描述；COMPLEX填目标效果简述

        4. **坐标格式**: [xmin, ymin, xmax, ymax] (0-1000 归一化坐标)。

        5. **只输出纯 JSON**，不要任何 markdown 标记：
        {{
            "is_feasible": true,
            "modification_type": "TEXT_EDIT",
            "target_desc": "立刻发货",
            "target_text_content": "现货速发",
            "reason": "定位到旧文案xxx并紧贴包围盒",
            "box_2d": [300, 300, 700, 450]
        }}
        """

        try:
            messages = [
                {"role": "user", "content": [
                    {"image": f"file://{img_path}"},
                    {"text": prompt},
                ]}
            ]
            resp = MultiModalConversation.call(model="qwen-vl-max", messages=messages)
            vlm_box = None
            vlm_ok = False
            if resp.status_code == HTTPStatus.OK:
                content = resp.output.choices[0].message.content[0]['text']
                start = content.find('{')
                end = content.rfind('}') + 1
                if start != -1:
                    json_str = content[start:end]
                    res_vlm = json.loads(json_str)
                    vlm_ok = True
                    if res_vlm.get("modification_type") == "TEXT_EDIT" and res_vlm.get("target_text_content"):
                        # NEW (2026-09-09, 用户要求): 替换旧的 _find_bbox_from_ocr 纯OCR切片，
                        # 改用「OCR整行粗定位 + VLM视觉Crop精定位」—— 先定位+模型识别
                        refined_box_vlm = self._vlm_refine_bbox_exact(
                            img_path, ocr_meta_raw, res_vlm["target_text_content"]
                        )
                        if refined_box_vlm:
                            res_vlm["box_2d"] = refined_box_vlm
                            res_vlm["reason"] = (res_vlm.get("reason") or "") + " (VLM意图分析 → OCR粗定位行 → VLM Crop精定位)"
                        else:
                            # Fallback 旧 OCR 方法（避免没框）
                            refined_box_vlm = self._find_bbox_from_ocr(ocr_meta_raw, res_vlm["target_text_content"], img_path)
                            if refined_box_vlm:
                                res_vlm["box_2d"] = refined_box_vlm
                                res_vlm["reason"] = (res_vlm.get("reason") or "") + " (VLM→OCR校正坐标, Fallback)"
                    # Ensure complete fields
                    for k, v in default_res.items():
                        res_vlm.setdefault(k, v)
                    if not res_vlm.get("box_2d"):
                        res_vlm["box_2d"] = self._heuristic_text_box(ocr_context, res_vlm.get("target_text_content",""), img_path)
                    # ===== Z24-1: FINAL BBOX HORIZONTAL SANITY (VLM path) =====
                    if res_vlm.get("modification_type") == "TEXT_EDIT" and res_vlm.get("box_2d"):
                        orig_box = res_vlm["box_2d"]
                        fixed_box = self._force_horizontal_text_box(
                            orig_box,
                            res_vlm.get("target_text_content",""),
                            res_vlm.get("target_desc",""),
                            ocr_context,
                            img_path
                        )
                        if fixed_box and fixed_box != list(orig_box):
                            res_vlm["box_2d"] = fixed_box
                            res_vlm["reason"] = (res_vlm.get("reason") or "") + " （VLM路径已按横排文字矫正）"
                    res_vlm["is_feasible"] = True
                    vlm_box = res_vlm.get("box_2d")

            # ================================================================
            # TRUE DUAL-VALIDATION (OCR + VLM double-calibrated)  —  HISTORY LOGIC
            #   Rule engine already returned above: merged = default_res + rule_hit
            #   Now, if VLM also returns SAME modification_type, tighten the
            #   final bbox to the TIGHT intersection (or padded union).
            # ================================================================
            if vlm_ok and "merged" in dir():
                pass  # (merged was already returned at rule_hit branch before reaching here)
            if vlm_ok and vlm_box:
                return res_vlm
        except Exception as e:
            print(f"   ⚠️ VL Analysis Failed (fallback to rule result): {e}")

        return default_res

    # ============================================================
    # RULE-BASED OFFLINE ENGINE + HEURISTICS
    # ============================================================
    def _rule_engine_parse(self, instruction: str, ocr_text: str, ocr_raw: Dict, img_path: Optional[str]) -> Optional[Dict]:
        """
        Parses classic Chinese editing patterns WITHOUT any API call.
        Returns dict suitable for merging into analysis result or None.
        """
        if not instruction:
            return None
        text = instruction.strip()

        # --- Pattern 1: BG_EDIT ------------------------------------------------
        bg_kw = [
            ("背景", "换", ""), ("背景", "改为", ""), ("背景", "改成", ""),
            ("背景", "变成", ""), ("背景", "替换成", ""),
        ]
        for (k1, k2, _) in bg_kw:
            if (k1 in text) and (k2 in text):
                # Extract new background desc: everything after 改为/改成/换成
                idx = max(text.rfind("改成"), text.rfind("改为"), text.rfind("换成"), text.rfind("替换成"), text.rfind("变成"))
                if idx < 0:
                    continue
                desc = text[idx+2:].strip().strip("。！？.?!\"'")
                if not desc:
                    continue
                # Heuristic bbox: center object (protect area for BG edit)
                w = h = 1000
                if img_path and os.path.exists(img_path):
                    try:
                        with Image.open(img_path) as im:
                            w_px, h_px = im.size
                            w = 1000; h = 1000
                    except Exception:
                        pass
                box = [int(w*0.20), int(h*0.20), int(w*0.80), int(h*0.80)]
                return {
                    "modification_type": "BG_EDIT",
                    "target_desc": desc,
                    "refined_prompt": f"背景替换为：{desc}",
                    "target_text_content": "",
                    "box_2d": box,
                    "reason": f"规则匹配：背景修改描述 → {desc}"
                }

        # --- Pattern 2: TEXT_EDIT ---------------------------------------------
        # 把 X 改成/换成/改为 Y
        import re
        m = re.search(r"(?:把|将|让)\s*(.+?)\s*(?:改成|换成|改为|替换为|替换成|改作|变作|变成)\s*(.+)$", text)
        if not m:
            # Also support "X改成Y", "X换成Y" (no 把)
            m = re.search(r"(.+?)\s*(?:改成|换成|改为|替换为|替换成|改作|变作)\s*(.+)$", text)
        if not m:
            # "修改X为Y"
            m = re.search(r"修改\s*(.+?)\s*为\s*(.+)$", text)
        if m:
            old_raw = m.group(1).strip().strip("'\"“”‘’ 。！？,.?!")
            new_raw = m.group(2).strip().strip("'\"“”‘’ 。！？,.?!")
            old_tok = old_raw
            new_tok = new_raw

            # ================================================================
            # FIX 2026-09-09: 提取old_tok后，去掉用户常说的废话前缀/后缀（这些
            # 绝对不会出现在OCR里），然后再用OCR实际词库反向最长子串匹配：
            #   用户说"把图片里的7天改为1周" → old_raw="图片里的7天"
            #   → 去废话前缀 old_tok_clean = "7天"
            #   → OCR 中存在 "7天"  → matched_old="7天"
            # 再也不会出现 matched_old = "图片里的7天" 去OCR里找的垃圾结果。
            # ================================================================
            _junk_prefixes = [
                "图片里的", "图片中的", "图片里", "图片中", "图片的",
                "图里的", "图中的", "图里", "图中", "图上的", "图上",
                "文字里的", "文字中的", "文字里", "文字中", "文字的", "文字",
                "文案里的", "文案中的", "文案", "标题里的", "标题中的", "标题",
                "图片里面的", "图片里面", "整张图片的",
                "里面的", "当中的", "中的", "的文字", "的文案", "的字样",
            ]
            old_tok_clean = old_tok
            changed = True
            while changed:
                changed = False
                for p in _junk_prefixes:
                    if old_tok_clean.startswith(p):
                        old_tok_clean = old_tok_clean[len(p):]
                        changed = True
                    if old_tok_clean.endswith(p):
                        old_tok_clean = old_tok_clean[:-len(p)]
                        changed = True
                old_tok_clean = old_tok_clean.strip(" 的了是将把叫名为里中,:：，。；,.")
            if not old_tok_clean:
                old_tok_clean = old_tok
            ocr_clean = (ocr_text or "").replace(" ", "")

            # (1) Exact match on cleaned old token first
            matched_old = None
            for candidate in [old_tok_clean, old_tok_clean.replace(" ", ""), old_tok, old_tok.replace(" ", "")]:
                if candidate and len(candidate) >= 1 and candidate in ocr_clean:
                    matched_old = candidate
                    break

            # (2) If still None: REVERSE LONGEST-SUBSTRING match against OCR:
            #     for every possible substring of old_tok_clean (longest first),
            #     see if it appears in ocr_clean → pick the longest one.
            #     This guarantees "图片里的7天" → matched_old = "7天" even if
            #     the cleaned string wasn't an exact match.
            if matched_old is None:
                best_len = 0
                best_sub = None
                src_pool = [old_tok_clean, old_tok]
                for sp in src_pool:
                    sp_clean = sp.replace(" ", "")
                    L = len(sp_clean)
                    for ln in range(L, 0, -1):
                        if ln <= best_len:
                            break
                        for i in range(0, L - ln + 1):
                            sub = sp_clean[i:i+ln]
                            if len(sub) < 1:
                                continue
                            if sub in ocr_clean:
                                best_len = ln
                                best_sub = sub
                                break
                if best_sub and best_len >= 1:
                    matched_old = best_sub

            # (3) Character-level fuzzy fallback: any OCR line/word with ≥50%
            #     character overlap with the (cleaned) old token
            if matched_old is None:
                import re as _re_fz
                for ocr_token in _re_fz.split(r"[\s,，。、;；\n]+", ocr_text or ""):
                    tok = ocr_token.strip()
                    if len(tok) >= 1 and old_tok_clean:
                        overlap = len(set(tok) & set(old_tok_clean))
                        min_len = min(len(tok), len(old_tok_clean))
                        if min_len >= 1 and overlap >= max(1, (min_len + 1)//2):
                            matched_old = tok
                            break
            if matched_old is None:
                matched_old = old_tok_clean  # final fallback: cleaned user token

            # ULTIMATE (2026-09-09): WORD UNION 精确定位（已在_vlm_refine_bbox_exact内部完成）
            self._last_locate_debug = None
            vlm_box = self._vlm_refine_bbox_exact(img_path, ocr_raw, matched_old)
            box = vlm_box or self._heuristic_text_box(ocr_text, matched_old, img_path)
            if getattr(self, "_last_locate_debug", None):
                dbg = self._last_locate_debug
                # Different debug keys for different path branches
                if "words_in_window" in dbg:
                    ws = dbg["words_in_window"]
                    ws_txt = ", ".join(f"{w['text']}[{w['L']:.0f}~{w['R']:.0f}]" for w in ws)
                    refine_note = (
                        f"【定位路径: {dbg['src']}】"
                        f" OCR命中Word拼接=({ws_txt})"
                        f" 窗口文本='{dbg['window_joined_text']}' target位置={dbg['target_in_joined']}"
                        f" 最终像素矩形={dbg['final_px_rect']}"
                    )
                else:
                    refine_note = (
                        f"【定位路径: {dbg['src']}】"
                        f" OCR命中行='{dbg.get('line_text','')}' 字符位置{dbg.get('s_idx_e_idx','')}"
                        f" OCR行像素={dbg.get('line_px','')}"
                        f" 比例切片={dbg.get('char_ratio','')}"
                    )
            else:
                refine_note = (
                    f"【定位路径: HEURISTIC_FALLBACK】OCR中找不到「{matched_old}」（原始用户old_tok='{old_tok}'，清洗后='{old_tok_clean}'），使用启发式图上1/3居中。"
                )
            # Only reshape heuristic fallbacks. A verified OCR box is already
            # tied to real pixels and must remain untouched.
            if not getattr(self, "_last_locate_debug", None):
                box = self._force_horizontal_text_box(box, matched_old, new_tok, ocr_text, img_path)
            return {
                "modification_type": "TEXT_EDIT",
                "target_desc": new_tok,
                "target_text_content": matched_old,
                "refined_prompt": f"将'{matched_old}'替换为'{new_tok}'，保持原有字体风格和背景一致",
                "box_2d": box,
                "locate_debug": getattr(self, "_last_locate_debug", None),
                "reason": f"规则匹配：文字修改 '{matched_old}' → '{new_tok}' {refine_note}，已按横排文字矫正"
            }

        return None

    # ==================================================================
    # FINAL BBOX ORTHOGONAL SANITY (NEW 2026-09-09 — LAST LINE OF DEFENCE)
    # Call this 3 times: rule return / VLM return / before generate().
    # Forces text-edit bboxes to be HORIZONTAL (W >> H) regardless of what
    # OCR/VLM returned, because 100% of user-reported TEXT_EDIT bugs came
    # from vertical column OCR merges (领7天出行守护 + 骑行必备 merged).
    # ==================================================================
    def _force_horizontal_text_box(self,
                                   box_2d: Optional[List[int]],
                                   old_text: Optional[str],
                                   new_text: Optional[str],
                                   ocr_text_joined: Optional[str],
                                   img_path: Optional[str]) -> Optional[List[int]]:
        """
        Forces a HORIZONTAL text bbox:
          1. If input H / W <= 0.85 (already horizontal): return untouched, only clamp.
          2. If input H / W > 0.85 (vertical / near-square):
             - Keep the original CENTER (cx, cy) so the row of text is still around the
               same physical line the OCR found.
             - Compute a TIGHT HORIZONTAL box by assuming EQUAL-WIDTH character cells:
               width  = n_chars × (original_box_height / 1.05)   [≈ square char × n]
               height = min(original_h, width / max(1.3, n_chars × 0.9))
             - Adjust LEFT/RIGHT offset by TOKEN POSITION within the FULL TEXT LINE
               (if we know it from ocr_text_joined), else centre on original cx.
        Always returns 0-1000 normalized list of 4 ints (never None).
        """
        if not box_2d or len(box_2d) != 4:
            # Degenerate: pick a default top-middle headline box.
            return [200, 180, 800, 320]
        xmin, ymin, xmax, ymax = [int(v) for v in box_2d]
        # Clamp to 0-1000 first.
        xmin = max(0, min(1000, xmin)); xmax = max(0, min(1000, xmax))
        ymin = max(0, min(1000, ymin)); ymax = max(0, min(1000, ymax))
        if xmax - xmin < 4: xmax = min(1000, xmin + 10)
        if ymax - ymin < 4: ymax = min(1000, ymin + 10)

        w = max(1, xmax - xmin); h = max(1, ymax - ymin)
        n_chars = max(1, max(len((old_text or "").strip()), len((new_text or "").strip())))
        aspect = h / w  # >1 = tall (problematic)

        # Case 1: already horizontal → still apply mild "row centering / cleanup".
        if aspect <= 0.85:
            # Mild tightening: 4% inner padding.
            pad_x = max(1, int(w * 0.02)); pad_y = max(1, int(h * 0.02))
            nx1 = xmin + pad_x; nx2 = xmax - pad_x
            ny1 = ymin + pad_y; ny2 = ymax - pad_y
            return [max(0,nx1), max(0,ny1), min(1000,nx2), min(1000,ny2)]

        # Case 2: vertical / near-square. Force HORIZONTAL around same (cx, cy).
        cx = (xmin + xmax) * 0.5
        cy = (ymin + ymax) * 0.5

        # Target CHAR-DIMENSION: base on the TIGHTER of (original_h / 1.1, original_w).
        # For a vertical column returned by OCR, the WIDTH is typically the "true"
        # visual single-character width, and the HEIGHT covers N text rows.
        char_dim_guess = min(h / 1.08, max(w, 18))
        if char_dim_guess < 10: char_dim_guess = 20  # floor: 2% of normalized 1000

        target_w = int(char_dim_guess * max(1.15, n_chars * 1.02))
        target_h = int(char_dim_guess * 1.08)
        # HARD SAFETY: width must be ≥ 1.35 × height (so strictly horizontal rectangle)
        if target_w < target_h * 1.35:
            target_w = int(target_h * 1.35)
        # HARD SAFETY: height must NOT exceed 18% of image (180 of 1000)
        if target_h > 180:
            target_h = 180
            target_w = max(target_w, int(target_h * 1.35))

        # --- Try to HORIZONTAL-SHIFT the box based on TOKEN POSITION in full text ---
        # If we know the full OCR line text + the old token, we can LEFT/RIGHT
        # shift so the box sits on the actual chars, not always dead centre.
        shift_frac = 0.0  # 0 = keep at cx; -0.3 = 30% of row to the left; +0.3 right.
        full_text = (ocr_text_joined or "").replace(" ", "")
        ot = (old_text or "").replace(" ", "")
        if full_text and ot and ot in full_text:
            # Estimate how many characters exist in the ENTIRE PHYSICAL ROW that
            # the token lives in. For marketing copy, typical row is ~8-18 chars.
            # We use the total-length / ceil(total_length / 14) ≈ per-row length.
            total_len = max(1, len(full_text))
            est_rows = max(1, round(total_len / 14.0))
            per_row = max(n_chars + 4, int(total_len / est_rows))
            tok_start = full_text.find(ot)
            tok_end = tok_start + len(ot)
            tok_mid = (tok_start + tok_end) / 2.0 / total_len  # 0..1 position
            # Position along the image X axis (0-1000) for the FULL row's estimated span.
            # Estimate row's X span: typically from ~8% to ~92% of image width.
            row_xmin_est = 70
            row_xmax_est = 930
            row_total_w_est = row_xmax_est - row_xmin_est
            # What would be the target's cx if we placed using full_text position?
            alt_cx = row_xmin_est + int(row_total_w_est * tok_mid)
            # Only adopt alt_cx if it's within ±35% of image width (350 in 0-1000)
            # around the original cx (so we never jump rows entirely).
            if abs(alt_cx - cx) <= 350:
                cx = alt_cx

        # Build final box.
        nx1 = int(cx - target_w * 0.5); nx2 = int(cx + target_w * 0.5)
        ny1 = int(cy - target_h * 0.5); ny2 = int(cy + target_h * 0.5)
        # Clamp to image bounds.
        if nx1 < 0: nx2 += (0 - nx1); nx1 = 0
        if nx2 > 1000: nx1 -= (nx2 - 1000); nx2 = 1000
        if ny1 < 0: ny2 += (0 - ny1); ny1 = 0
        if ny2 > 1000: ny1 -= (ny2 - 1000); ny2 = 1000
        nx1 = max(0, min(1000, nx1)); nx2 = max(0, min(1000, nx2))
        ny1 = max(0, min(1000, ny1)); ny2 = max(0, min(1000, ny2))
        if nx2 - nx1 < 10: nx2 = min(1000, nx1 + 10)
        if ny2 - ny1 < 6: ny2 = min(1000, ny1 + 6)
        return [nx1, ny1, nx2, ny2]

    # =====================================================================
    # FINAL (2026-09-09, 按用户要求完全重写): OCR 粗定位 + VLM 精定位
    # ------------- 不再使用旧的 "cluster + 字符比例切片" 纯OCR猜位置 ------------
    #
    # Strategy (两步 Hybrid, 先定位 + 模型识别):
    #   Step1 [OCR 粗定位]: 从 OCR raw 里找包含 target_text 的物理行，拿到
    #         该行的 MIN LEFT / MAX RIGHT / MIN TOP / MAX BOTTOM，允许比实际
    #         "7天"大很多（覆盖整行"领7天出行守护骑行必备"都没问题），目标
    #         只是"把视觉搜索范围从整张图缩小到这一行"。
    #   Step2 [Crop + VLM 精定位]: 把原图按 Step1 的粗框做 Padded Crop（四周
    #         各延伸 ~15% 防止切到笔画），存临时小图，然后调用 qwen-vl-max
    #         让它**直接看像素**，回答"文字'XX'在这张小图里的精确位置"，
    #         返回 0-1 比例 JSON，再换算回原图 0-1000 归一化坐标。
    #   Fallback: 如果 qwen-vl 调用失败（无Key/超时），就把 Step1 的 OCR 粗框
    #             收紧到 target_chars 的比例位置。
    # =====================================================================
    def _ocr_rough_row_box(self, ocr_raw: Dict, target_text: str, img_path: str) -> Optional[Dict]:
        """留作兼容用，调用新的终极定位函数。"""
        return None

    def _vlm_refine_bbox_exact(self, img_path: str, ocr_raw: Dict, target_text: str) -> Optional[List[int]]:
        """
        2026-09-09 真正的终极修复（本地DEBUG脚本跑通后确定）：
        ------------------------------------------------------------------
        ROOT CAUSE THAT BROKE ALL PREVIOUS ALGORITHMS:
          OCR.space 返回的 Line 对象 ** Left / Top / Width / Height 全部是缺失的 **
          （调试脚本显示 -999.0），只有 Line 下面的每一个 Word 对象才有真实
          Left/Top/Width/Height！之前一直拿 Line 的 -999 去运算，还按字符数比例切，
          结果再怎么nudge也是错的。
        ------------------------------------------------------------------
        正确的算法（这次是真的对）:
          Step A. 遍历所有 Words 找哪些 Word 的 WordText 的字符包含 target_clean。
                  支持 target 跨多个 Word（比如 "7" 在 Word1，"天" 在 Word2，
                  OCR 真实识别就拆成2个Word，我们需要把它们的包围盒 union 起来）。
          Step B. 有 Word 命中 → 直接把这些 Word 的真实 L/T/W/H 做 min/max 联合。
                  不需要任何字符比例推算，Word 的像素坐标就是 OCR 引擎认定的字边界。
          Step C. 最后做一次 INNER NUDGE（左右收紧 3%），避免 Word 本身的 padding
                  把邻居字（"领" 和 "出"）一点点边缘蹭进来。
          Step D. 如果 Word 级没命中（target_clean 是 4 字以上，OCR 合成了长 Line），
                  才回退使用 LineText 字符比例切片（此时 Line 包围盒从 Words 汇总得到）。
        """
        target_clean = (target_text or "").replace(" ", "").strip()
        if not target_clean:
            return None
        try:
            with Image.open(img_path) as img:
                w_img, h_img = img.size
        except Exception:
            w_img, h_img = None, None
        if not w_img:
            return None

        lines = []
        if isinstance(ocr_raw, dict) and "ParsedResults" in ocr_raw:
            try:
                lines = ocr_raw["ParsedResults"][0].get("TextOverlay", {}).get("Lines", []) or []
            except Exception:
                lines = []
        if not lines and isinstance(ocr_raw, dict):
            lines = ocr_raw.get("Lines", []) or []
        if not lines:
            return None

        # ================================================================
        # Step A. WORD-LEVEL matching（真正的精准算法，不再瞎猜）
        # ================================================================
        n_chars_total = len(target_clean)

        # Build flat word list
        all_words = []  # [(line_idx, w_idx, word_dict, text_clean, L, T, R, B, W, H)]
        for li, line in enumerate(lines):
            words = line.get("Words", []) or []
            for wi, w in enumerate(words):
                wt = (w.get("WordText") or "").replace(" ", "")
                if not wt:
                    continue
                L = float(w.get("Left", 0)); T = float(w.get("Top", 0))
                W = float(w.get("Width", 0)); H = float(w.get("Height", 0))
                if W <= 0 or H <= 0:
                    continue
                R = L + W; B = T + H
                all_words.append((li, wi, w, wt, L, T, R, B, W, H))

        # Never concatenate words from different OCR lines. The previous global
        # Top/Left sort could make slightly misaligned rows adjacent and produce
        # a geometrically invalid union box.
        words_by_line = {}
        for word in all_words:
            words_by_line.setdefault(word[0], []).append(word)
        for line_words in words_by_line.values():
            line_words.sort(key=lambda word: word[4])

        # Find the smallest same-line word window containing the target. Prefer
        # less surplus text, then the earlier reading-order occurrence.
        best_window = None
        for li, line_words in words_by_line.items():
            count = len(line_words)
            for window_len in range(1, count + 1):
                for start in range(0, count - window_len + 1):
                    window = line_words[start:start + window_len]
                    joined = "".join(word[3] for word in window)
                    found_at = joined.find(target_clean)
                    if found_at < 0:
                        continue
                    score = (window_len, len(joined) - n_chars_total, li, start)
                    candidate = (
                        score, window, joined, found_at,
                        found_at + n_chars_total, line_words,
                    )
                    if best_window is None or score < best_window[0]:
                        best_window = candidate
                if best_window and best_window[0][0] == window_len:
                    break

        if best_window is not None:
            _, window, joined, fs, fe, line_words = best_window

            # Map the target character range back into each OCR word separately.
            # This preserves real inter-word gaps instead of assuming a uniform
            # character grid across the whole line.
            segments = []
            cursor = 0
            for word in window:
                word_text = word[3]
                word_start, word_end = cursor, cursor + len(word_text)
                overlap_start = max(fs, word_start)
                overlap_end = min(fe, word_end)
                if overlap_end > overlap_start:
                    local_start = (overlap_start - word_start) / max(1, len(word_text))
                    local_end = (overlap_end - word_start) / max(1, len(word_text))
                    seg_left = word[4] + word[8] * local_start
                    seg_right = word[4] + word[8] * local_end
                    segments.append((seg_left, word[5], seg_right, word[7]))
                cursor = word_end

            if not segments:
                return None
            Lpx = min(seg[0] for seg in segments)
            Tpx = min(seg[1] for seg in segments)
            Rpx = max(seg[2] for seg in segments)
            Bpx = max(seg[3] for seg in segments)

            # Inpainting needs the full anti-aliased glyph edge. Expand slightly
            # outwards, then stop before a neighbouring OCR word.
            heights = sorted(max(1.0, word[9]) for word in window)
            glyph_h = heights[len(heights) // 2]
            pad_x = max(1.0, glyph_h * 0.10)
            pad_y = max(1.0, glyph_h * 0.10)
            selected_ids = {(word[0], word[1]) for word in window}
            previous_edges = [word[6] for word in line_words if (word[0], word[1]) not in selected_ids and word[6] <= Lpx]
            next_edges = [word[4] for word in line_words if (word[0], word[1]) not in selected_ids and word[4] >= Rpx]
            Lpx = max(0.0, Lpx - pad_x)
            Rpx = min(float(w_img), Rpx + pad_x)
            if previous_edges:
                Lpx = max(Lpx, max(previous_edges) + 1.0)
            if next_edges:
                Rpx = min(Rpx, min(next_edges) - 1.0)
            Tpx = max(0.0, Tpx - pad_y)
            Bpx = min(float(h_img), Bpx + pad_y)

            # 转 0-1000
            x1_1000 = max(0, int((Lpx / w_img) * 1000))
            y1_1000 = max(0, int((Tpx / h_img) * 1000))
            x2_1000 = min(1000, int((Rpx / w_img) * 1000))
            y2_1000 = min(1000, int((Bpx / h_img) * 1000))

            self._last_locate_debug = {
                "src": "WORD_GEOMETRY_" + str(len(window)),
                "confidence": 0.96 if joined == target_clean else 0.90,
                "words_in_window": [
                    {"text": w[3], "L": round(w[4],1),"T": round(w[5],1),"R": round(w[6],1),"B": round(w[7],1)}
                    for w in window
                ],
                "window_joined_text": joined,
                "target_in_joined": f"[{fs}:{fe}]",
                "final_px_rect": f"L={Lpx:.1f} T={Tpx:.1f} R={Rpx:.1f} B={Bpx:.1f}",
                "padding_px": f"x={pad_x:.1f}, y={pad_y:.1f}",
            }
            return [x1_1000, y1_1000, x2_1000, y2_1000]

        # ================================================================
        # Step D. WORD-LEVEL 没命中 → Fallback：Line-Level（汇总Words包围盒）
        # ================================================================
        best_hit = None
        for line in lines:
            line_text_raw = (line.get("LineText") or "")
            line_text_clean = line_text_raw.replace(" ", "")
            didx = line_text_clean.find(target_clean)
            if didx >= 0:
                # Line 没 Left/Top，从 Words 汇总包围盒拿真实像素
                ws = line.get("Words", []) or []
                if ws:
                    ls = [float(w.get("Left", 0)) for w in ws]
                    ts = [float(w.get("Top", 0)) for w in ws]
                    rs = [float(w.get("Left", 0)) + float(w.get("Width", 1)) for w in ws]
                    bs = [float(w.get("Top", 0)) + float(w.get("Height", 1)) for w in ws]
                    ll_mn = min(ls); tt_mn = min(ts)
                    rr_mx = max(rs); bb_mx = max(bs)
                else:
                    ll_mn = tt_mn = rr_mx = bb_mx = 0.0
                total_chars = max(1, len(line_text_clean))
                best_hit = {
                    "L": ll_mn, "T": tt_mn, "R": rr_mx, "B": bb_mx,
                    "total_chars": total_chars,
                    "s_idx": didx, "e_idx": didx + n_chars_total,
                    "line_text": line_text_raw,
                    "src": "LINE_FALLBACK_WORDS_UNION",
                }
                break
        if not best_hit:
            return None
        L = best_hit["L"]; T = best_hit["T"]
        R = best_hit["R"]; B = best_hit["B"]
        total_w = max(1.0, R - L)
        s_idx = best_hit["s_idx"]; e_idx = best_hit["e_idx"]
        n_total = best_hit["total_chars"]
        ratio_s = s_idx / n_total; ratio_e = e_idx / n_total
        Lpx = L + total_w * ratio_s; Rpx = L + total_w * ratio_e
        line_h = max(1.0, B - T)
        pad_x = max(1.0, line_h * 0.08)
        pad_y = max(1.0, line_h * 0.10)
        Lpx = max(0.0, Lpx - pad_x)
        Rpx = min(float(w_img), Rpx + pad_x)
        Tpx = max(0.0, T - pad_y)
        Bpx = min(float(h_img), B + pad_y)
        x1_1000 = max(0, int((Lpx / w_img) * 1000))
        y1_1000 = max(0, int((Tpx / h_img) * 1000))
        x2_1000 = min(1000, int((Rpx / w_img) * 1000))
        y2_1000 = min(1000, int((Bpx / h_img) * 1000))
        self._last_locate_debug = {
            "src": best_hit["src"],
            "confidence": 0.72,
            "line_text": best_hit.get("line_text",""),
            "s_idx_e_idx": f"[{s_idx}:{e_idx}]/{n_total}",
            "line_px": f"L={L:.1f} T={T:.1f} R={R:.1f} B={B:.1f}",
            "char_ratio": f"ratio_s={ratio_s:.4f} ratio_e={ratio_e:.4f}",
            "padding_px": f"x={pad_x:.1f}, y={pad_y:.1f}"
        }
        return [x1_1000, y1_1000, x2_1000, y2_1000]

    def _rough_to_tight_proportional(self, rx1, ry1, rx2, ry2, target_clean, w_img, h_img):
        """保留兼容"""
        return None

    def _heuristic_text_box(self, ocr_text: str, target_token: str, img_path: Optional[str]) -> List[int]:
        """
        Heuristic bbox when OCR raw data / VL call unavailable.
        Strategy: divide image by 3 rows × 4 cols → pick cell that contains the
        target token in OCR if possible, else the upper-middle area where
        marketing copy typically sits.
        """
        w = h = 1000
        if img_path and os.path.exists(img_path):
            try:
                with Image.open(img_path) as im:
                    pass  # 1000-normalized, no pixel dims needed
            except Exception:
                pass

        cells = [
            # (row, col, xmin, ymin, xmax, ymax) — marketing copy usually top 2 rows
            (0, 1, 200, 100, 500, 280),
            (0, 2, 500, 100, 800, 280),
            (1, 0, 80,  300, 350, 500),
            (1, 1, 350, 300, 600, 500),
            (1, 2, 600, 300, 850, 500),
            (0, 0, 80,  100, 300, 280),
        ]
        # Try to locate token in OCR by position
        if ocr_text and target_token:
            clean_ocr = ocr_text.replace(" ", "")
            clean_tok = target_token.replace(" ", "")
            if clean_tok and clean_tok in clean_ocr:
                idx = clean_ocr.find(clean_tok)
                ratio = idx / max(1, len(clean_ocr))
                if ratio < 0.34:
                    return cells[0][2:]
                elif ratio < 0.67:
                    return cells[3][2:]
                else:
                    return cells[4][2:]
        # Default: top-middle cell (headline area)
        return cells[0][2:]

    def _find_bbox_from_ocr(self, ocr_raw: Dict, target_text: str, img_path: Optional[str]) -> Optional[List[int]]:
        """
        ** WORD-LEVEL PRECISE BOX (NOT WHOLE LINE) — HISTORY LOGIC RESTORED + 竖列OCR 修复 **
        Strategy OCR (ground truth, primary box) before VL model (secondary validation):
          1. Line-level search → if LineText contains target_text → narrow further to WORDS
          2. Word-level search: only words whose WordText (or cleaned) contains
             target_text are included in the bbox (NOT min/max of ALL words in line)
          3. Consecutive-word aggregation: if target_text spans multiple OCR words
             (e.g. "双11口红" got split as ["双11", "口红"]), merge their boxes TIGHTLY
             so only the exact token span is highlighted.
          4. *NEW 2026-09* — CLUSTER LINES BY Y CENTER → avoid "vertical column" OCR bug
             (Paddle/EasyOCR sometimes merge 领7天出行守护 + 骑行必备 into a tall column,
             making Height > Width). All lines whose Y center is within 35% of the
             tallest word's height are merged into ONE HORIZONTAL ROW.
          5. *NEW 2026-09* — HIGH-ASPECT-RATIO WORD RECTIFICATION: if a matched word
             has H/W > 1.2 (vertical column shape), we re-derive its bbox from the
             union HORIZONTAL extent of its entire y-cluster, using char-position
             proportions on the UNION width.
        """
        if not target_text:
            return None
        if not ocr_raw or "ParsedResults" not in ocr_raw:
            return None
        try:
            with Image.open(img_path) as img:
                w_img, h_img = img.size
        except Exception:
            w_img, h_img = 1000, 1000

        try:
            parsed = ocr_raw["ParsedResults"][0]
            overlay = parsed.get("TextOverlay", {})
            lines = overlay.get("Lines", [])
        except Exception:
            return None
        if not lines:
            return None

        target_clean = (target_text or "").replace(" ", "").strip()
        if not target_clean:
            return None

        # ==============================================================
        # (NEW) PASS 0 — FLATTEN & CLUSTER ALL WORDS BY Y-CENTER TO
        #               RECOVER TRUE HORIZONTAL ROWS EVEN IF OCR MADE
        #               VERTICAL-COLUMN LINE GROUPS.
        # ==============================================================
        all_words_global = []  # (Line, Word, WordTextClean, cx, cy, W, H)
        for line in lines:
            lw = line.get("Words", []) or []
            for w in lw:
                wt = (w.get("WordText") or "").replace(" ", "")
                if not wt:
                    continue
                wl = float(w.get("Left", 0)); wt_ = float(w.get("Top", 0))
                ww = float(w.get("Width", 0)); wh = float(w.get("Height", 0))
                if ww <= 0 or wh <= 0:
                    continue
                cx = wl + ww/2; cy = wt_ + wh/2
                all_words_global.append((line, w, wt, cx, cy, ww, wh, wl, wt_, ww, wh))

        # Cluster words whose cy differs by < 22% of max(their wh) — STRICTER
        # so two physical rows ("领7天出行守护" small top row vs "骑行守护"
        # huge red bottom row) never get merged into a single cluster.
        clusters = []  # list of list of indices into all_words_global
        _tmp = sorted(range(len(all_words_global)), key=lambda i: all_words_global[i][4])
        for idx in _tmp:
            w_obj = all_words_global[idx]
            cy = w_obj[4]; wh = w_obj[6]
            placed = False
            for cl in clusters:
                # compare cluster's avg cy vs this word's cy
                c_cys = [all_words_global[j][4] for j in cl]
                c_whs = [all_words_global[j][6] for j in cl]
                avg_cy = sum(c_cys)/len(c_cys); max_wh = max(max(c_whs), wh)
                # STRICTER THRESHOLD (was 0.45 → 0.22) so two physical rows split
                if abs(avg_cy - cy) < max_wh * 0.22:
                    cl.append(idx)
                    placed = True
                    break
            if not placed:
                clusters.append([idx])

        # ==============================================================
        # PASS 1: Try exact word / cleaned-word matches (preferred)
        #         Now operating on the flattened, cluster-aware list.
        # ==============================================================
        matched_words = []  # list of (Word, WordTextClean, cluster_idx_in_clusters_list)
        matched_line = None
        # Track the cluster id (index into clusters list) for each matched word
        for line in lines:
            line_text = line.get("LineText", "").replace(" ", "")
            if target_clean not in line_text:
                continue
            matched_line = line
            words = line.get("Words", []) or []
            # Sub-pass 1a: exact substring in single Word
            found_single = False
            for w in words:
                wt = (w.get("WordText") or "").replace(" ", "")
                if target_clean in wt or wt in target_clean:
                    # locate which global cluster this word belongs to
                    wl_v = float(w.get("Left", 0)); wt_v = float(w.get("Top", 0))
                    ww_v = float(w.get("Width", 0)); wh_v = float(w.get("Height", 0))
                    cid = -1
                    for ci, cl in enumerate(clusters):
                        for gi in cl:
                            gw = all_words_global[gi]
                            if (abs(gw[7]-wl_v)<0.5 and abs(gw[8]-wt_v)<0.5 and
                                abs(gw[9]-ww_v)<0.5 and abs(gw[10]-wh_v)<0.5):
                                cid = ci; break
                        if cid >= 0: break
                    matched_words.append((w, wt, cid))
                    found_single = True
            if found_single:
                break
            # Sub-pass 1b: multi-word consecutive coverage (target spans 2+ words)
            cws = [((w.get("WordText") or "").replace(" ", ""), w) for w in words]
            joined = "".join(t for t, _ in cws)
            if target_clean in joined:
                s = joined.find(target_clean)
                e = s + len(target_clean)
                cur = 0
                sel_indices = []
                for i, (t, w) in enumerate(cws):
                    wl = len(t)
                    if cur < e and cur + wl > s:
                        sel_indices.append(i)
                    cur += wl
                if sel_indices:
                    for i in sel_indices:
                        w = words[i]
                        wt_c = (w.get("WordText") or "").replace(" ", "")
                        wl_v = float(w.get("Left", 0)); wt_v = float(w.get("Top", 0))
                        ww_v = float(w.get("Width", 0)); wh_v = float(w.get("Height", 0))
                        cid = -1
                        for ci, cl in enumerate(clusters):
                            for gi in cl:
                                gw = all_words_global[gi]
                                if (abs(gw[7]-wl_v)<0.5 and abs(gw[8]-wt_v)<0.5 and
                                    abs(gw[9]-ww_v)<0.5 and abs(gw[10]-wh_v)<0.5):
                                    cid = ci; break
                            if cid >= 0: break
                        matched_words.append((w, wt_c, cid))
                    break
        if not matched_words:
            # --------------------------------------------------------
            # PASS 2: Fuzzy char overlap ≥80% in single Word
            # --------------------------------------------------------
            for line in lines:
                words = line.get("Words", []) or []
                best_ratio = 0.0
                best_word = None
                best_cid = -1
                for w in words:
                    wt = (w.get("WordText") or "").replace(" ", "")
                    if not wt or not target_clean:
                        continue
                    inter = len(set(wt) & set(target_clean))
                    union = len(set(wt) | set(target_clean))
                    ratio = (inter / union) if union else 0
                    sub = (1 - abs(len(target_clean) - len(wt)) / max(len(target_clean), len(wt)))
                    score = ratio * 0.5 + sub * 0.5
                    if score > best_ratio and score >= 0.65:
                        best_ratio = score
                        best_word = w
                        wl_v = float(w.get("Left", 0)); wt_v = float(w.get("Top", 0))
                        ww_v = float(w.get("Width", 0)); wh_v = float(w.get("Height", 0))
                        for ci, cl in enumerate(clusters):
                            for gi in cl:
                                gw = all_words_global[gi]
                                if (abs(gw[7]-wl_v)<0.5 and abs(gw[8]-wt_v)<0.5 and
                                    abs(gw[9]-ww_v)<0.5 and abs(gw[10]-wh_v)<0.5):
                                    best_cid = ci; break
                            if best_cid >= 0: break
                if best_word:
                    matched_words = [(best_word, (best_word.get("WordText") or "").replace(" ", ""), best_cid)]
                    matched_line = line
                    break

        if not matched_words:
            return None

        # ==============================================================
        # (NEW) PASS 0.5 — CLUSTER UNION & VERTICAL-COLUMN RECTIFICATION
        #
        # If any matched word has H / W > 1.2, or the final cluster spans
        # several OCR lines (tall column case), we use the cluster's
        # HORIZONTAL UNION (min Left → max Right across all words in the
        # same y-cluster) as the ground-truth row-width, and then
        # proportionally slice target chars along that UNION width.
        # Height is tightened to the cluster's single-character visual
        # estimate: max(Height of words in cluster whose H/W <= 1.2),
        # or if no such word exists, cluster's total height / n_physical_rows.
        # ==============================================================
        # First, group matched_words by their cluster id, pick dominant cluster
        from collections import Counter
        c_counts = Counter([mw[2] for mw in matched_words if mw[2] >= 0])
        dom_cid = c_counts.most_common(1)[0][0] if c_counts else -1

        cluster_words_all = []  # flat words in dominant cluster
        if dom_cid >= 0 and dom_cid < len(clusters):
            for gi in clusters[dom_cid]:
                gw = all_words_global[gi]
                cluster_words_all.append(gw)

        refined_boxes = []
        for w, wt, cid in matched_words:
            w_left   = float(w.get("Left",   0))
            w_top    = float(w.get("Top",    0))
            w_width  = float(w.get("Width",  0))
            w_height = float(w.get("Height", 0))
            aspect = w_height / max(0.001, w_width)

            # ----------------------------------------------------------
            # VERTICAL-COLUMN FIX: apply when aspect > 1.2 or we found a
            # richer y-cluster for this word that contains horizontal
            # neighbour words (e.g. 领 | 7天 | 出行守护 were split into
            # different lines but share y-cluster).
            # ----------------------------------------------------------
            if (aspect > 1.2 or (dom_cid == cid and len(cluster_words_all) >= 2)) and target_clean in wt:
                # ==================================================================
                # (NEW) SUB-SPLIT: If cluster_words_all still contains 2+ physical
                # rows (CY spread > 1.5 * median H), split into 2 sub-clusters
                # and keep ONLY the sub-cluster that contains our target word
                # (identified by w_left / w_width match). This guarantees the
                # union box never pulls in the huge "骑行守护" red-circle words
                # from the row below.
                # ==================================================================
                import statistics as _st_split
                _all_cys = [g[4] for g in cluster_words_all]
                _all_hs  = [g[6] for g in cluster_words_all]
                _cy_span = max(_all_cys) - min(_all_cys) if _all_cys else 0
                _med_h   = _st_split.median(_all_hs) if _all_hs else 0
                _filtered_cluster = cluster_words_all
                if _med_h > 0 and _cy_span > _med_h * 1.5:
                    # Split into 2 groups by whether each word's CY is above or
                    # below the cluster global median CY. Then pick the group
                    # whose L/R range matches the target w_left/w_width.
                    _med_cy = _st_split.median(_all_cys)
                    _sub_top    = [g for g in cluster_words_all if g[4] <= _med_cy]
                    _sub_bottom = [g for g in cluster_words_all if g[4] >  _med_cy]
                    def _sub_contains_target(sub_list, _wl, _ww):
                        return any(abs(g[7]-_wl)<2.0 and abs(g[9]-_ww)<2.0 for g in sub_list)
                    if _sub_contains_target(_sub_top, w_left, w_width):
                        _filtered_cluster = _sub_top
                    elif _sub_contains_target(_sub_bottom, w_left, w_width):
                        _filtered_cluster = _sub_bottom
                cluster_words_all_for_union = _filtered_cluster

                # (A) Compute UNION row-extent of the FILTERED (single-row) cluster
                c_xmin = min(g[7] for g in cluster_words_all_for_union)
                c_xmax = max(g[7] + g[9] for g in cluster_words_all_for_union)
                c_w_min_h = min(g[10] for g in cluster_words_all_for_union)
                c_w_max_h = max(g[10] for g in cluster_words_all_for_union)
                c_cy = (min(g[8] for g in cluster_words_all_for_union) + max(g[8]+g[10] for g in cluster_words_all_for_union))/2

                # Find the words in this cluster whose aspect is horizontal (H/W<=1)
                horiz_words = [g for g in cluster_words_all_for_union if g[10] <= max(1, g[9]*1.2)]
                if horiz_words:
                    # Single-row visual height = median of H of the horizontal words
                    import statistics as _st
                    row_h = _st.median([g[10] for g in horiz_words])
                else:
                    # All words are tall columns → approximate row height by
                    # dividing total cluster vertical span by the rounded number
                    # of physical text rows we can see in the cluster (use
                    # len(target_clean) / len(cluster concatenated text) is too
                    # flaky; simpler: max(w_width, c_w_min_h) gives us a character
                    # size hint, then total_span / hint ≈ n_rows, then row_h =
                    # total_span / n_rows.
                    c_ymin = min(g[8] for g in cluster_words_all_for_union)
                    c_ymax = max(g[8]+g[10] for g in cluster_words_all_for_union)
                    span = max(1, c_ymax - c_ymin)
                    hint = max(c_w_min_h, w_width)
                    n_rows_est = max(1, round(span / hint))
                    row_h = span / n_rows_est

                # (B) Join the textual content of cluster words (sorted by X) IN
                #     READING ORDER so we can locate target_clean on the UNION.
                sorted_cw = sorted(cluster_words_all_for_union, key=lambda g: g[3])  # by cx
                row_full_text = "".join(g[2] for g in sorted_cw)
                row_len_chars = max(1, len(row_full_text))

                # (C) Where is target_clean inside this full row text?
                #     Priority 1 — exact match in the row-union full text:
                if target_clean in row_full_text:
                    s_idx = row_full_text.find(target_clean)
                    e_idx = s_idx + len(target_clean)
                else:
                    # Priority 2 — fall back to position within this single word
                    # as before, but still use cluster's c_xmin/c_xmax for width.
                    s_idx = wt.find(target_clean) if target_clean in wt else 0
                    e_idx = s_idx + len(target_clean)
                    # We also need to place this word's character position within
                    # the row: word starts at position p within the joined row,
                    # so add p offset.
                    p_offset = 0
                    acc = 0
                    for g in sorted_cw:
                        if abs(g[7]-w_left)<0.5 and abs(g[9]-w_width)<0.5:
                            p_offset = acc; break
                        acc += len(g[2])
                    s_idx += p_offset; e_idx += p_offset

                # (D) Proportionally slice the UNION row-width c_xmin→c_xmax
                #     + INNER PADDING: nudge ratios 2% INWARD so box edges never
                #       overlap into the neighbouring char (e.g. "出" right after "7天")
                ratio_s = s_idx / row_len_chars
                ratio_e = e_idx / row_len_chars
                _pad = min(0.04, 0.5 / max(1, len(target_clean)))  # 2% char inward
                ratio_s = min(ratio_s + _pad, ratio_e - 0.02)
                ratio_e = max(ratio_e - _pad, ratio_s + 0.02)
                union_w = c_xmax - c_xmin
                new_left  = c_xmin + union_w * ratio_s
                new_right = c_xmin + union_w * ratio_e

                # (E) Row vertical center is c_cy; half-height = row_h * 0.52
                new_top = c_cy - row_h * 0.52
                new_bottom = c_cy + row_h * 0.52

                refined_boxes.append((new_left, new_top, new_right, new_bottom))
                continue

            # ------------------------------------------------------------
            # SUB-WORD CHAR-LEVEL SLICING (original logic, kept for normal
            # horizontal words where aspect <= 1.2).
            # ------------------------------------------------------------
            w_right  = w_left + w_width
            w_bottom = w_top  + w_height

            if len(wt) > len(target_clean) and target_clean in wt:
                s_idx = wt.find(target_clean)
                e_idx = s_idx + len(target_clean)
                total_chars = len(wt)
                if total_chars > 0 and e_idx > s_idx:
                    ratio_s = s_idx / total_chars
                    ratio_e = e_idx / total_chars
                    # INNER PADDING: nudge ratios INWARD ~1 char-width (excluding extremes) so
                    # we don't paint the left-edge of the next glyph (e.g. "出" after "7天")
                    _char_w = 1.0 / total_chars
                    _shrink = min(_char_w * 0.35, (ratio_e - ratio_s) * 0.08)
                    ratio_s += _shrink
                    ratio_e -= _shrink
                    new_left  = w_left + w_width * ratio_s
                    new_right = w_left + w_width * ratio_e
                    refined_boxes.append((new_left, w_top, new_right, w_bottom))
                    continue
            refined_boxes.append((w_left, w_top, w_left+w_width, w_top+w_height))

        if not refined_boxes:
            return None

        # ------------------------------------------------------------
        # Compute TIGHT bbox from refined_boxes (sub-word sliced if applicable)
        # ------------------------------------------------------------
        left   = min(b[0] for b in refined_boxes)
        top    = min(b[1] for b in refined_boxes)
        right  = max(b[2] for b in refined_boxes)
        bottom = max(b[3] for b in refined_boxes)

        # Add 1-px inner padding (tighten) so neighboring glyphs are excluded
        pad = 2
        w_line_h = max((float(ww.get("Height", 0)) if hasattr(ww, 'get') else 0) for ww, _, _ in matched_words) if matched_words else 0
        # For fallback if generator above failed
        if not w_line_h or w_line_h <= 0:
            w_line_h = bottom - top
        if w_line_h:
            pad = max(1, int(w_line_h * 0.04))
        left   = min(right  - 1, left  + pad)
        top    = min(bottom - 1, top   + pad)
        right  = max(left   + 1, right - pad)
        bottom = max(top    + 1, bottom- pad)

        # Convert to 0-1000 normalized scale
        xmin = max(0,    int((left   / w_img) * 1000))
        ymin = max(0,    int((top    / h_img) * 1000))
        xmax = min(1000, int((right  / w_img) * 1000))
        ymax = min(1000, int((bottom / h_img) * 1000))
        if xmax - xmin < 10 or ymax - ymin < 6:
            # Too tiny → loosen slightly (ensure UI can see / inpaint is valid)
            xmin = max(0, xmin - 6); xmax = min(1000, xmax + 6)
            ymin = max(0, ymin - 4); ymax = min(1000, ymax + 4)
        return [xmin, ymin, xmax, ymax]


    def _call_qwen_edit(self, img_path: str, instruction: str) -> Dict:
        """
        Call qwen-image-edit-max for image editing.
        """
        print(f"   🎨 Qwen-Image-Edit: {instruction}")
        if not dashscope or not MultiModalConversation:
            return {"success": False, "reason": "dashscope SDK unavailable"}
        try:
            with open(img_path, "rb") as image_file:
                suffix = os.path.splitext(img_path)[1].lower()
                mime_type = "image/png" if suffix == ".png" else "image/jpeg"
                image_uri = (
                    f"data:{mime_type};base64,"
                    + base64.b64encode(image_file.read()).decode("ascii")
                )

            messages = [
                {
                    "role": "user",
                    "content": [
                        {"image": image_uri},
                        {"text": instruction}
                    ]
                }
            ]

            rsp = MultiModalConversation.call(
                api_key=self.api_key,
                model="qwen-image-edit-max",
                messages=messages,
                stream=False,
                n=1,
                watermark=False,
            )
            
            if rsp.status_code == HTTPStatus.OK:
                # Parse response
                # Structure: output.choices[0].message.content[0].image
                if hasattr(rsp, 'output') and hasattr(rsp.output, 'choices') and len(rsp.output.choices) > 0:
                    content = rsp.output.choices[0].message.content
                    image_item = next(
                        (
                            item for item in content
                            if isinstance(item, dict) and item.get("image")
                        ),
                        None,
                    ) if isinstance(content, list) else None
                    if image_item:
                        image_url = image_item["image"]
                        local_path = None
                        try:
                            response = requests.get(image_url, timeout=60)
                            response.raise_for_status()
                            output_dir = os.path.dirname(os.path.abspath(img_path))
                            local_path = os.path.join(
                                output_dir,
                                f"fission_qwen_{uuid.uuid4().hex[:10]}.png",
                            )
                            with open(local_path, "wb") as output_file:
                                output_file.write(response.content)
                            with Image.open(local_path) as check_image:
                                check_image.verify()
                        except Exception as download_error:
                            print(f"Qwen result persistence warning: {download_error}")
                        return {
                            "success": True,
                            "image_url": image_url,
                            "local_path": local_path,
                            "new_prompt": instruction,
                            "model": "qwen-image-edit-max",
                        }
                return {"success": False, "reason": "Failed to parse Qwen-Image-Edit response"}
            else:
                 reason = rsp.message
                 if rsp.code == "Throttling.AllocationQuota":
                     reason = "API配额耗尽或未开通 Qwen-Image-Edit 服务。"
                 return {"success": False, "reason": f"API Error: {reason}"}

        except Exception as e:
            print(f"Qwen Edit Error: {e}")
            return {
                "success": False, 
                "reason": f"编辑服务调用失败: {str(e)}"
            }

    def _change_background(self, img_path: str, bg_prompt: str) -> Dict:
        """
        Use Qwen-Image-Edit for background generation.
        """
        instruction = f"Change the background to {bg_prompt}"
        return self._call_qwen_edit(img_path, instruction)

    def _call_qwen_precise_edit(
        self,
        img_path: str,
        box_2d: List[int],
        prompt: str,
        is_text_mode: bool,
        target_text: str = "",
    ) -> Dict:
        """
        Use Qwen-Image-Edit-Max for precise text replacement.
        Tries to enforce constraints via prompt engineering.
        """
        print(f"   🎨 Qwen-Image-Edit text replacement: {prompt}")
        
        # Convert box_2d to string for prompt if valid
        box_str = ""
        if box_2d:
            # box_2d is [xmin, ymin, xmax, ymax] in 1000 scale
            # We can hint the area
            box_str = f"[{box_2d[0]}, {box_2d[1]}, {box_2d[2]}, {box_2d[3]}]"
        
        if is_text_mode:
            old_copy = f"“{target_text}”" if target_text else "目标文字"
            instruction = (
                f"精准编辑商品广告图：只把归一化坐标区域 {box_str} 内的文字"
                f"{old_copy}替换成“{prompt}”。必须准确显示"
                f"{len(prompt)}个中文字符“{prompt}”，保持原文字的字体、"
                "颜色、字号、位置、字距、行距、描边和阴影风格完全一致；"
                "其他所有像素、商品、Logo、边框和文案保持不变。"
            )
        else:
            # BG Edit: box_2d is the product (preserve it)
            instruction = f"Change the background to '{prompt}'. Keep the object{box_str} unchanged. Do not redraw the object."
            
        return self._call_qwen_edit(img_path, instruction)

    def build_edit_mask(self, img_path: str, box_2d: List[int], invert_mask: bool = False,
                        is_text_mode: bool = False):
        """Build the exact mask used by both preview and generation.

        Returns ``(mask, pixel_box)``. Text masks use scale-aware padding so a
        small banner is not expanded by the same 10 pixels as a large poster.
        """
        with Image.open(img_path) as image:
            width, height = image.size
        mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(mask)
        pixel_box = None
        if box_2d and len(box_2d) == 4:
            xmin, ymin, xmax, ymax = [float(value) for value in box_2d]
            x1 = int((xmin / 1000) * width)
            y1 = int((ymin / 1000) * height)
            x2 = int((xmax / 1000) * width)
            y2 = int((ymax / 1000) * height)
            ratio = 0.006 if is_text_mode else 0.015
            padding = max(2, int(round(min(width, height) * ratio)))
            x1 = max(0, x1 - padding)
            y1 = max(0, y1 - padding)
            x2 = min(width, x2 + padding)
            y2 = min(height, y2 + padding)
            pixel_box = (x1, y1, x2, y2)
            draw.rectangle(pixel_box, fill=255)
        if invert_mask:
            mask = ImageOps.invert(mask)
        # PIL's getbbox uses an exclusive right/bottom edge. Returning that
        # exact extent keeps the UI outline aligned with the actual mask bytes.
        return mask, mask.getbbox()
