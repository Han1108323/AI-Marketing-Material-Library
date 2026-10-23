"""Public, zero-config standalone (lite) entry point."""

from __future__ import annotations

from collections import Counter
from io import BytesIO
import math
import os

import altair as alt
import pandas as pd
from PIL import Image, ImageStat
import streamlit as st

from catalog import MATERIALS, categories, search_materials


st.set_page_config(page_title="TrendCrafter AI 素材库 · 基于历史素材数据资产的AI供给决策系统", page_icon="✦", layout="wide")

st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Noto+Sans+SC:wght@400;500;600;700&display=swap');
:root { --ink:#17211b; --muted:#68736b; --line:#dfe6df; --paper:#f5f7f2; --accent:#176b45; --lime:#dff56a; }
.stApp { background:linear-gradient(140deg,#f7f8f3 0%,#eef4ee 55%,#f8f3e9 100%); color:var(--ink); }
.block-container { max-width:1280px; padding-top:2.2rem; padding-bottom:4rem; }
html, body, [class*="css"] { font-family:'DM Sans','Noto Sans SC',sans-serif; }
h1,h2,h3 { letter-spacing:-.025em; }
[data-testid="stSidebar"] { background:#17211b; }
[data-testid="stSidebar"] * { color:#f4f7f3; }
.hero { padding:2.8rem 3rem; border:1px solid #d9e2d8; border-radius:28px; background:radial-gradient(circle at 85% 15%,rgba(223,245,106,.55),transparent 28%),#fcfdf9; box-shadow:0 22px 60px rgba(31,54,40,.09); margin-bottom:1.3rem; }
.eyebrow { font-size:.78rem; font-weight:700; letter-spacing:.15em; color:#176b45; text-transform:uppercase; }
.hero h1 { font-size:clamp(2.3rem,5vw,4.8rem); line-height:1.03; margin:.7rem 0 1rem; max-width:860px; color:#17211b; }
.hero p { color:#5f6b63; font-size:1.05rem; max-width:740px; line-height:1.75; }
.badge { display:inline-block; padding:.35rem .7rem; margin:.2rem .35rem .2rem 0; border-radius:999px; background:#edf3eb; color:#365142; font-size:.78rem; }
.metric-card { background:rgba(255,255,255,.72); border:1px solid var(--line); border-radius:18px; padding:1rem 1.1rem; min-height:112px; }
.metric-label { color:var(--muted); font-size:.78rem; }
.metric-value { font-size:1.8rem; font-weight:700; color:var(--ink); margin:.25rem 0; }
.metric-note { color:#7a857d; font-size:.76rem; }
.asset-card { border:1px solid var(--line); background:rgba(255,255,255,.8); padding:1rem; border-radius:20px; min-height:250px; }
.score-pill { float:right; background:#17211b; color:#dff56a; padding:.25rem .55rem; border-radius:999px; font-weight:700; font-size:.78rem; }
.decision { border-left:4px solid #176b45; background:#eef5ee; padding:1rem 1.2rem; border-radius:0 14px 14px 0; }
.footer { color:#718078; border-top:1px solid #dce4dc; padding-top:1.2rem; margin-top:2.5rem; font-size:.82rem; }
.stButton>button { border-radius:999px; border:1px solid #176b45; font-weight:600; }
.stButton>button[kind="primary"] { background:#176b45; color:white; }
</style>
""",
    unsafe_allow_html=True,
)


def metric_card(label: str, value: str, note: str) -> None:
    st.markdown(
        f'<div class="metric-card"><div class="metric-label">{label}</div><div class="metric-value">{value}</div><div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def score_decision(score: int, relevance: float) -> tuple[str, str]:
    if score >= 24 and relevance >= 0.35:
        return "直接复用", "匹配度与CTR预估都较高，优先复用，节省生成成本。"
    if score >= 20 and relevance >= 0.2:
        return "智能微调", "素材基础良好，建议保留构图并针对本次人群或节点裂变。"
    return "RAG 参考生成", "将高分视觉基因作为参考，结合新需求生成一版新素材。"


def palette_and_quality(image: Image.Image) -> dict:
    rgb = image.convert("RGB")
    thumb = rgb.copy()
    thumb.thumbnail((320, 320))
    palette = thumb.quantize(colors=5, method=Image.Quantize.MEDIANCUT).convert("RGB")
    counts = Counter(palette.getdata()).most_common(5)
    total = max(sum(count for _, count in counts), 1)
    colors = [{"hex": "#%02x%02x%02x" % color, "share": count / total} for color, count in counts]
    stat = ImageStat.Stat(thumb)
    brightness = sum(stat.mean) / 3
    contrast = sum(stat.stddev) / 3
    score = round(min(10, 4.5 + contrast / 18 + (1 - abs(brightness - 145) / 145) * 2.2), 1)
    return {"width": rgb.width, "height": rgb.height, "brightness": brightness, "contrast": contrast, "score": score, "colors": colors}


with st.sidebar:
    st.markdown("## ✦ TrendCrafter AI")
    st.caption("AI供给决策系统 · 基于历史素材数据资产")
    st.markdown("---")
    st.markdown("**运行状态**")
    api_ready = bool(os.getenv("DASHSCOPE_API_KEY"))
    st.info("零配置预览 · 内置行业样本数据集")
    if api_ready:
        st.success("DashScope · 已连接")
    else:
        st.caption("未配置云端模型；检索与图像体检功能仍可用，完整管线请配置环境变量。")
    st.markdown("---")
    st.markdown("**供给决策流程**")
    st.caption("01 解析商家需求\n\n02 多模态检索库存\n\n03 三角色评分与CTR预估\n\n04 复用 / 微调 / 生成分流")
    st.markdown("---")
    st.caption("TrendCrafter AI · v1.0")


st.markdown(
    """
<section class="hero">
  <div class="eyebrow">AI 供给决策系统 · 基于历史素材数据资产</div>
  <h1>TrendCrafter AI · 让营销素材生产<br/>从"拍脑袋做图"走向数据驱动供给。</h1>
  <p>核心逻辑不是"能不能生成一张图"，而是"该不该做 + 怎么做最优"：先判断素材库里有没有值得复用的高转化资产，再决定 REUSE 直接复用 / FISSION 智能微调 / GENERATE RAG 参考生成，感知 → 认知 → 决策 → 执行 → 反馈，5 层 AI 闭环。</p>
  <span class="badge">多模态理解入库（PROCESS 01）</span><span class="badge">智能检索三重召回（PROCESS 02）</span><span class="badge">三角色 AI 评审与CTR预估</span><span class="badge">FISSION智能微调（PROCESS 03）</span><span class="badge">RAG 参考生图（PROCESS 04）</span>
</section>
""",
    unsafe_allow_html=True,
)

avg_score = sum(m.score for m in MATERIALS) / len(MATERIALS)
best_ctr = max(m.ctr for m in MATERIALS)
m1, m2, m3, m4 = st.columns(4)
with m1: metric_card("样本素材库存", str(len(MATERIALS)), "行业样本 · 服饰/美妆/3C")
with m2: metric_card("平均三角色评分", f"{avg_score:.1f}/30", "视觉 · 文案 · 投放")
with m3: metric_card("样本最佳 CTR", f"{best_ctr:.1%}", "用于校准供给决策评分")
with m4: metric_card("供给决策路径", "3", "复用 · 智能微调 · RAG生成")

st.markdown("## 从一个真实需求开始")
query = st.text_input("需求描述", placeholder="例如：适合年轻人的高转化智能手表信息流广告", label_visibility="collapsed")
suggestions = ["智能手表 科技感", "年轻女性 清爽 美妆", "双11 红色促销", "高端男士礼赠"]
buttons = st.columns(4)
for idx, suggestion in enumerate(suggestions):
    if buttons[idx].button(suggestion, width="stretch"):
        st.session_state["suggested_query"] = suggestion
        st.rerun()
if not query and st.session_state.get("suggested_query"):
    query = st.session_state.pop("suggested_query")

filter_col, score_col = st.columns([2, 1])
with filter_col:
    category = st.selectbox("品类", categories(), label_visibility="collapsed")
with score_col:
    min_score = st.select_slider("最低预测分", options=[0, 18, 20, 22, 24], value=0, label_visibility="collapsed")

results = search_materials(query, category, min_score)
st.caption(f"找到 {len(results)} 个匹配素材 · 结果按需求相关度、预测分与历史 CTR 综合排序")

if query and results:
    top = results[0]
    action, reason = score_decision(top["score"], top["relevance"])
    st.markdown(
        f'<div class="decision"><b>Agent 建议：{action}</b><br/><span style="color:#5f6b63">{reason} 当前首选「{top["title"]}」，预测分 {top["score"]}/30。</span></div>',
        unsafe_allow_html=True,
    )

visible = results[:12]
for row_start in range(0, len(visible), 3):
    cols = st.columns(3)
    for col, item in zip(cols, visible[row_start : row_start + 3]):
        with col:
            st.image(item["path"], width="stretch")
            st.markdown(f'<span class="score-pill">{item["score"]}/30</span>', unsafe_allow_html=True)
            st.markdown(f"### {item['title']}")
            st.caption(f"{item['category']} · {item['purpose']} · CTR {item['ctr']:.1%}")
            st.markdown(" ".join(f'<span class="badge">{tag}</span>' for tag in [item["style"], *item["colors"][:2], *item["audience"][:1]]), unsafe_allow_html=True)
            with st.expander("查看 AI 评审"):
                c1, c2, c3 = st.columns(3)
                c1.metric("视觉", f"{item['visual_score']}/10")
                c2.metric("文案", f"{item['copy_score']}/10")
                c3.metric("投放", f"{item['ad_score']}/10")
                st.write(item["insight"])

if not results:
    st.info("当前精选库没有匹配结果。放宽筛选条件，或在下方上传素材进行即时体检。")

st.markdown("## 上传一张图，做即时视觉体检")
st.caption("该能力完全在本地运行，不上传图片；用于在没有模型 Key 时依然提供真实、可验证的工程链路验证。")
upload = st.file_uploader("上传 JPG / PNG / WebP", type=["jpg", "jpeg", "png", "webp"], label_visibility="collapsed")
if upload:
    image = Image.open(BytesIO(upload.getvalue()))
    analysis = palette_and_quality(image)
    image_col, report_col = st.columns([1, 1.15])
    with image_col:
        st.image(image, width="stretch")
    with report_col:
        st.markdown("### 视觉信号报告")
        q1, q2, q3 = st.columns(3)
        q1.metric("尺寸", f"{analysis['width']}×{analysis['height']}")
        q2.metric("对比度", f"{analysis['contrast']:.0f}")
        q3.metric("视觉基准分", f"{analysis['score']}/10")
        st.markdown("**主色板**")
        st.markdown("".join(f'<span title="{c["hex"]}" style="display:inline-block;width:54px;height:38px;background:{c["hex"]};margin-right:8px;border-radius:9px;border:1px solid #ddd"></span>' for c in analysis["colors"]), unsafe_allow_html=True)
        if analysis["contrast"] < 35:
            st.warning("画面对比较弱：用于信息流时，可强化主体/背景明暗差或加入单一高饱和强调色。")
        else:
            st.success("画面对比度适合快速建立视觉焦点；下一步重点检查利益点和 CTA 层级。")

st.markdown("## 数据闭环")
chart_data = pd.DataFrame([{"品类": m.category, "预测分": m.score, "历史CTR": m.ctr, "素材": m.title} for m in MATERIALS])
chart = (
    alt.Chart(chart_data)
    .mark_circle(opacity=0.82, stroke="#ffffff", strokeWidth=1.5)
    .encode(
        x=alt.X("预测分:Q", scale=alt.Scale(domain=[10, 30])),
        y=alt.Y("历史CTR:Q", axis=alt.Axis(format="%")),
        color=alt.Color("品类:N", legend=alt.Legend(orient="bottom")),
        size=alt.value(150),
        tooltip=["素材:N", "品类:N", "预测分:Q", alt.Tooltip("历史CTR:Q", format=".1%")],
    )
    .properties(height=360)
    .interactive()
)
st.altair_chart(chart, width="stretch")
st.caption("预测评分不是结论，而是假设。系统用真实 CTR 持续回测，识别评分与业务结果的偏差。")

st.markdown('<div class="footer">TrendCrafter AI • Empowering Creativity with Data<br/>感知→认知→决策→执行→反馈 · 5层AI闭环 · 3R决策矩阵（REUSE/FISSION/GENERATE）</div>', unsafe_allow_html=True)
