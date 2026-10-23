import streamlit as st
import os
import glob
from PIL import Image, ImageDraw
import time
import uuid
import pandas as pd
try:
    from taobao_utils import get_main_image_from_item_url
except ImportError:
    from src.taobao_utils import get_main_image_from_item_url
from pipeline import MaterialPipeline
from agent import MaterialAgent
from feedback import FeedbackSystem
from auth import AuthManager
from batch_utils import BatchProcessor
try:
    from dotenv import load_dotenv
except ImportError:
    load_dotenv = None
import altair as alt
import sys
import json

# --- Helper Functions for Background Import ---
KNOWN_IDS_FILE = "known_ids.json"
PENDING_DIR = "pending_ingest"

def export_known_ids():
    """Exports all current image IDs from Qdrant to known_ids.json"""
    if not st.session_state.agent:
        return
        
    try:
        # We need to scroll all points. 
        # Ideally, we should iterate, but for 50-100 items, limit=1000 is fine.
        points, _ = st.session_state.agent.qdrant.scroll(
            collection_name=st.session_state.agent.collection_name,
            limit=1000,
            with_payload=False,
            with_vectors=False
        )
        ids = [str(p.id) for p in points]
        with open(KNOWN_IDS_FILE, 'w') as f:
            json.dump(ids, f)
        # st.toast(f"Exported {len(ids)} known IDs for background process.")
    except Exception as e:
        print(f"Error exporting IDs: {e}")

def sync_pending_to_db():
    """Reads pending JSON files and upserts to Qdrant."""
    if not os.path.exists(PENDING_DIR):
        return
    
    files = [f for f in os.listdir(PENDING_DIR) if f.endswith('.json')]
    if not files:
        return
        
    # st.toast(f"🔄 Found {len(files)} pending items. Syncing...")
    
    from qdrant_client.models import PointStruct
    points = []
    processed_ids = []
    processed_files = []
    
    for f in files:
        path = os.path.join(PENDING_DIR, f)
        try:
            with open(path, 'r') as jf:
                data = json.load(jf)
                points.append(PointStruct(
                    id=data['id'],
                    vector=data['vector'],
                    payload=data['payload']
                ))
                processed_ids.append(data['id'])
                processed_files.append(f)
        except Exception as e:
            st.error(f"Error reading {f}: {e}")

    if points and st.session_state.agent:
        try:
            st.session_state.agent.qdrant.upsert(
                collection_name=st.session_state.agent.collection_name,
                points=points
            )
            # Remove processed files
            for f in processed_files:
                try:
                    os.remove(os.path.join(PENDING_DIR, f))
                except: pass

            # Update known_ids.json
            known_ids = []
            if os.path.exists(KNOWN_IDS_FILE):
                try:
                    with open(KNOWN_IDS_FILE, 'r') as kf:
                        known_ids = json.load(kf)
                except: pass
            
            known_ids.extend(processed_ids)
            known_ids = list(set(known_ids))
            
            with open(KNOWN_IDS_FILE, 'w') as kf:
                json.dump(known_ids, kf)
                
            st.success(f"✅ 成功同步 {len(processed_ids)} 个素材到数据库！")
            time.sleep(1)
            st.rerun()
        except Exception as e:
            st.error(f"Sync to DB failed: {e}")


# --- Custom CSS for Dark/Gradient Theme (Screenshot 1 Style) ---
st.markdown("""
<style>
    /* Global Background */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Headers (White text) */
    h1, h2, h3, h4, h5, h6 {
        color: #ffffff !important;
        font-family: 'Inter', sans-serif;
    }
    
    /* Buttons - Gradient Style */
    .stButton > button {
        background: linear-gradient(90deg, #0066cc 0%, #0099ff 100%);
        color: white !important;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        opacity: 0.9;
        box-shadow: 0 4px 12px rgba(0, 153, 255, 0.3);
    }
    
    /* Secondary Buttons */
    .stButton > button[kind="secondary"] {
        background: transparent;
        border: 1px solid #333;
        color: #ccc !important;
    }
    
    /* Inputs */
    .stTextInput > div > div > input {
        background-color: #1c1f26;
        color: white;
        border: 1px solid #333;
        border-radius: 6px;
    }
    
    /* Expander/Containers */
    .streamlit-expanderHeader {
        background-color: #161920;
        color: white;
    }
    
    /* Custom Info Boxes (Green/Blue/etc) overrides */
    div[data-testid="stMarkdownContainer"] p {
        color: #e0e0e0;
    }
    
    /* Sidebar */
    [data-testid="stSidebar"] {
        background-color: #0e1117;
        border-right: 1px solid #262730;
    }
</style>
""", unsafe_allow_html=True)

def display_score(payload):
    """Helper to display score and ladder tags"""
    score = payload.get("predicted_score", 0)
    # If score is 0, try to map from historical_ctr if available (Legacy fallback)
    if score == 0:
        ctr = payload.get("historical_ctr", 0.0)
        score = min(int(ctr * 600), 30)
    
    breakdown = payload.get("score_breakdown", {})
    if isinstance(breakdown, str):
        import json
        try: breakdown = json.loads(breakdown)
        except: breakdown = {}
        
    # Default breakdown if missing but score exists
    if score > 0 and not breakdown:
        avg = score // 3
        breakdown = {"visual": avg, "copy": avg, "ad": avg}
    
    # Color logic
    color = "#28a745" if score >= 24 else "#ffc107" if score >= 18 else "#dc3545"
    
    # Process Tags for Display
    tags = payload.get('tags', {})
    if isinstance(tags, str):
        import json
        try: tags = json.loads(tags)
        except: tags = {}
        
    tag_html = ""
    if tags:
        # Define some key tags to show prominently
        key_tags = ['visual_style', 'color_tone', 'composition', 'material_purpose', 'target_audience']
        for k, v in tags.items():
            if k in key_tags and v:
                # Dark theme friendly tag style
                tag_html += f'<span style="background-color: #2b303b; color: #e0e0e0; padding: 2px 8px; border-radius: 12px; font-size: 0.8em; margin-right: 5px; border: 1px solid #444;">{v}</span>'

    st.markdown(f"""
    <div style="background-color: #1c1f26; padding: 15px; border-radius: 8px; border-left: 5px solid {color}; margin-bottom: 10px; box-shadow: 0 1px 3px rgba(0,0,0,0.3);">
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 10px;">
            <h4 style="margin:0; color: {color}; font-size: 1.1em;">素材预测分：{score}/30 分</h4>
            <div style="font-size: 0.8em; color: #aaa;">多Agent专家评审</div>
        </div>
        
        <div style="display: flex; gap: 15px; margin-bottom: 10px; font-size: 0.9em; background: #2b303b; padding: 8px; border-radius: 6px;">
            <span title="Stopping Power">🎨 视觉: <b style="color: {color}">{breakdown.get('visual',0)}</b>/10</span>
            <span style="border-left: 1px solid #555;"></span>
            <span title="Readability">✍️ 文案: <b style="color: {color}">{breakdown.get('copy',0)}</b>/10</span>
            <span style="border-left: 1px solid #555;"></span>
            <span title="Click Desire">🎯 投放: <b style="color: {color}">{breakdown.get('ad',0)}</b>/10</span>
        </div>
        
        <div style="margin-bottom: 10px;">
            {tag_html}
        </div>

        <div style="font-style: italic; font-size: 0.85em; color: #bbb; border-top: 1px dashed #444; padding-top: 8px;">
            💡 <b>评价:</b> {payload.get('score_reasoning', '暂无详细评价')}
        </div>
    </div>
    """, unsafe_allow_html=True)

def display_gallery():
    """Display all images in the gallery directory"""
    st.markdown("### 📚 素材全览 (Gallery)")
    
    # Large Image Viewer (Modal-like)
    if st.session_state.get("view_image"):
        with st.container():
            st.info("🔍 查看大图模式")
            cols = st.columns([1, 10, 1])
            with cols[1]:
                st.image(st.session_state.view_image, use_container_width=True)
                if st.button("❌ 关闭大图", type="primary", use_container_width=True):
                    st.session_state.view_image = None
                    st.rerun()
            st.divider()

    img_dir = "AI素材案例"
    if not os.path.exists(img_dir):
        st.info("暂无素材，请上传或生成")
        return

    # Get all images
    files = glob.glob(os.path.join(img_dir, "*.*"))
    images = [f for f in files if f.lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))]
    
    if not images:
        st.info("暂无素材")
        return
        
    # Grid Layout
    # Use standard columns but make them responsive-ish by checking list length
    cols = st.columns(4)
    for idx, img_path in enumerate(images):
        with cols[idx % 4]:
            try:
                if os.path.exists(img_path) and os.path.getsize(img_path) > 0:
                    try:
                        with Image.open(img_path) as img:
                            img.verify()
                        st.image(img_path, use_container_width=True)
                        st.caption(os.path.basename(img_path))
                    except:
                        continue 
                else:
                    continue
            except:
                continue

@st.dialog("用户素材管理", width="large")
def show_user_materials_modal(uid, masked_phone):
    # Ensure session state storage for this user
    key_mats = f"mats_{uid}"
    key_offset = f"offset_{uid}"
    
    if key_mats not in st.session_state:
        # Initial Load
        with st.spinner("正在加载素材..."):
            mats, next_offset = st.session_state.agent.get_materials_by_user(uid, limit=20)
            st.session_state[key_mats] = mats
            st.session_state[key_offset] = next_offset

    mats = st.session_state[key_mats]
    next_offset = st.session_state[key_offset]
    
    st.caption(f"用户: {masked_phone} | 当前已加载: {len(mats)}")
    
    if not mats:
        st.info("该用户暂无管理素材")
    else:
        # Grid Display
        cols = st.columns(4)
        for idx, m in enumerate(mats):
            p = m['payload']
            fname = p.get('filename', 'unknown')
            img_path = os.path.join("AI素材案例", fname)
            with cols[idx % 4]:
                if os.path.exists(img_path):
                    st.image(img_path, use_container_width=True)
                    st.caption(fname)
                else:
                    st.caption(f"Missing: {fname}")
        
        # Load More
        if next_offset:
            if st.button("⬇️ 加载更多", key=f"btn_more_{uid}", use_container_width=True):
                with st.spinner("加载更多..."):
                    new_mats, new_next = st.session_state.agent.get_materials_by_user(uid, limit=20, offset=next_offset)
                    st.session_state[key_mats].extend(new_mats)
                    st.session_state[key_offset] = new_next
                    st.rerun()

# Load env vars
if load_dotenv is not None:
    try:
        load_dotenv(override=True)
    except Exception:
        pass

# Page Config
st.set_page_config(
    page_title="TrendCrafter AI 素材库 · 基于历史素材数据资产的AI供给决策系统",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS for cleaner UI (Dark Theme - Screenshot 1 Style)
st.markdown("""
<style>
    /* Global Background & Text */
    .stApp {
        background-color: #0e1117;
        color: #ffffff;
    }
    
    /* Inputs */
    .stTextInput input {
        background-color: #1c1f26;
        color: white;
        border: 1px solid #333;
    }
    
    /* Buttons - Gradient Style (Blue-ish) */
    .stButton > button {
        background: linear-gradient(90deg, #0066cc 0%, #0099ff 100%);
        color: white !important;
        border: none;
        border-radius: 6px;
        padding: 0.5rem 1rem;
        font-weight: 600;
        transition: all 0.3s ease;
    }
    .stButton > button:hover {
        opacity: 0.9;
        box-shadow: 0 4px 12px rgba(0, 153, 255, 0.3);
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background-color: #161920;
    }

    /* Expander */
    .streamlit-expanderHeader {
        background-color: #1c1f26 !important;
        color: white !important;
    }
    div[data-testid="stExpander"] {
        border: 1px solid #333;
        background-color: #1c1f26;
        border-radius: 8px;
    }
    
    /* Code/Tags */
    code {
        color: #e0e0e0 !important;
        background-color: #2b303b !important;
        border-radius: 4px;
    }

    /* Info/Success/Warning Boxes */
    .stAlert {
        background-color: #1c1f26;
        color: white;
        border: 1px solid #333;
    }

    .main .block-container {
        padding-top: 2rem;
    }
    
    /* File Uploader Localization Hack (Safe, no font-size:0 on button to avoid disabled visual on new Streamlit) */
    [data-testid="stFileUploaderDropzone"] div div::before {
       content: "将文件拖拽至此";
       display: block; 
       font-weight: bold;
    }
    [data-testid="stFileUploaderDropzone"] div div span {
       display: none;
    }
    [data-testid="stFileUploaderDropzone"] button span {
        /* Only hide the child span with default EN text, keep actual button text visible via streamlit's rendered label */
        color: transparent !important;
        font-size: 1px !important;
        position: absolute;
        pointer-events: none;
        visibility: hidden;
    }
    [data-testid="stFileUploaderDropzone"] button {
        /* Force button always active; streamlit manages real disabled state only when file_uploader(..., disabled=True) */
        pointer-events: auto !important;
        opacity: 1 !important;
        cursor: pointer !important;
        background: var(--secondary-background-color, #f0f2f6) !important;
        color: var(--text-color, inherit) !important;
        border: 1px solid rgba(49, 51, 63, 0.2) !important;
        padding: 0.35rem 0.9rem !important;
        font-size: 14px !important;
        border-radius: 0.5rem !important;
    }
    [data-testid="stFileUploaderDropzone"] button::after {
        content: "📂 浏览本地文件";
        display: inline-block !important;
        visibility: visible !important;
        opacity: 1 !important;
        pointer-events: none; /* Let real button beneath receive the click; this layer only shows the label */
    }
    [data-testid="stFileUploaderDropzone"] small {
        display: none;
    }
</style>
""", unsafe_allow_html=True)

# Initialize Session State
if "pipeline" not in st.session_state:
    st.session_state.pipeline = None
if "agent" not in st.session_state:
    st.session_state.agent = None
if "view_image" not in st.session_state:
    st.session_state.view_image = None
if "batch_processor" not in st.session_state:
    st.session_state.batch_processor = BatchProcessor()

# --- Authentication ---
if "auth" not in st.session_state:
    st.session_state.auth = AuthManager()

# Opt-in local showcase mode. It keeps production authentication unchanged,
# while allowing a reviewer to open the portfolio demo without receiving a
# reusable account password.
if os.getenv("TRENDCRAFTER_LOCAL_PREVIEW") == "1" and not st.session_state.get("user"):
    preview_user = st.session_state.auth.get_user("13800138000") or {}
    st.session_state.user = {
        "phone": preview_user.get("phone", "13800138000"),
        "role": "admin",
        "whitelist": True,
        "shop_name": preview_user.get("shop_name", "Vibe Coding 本地预览"),
        "upload_count": preview_user.get("upload_count", 0),
    }

if "user" not in st.session_state or not st.session_state.user:
    st.markdown("<br><br><br>", unsafe_allow_html=True)
    c1, c2, c3 = st.columns([1, 1.6, 1])
    with c2:
        st.title("✦ TrendCrafter AI 素材库")
        st.caption("基于历史素材数据资产的 AI 供给决策系统 · 感知→认知→决策→执行→反馈")
        with st.container(border=True):
            st.markdown("### 🔐 账号登录 / 注册")

            auth_mode = st.radio("操作类型", ["登录", "商家入驻注册"], horizontal=True, label_visibility="collapsed")

            phone = st.text_input("手机号", placeholder="请输入 11 位手机号", max_chars=11)
            password = st.text_input("密码", placeholder="请输入密码（至少 6 位）", type="password")

            if auth_mode == "登录":
                if st.button("🚀 登录", type="primary", use_container_width=True):
                    user, msg = st.session_state.auth.login(phone, password)
                    if user:
                        st.session_state.user = user
                        st.success(f"欢迎回来，{st.session_state.auth.mask_phone(phone)}")
                        time.sleep(0.6)
                        st.rerun()
                    else:
                        st.error(f"登录失败：{msg}")

                st.caption("平台环境访问权限由运营统一管理；白名单与管理员权限请联系管理员。本地部署按工程文档配置即可。")
            else:
                shop_name = st.text_input("店铺名称", placeholder="例如：XX美妆淘宝旗舰店")
                category = st.selectbox(
                    "主营品类",
                    ["请选择品类", "服饰鞋包", "美妆个护", "食品饮料", "3C数码", "运动户外", "母婴玩具", "家居日用", "珠宝配饰", "宠物用品", "其他"],
                    index=0
                )
                if st.button("📝 提交入驻", type="primary", use_container_width=True):
                    if len(phone) < 7:
                        st.error("请输入合法手机号")
                    elif len(password) < 6:
                        st.error("密码至少 6 位")
                    elif category == "请选择品类":
                        st.error("请选择主营品类")
                    elif not shop_name.strip():
                        st.error("请填写店铺名称")
                    else:
                        reg_ok, reg_msg = st.session_state.auth.register(phone, password)
                        if reg_ok:
                            st.session_state.auth.update_profile(phone, shop_name=shop_name, category=category, channel_prefs=["直通车"])
                            st.success("入驻提交成功！请联系平台运营将账号加入白名单后登录。")
                        else:
                            st.error(f"注册失败：{reg_msg}")

    st.stop()

try:
    required_keys = ["DASHSCOPE_API_KEY", "QDRANT_API_KEY"]
    missing_keys = [k for k in required_keys if not os.getenv(k)]
    
    if missing_keys:
        st.session_state.startup_warning = f"Missing API Keys: {', '.join(missing_keys)}. Some features may be disabled."
    else:
        st.session_state.startup_warning = None
    st.session_state.system_ready = True

    # Attempt to init pipeline with Retry Logic
    if st.session_state.agent is None or st.session_state.pipeline is None:
        max_retries = 3
        for attempt in range(max_retries):
            try:
                if st.session_state.pipeline is None:
                    st.session_state.pipeline = MaterialPipeline()
                if st.session_state.agent is None:
                    st.session_state.agent = MaterialAgent()
                if "feedback" not in st.session_state:
                    st.session_state.feedback = FeedbackSystem()
                break # Success
            except Exception as e:
                if attempt < max_retries - 1:
                    time.sleep(1)
                else:
                    st.session_state.startup_error = f"Core Init Failed after {max_retries} attempts: {str(e)}"
                    st.session_state.system_ready = True
        
except Exception as e:
    st.session_state.startup_error = str(e)
    st.session_state.system_ready = True

# --- Sidebar: Status & Config ---
with st.sidebar:
    # Current Account Info
    st.markdown("### 👤 当前账号")
    user = st.session_state.user
    if user:
        phone = user.get("phone", "")
        role = user.get("role", "user")
        shop = user.get("shop_name") or st.session_state.auth.get_user(phone).get("shop_name") if phone else None

        role_label_map = {"admin": "🛡️ 平台管理员", "designer": "🎨 设计岗", "user": "🏪 商家运营"}
        role_label = role_label_map.get(role, "🏪 商家运营")

        mask_phone = st.session_state.auth.mask_phone(phone)
        st.markdown(f"**账号**：{mask_phone}")
        st.markdown(f"**角色**：{role_label}")
        if shop:
            st.markdown(f"**店铺**：{shop}")

        if st.button("🚪 退出登录", use_container_width=True):
            st.session_state.user = None
            st.rerun()

        st.divider()

    # Step-by-step Debug Mode Toggle
    st.markdown("### 🛠 流程调试辅助")
    demo_mode = st.checkbox("🔍 分步调试模式 (Step-by-step Trace)",
                            value=False,
                            key="demo_mode",
                            help="开启后会减慢处理流程并可视化中间状态，用于分步输出入库/检索/微调的调用链日志与链路排查")
    
    if demo_mode:
        st.info("ℹ️ 分步调试模式已开启：流程将自动延时输出中间步骤与调用栈")

    # Auto-Sync Pending Items
    if st.session_state.agent:
        sync_pending_to_db()

    st.title("⚙️ 设置")
    

    with st.expander("环境状态检查", expanded=True):
        keys = {
            "DashScope (通义)": os.getenv("DASHSCOPE_API_KEY"),
            "OCR.space": os.getenv("OCR_KEY") or os.getenv("OCRSPACE_API_KEY"),
            "Qdrant": os.getenv("QDRANT_API_KEY")
        }
        
        all_ready = True
        for name, value in keys.items():
            if value:
                st.success(f"✅ {name} 已连接")
            else:
                st.error(f"❌ {name} 未配置")
                all_ready = False
        
        if not all_ready:
            st.warning("请在 .env 文件中配置缺失的 Key")
            st.info("提示: 可以在项目根目录创建 .env 文件")
            if st.button("🔄 刷新状态"):
                load_dotenv()
                st.rerun()
        
        if st.session_state.get("startup_error"):
            st.error(f"启动错误: {st.session_state.startup_error}")
            if st.button("🔄 重试连接"):
                st.session_state.clear()
                st.rerun()
    
    # Batch Task Status in Sidebar
    if "batch_processor" in st.session_state:
        bp = st.session_state.batch_processor
        if bp.progress["status"] == "running":
            with st.status("🔄 批量任务运行中...", expanded=False) as status:
                p = bp.progress
                st.progress(p["processed"] / p["total"] if p["total"] > 0 else 0)
                st.caption(f"进度: {p['processed']}/{p['total']}")
                st.caption(f"当前: {os.path.basename(p['current_file'])}")
                if st.button("查看详情"):
                     # Just a way to prompt user to go to tab
                     st.info("请前往“素材入库与批量处理”查看详情")

# --- Main Interface ---
st.title("✦ TrendCrafter AI 素材库")
st.caption("基于历史素材数据资产的 AI 供给决策系统 · 5 层 AI 闭环 + 3R 决策矩阵")

# ============================================================
# TAB 持久化修复（V2 终极版：JS 监听 tab button click，不在 with 块执行_save_tab）
#   - tab 点击时更新 URL query_params['tab'] 和 session_state['_tab_idx']
#   - 首次加载 / rerun 后，JS 根据 session_state 高亮正确的 tab，不因为 with 块
#     都被执行而被"最后一个 tab" 的 save 覆盖
# ============================================================
_query_tab = st.query_params.get("tab", None)
try:
    _default_tab = max(0, min(10, int(_query_tab))) if _query_tab is not None else 0
except Exception:
    _default_tab = 0
if "_tab_idx" not in st.session_state or _query_tab is not None:
    st.session_state["_tab_idx"] = _default_tab
_current_tab_for_js = int(st.session_state.get("_tab_idx", 0))

_tab_manager_js = f"""
<script>
(function() {{
  const SESSION_KEY = '_trendcrafter_tab_idx';
  const TARGET_IDX = {_current_tab_for_js};

  // (A) Restore query param from sessionStorage if Python hasn't received it yet
  try {{
    if (!window.location.search.includes('tab=')) {{
      const s = window.sessionStorage.getItem(SESSION_KEY);
      if (s != null) {{
        const n = parseInt(s, 10);
        if (!isNaN(n)) {{
          const u = new URL(window.location.href);
          u.searchParams.set('tab', String(n));
          window.history.replaceState(null, '', u.toString());
        }}
      }}
    }}
  }} catch(e) {{}}

  let tries = 0;
  function setupTabs() {{
    tries++;
    const hostDoc = parent.document || document;
    const tabBtns = hostDoc.querySelectorAll('.stTabs button[role="tab"]');
    if (!tabBtns || tabBtns.length === 0) {{
      if (tries < 40) setTimeout(setupTabs, 80);
      return;
    }}

    // --- 1) Persist user's manual click into URL / sessionStorage / session state sync ---
    tabBtns.forEach((btn, i) => {{
      if (btn.dataset._tc_bound === '1') return;
      btn.dataset._tc_bound = '1';
      btn.addEventListener('click', () => {{
        try {{
          const u = new URL(window.location.href);
          u.searchParams.set('tab', String(i));
          window.history.replaceState(null, '', u.toString());
          window.sessionStorage.setItem(SESSION_KEY, String(i));
        }} catch(e) {{}}
      }});
    }});

    // --- 2) Activate correct tab WITH ONE REAL .click() ONLY IF mismatch ---
    //   We ONLY call .click() if Streamlit internal active tab != TARGET_IDX.
    //   Calling .click() forces Streamlit to sync its internal active-tab
    //   state (and therefore rerender the correct tabpanel content). This
    //   guarantees "智能检索" tab always shows 智能检索内容, never the
    //   previous tab's leftover panel.
    try {{
      // Find currently active button according to Streamlit internal state
      let activeIdx = 0;
      tabBtns.forEach((b, i) => {{
        if (String(b.getAttribute('aria-selected') || '').toLowerCase() === 'true') {{
          activeIdx = i;
        }}
      }});

      if (activeIdx !== TARGET_IDX && tabBtns[TARGET_IDX]) {{
        tabBtns[TARGET_IDX].click();
        return; // let Streamlit rerun handle the rest; no DOM hacks needed
      }}

      // Streamlit may virtualize tab panels, so the DOM can contain only one
      // tabpanel even when activeIdx is greater than zero. Never map panel
      // indexes back to tab indexes here; Streamlit owns panel visibility.
    }} catch(e) {{}}
  }}
  setTimeout(setupTabs, 40);
}})();
</script>
"""
st.components.v1.html(_tab_manager_js, height=0, width=0)

# Tabs for Main Functions
tab_titles = [
    "🧠 智能检索与供给决策（PROCESS 02）",
    "🎨 智能微调 FISSION（PROCESS 03）",
    "🧬 RAG 智能生图（PROCESS 04）",
    # "🔍 智能搜索", # Hidden
    "📤 素材入库自动打标（PROCESS 01）",
    "🏷️ 素材标签数据资产",
    "📊 CTR 决策看板与闭环反馈",
    "⚙️ 商家配置中心"
]

# Check if admin (Default is admin for demo, but check role safely)
is_admin = st.session_state.user and st.session_state.user.get('role') == 'admin'
if is_admin:
    tab_titles.append("👥 用户管理")

tabs = st.tabs(tab_titles)

tab_library = tabs[0]
tab_fission = tabs[1]
tab_rag = tabs[2]
tab_upload = tabs[3]
tab_tags = tabs[4]
tab_dashboard = tabs[5]
tab_shop_config = tabs[6]
tab_users = tabs[7] if is_admin and len(tabs) > 7 else None

# Hidden Tab Placeholder
tab_search = None

# ============================================================
# 🔐 多租户数据隔离：空库商家（upload_count=0 且 非admin）能力边界限制
# 仅开放 智能微调 Fission + 商家配置中心；其余 Tab 显示统一锁定引导
# ============================================================
_shop_upload_count = st.session_state.user.get('upload_count', 0) if st.session_state.user else 0
# 本地验证/面试预览阶段：默认所有账号（包括空库）全部解锁，避免功能看起来不可用
is_empty_shop = False
# 正式多租户生产环境用下面一行（已注释）:
# is_empty_shop = (not is_admin) and _shop_upload_count == 0

def _render_locked_tab(tab_name_cn: str):
    """空库普通用户点击受限 Tab 时，渲染统一的禁用+解锁路径卡片，并停止渲染后续业务内容。"""
    st.info("🔒 功能未解锁：您当前的店铺素材库是空的")
    _c1, _c2, _c3 = st.columns([0.8, 2.4, 0.8])
    with _c2:
        _card = st.container(border=True)
        _card.markdown(f"""
### 🔐 「{tab_name_cn}」暂时不可用

**当前店铺状态**：您还没有上传任何自有商品素材，因此仅开放两项**不依赖素材库**的独立能力：
- ✅ 🎨 **智能微调 Fission**（直接上传单张图片即可生成变体）
- ✅ ⚙️ **商家配置中心**（填写店铺名称、主营品类、常用投放渠道）

---

**🔓 解锁其他业务模块（二选一）：**
1. **📤 自助批量入库（推荐）**：联系管理员为您开启「批量入库」权限，上传 **10 张以上**商品主图完成自动打标与向量化，入库完成后自动解锁本模块
2. **👔 管理员协助开通**：联系平台管理员为您的账号分配预置行业素材包，可跳过自行入库步骤直接验证全链路

> 新商家能力隔离策略见「⚙️ 商家配置中心 → 🔐 多租户隔离与数据安全」说明卡片。
        """)
    st.stop()


# --- Tab 1: Smart Retrieval & Supply Decision (PROCESS 02) ---
with tab_library:
    if is_empty_shop:
        _render_locked_tab("智能检索与供给决策")

    st.header("🧠 智能检索与供给决策")
    st.caption("5 层 AI 闭环 L1-L3：感知认知→检索召回→供给决策，3R 决策矩阵 REUSE/FISSION/GENERATE 分流")
    
    # --- Fine-tune UI ---
    if st.session_state.get("fine_tune_mode"):
        target = st.session_state.get("fine_tune_target", {})
        payload = target.get("payload", {})
        filename = payload.get("filename", "")
        
        with st.container():
            st.info(f"🎨 正在针对素材进行微调: {filename}")
            c_img, c_inp = st.columns([1, 3])
            with c_img:
                img_path = os.path.join("AI素材案例", filename)
                if os.path.exists(img_path):
                    st.image(img_path, width=200)
            with c_inp:
                ft_prompt = st.text_input("微调需求", placeholder="例如: 保持构图，换成春节氛围...", key="ft_prompt_manual")
                c_btn1, c_btn2 = st.columns([1, 1])
                with c_btn1:
                    if st.button("🚀 开始微调", type="primary", key="btn_ft_manual"):
                        if not ft_prompt:
                            st.warning("请输入微调需求")
                        else:
                            with st.spinner("微调中..."):
                                try:
                                    res = st.session_state.agent.fission_agent.execute(payload, ft_prompt)
                                    if res.get("success"):
                                        st.success("微调成功！")
                                        # Show preview...
                                        st.session_state.gen_preview = {
                                            "image_url": res.get("image_url"),
                                            "filename": f"fission_manual_{int(time.time())}.png",
                                            "prompt": res.get("new_prompt")
                                        }
                                        # Load image for preview
                                        import requests
                                        from io import BytesIO
                                        resp = requests.get(res["image_url"])
                                        st.session_state.gen_preview["image"] = Image.open(BytesIO(resp.content))
                                        
                                        st.session_state.fine_tune_mode = False
                                        st.rerun()
                                    else:
                                        st.error(f"微调失败: {res.get('reason')}")
                                except Exception as e:
                                    st.error(f"Error: {e}")
                with c_btn2:
                    if st.button("取消", key="btn_ft_cancel"):
                        st.session_state.fine_tune_mode = False
                        st.rerun()
            st.divider()

    col1, col2 = st.columns([5, 1])
    with col1:
        produce_query = st.text_input(
            "需求描述", 
            placeholder="例如: 双11美妆海报",
            label_visibility="collapsed",
            key="produce_input"
        )
    with col2:
        # Renamed button to "智能搜索" and attempted to make it visually distinct (Streamlit limits color options)
        produce_btn = st.button("智能搜索", type="primary", use_container_width=True, key="produce_btn")

    if produce_btn and produce_query:
        if not st.session_state.get("system_ready", False):
            st.error(f"系统未连接: {st.session_state.get('startup_warning', 'Unknown Error')}")
            # Debug info
            missing = [k for k in ["DASHSCOPE_API_KEY", "QDRANT_API_KEY"] if not os.getenv(k)]
            st.caption(f"Debug: Missing keys in env: {missing}")
        elif st.session_state.agent is None:
             st.error(f"Agent 初始化失败: {st.session_state.get('startup_error', '未知错误')}")
        else:
            # Use status instead of spinner for better visibility
            with st.status("🤖 智能检索中...", expanded=True) as status:
                try:
                    status.write("🔍 分析用户意图 (Intent Analysis)...")
                    if st.session_state.get("demo_mode"):
                        time.sleep(2)

                    status.write("🧬 生成查询向量 (DashScope Embedding)...")
                    if st.session_state.get("demo_mode"):
                        time.sleep(2)
                    
                    status.write("🧠 执行语义检索 (Semantic Search)...")
                    if st.session_state.get("demo_mode"):
                        time.sleep(2)

                    start_t = time.time()
                    
                    # Pass user_id for data isolation
                    current_user_id = st.session_state.user.get('phone')
                    # If admin, user_id=None means search global? 
                    # User requirement: "Super admin can see all materials" -> user_id=None
                    # "Other users can only see their own" -> user_id=phone
                    
                    search_user_id = current_user_id
                    if st.session_state.user.get('role') == 'admin':
                        search_user_id = None
                    
                    # Call the evaluate_supply method
                    decision = st.session_state.agent.evaluate_supply(produce_query, user_id=search_user_id)
                    
                    # DEBUG LOG
                    print(f"DEBUG: Decision for '{produce_query}':")
                    print(f"  - Action: {decision.get('action')}")
                    print(f"  - Candidates: {len(decision.get('top_candidates', []))}")
                    for idx, c in enumerate(decision.get('top_candidates', [])):
                        print(f"    [{idx}] {c.get('payload', {}).get('filename')} (Score: {c.get('score')})")

                    duration = time.time() - start_t
                    status.write(f"✅ 检索完成 (耗时 {duration:.2f}s)")
                    status.update(label="✅ 检索完成", state="complete", expanded=False)
                    
                    # Store decision in session state to persist across reruns
                    st.session_state.last_decision = decision
                    
                except Exception as e:
                    status.update(label="❌ 检索出错", state="error")
                    st.error(f"决策过程出错: {e}")
                    # Debug fallback
                    st.write(f"Debug Info: {str(e)}")
                    import traceback
                    st.code(traceback.format_exc())

    # Display Result (from session state to handle button clicks below)
    if "last_decision" in st.session_state:
        decision = st.session_state.last_decision
        
        # --- Step 0: Retrieval Analysis & Results (Redesigned) ---
        with st.expander("🕵️ 检索与分析过程 (Analysis Process)", expanded=True):
            # 1. Intent Analysis Info
            intent = decision.get("intent", {})
            # Dark theme analysis box
            st.markdown(f"""
            <div style="background:rgba(0, 102, 204, 0.2); padding:10px; border-radius:8px; margin-bottom:20px; border: 1px solid rgba(0, 102, 204, 0.3);">
                <span style="margin-right:15px; color: #e0e0e0;"><b>意图识别:</b> {intent.get('level', 'N/A')}</span>
                <span style="color: #e0e0e0;"><b>搜索词:</b> {decision.get('query')}</span>
            </div>
            """, unsafe_allow_html=True)
            if intent.get("error"):
                 st.error(f"⚠️ {intent.get('error')}")
            
            # 2. Candidate List (Vertical Layout)
            candidates = decision.get("top_candidates", [])
            
            # Filter Logic: Show all hits above a relaxed threshold, including fallback matches so users never see empty state
            valid_candidates = [c for c in candidates if c.get("score", 0) >= 0.15]
            
            if not valid_candidates:
                st.warning("⚠️ 暂无高分精准匹配较少，已展示参考素材：")
                valid_candidates = candidates[:15]
            else:
                st.markdown(f"### 🔍 召回候选 ({len(valid_candidates)})")

                for idx, item in enumerate(valid_candidates):
                    payload = item.get("payload", {})
                    score = item.get("score", 0)
                    filename = payload.get("filename", "unknown")
                    predicted_score = payload.get("predicted_score", 0)
                    
                    with st.container():
                        # Highlight Top 1
                        bg_color = "#f8f9fa" if idx == 0 else "white"
                        border = "2px solid #28a745" if idx == 0 else "1px solid #e0e0e0"
                        
                        c1, c2, c3 = st.columns([1, 2, 1])
                        
                        # Col 1: Image
                        with c1:
                            img_path = os.path.join("AI素材案例", filename)
                            if os.path.exists(img_path):
                                st.image(img_path, use_container_width=True)
                            else:
                                st.caption(f"Image Missing: {filename}")
                                
                        # Col 2: Info
                        with c2:
                            if idx == 0:
                                st.markdown("#### 🏆 最佳匹配 (Best Match)")
                            else:
                                st.markdown(f"**候选 #{idx+1}**")
                                
                            st.caption(f"文件名: {filename}")
                            st.markdown(f"""
                            - **语义相似度**: `{score:.2f}`
                            - **预测效果分**: `{predicted_score}/30`
                            """)
                            # Tags pill
                            tags = payload.get('tags', {})
                            if isinstance(tags, str):
                                import json
                                try: tags = json.loads(tags)
                                except: tags = {}
                            if tags:
                                t_str = " ".join([f"`{v}`" for k,v in tags.items() if k in ['visual_style', 'color_tone', 'material_purpose']][:3])
                                st.markdown(t_str)

                        # Col 3: Action (Fine-tune)
                        with c3:
                            st.write("") # Spacer
                            st.write("") 
                            # Toggle Fine-tune
                            ft_key = f"ft_active_{idx}"
                            is_active = st.session_state.get(ft_key, False)
                            btn_label = "🔼 收起" if is_active else "🎨 微调此图"

                            if st.button(btn_label, key=f"btn_ft_inline_{idx}", use_container_width=True):
                                st.session_state[ft_key] = not is_active
                                st.rerun()

                        # --- Inline Fine-tune Area ---
                        if st.session_state.get(f"ft_active_{idx}", False):
                            with st.container():
                                # Use RGBA for Dark/Light mode compatibility
                                st.markdown(f"""<div style="background:rgba(0, 123, 255, 0.15); padding:15px; border-radius:8px; margin-top:10px; margin-bottom:10px; border:1px dashed #007bff;">
                                    <small>🔧 <b>微调工作台</b> (基于: {filename})</small>
                                </div>""", unsafe_allow_html=True)
                                
                                ft_q = st.text_input("请输入微调需求:", value=decision.get("query"), key=f"input_ft_{idx}")
                                
                                if st.button("🚀 立即微调", key=f"run_ft_{idx}", type="primary"):
                                    with st.spinner("⚡️ 正在裂变生成中..."):
                                        try:
                                            # Execute Fission
                                            res = st.session_state.agent.fission_agent.execute(payload, ft_q)
                                            if res.get("success"):
                                                st.success("微调成功！")
                                                st.session_state.gen_preview = {
                                                    "image_url": res.get("image_url"),
                                                    "filename": f"fission_{int(time.time())}.png",
                                                    "prompt": res.get("new_prompt")
                                                }
                                                st.rerun()
                                            else:
                                                st.error(f"失败: {res.get('reason')}")
                                        except Exception as e:
                                            st.error(f"Error: {e}")
                                            
                        st.divider()

        # --- Step 1: Decision Suggestion (Below List) ---
        action = decision["action"]
        reason = decision["reason"]
        
        st.markdown("### 💡 决策建议 (Decision)")
        
        if action == "REUSE":
            st.success(f"✅ 建议：**直接复用 (REUSE)**")
            st.markdown(f"> **理由**: {reason}")
            # Show simplified champion view since it's already top of list
            st.caption("请参考上方列表中的【最佳匹配】素材。")
            
        elif action == "FISSION":
            st.info(f"⚡ 建议：**智能微调 (FISSION)**")
            st.markdown(f"> **理由**: {reason}")
            st.markdown("👉 **操作指引**: 请点击上方列表中最符合预期的素材右侧的 **'微调此图'** 按钮开始。")
            
        else:
            # st.warning(f"🚀 建议：**生产新图 (GENERATE)**")
            # st.markdown(f"> **理由**: {reason}")
            
            # Custom styled box for GENERATE (User requested color fix)
            st.markdown(f"""
            <div style="background:rgba(40, 167, 69, 0.15); padding:15px; border-radius:8px; margin-bottom:15px; border:1px solid rgba(40, 167, 69, 0.2);">
                <h4 style="margin:0; display:inline;">🚀 建议：生产新图 (GENERATE)</h4>
                <div style="margin-top:8px; opacity:0.9;">
                    <b>💡 理由:</b> {reason}
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Show Generation Strategy
            strategy = decision.get("strategy", {})
            st.markdown("#### 生成策略")
            c1, c2 = st.columns(2)
            with c1: 
                st.markdown("**✅ 参考:** " + ", ".join([f"`{t}`" for t in strategy.get('positive_cues', [])]))
            with c2:
                st.markdown("**❌ 避免:** " + ", ".join([f"`{t}`" for t in strategy.get('negative_cues', [])]))
            
            st.info(f"指导: {strategy.get('advice')}")
            
            # --- Action (Auto-Generate) ---
            if st.button("🎨 执行 RAG 生成", type="primary"):
                with st.spinner("🤖 正在检索参考素材并生成新图..."):
                    try:
                        # Use RAG Agent
                        result = st.session_state.agent.execute_decision(decision)
                        
                        if result.get("success"):
                            # Load image
                            generated_path = result.get("image_path") or result.get("image_url")
                            if generated_path and os.path.exists(generated_path):
                                img = Image.open(generated_path)
                                st.session_state.gen_preview = {
                                    "image": img, 
                                    "filename": f"gen_{int(time.time())}.png",
                                    "prompt": result.get("used_prompt", "")
                                }
                            elif result.get("image_url"):
                                # If it's a URL (from Fission or elsewhere)
                                import requests
                                from io import BytesIO
                                resp = requests.get(result["image_url"])
                                img = Image.open(BytesIO(resp.content))
                                st.session_state.gen_preview = {
                                    "image": img,
                                    "filename": f"gen_{int(time.time())}.png",
                                    "prompt": result.get("new_prompt", "")
                                }
                            st.rerun()
                        else:
                            st.error(f"生成失败: {result.get('reason')}")
                            
                    except Exception as e:
                        st.error(f"执行出错: {e}")


            if st.session_state.get("gen_preview"):
                st.image(st.session_state.gen_preview["image"], width=400, caption="生成预览")
                st.caption(f"Prompt: {st.session_state.gen_preview.get('prompt')}")
                
                c1, c2 = st.columns(2)
                with c1:
                    if st.button("👍 满意，入库", type="primary"):
                        with st.status("🔄 正在完成全链路处理...", expanded=True) as status:
                            status.write("2. 视觉理解与OCR提取")
                            status.write("3. 智能打标")
                            status.write("4. 效果预测")
                            status.write("5. 向量化入库")
                            # Pass user_id
                            current_uid = st.session_state.user.get('phone') if st.session_state.user else None
                            res_tuple = st.session_state.pipeline.process_image(
                                st.session_state.gen_preview["image"],
                                filename=st.session_state.gen_preview["filename"],
                                user_id=current_uid
                            )
                            new_payload = None
                            if isinstance(res_tuple, tuple):
                                new_payload = res_tuple[0]
                            else:
                                new_payload = res_tuple
                                
                            if new_payload:
                                if current_uid:
                                    st.session_state.auth.increment_upload_count(current_uid)
                                status.update(label="✅ 已完成打标并入库", state="complete", expanded=False)
                                st.success(f"✅ 素材已入库：{st.session_state.gen_preview['filename']}")
                                st.session_state.last_ingest_success = st.session_state.gen_preview["filename"]
                                st.session_state.gen_preview = None
                            else:
                                status.update(label="❌ 处理失败", state="error")
                with c2:
                    if st.button("🔁 重新生成"):
                        st.session_state.gen_preview = None
        
    # --- Always Show Gallery Below ---
    st.divider()
    display_gallery()
with tab_fission:
    st.header("🎨 智能微调 (Fission)")
    st.caption("Qwen-VL/OCR 精确定位 + Qwen-Image-Edit-Max 高质量文字微调")

    # Initialize State
    if "fission_state" not in st.session_state:
        st.session_state.fission_state = {"step": "init", "analysis": None, "payload": None}

    # Step 1: Upload & Analyze
    if st.session_state.fission_state["step"] == "init":
        c1, c2 = st.columns([1, 1])
        with c1:
            # Input Method Selection
            input_tab1, input_tab2 = st.tabs(["📤 本地上传", "🔗 淘宝链接"])
            
            uploaded_file = None
            target_image = None # This will hold the final PIL Image
            real_filename = "upload.png" # Default filename

            with input_tab1:
                uploaded_file = st.file_uploader("上传参考素材", type=["png", "jpg", "jpeg", "webp"], key="fis_up")
                if os.getenv("TRENDCRAFTER_LOCAL_PREVIEW") == "1":
                    if st.button("🪥 一键载入高质量文字微调案例", use_container_width=True, key="fis_quality_demo_load"):
                        st.session_state["fission_demo_enabled"] = True
                        st.session_state["fission_demo_path"] = os.path.join(
                            os.path.dirname(__file__), "AI素材案例", "shop_local_253.jpg"
                        )
                        st.session_state["fission_demo_caption"] = "历史目标案例：将“现货速发”改为“当日发货”"
                        st.session_state["fis_ins"] = "把“现货速发”改为“当日发货”"

                if uploaded_file:
                    # Logic to ensure consistent unique filename across reruns
                    file_key = f"{uploaded_file.name}_{uploaded_file.size}"
                    
                    # Check if new file or need to generate name
                    if st.session_state.get("last_upload_key") != file_key:
                        import uuid
                        unique_name = f"upload_{uuid.uuid4().hex[:8]}_{uploaded_file.name}"
                        st.session_state["last_upload_key"] = file_key
                        st.session_state["last_upload_filename"] = unique_name
                    else:
                        unique_name = st.session_state["last_upload_filename"]
                    
                    # Save to disk immediately (ensure existence)
                    save_dir = "AI素材案例"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_path = os.path.join(save_dir, unique_name)
                    
                    # Always write if missing (or just to be safe, overwrite is cheap for single image)
                    if not os.path.exists(save_path):
                        with open(save_path, "wb") as f:
                            f.write(uploaded_file.getbuffer())
                    
                    st.image(save_path, caption="参考图", use_container_width=True)
                    target_image = Image.open(save_path)
                    real_filename = unique_name
                elif st.session_state.get("fission_demo_enabled"):
                    demo_path = st.session_state.get(
                        "fission_demo_path",
                        os.path.join(os.path.dirname(__file__), "AI素材案例", "shop_local_253.jpg"),
                    )
                    if os.path.exists(demo_path):
                        save_path = demo_path
                        real_filename = os.path.basename(demo_path)
                        target_image = Image.open(demo_path)
                        st.image(
                            demo_path,
                            caption=st.session_state.get("fission_demo_caption", "微调演示案例"),
                            use_container_width=True,
                        )

            with input_tab2:
                tb_url = st.text_input("输入淘宝商品链接", placeholder="https://item.taobao.com/...", key="fis_url")
                if tb_url:
                    if st.button("提取主图", key="btn_fetch_tb", use_container_width=True):
                        with st.spinner("正在提取淘宝主图..."):
                            img = get_main_image_from_item_url(tb_url)
                            if img:
                                st.session_state.fission_state["fetched_image"] = img
                                import uuid
                                st.session_state.fission_state["fetched_filename"] = f"taobao_{uuid.uuid4().hex[:8]}.png"
                                st.success("提取成功")
                            else:
                                st.error("提取失败。可能是淘宝反爬限制。💡 建议：右键点击淘宝主图 -> '复制图片地址'，然后粘贴到这里。")
                
                # Use fetched image if available
                if st.session_state.fission_state.get("fetched_image"):
                    img = st.session_state.fission_state["fetched_image"]
                    fname = st.session_state.fission_state.get("fetched_filename", "taobao_fetched.png")
                    
                    # Ensure it exists on disk
                    save_dir = "AI素材案例"
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_path = os.path.join(save_dir, fname)
                    img.save(save_path)
                    
                    st.image(save_path, caption="已提取淘宝主图", use_container_width=True)
                    target_image = Image.open(save_path)
                    real_filename = fname

        with c2:
            instruction = st.text_area("2. 修改指令", placeholder="例如: 背景换成红色春节氛围 (保留商品)\n或者: 把标题文字改成双11大促", height=150, key="fis_ins")
            
            if st.button("🔍 分析意图 (Analyze)", type="primary", use_container_width=True):
                if not target_image or not instruction:
                    st.warning("请提供图片并输入指令")
                elif not st.session_state.get("system_ready"):
                    st.error("系统未连接")
                else:
                    with st.spinner("正在分析微调意图..."):
                        try:
                            # Ingest/Get Payload Logic
                            import uuid
                            img = target_image
                            
                            # Process image to get payload (Optimized for Speed)
                            # We only need OCR for Fission Analysis, skipping Vision/TagGen/CTR/Embedding
                            from pipeline import OCRProcessor
                            ocr_proc = OCRProcessor()
                            with st.spinner("正在提取文字信息 (OCR)..."):
                                ocr_data = ocr_proc.process(img)
                            
                            payload = {
                                "filename": real_filename,
                                "img_path": save_path,
                                "ocr_text": ocr_data.get("text", ""),
                                "ocr_meta": ocr_data # Keep full meta for coordinate refinement
                            }
                                
                            if payload:
                                # Analyze
                                # Fix SQLite Threading Issue: Instantiate FissionAgent locally to ensure thread safety
                                from fission import FissionAgent
                                local_fission_agent = FissionAgent()
                                analysis = local_fission_agent.analyze(payload, instruction)
                                
                                st.session_state.fission_state = {
                                    "step": "analyzed",
                                    "analysis": analysis,
                                    "payload": payload,
                                    "uploaded_file": img, 
                                    "instruction": instruction,
                                    "fission_agent": local_fission_agent,
                                }
                                st.rerun()
                            else:
                                st.error("图片处理失败")
                        except Exception as e:
                            st.error(f"分析错误: {e}")

    # Step 2: Confirm & Visualize
    elif st.session_state.fission_state["step"] == "analyzed":
        state = st.session_state.fission_state
        analysis = state["analysis"]
        
        st.divider()
        st.markdown("### 🎯 确认修改范围")
        
        c1, c2 = st.columns([1, 1])
        with c1:
            img_to_show = state["uploaded_file"]
            box_2d = analysis.get("box_2d")

            # Keep a human-in-the-loop escape hatch for low-confidence or
            # repeated text. Values are normalized to 0-1000 and immediately
            # become the source of truth for both preview and generation.
            if box_2d and isinstance(box_2d, list) and len(box_2d) == 4:
                locate_debug = analysis.get("locate_debug") or {}
                confidence = float(locate_debug.get("confidence", 0.55))
                confidence_label = "高" if confidence >= 0.85 else "中" if confidence >= 0.65 else "低"
                st.caption(f"定位置信度：{confidence_label}（{confidence:.0%}） · 红色区域是模型文字替换的定位参考")
                with st.expander("手动校准区域（定位不准时展开）", expanded=confidence < 0.65):
                    edit_cols = st.columns(4)
                    labels = ["左边界", "上边界", "右边界", "下边界"]
                    region_key = f"{state.get('payload', {}).get('filename', 'image')}_{analysis.get('target_text_content', 'target')}"
                    adjusted = []
                    for idx, (col, label) in enumerate(zip(edit_cols, labels)):
                        with col:
                            adjusted.append(st.number_input(
                                label, min_value=0, max_value=1000,
                                value=int(box_2d[idx]), step=2,
                                key=f"fission_box_{region_key}_{idx}",
                            ))
                    if adjusted[0] < adjusted[2] and adjusted[1] < adjusted[3]:
                        box_2d = [int(value) for value in adjusted]
                        analysis["box_2d"] = box_2d
                    else:
                        st.error("区域边界无效：左/上边界必须小于右/下边界。")

            # --- DEBUG: 默认折叠，不占主界面空间；点进去可看 box_2d / OCR / VLM 真实值 ---
            if box_2d:
                with st.expander("🔍 区域定位调试信息（OCR + VLM 双重验证）", expanded=False):
                    st.code(f"box_2d (归一化 0-1000): {box_2d}\n定位依据: {analysis.get('reason','未说明')}\n修改类型: {analysis.get('modification_type')}\n旧文案: {analysis.get('target_text_content')} → 新文案: {analysis.get('target_desc')}\nimg_path 传入: {state.get('payload',{}).get('img_path')}")

            # Preview the exact generated mask, including adaptive padding.
            draw_ok = False
            if box_2d and isinstance(box_2d, list) and len(box_2d) == 4:
                try:
                    if not isinstance(img_to_show, Image.Image):
                        img_to_show = Image.open(img_to_show)
                    base_rgba = img_to_show.convert("RGBA")
                    from fission import FissionAgent
                    preview_agent = FissionAgent()
                    mask, pixel_box = preview_agent.build_edit_mask(
                        state.get("payload", {}).get("img_path"),
                        box_2d,
                        invert_mask=analysis.get("modification_type") == "BG_EDIT",
                        is_text_mode=analysis.get("modification_type") == "TEXT_EDIT",
                    )
                    red = Image.new("RGBA", base_rgba.size, (255, 38, 38, 0))
                    red.putalpha(mask.point(lambda value: int(value * 0.28)))
                    draw_layer = Image.alpha_composite(base_rgba, red)
                    if pixel_box and analysis.get("modification_type") == "TEXT_EDIT":
                        draw = ImageDraw.Draw(draw_layer, "RGBA")
                        stroke_w = max(2, int(min(base_rgba.size) * 0.006))
                        draw.rectangle(pixel_box, outline=(255, 0, 0, 255), width=stroke_w)
                    img_to_show = draw_layer
                    draw_ok = True
                except Exception as _e:
                    st.error(f"⚠️ 定位区域预览失败（不影响生成）: {_e}")
            
            if not draw_ok:
                # box_2d 为空时兜底（避免用户什么框都看不到）
                try:
                    if not isinstance(img_to_show, Image.Image):
                        img_to_show = Image.open(img_to_show)
                    with st.expander("⚠️ 未精准定位，展示参考图（可继续执行生成）", expanded=True):
                        st.info(f"定位原因: {analysis.get('reason','未说明')} → 执行生成时仍会尝试精准修改")
                except Exception:
                    pass

            st.image(img_to_show, caption="最终重绘区域预览（红色半透明区域）", use_container_width=True)
        
        with c2:
            st.info(f"🧐 识别意图: {analysis['modification_type']}")
            st.markdown(f"**目标描述**: {analysis.get('target_desc') or analysis.get('refined_prompt')}")
            
            # Simulated Red Box / Mask Visualization
            if analysis['modification_type'] == "BG_EDIT":
                st.warning("⚠️ 将保留主体，替换背景区域 (红框示意范围)")
                # In a real app, we'd draw a box. Here we just show text.
                st.markdown("""
                <div style="border: 2px dashed red; padding: 10px; color: red; text-align: center;">
                   [ 模拟: 背景区域将被重绘 ]
                </div>
                """, unsafe_allow_html=True)
                
            elif analysis['modification_type'] == "TEXT_EDIT":
                 st.info("ℹ️ 识别到局部修改/文字处理需求")
                 st.markdown("""
                <div style="border: 2px dashed red; padding: 10px; color: red; text-align: center;">
                   [ AI 将参考红框位置执行精准文字替换 ]
                </div>
                """, unsafe_allow_html=True)
            else:
                 st.info("ℹ️ 复杂修改，将生成新变体")
            
            st.divider()
            
            col_conf, col_cancel = st.columns(2)
            with col_conf:
                if st.button("✅ 确认并执行", type="primary", use_container_width=True):
                    # Use status container for detailed step-by-step progress visualization
                    with st.status("🎨 正在进行智能微调...", expanded=True) as status:
                        if st.session_state.get("demo_mode"):
                            status.write("1. 加载 Fission 策略...")
                            time.sleep(2)
                            status.write("2. 正在应用 Visual Prompting (红框约束)...")
                            time.sleep(2)
                            status.write("3. 调用 Qwen-Image-Edit-Max 精准替换...")
                            time.sleep(2)
                        
                        # Keep Fission independent from the search/vector-store
                        # agent: a Qdrant lock must not block DashScope editing.
                        from fission import FissionAgent
                        fission_agent = state.get("fission_agent") or FissionAgent()
                        res = fission_agent.generate(state["payload"], analysis)
                        state["result"] = res
                        state["step"] = "generated"
                        
                        status.update(label="✅ 微调完成", state="complete", expanded=False)
                        time.sleep(1) # Brief pause to show completion
                        st.rerun()
            with col_cancel:
                if st.button("❌ 取消/重试", use_container_width=True):
                    st.session_state.fission_state["step"] = "init"
                    st.rerun()

    # Step 3: Result & Ingest
    elif st.session_state.fission_state["step"] == "generated":
        state = st.session_state.fission_state
        res = state["result"]
        
        if res.get("success"):
            st.success("✅ 微调完成！")
            if res.get("reason"):
                st.info(res.get("reason"))
            
            c1, c2 = st.columns(2)
            with c1:
                st.image(state["uploaded_file"], caption="原图")
            with c2:
                new_img = None
                load_err = None
                # ========== 支持三种结果源：本地路径 / file:// URL / http(s) URL ==========
                try:
                    local_path = res.get("local_path") or res.get("image_path")
                    img_url = res.get("image_url")

                    if local_path and os.path.exists(local_path):
                        new_img = Image.open(local_path).convert("RGBA")
                    elif img_url and isinstance(img_url, str) and img_url.startswith("file://"):
                        real_path = img_url.replace("file://", "", 1)
                        if os.path.exists(real_path):
                            new_img = Image.open(real_path).convert("RGBA")
                        else:
                            load_err = f"本地文件不存在: {real_path}"
                    elif img_url and isinstance(img_url, str) and (img_url.startswith("http://") or img_url.startswith("https://")):
                        import requests
                        from io import BytesIO
                        resp = requests.get(img_url, timeout=30)
                        if resp.status_code == 200:
                            new_img = Image.open(BytesIO(resp.content)).convert("RGBA")
                        else:
                            load_err = f"下载图片失败 HTTP {resp.status_code}"
                    else:
                        load_err = f"未找到可显示的图片源 (local_path={res.get('local_path')}, image_url={str(res.get('image_url',''))[:80]})"
                except Exception as _e_load:
                    load_err = f"加载结果图失败: {_e_load}"

                if new_img is not None:
                    st.image(new_img, caption="微调结果（新）")
                     
                    # Ingest Button
                    if st.button("📥 满意，入库", type="primary"):
                        new_filename = f"fission_{int(time.time())}.png"
                        current_uid = st.session_state.user.get('phone') if st.session_state.user else None
                        res_tuple = st.session_state.pipeline.process_image(new_img, filename=new_filename, user_id=current_uid)
                        if isinstance(res_tuple, tuple):
                            new_payload = res_tuple[0]
                        else:
                            new_payload = res_tuple

                        if new_payload:
                            if current_uid:
                                st.session_state.auth.increment_upload_count(current_uid)
                            st.success("入库成功！")
                            display_score(new_payload)
                            # Reset
                            col_restart, col_refine = st.columns(2)
                            with col_restart:
                                if st.button("✨ 开始新任务", use_container_width=True):
                                    st.session_state.fission_state = {"step": "init", "analysis": None, "payload": None}
                                    st.rerun()
                            with col_refine:
                                if st.button("🔄 不满意，重新调整", use_container_width=True):
                                    st.session_state.fission_state["step"] = "init"
                                    st.rerun()
                else:
                    # 结果加载失败：显示失败原因，并退回到原图大小的占位提示（不中断）
                    st.error(f"❌ 结果图加载失败: {load_err or '未知原因'}")
                    with st.expander("📦 生成结果原始 payload（便于定位）", expanded=False):
                        st.code({k: (str(v)[:200] if isinstance(v,str) else v) for k,v in res.items()})

                    col_restart2, col_refine2 = st.columns(2)
                    with col_restart2:
                        if st.button("✨ 开始新任务", key="restart_fallback", use_container_width=True):
                            st.session_state.fission_state = {"step": "init", "analysis": None, "payload": None}
                            st.rerun()
                    with col_refine2:
                        if st.button("🔄 不满意，重新调整", key="refine_fallback", use_container_width=True):
                            st.session_state.fission_state["step"] = "init"
                            st.rerun()
        else:
            st.error(f"失败: {res.get('reason')}")
            if st.button("重试"):
                st.session_state.fission_state["step"] = "init"
                st.rerun()

# --- Tab: RAG ---
with tab_rag:
    if is_empty_shop:
        _render_locked_tab("RAG智能生图")

    st.header("🧬 RAG 智能生图")
    st.caption("检索优秀素材基因，生成全新设计，并自动入库")
    
    rag_mode = st.radio("生成模式", ["自然语言生成", "基于产品图智能设计"], horizontal=True)
    
    if rag_mode == "自然语言生成":
        rag_query = st.text_area("设计需求", placeholder="例如: 夏季清凉饮料海报，蓝色调，极简风格", key="rag_query")
        
        if st.button("✨ 生成新图", type="primary", key="rag_btn"):
            if not st.session_state.get("system_ready"):
                st.error("系统未连接")
            elif st.session_state.agent is None:
                 st.error(f"Agent 初始化失败: {st.session_state.get('startup_error', '未知错误')}")
            elif not rag_query:
                st.warning("请输入设计需求")
            else:
                with st.status("🔄 正在执行 RAG 生成...", expanded=True) as status:
                    decision = {
                        "action": "GENERATE",
                        "query": rag_query
                    }
                    try:
                        status.write("🧬 1. 正在检索高分基因并生成...")
                        if st.session_state.get("demo_mode"):
                            time.sleep(3)
                            status.write("🧠 2. 融合用户 Prompt 与素材基因...")
                            time.sleep(2)
                            status.write("🎨 3. 调用生图模型 (Qwen-Image)...")
                            time.sleep(2)

                        res = st.session_state.agent.execute_decision(decision)
                        
                        if res.get("success"):
                            st.session_state.rag_result = res
                            status.update(label="✅ 生成成功", state="complete", expanded=False)
                        else:
                            status.update(label=f"❌ 生成失败: {res.get('reason')}", state="error")
                    except Exception as e:
                        status.update(label=f"❌ 错误: {e}", state="error")

    else: # 基于产品图智能设计
        st.info("💡 上传产品白底图或实拍图，AI自动提取主体并生成场景背景。")
        
        c1, c2 = st.columns([1, 1])
        with c1:
            product_file = st.file_uploader("上传产品图", type=["png", "jpg", "jpeg", "webp"], key="rag_prod_up")
            prod_img = None
            if product_file:
                prod_img = Image.open(product_file)
                st.image(prod_img, caption="产品原图", width=200)
        
        with c2:
            design_prompt = st.text_area("场景/背景描述 (可选)", placeholder="例如: 圣诞节氛围，雪地背景，温暖灯光\n(留空则由 AI 自动推荐高转化背景)", height=100)
            use_rag_factors = st.checkbox("启用 RAG 高转化因子", value=True, help="参考历史高点击率素材的风格和构图")
        
        # --- Step 1: Matting & Confirmation ---
        if "matting_result" not in st.session_state:
            st.session_state.matting_result = None

        if st.button("✂️ 1. 智能识别主体 (Extract Subject)", type="secondary", key="step1_matting"):
            if not prod_img:
                st.warning("请先上传产品图")
            else:
                 with st.status("正在智能识别并提取主体...", expanded=True) as status:
                     # Call detection then matting
                     from design_utils import smart_extract_product
                     try:
                        extracted, mask = smart_extract_product(prod_img)
                        st.session_state.matting_result = extracted
                        status.update(label="✅ 主体提取完成，请确认范围", state="complete", expanded=False)
                     except Exception as e:
                        status.update(label=f"❌ 提取失败: {e}", state="error")
                        st.error(f"提取失败: {e}")

        if st.session_state.matting_result:
            st.markdown("##### 确认提取结果")
            st.image(st.session_state.matting_result, caption="提取的主体 (透明背景)", width=200)
            st.caption("⚠️ 如果提取不完整（如包含手部），请重新上传更清晰的图片或尝试手动裁剪原图后上传。")
            
            # --- Step 2: Generation ---
            if st.button("🎨 2. 确认并生成场景 (Generate)", type="primary", key="rag_design_btn_confirm"):
                 if not st.session_state.get("system_ready"):
                    st.error("系统未连接")
                 else:
                    with st.status("🔄 正在进行智能设计...", expanded=True) as status:
                        try:
                            status.write("🧠 1. 构思高转化场景 (RAG Strategy)...")
                            if st.session_state.get("demo_mode"):
                                time.sleep(2)
                                
                            status.write("🎨 2. 生成背景并融合 (Composition)...")
                            
                            # Call generator with PRE-MATTED image
                            # We need to modify the generator to accept pre-matted image or bypass matting
                            # For now, we pass the original and let it re-mat OR pass the matted one if supported.
                            # Actually, rag_generator.generate_composite_design usually does matting internally.
                            # We should pass the matted image as 'base_image' and tell it NOT to mat again?
                            # Or update generate_composite_design to check alpha channel.
                            
                            # Hack: If image is RGBA, assume matted?
                            # Let's pass the matted image from session state.
                            matted_img = st.session_state.matting_result
                            
                            res = st.session_state.agent.rag_generator.generate_composite_design(
                                base_image=matted_img, # Pass the matted one
                                user_query=design_prompt,
                                use_rag=use_rag_factors,
                                skip_matting=True # We need to add this param or logic
                            )
                            
                            if res.get("success"):
                                st.session_state.rag_result = res
                                status.update(label="✅ 设计完成", state="complete", expanded=False)
                            else:
                                status.update(label=f"❌ 设计失败: {res.get('reason')}", state="error")
                                
                        except Exception as e:
                            status.update(label=f"❌ 错误: {e}", state="error")
                            st.error(f"Detail: {e}")

    # Preview & Ingest Section
    if "rag_result" in st.session_state:
        res = st.session_state.rag_result
        path = res.get("image_path") or res.get("image_url")
        
        # Load Image
        img = None
        if path and os.path.exists(path):
            img = Image.open(path)
        elif res.get("image_url"):
            import requests
            from io import BytesIO
            resp = requests.get(res["image_url"])
            if resp.status_code == 200:
                img = Image.open(BytesIO(resp.content))
        
        if img:
            st.image(img, caption="生成结果", width=400)
            
            # --- New: Display Smart Design Factors ---
            m_factors = res.get("marketing_factors")
            if m_factors:
                st.markdown("### 🧠 智能设计策略 (The Brain)")
                st.caption("基于高转化因子自动生成的营销要素")
                
                mf1, mf2, mf3 = st.columns(3)
                with mf1:
                    st.info(f"📋 **营销文案**\n\n{m_factors.get('copy', '无')}")
                with mf2:
                    st.info(f"🎨 **文案样式**\n\n{m_factors.get('style', '无')}")
                with mf3:
                    st.info(f"📍 **建议位置**\n\n{m_factors.get('position', '无')}")
                
                st.divider()
            # -----------------------------------------

            if res.get("references"):
                st.info(f"参考基因来源: {', '.join(res['references'])}")
            
            c1, c2 = st.columns(2)
            with c1:
                if st.button("📥 确认入库", type="primary", key="rag_ingest"):
                     with st.status("📥 正在全链路入库...", expanded=True) as status:
                        status.write("2. 正在全链路入库 (打标/评分/向量化)...")
                        if st.session_state.get("demo_mode"):
                            time.sleep(3)
                        
                        new_filename = f"rag_{int(time.time())}.png"
                        # Handle tuple return
                        current_uid = st.session_state.user.get('phone') if st.session_state.user else None
                        res_tuple = st.session_state.pipeline.process_image(img, filename=new_filename, user_id=current_uid)
                        if isinstance(res_tuple, tuple):
                             new_payload = res_tuple[0]
                        else:
                             new_payload = res_tuple
                        
                        if new_payload:
                            if current_uid:
                                st.session_state.auth.increment_upload_count(current_uid)
                            status.update(label="✅ 入库完成！", state="complete", expanded=False)
                            st.success("🎉 RAG生成素材已入库！")
                            display_score(new_payload)
                            # Clear state
                            del st.session_state.rag_result
                            time.sleep(2)
                            st.rerun()
                        else:
                            status.update(label="❌ 入库失败", state="error")
            with c2:
                if st.button("🗑️ 放弃结果"):
                    del st.session_state.rag_result
                    st.rerun()
        else:
             st.error("❌ 无法加载生成结果")


# --- Tab 2: Search ---
if False: # tab_search (Hidden)
    col1, col2 = st.columns([5, 1])
    with col1:
        query = st.text_input(
            "搜索", 
            placeholder="描述你的需求... (例如: '红色促销横幅' 或 '夏季清爽风格')",
            label_visibility="collapsed"
        )
    with col2:
        search_btn = st.button("搜索", type="primary", use_container_width=True)

    if search_btn and query:
        if not st.session_state.get("system_ready", False):
            st.error(f"系统未连接: {st.session_state.get('startup_error', 'Unknown Error')}")
        elif st.session_state.agent is None:
             st.error(f"Agent 初始化失败: {st.session_state.get('startup_error', '未知错误')}")
        else:
            with st.spinner("🧠 Agent 正在分析意图并检索..."):
                try:
                    results, intent = st.session_state.agent.search(query)
                    
                    # 1. Display Agent Thought Process
                    with st.expander("🤖 Agent 思考过程 (Thought Process)", expanded=True):
                        st.markdown(f"**策略选择**: `{intent['strategy']}`")
                        if intent['strategy'] == 'L1':
                            st.write(f"🎯 识别到具体属性: {intent.get('extracted_keywords')}")
                            st.info("💡 策略: 使用 L1 关键词精准过滤")
                        elif intent['strategy'] == 'L2':
                            st.write(f"🌊 识别为语义描述")
                            st.info("💡 策略: 使用 L2 向量语义检索，寻找概念匹配的素材")
                        elif intent['strategy'] == 'L3':
                            st.write(f"🧠 识别为复杂/模糊需求")
                            st.info("💡 策略: 使用 L3 推理链 (CoT)，先扩展查询词再进行检索")
                            st.caption(f"扩展后的查询: '{intent.get('search_query')} marketing high conversion'")

                    st.divider()

                    if not results:
                        st.info("👋 未找到相关素材，尝试上传一些图片？")
                    else:
                        st.subheader(f"找到 {len(results)} 个结果")
                        
                        # Translation Map
                        tag_map = {
                            "material_purpose": "用途",
                            "visual_style": "风格",
                            "target_audience": "受众",
                            "core_content": "核心内容"
                        }

                        # Vertical List Layout
                        for idx, item in enumerate(results):
                            payload = item['payload']
                            score = item['score']
                            filename = payload.get('filename')
                            
                            with st.container():
                                st.markdown("---")
                                col_img, col_info = st.columns([1, 2])
                                
                                # Left: Image
                                with col_img:
                                    img_path = os.path.join("AI素材案例", filename)
                                    if os.path.exists(img_path):
                                        st.image(img_path, width=250)
                                    else:
                                        st.warning("🖼️ 图片不可用")
                                        
                                # Right: Info
                                with col_info:
                                    st.markdown(f"### 📄 {filename}")
                                    st.caption(f"🔥 匹配度: {int(score*100)}%")
                                    
                                    # Tags Display (Chinese Keys)
                                    tags = payload.get('tags', {})
                                    if isinstance(tags, str):
                                        import json
                                        try: tags = json.loads(tags)
                                        except: pass
                                    
                                    # Display Performance Data if available (Agent Feature)
                                    display_score(payload)

                                    if isinstance(tags, dict):
                                        for en_key, val in tags.items():
                                            if en_key in ["predicted_ctr", "predicted_cvr", "historical_impressions"]:
                                                continue # Skip displaying raw metrics in tag list
                                            cn_key = tag_map.get(en_key, en_key)
                                            # Truncate long values for preview
                                            preview_val = str(val)[:100] + "..." if len(str(val)) > 100 else str(val)
                                            st.markdown(f"**{cn_key}:** {preview_val}")
                                    
                                    # Details Expander
                                    with st.expander("🔍 查看完整详情"):
                                        st.markdown("#### 完整标签")
                                        st.json(tags)
                                        st.markdown("#### 视觉描述")
                                        st.write(payload.get('caption', '无描述'))
                                        st.markdown("#### OCR 文字")
                                        st.text_area("OCR Content", payload.get('ocr_text', ''), height=100, disabled=True, key=f"ocr_{idx}")

                                    # Fission & Feedback
                                    st.markdown("#### ⚡️ 智能操作")
                                    f1, f2, f3 = st.columns([3, 1, 1])
                                    with f1:
                                        with st.expander("🛠️ 智能微调 (Fission)"):
                                            fission_input = st.text_input("修改指令", placeholder="如: 换成红色背景", key=f"fis_in_{idx}")
                                            if st.button("执行", key=f"fis_btn_{idx}"):
                                                if not fission_input:
                                                    st.error("请输入修改指令")
                                                else:
                                                    with st.spinner("正在进行智能微调..."):
                                                        decision_req = {
                                                            "action": "FISSION",
                                                            "reference_material_id": item['id'],
                                                            "query": fission_input
                                                        }
                                                        try:
                                                            res = st.session_state.agent.execute_decision(decision_req)
                                                            if res.get("success"):
                                                                st.success("微调成功！请保存结果。")
                                                                st.image(res["image_url"], caption=f"微调结果: {fission_input}")
                                                            else:
                                                                st.error(f"微调失败: {res.get('reason')}")
                                                        except Exception as e:
                                                            st.error(f"Error: {e}")

                                    with f2:
                                        if st.button("👍", key=f"like_{idx}", help="有用"):
                                            st.session_state.feedback.record(item['id'], "like", query)
                                            st.toast("已反馈：有用")
                                    with f3:
                                         if st.button("👎", key=f"dislike_{idx}", help="无用"):
                                            st.session_state.feedback.record(item['id'], "dislike", query)
                                            st.toast("已反馈：无用")

                except Exception as e:
                    st.error(f"搜索出错: {e}")

# --- Tab 3: Strategy Dashboard ---
with tab_dashboard:
    if is_empty_shop:
        _render_locked_tab("决策看板")

    st.header("📊 营销策略洞察看板")
    st.caption("基于全量素材的归因分析 (Attribution Analysis)")
    
    # Auto-load on tab entry is hard in Streamlit, so use a button or session state check
    # We'll use a button for explicit action
    if st.button("🔄 刷新数据", type="primary"):
        # Fix: Ensure Agent
        if st.session_state.agent is None:
             try:
                 st.session_state.agent = MaterialAgent()
             except: pass

        with st.spinner("正在聚合分析全量数据..."):
            try:
                # User Isolation
                dash_user_id = st.session_state.user.get('phone')
                if st.session_state.user.get('role') == 'admin':
                    dash_user_id = None

                analysis = st.session_state.agent.analyze_assets(user_id=dash_user_id)
                
                if "error" in analysis:
                    st.error(f"分析失败: {analysis['error']}")
                else:
                    tag_perf = analysis["tag_performance"]
                    insights = analysis["insights"]
                    global_ctr = analysis["global_avg_ctr"]
                    
                    # 1. Key Metrics
                    m1, m2 = st.columns(2)
                    m1.metric("全局平均 CTR", f"{global_ctr:.2%}")
                    m1.metric("分析素材样本数", f"{analysis.get('sample_count', 0)}")
                    
                    if not tag_perf.empty:
                        best = tag_perf.iloc[0]
                        if global_ctr > 0:
                            delta = (best['avg_ctr']-global_ctr)/global_ctr
                        else:
                            delta = 0
                        m2.metric("🏆 最佳风格", best['tag'], delta=f"{delta:.0%}")
                    
                    st.divider()
                    
                    # 2. Insights
                    st.subheader("💡 智能策略发现")
                    for insight in insights:
                        st.info(insight)
                        
                    if not insights:
                        st.info("数据量不足，暂无显著策略发现。")
                        
                    st.divider()
                    
                    # 3. Visualization
                    st.subheader("📈 风格标签效能归因")
                    if not tag_perf.empty:
                        chart = alt.Chart(tag_perf).mark_bar().encode(
                            x=alt.X('tag:N', axis=alt.Axis(labelAngle=0, labelLimit=160)),
                            y=alt.Y('avg_ctr:Q', title='平均CTR'),
                            tooltip=['tag', alt.Tooltip('avg_ctr:Q', format='.2%'), 'count']
                        ).properties(height=320)
                        st.altair_chart(chart, use_container_width=True)
                    
                    # Detailed Data
                    with st.expander("查看详细数据"):
                        st.dataframe(tag_perf.style.format({"avg_ctr": "{:.2%}"}))
                        
                    st.divider()
                    
                    # 4. Simulation Logs Visualization
                    st.subheader("👥 虚拟用户仿真评测记录")
                    st.caption("基于 User Persona Agent 的批量仿真数据")
                    
                    sim_csv_path = "simulation_results.csv"
                    if os.path.exists(sim_csv_path):
                        import pandas as pd
                        sim_df = pd.read_csv(sim_csv_path)
                        
                        # Metrics Summary
                        s1, s2, s3 = st.columns(3)
                        s1.metric("仿真样本数", len(sim_df))
                        s2.metric("平均模拟点击率", f"{sim_df['simulated_ctr'].mean():.2%}")
                        s3.metric("平均综合得分", f"{sim_df['avg_score'].mean():.1f}/10")
                        
                        # Persona Preference Chart
                        # Melt for chart
                        try:
                            persona_scores = sim_df[['filename', 'junior_score', 'senior_score', 'growth_score']]
                            persona_melted = persona_scores.melt('filename', var_name='Persona', value_name='Score')
                            
                            chart_p = alt.Chart(persona_melted).mark_circle(size=100).encode(
                                x=alt.X('Persona', axis=alt.Axis(title='角色')),
                                y=alt.Y('Score', scale=alt.Scale(domain=[0, 10])),
                                color='Persona',
                                tooltip=['filename', 'Score', 'Persona']
                            ).properties(height=300, title="不同角色的评分分布")
                            st.altair_chart(chart_p, use_container_width=True)
                            
                            # Detailed Table
                            with st.expander("📄 查看完整评测日志"):
                                # Format boolean columns
                                display_df = sim_df.copy()
                                if 'simulated_ctr' in display_df.columns:
                                    display_df['simulated_ctr'] = display_df['simulated_ctr'].apply(lambda x: f"{x:.2%}")
                                if 'consistency' in display_df.columns:
                                    st.markdown("##### 👥 角色一致性分析")
                                    consistency_counts = display_df['consistency'].value_counts()
                                    st.bar_chart(consistency_counts)
                                
                                st.dataframe(display_df)
                        except Exception as e:
                            st.error(f"图表渲染错误: {e}")
                    else:
                        st.info("暂无仿真数据，请运行 `src/simulation.py` 生成数据。")
                    
                    st.divider()
                    
                    # 5. High Conversion Factor Insights (Merged from Batch Tab)
                    st.subheader("🏆 高转化因子洞察")
                    st.caption("基于全库素材的品类与转化因子分析")
                    
                    # Category Filter
                    # Use a cached way to get categories if possible, or fetch efficiently
                    mats_insight = st.session_state.agent.list_materials(limit=1000) 
                    
                    categories = set()
                    if mats_insight:
                         for m in mats_insight:
                             t = m['payload'].get('tags', {})
                             if isinstance(t, str):
                                 import json
                                 try: t = json.loads(t)
                                 except: t = {}
                             if isinstance(t, dict) and t.get('product_category'):
                                 categories.add(t['product_category'])
                    
                    selected_cat_insight = st.selectbox("选择品类 (Product Category)", ["全部"] + list(categories), key="insight_cat_filter")
                    
                    if st.button("🔄 刷新洞察", key="refresh_insight_tab3"):
                        if mats_insight:
                            data_insight = []
                            for m in mats_insight:
                                p = m['payload']
                                tags = p.get('tags', {})
                                if isinstance(tags, str):
                                    import json
                                    try: tags = json.loads(tags)
                                    except: tags = {}
                                
                                # Filter by Category
                                cat = tags.get('product_category', 'Unknown')
                                if selected_cat_insight != "全部" and cat != selected_cat_insight:
                                    continue
                                    
                                score = p.get('predicted_score', 0)
                                if score > 0:
                                    data_insight.append({
                                        "category": cat,
                                        "style": tags.get('copy_style', 'Unknown'),
                                        "position": tags.get('copy_position', 'Unknown'),
                                        "scene": tags.get('scene_type', 'Unknown'),
                                        "copy": tags.get('marketing_copy', ''),
                                        "filename": p.get('filename', ''),
                                        "score": score
                                    })
                            
                            if data_insight:
                                df_insight = pd.DataFrame(data_insight)
                                st.caption(f"分析样本数: {len(df_insight)}")
                                
                                # Charts
                                c_i1, c_i2 = st.columns(2)
                                
                                with c_i1:
                                    # Best Scene
                                    avg_scene = df_insight.groupby("scene")["score"].mean().reset_index().sort_values("score", ascending=False).head(5)
                                    st.markdown("**🏆 最佳背景 (Scene)**")
                                    chart_scene = alt.Chart(avg_scene).mark_bar().encode(
                                        x=alt.X('scene', sort='-y', axis=alt.Axis(labelLimit=100)),
                                        y='score',
                                        color=alt.value('#ffc107'),
                                        tooltip=['scene', 'score']
                                    ).properties(height=200)
                                    st.altair_chart(chart_scene, use_container_width=True)

                                with c_i2:
                                    # Best Style
                                    avg_style = df_insight.groupby("style")["score"].mean().reset_index().sort_values("score", ascending=False).head(5)
                                    st.markdown("**✍️ 最佳文案风格 (Style)**")
                                    chart_style = alt.Chart(avg_style).mark_bar().encode(
                                        x=alt.X('style', sort='-y', axis=alt.Axis(labelLimit=100)),
                                        y='score',
                                        color=alt.value('#28a745'),
                                         tooltip=['style', 'score']
                                    ).properties(height=200)
                                    st.altair_chart(chart_style, use_container_width=True)
                                
                                # Best Position
                                avg_pos = df_insight.groupby("position")["score"].mean().reset_index().sort_values("score", ascending=False)
                                st.markdown("**📍 最佳文案位置 (Position)**")
                                chart_pos = alt.Chart(avg_pos).mark_bar().encode(
                                    x=alt.X('position', sort='-y'),
                                    y='score',
                                    color=alt.value('#17a2b8'),
                                    tooltip=['position', 'score']
                                ).properties(height=150)
                                st.altair_chart(chart_pos, use_container_width=True)

                                # Top Copies Table
                                st.markdown("**📝 最佳营销文案 (Top Copies)**")
                                top_copies = df_insight[['filename', 'copy', 'score']].sort_values('score', ascending=False).head(10)
                                st.dataframe(top_copies, use_container_width=True, hide_index=True)
                            else:
                                st.info(f"暂无 {selected_cat_insight} 品类的评分数据")
                        else:
                            st.info("暂无素材")

            except Exception as e:
                st.error(f"系统错误: {e}")
                st.exception(e)
    else:
        st.info("点击上方按钮开始分析")

# --- Tab 4: Upload ---
with tab_upload:
    if is_empty_shop:
        _render_locked_tab("素材入库与批量处理")

    st.header("📤 素材入库")
    
    uploaded_files = st.file_uploader("选择图片素材", accept_multiple_files=True, type=['png', 'jpg', 'jpeg', 'webp'])
    
    if st.button("开始入库", type="primary") and uploaded_files:
        if not st.session_state.get("system_ready", False):
            st.error("系统未连接")
        else:
            # Progress bar
            progress_bar = st.progress(0)
            success_count = 0
            
            for i, uploaded_file in enumerate(uploaded_files):
                try:
                    # Save temp
                    temp_path = os.path.join("AI素材案例", uploaded_file.name)
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    
                    st.caption(f"正在处理: {uploaded_file.name} ...")
                    
                    # Full Pipeline Visualization for Upload
                    with st.status(f"🔄 处理中: {uploaded_file.name}", expanded=True) as status:
                        st.write("1. 👁️ 视觉理解与OCR提取...")
                        st.write("2. 🧠 智能打标 (Tagging)...")
                        st.write("3. 📈 效果预测 (CTR Simulation)...")
                        st.write("4. 💾 向量化入库...")
                        
                        image = Image.open(temp_path)
                        # Process
                    # Fix: Handle tuple return explicitly
                    current_uid = st.session_state.user.get('phone') if st.session_state.user else None
                    res = st.session_state.pipeline.process_image(image, filename=uploaded_file.name, user_id=current_uid)
                    is_success = False
                    if isinstance(res, tuple):
                         # (payload, vector)
                         if res[0] is not None:
                             is_success = True
                    elif res:
                         is_success = True
                         
                    if is_success:
                        if current_uid:
                            st.session_state.auth.increment_upload_count(current_uid)
                        success_count += 1
                        status.update(label=f"✅ {uploaded_file.name} 入库成功", state="complete", expanded=False)
                    else:
                        status.update(label=f"❌ {uploaded_file.name} 处理失败", state="error")
                
                except Exception as e:
                    st.error(f"Error processing {uploaded_file.name}: {e}")
                
                progress_bar.progress((i + 1) / len(uploaded_files))
                
            if success_count == len(uploaded_files):
                st.success(f"🎉 全部 {success_count} 张素材入库完成！")
            else:
                st.warning(f"完成，但有 {len(uploaded_files) - success_count} 张失败。")

    st.divider()
    st.header("⚡️ 素材批量处理")
    st.caption("一键管理素材全览中的图片，执行入库、打标与评分。")
    
    # --- One-Click Batch Processing (Background Task Mode) ---
    st.info("💡 点击下方按钮，系统将扫描 `AI素材案例` 目录，启动后台任务对所有未处理图片进行智能打标与评分。您可以切换到其他页面继续工作。")
    
    bp = st.session_state.batch_processor
    
    # Control Panel
    c_start, c_pause, c_stop = st.columns([1, 1, 1])
    
    is_running = bp.progress["status"] == "running"
    is_paused = bp.progress["status"] == "paused"
    
    with c_start:
        if bp.progress["status"] in ["idle", "completed", "stopped", "error"]:
            if st.button("🚀 启动后台任务", type="primary", key="start_batch_bg"):
                # Check prerequisites
                if not st.session_state.get("system_ready"):
                    st.error("系统未连接")
                elif st.session_state.agent is None:
                    st.error("Agent未初始化")
                else:
                    # Init pipeline if needed
                    if st.session_state.pipeline is None:
                        st.session_state.pipeline = MaterialPipeline()
                        
                    current_uid = st.session_state.user.get('phone') if st.session_state.user else None
                    bp.start(
                        pipeline=st.session_state.pipeline,
                        agent=st.session_state.agent,
                        auth_manager=st.session_state.auth,
                        user_id=current_uid,
                        img_dir="AI素材案例"
                    )
                    # st.success("后台任务已启动！您可以去其他 Tab 操作。") # User requested removal
                    time.sleep(1)
                    st.rerun()
        else:
            st.button("🚀 运行中...", disabled=True)

    with c_pause:
        if is_running:
            if st.button("⏸ 暂停", key="pause_batch_bg"):
                bp.pause()
                st.rerun()
        elif is_paused:
            if st.button("▶️ 继续", key="resume_batch_bg"):
                bp.resume()
                st.rerun()
        else:
            st.button("⏸ 暂停", disabled=True)
            
    with c_stop:
        if is_running or is_paused:
            if st.button("⏹ 停止", type="secondary", key="stop_batch_bg"):
                bp.stop()
                st.rerun()
        else:
             st.button("⏹ 停止", disabled=True)

    # Status Dashboard (Live Update)
    if bp.progress["status"] != "idle":
        st.divider()
        status_container = st.container(border=True)
        
        p = bp.progress
        total = p["total"]
        processed = p["processed"]
        
        # Progress Bar
        progress_val = 0.0
        if total > 0:
            progress_val = processed / total
            
        status_container.progress(progress_val)
        
        # Metrics
        m1, m2, m3, m4 = status_container.columns(4)
        m1.metric("状态", "运行中" if is_running else "暂停" if is_paused else p["status"])
        m2.metric("进度", f"{processed}/{total}")
        m3.metric("成功", p["success_count"])
        m4.metric("失败", p["fail_count"])
        
        # Current File
        if p["current_file"]:
            status_container.caption(f"🔄 正在处理: {p['current_file']}")
            
        # Logs
        with status_container.expander("查看日志", expanded=False):
            for log in p["logs"]:
                st.text(log)
                
        # Auto-refresh logic (only if running)
        # Removed auto-refresh loop to prevent blocking UI/spinner
        if is_running:
            if st.button("🔄 刷新状态"):
                st.rerun()
            st.caption("后台任务正在进行中，您可以点击上方按钮手动刷新状态。")
                
    elif bp.progress["status"] == "completed":
        st.success(f"✅ 批量任务完成！成功: {bp.progress['success_count']}, 失败: {bp.progress['fail_count']}")


    st.divider()

    # Image Preview Modal
    if "view_image_path" in st.session_state:
        with st.container():
            st.info("🔍 高清大图预览")
            if st.button("❌ 关闭预览", key="close_preview", type="primary"):
                del st.session_state.view_image_path
                st.rerun()
            
            if os.path.exists(st.session_state.view_image_path):
                # Use full width
                st.image(st.session_state.view_image_path, caption=os.path.basename(st.session_state.view_image_path), use_container_width=True)
            else:
                st.error("图片文件不存在")
            st.divider()

# --- Tab 5: Material Management (Empty) ---
with tab_tags:
    if is_empty_shop:
        _render_locked_tab("素材标签数据")

    st.header("🏷️ 素材标签数据")
    st.caption("查看素材标签、预测分及真实CTR数据")
    
    with st.expander("🔧 批量上传CTR数据", expanded=False):
        st.caption("批量上传 (格式: 文件名,CTR)")
        ctr_file = st.file_uploader("上传 CSV/TXT", type=['csv', 'txt'], key="ctr_uploader_tab")
        if ctr_file:
            if st.button("开始导入", key="ctr_import_btn"):
                try:
                    content = ctr_file.getvalue().decode("utf-8")
                    lines = content.splitlines()
                    success_count = 0
                    for line in lines:
                        if "," in line:
                            parts = line.split(",")
                            if len(parts) >= 2:
                                fname = parts[0].strip()
                                try:
                                    ctr_val = float(parts[1].strip())
                                    if st.session_state.agent.update_material_ctr(fname, ctr_val):
                                        success_count += 1
                                except:
                                    pass
                    if success_count > 0:
                        st.success(f"成功更新 {success_count} 条数据")
                        time.sleep(1)
                        st.rerun()
                    else:
                        st.warning("未找到有效数据或更新失败")
                except Exception as e:
                    st.error(f"Error: {e}")
    
    st.divider()
    
    # Helper to get materials
    # Fix: Ensure Agent
    if 'agent' not in st.session_state or st.session_state.agent is None:
         try:
             st.session_state.agent = MaterialAgent()
         except: pass

    if st.session_state.agent:
        # User Data Isolation for List
        list_user_id = st.session_state.user.get('phone')
        if st.session_state.user.get('role') == 'admin':
            list_user_id = None # See all
            
        materials = st.session_state.agent.list_materials(limit=50, user_id=list_user_id)
        
        if not materials:
            st.info("暂无素材数据")
        else:
            # Header
            c1, c2, c3, c4 = st.columns([2, 2, 1, 1])
            c1.markdown("**🖼️ 图片 & 标题**")
            c2.markdown("**🏷️ 标签 (Tags)**")
            c3.markdown("**📈 预测分**")
            c4.markdown("**📊 真实CTR**")
            st.divider()
            
            for m in materials:
                p = m['payload']
                filename = p.get('filename', 'Unknown')
                score = p.get('predicted_score', 0)
                
                tags = p.get('tags', {})
                if isinstance(tags, str):
                    import json
                    try: tags = json.loads(tags)
                    except: tags = {}
                
                # Display Row
                with st.container():
                    rc1, rc2, rc3, rc4 = st.columns([2, 2, 1, 1])
                    
                    with rc1:
                        img_path = os.path.join("AI素材案例", filename)
                        if os.path.exists(img_path):
                            rc1.image(img_path, width=100)
                            if st.button("🔍 放大", key=f"view_{m['id']}", help="点击查看高清大图"):
                                st.session_state.view_image_path = img_path
                                st.rerun()
                        rc1.caption(filename)
                        
                    with rc2:
                        if isinstance(tags, dict):
                            # Helper to format tag values (handle lists)
                            def fmt(v):
                                if isinstance(v, list):
                                    return " ".join(str(x) for x in v)
                                return str(v) if v else "-"

                            # Phase 1: Visual
                            st.caption("👁️ 视觉感知")
                            t1 = f"风格: `<span style='color:#e0e0e0'>{fmt(tags.get('visual_style'))}</span>` | 色彩: `<span style='color:#e0e0e0'>{fmt(tags.get('color_tone'))}</span>` | 构图: `<span style='color:#e0e0e0'>{fmt(tags.get('composition'))}</span>`"
                            st.markdown(t1, unsafe_allow_html=True)
                            
                            # Phase 2: Marketing
                            st.caption("🧠 营销策略")
                            t2 = f"场景: `<span style='color:#e0e0e0'>{fmt(tags.get('scene_type'))}</span>` | 受众: `<span style='color:#e0e0e0'>{fmt(tags.get('target_audience'))}</span>` | 利益: `<span style='color:#e0e0e0'>{fmt(tags.get('key_benefit'))}</span>`"
                            st.markdown(t2, unsafe_allow_html=True)

                            # Description (Always Visible)
                            if tags.get('description'):
                                st.caption("📝 融合描述")
                                st.markdown(f"**{tags.get('description')}**")
                        
                    with rc3:
                        color = "green" if score > 20 else "orange" if score > 10 else "red"
                        rc3.markdown(f":{color}[**{score}**]")
                        
                    with rc4:
                        real_ctr = p.get('real_ctr')
                        val = float(real_ctr) if real_ctr is not None else 0.0
                        
                        # Display as Percentage String (e.g., "1.23%")
                        display_val = f"{val * 100:.2f}%"

                        def update_ctr_callback(mid=m['id'], fname=filename):
                            new_val_str = st.session_state.get(f"ctr_{mid}")
                            if new_val_str:
                                try:
                                    # Strip % and convert
                                    clean_val = new_val_str.replace('%', '').strip()
                                    float_val = float(clean_val)
                                    # Convert back to decimal (1.23 -> 0.0123)
                                    decimal_val = float_val / 100.0
                                    
                                    st.session_state.agent.update_material_ctr(fname, decimal_val)
                                    st.toast(f"Updated CTR for {fname} to {decimal_val:.4f}")
                                except ValueError:
                                    st.toast("Invalid CTR format. Please enter a number (e.g. 1.5)", icon="⚠️")

                        st.text_input(
                            "真实CTR", 
                            value=display_val,
                            key=f"ctr_{m['id']}",
                            on_change=update_ctr_callback,
                            label_visibility="collapsed"
                        )
                        
                    st.divider()
    else:
        st.error("Agent not initialized")

# --- Tab 7: User Management (Admin Only) ---
if tab_users:
    with tab_users:
        # Check for Admin Redirect (Gallery View)
        if st.session_state.get("admin_view_gallery", False):
            c_back, c_title = st.columns([1, 5])
            with c_back:
                if st.button("⬅️ 返回用户列表", key="btn_back_users"):
                    st.session_state.admin_view_gallery = False
                    st.rerun()
            
            # Show Gallery
            display_gallery()
            
        else:
            st.header("👥 用户管理 (管理员)")
            st.caption("查看所有用户及其管理素材统计 (点击数字查看详情)")
            
            users = st.session_state.auth.get_all_users()
            if users:
                import pandas as pd
                
                # Header
                h1, h2, h2b, h3, h4 = st.columns([2, 1, 1, 1, 2])
                h1.markdown("**用户账号**")
                h2.markdown("**权限角色**")
                h2b.markdown("**白名单**")
                h3.markdown("**管理素材数**")
                h4.markdown("**注册时间**")
                st.divider()
                
                # Sort: Admin first, then by time desc
                users.sort(key=lambda x: (x.get('role') != 'admin', x.get('created_at', 0)), reverse=False)
                
                for u in users:
                    with st.container():
                        c1, c2, c2b, c3, c4 = st.columns([2, 1, 1, 1, 2])
                        
                        # Account
                        c1.write(u.get('masked_phone'))
                        
                        # Role
                        role = u.get('role', 'user')
                        if role == 'admin':
                            c2.markdown("🛡️ **管理员**")
                        else:
                            c2.write("普通用户")
                        
                        # Whitelist
                        is_wl = u.get('whitelist', False)
                        if role != 'admin':
                            new_wl = c2b.checkbox("白名单", value=is_wl, key=f"wl_{u['phone']}", label_visibility="collapsed")
                            if new_wl != is_wl:
                                st.session_state.auth.toggle_whitelist(u['phone'], new_wl)
                                st.rerun()
                        else:
                            c2b.markdown("✅")
                            
                        # Count (Clickable)
                        # count = u.get('upload_count', 0) 
                        # Fix: Get real count from Qdrant
                        count = 0
                        if st.session_state.agent:
                            count = st.session_state.agent.get_user_material_count(u['phone'])
                        
                        # Use a unique key
                        if c3.button(f"{count}", key=f"btn_cnt_{u['phone']}"):
                            if role == 'admin':
                                # Redirect to Gallery View
                                st.session_state.admin_view_gallery = True
                                st.rerun()
                            else:
                                # Open Modal
                                # Reset state for this user to ensure fresh load
                                if f"mats_{u['phone']}" in st.session_state:
                                    del st.session_state[f"mats_{u['phone']}"]
                                if f"offset_{u['phone']}" in st.session_state:
                                    del st.session_state[f"offset_{u['phone']}"]
                                    
                                show_user_materials_modal(u['phone'], u.get('masked_phone'))
                            
                        # Time (UTC+8 Fix)
                        ts = u.get('created_at')
                        if ts:
                            try:
                                # Ensure float
                                ts_val = float(ts)
                                # Assuming ts is timestamp, convert to UTC+8
                                dt = pd.to_datetime(ts_val, unit='s', utc=True).tz_convert('Asia/Shanghai')
                                c4.write(dt.strftime('%Y-%m-%d %H:%M:%S'))
                            except Exception as e:
                                c4.caption(f"Error: {ts}")
                        else:
                            c4.write("-")
                        
                        st.markdown("<hr style='margin: 5px 0; border-top: 1px solid #333;'>", unsafe_allow_html=True)

                        # --- Row 2: Admin Operations (Delete / Reset Password / Change Role) ---
                        if role != 'admin' or u['phone'] != '13800138000':
                            with st.container():
                                opc1, opc2, opc3, opc4 = st.columns([1.2, 1.2, 1.5, 2.1])

                                with opc1:
                                    with st.popover("🔑 重置密码", disabled=u['phone'] == '13800138000'):
                                        new_pwd = st.text_input("新密码（至少 6 位）", type="password", key=f"rpwd_txt_{u['phone']}")
                                        if st.button("确认重置", key=f"rpwd_confirm_{u['phone']}", type="primary"):
                                            ok, msg = st.session_state.auth.reset_password(u['phone'], new_pwd, st.session_state.user.get('role'))
                                            if ok:
                                                st.success(msg)
                                                st.rerun()
                                            else:
                                                st.error(msg)

                                with opc2:
                                    target_role = "user" if role != "designer" else "designer"
                                    if st.button(f"↔️ 转为{('设计岗' if target_role=='designer' else '运营岗')}",
                                                 key=f"chrole_{u['phone']}", disabled=u['phone'] == '13800138000'):
                                        ok, msg = st.session_state.auth.change_role(u['phone'], target_role, st.session_state.user.get('role'))
                                        if ok:
                                            st.success(msg)
                                            st.rerun()
                                        else:
                                            st.error(msg)

                                with opc3:
                                    if role != 'admin' and st.button("🛡️ 设为管理员", key=f"mkadmin_{u['phone']}"):
                                        ok, msg = st.session_state.auth.change_role(u['phone'], "admin", st.session_state.user.get('role'))
                                        if ok:
                                            st.success(msg)
                                            st.rerun()
                                        else:
                                            st.error(msg)
                                    elif role == 'admin' and u['phone'] != '13800138000' and st.button("⬇️ 降级运营", key=f"rmadmin_{u['phone']}"):
                                        ok, msg = st.session_state.auth.change_role(u['phone'], "user", st.session_state.user.get('role'))
                                        if ok:
                                            st.success(msg)
                                            st.rerun()
                                        else:
                                            st.error(msg)

                                with opc4:
                                    if st.button("🗑️ 删除账号", key=f"del_{u['phone']}", disabled=u['phone'] == '13800138000'):
                                        ok, msg = st.session_state.auth.delete_user(u['phone'], st.session_state.user.get('role'))
                                        if ok:
                                            st.success(msg)
                                            st.rerun()
                                        else:
                                            st.error(msg)
                        
            else:
                st.info("暂无用户数据")

# --- Tab 7: 商家配置中心 (所有登录用户可见) ---
with tab_shop_config:
    st.header("⚙️ 商家配置中心")
    st.caption("店铺基础信息 / 品类与投放偏好 / AI 用量配额与订阅 / 平台级 API 连接状态")

    with st.container(border=True):
        sc1, sc2 = st.columns([0.2, 3])
        sc1.info("🔐")
        sc2.markdown("""
**多租户数据隔离策略（当前版本逻辑隔离）**：
- 非管理员账号的素材归属与检索默认按 `owner_phone`（店铺归属）过滤，不会看到其他商家素材；
- **空库新商家（upload_count = 0）**默认仅开放「🎨 智能微调」和本配置中心，避免新入驻用户看到无意义的系统级行业样本数据集；
- 管理员在「👥 用户管理」内开启白名单 + 为用户分配行业素材包后，其余 Tab 自动解锁；
- 生产版 Beta 规划：OSS 分桶 / Postgres 行级 RLS / Qdrant 按 tenant_id 分 collection，实现物理级隔离。
        """)
    st.divider()
    st.caption("管理店铺基础信息、投放偏好与 API 用量")

    current_phone = st.session_state.user.get("phone", "")
    profile = st.session_state.auth.get_user(current_phone) or {}
    role = st.session_state.user.get("role", "user")

    col_info, col_usage = st.columns([2, 1])

    with col_info:
        with st.container(border=True):
            st.subheader("🏪 店铺与账号信息")

            default_shop = profile.get("shop_name", "")
            default_cat = profile.get("category", "请选择品类")
            default_channels = profile.get("channel_prefs", [])

            new_shop = st.text_input("店铺名称", value=default_shop, placeholder="例如：XX美妆官方旗舰店")
            category_options = ["请选择品类", "服饰鞋包", "美妆个护", "食品饮料", "3C数码", "运动户外", "母婴玩具", "家居日用", "珠宝配饰", "宠物用品", "其他"]
            cat_idx = category_options.index(default_cat) if default_cat in category_options else 0
            new_cat = st.selectbox("主营品类", category_options, index=cat_idx)

            all_channels = ["直通车", "引力魔方", "万相台无界", "极速推", "品销宝", "超级直播", "搜索推广", "推荐推广", "站外抖音", "站外小红书"]
            default_selected = [c for c in default_channels if c in all_channels]
            new_channels = st.multiselect(
                "常用投放渠道（决定 CTR 看板对比基线）",
                options=all_channels,
                default=default_selected
            )

            st.markdown("##### 👤 账号信息")
            c_a, c_b = st.columns([1, 1])
            c_a.markdown(f"**手机号**：{st.session_state.auth.mask_phone(current_phone)}")
            role_map = {"admin": "🛡️ 平台管理员", "designer": "🎨 设计岗", "user": "🏪 商家运营"}
            c_b.markdown(f"**权限角色**：{role_map.get(role, '商家运营')}")

            if st.button("💾 保存配置", type="primary", key="save_shop_profile"):
                cat_save = None if new_cat == "请选择品类" else new_cat
                ok, msg = st.session_state.auth.update_profile(current_phone, new_shop, cat_save, new_channels)
                if ok:
                    if "shop_name" not in st.session_state.user:
                        st.session_state.user["shop_name"] = new_shop
                    else:
                        st.session_state.user["shop_name"] = new_shop
                    st.success(msg)
                    time.sleep(0.5)
                    st.rerun()
                else:
                    st.error(msg)

    with col_usage:
        with st.container(border=True):
            st.subheader("📊 AI 用量配额（本月）")

            upload_count = profile.get("upload_count", 0)
            generate_count = profile.get("gen_count", 0) or 0

            # Demo-tier quota mock data
            quota_upload = 500
            quota_generate = 100
            quota_search = 3000

            st.metric("本月素材入库", f"{upload_count} / {quota_upload}")
            st.progress(min(upload_count / quota_upload, 1.0))

            st.divider()
            st.metric("本月生图/微调", f"{generate_count} / {quota_generate}")
            st.progress(min(generate_count / quota_generate, 1.0))

            st.divider()
            st.metric("本月检索调用", f"~{min(1200, quota_search)} / {quota_search}")
            st.progress(min(1200 / quota_search, 1.0))

            with st.expander("📝 订阅与计费说明"):
                st.markdown(
                    "**社区版（免费）** 配额：\n"
                    "- 入库：500 张/月\n"
                    "- 生图/微调：100 张/月\n"
                    "- 检索：3000 次/月\n\n"
                    "**商家标准版**（¥199/月）：\n"
                    "- 入库：5000 张/月\n"
                    "- 生图/微调：800 张/月\n"
                    "- 团队席位：3 个\n\n"
                    "如需升级请联系平台管理员。"
                )

    st.divider()

    # 管理员才看得到的平台级配置
    if role == 'admin':
        with st.container(border=True):
            st.subheader("🛡️ 平台级配置（仅平台管理员可见）")
            env_cols = st.columns([1, 1, 1])
            with env_cols[0]:
                st.metric("DashScope 状态", "✅ 已连接" if os.getenv("DASHSCOPE_API_KEY") else "❌ 未配置")
            with env_cols[1]:
                st.metric("Qdrant 状态", "✅ 已连接" if os.getenv("QDRANT_API_KEY") else "❌ 未配置")
            with env_cols[2]:
                st.metric("OCR.space 状态", "✅ 已连接" if (os.getenv("OCR_KEY") or os.getenv("OCRSPACE_API_KEY")) else "❌ 未配置")

            st.info("提示：修改平台级配置需要编辑服务器 `~/.env` 文件后重启服务。用户侧不需要关心此项。")
