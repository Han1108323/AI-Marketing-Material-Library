
# L1 Entity Dictionaries
# Used for explicit filtering (The Filter)

# 1. Colors (Maps to 'color_tone')
COLORS = [
    "红色", "蓝色", "绿色", "黑色", "白色", "金色", "紫色", "粉色", "黄色",
    "高饱和", "莫兰迪", "渐变", "撞色", "同色系", "金属", "霓虹", "暖色", "冷色"
]

# 2. Purposes (Maps to 'material_purpose')
PURPOSES = [
    "海报", "banner", "详情页", "主图", "封面", "朋友圈", "开屏", "信息流", "广告"
]

# 3. Styles (Maps to 'visual_style')
STYLES = [
    "极简", "赛博朋克", "国潮", "3D", "手绘", "插画", "孟菲斯", "新拟态", 
    "故障艺术", "波普", "蒸汽波", "复古", "未来主义", "扁平", "简约"
]

# 4. Audiences (Maps to 'target_audience')
AUDIENCES = [
    "Z世代", "白领", "宝妈", "银发族", "学生", "青年", "高端"
]

# 5. Compositions (Maps to 'composition')
COMPOSITIONS = [
    "中心", "对称", "留白", "满版", "九宫格", "对角线", "三分法",
    "横版", "竖版", "方形"
]

# 6. Scenes/Festivals (Triggers L3 Expansion)
SCENES = [
    "母亲节", "父亲节", "情人节", "七夕", "春节", "中秋", "圣诞", "元旦",
    "双11", "618", "大促", "开学季", "毕业季", "夏季", "冬季", "春天", "秋天",
    "营销", "活动"
]

# L2 Abstract Keywords (The Finder)
# Triggers semantic vector search
ABSTRACT_KEYWORDS = [
    "的", "感", "风", "调性", "氛围", "风格", "视觉", "冲击力", "高级",
    "温情", "温馨", "热闹", "喜庆", "简约", "大气", "小清新"
]

# L3 Complex Logic Keywords (The Reasoner)
# Triggers LLM reasoning, rewriting, and expansion
COMPLEX_KEYWORDS = [
    "适合", "面向", "针对", "且", "并且", "同时", "要", "不要", 
    "找一张", "帮我", "推荐", "逻辑", "虽然", "但是"
]
