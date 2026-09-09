"""Small, versioned demo catalogue used by the standalone lite demo app.

The production pipeline can replace this module with Qdrant records. Keeping a
curated catalogue in git makes the public demo useful without API keys or a
database service.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import re
from typing import Iterable


ASSET_DIR = Path(__file__).resolve().parent / "AI素材案例"


@dataclass(frozen=True)
class Material:
    filename: str
    title: str
    category: str
    style: str
    colors: tuple[str, ...]
    audience: tuple[str, ...]
    purpose: str
    visual_score: int
    copy_score: int
    ad_score: int
    ctr: float
    insight: str

    @property
    def score(self) -> int:
        return self.visual_score + self.copy_score + self.ad_score

    @property
    def path(self) -> Path:
        return ASSET_DIR / self.filename

    def to_dict(self) -> dict:
        value = asdict(self)
        value["score"] = self.score
        value["path"] = str(self.path)
        return value


MATERIALS = (
    Material("product_hq_1.jpg", "爆款运动耳机场景图", "3C数码", "活力", ("橙色", "黑色", "白色"), ("Z世代", "运动人群", "学生"), "信息流广告", 8, 7, 8, 0.048, "场景代入感强，利益点清晰，适合直接复用为新品冷启动素材。"),
    Material("product_hq_2.jpg", "高对比无线耳机主图", "3C数码", "极简", ("黄色", "黑色"), ("Z世代", "都市白领"), "电商主图", 9, 3, 5, 0.031, "主体聚焦和色彩对比突出，补充利益点与 CTA 后更适合投放。"),
    Material("product_hq_3.jpg", "质感腕表静物", "腕表配饰", "轻奢", ("黑色", "金色"), ("品质男性", "礼赠人群"), "品牌海报", 9, 5, 6, 0.037, "材质表现优秀，适合作为品牌心智素材或高客单详情页首屏。"),
    Material("product_hq_4.jpg", "高定口红产品棚拍", "美妆个护", "高级感", ("红色", "黑色"), ("年轻女性", "精致妈妈", "礼赠人群"), "电商主图", 9, 7, 8, 0.050, "双11/618大促口红爆款主图，高饱和红金配色与质感拉满，适合美妆节点复用。"),
    Material("product_hq_6.jpg", "家居生活场景组合", "家居生活", "北欧", ("白色", "木色"), ("新中产", "年轻女性", "家装人群"), "场景海报", 7, 5, 5, 0.029, "生活化场景完整，适合叠加价格利益点与投放渠道的强CTA。"),
    Material("product_hq_7.jpg", "有机生活方式组合", "食品健康", "自然极简", ("米色", "棕色", "绿色"), ("健康人群", "都市白领"), "电商主图", 9, 6, 5, 0.033, "动态堆叠构图有记忆点，产品利益表达仍可进一步前置。"),
    Material("product_hq_8.jpg", "精华护肤品组合", "美妆个护", "清透", ("粉色", "白色", "金色"), ("年轻女性", "精致妈妈", "敏感肌人群"), "活动Banner", 8, 8, 7, 0.046, "母亲节/女神节美妆护肤活动Banner模板，适合直接裂变节日文案。"),
    Material("product_hq_9.jpg", "潮流运动鞋场景", "运动户外", "潮流", ("白色", "黑色", "红色"), ("Z世代", "学生", "年轻男性"), "信息流广告", 8, 6, 7, 0.043, "潮流风格与鞋类品类匹配，裂变多配色版本高效。"),
    Material("product_hq_10.jpg", "高定口红美妆棚拍", "美妆个护", "高级感", ("粉色", "白色"), ("年轻女性", "精致妈妈"), "社媒种草", 8, 6, 7, 0.041, "柔和光线适合美妆口红种草，可加入功效数字增强转化确定性。"),
    Material("product_hq_11.jpg", "现代客厅场景", "家居生活", "北欧", ("米色", "棕色"), ("新中产", "家装人群"), "场景海报", 8, 4, 5, 0.028, "空间氛围完整，适合承载生活方式叙事和品牌价值表达。"),
    Material("product_hq_13.jpg", "自然家居品牌空间", "家居生活", "波西米亚", ("米色", "紫色", "绿色"), ("新中产", "年轻女性"), "品牌海报", 9, 3, 4, 0.027, "视觉氛围强但缺少行动引导，建议补充产品锚点和核心卖点。"),
    Material("product_hq_14.jpg", "清爽口红护肤组合", "美妆个护", "清透", ("蓝色", "白色", "粉色"), ("年轻女性", "敏感肌人群"), "电商主图", 8, 7, 7, 0.044, "品类识别快、信息负担低，适合复用为新品冷启动口红/护肤素材。"),
    Material("product_hq_16.jpg", "潮流饮品视觉", "食品饮料", "潮流", ("橙色", "红色"), ("Z世代", "学生"), "信息流广告", 9, 7, 8, 0.052, "高饱和配色有强停留力，利益点和场景结合完整，可直接复用。"),
    Material("product_hq_17.jpg", "运动生活方式", "运动户外", "活力", ("蓝色", "橙色"), ("运动人群", "年轻男性"), "信息流广告", 8, 6, 8, 0.046, "运动场景明确，建议用限时机制进一步提高点击动机。"),
    Material("product_hq_19.jpg", "极简桌面科技", "3C数码", "科技感", ("灰色", "黑色"), ("极客", "都市白领"), "品牌海报", 8, 4, 6, 0.034, "质感稳定且留白充足，适合叠加一句核心功能文案。"),
    Material("product_hq_20.jpg", "双11美妆大促组合", "美妆个护", "促销", ("红色", "金色", "粉色"), ("年轻女性", "精致妈妈"), "活动Banner", 9, 8, 8, 0.051, "双11/618大促节点美妆组合Banner，促销信息层级清晰，可直接复用裂变。"),
    Material("watch_black_1.jpg", "黑色商务腕表", "腕表配饰", "商务", ("黑色", "灰色"), ("品质男性", "职场人群"), "电商主图", 8, 5, 6, 0.035, "产品轮廓清晰，适合价格利益点明确的效果广告。"),
    Material("watch_classic_1.jpg", "经典机械腕表", "腕表配饰", "经典", ("棕色", "金色"), ("品质男性", "礼赠人群"), "详情页", 8, 6, 6, 0.038, "经典调性与礼赠场景匹配，可裂变节日版本。"),
    Material("watch_luxury_1.jpg", "奢华腕表氛围图", "腕表配饰", "轻奢", ("金色", "黑色"), ("高净值人群", "礼赠人群"), "品牌海报", 10, 5, 6, 0.042, "停留力和质感出色，适合高端品牌认知投放。"),
    Material("watch_smart_1.jpg", "智能手表功能展示", "3C数码", "科技感", ("蓝色", "黑色"), ("运动人群", "极客"), "功能海报", 8, 8, 8, 0.055, "功能、视觉与行动动机均衡，是当前库存中的优先复用素材。"),
    Material("watch_sport_1.jpg", "运动手表户外场景", "运动户外", "活力", ("橙色", "黑色"), ("运动人群", "年轻男性"), "信息流广告", 9, 6, 8, 0.049, "场景感强且产品清晰，可围绕赛事和训练计划继续裂变。"),
    Material("gen_1766715957_双11猫粮.png", "双11猫粮促销概念", "宠物食品", "促销", ("红色", "黄色"), ("养宠人群", "年轻女性"), "活动Banner", 7, 8, 8, 0.047, "大促信息完整，可进一步统一字体层级并强化品牌露出。"),
    Material("gen_1766721883_双11猫粮.png", "双11猫粮大促主图", "宠物食品", "促销", ("红色", "金色", "橙色"), ("养宠人群", "年轻女性"), "电商主图", 8, 7, 8, 0.049, "双11节点宠物食品主图模板，价格锚点和促销信息完整，适合裂变同类宠物素材。"),
    Material("素材示例1.jpg", "节点营销横幅", "节日营销", "促销", ("红色", "金色"), ("大众消费者",), "活动Banner", 8, 8, 7, 0.045, "信息层级清楚，适合作为节点营销模板快速复用。"),
    Material("素材示例2.jpg", "双11美妆口红大促海报", "美妆个护", "促销", ("红色", "粉色", "金色"), ("年轻女性", "精致妈妈"), "活动Banner", 9, 8, 8, 0.052, "双11口红美妆爆款主图，红金大促配色+强利益点结构，适合直接复用裂变。"),
    Material("素材示例3.jpg", "618数码家电大促", "3C数码", "促销", ("红色", "蓝色", "白色"), ("都市白领", "学生", "Z世代"), "活动Banner", 8, 7, 7, 0.046, "618大促节点数码家电模板，价格机制清晰。"),
    Material("素材示例4.jpg", "食品健康礼盒营销", "食品健康", "礼赠", ("红色", "金色", "棕色"), ("礼赠人群", "健康人群", "都市白领"), "活动Banner", 7, 7, 6, 0.040, "年货节/中秋礼赠场景食品礼盒模板，节日氛围浓厚。"),
    Material("素材示例5.jpg", "女性护肤精致场景", "美妆个护", "高级感", ("粉色", "白色", "米色"), ("年轻女性", "精致妈妈"), "社媒种草", 8, 6, 7, 0.043, "护肤美妆种草图，氛围与产品锚点均衡，适合裂变多品类。"),
    Material("素材示例6.jpg", "家居场景暖冬氛围", "家居生活", "北欧", ("米色", "棕色", "橙色"), ("新中产", "年轻女性"), "场景海报", 7, 5, 6, 0.030, "暖冬家居场景完整，适合结合双11/双12大促裂变文案。"),
    Material("素材示例7.jpg", "运动户外训练场景", "运动户外", "活力", ("蓝色", "黑色", "橙色"), ("运动人群", "年轻男性", "学生"), "信息流广告", 8, 6, 7, 0.044, "运动场景明确，行动动机充足，适合裂变跑鞋/运动服饰。"),
    Material("素材示例8.jpg", "高端腕表礼赠海报", "腕表配饰", "轻奢", ("黑色", "金色"), ("品质男性", "高净值人群", "礼赠人群"), "品牌海报", 9, 6, 7, 0.045, "高端腕表礼赠海报，质感与调性匹配高客单投放。"),
    Material("素材示例9.jpg", "宠物食品温馨场景", "宠物食品", "温暖", ("米色", "橙色"), ("养宠人群", "年轻女性"), "社媒种草", 8, 7, 6, 0.041, "宠物食品温馨场景图，情感共鸣强，适合双11裂变促销文案。"),
    Material("素材示例10.jpg", "双12服饰穿搭大促", "服饰穿搭", "促销", ("红色", "白色", "黑色"), ("年轻女性", "学生", "Z世代"), "活动Banner", 8, 7, 7, 0.047, "双12大促服饰穿搭模板，模特与价格利益点结构完整。"),
    Material("素材示例11.jpg", "母婴护肤温和场景", "母婴亲子", "温暖", ("粉色", "白色", "米色"), ("精致妈妈", "母婴人群"), "社媒种草", 8, 7, 7, 0.044, "母婴护肤温和场景图，信任感营造充分，可裂变母婴多品类素材。"),
    Material("素材示例12.jpg", "双11大促主会场横幅", "节日营销", "促销", ("红色", "金色", "橙色"), ("大众消费者", "Z世代", "都市白领"), "活动Banner", 9, 9, 8, 0.053, "双11/618大促主会场横幅模板，信息层级与促销氛围最强，可直接复用。"),
)


ALIASES = {
    "耳机": "3C数码 科技感",
    "手表": "腕表配饰 智能手表 腕表",
    "腕表": "手表",
    "双11": "促销 活动Banner 红色 金色 大促",
    "双12": "促销 活动Banner 大促 红色",
    "618": "促销 活动Banner 大促 红色",
    "大促": "促销 活动Banner 双11 双12 618",
    "促销": "大促 红色 金色 活动Banner",
    "口红": "美妆个护 粉色 红色 女性 年轻女性 精致妈妈 高级感 大促",
    "粉底": "美妆个护 粉色 白色 女性 年轻女性 精致妈妈 清透",
    "眼影": "美妆个护 彩色 粉色 女性 年轻女性 高级感",
    "美妆": "美妆个护 粉色 女性 年轻女性 精致妈妈 高级感 清透",
    "护肤": "美妆个护 粉色 白色 蓝色 女性 年轻女性 精致妈妈 敏感肌 清透",
    "女性": "年轻女性 精致妈妈",
    "男士": "品质男性 年轻男性",
    "高端": "高级感 轻奢 奢华",
    "清爽": "清透 蓝色 白色",
    "猫粮": "宠物食品 宠物 养宠人群 年轻女性 促销",
    "狗粮": "宠物食品 宠物 养宠人群",
    "宠物": "宠物食品 养宠人群 年轻女性 温馨",
}


def _tokens(text: str) -> set[str]:
    normalized = text.lower().strip()
    for source, target in ALIASES.items():
        if source in normalized:
            normalized += " " + target.lower()
    latin = re.findall(r"[a-z0-9]+", normalized)
    chinese = [normalized[i : i + 2] for i in range(max(len(normalized) - 1, 0))]
    return {part for part in latin + chinese if part.strip()}


def search_materials(query: str = "", category: str = "全部", min_score: int = 0) -> list[dict]:
    query_tokens = _tokens(query)
    ranked: list[tuple[float, Material]] = []
    for item in MATERIALS:
        if category != "全部" and item.category != category:
            continue
        if item.score < min_score:
            continue
        haystack = " ".join(
            [item.title, item.category, item.style, *item.colors, *item.audience, item.purpose, item.insight]
        ).lower()
        item_tokens = _tokens(haystack)
        lexical = len(query_tokens & item_tokens) / max(len(query_tokens), 1)
        phrase = 1.0 if query and query.lower() in haystack else 0.0
        relevance = lexical * 0.68 + phrase * 0.22 + (item.score / 30) * 0.10
        if not query or lexical > 0 or phrase > 0:
            ranked.append((relevance, item))
    ranked.sort(key=lambda row: (row[0], row[1].score, row[1].ctr), reverse=True)
    return [{**item.to_dict(), "relevance": round(score, 3)} for score, item in ranked]


def categories(materials: Iterable[Material] = MATERIALS) -> list[str]:
    return ["全部", *sorted({item.category for item in materials})]

