"""股息率人工修正数据

腾讯财经API返回的股息率部分股票不准确，此模块提供手工校正值。
数据来源: https://github.com/stevenwxz/JYSstock_analyzer config/dividend_override.py
"""

from typing import Dict, Optional

# 已知API返回股息率不准确的股票，key为6位代码
_OVERRIDES: Dict[str, Dict] = {
    # ---- 金融 ----
    "601318": {"dividend_yield": 2.36, "dividend_per_share": 1.93, "year": 2024, "note": "中国平安"},
    "601398": {"dividend_yield": 5.76, "dividend_per_share": 0.3064, "year": 2024, "note": "工商银行"},
    "601288": {"dividend_yield": 5.15, "dividend_per_share": 0.2302, "year": 2024, "note": "农业银行"},
    "601939": {"dividend_yield": 5.34, "dividend_per_share": 0.3640, "year": 2024, "note": "建设银行"},
    "601988": {"dividend_yield": 5.32, "dividend_per_share": 0.2364, "year": 2024, "note": "中国银行"},
    "600036": {"dividend_yield": 4.56, "dividend_per_share": 1.9730, "year": 2024, "note": "招商银行"},
    "601166": {"dividend_yield": 5.55, "dividend_per_share": 0.1040, "year": 2024, "note": "兴业银行"},
    "600000": {"dividend_yield": 4.72, "dividend_per_share": 0.3510, "year": 2024, "note": "浦发银行"},
    "601838": {"dividend_yield": 5.75, "dividend_per_share": 0.3260, "year": 2024, "note": "成都银行"},
    "600016": {"dividend_yield": 5.25, "dividend_per_share": 0.2160, "year": 2024, "note": "民生银行"},
    "601628": {"dividend_yield": 1.48, "dividend_per_share": 0.4300, "year": 2024, "note": "中国人寿 API返回5.49%不准"},
    "601601": {"dividend_yield": 2.85, "dividend_per_share": 1.0200, "year": 2024, "note": "中国太保"},
    "601336": {"dividend_yield": 3.07, "dividend_per_share": 0.8600, "year": 2024, "note": "新华保险"},
    "601688": {"dividend_yield": 2.83, "dividend_per_share": 0.5400, "year": 2024, "note": "华泰证券"},
    "600030": {"dividend_yield": 2.21, "dividend_per_share": 0.3400, "year": 2024, "note": "中信证券"},
    # ---- 能源 ----
    "601857": {"dividend_yield": 4.72, "dividend_per_share": 0.2362, "year": 2024, "note": "中国石油"},
    "600028": {"dividend_yield": 5.61, "dividend_per_share": 0.2920, "year": 2024, "note": "中国石化"},
    "601088": {"dividend_yield": 5.86, "dividend_per_share": 2.2600, "year": 2024, "note": "中国神华"},
    "600900": {"dividend_yield": 3.02, "dividend_per_share": 0.8200, "year": 2024, "note": "长江电力"},
    "601985": {"dividend_yield": 2.32, "dividend_per_share": 0.1780, "year": 2024, "note": "中国核电"},
    "003816": {"dividend_yield": 2.56, "dividend_per_share": 0.2080, "year": 2024, "note": "中国广核"},
    # ---- 消费 ----
    "600519": {"dividend_yield": 2.98, "dividend_per_share": 30.8640, "year": 2024, "note": "贵州茅台"},
    "000858": {"dividend_yield": 3.52, "dividend_per_share": 4.6600, "year": 2024, "note": "五粮液"},
    "000568": {"dividend_yield": 3.78, "dividend_per_share": 5.4900, "year": 2024, "note": "泸州老窖"},
    "000333": {"dividend_yield": 4.51, "dividend_per_share": 3.0000, "year": 2024, "note": "美的集团"},
    "000651": {"dividend_yield": 4.93, "dividend_per_share": 2.3800, "year": 2024, "note": "格力电器"},
    "600887": {"dividend_yield": 4.21, "dividend_per_share": 1.2000, "year": 2024, "note": "伊利股份"},
    "603288": {"dividend_yield": 1.78, "dividend_per_share": 0.6600, "year": 2024, "note": "海天味业"},
    # ---- 医药 ----
    "600276": {"dividend_yield": 1.42, "dividend_per_share": 0.4800, "year": 2024, "note": "恒瑞医药"},
    "000538": {"dividend_yield": 3.15, "dividend_per_share": 2.1200, "year": 2024, "note": "云南白药"},
    "600161": {"dividend_yield": 0.82, "dividend_per_share": 0.2200, "year": 2024, "note": "天坛生物"},
    # ---- 制造 ----
    "601899": {"dividend_yield": 2.98, "dividend_per_share": 0.4000, "year": 2024, "note": "紫金矿业"},
    "600031": {"dividend_yield": 3.27, "dividend_per_share": 0.2800, "year": 2024, "note": "三一重工"},
    "601766": {"dividend_yield": 1.12, "dividend_per_share": 0.0580, "year": 2024, "note": "中国中车"},
    "600585": {"dividend_yield": 4.25, "dividend_per_share": 0.9000, "year": 2024, "note": "海螺水泥"},
    # ---- 地产/基建 ----
    "601668": {"dividend_yield": 3.86, "dividend_per_share": 0.2680, "year": 2024, "note": "中国建筑"},
    "601669": {"dividend_yield": 2.74, "dividend_per_share": 0.1680, "year": 2024, "note": "中国电建"},
    "601186": {"dividend_yield": 3.05, "dividend_per_share": 0.2200, "year": 2024, "note": "中国铁建"},
    # ---- 科技 ----
    "002415": {"dividend_yield": 1.25, "dividend_per_share": 0.5600, "year": 2024, "note": "海康威视"},
    "600036": {"dividend_yield": 4.56, "dividend_per_share": 1.9730, "year": 2024, "note": "招商银行"},
    "300750": {"dividend_yield": 0.62, "dividend_per_share": 2.5300, "year": 2024, "note": "宁德时代"},
    "002594": {"dividend_yield": 0.93, "dividend_per_share": 3.0600, "year": 2024, "note": "比亚迪"},
    "601012": {"dividend_yield": 1.52, "dividend_per_share": 0.1400, "year": 2024, "note": "隆基绿能"},
    # ---- 交运 ----
    "601111": {"dividend_yield": 0.00, "dividend_per_share": 0.0000, "year": 2024, "note": "中国国航 亏损不分红"},
    "600104": {"dividend_yield": 3.67, "dividend_per_share": 0.3700, "year": 2024, "note": "上汽集团"},
    # ---- 其他 ----
    "601225": {"dividend_yield": 1.85, "dividend_per_share": 1.0400, "year": 2024, "note": "陕西煤业"},
    "601872": {"dividend_yield": 2.15, "dividend_per_share": 0.1950, "year": 2024, "note": "招商港口"},
    "002714": {"dividend_yield": 2.36, "dividend_per_share": 0.6800, "year": 2024, "note": "牧原股份"},
    "600309": {"dividend_yield": 1.92, "dividend_per_share": 0.6500, "year": 2024, "note": "万华化学"},
    "002230": {"dividend_yield": 0.32, "dividend_per_share": 0.0800, "year": 2024, "note": "科大讯飞"},
    "688981": {"dividend_yield": 0.38, "dividend_per_share": 0.2000, "year": 2024, "note": "中芯国际"},
    "603501": {"dividend_yield": 1.05, "dividend_per_share": 0.3100, "year": 2024, "note": "韦尔股份"},
    "002049": {"dividend_yield": 0.46, "dividend_per_share": 0.0850, "year": 2024, "note": "紫光国微"},
    "688012": {"dividend_yield": 0.68, "dividend_per_share": 0.3000, "year": 2024, "note": "中微公司"},
    "300059": {"dividend_yield": 0.25, "dividend_per_share": 0.0400, "year": 2024, "note": "东方财富"},
}


def get_dividend_override(code: str) -> Optional[Dict]:
    """查询股息率人工修正值

    Args:
        code: 6位股票代码 (如 "601318")

    Returns:
        修正数据字典或None
    """
    # 支持 .SH/.SZ 后缀格式
    key = code.replace(".SH", "").replace(".SZ", "")
    return _OVERRIDES.get(key)


def all_overrides() -> Dict[str, Dict]:
    """返回全部修正数据"""
    return dict(_OVERRIDES)
