"""JYS五维选股模块

核心组件:
    ScreenerBase    - 选股筛选器抽象基类
    JYSScorer       - 五维评分引擎 (技术面+估值+盈利质量+安全性+分红)
    JYSScoreResult  - 评分结果数据类
    JYSScreener     - 五维选股筛选器 (对接portfolio pipeline)
    TencentFetcher  - 腾讯财经API数据获取器 (A股+港股通)

注册表:
    create_screener - 工厂函数，按方法名创建筛选器
    register_screener - 注册新选股方法
    available_methods - 查看已注册方法

股票池加载:
    load_csi300_codes  - 沪深300 (~300只)
    load_a_share_codes - A股全量 (~5000只)
    load_hk_connect_codes - 港股通 (~800只)
    load_all_codes     - A股+港股通 (~5800只)
    load_codes_by_market - 按市场加载 ("all"/"a"/"hk"/"csi300")
"""

from .base import ScreenerBase
from .registry import create_screener, register_screener, available_methods
from .jys_scorer import JYSScorer, JYSScoreResult
from .jys_screener import JYSScreener
from .trend_scorer import TrendScorer, TrendScoreResult
from .trend_screener import TrendScreener
from .tencent_fetcher import (
    TencentFetcher,
    load_csi300_codes,
    load_a_share_codes,
    load_hk_connect_codes,
    load_all_codes,
    load_codes_by_market,
)

__all__ = [
    "ScreenerBase",
    "create_screener",
    "register_screener",
    "available_methods",
    "JYSScorer",
    "JYSScoreResult",
    "JYSScreener",
    "TrendScorer",
    "TrendScoreResult",
    "TrendScreener",
    "TencentFetcher",
    "load_csi300_codes",
    "load_a_share_codes",
    "load_hk_connect_codes",
    "load_all_codes",
    "load_codes_by_market",
]
