"""JYS五维评分选股引擎

五维评分体系 (满分100分):
  技术面(30分): 日涨跌幅(10) + 20日动量(15) + 换手率(5)
  估值面(25分): PE(10) + PB(10) + PR市赚率(5)
  盈利质量(30分): ROE(15) + 利润增长(15)
  安全性(10分): PB安全边际(3) + 股息稳定性(3) + 换手率波动(4)
  分红(5分): 股息率

评级: A+(85-100) / A(75-84) / B+(65-74) / B(55-64) / C(45-54) / D(<45)

参考: https://github.com/stevenwxz/JYSstock_analyzer
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .dividend_override import get_dividend_override


# ---------------------------------------------------------------------------
# 数据类
# ---------------------------------------------------------------------------

@dataclass
class JYSScoreResult:
    """五维评分结果"""
    code: str
    name: str
    total_score: int
    grade: str
    detail: Dict[str, int] = field(default_factory=dict)
    stock_data: Dict = field(default_factory=dict)

    # 各维度小计
    tech_score: int = 0        # 技术面 (满分30)
    valuation_score: int = 0   # 估值面 (满分25)
    profit_score: int = 0      # 盈利质量 (满分30)
    safety_score: int = 0      # 安全性 (满分10)
    dividend_score: int = 0    # 分红 (满分5)

    # 市场标识
    market: str = "A"          # "A" (A股) / "HK" (港股通)


# ---------------------------------------------------------------------------
# 评级映射
# ---------------------------------------------------------------------------

def _grade(score: int) -> str:
    if score >= 85:
        return "A+"
    if score >= 75:
        return "A"
    if score >= 65:
        return "B+"
    if score >= 55:
        return "B"
    if score >= 45:
        return "C"
    return "D"


# ---------------------------------------------------------------------------
# 阶梯评分辅助
# ---------------------------------------------------------------------------

def _step_score(value: Optional[float], thresholds: List[tuple]) -> int:
    """阶梯评分: thresholds 为 [(条件值, 得分), ...]，从高到低匹配"""
    if value is None:
        return 0
    for threshold, score in thresholds:
        if value > threshold:
            return score
    return 0


def _step_score_lt(value: Optional[float], thresholds: List[tuple]) -> int:
    """低于阈值得分: thresholds 为 [(条件值, 得分), ...]，从低到高匹配"""
    if value is None:
        return 0
    for threshold, score in thresholds:
        if value < threshold:
            return score
    return 0


def _step_score_range(value: Optional[float], ranges: List[tuple]) -> int:
    """区间评分: ranges 为 [(low, high, score), ...]"""
    if value is None:
        return 0
    for low, high, score in ranges:
        if low <= value < high:
            return score
    return 0


# ---------------------------------------------------------------------------
# 五维评分器
# ---------------------------------------------------------------------------

class JYSScorer:
    """JYS五维评分选股器

    Args:
        config: 可选配置覆盖，支持键:
            max_pe_ratio: 最大PE (默认30)
            min_turnover_rate: 最小换手率 (默认1.0)
            min_price: 最低股价 (默认1.0)
            min_strength_score: 最低综合分 (默认40)
            momentum_days: 动量天数 (默认20)
    """

    DEFAULT_CONFIG = {
        "max_pe_ratio": 30,
        "min_turnover_rate": 1.0,
        "min_price": 1.0,
        "min_strength_score": 40,
        "momentum_days": 20,
    }

    def __init__(self, config: Optional[Dict] = None):
        self._config = {**self.DEFAULT_CONFIG, **(config or {})}

    # ---- 1. 技术面 (满分30) ----

    def _score_technical(self, stock: Dict) -> Dict[str, int]:
        """技术面评分: 日涨跌幅(10) + 20日动量(15) + 换手率(5)"""
        scores = {}

        # 日涨跌幅 (满分10)
        change = stock.get("change_pct", 0) or 0
        scores["daily_change"] = _step_score(change, [
            (5, 10), (2, 7), (0, 4), (-2, 2),
        ])

        # 20日动量 (满分15)
        momentum = stock.get("momentum_20d", 0) or 0
        scores["momentum_20d"] = _step_score(momentum, [
            (15, 15), (10, 12), (5, 8), (0, 4),
        ])

        # 换手率 (满分5): 1-3%最佳
        turnover = stock.get("turnover_rate", 0) or 0
        scores["turnover_rate_tech"] = _step_score_range(turnover, [
            (1, 3, 5), (3, 5, 4), (5, 8, 3), (0.5, 1, 2),
        ])
        if turnover >= 8:
            scores["turnover_rate_tech"] = 1

        return scores

    # ---- 2. 估值面 (满分25) ----

    def _score_valuation(self, stock: Dict) -> Dict[str, int]:
        """估值面评分: PE(10) + PB(10) + PR市赚率(5)"""
        scores = {}

        # PE (满分10): 越低越好
        pe = stock.get("pe_ratio")
        scores["pe_ratio"] = _step_score_lt(pe, [
            (10, 10), (20, 7), (30, 4),
        ])

        # PB (满分10): 越低越好
        pb = stock.get("pb_ratio")
        scores["pb_ratio"] = _step_score_lt(pb, [
            (2, 10), (4, 8), (7, 5), (10, 2),
        ])

        # PR市赚率 (满分5): PR = PE / ROE百分比, <0.8为低估
        roe_pct = stock.get("roe")  # ROE已经是百分比形式
        pr = None
        if pe and roe_pct and roe_pct > 0:
            pr = pe / roe_pct
        scores["pr"] = _step_score_lt(pr, [
            (0.8, 5), (1.0, 3), (1.2, 2),
        ])

        return scores

    # ---- 3. 盈利质量 (满分30) ----

    def _score_profit(self, stock: Dict) -> Dict[str, int]:
        """盈利质量评分: ROE(15) + 利润增长(15)"""
        scores = {}

        # ROE (满分15): 从PB/PE推导
        roe = stock.get("roe")
        scores["roe"] = _step_score(roe, [
            (20, 15), (15, 12), (10, 8), (5, 4),
        ])

        # 利润增长 (满分15)
        # 港股有API真实利润增长字段[51]，优先使用
        # A股无此字段，走 ROE*(1-分红支付率) 推导
        profit_growth = stock.get("profit_growth")
        if profit_growth is None:
            dividend_yield = stock.get("dividend_yield", 0) or 0
            if roe and roe > 0 and dividend_yield > 0:
                payout_ratio = min(dividend_yield / roe, 0.9)
                profit_growth = roe * (1 - payout_ratio)
            elif roe and roe > 0:
                profit_growth = roe

        scores["profit_growth"] = _step_score(profit_growth, [
            (30, 15), (20, 12), (10, 8), (0, 4),
        ])

        return scores

    # ---- 4. 安全性 (满分10) ----

    def _score_safety(self, stock: Dict) -> Dict[str, int]:
        """安全性评分: PB安全边际(3) + 股息稳定性(3) + 换手率波动(4)"""
        scores = {}

        # PB安全边际 (满分3): PB<1 = 低于账面价值
        pb = stock.get("pb_ratio")
        scores["pb_safety"] = _step_score_lt(pb, [
            (1.0, 3), (1.5, 2), (2.5, 1),
        ])

        # 股息稳定性 (满分3): 用股息率作为代理
        dividend = stock.get("dividend_yield", 0) or 0
        scores["dividend_stability"] = _step_score(dividend, [
            (5, 3), (3, 2), (1, 1),
        ])

        # 换手率波动 (满分4): 低换手率=稳定
        turnover = stock.get("turnover_rate", 0) or 0
        scores["turnover_volatility"] = _step_score_lt(turnover, [
            (2, 4), (5, 3), (10, 1),
        ])

        return scores

    # ---- 5. 分红 (满分5) ----

    def _score_dividend(self, stock: Dict) -> Dict[str, int]:
        """分红评分: 股息率 (满分5)"""
        dividend = stock.get("dividend_yield", 0) or 0
        score = _step_score(dividend, [
            (5, 5), (3, 4), (2, 3), (1, 2), (0.5, 1),
        ])
        return {"dividend_yield": score}

    # ---- 公开接口 ----

    def calculate_score(self, stock_data: Dict) -> JYSScoreResult:
        """计算单只股票五维评分

        Args:
            stock_data: 包含 code, name, pe_ratio, pb_ratio, roe,
                        change_pct, momentum_20d, turnover_rate,
                        dividend_yield, market 等字段的字典

        Returns:
            JYSScoreResult 评分结果
        """
        # 确定市场
        market = stock_data.get("market", "A")

        # 股息率修正: A股override表覆盖API近似值(字段[64])，港股API可靠直接使用
        code = stock_data.get("code", "")
        if market != "HK":
            override = get_dividend_override(code)
            if override:
                stock_data = dict(stock_data)
                stock_data["dividend_yield"] = override["dividend_yield"]
                stock_data["dividend_per_share"] = override["dividend_per_share"]

        # 计算各维度
        tech = self._score_technical(stock_data)
        valuation = self._score_valuation(stock_data)
        profit = self._score_profit(stock_data)
        safety = self._score_safety(stock_data)
        dividend = self._score_dividend(stock_data)

        # 汇总
        tech_total = sum(tech.values())
        val_total = sum(valuation.values())
        profit_total = sum(profit.values())
        safety_total = sum(safety.values())
        div_total = sum(dividend.values())
        total = tech_total + val_total + profit_total + safety_total + div_total

        detail = {**tech, **valuation, **profit, **safety, **dividend}

        return JYSScoreResult(
            code=code,
            name=stock_data.get("name", ""),
            total_score=total,
            grade=_grade(total),
            detail=detail,
            stock_data=stock_data,
            tech_score=tech_total,
            valuation_score=val_total,
            profit_score=profit_total,
            safety_score=safety_total,
            dividend_score=div_total,
            market=market,
        )

    def score_batch(self, stocks: List[Dict]) -> List[JYSScoreResult]:
        """批量评分"""
        return [self.calculate_score(s) for s in stocks]

    def select_top(self, stocks: List[Dict], top_n: int = 10) -> List[JYSScoreResult]:
        """批量评分 + 过滤 + 排序，返回 Top N

        过滤规则:
            - PE > max_pe_ratio 或 PE <= 0
            - 换手率 < min_turnover_rate
            - 股价 < min_price
            - 综合分 < min_strength_score
        """
        results = self.score_batch(stocks)

        # 过滤
        filtered = []
        for r in results:
            d = r.stock_data
            pe = d.get("pe_ratio")
            if pe is not None and (pe <= 0 or pe > self._config["max_pe_ratio"]):
                continue
            if (d.get("turnover_rate", 0) or 0) < self._config["min_turnover_rate"]:
                continue
            if (d.get("price", 0) or 0) < self._config["min_price"]:
                continue
            if r.total_score < self._config["min_strength_score"]:
                continue
            filtered.append(r)

        # 排序
        filtered.sort(key=lambda r: r.total_score, reverse=True)
        return filtered[:top_n]
