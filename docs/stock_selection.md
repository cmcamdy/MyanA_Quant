# 选股模块 (`src/stock_selection/`)

## 模块概述

多策略选股模块，统一接口支持多种选股方法，共享腾讯财经API数据获取和股票池加载。当前支持：

| 方法 | 核心逻辑 | 适用场景 |
|------|----------|----------|
| **JYS 五维评分** | PE/PB/ROE/股息 + 技术面综合评分 | 价值投资，低估值好公司 |
| **因子筛选** | 技术因子 z-score 标准化 + 加权 | 历史回测中的标的筛选 |
| **Minervini 趋势** | Stage 2 上升趋势 + VCP形态 + 相对强度 | 趋势跟踪，强势突破股 |

> JYS 参考: [https://github.com/stevenwxz/JYSstock_analyzer](https://github.com/stevenwxz/JYSstock_analyzer)
> 趋势 参考: [https://github.com/RyanJHamby/stock-screener](https://github.com/RyanJHamby/stock-screener)

## 模块架构

```
src/stock_selection/
├── __init__.py             # 模块导出
├── base.py                 # ScreenerBase 抽象基类
├── registry.py             # 选股方法注册表 + create_screener() 工厂
├── tencent_fetcher.py      # 腾讯财经API数据获取（异步+同步）
├── jys_scorer.py           # 五维评分引擎
├── jys_screener.py         # 五维选股筛选器 (ScreenerBase)
├── factor_adapter.py       # 因子筛选器适配器 (ScreenerBase)
├── trend_scorer.py         # Minervini趋势评分引擎
├── trend_screener.py       # 趋势选股筛选器 (ScreenerBase)
└── dividend_override.py    # 股息率人工修正数据
```

### 统一接口

所有选股方法继承 `ScreenerBase`，通过 `create_screener()` 工厂创建：

```python
from stock_selection import create_screener, available_methods

# 查看可用方法
print(available_methods())  # ['factor', 'jys', 'trend']

# 创建筛选器
screener = create_screener("jys", top_n=10, market="all")
top_df, all_df = screener.screen_all()

# 或直接实例化
from stock_selection import JYSScreener
screener = JYSScreener(top_n=10)
top_df, all_df = screener.screen_all()
```

### 新增选股方法

1. 继承 `ScreenerBase`，实现 `screen_all()` 方法
2. 在 `registry.py` 中调用 `register_screener("name", factory)` 注册

## 核心组件

### TencentFetcher — 腾讯财经API数据获取器

从腾讯财经API获取A股和港股通实时行情及历史K线数据，支持异步和同步两种模式。

```python
from stock_selection import TencentFetcher

fetcher = TencentFetcher(max_concurrent=20)

# 异步模式（推荐，需安装 aiohttp）
stocks = await fetcher.fetch_all(codes=["601318", "600519"])

# 同步模式（自动降级，无需 aiohttp）
stocks = fetcher.fetch_all_sync(codes=["601318", "600519"])
```

**数据字段**:

| 字段 | 说明 | 来源 |
|------|------|------|
| `code` | 6位股票代码 | API |
| `name` | 股票名称 | API |
| `price` | 当前价格 | 实时行情 |
| `prev_close` | 前收盘价 | 实时行情 |
| `change_pct` | 日涨跌幅(%) | 实时行情 |
| `pe_ratio` | 市盈率 | API字段[39] |
| `pb_ratio` | 市净率 | API字段[46] |
| `roe` | 净资产收益率(%) | A股: API字段[65](财报值); 港股: 推导PB/PE*100 |
| `turnover_rate` | 换手率(%) | API字段[38] |
| `market_cap` | 总市值(亿元) | API字段[44] |
| `dividend_yield` | 股息率(%) | A股: API字段[64](近似值)+override修正; 港股: API字段[47] |
| `dividend_per_share` | 每股股息(元) | A股: 由股息率反算+override修正; 港股: API字段[72] |
| `momentum_20d` | 20日动量(%) | 历史K线计算 |

**A股 vs 港股通 API字段差异**（实测确认）：

| 字段 | A股索引 | 港股索引 | 说明 |
|------|---------|----------|------|
| PE(动态) | [39] | [39] | 相同 |
| PB | [46] | **[58]** | 港股移位 |
| 换手率 | [38] | **需计算** | 港股API[38]始终为0，通过成交量/总股本近似计算 |
| 股息率 | [64]近似(override修正) | **[47]** | A股字段[64]为近似值，部分股票需override表修正；港股[47]可靠 |
| 每股股息 | 由股息率反算(override修正) | **[72]** | 港股可用 |
| ROE | **[65]**(财报值) | 推导PB/PE*100 | A股字段[65]为财报ROE，比PB/PE推导更准确 |
| 利润增长 | 不可用 | **[51]** | 港股独有，真实净利润同比 |
| 总市值(亿) | [44] | [44] | 相同（港股单位为港元） |
| 货币 | CNY | **[75]=HKD** | 用于识别市场 |

**港股独特优势**：港股API直接提供股息率[47]和利润增长[51]，无需推导，数据质量反而优于A股。

**ROE数据来源**: A股使用API字段[65]的财报ROE值（比PB/PE推导更准确），若字段不可用则fallback到PB/PE*100推导。港股使用PB/PE*100推导（`ROE = (市值/净利润)^{-1} × (市值/净资产) = PB/PE`，乘以100转换为百分比形式）。

**A股股息率数据**: 使用API字段[64]作为股息率近似值，覆盖约89%的A股股票。对于字段[64]数据不准确的约50只股票，由`dividend_override.py`中的手工校正值覆盖修正。港股API字段[47]直接提供可靠股息率，无需修正。

**性能对比**:

| 模式 | 依赖 | 300只耗时 | 5800只耗时 | 适用场景 |
|------|------|-----------|------------|----------|
| 异步 | aiohttp | ~3秒 | ~15-20秒 | 生产环境（默认） |
| 同步 | requests | ~73秒 | ~20分钟+ | 无aiohttp时自动降级 |

**API端点**:

| 用途 | URL | 说明 |
|------|-----|------|
| 实时行情 | `qt.gtimg.cn/q={symbol}` | 批量查询，每批最多50只 |
| 历史K线 | `web.ifzq.gtimg.cn/appstock/app/fqkline/get` | 前复权日K |
| 市场指数 | `qt.gtimg.cn=q=sh000001,sz399001,sz399006` | 上证/深证/创业板 |

### JYSScorer — 五维评分引擎

五维评分体系满分100分，各维度权重与子项得分如下：

#### 维度1: 技术面 (满分30分)

| 子项 | 满分 | 评分规则 |
|------|------|----------|
| 日涨跌幅 | 10 | >5%→10, >2%→7, >0%→4, >-2%→2 |
| 20日动量 | 15 | >15%→15, >10%→12, >5%→8, >0%→4 |
| 换手率 | 5 | 1-3%→5(最佳), 3-5%→4, 5-8%→3, 0.5-1%→2, >8%→1(投机) |

**设计逻辑**: 技术面关注短期趋势和流动性。换手率1-3%为最佳区间——过低流动性差，过高可能是投机炒作。

#### 维度2: 估值面 (满分25分)

| 子项 | 满分 | 评分规则 |
|------|------|----------|
| PE(市盈率) | 10 | <10→10, <20→7, <30→4 |
| PB(市净率) | 10 | <2→10, <4→8, <7→5, <10→2 |
| PR(市赚率) | 5 | <0.8→5, <1.0→3, <1.2→2 |

**PR(市赚率)**: `PR = PE / ROE(百分比形式)`，PR < 1表示公司盈利能力相对估值被低估。由雪球用户丁宁发明，是对PEG的改进——用ROE替代增长率，更稳定。

#### 维度3: 盈利质量 (满分30分) — 核心权重

| 子项 | 满分 | 评分规则 |
|------|------|----------|
| ROE(净资产收益率) | 15 | >20%→15, >15%→12, >10%→8, >5%→4 |
| 利润增长 | 15 | >30%→15, >20%→12, >10%→8, >0%→4 |

**利润增长推导**: A股无利润增长API字段，使用 `利润增长 = ROE × (1 - 分红支付率)` 推导，其中 `分红支付率 = min(股息率/ROE, 0.9)`。港股有API字段[51]提供真实净利润同比增长，评分时优先使用。

**为什么盈利质量权重最高(30%)**: 巴菲特选股核心标准是ROE持续高于15%，高ROE意味着公司有护城河，能持续创造超额利润。

#### 维度4: 安全性 (满分10分)

| 子项 | 满分 | 评分规则 |
|------|------|----------|
| PB安全边际 | 3 | PB<1.0→3(破净), <1.5→2, <2.5→1 |
| 股息稳定性 | 3 | >5%→3, >3%→2, >1%→1 |
| 换手率波动 | 4 | <2%→4, <5%→3, <10%→1 |

**PB安全边际**: PB < 1意味着股价低于账面价值（破净），提供价值底线保护。

#### 维度5: 分红 (满分5分)

| 子项 | 满分 | 评分规则 |
|------|------|----------|
| 股息率 | 5 | >5%→5, >3%→4, >2%→3, >1%→2, >0.5%→1 |

#### 评级映射

| 评级 | 分数区间 | 含义 |
|------|----------|------|
| A+ | 85-100 | 极优 |
| A | 75-84 | 优秀 |
| B+ | 65-74 | 良好 |
| B | 55-64 | 中等 |
| C | 45-54 | 一般 |
| D | <45 | 较差 |

#### 使用示例

```python
from stock_selection import JYSScorer

scorer = JYSScorer()

# 单只股票评分
result = scorer.calculate_score({
    'code': '601318',
    'name': '中国平安',
    'price': 54.0,
    'change_pct': 2.5,        # 日涨跌幅%
    'momentum_20d': 8.0,      # 20日动量%
    'turnover_rate': 1.5,     # 换手率%
    'pe_ratio': 7.4,          # PE
    'pb_ratio': 0.96,         # PB
    'roe': 13.0,              # ROE%
    'dividend_yield': 2.36,   # 股息率%
})

print(f"总分: {result.total_score} [{result.grade}]")
print(f"技术面={result.tech_score}/30 估值={result.valuation_score}/25 "
      f"盈利={result.profit_score}/30 安全={result.safety_score}/10 分红={result.dividend_score}/5")

# 批量评分
results = scorer.score_batch(stock_data_list)

# 评分+过滤+排序
top10 = scorer.select_top(stock_data_list, top_n=10)
```

**过滤规则** (select_top默认):
- PE > 30 或 PE ≤ 0 过滤
- 换手率 < 0.3% 过滤
- 股价 < 1元 过滤
- 综合分 < 40 过滤

### 股票池加载

```python
from stock_selection import load_csi300_codes, load_a_share_codes, load_hk_connect_codes, load_all_codes, load_codes_by_market

# 按市场加载
codes = load_codes_by_market("all")      # A股+港股通 (~5800只, 默认)
codes = load_codes_by_market("a")         # 仅A股 (~5000只)
codes = load_codes_by_market("hk")        # 仅港股通 (~800只)
codes = load_codes_by_market("csi300")    # 仅沪深300 (~300只)

# 或直接调用
a_codes = load_a_share_codes()            # A股全量
hk_codes = load_hk_connect_codes()        # 港股通
all_codes = load_all_codes()              # A股 + 港股通
```

**数据来源与缓存**:
- A股全量: `akshare.stock_zh_a_spot_em()`，缓存到 `data/a_share_codes.json`
- 港股通: `akshare.stock_hk_ggt_components_em()`，缓存到 `data/hk_connect_codes.json`
- 沪深300: `akshare.index_stock_cons_csindex(symbol="000300")`，缓存到 `data/csi300_codes.json`
- 首次获取后自动缓存到本地JSON，后续直接读取

### JYSScreener — 选股筛选器

组合 `TencentFetcher` + `JYSScorer` 的一站式选股工具，输出格式与现有 `StockScreener.screen()` 兼容。

```python
from stock_selection import JYSScreener

screener = JYSScreener(
    top_n=10,           # 返回前10只
    min_score=40,       # 最低综合分
    max_pe=30,          # 最大PE
    min_turnover=0.3,   # 最低换手率%
    max_concurrent=20,  # 异步并发数
    market="all",       # "all"(A股+港股通) / "a" / "hk" / "csi300"
)

# 全量选股 (A股+港股通)
top_df, all_df = screener.screen_all()

# 仅港股通
screener_hk = JYSScreener(market="hk", top_n=5)
top_df, all_df = screener_hk.screen_all()

# 指定股票池（A股+港股混合）
top_df, all_df = screener.screen_all(codes=["601318", "00700", "600519", "09988"])

# top_df: Top N 结果
# all_df: 全部候选（过滤后）
# 均为 DataFrame，含 symbol, name, composite_score, grade,
# tech_score, valuation_score, profit_score, safety_score, dividend_score,
# pe_ratio, pb_ratio, roe, dividend_yield 等列
```

**输出列说明**:

| 列名 | 类型 | 说明 |
|------|------|------|
| `symbol` | str | 标准代码 (如 601318.SH) |
| `name` | str | 股票名称 |
| `composite_score` | int | 综合得分 (0-100) |
| `grade` | str | 评级 (A+/A/B+/B/C/D) |
| `tech_score` | int | 技术面得分 (0-30) |
| `valuation_score` | int | 估值面得分 (0-25) |
| `profit_score` | int | 盈利质量得分 (0-30) |
| `safety_score` | int | 安全性得分 (0-10) |
| `dividend_score` | int | 分红得分 (0-5) |
| `pe_ratio` | float | 市盈率 |
| `pb_ratio` | float | 市净率 |
| `roe` | float | 净资产收益率(%) |
| `dividend_yield` | float | 股息率(%) |
| `profit_growth` | float | 净利润增长(%) — 港股独有API数据 |
| `market` | str | 市场: "A" (A股) / "HK" (港股通) |

### dividend_override — 股息率修正

腾讯财经API返回的股息率数据部分不准确，此模块提供50+只股票的手工校正值。

```python
from stock_selection.dividend_override import get_dividend_override, all_overrides

# 查询单只股票修正
override = get_dividend_override("601628")
# → {"dividend_yield": 1.48, "dividend_per_share": 0.43, "year": 2024, "note": "中国人寿 API返回5.49%不准"}

# 获取全部修正数据
all_data = all_overrides()  # Dict[str, Dict]
```

**修正机制**: `JYSScorer.calculate_score()` 自动应用override，无需手动调用。对API数据中股息率异常的股票（如中国人寿API返回5.49%实际为1.48%），用修正值替换。**注意：override仅应用于A股**，港股API字段[47]返回的股息率可靠，直接使用。

### TrendScreener — Minervini Stage 2 趋势选股

基于 Mark Minervini 《Trade Like a Stock Market Wizard》的趋势跟踪选股方法，核心是识别处于 Stage 2 上升趋势的股票。组合 `TencentFetcher` + `TrendScorer` 的一站式趋势选股工具。

```python
from stock_selection import TrendScreener

screener = TrendScreener(
    top_n=10,               # 返回前10只
    min_score=50,           # 最低综合分
    market="csi300",        # "csi300"(默认) / "all" / "a" / "hk"
    benchmark="sh000001",   # 基准指数 (sh000001=上证 / sz399001=深证 / sz399006=创业板)
    max_concurrent=20,      # 异步并发数
    sma_periods=[50, 150, 200],  # SMA 周期
    min_criteria_pass=7,    # Minervini模板最少通过条件数
)

# 沪深300趋势选股
top_df, all_df = screener.screen_all()

# 指定股票池
top_df, all_df = screener.screen_all(codes=["601318", "600519", "600036"])
```

**为什么默认 market="csi300"**: 趋势选股需要 230 天 K 线数据（SMA200 计算），全量 5800 只股票的 K 线获取耗时约 65 秒以上，CSI300 约 3-5 秒。

#### 4阶段分类 (Phase Classification)

| Phase | 含义 | 判定条件 |
|-------|------|----------|
| **Uptrend** (Stage 2) | 上升趋势 | price > SMA50 > SMA150 > SMA200，且 SMA50/SMA200 斜率向上 |
| **Base** (Stage 1) | 底部整理 | 不满足其他阶段条件 |
| **Distribution** (Stage 3) | 派发 | SMA50 > SMA200 且 price > SMA50 × 1.25（过度偏离） |
| **Downtrend** (Stage 4) | 下降趋势 | price < SMA50 且 price < SMA200 且 SMA50 < SMA200 |

优先级: Downtrend > Uptrend > Distribution > Base

#### Minervini 8条件趋势模板

| # | 条件 | 说明 |
|---|------|------|
| 1 | price > SMA150 且 > SMA200 | 价格在长期均线上方 |
| 2 | SMA150 > SMA200 | 长期均线多头排列 |
| 3 | SMA200 趋势向上 | 200日均线至少1个月上升 |
| 4 | SMA50 > SMA150 | 短期均线在长期上方（级联排列） |
| 5 | price > SMA50 | 价格在短期均线上方 |
| 6 | price >= 52周最低 × 1.30 | 距年内低点至少涨30% |
| 7 | price >= 52周最高 × 0.75 | 距年内高点不超过25% |
| 8 | Phase = Uptrend | 当前处于 Stage 2 上升趋势 |

**通过标准**: 7/8 条件通过即视为趋势确认。

#### VCP 波动收缩形态

VCP (Volatility Contraction Pattern) 是 Minervini 识别突破前整理形态的核心方法：

```
价格
  │    /\      /\
  │   /  \    /  \    ← 每次回撤幅度递减
  │  /    \  /    \
  │ /      \/      \___  ← 缩量横盘，蓄势突破
  │/
  └──────────────────── 时间
    回撤1   回撤2   回撤3 (更小)
```

**检测条件**:
- 至少 2 次回撤，每次幅度小于前一次（收缩）
- 后半段成交量低于前半段（缩量确认）
- 接近 52 周高点加分

**输出**: `(detected, contraction_count, quality 0-100)`

#### 相对强度 (Relative Strength)

RS = (股价 / 基准指数) 的 63 日线性回归斜率，映射到 0-10 分：

| RS 分数 | 含义 |
|---------|------|
| 8-10 | 强势跑赢大盘 |
| 5-7 | 与大盘同步 |
| 0-4 | 弱于大盘 |

#### ATR 止损 + 仓位管理

基于 ATR (Average True Range) 的动态止损和仓位计算：

- **止损价** = 当前价 - 2 × ATR(14)
- **仓位比例** = (单笔风险比例 × 股价) / (股价 - 止损价) × 100%
- **单笔风险**: 默认 1% 总资金

#### 综合评分 (0-100)

| 维度 | 满分 | 计算方式 |
|------|------|----------|
| Phase | 25 | Uptrend=25, Base=10, Distribution=5, Downtrend=0 |
| 趋势模板 | 25 | 通过条件数/8 × 25 |
| 相对强度 | 15 | RS/10 × 15 |
| VCP | 15 | 检测到: 10 + 收缩次数×2 (上限15); 未检测但≥2次收缩: 5 |
| 成交量突破 | 10 | 放量突破(vol≥1.5倍均量): 10; 适度放量(1.2倍): 5 |
| 52周位置 | 10 | 距高点<5%: 10, <10%: 7, <20%: 4 |

#### 买卖信号

**买入信号**: Phase=Uptrend AND 趋势模板通过≥7 AND 综合分≥50

**卖出信号**: Phase=Downtrend OR (Phase=Distribution AND 趋势模板通过<4)

#### 输出列说明

| 列名 | 类型 | 说明 |
|------|------|------|
| `symbol` | str | 标准代码 (如 601318.SH) |
| `name` | str | 股票名称 |
| `market` | str | 市场: "A" / "HK" |
| `composite_score` | int | 综合得分 (0-100) |
| `phase` | str | 阶段: Uptrend/Base/Distribution/Downtrend |
| `criteria_passed` | int | 趋势模板通过条件数 (0-8) |
| `criteria_total` | int | 趋势模板总条件数 (固定8) |
| `vcp_detected` | bool | 是否检测到 VCP 形态 |
| `vcp_contractions` | int | VCP 收缩次数 |
| `vcp_quality` | int | VCP 质量 (0-100) |
| `relative_strength` | float | 相对强度 (0-10) |
| `volume_breakout` | bool | 是否成交量突破 |
| `volume_ratio` | float | 当前量/均量比 |
| `sma_50` | float | 50日均线 |
| `sma_150` | float | 150日均线 |
| `sma_200` | float | 200日均线 |
| `atr_stop_loss` | float | ATR止损价 |
| `position_size_pct` | float | 建议仓位比例(%) |
| `buy_signal` | bool | 买入信号 |
| `sell_signal` | bool | 卖出信号 |

## 使用方式

### 方式1: 独立运行脚本（推荐）

```bash
# A股+港股通全量选股 (默认)
python3 scripts/demo_jys_screen.py

# 仅A股 (~5000只)
python3 scripts/demo_jys_screen.py --market a

# 仅港股通 (~800只)
python3 scripts/demo_jys_screen.py --market hk

# 仅沪深300
python3 scripts/demo_jys_screen.py --market csi300

# 指定参数
python3 scripts/demo_jys_screen.py --top 5 --detail
python3 scripts/demo_jys_screen.py --codes 601318,600519,00700,09988
python3 scripts/demo_jys_screen.py --min-score 50 --max-pe 25
```

### 方式2: Python API

```python
from stock_selection import JYSScreener

# 一键选股
screener = JYSScreener(top_n=10)
top_df, all_df = screener.screen_all()

# 选出的标的直接用于回测
symbols = top_df['symbol'].tolist()
# → ["600887.SH", "000333.SZ", "601318.SH", ...]
```

### 方式3: YAML配置（与 Portfolio Pipeline 集成）

```yaml
# config/portfolio.yaml
screening:
  enabled: true
  method: jys                    # factor(因子筛选) / jys(五维评分) / trend(趋势选股)

  jys:
    top_n: 10                    # 选前10只
    min_score: 40                # 最低综合分
    max_pe: 30                   # 最大PE
    min_turnover: 0.3            # 最低换手率%
    market: all                  # all(A股+港股通) / a / hk / csi300

  factor:
    top_n: 20
    min_bars: 60
    factors:
      momentum: {period: 20}
      volatility: {period: 20}
    indicator_factors:
      ma_alignment: true
    weights:
      momentum_20: 0.25
    filters:
      avg_volume: {min: 1000000}

  trend:
    top_n: 10                    # 选前10只
    min_score: 50                # 最低综合分
    market: csi300               # csi300(默认) / all / a / hk
    benchmark: sh000001          # sh000001=上证 / sz399001=深证 / sz399006=创业板
    sma_periods: [50, 150, 200]  # SMA 周期
    min_criteria_pass: 7         # Minervini模板最少通过条件数
```

### 方式4: Streamlit 可视化系统

```bash
# 安装 Web 依赖
pip install streamlit aiohttp

# 启动
streamlit run web/app.py
```

**功能**:

| 功能 | 说明 |
|------|------|
| 选股方法切换 | 侧边栏选择 JYS/因子/趋势，预留扩展 |
| JYS 参数调节 | 市场、Top N、最低分、PE/换手率/股价阈值 |
| 单股评分 | 输入代码 → 五维雷达图 + 维度柱状图 + 评分明细 |
| 批量筛选 | 筛选结果表 + CSV下载 |
| 五维分析 | 雷达图(Scatterpolar) + 维度得分柱状图 + 子项明细 |
| 分布统计 | 评级分布、市场饼图、得分直方图、PE-Score散点图 |
| 数据缓存 | `@st.cache_data(ttl=3600)` + 手动清除 |

**异步兼容**: Streamlit 运行在 Tornado 事件循环上，`TencentFetcher` 检测到后会降级为纯同步模式（20分钟+）。Web 应用通过 `ThreadPoolExecutor` 在独立线程中运行筛选，使 `asyncio.run()` 正常执行，保持异步模式速度（~15-20秒/5800只）。

## 三种选股方法对比

| 特性 | 因子选股 (FactorScreener) | 五维选股 (JYSScreener) | 趋势选股 (TrendScreener) |
|------|--------------------------|------------------------|--------------------------|
| 数据源 | 本地Parquet历史数据 | 腾讯财经API实时数据 | 腾讯财经API实时+230天K线 |
| 支持市场 | A股 | A股 + 港股通 | A股 + 港股通 |
| 评分维度 | 技术因子（动量/波动率/均线等） | 基本面+技术面五维 | 趋势形态+相对强度+量价 |
| PE/PB/ROE | 无 | 有（核心维度） | 无 |
| 利润增长 | 无 | 港股有真实API数据 | 无 |
| 分红 | 无 | 有 | 无 |
| Phase分类 | 无 | 无 | 4阶段（Base/Up/Dist/Down） |
| 形态识别 | 无 | 无 | VCP波动收缩 |
| 止损/仓位 | 无 | 无 | ATR止损+仓位计算 |
| 数据时效 | 历史回测 | 实时盘后 | 实时盘后 |
| 适用场景 | 回测中的标的筛选 | 价值投资选股 | 趋势跟踪，强势突破股 |
| 评分方法 | z-score标准化+加权 | 阈值阶梯+加权 | 趋势模板+形态+RS |
| 全量耗时 | 依赖本地数据量 | ~3秒(CSI300) / ~15秒(全量) | ~3-5秒(CSI300) / ~65秒+(全量) |

**推荐组合**: 先用JYS五维选股筛选出基本面优秀的标的池，再用趋势选股识别其中处于上升趋势的标的，最后用因子选股在历史数据上验证和排序，接入策略回测。

## 已知局限

1. **ROE来源**: A股优先使用API字段[65]的财报ROE值，fallback到PB/PE*100推导。在PE或PB为负时仍无法计算。港股使用PB/PE推导。
2. **A股利润增长推导**: A股无利润增长API字段，使用 ROE * (1 - 分红支付率) 推导，不是真实的财报利润增长率，仅作为近似。港股有API字段[51]的真实数据。
3. **A股股息率近似**: A股API字段[64]为股息率近似值，约89%的股票有数据，但部分股票精度有限（偏差可达2-3个百分点）。约50只股票由override表手工修正。港股API字段[47]可靠，无需修正。
4. **港股换手率**: 港股API不返回换手率，通过成交量/总股本近似计算，可能不够精确。
5. **阶梯评分**: JYS所有因子使用离散阈值，在边界处存在阶跃效应（如PE从20.01降到19.99，PE得分从7跳到10）。
6. **无行业分散**: 当前按综合得分排名，不限制行业集中度，可能出现同行业多只入选。
7. **港股货币**: 港股价格和市值为港元，与A股混合排名时货币差异未做汇率转换。
8. **趋势选股数据量大**: 趋势选股需要230天K线数据（SMA200），全量5800只股票的K线获取耗时约65秒以上，建议使用CSI300（3-5秒）。
9. **趋势选股A股适应性**: Minervini模板源自美股，A股T+1和涨跌停制度可能导致部分条件需调整（如成交量突破阈值）。
10. **VCP检测局限**: VCP检测基于峰谷识别，在横盘整理或缓慢上涨中可能漏检或误检。
