# 数据获取与存储模块 (`src/data/`)

## 模块概述

数据模块负责 A 股行情数据的获取、存储、清洗和管理，是整个系统的数据基础层。支持多数据源自动切换、Parquet 本地持久化、增量更新、行业分类查询和数据质量检测。

## 模块架构

```
src/data/
├── base.py           # KlineData 数据容器 + DataProvider 数据源抽象基类
├── storage.py        # ParquetStorage Parquet 格式本地存储
├── manager.py        # DataManager 统一数据管理入口
├── cleaner.py        # DataCleaner 数据清洗与质量检测
├── industry.py       # IndustryLookup 证监会行业分类查询
└── providers/
    ├── akshare.py    # AkShare 数据源 (日线/周线/月线)
    └── baostock.py   # Baostock 数据源 (分钟线/日线/周线/月线)
```

## 核心组件

### KlineData — K 线数据容器

统一的 K 线数据结构，封装标的信息和行情 DataFrame。

```python
from data.base import KlineData

kline = KlineData(
    symbol="000807.SZ",
    freq="1d",
    start=datetime(2024, 1, 1),
    end=datetime(2024, 12, 31),
    df=df,  # DataFrame with DatetimeIndex, columns: open/high/low/close/volume
)

kline.open    # Series — 开盘价
kline.high    # Series — 最高价
kline.low     # Series — 最低价
kline.close   # Series — 收盘价
kline.volume  # Series — 成交量
kline.amount  # Series or None — 成交额
```

### DataProvider — 数据源抽象基类

定义数据获取接口，AkShareProvider 和 BaostockProvider 为具体实现。

```python
from data.base import DataProvider

class DataProvider(ABC):
    def get_kline(self, symbol, start, end, freq='1d', adjust='qfq') -> KlineData
    def get_stock_list(self) -> pd.DataFrame
    def normalize_symbol(self, symbol: str) -> str
    def supports_freq(self, freq: str) -> bool
```

**AkShareProvider**: 支持 `1d`, `1w`, `1m`。无需注册，直接调用 AkShare 接口。

**BaostockProvider**: 支持 `5min`, `15min`, `30min`, `60min`, `1d`, `1w`, `1m`。每次调用内部自动 login/logout。

### ParquetStorage — 本地 Parquet 存储

基于 Parquet 格式的层级目录存储，支持自动合并去重和日期范围查询。

```python
from data.storage import ParquetStorage

storage = ParquetStorage(data_dir="./data")

storage.save(kline)                               # 保存，自动合并去重
df = storage.load(symbol, freq, start, end)       # 加载，可选日期范围过滤
symbols = storage.list_symbols(market="SH")        # 列出已存储标的
date_range = storage.get_date_range(symbol, freq)  # 获取数据日期范围 → (start, end) or None
```

存储路径格式: `{data_dir}/{market}/{code}/{freq}.parquet`，如 `data/SZ/000807/1d.parquet`

### DataManager — 统一数据管理

Cache-aside 模式的统一数据入口，优先读取本地缓存，缺失时自动从数据源下载。按频率自动选择最优数据源，支持增量更新。

```python
from data.manager import DataManager

mgr = DataManager(data_dir="./data")

# 获取数据（优先缓存，缺失时自动从数据源下载）
kline = mgr.get_kline("000807.SZ", start="2024-01-01", end="2024-12-31", freq="1d")

# 指定数据源
kline = mgr.get_kline("000807.SZ", start="2024-01-01", end="2024-12-31", source="baostock")

# 增量更新
new_data = mgr.update_kline("000807.SZ", freq="1d")

# 列出已缓存标的
symbols = mgr.list_cached_symbols(market="SH")

# 获取股票列表
stock_list = mgr.get_stock_list(source="akshare")
```

数据源自动选择规则:

| 频率 | 优先数据源 | 备选 |
|------|-----------|------|
| 1d | AkShare | Baostock |
| 1w | AkShare | Baostock |
| 1m | AkShare | Baostock |
| 5min/15min/30min/60min | Baostock | — |

### DataCleaner — 数据清洗与质量检测

对 OHLCV 行情数据执行缺失值检测、异常值检测、填充替换和数据约束校验。

```python
from data.cleaner import DataCleaner

cleaner = DataCleaner()

# 生成数据质量报告
report = cleaner.quality_report(df)
print(report.summary())
# === Data Quality Report ===
# Total rows: 2500
# Missing values per column:
#   open: 0
#   close: 3
#   ...
# Validation: PASSED (no errors)

# 完整清洗流程 (异常值→NaN → 填充缺失值)
df_clean = cleaner.clean(df, outlier_method='mad', outlier_threshold=3.0,
                          fill_method='ffill', fill_limit=5)

# 单独使用各功能
missing_mask = cleaner.detect_missing(df)             # 布尔矩阵，True=缺失
outlier_mask = cleaner.detect_outliers(df, method='mad', threshold=3.0)  # 异常值检测
df_filled = cleaner.fill_missing(df, method='ffill')  # 填充缺失值
df_replaced = cleaner.replace_outliers(df, replacement='clip')  # 替换异常值
errors = cleaner.validate_ohlcv(df)                   # OHLCV 约束校验
```

**异常值检测方法**:
- `mad`: 基于中位数绝对偏差，稳健不受极端值影响
- `zscore`: 基于均值/标准差的 Z 分数法
- `iqr`: 基于四分位距的箱线图法

**异常值替换策略**:
- `clip`: 截断到阈值边界 (默认)
- `nan`: 替换为 NaN
- `median`: 替换为中位数

**OHLCV 约束校验**: 检查 high >= low, high >= open/close, low <= open/close, volume >= 0, DatetimeIndex 单调递增且无重复。

### IndustryLookup — 行业分类查询

基于证监会行业分类，支持行业→股票和股票→行业双向查询。

```python
from data import IndustryLookup

lk = IndustryLookup(data_dir="./data")

stocks = lk.get_stocks("N78公共设施管理业")      # → ["000430.SZ", "000558.SZ", ...]
details = lk.get_stock_details("N78公共设施管理业")  # → DataFrame(code, name)
industry = lk.get_industry("600874.SH")          # → "D46水的生产和供应业"
industries = lk.list_industries()                # → 83个行业列表
```

数据源: Baostock 证监会行业分类，通过 `scripts/download_industry_stock.py` 下载，存储于 `data/industry-stock/industry_stock.csv`。

## 数据下载脚本

### download_daily.py

```bash
# 全量下载（默认从 2000-01-01 开始）
python scripts/download_daily.py

# 增量更新（只下载本地缺失的日期段）
python scripts/download_daily.py --incremental

# 指定股票和日期
python scripts/download_daily.py --symbols 000807.SZ,600036.SH --start 2020-01-01

# 自定义数据目录
python scripts/download_daily.py --data-dir /path/to/data

# 复权方式: qfq(前复权,默认) / hfq(后复权) / 不复权
python scripts/download_daily.py --adjust hfq

# 重试和延迟
python scripts/download_daily.py --retry 5 --delay 0.5
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--incremental` | False | 增量更新模式 |
| `--symbols` | None | 指定股票代码，逗号分隔 |
| `--data-dir` | ./data | 数据存储目录 |
| `--start` | 2000-01-01 | 全量下载起始日期 |
| `--end` | 今天 | 结束日期 |
| `--adjust` | qfq | 复权方式: qfq/hfq/空 |
| `--retry` | 3 | 单只股票失败重试次数 |
| `--delay` | 0.3 | 下载间隔秒数 |

### download_industry_stock.py

```bash
# 下载行业-股票映射（证监会行业分类）
python scripts/download_industry_stock.py
# 输出: data/industry-stock/industry_stock.csv + industry_summary.txt
```

### watchdog_download.py

下载守护脚本，监控下载进程，卡住时自动重启。适用于大批量下载场景。
