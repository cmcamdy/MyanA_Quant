# 实时监控模块 (`src/monitor/`)

## 模块概述

实时监控模块提供策略的在线运行能力：定期轮询最新行情数据、驱动策略产生信号、执行风控检查、更新组合状态，并通过终端仪表盘和告警系统实时展示运行状态。支持状态持久化，重启后可恢复到上次运行状态。

## 模块架构

```
src/monitor/
├── config.py      # MonitorConfig 监控配置
├── poller.py      # DataPoller 数据轮询器
├── live_engine.py # LiveEngine 实时交易引擎
├── state.py       # MonitorState 状态持久化
├── alerts.py      # AlertManager 告警管理
└── dashboard.py   # ConsoleDashboard 终端仪表盘
```

## 核心组件

### MonitorConfig

```python
from monitor import MonitorConfig

config = MonitorConfig(
    symbols=["600036.SH", "000001.SZ"],  # 监控标的列表
    freq="1d",                            # 数据频率
    poll_interval=60,                     # 轮询间隔(秒)
    initial_capital=100000.0,
    commission=0.0003,
    slippage=0.0001,
    state_dir="./data/monitor",           # 状态文件目录
    data_dir="./data",                    # DataManager 数据目录
    data_source=None,                     # 数据源 (None=自动)
    adjust="qfq",
    initial_positions=None,               # Dict[str, {avg_cost, quantity}]
    dashboard_refresh=5,                  # 仪表盘刷新间隔(秒)
    alert_on_signal=True,
    alert_on_risk_trigger=True,
    alert_on_equity_change_pct=0.05,      # 权益变动超过5%告警
)
```

### DataPoller — 数据轮询器

定期轮询 DataManager 获取最新数据，只返回新增的 K 线数据。

```python
from monitor.poller import DataPoller

poller = DataPoller(
    data_manager=mgr,
    symbols=["600036.SH", "000001.SZ"],
    freq="1d",
    poll_interval=60,
)

# 首次加载历史数据
df = poller.warmup("600036.SH", start="2024-01-01", end="2024-12-31")

# 拉取新增数据
new_data = poller.poll()  # → Dict[str, pd.DataFrame]
```

### LiveEngine — 实时交易引擎

主循环：轮询数据 → 驱动策略 → 执行信号 → 更新仪表盘 → 保存状态。

```python
from monitor import LiveEngine
from strategies.examples.ma_cross import MACrossStrategy
from strategies.risk import RiskConfig

strategy = MACrossStrategy(fast_period=5, slow_period=20)
risk_config = RiskConfig(stop_loss_pct=0.05, trailing_stop_pct=0.03)

engine = LiveEngine(
    config=config,
    strategy=strategy,
    risk_config=risk_config,
)

# 启动监控（默认恢复上次状态）
engine.start(resume=True)

# 优雅停止
engine.stop()
```

支持普通 `Strategy`（逐标的调用 `on_bar`）和 `MultiStrategy`（一次调用 `on_bar_multi`）。Ctrl+C 触发优雅退出。

### MonitorState — 状态持久化

原子写入 JSON 文件，防崩溃。支持组合状态和风控状态的完整持久化。

```python
from monitor import MonitorState

# 从 Portfolio 创建
state = MonitorState.from_portfolio(portfolio, monitor_id="test")

# 保存/加载
state.save("./data/monitor/test.json")
loaded = MonitorState.load("./data/monitor/test.json")

# 重建 Portfolio
portfolio = state.to_portfolio()

# RiskManager 状态持久化
state.extract_risk_state(risk_manager)  # 从 RiskManager 提取
state.apply_risk_state(risk_manager)    # 恢复到 RiskManager
```

### AlertManager — 告警管理

可扩展的告警系统，支持多种告警类型和自定义回调。

```python
from monitor import AlertManager, AlertEvent, LoggingCallback

class DingTalkCallback:
    def __call__(self, event: AlertEvent) -> None:
        # 发送钉钉/微信/邮件通知
        ...

alerts = AlertManager(
    on_signal=True,                      # 信号产生时告警
    on_risk_trigger=True,                # 风控触发时告警
    equity_change_threshold=0.05,        # 权益变动超过5%告警
    callbacks=[LoggingCallback(), DingTalkCallback()],
    log_file="./data/monitor/alerts.log",
)
```

**AlertEvent** 属性: `timestamp`, `event_type`("signal"/"risk_trigger"/"equity_change"/"trade"), `symbol`, `details`, `severity`("info"/"warning"/"critical")

### ConsoleDashboard — 终端仪表盘

自动在 `LiveEngine.start()` 中显示，实时展示组合状态：

```
============================================
  Strategy Monitor - MACrossStrategy
  Symbols: 600036.SH, 000001.SZ | Freq: 1d
  Poll: every 60s | Uptime: 2h 15m
============================================

  PORTFOLIO
  Equity: 105,230.45  Cash: 52,115.23  Return: +5.23%

  POSITIONS
  Symbol       Qty     AvgCost   Price    PnL       PnL%
  600036.SH    800     38.50     40.12    +1,296    +4.21%

  RECENT SIGNALS (last 5)
  14:30  600036.SH  BUY   (MA金叉)

  RISK ALERTS (last 3)
  (none)

  EQUITY
  ▂▃▃▄▅▅▆▇▆▇██▇▇█

  Polls: 42 | Last: 2026-05-20 14:30:00
============================================
```

## 监控脚本

```bash
# 基本用法
python scripts/monitor_strategy.py \
    --symbols 600036.SH,000001.SZ \
    --freq 1d --interval 60

# 带风控
python scripts/monitor_strategy.py \
    --symbols 600036.SH \
    --stop-loss 0.05 --trailing-stop 0.03 --take-profit 0.15

# 不恢复上次状态
python scripts/monitor_strategy.py \
    --symbols 600036.SH --no-resume

# 指定数据源
python scripts/monitor_strategy.py \
    --symbols 600036.SH --source baostock
```

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--symbols` | 必填 | 逗号分隔的标的代码 |
| `--freq` | 1d | 数据频率 |
| `--interval` | 60 | 轮询间隔(秒) |
| `--capital` | 100000 | 初始资金 |
| `--fast` | 5 | MA快线周期 |
| `--slow` | 20 | MA慢线周期 |
| `--stop-loss` | 0 | 止损比例 |
| `--trailing-stop` | 0 | 追踪止损比例 |
| `--take-profit` | 0 | 止盈比例 |
| `--max-positions` | 0 | 最大持仓数(0=不限) |
| `--source` | auto | 数据源 |
| `--no-resume` | False | 不恢复上次状态 |
