"""HTML回测报告生成器"""

from datetime import datetime
from io import StringIO
from typing import List

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from strategies.result import BacktestResult, TradeRecord


class ReportGenerator:
    """从BacktestResult生成自包含的HTML报告"""

    def __init__(self, title: str = "回测报告"):
        self._title = title

    def generate(self, result: BacktestResult, output_path: str = "report.html") -> str:
        """生成HTML报告，返回HTML字符串并保存到文件"""
        html = self.to_html(result)
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(html)
        return html

    def to_html(self, result: BacktestResult) -> str:
        """返回HTML字符串，不保存文件"""
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        metrics_html = self._render_metrics(result)
        equity_html = self._render_equity_chart(result)
        drawdown_html = self._render_drawdown_chart(result)
        trades_html = self._render_trades_table(result)
        pnl_html = self._render_pnl_chart(result)

        html = _HTML_TEMPLATE.format(
            title=self._title,
            timestamp=timestamp,
            metrics=metrics_html,
            equity_chart=equity_html,
            drawdown_chart=drawdown_html,
            trades_table=trades_html,
            pnl_chart=pnl_html,
        )
        return html

    def _render_metrics(self, result: BacktestResult) -> str:
        """渲染指标汇总表格"""
        metrics = [
            ('总收益率', f'{result.total_return:.2%}'),
            ('年化收益率', f'{result.annual_return:.2%}'),
            ('夏普比率', f'{result.sharpe_ratio:.2f}'),
            ('最大回撤', f'{result.max_drawdown:.2%}'),
            ('最大回撤持续', f'{result.max_drawdown_duration} 天'),
            ('胜率', f'{result.win_rate:.2%}'),
            ('盈亏比', f'{result.profit_loss_ratio:.2f}'),
            ('总交易次数', f'{result.total_trades}'),
            ('初始资金', f'{result.initial_capital:,.2f}'),
        ]
        rows = ''
        for label, value in metrics:
            rows += f'<tr><td>{label}</td><td class="metric-value">{value}</td></tr>\n'
        return f'<table>\n{rows}</table>'

    def _render_equity_chart(self, result: BacktestResult) -> str:
        """渲染权益曲线为嵌入式SVG"""
        if result.equity_curve is None or result.equity_curve.empty:
            return '<p>无权益曲线数据</p>'

        fig, ax = plt.subplots(figsize=(10, 4))
        equity = result.equity_curve
        if 'equity' in equity.columns:
            ax.plot(equity.index, equity['equity'], label='权益', color='#2563eb')
        if 'cash' in equity.columns:
            ax.plot(equity.index, equity['cash'], label='现金', color='#16a34a', linestyle='--')
        ax.set_title('权益曲线')
        ax.set_xlabel('日期')
        ax.set_ylabel('金额')
        ax.legend()
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()

        buf = StringIO()
        fig.savefig(buf, format='svg')
        plt.close(fig)
        return buf.getvalue()

    def _render_drawdown_chart(self, result: BacktestResult) -> str:
        """渲染回撤曲线为嵌入式SVG"""
        if result.equity_curve is None or result.equity_curve.empty:
            return '<p>无回撤数据</p>'

        equity = result.equity_curve
        if 'equity' not in equity.columns:
            return '<p>无回撤数据</p>'

        equity_series = equity['equity']
        cummax = equity_series.cummax()
        drawdown = (cummax - equity_series) / cummax

        fig, ax = plt.subplots(figsize=(10, 3))
        ax.fill_between(drawdown.index, drawdown.values, color='#dc2626', alpha=0.4)
        ax.plot(drawdown.index, drawdown.values, color='#dc2626', linewidth=0.8)
        ax.set_title('回撤曲线')
        ax.set_xlabel('日期')
        ax.set_ylabel('回撤')
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda y, _: f'{y:.0%}'))
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()

        buf = StringIO()
        fig.savefig(buf, format='svg')
        plt.close(fig)
        return buf.getvalue()

    def _render_trades_table(self, result: BacktestResult) -> str:
        """渲染交易记录表格"""
        trades: List[TradeRecord] = result.trades
        if not trades:
            return '<p>无交易记录</p>'

        show_trades = trades[:100]
        note = ''
        if len(trades) > 100:
            note = f'<p>显示前100条交易，共{len(trades)}条</p>'

        rows = ''
        for t in show_trades:
            pnl_class = 'positive' if t.pnl >= 0 else 'negative'
            entry_time = t.entry_time.strftime('%Y-%m-%d %H:%M') if t.entry_time is not None else ''
            exit_time = t.exit_time.strftime('%Y-%m-%d %H:%M') if t.exit_time is not None else ''
            entry_price = f'{t.entry_price:.2f}' if t.entry_price is not None else ''
            exit_price = f'{t.exit_price:.2f}' if t.exit_price is not None else ''
            rows += (
                f'<tr>'
                f'<td style="text-align:left">{t.symbol}</td>'
                f'<td>{entry_time}</td>'
                f'<td>{exit_time}</td>'
                f'<td>{entry_price}</td>'
                f'<td>{exit_price}</td>'
                f'<td>{t.quantity:.0f}</td>'
                f'<td class="{pnl_class}">{t.pnl:.2f}</td>'
                f'<td>{t.commission:.2f}</td>'
                f'</tr>\n'
            )

        table = (
            '<table>\n'
            '<tr>'
            '<th>标的</th><th>入场时间</th><th>出场时间</th>'
            '<th>入场价</th><th>出场价</th><th>数量</th>'
            '<th>盈亏</th><th>手续费</th>'
            '</tr>\n'
            f'{rows}'
            '</table>\n'
        )
        return note + table

    def _render_pnl_chart(self, result: BacktestResult) -> str:
        """渲染交易盈亏分布柱状图为嵌入式SVG"""
        trades: List[TradeRecord] = result.trades
        if not trades:
            return '<p>无盈亏数据</p>'

        pnls = [t.pnl for t in trades]
        fig, ax = plt.subplots(figsize=(10, 4))
        colors = ['#16a34a' if p >= 0 else '#dc2626' for p in pnls]
        x = list(range(len(pnls)))
        ax.bar(x, pnls, color=colors, alpha=0.8)
        ax.set_title('交易盈亏分布')
        ax.set_xlabel('交易序号')
        ax.set_ylabel('盈亏')
        ax.grid(True, alpha=0.3, axis='y')
        fig.tight_layout()

        buf = StringIO()
        fig.savefig(buf, format='svg')
        plt.close(fig)
        return buf.getvalue()


_HTML_TEMPLATE = """<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{title}</title>
    <style>
        body {{ font-family: -apple-system, sans-serif; max-width: 1200px; margin: 0 auto; padding: 20px; }}
        h1, h2 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: right; }}
        th {{ background-color: #f5f5f5; }}
        .positive {{ color: #16a34a; }}
        .negative {{ color: #dc2626; }}
        .metric-value {{ font-weight: bold; font-size: 1.1em; }}
        .chart {{ margin: 20px 0; }}
        .section {{ margin: 30px 0; }}
    </style>
</head>
<body>
    <h1>{title}</h1>
    <p>生成时间: {timestamp}</p>

    <div class="section">
        <h2>绩效指标</h2>
        {metrics}
    </div>

    <div class="section chart">
        <h2>权益曲线</h2>
        {equity_chart}
    </div>

    <div class="section chart">
        <h2>回撤曲线</h2>
        {drawdown_chart}
    </div>

    <div class="section">
        <h2>交易记录</h2>
        {trades_table}
    </div>

    <div class="section chart">
        <h2>交易盈亏分布</h2>
        {pnl_chart}
    </div>
</body>
</html>"""
