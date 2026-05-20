"""ReportGenerator单元测试"""

import os
import tempfile

import pandas as pd
import pytest

from strategies.result import BacktestResult, TradeRecord
from reports.generator import ReportGenerator


def _make_result(trades=None):
    """构造测试用的BacktestResult"""
    dates = pd.date_range('2024-01-01', periods=20, freq='D')
    equity_curve = pd.DataFrame(
        {
            'equity': [100000 + i * 100 for i in range(20)],
            'cash': [50000] * 20,
            'market_value': [50000 + i * 100 for i in range(20)],
        },
        index=dates,
    )
    return BacktestResult(
        total_return=0.10,
        annual_return=0.25,
        sharpe_ratio=1.5,
        max_drawdown=0.05,
        max_drawdown_duration=3,
        win_rate=0.6,
        profit_loss_ratio=2.0,
        total_trades=5,
        equity_curve=equity_curve,
        trades=trades if trades is not None else [],
        initial_capital=100000.0,
    )


def _make_trades():
    """构造测试用的交易记录"""
    return [
        TradeRecord(
            symbol='AAPL',
            entry_time=pd.Timestamp('2024-01-05'),
            exit_time=pd.Timestamp('2024-01-10'),
            entry_price=150.0,
            exit_price=155.0,
            quantity=100,
            pnl=500.0,
            commission=1.0,
        ),
        TradeRecord(
            symbol='GOOG',
            entry_time=pd.Timestamp('2024-01-08'),
            exit_time=pd.Timestamp('2024-01-12'),
            entry_price=2800.0,
            exit_price=2750.0,
            quantity=10,
            pnl=-500.0,
            commission=1.5,
        ),
        TradeRecord(
            symbol='MSFT',
            entry_time=pd.Timestamp('2024-01-11'),
            exit_time=pd.Timestamp('2024-01-15'),
            entry_price=300.0,
            exit_price=310.0,
            quantity=50,
            pnl=500.0,
            commission=0.8,
        ),
    ]


class TestReportGeneratorGenerate:
    """测试generate方法（保存文件）"""

    def test_generate_creates_file(self):
        result = _make_result()
        gen = ReportGenerator()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'report.html')
            html = gen.generate(result, output_path=path)
            assert os.path.isfile(path)
            assert len(html) > 0

    def test_generate_file_content_matches_return(self):
        result = _make_result()
        gen = ReportGenerator()
        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, 'report.html')
            html = gen.generate(result, output_path=path)
            with open(path, 'r', encoding='utf-8') as f:
                file_content = f.read()
            assert html == file_content


class TestReportGeneratorToHtml:
    """测试to_html方法"""

    def test_to_html_returns_string(self):
        result = _make_result()
        gen = ReportGenerator()
        html = gen.to_html(result)
        assert isinstance(html, str)

    def test_html_contains_doctype_and_tags(self):
        result = _make_result()
        gen = ReportGenerator()
        html = gen.to_html(result)
        assert '<!DOCTYPE html>' in html
        assert '<html>' in html or '<html ' in html
        assert '</html>' in html

    def test_html_contains_title(self):
        result = _make_result()
        gen = ReportGenerator(title='测试报告')
        html = gen.to_html(result)
        assert '测试报告' in html
        assert '<title>测试报告</title>' in html

    def test_html_contains_timestamp(self):
        result = _make_result()
        gen = ReportGenerator()
        html = gen.to_html(result)
        assert '生成时间:' in html


class TestReportGeneratorMetrics:
    """测试指标渲染"""

    def test_html_contains_key_metrics(self):
        result = _make_result()
        gen = ReportGenerator()
        html = gen.to_html(result)
        assert '总收益率' in html
        assert '10.00%' in html
        assert '年化收益率' in html
        assert '25.00%' in html
        assert '夏普比率' in html
        assert '1.50' in html
        assert '最大回撤' in html
        assert '5.00%' in html
        assert '胜率' in html
        assert '60.00%' in html
        assert '盈亏比' in html
        assert '2.00' in html
        assert '总交易次数' in html
        assert '初始资金' in html
        assert '100,000.00' in html

    def test_metrics_table_has_metric_value_class(self):
        result = _make_result()
        gen = ReportGenerator()
        html = gen.to_html(result)
        assert 'metric-value' in html


class TestReportGeneratorCharts:
    """测试图表渲染"""

    def test_html_contains_svg_equity_chart(self):
        result = _make_result()
        gen = ReportGenerator()
        html = gen.to_html(result)
        assert '<svg' in html
        assert '权益曲线' in html

    def test_html_contains_svg_drawdown_chart(self):
        result = _make_result()
        gen = ReportGenerator()
        html = gen.to_html(result)
        assert '回撤曲线' in html


class TestReportGeneratorTrades:
    """测试交易记录渲染"""

    def test_empty_trades_shows_message(self):
        result = _make_result(trades=[])
        gen = ReportGenerator()
        html = gen.to_html(result)
        assert '无交易记录' in html

    def test_trades_table_shows_trade_data(self):
        trades = _make_trades()
        result = _make_result(trades=trades)
        gen = ReportGenerator()
        html = gen.to_html(result)
        assert 'AAPL' in html
        assert 'GOOG' in html
        assert 'MSFT' in html
        assert '500.00' in html
        assert '-500.00' in html

    def test_winning_trade_has_positive_class(self):
        trades = _make_trades()
        result = _make_result(trades=trades)
        gen = ReportGenerator()
        html = gen.to_html(result)
        assert 'class="positive"' in html

    def test_losing_trade_has_negative_class(self):
        trades = _make_trades()
        result = _make_result(trades=trades)
        gen = ReportGenerator()
        html = gen.to_html(result)
        assert 'class="negative"' in html

    def test_pnl_chart_with_trades(self):
        trades = _make_trades()
        result = _make_result(trades=trades)
        gen = ReportGenerator()
        html = gen.to_html(result)
        assert '交易盈亏分布' in html

    def test_pnl_chart_without_trades(self):
        result = _make_result(trades=[])
        gen = ReportGenerator()
        html = gen.to_html(result)
        assert '无盈亏数据' in html

    def test_long_trade_list_truncation(self):
        """测试超过100条交易时截断显示"""
        trades = [
            TradeRecord(
                symbol=f'STOCK{i}',
                entry_time=pd.Timestamp('2024-01-01'),
                exit_time=pd.Timestamp('2024-01-05'),
                entry_price=100.0,
                exit_price=101.0,
                quantity=10,
                pnl=10.0,
                commission=0.5,
            )
            for i in range(150)
        ]
        result = _make_result(trades=trades)
        gen = ReportGenerator()
        html = gen.to_html(result)
        assert '显示前100条交易' in html
        assert '共150条' in html
        # First trade should be present
        assert 'STOCK0' in html
        # Trade #100 should not be present (0-indexed, only 0-99 shown)
        assert 'STOCK100' not in html


class TestReportGeneratorDefaultTitle:
    """测试默认标题"""

    def test_default_title(self):
        result = _make_result()
        gen = ReportGenerator()
        html = gen.to_html(result)
        assert '<h1>回测报告</h1>' in html
        assert '<title>回测报告</title>' in html
