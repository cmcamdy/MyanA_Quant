"""因子分析模块单元测试"""

import numpy as np
import pandas as pd
import pytest

from factors.base import Factor, FactorResult
from factors.factors import (
    BiasFactor,
    MomentumFactor,
    PriceVolumeFactor,
    ReversalFactor,
    TurnoverFactor,
    VolatilityFactor,
)
from factors.analyzer import FactorAnalyzer, cross_section_ic
from factors.screener import FactorScreener


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_df() -> pd.DataFrame:
    """构造 100 行的单股票行情数据"""
    np.random.seed(42)
    n = 100
    dates = pd.date_range('2020-01-01', periods=n, freq='D')
    close = 100 + np.cumsum(np.random.randn(n) * 0.5)
    volume = 1_000_000 + np.random.randint(-200_000, 200_000, size=n).cumsum()
    volume = np.maximum(volume, 100_000)  # 确保为正
    return pd.DataFrame({
        'date': dates,
        'close': close,
        'volume': volume.astype(float),
    }).reset_index(drop=True)


@pytest.fixture
def analyzer() -> FactorAnalyzer:
    return FactorAnalyzer(forward_period=1)


# ---------------------------------------------------------------------------
# Factor compute tests
# ---------------------------------------------------------------------------

class TestMomentumFactor:
    def test_column_name(self, sample_df: pd.DataFrame) -> None:
        f = MomentumFactor(period=20)
        result = f.compute(sample_df)
        assert result.name == 'momentum_20'

    def test_length_matches_input(self, sample_df: pd.DataFrame) -> None:
        f = MomentumFactor(period=20)
        result = f.compute(sample_df)
        assert len(result) == len(sample_df)

    def test_same_index(self, sample_df: pd.DataFrame) -> None:
        f = MomentumFactor(period=20)
        result = f.compute(sample_df)
        assert result.index.equals(sample_df.index)

    def test_initial_nan(self, sample_df: pd.DataFrame) -> None:
        f = MomentumFactor(period=20)
        result = f.compute(sample_df)
        assert result.iloc[:20].isna().all()

    def test_custom_period(self, sample_df: pd.DataFrame) -> None:
        f = MomentumFactor(period=10)
        result = f.compute(sample_df)
        assert result.name == 'momentum_10'


class TestVolatilityFactor:
    def test_column_name(self, sample_df: pd.DataFrame) -> None:
        f = VolatilityFactor(period=20)
        result = f.compute(sample_df)
        assert result.name == 'volatility_20'

    def test_length_matches_input(self, sample_df: pd.DataFrame) -> None:
        f = VolatilityFactor(period=20)
        result = f.compute(sample_df)
        assert len(result) == len(sample_df)

    def test_non_negative(self, sample_df: pd.DataFrame) -> None:
        f = VolatilityFactor(period=20)
        result = f.compute(sample_df)
        assert (result.dropna() >= 0).all()


class TestTurnoverFactor:
    def test_column_name(self, sample_df: pd.DataFrame) -> None:
        f = TurnoverFactor(period=20)
        result = f.compute(sample_df)
        assert result.name == 'turnover_20'

    def test_length_matches_input(self, sample_df: pd.DataFrame) -> None:
        f = TurnoverFactor(period=20)
        result = f.compute(sample_df)
        assert len(result) == len(sample_df)

    def test_no_inf(self, sample_df: pd.DataFrame) -> None:
        f = TurnoverFactor(period=20)
        result = f.compute(sample_df)
        assert not np.isinf(result.dropna()).any()


class TestReversalFactor:
    def test_column_name(self, sample_df: pd.DataFrame) -> None:
        f = ReversalFactor(period=5)
        result = f.compute(sample_df)
        assert result.name == 'reversal_5'

    def test_length_matches_input(self, sample_df: pd.DataFrame) -> None:
        f = ReversalFactor(period=5)
        result = f.compute(sample_df)
        assert len(result) == len(sample_df)

    def test_negates_pct_change(self, sample_df: pd.DataFrame) -> None:
        f = ReversalFactor(period=5)
        result = f.compute(sample_df)
        expected = -sample_df['close'].pct_change(5)
        pd.testing.assert_series_equal(result, expected, check_names=False)


class TestPriceVolumeFactor:
    def test_column_name(self, sample_df: pd.DataFrame) -> None:
        f = PriceVolumeFactor(period=10)
        result = f.compute(sample_df)
        assert result.name == 'price_volume_10'

    def test_length_matches_input(self, sample_df: pd.DataFrame) -> None:
        f = PriceVolumeFactor(period=10)
        result = f.compute(sample_df)
        assert len(result) == len(sample_df)

    def test_range_minus1_to_1(self, sample_df: pd.DataFrame) -> None:
        f = PriceVolumeFactor(period=10)
        result = f.compute(sample_df)
        valid = result.dropna()
        assert (valid >= -1.01).all() and (valid <= 1.01).all()


class TestBiasFactor:
    def test_column_name(self, sample_df: pd.DataFrame) -> None:
        f = BiasFactor(period=20)
        result = f.compute(sample_df)
        assert result.name == 'bias_20'

    def test_length_matches_input(self, sample_df: pd.DataFrame) -> None:
        f = BiasFactor(period=20)
        result = f.compute(sample_df)
        assert len(result) == len(sample_df)


# ---------------------------------------------------------------------------
# FactorAnalyzer tests
# ---------------------------------------------------------------------------

class TestFactorAnalyzerComputeIC:
    def test_known_data(self) -> None:
        """构造一个因子值与远期收益完全单调相关的场景，|IC| 应接近 1"""
        n = 50
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        close = pd.Series(np.arange(1, n + 1, dtype=float), index=dates)
        volume = pd.Series(np.full(n, 1_000_000.0), index=dates)
        df = pd.DataFrame({'close': close, 'volume': volume})

        analyzer = FactorAnalyzer(forward_period=1)
        factor_values = pd.Series(np.arange(n, dtype=float), index=dates, name='test')
        ic = analyzer.compute_ic(factor_values, df)
        # 因子值与价格单调增，但远期收益 = (i+1)/i - 1 随 i 增大而减小
        # 所以正相关因子与远期收益呈负相关
        assert abs(ic) > 0.5

    def test_negative_correlation(self) -> None:
        """反转因子值后，IC 符号应反转"""
        n = 50
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        close = pd.Series(np.arange(1, n + 1, dtype=float), index=dates)
        volume = pd.Series(np.full(n, 1_000_000_0), index=dates)
        df = pd.DataFrame({'close': close, 'volume': volume})

        analyzer = FactorAnalyzer(forward_period=1)
        factor_values = pd.Series(np.arange(n, dtype=float), index=dates, name='test')
        factor_inv = pd.Series(np.arange(n, 0, -1, dtype=float), index=dates, name='test')
        ic_orig = analyzer.compute_ic(factor_values, df)
        ic_inv = analyzer.compute_ic(factor_inv, df)
        # 反转因子值后 IC 符号应相反
        assert ic_orig * ic_inv < 0


class TestFactorAnalyzerQuantileAnalysis:
    def test_monotonicity_for_momentum(self, sample_df: pd.DataFrame) -> None:
        """对单调上升数据，动量因子的分位收益应大致单调"""
        n = 200
        dates = pd.date_range('2020-01-01', periods=n, freq='D')
        close = pd.Series(np.cumsum(np.ones(n) * 0.1), index=dates)
        volume = pd.Series(np.full(n, 1_000_000.0), index=dates)
        df = pd.DataFrame({'close': close, 'volume': volume}).reset_index(drop=True)

        analyzer = FactorAnalyzer(forward_period=1)
        factor = MomentumFactor(period=20)
        factor_values = factor.compute(df)
        qr = analyzer.quantile_analysis(factor_values, df, n_quantiles=5)
        # 在趋势向上时，高动量分位的收益应大于低动量分位
        assert qr[5] > qr[1]


class TestFactorAnalyzerAnalyze:
    def test_returns_factor_result(self, sample_df: pd.DataFrame, analyzer: FactorAnalyzer) -> None:
        factor = MomentumFactor(period=20)
        result = analyzer.analyze(factor, sample_df)
        assert isinstance(result, FactorResult)
        assert result.factor_name == 'momentum_20'
        assert isinstance(result.values, pd.Series)
        assert isinstance(result.ic, float)
        # IR 可能为 NaN（数据不足），但不应抛异常
        assert isinstance(result.quantile_returns, dict)
        assert len(result.quantile_returns) == 5

    def test_valid_ic(self, sample_df: pd.DataFrame, analyzer: FactorAnalyzer) -> None:
        factor = MomentumFactor(period=20)
        result = analyzer.analyze(factor, sample_df)
        assert not np.isnan(result.ic)


class TestFactorAnalyzerAnalyzeBatch:
    def test_returns_dataframe(self, sample_df: pd.DataFrame, analyzer: FactorAnalyzer) -> None:
        factors = [MomentumFactor(20), VolatilityFactor(20), ReversalFactor(5)]
        result_df = analyzer.analyze_batch(factors, sample_df)
        assert isinstance(result_df, pd.DataFrame)
        assert len(result_df) == 3
        assert 'factor_name' in result_df.columns
        assert 'ic' in result_df.columns
        assert 'ir' in result_df.columns
        assert 'q1_return' in result_df.columns
        assert 'q5_return' in result_df.columns


# ---------------------------------------------------------------------------
# FactorScreener tests
# ---------------------------------------------------------------------------

class TestFactorScreenerScore:
    def test_with_multiple_factors(self, sample_df: pd.DataFrame) -> None:
        factors = [MomentumFactor(20), VolatilityFactor(20)]
        screener = FactorScreener(factors=factors)
        scores = screener.score(sample_df)
        assert isinstance(scores, pd.Series)
        assert len(scores) == len(sample_df)
        assert scores.name == 'composite_score'

    def test_with_weights(self, sample_df: pd.DataFrame) -> None:
        factors = [MomentumFactor(20), VolatilityFactor(20)]
        weights = {'momentum_20': 2.0, 'volatility_20': 1.0}
        screener = FactorScreener(factors=factors, weights=weights)
        scores = screener.score(sample_df)
        assert isinstance(scores, pd.Series)

    def test_equal_weights_default(self, sample_df: pd.DataFrame) -> None:
        factors = [MomentumFactor(20), VolatilityFactor(20)]
        screener = FactorScreener(factors=factors)
        scores = screener.score(sample_df)
        # 非全 NaN
        assert scores.dropna().shape[0] > 0


class TestFactorScreenerRank:
    def test_returns_correct_number(self) -> None:
        """构造多股票数据，验证 rank 返回数量"""
        n_per_stock = 50
        frames = []
        for symbol in ['A', 'B', 'C', 'D', 'E']:
            np.random.seed(hash(symbol) % 2**31)
            close = 100 + np.cumsum(np.random.randn(n_per_stock) * 0.5)
            volume = 1_000_000 + np.random.randint(-100_000, 100_000, size=n_per_stock).cumsum()
            volume = np.maximum(volume, 100_000).astype(float)
            df = pd.DataFrame({
                'close': close,
                'volume': volume,
                'symbol': symbol,
            })
            frames.append(df)
        df = pd.concat(frames, ignore_index=True)

        factors = [MomentumFactor(20)]
        screener = FactorScreener(factors=factors)
        top3 = screener.rank(df, top_n=3)
        assert len(top3) == 3
        assert all(s in ['A', 'B', 'C', 'D', 'E'] for s in top3)


class TestFactorScreenerFilter:
    def test_filter_by_condition(self, sample_df: pd.DataFrame) -> None:
        factors = [MomentumFactor(20)]
        screener = FactorScreener(factors=factors)
        filtered = screener.filter(
            sample_df,
            condition=lambda df: df['momentum_20'] > 0,
        )
        assert isinstance(filtered, pd.DataFrame)
        assert len(filtered) < len(sample_df)
        assert (filtered['momentum_20'] > 0).all()


# ---------------------------------------------------------------------------
# Cross-section IC test
# ---------------------------------------------------------------------------

class TestCrossSectionIC:
    def test_returns_series(self) -> None:
        dates = pd.date_range('2020-01-01', periods=20, freq='D')
        stocks = ['A', 'B', 'C', 'D']
        np.random.seed(0)
        factor_df = pd.DataFrame(
            np.random.randn(20, 4),
            index=dates,
            columns=stocks,
        )
        return_df = pd.DataFrame(
            np.random.randn(20, 4),
            index=dates,
            columns=stocks,
        )
        result = cross_section_ic(factor_df, return_df)
        assert isinstance(result, pd.Series)
        assert len(result) == 20
