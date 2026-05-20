#!/usr/bin/env python3
"""JYS五维选股可视化系统 — Streamlit Web应用

启动: streamlit run web/app.py
"""

import concurrent.futures
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from stock_selection import JYSScorer, JYSScreener, TencentFetcher, load_codes_by_market
from stock_selection.jys_scorer import JYSScoreResult
from stock_selection.registry import available_methods, create_screener

# ---------------------------------------------------------------------------
# 页面配置
# ---------------------------------------------------------------------------

st.set_page_config(
    page_title="五维选股系统",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# 常量
# ---------------------------------------------------------------------------

MARKET_OPTIONS = {
    "沪深300": "csi300",
    "A股全量": "a",
    "港股通": "hk",
    "A股+港股通": "all",
}

METHOD_LABELS = {
    "jys": "JYS五维评分",
    "factor": "因子选股",
    "trend": "趋势选股",
}

DIMENSION_NAMES = ["技术面(30)", "估值面(25)", "盈利质量(30)", "安全性(10)", "分红(5)"]
DIMENSION_KEYS = ["tech_score", "valuation_score", "profit_score", "safety_score", "dividend_score"]
DIMENSION_MAX = [30, 25, 30, 10, 5]
DIMENSION_COLORS = ["#4C78A8", "#54A24B", "#EECA3B", "#E45756", "#B279A2"]

GRADE_ORDER = ["A+", "A", "B+", "B", "C", "D"]
GRADE_COLORS = {
    "A+": "#2ecc71", "A": "#27ae60", "B+": "#f39c12",
    "B": "#e67e22", "C": "#e74c3c", "D": "#c0392b",
}

PLOTLY_FONT = dict(family="PingFang SC, Microsoft YaHei, Arial, sans-serif")

# ---------------------------------------------------------------------------
# 缓存的筛选函数
# ---------------------------------------------------------------------------


@st.cache_data(ttl=3600, show_spinner=False)
def cached_screening(
    market: str,
    top_n: int,
    min_score: int,
    max_pe: float,
    min_turnover: float,
    min_price: float,
    max_concurrent: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """批量筛选（带缓存），在独立线程中运行以兼容 Streamlit 事件循环"""
    def _work():
        screener = JYSScreener(
            top_n=top_n,
            min_score=min_score,
            max_pe=max_pe,
            min_turnover=min_turnover,
            min_price=min_price,
            max_concurrent=max_concurrent,
            market=market,
        )
        return screener.screen_all()

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_work)
        return future.result()


@st.cache_data(ttl=3600, show_spinner=False)
def cached_fetch(codes_tuple: tuple) -> List[Dict]:
    """批量获取行情数据（带缓存）"""
    codes = list(codes_tuple)
    fetcher = TencentFetcher(max_concurrent=20)

    def _work():
        return fetcher.fetch_all_sync(codes)

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(_work)
        return future.result()


# ---------------------------------------------------------------------------
# 图表构建函数
# ---------------------------------------------------------------------------


def build_radar_chart(result: JYSScoreResult) -> go.Figure:
    """五维雷达图"""
    scores = [getattr(result, k) for k in DIMENSION_KEYS]

    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=DIMENSION_MAX,
        theta=DIMENSION_NAMES,
        fill="none",
        line=dict(color="lightgray", dash="dash", width=1),
        name="满分",
    ))
    fig.add_trace(go.Scatterpolar(
        r=scores,
        theta=DIMENSION_NAMES,
        fill="toself",
        fillcolor="rgba(76, 120, 168, 0.3)",
        line=dict(color="#4C78A8", width=2),
        name="实际得分",
    ))
    fig.update_layout(
        polar=dict(radialaxis=dict(visible=True, range=[0, max(DIMENSION_MAX)])),
        showlegend=True,
        margin=dict(l=20, r=20, t=30, b=20),
        font=PLOTLY_FONT,
    )
    return fig


def build_dimension_bars(result: JYSScoreResult) -> go.Figure:
    """五维水平柱状图"""
    scores = [getattr(result, k) for k in DIMENSION_KEYS]
    labels = [f"{n}: {s}/{m}" for n, s, m in zip(DIMENSION_NAMES, scores, DIMENSION_MAX)]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        y=labels,
        x=scores,
        orientation="h",
        marker_color=DIMENSION_COLORS,
        text=[f"{s}分" for s in scores],
        textposition="auto",
    ))
    fig.update_layout(
        xaxis_title="得分",
        margin=dict(l=120, r=20, t=30, b=20),
        height=250,
        font=PLOTLY_FONT,
    )
    return fig


def build_grade_distribution(all_df: pd.DataFrame) -> go.Figure:
    """评级分布柱状图"""
    grade_counts = all_df["grade"].value_counts()
    grades = [g for g in GRADE_ORDER if g in grade_counts.index]
    counts = [grade_counts[g] for g in grades]
    colors = [GRADE_COLORS[g] for g in grades]

    fig = go.Figure(go.Bar(
        x=grades,
        y=counts,
        marker_color=colors,
        text=counts,
        textposition="auto",
    ))
    fig.update_layout(
        title="评级分布",
        xaxis_title="评级",
        yaxis_title="数量",
        margin=dict(l=20, r=20, t=40, b=20),
        font=PLOTLY_FONT,
    )
    return fig


def build_market_distribution(all_df: pd.DataFrame) -> go.Figure:
    """市场分布饼图"""
    market_counts = all_df["market"].value_counts()
    labels = ["A股" if m == "A" else "港股通" for m in market_counts.index]

    fig = go.Figure(go.Pie(
        labels=labels,
        values=market_counts.values,
        hole=0.4,
        marker_colors=["#4C78A8", "#E45756"],
    ))
    fig.update_layout(
        title="市场分布",
        margin=dict(l=20, r=20, t=40, b=20),
        font=PLOTLY_FONT,
    )
    return fig


def build_score_histogram(all_df: pd.DataFrame, min_score: int) -> go.Figure:
    """综合得分直方图"""
    fig = go.Figure(go.Histogram(
        x=all_df["composite_score"],
        nbinsx=25,
        marker_color="#4C78A8",
        opacity=0.7,
    ))
    fig.add_vline(
        x=min_score,
        line_dash="dash",
        line_color="red",
        annotation_text=f"最低分 {min_score}",
    )
    fig.update_layout(
        title="综合得分分布",
        xaxis_title="综合得分",
        yaxis_title="数量",
        margin=dict(l=20, r=20, t=40, b=20),
        font=PLOTLY_FONT,
    )
    return fig


def build_pe_score_scatter(all_df: pd.DataFrame) -> go.Figure:
    """PE vs 综合得分散点图"""
    plot_df = all_df.dropna(subset=["pe_ratio"]).copy()
    plot_df = plot_df[plot_df["pe_ratio"] > 0]
    plot_df = plot_df[plot_df["pe_ratio"] <= 50]

    fig = px.scatter(
        plot_df,
        x="pe_ratio",
        y="composite_score",
        color="grade",
        category_orders={"grade": GRADE_ORDER},
        color_discrete_map=GRADE_COLORS,
        hover_data=["symbol", "name"],
        labels={"pe_ratio": "PE", "composite_score": "综合得分", "grade": "评级"},
    )
    fig.update_layout(
        title="PE vs 综合得分",
        margin=dict(l=20, r=20, t=40, b=20),
        font=PLOTLY_FONT,
    )
    return fig


# ---------------------------------------------------------------------------
# UI 渲染函数
# ---------------------------------------------------------------------------


def render_sidebar() -> Dict:
    """渲染侧边栏，返回配置字典"""
    with st.sidebar:
        st.title("📊 五维选股系统")

        # ---- 选股方法 ----
        methods = available_methods()
        method_labels = [METHOD_LABELS.get(m, m) for m in methods]
        selected_label = st.selectbox("选股方法", method_labels, index=0)
        method = methods[method_labels.index(selected_label)]
        st.session_state["method"] = method

        st.divider()

        # ---- JYS 参数 ----
        config = {}
        if method == "jys":
            st.subheader("JYS 参数")

            market_label = st.selectbox(
                "市场", list(MARKET_OPTIONS.keys()), index=0,
            )
            config["market"] = MARKET_OPTIONS[market_label]

            config["top_n"] = st.number_input("返回数量 Top N", min_value=1, max_value=100, value=10)
            config["min_score"] = st.slider("最低综合分", 0, 100, 40)
            config["max_pe"] = st.number_input("最大PE", value=30.0, min_value=1.0, max_value=200.0)
            config["min_turnover"] = st.number_input("最低换手率(%)", value=0.3, min_value=0.0, max_value=50.0, step=0.1)
            config["min_price"] = st.number_input("最低股价", value=1.0, min_value=0.0, step=0.5)
            config["max_concurrent"] = st.slider("并发数", 5, 50, 20)

            # 大池预估提示
            pool_sizes = {"csi300": "~300", "a": "~5000", "hk": "~800", "all": "~5800"}
            pool_hint = pool_sizes.get(config["market"], "")
            if config["market"] in ("a", "all"):
                st.info(f"股票池 {pool_hint} 只，预计耗时 15-30 秒")
            elif config["market"] == "csi300":
                st.info(f"股票池 {pool_hint} 只，预计耗时 3-5 秒")

        st.divider()

        # ---- 单股评分 ----
        st.subheader("单股评分")
        single_code = st.text_input("股票代码", placeholder="如 601318, 00700")
        score_btn = st.button("🔍 评分", use_container_width=True)

        st.divider()

        # ---- 批量筛选 ----
        st.subheader("批量筛选")
        screen_btn = st.button("🚀 开始筛选", type="primary", use_container_width=True)

        st.divider()

        # ---- 缓存 ----
        st.subheader("数据缓存")
        if "last_fetch_time" in st.session_state:
            st.caption(f"上次获取: {st.session_state['last_fetch_time']}")
        else:
            st.caption("尚未获取数据")
        if st.button("🗑️ 清除缓存"):
            st.cache_data.clear()
            for key in ["top_df", "all_df", "elapsed", "last_fetch_time", "single_result"]:
                st.session_state.pop(key, None)
            st.toast("缓存已清除")

    # 保存按钮状态供主区域使用
    st.session_state["_score_btn"] = score_btn
    st.session_state["_screen_btn"] = screen_btn
    st.session_state["_single_code"] = single_code

    return config


def render_welcome():
    """欢迎页"""
    st.header("欢迎使用五维选股系统")
    st.markdown("""
    ### 评分体系 (满分100分)

    | 维度 | 满分 | 组成 |
    |------|------|------|
    | 技术面 | 30 | 日涨跌幅(10) + 20日动量(15) + 换手率(5) |
    | 估值面 | 25 | PE(10) + PB(10) + PR市赚率(5) |
    | 盈利质量 | 30 | ROE(15) + 利润增长(15) |
    | 安全性 | 10 | PB安全边际(3) + 股息稳定性(3) + 换手率波动(4) |
    | 分红 | 5 | 股息率(5) |

    **评级**: A+(85-100) / A(75-84) / B+(65-74) / B(55-64) / C(45-54) / D(<45)

    ---

    👈 在左侧选择市场、调整参数，然后点击 **开始筛选** 或输入代码进行 **单股评分**
    """)


def render_single_result(result: JYSScoreResult):
    """渲染单股评分结果"""
    col1, col2, col3 = st.columns(3)
    col1.metric("综合得分", f"{result.total_score}分")
    col2.metric("评级", result.grade)
    col3.metric("市场", "A股" if result.market == "A" else "港股通")

    col_radar, col_bars = st.columns([1, 1])
    with col_radar:
        st.subheader("五维雷达图")
        st.plotly_chart(build_radar_chart(result), use_container_width=True)
    with col_bars:
        st.subheader("维度得分")
        st.plotly_chart(build_dimension_bars(result), use_container_width=True)

    with st.expander("评分明细"):
        detail = result.detail
        detail_data = {
            "子项": list(detail.keys()),
            "得分": list(detail.values()),
        }
        st.dataframe(pd.DataFrame(detail_data), use_container_width=True, hide_index=True)

        st.markdown(f"**原始数据**: 价格={result.stock_data.get('price', 'N/A')}, "
                     f"PE={result.stock_data.get('pe_ratio', 'N/A')}, "
                     f"PB={result.stock_data.get('pb_ratio', 'N/A')}, "
                     f"ROE={result.stock_data.get('roe', 'N/A')}%, "
                     f"股息率={result.stock_data.get('dividend_yield', 'N/A')}%")


def render_jys_results(top_df: pd.DataFrame, all_df: pd.DataFrame, elapsed: float, config: Dict):
    """渲染 JYS 方法的结果（三个 Tab）"""
    tab_result, tab_detail, tab_stats = st.tabs(["📋 筛选结果", "🎯 五维分析", "📊 分布统计"])

    with tab_result:
        # 摘要指标
        pool_size = len(load_codes_by_market(config["market"]))
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("分析", f"{pool_size} 只")
        col2.metric("过滤后", f"{len(all_df)} 只")
        col3.metric("入选", f"{len(top_df)} 只")
        col4.metric("耗时", f"{elapsed:.1f}s")

        if top_df.empty:
            st.warning("未找到符合条件的标的，请尝试放宽筛选条件")
            return

        # 结果表格
        display_cols = [
            "symbol", "name", "market", "composite_score", "grade",
            "tech_score", "valuation_score", "profit_score", "safety_score", "dividend_score",
            "pe_ratio", "pb_ratio", "roe", "dividend_yield",
        ]
        existing_cols = [c for c in display_cols if c in top_df.columns]
        display_df = top_df[existing_cols].copy()

        # 重命名列
        rename_map = {
            "symbol": "代码", "name": "名称", "market": "市场",
            "composite_score": "综合分", "grade": "评级",
            "tech_score": "技术面", "valuation_score": "估值面",
            "profit_score": "盈利质量", "safety_score": "安全性", "dividend_score": "分红",
            "pe_ratio": "PE", "pb_ratio": "PB", "roe": "ROE%", "dividend_yield": "股息率%",
        }
        display_df = display_df.rename(columns={k: v for k, v in rename_map.items() if k in display_df.columns})

        st.dataframe(display_df, use_container_width=True, height=400)

        # CSV 下载
        csv = top_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button(
            "📥 下载 CSV",
            data=csv,
            file_name=f"jys_screen_{config['market']}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv",
        )

    with tab_detail:
        if top_df.empty:
            st.warning("无数据")
            return

        # 股票选择器
        stock_options = [
            f"{row['symbol']} - {row['name']} ({row['composite_score']}分 {row['grade']})"
            for _, row in top_df.iterrows()
        ]
        selected_idx = st.selectbox("选择股票", range(len(stock_options)), format_func=lambda i: stock_options[i])
        selected_row = top_df.iloc[selected_idx]

        # 获取完整 JYSScoreResult
        if "all_results_raw" in st.session_state and st.session_state["all_results_raw"]:
            raw_results = st.session_state["all_results_raw"]
            match = [r for r in raw_results if r.code == selected_row.get("symbol", "").split(".")[0].lstrip("0") or r.name == selected_row["name"]]
            if match:
                render_single_result(match[0])
            else:
                _render_detail_from_row(selected_row)
        else:
            _render_detail_from_row(selected_row)

    with tab_stats:
        if all_df.empty:
            st.warning("无数据")
            return

        col1, col2 = st.columns(2)
        with col1:
            st.plotly_chart(build_grade_distribution(all_df), use_container_width=True)
            st.plotly_chart(build_market_distribution(all_df), use_container_width=True)
        with col2:
            st.plotly_chart(build_score_histogram(all_df, config.get("min_score", 40)), use_container_width=True)
            st.plotly_chart(build_pe_score_scatter(all_df), use_container_width=True)


def _render_detail_from_row(row: pd.Series):
    """从 DataFrame 行数据构建简化的五维展示"""
    scores = [row.get(k, 0) for k in DIMENSION_KEYS]

    col_radar, col_bars = st.columns([1, 1])
    with col_radar:
        st.subheader("五维雷达图")

        fig = go.Figure()
        fig.add_trace(go.Scatterpolar(
            r=DIMENSION_MAX,
            theta=DIMENSION_NAMES,
            fill="none",
            line=dict(color="lightgray", dash="dash", width=1),
            name="满分",
        ))
        fig.add_trace(go.Scatterpolar(
            r=scores,
            theta=DIMENSION_NAMES,
            fill="toself",
            fillcolor="rgba(76, 120, 168, 0.3)",
            line=dict(color="#4C78A8", width=2),
            name="实际得分",
        ))
        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, max(DIMENSION_MAX)])),
            showlegend=True,
            margin=dict(l=20, r=20, t=30, b=20),
            font=PLOTLY_FONT,
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_bars:
        st.subheader("维度得分")
        labels = [f"{n}: {s}/{m}" for n, s, m in zip(DIMENSION_NAMES, scores, DIMENSION_MAX)]
        fig = go.Figure(go.Bar(
            y=labels,
            x=scores,
            orientation="h",
            marker_color=DIMENSION_COLORS,
            text=[f"{s}分" for s in scores],
            textposition="auto",
        ))
        fig.update_layout(
            xaxis_title="得分",
            margin=dict(l=120, r=20, t=30, b=20),
            height=250,
            font=PLOTLY_FONT,
        )
        st.plotly_chart(fig, use_container_width=True)


def render_factor_placeholder():
    """因子选股占位页"""
    st.header("因子选股")
    st.info("🚧 因子选股功能即将上线，敬请期待...")
    st.markdown("""
    **因子选股方法**基于本地 Parquet 历史数据，计算量化因子（动量、波动率、换手率、反转、价量等），
    通过 z-score 标准化和加权评分进行筛选。

    相比 JYS 五维评分：
    - JYS 侧重**基本面**（PE/PB/ROE/股息率）+ 实时行情
    - 因子选股侧重**技术面**（多因子模型）+ 历史回测

    两者可组合使用：先用 JYS 筛出基本面优质标的，再用因子选股做技术面二次筛选。
    """)


# ---------------------------------------------------------------------------
# 主函数
# ---------------------------------------------------------------------------


def main():
    config = render_sidebar()
    method = st.session_state.get("method", "jys")

    # ---- 单股评分 ----
    if st.session_state.get("_score_btn") and st.session_state.get("_single_code"):
        code = st.session_state["_single_code"].strip()
        if not code:
            st.error("请输入股票代码")
        else:
            with st.spinner("正在获取数据并评分..."):
                try:
                    stocks = cached_fetch((code,))
                    if stocks:
                        scorer = JYSScorer()
                        result = scorer.calculate_score(stocks[0])
                        st.session_state["single_result"] = result
                    else:
                        st.error(f"未找到代码 {code} 的数据，请检查代码是否正确")
                except Exception as e:
                    st.error(f"评分失败: {e}")

    # ---- 批量筛选 ----
    if st.session_state.get("_screen_btn") and method == "jys":
        market = config.get("market", "csi300")
        pool_sizes = {"csi300": "~300", "a": "~5000", "hk": "~800", "all": "~5800"}
        pool_hint = pool_sizes.get(market, "")

        with st.spinner(f"正在获取 {pool_hint} 只股票实时行情数据，请耐心等待..."):
            t0 = time.time()
            try:
                top_df, all_df = cached_screening(
                    market=config["market"],
                    top_n=config["top_n"],
                    min_score=config["min_score"],
                    max_pe=config["max_pe"],
                    min_turnover=config["min_turnover"],
                    min_price=config["min_price"],
                    max_concurrent=config["max_concurrent"],
                )
                elapsed = time.time() - t0
                st.session_state["top_df"] = top_df
                st.session_state["all_df"] = all_df
                st.session_state["elapsed"] = elapsed
                st.session_state["last_fetch_time"] = datetime.now().strftime("%H:%M:%S")
                st.toast(f"筛选完成！耗时 {elapsed:.1f}s，入选 {len(top_df)} 只")
            except Exception as e:
                st.error(f"筛选失败: {e}")

    # ---- 主区域内容 ----
    # 显示单股评分结果
    if "single_result" in st.session_state and st.session_state["single_result"]:
        st.subheader("单股评分结果")
        render_single_result(st.session_state["single_result"])
        st.divider()

    # 显示批量筛选结果
    if method == "jys":
        if "top_df" in st.session_state and st.session_state["top_df"] is not None:
            render_jys_results(
                st.session_state["top_df"],
                st.session_state["all_df"],
                st.session_state.get("elapsed", 0),
                config,
            )
        else:
            if "single_result" not in st.session_state or not st.session_state["single_result"]:
                render_welcome()
    elif method == "factor":
        render_factor_placeholder()
    else:
        st.info(f"🚧 {METHOD_LABELS.get(method, method)} 功能即将上线")


if __name__ == "__main__":
    main()
