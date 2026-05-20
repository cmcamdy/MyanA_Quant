from setuptools import setup, find_packages

setup(
    name="quant-analysis",
    version="0.1.0",
    description="量化交易分析系统",
    packages=find_packages(),
    python_requires=">=3.10",
    install_requires=[
        "pandas>=2.0.0",
        "numpy>=1.24.0",
        "matplotlib>=3.7.0",
        "plotly>=5.14.0",
        "ta-lib>=0.4.26",
        "tushare>=1.2.89",
        "akshare>=1.11.0",
        "pyyaml>=6.0",
        "requests>=2.31.0",
        "streamlit>=1.30.0",
        "aiohttp>=3.9.0",
    ],
)