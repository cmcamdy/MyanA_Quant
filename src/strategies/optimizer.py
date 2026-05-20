"""参数优化器"""

import random
import math
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Any, Callable, Union
from itertools import product

import pandas as pd
import numpy as np

from .base import Strategy
from .result import BacktestResult


@dataclass
class ParamRange:
    """参数范围定义"""
    name: str
    values: Optional[List[Any]] = None
    start: Optional[float] = None
    stop: Optional[float] = None
    step: Optional[float] = None

    def iter_values(self) -> List[Any]:
        if self.values is not None:
            return self.values
        if self.start is not None and self.stop is not None:
            step = self.step or 1
            if isinstance(self.start, int) and isinstance(self.step if self.step else 1, int):
                return list(range(int(self.start), int(self.stop) + 1, int(step)))
            vals = []
            v = self.start
            while v <= self.stop + 1e-10:
                vals.append(type(self.start)(v))
                v += step
            return vals
        raise ValueError(f"ParamRange {self.name} 必须指定 values 或 start/stop")


@dataclass
class OptimizationResult:
    """单次优化结果"""
    params: Dict[str, Any]
    objective_value: float
    backtest_result: Optional[BacktestResult] = None


StrategyFactory = Callable[[Dict[str, Any]], Strategy]


def sharpe_objective(result: BacktestResult) -> float:
    return result.sharpe_ratio


def total_return_objective(result: BacktestResult) -> float:
    return result.total_return


def results_to_dataframe(results: List[OptimizationResult]) -> pd.DataFrame:
    """将优化结果转为 DataFrame"""
    rows = []
    for r in results:
        row = dict(r.params)
        row['objective'] = r.objective_value
        if r.backtest_result is not None:
            row['total_return'] = r.backtest_result.total_return
            row['sharpe_ratio'] = r.backtest_result.sharpe_ratio
            row['max_drawdown'] = r.backtest_result.max_drawdown
            row['total_trades'] = r.backtest_result.total_trades
        rows.append(row)
    return pd.DataFrame(rows)


class Optimizer:
    """参数优化器"""

    def __init__(
        self,
        engine: Any,
        strategy_factory: StrategyFactory,
        param_ranges: List[ParamRange],
        objective: Union[str, Callable[[BacktestResult], float]] = 'sharpe',
        top_n: int = 10,
        max_workers: Optional[int] = None,
    ):
        self._engine = engine
        self._strategy_factory = strategy_factory
        self._param_ranges = param_ranges
        self._objective = objective
        self._top_n = top_n
        self._max_workers = max_workers or 1

    def run(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        symbol: str = "UNKNOWN",
    ) -> List[OptimizationResult]:
        """运行参数优化"""
        grid = self._generate_grid()
        resolve_obj = self._resolve_objective()

        results: List[OptimizationResult] = []
        for params in grid:
            strategy = self._strategy_factory(params)
            if isinstance(data, dict):
                result = self._engine.run(strategy, data)
            else:
                result = self._engine.run(strategy, data, symbol=symbol)
            obj_value = resolve_obj(result)
            results.append(OptimizationResult(
                params=params,
                objective_value=obj_value,
                backtest_result=result,
            ))

        # 按目标函数降序
        results.sort(key=lambda r: r.objective_value, reverse=True)

        # 仅 top_n 保留完整结果
        for i in range(self._top_n, len(results)):
            results[i].backtest_result = None

        return results

    def _generate_grid(self) -> List[Dict[str, Any]]:
        """生成参数网格（笛卡尔积）"""
        param_lists = [pr.iter_values() for pr in self._param_ranges]
        param_names = [pr.name for pr in self._param_ranges]
        grid = []
        for combo in product(*param_lists):
            grid.append(dict(zip(param_names, combo)))
        return grid

    def _resolve_objective(self) -> Callable[[BacktestResult], float]:
        if isinstance(self._objective, str):
            if self._objective == 'sharpe':
                return sharpe_objective
            elif self._objective == 'return':
                return total_return_objective
            else:
                raise ValueError(f"未知目标函数: {self._objective}")
        return self._objective


class GeneticOptimizer:
    """遗传算法参数优化器

    通过选择、交叉、变异操作在参数空间中搜索最优解。
    适用于参数空间较大的场景，相比网格搜索更高效。
    """

    def __init__(
        self,
        engine: Any,
        strategy_factory: StrategyFactory,
        param_ranges: List[ParamRange],
        objective: Union[str, Callable[[BacktestResult], float]] = 'sharpe',
        population_size: int = 30,
        generations: int = 20,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elite_ratio: float = 0.1,
        top_n: int = 10,
        seed: Optional[int] = None,
    ):
        self._engine = engine
        self._strategy_factory = strategy_factory
        self._param_ranges = param_ranges
        self._objective = objective
        self._population_size = population_size
        self._generations = generations
        self._crossover_rate = crossover_rate
        self._mutation_rate = mutation_rate
        self._elite_ratio = elite_ratio
        self._top_n = top_n
        self._rng = random.Random(seed)
        self._param_values_cache: Dict[str, List[Any]] = {}

    def run(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        symbol: str = "UNKNOWN",
    ) -> List[OptimizationResult]:
        """运行遗传算法优化"""
        resolve_obj = self._resolve_objective()
        self._param_values_cache = {
            pr.name: pr.iter_values() for pr in self._param_ranges
        }
        param_names = [pr.name for pr in self._param_ranges]

        # 初始化种群
        population = [self._random_individual() for _ in range(self._population_size)]
        all_results: List[OptimizationResult] = []
        best_ever: Optional[OptimizationResult] = None

        for gen in range(self._generations):
            # 评估适应度
            fitnesses: List[float] = []
            for individual in population:
                params = dict(zip(param_names, individual))
                strategy = self._strategy_factory(params)
                if isinstance(data, dict):
                    result = self._engine.run(strategy, data)
                else:
                    result = self._engine.run(strategy, data, symbol=symbol)
                obj_value = resolve_obj(result)
                fitnesses.append(obj_value)
                all_results.append(OptimizationResult(
                    params=params, objective_value=obj_value,
                    backtest_result=result,
                ))

            # 更新全局最优
            gen_best_idx = int(np.argmax(fitnesses))
            gen_best = all_results[len(all_results) - len(population) + gen_best_idx]
            if best_ever is None or gen_best.objective_value > best_ever.objective_value:
                best_ever = gen_best

            # 选择（轮盘赌）
            selected = self._select(population, fitnesses)

            # 交叉 + 变异 → 新种群
            new_population = []
            n_elite = max(1, int(self._population_size * self._elite_ratio))

            # 精英保留
            elite_indices = sorted(range(len(fitnesses)), key=lambda i: fitnesses[i], reverse=True)[:n_elite]
            for idx in elite_indices:
                new_population.append(list(population[idx]))

            while len(new_population) < self._population_size:
                p1, p2 = self._rng.sample(selected, 2)
                child = self._crossover(p1, p2)
                child = self._mutate(child)
                new_population.append(child)

            population = new_population

        # 去重 + 排序
        seen: set = set()
        unique_results: List[OptimizationResult] = []
        for r in all_results:
            key = tuple(sorted(r.params.items()))
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        unique_results.sort(key=lambda r: r.objective_value, reverse=True)

        for i in range(self._top_n, len(unique_results)):
            unique_results[i].backtest_result = None

        return unique_results

    def _random_individual(self) -> List[Any]:
        return [self._rng.choice(self._param_values_cache[pr.name]) for pr in self._param_ranges]

    def _select(self, population: List[List[Any]], fitnesses: List[float]) -> List[List[Any]]:
        min_f = min(fitnesses)
        shifted = [f - min_f + 1e-10 for f in fitnesses]
        total = sum(shifted)
        probs = [f / total for f in shifted]
        selected = self._rng.choices(population, weights=probs, k=self._population_size)
        return selected

    def _crossover(self, p1: List[Any], p2: List[Any]) -> List[Any]:
        if self._rng.random() > self._crossover_rate:
            return list(p1)
        if len(p1) <= 1:
            return list(p1)
        point = self._rng.randint(1, len(p1) - 1)
        return list(p1[:point]) + list(p2[point:])

    def _mutate(self, individual: List[Any]) -> List[Any]:
        result = list(individual)
        for i, pr in enumerate(self._param_ranges):
            if self._rng.random() < self._mutation_rate:
                result[i] = self._rng.choice(self._param_values_cache[pr.name])
        return result

    def _resolve_objective(self) -> Callable[[BacktestResult], float]:
        if isinstance(self._objective, str):
            if self._objective == 'sharpe':
                return sharpe_objective
            elif self._objective == 'return':
                return total_return_objective
            else:
                raise ValueError(f"未知目标函数: {self._objective}")
        return self._objective


class BayesianOptimizer:
    """贝叶斯优化参数搜索

    使用高斯过程代理模型 + 采集函数（EI）指导搜索。
    适用于评估成本高（回测耗时）的场景，用更少的评估找到最优解。

    注意：若未安装 scikit-learn，回退到随机搜索。
    """

    def __init__(
        self,
        engine: Any,
        strategy_factory: StrategyFactory,
        param_ranges: List[ParamRange],
        objective: Union[str, Callable[[BacktestResult], float]] = 'sharpe',
        n_initial: int = 5,
        n_iterations: int = 20,
        top_n: int = 10,
        seed: Optional[int] = None,
    ):
        self._engine = engine
        self._strategy_factory = strategy_factory
        self._param_ranges = param_ranges
        self._objective = objective
        self._n_initial = n_initial
        self._n_iterations = n_iterations
        self._top_n = top_n
        self._seed = seed

    def run(
        self,
        data: Union[pd.DataFrame, Dict[str, pd.DataFrame]],
        symbol: str = "UNKNOWN",
    ) -> List[OptimizationResult]:
        """运行贝叶斯优化"""
        resolve_obj = self._resolve_objective()
        param_names = [pr.name for pr in self._param_ranges]
        param_value_lists = [pr.iter_values() for pr in self._param_ranges]

        # 尝试导入 sklearn
        try:
            from sklearn.gaussian_process import GaussianProcessRegressor
            from sklearn.gaussian_process.kernels import Matern
            from sklearn.preprocessing import MinMaxScaler
            use_gp = True
        except ImportError:
            use_gp = False

        all_results: List[OptimizationResult] = []
        X_observed: List[List[float]] = []
        y_observed: List[float] = []

        # 编码：将参数值映射到 [0, 1]
        def encode(params: Dict[str, Any]) -> List[float]:
            encoded = []
            for i, pr in enumerate(self._param_ranges):
                vals = param_value_lists[i]
                idx = vals.index(params[pr.name]) if params[pr.name] in vals else 0
                encoded.append(idx / max(len(vals) - 1, 1))
            return encoded

        def decode(encoded: List[float]) -> Dict[str, Any]:
            params = {}
            for i, pr in enumerate(self._param_ranges):
                vals = param_value_lists[i]
                idx = round(encoded[i] * (len(vals) - 1))
                idx = max(0, min(idx, len(vals) - 1))
                params[pr.name] = vals[idx]
            return params

        def evaluate(params: Dict[str, Any]) -> float:
            strategy = self._strategy_factory(params)
            if isinstance(data, dict):
                result = self._engine.run(strategy, data)
            else:
                result = self._engine.run(strategy, data, symbol=symbol)
            obj_value = resolve_obj(result)
            all_results.append(OptimizationResult(
                params=params, objective_value=obj_value,
                backtest_result=result,
            ))
            return obj_value

        # 随机初始探索
        rng = random.Random(self._seed)
        grid = list(product(*param_value_lists))
        initial_points = rng.sample(grid, min(self._n_initial, len(grid)))

        for combo in initial_points:
            params = dict(zip(param_names, combo))
            obj = evaluate(params)
            X_observed.append(encode(params))
            y_observed.append(obj)

        # 贝叶斯迭代
        if use_gp:
            X_arr = np.array(X_observed)
            y_arr = np.array(y_observed)

            for _ in range(self._n_iterations):
                kernel = Matern(nu=2.5)
                gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=3, random_state=self._seed)
                gp.fit(X_arr, y_arr)

                # Expected Improvement 采集函数
                best_y = max(y_observed)
                candidates = []
                for combo in grid:
                    x = encode(dict(zip(param_names, combo)))
                    x_arr = np.array(x).reshape(1, -1)
                    mu, sigma = gp.predict(x_arr, return_std=True)
                    sigma = max(sigma[0], 1e-10)
                    z = (mu[0] - best_y) / sigma
                    ei = sigma * (z * norm_cdf(z) + norm_pdf(z))
                    candidates.append((dict(zip(param_names, combo)), ei))

                # 选择 EI 最大的候选
                candidates.sort(key=lambda c: c[1], reverse=True)
                next_params = candidates[0][0]

                obj = evaluate(next_params)
                X_observed.append(encode(next_params))
                y_observed.append(obj)
                X_arr = np.vstack([X_arr, [encode(next_params)]])
                y_arr = np.append(y_arr, obj)
        else:
            # Fallback: random search
            remaining = [c for c in grid if c not in initial_points]
            sample_size = min(self._n_iterations, len(remaining))
            for combo in rng.sample(remaining, sample_size):
                params = dict(zip(param_names, combo))
                evaluate(params)

        # 去重 + 排序
        seen: set = set()
        unique_results: List[OptimizationResult] = []
        for r in all_results:
            key = tuple(sorted(r.params.items()))
            if key not in seen:
                seen.add(key)
                unique_results.append(r)
        unique_results.sort(key=lambda r: r.objective_value, reverse=True)

        for i in range(self._top_n, len(unique_results)):
            unique_results[i].backtest_result = None

        return unique_results

    def _resolve_objective(self) -> Callable[[BacktestResult], float]:
        if isinstance(self._objective, str):
            if self._objective == 'sharpe':
                return sharpe_objective
            elif self._objective == 'return':
                return total_return_objective
            else:
                raise ValueError(f"未知目标函数: {self._objective}")
        return self._objective


def norm_pdf(x: float) -> float:
    """标准正态分布概率密度"""
    return math.exp(-0.5 * x * x) / math.sqrt(2 * math.pi)


def norm_cdf(x: float) -> float:
    """标准正态分布累积分布函数（近似）"""
    return 0.5 * (1 + math.erf(x / math.sqrt(2)))
