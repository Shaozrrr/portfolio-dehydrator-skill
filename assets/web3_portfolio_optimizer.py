#!/usr/bin/env python3
from __future__ import annotations

import argparse
import difflib
import hashlib
import math
import re
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import requests
from requests import Session
from scipy.optimize import minimize

# =========================
# 全局配置
# =========================

OKX_BASE_URLS: tuple[str, ...] = (
    "https://aws.okx.com",
    "https://www.okx.com",
)
GATE_API_BASE_URL = "https://api.gateio.ws/api/v4"
OKX_KLINE_ENDPOINT = "/api/v5/market/candles"
GATE_KLINE_ENDPOINT = "/spot/candlesticks"
REQUEST_TIMEOUT_SECONDS = 6
REQUEST_RETRY_COUNT = 3
REQUEST_BACKOFF_SECONDS = 0.8
CANDLE_INTERVAL_OKX = "4H"
CANDLE_INTERVAL_GATE = "4h"
LOOKBACK_DAYS = 30
CANDLE_LIMIT = 180
PERIODS_PER_YEAR = 6 * 365  # 4 小时频率，一年约 6 * 365 个周期
RISK_FREE_RATE = 0.04
HIGH_CORRELATION_THRESHOLD = 0.85
NEW_TOKEN_MIN_CANDLES = 84  # 14 天 * 6 根/天
MIN_USABLE_CANDLES = 20
MIN_ALIGNMENT_POINTS = 20
NUMERICAL_EPSILON = 1e-10
DOMINATED_ASSET_MAX_WEIGHT = 0.02
OVERLAP_WEIGHT_PENALTY = 0.20
LOW_QUALITY_WEIGHT_PENALTY = 0.22
DRAWDOWN_WEIGHT_PENALTY = 0.18
DIVERSIFICATION_PENALTY = 0.01
MIN_ANNUAL_RETURN = -0.95
MAX_ANNUAL_RETURN = 10.0
SORTINO_DOWNSIDE_FLOOR = 1e-6
CALMAR_DRAWDOWN_FLOOR = 1e-6
LOW_QUALITY_MAX_WEIGHT = 0.05
VERY_LOW_QUALITY_MAX_WEIGHT = 0.03
DEEP_DRAWDOWN_MAX_WEIGHT = 0.03

STABLECOINS = {"USDT", "USDC", "DAI"}
MEGA_CAP_TOKENS = {"BTC", "ETH"}
BLUE_CHIP_TOKENS = {
    "SOL",
    "BNB",
    "LINK",
    "XRP",
    "ADA",
    "DOGE",
    "AVAX",
    "TON",
    "TRX",
    "DOT",
    "SUI",
    "APT",
    "ATOM",
    "NEAR",
    "LTC",
    "BCH",
    "ETC",
    "FIL",
    "AAVE",
    "UNI",
    "ARB",
    "OP",
}
KNOWN_TOKENS = sorted(
    STABLECOINS
    | MEGA_CAP_TOKENS
    | BLUE_CHIP_TOKENS
    | {
        "PEPE",
        "WIF",
        "BONK",
        "SHIB",
        "FLOKI",
        "DOG",
        "ENA",
        "SEI",
        "TIA",
        "INJ",
        "JUP",
        "PYTH",
        "WLD",
        "BOME",
        "MOG",
        "PENDLE",
        "RENDER",
        "RUNE",
        "ICP",
        "HBAR",
        "TAO",
    }
)


class TokenNotFoundError(Exception):
    """代币代码不存在或无法映射到现货交易对。"""


class DataSourceUnavailableError(Exception):
    """交易所接口暂时不可用。"""


@dataclass
class AssetData:
    token: str
    prices: pd.Series
    volumes: pd.Series
    source: str
    is_new: bool = False
    is_stablecoin: bool = False
    used_mock: bool = False
    warnings: List[str] = field(default_factory=list)


@dataclass
class AssetStats:
    token: str
    expected_return: float
    volatility: float
    sortino: float
    calmar: float
    max_drawdown: float
    cap: float
    effective_cap: float
    data_points: int
    is_new: bool
    used_mock: bool
    source: str


@dataclass
class OptimizationResult:
    weights: Dict[str, float]
    risky_weights: np.ndarray
    cash_weight: float
    expected_return: float
    volatility: float
    sortino: float
    calmar: float
    max_drawdown: float
    used_fallback: bool


@dataclass
class ReferencePortfolio:
    label: str
    weights: Dict[str, float]
    risky_weights: np.ndarray
    cash_weight: float
    expected_return: float
    volatility: float
    sortino: float
    calmar: float
    max_drawdown: float


def normalize_token(token: str) -> str:
    """清洗用户输入的代币符号，尽量兼容常见写法。"""
    normalized = token.strip().upper()
    normalized = normalized.replace("$", "")
    normalized = normalized.replace("/USDT", "")
    normalized = normalized.replace("-USDT", "")
    normalized = normalized.replace("_USDT", "")
    return normalized


def parse_token_text(token_text: str) -> List[str]:
    """将逗号或空格分隔的字符串转换为代币列表。"""
    raw_tokens = token_text.replace(",", " ").split()
    return [normalize_token(token) for token in raw_tokens if token.strip()]


def _parse_weight_chunk(chunk: str) -> Optional[Tuple[str, float]]:
    """从单个片段中提取代币与权重，兼容自然语言持仓描述。"""
    cleaned = (
        chunk.strip()
        .upper()
        .replace("％", "%")
        .replace("：", ":")
        .replace("，", ",")
        .replace("；", ";")
        .replace("、", " ")
    )
    if not cleaned:
        return None

    if "=" in cleaned:
        token_text, value_text = cleaned.split("=", 1)
        token = normalize_token(token_text)
        try:
            return token, float(value_text.strip().replace("%", ""))
        except ValueError:
            return None

    if ":" in cleaned:
        token_text, value_text = cleaned.split(":", 1)
        token = normalize_token(token_text)
        try:
            return token, float(value_text.strip().replace("%", ""))
        except ValueError:
            return None

    natural_language_patterns = (
        re.compile(r"(?P<token>\$?[A-Z][A-Z0-9]{0,14})\s*(?:持仓|仓位|占比|配置|比例)?\s*(?P<value>\d+(?:\.\d+)?)\s*%?"),
        re.compile(r"(?P<value>\d+(?:\.\d+)?)\s*%?\s*(?P<token>\$?[A-Z][A-Z0-9]{0,14})"),
    )
    for pattern in natural_language_patterns:
        match = pattern.search(cleaned)
        if not match:
            continue
        token = normalize_token(match.group("token"))
        if not token:
            continue
        try:
            return token, float(match.group("value"))
        except ValueError:
            return None
    return None


def parse_weights_text(weights_text: str) -> Dict[str, float]:
    """解析用户输入的初始持仓比例，兼容显式格式与自然语言描述。"""
    parsed: Dict[str, float] = {}
    if not weights_text.strip():
        return parsed

    normalized_text = (
        weights_text.replace("\n", ",")
        .replace("；", ",")
        .replace("，", ",")
        .replace("、", ",")
    )
    chunks = [chunk.strip() for chunk in normalized_text.replace(";", ",").split(",") if chunk.strip()]
    for chunk in chunks:
        parsed_pair = _parse_weight_chunk(chunk)
        if not parsed_pair:
            continue
        token, value = parsed_pair
        if value < 0:
            continue
        parsed[token] = value
    return parsed


def get_weight_cap(token: str, is_new: bool) -> float:
    """根据资产类型返回硬性权重上限。"""
    if is_new:
        return 0.05
    if token in MEGA_CAP_TOKENS:
        return 0.50
    if token in BLUE_CHIP_TOKENS:
        return 0.30
    return 0.15


def _risk_free_period_return() -> float:
    """把年化无风险利率换算成单个 4 小时周期收益。"""
    return float((1.0 + RISK_FREE_RATE) ** (1.0 / PERIODS_PER_YEAR) - 1.0)


def _annualize_return_from_periodic_returns(returns: pd.Series) -> float:
    """把 4 小时收益序列换算成年化收益。"""
    if returns.empty:
        return RISK_FREE_RATE
    clipped = returns.clip(lower=-0.999999)
    log_returns = np.log1p(clipped)
    annual_return = float(np.exp(log_returns.mean() * PERIODS_PER_YEAR) - 1.0)
    return float(np.clip(annual_return, MIN_ANNUAL_RETURN, MAX_ANNUAL_RETURN))


def _compute_max_drawdown(returns: pd.Series) -> float:
    """根据收益序列计算最大回撤，返回正数比例。"""
    if returns.empty:
        return 0.0
    clipped = returns.clip(lower=-0.999999)
    wealth = (1.0 + clipped).cumprod()
    running_peak = wealth.cummax()
    drawdown = wealth / running_peak - 1.0
    return float(abs(drawdown.min()))


def _compute_sortino_ratio(annual_return: float, returns: pd.Series) -> float:
    """计算 Sortino Ratio，只惩罚下行波动。"""
    if returns.empty:
        return 0.0
    target = _risk_free_period_return()
    downside = np.minimum(returns.to_numpy(dtype=float) - target, 0.0)
    downside_deviation = float(np.sqrt(np.mean(np.square(downside))) * math.sqrt(PERIODS_PER_YEAR))
    if downside_deviation <= SORTINO_DOWNSIDE_FLOOR:
        return 6.0 if annual_return > RISK_FREE_RATE else 0.0
    return float((annual_return - RISK_FREE_RATE) / downside_deviation)


def _compute_calmar_ratio(annual_return: float, max_drawdown: float) -> float:
    """计算 Calmar Ratio，用收益与最大回撤衡量性价比。"""
    if max_drawdown <= CALMAR_DRAWDOWN_FLOOR:
        return 6.0 if annual_return > RISK_FREE_RATE else 0.0
    return float((annual_return - RISK_FREE_RATE) / max_drawdown)


def deduplicate_messages(messages: Sequence[str]) -> List[str]:
    """按原顺序去重，避免报告中重复提示。"""
    seen: set[str] = set()
    ordered: List[str] = []
    for message in messages:
        cleaned = message.strip()
        if cleaned and cleaned not in seen:
            ordered.append(cleaned)
            seen.add(cleaned)
    return ordered


def deduplicate_tokens(tokens: Sequence[str]) -> List[str]:
    """按输入顺序去重，避免同一资产重复进入优化矩阵。"""
    seen: set[str] = set()
    ordered: List[str] = []
    for token in tokens:
        if token and token not in seen:
            ordered.append(token)
            seen.add(token)
    return ordered


def looks_like_missing_instrument(message: str) -> bool:
    """判断交易所错误信息是否在表达“该交易对不存在”。"""
    normalized = message.strip().lower()
    patterns = (
        "doesn't exist",
        "does not exist",
        "not exist",
        "invalid instrument",
        "instrument id",
        "invalid instid",
        "instrument doesn't exist",
        "currency pair",
    )
    return any(pattern in normalized for pattern in patterns)


class Web3PortfolioOptimizer:
    """基于 MPT 的 Web3 组合优化器。"""

    def __init__(self, session: Optional[Session] = None) -> None:
        self.session = session or requests.Session()
        self.session.headers.update(
            {
                "Accept": "application/json",
                "User-Agent": "Web3PortfolioOptimizer/1.0",
            }
        )

    def optimize(
        self,
        tokens: Sequence[str],
        total_capital: float = 10000.0,
        current_weights: Optional[Dict[str, float]] = None,
    ) -> str:
        """对外主入口：接收代币列表和总资金，返回中文 Markdown 报告。"""
        normalized_tokens = deduplicate_tokens(
            [normalize_token(token) for token in tokens if normalize_token(token)]
        )
        if not normalized_tokens:
            return self._format_empty_report(total_capital)

        warnings: List[str] = []
        explicit_stables = [token for token in normalized_tokens if token in STABLECOINS]
        risky_assets: List[AssetData] = []

        for token in normalized_tokens:
            if token in STABLECOINS:
                warnings.append(f"{token} 被识别为稳定币，已作为无风险基准与现金仓位处理。")
                continue

            try:
                asset = self._fetch_asset_data(token)
                risky_assets.append(asset)
                warnings.extend(asset.warnings)
            except TokenNotFoundError as exc:
                warnings.append(str(exc))
            except Exception as exc:  # noqa: BLE001
                mock_asset = self._build_mock_asset(token)
                mock_asset.warnings.append(
                    f"{token} 抓取过程中出现未预期异常：{exc}；已改用 Mock Data 保证流程可演示。"
                )
                risky_assets.append(mock_asset)
                warnings.extend(mock_asset.warnings)

        cash_symbol = explicit_stables[0] if explicit_stables else "USDT"

        if not risky_assets:
            return self._format_cash_only_report(
                cash_symbol=cash_symbol,
                total_capital=total_capital,
                warnings=warnings,
            )

        risky_assets = self._ensure_minimum_series_quality(risky_assets, warnings)
        stats_map = {asset.token: self._compute_asset_stats(asset) for asset in risky_assets}
        aligned_prices = self._build_aligned_price_frame(risky_assets, warnings)
        returns_frame = aligned_prices.pct_change(fill_method=None).dropna(how="any")

        if returns_frame.empty or returns_frame.shape[0] < MIN_ALIGNMENT_POINTS:
            warnings.append("资产共同有效样本不足，已切换为统一 Mock 对齐数据以完成组合求解。")
            risky_assets = [self._build_mock_asset(asset.token) for asset in risky_assets]
            stats_map = {asset.token: self._compute_asset_stats(asset) for asset in risky_assets}
            aligned_prices = self._build_aligned_price_frame(risky_assets, warnings)
            returns_frame = aligned_prices.pct_change(fill_method=None).dropna(how="any")

        risky_tokens = [asset.token for asset in risky_assets]
        returns_frame = returns_frame[risky_tokens]
        returns_matrix = returns_frame.to_numpy(dtype=float)

        correlation_matrix = returns_frame.corr(method="pearson")
        overlap_pairs, dominated_tokens = self._detect_overlap_pairs(correlation_matrix, stats_map)

        adjusted_caps = np.array(
            [self._effective_cap_for_asset(token, stats_map[token], dominated_tokens) for token in risky_tokens],
            dtype=float,
        )
        for token, effective_cap in zip(risky_tokens, adjusted_caps):
            stats_map[token].effective_cap = float(effective_cap)
        dominated_mask = np.array([1.0 if token in dominated_tokens else 0.0 for token in risky_tokens], dtype=float)
        quality_penalty_vector = np.array(
            [
                min(
                    max(-stats_map[token].sortino, 0.0) * 0.45
                    + max(-stats_map[token].calmar, 0.0) * 0.35
                    + max(stats_map[token].max_drawdown - 0.35, 0.0) * 1.5,
                    3.0,
                )
                for token in risky_tokens
            ],
            dtype=float,
        )
        drawdown_penalty_vector = np.array(
            [max(stats_map[token].max_drawdown - 0.30, 0.0) for token in risky_tokens],
            dtype=float,
        )

        optimization = self._optimize_weights(
            risky_tokens=risky_tokens,
            returns_matrix=returns_matrix,
            adjusted_caps=adjusted_caps,
            dominated_mask=dominated_mask,
            quality_penalty_vector=quality_penalty_vector,
            drawdown_penalty_vector=drawdown_penalty_vector,
            cash_symbol=cash_symbol,
        )

        reference_portfolio = self._build_reference_portfolio(
            risky_tokens=risky_tokens,
            returns_matrix=returns_matrix,
            cash_symbol=cash_symbol,
            current_weights=current_weights,
        )
        final_warnings = deduplicate_messages(warnings)
        return self._format_report(
            risky_tokens=risky_tokens,
            stats_map=stats_map,
            overlap_pairs=overlap_pairs,
            correlation_matrix=correlation_matrix,
            returns_frame=returns_frame,
            optimization=optimization,
            reference_portfolio=reference_portfolio,
            total_capital=total_capital,
            cash_symbol=cash_symbol,
            warnings=final_warnings,
        )

    def _fetch_asset_data(self, token: str) -> AssetData:
        """优先从 OKX 抓取；连续失败两次后自动降级到 Gate.io。"""
        okx_failures = 0
        okx_not_found = False

        for base_url in OKX_BASE_URLS:
            try:
                asset = self._fetch_okx_candles(token, base_url)
                return asset
            except TokenNotFoundError:
                okx_not_found = True
                break
            except DataSourceUnavailableError:
                okx_failures += 1
                if okx_failures >= 2:
                    break

        try:
            asset = self._fetch_gate_candles(token)
            if okx_failures >= 2:
                asset.warnings.append(f"{token} 的 OKX 节点短暂波动，系统已自动切换到 Gate.io 并完成分析。")
            return asset
        except TokenNotFoundError:
            suggestion = self._build_unknown_token_warning(token)
            raise TokenNotFoundError(suggestion)
        except DataSourceUnavailableError as exc:
            if okx_not_found:
                suggestion = self._build_unknown_token_warning(token)
                raise TokenNotFoundError(suggestion)

            mock_asset = self._build_mock_asset(token)
            mock_asset.warnings.append(
                f"{token} 的 OKX 与 Gate.io 均不可用（{exc}），已自动启用 Mock Data 继续演示优化逻辑。"
            )
            return mock_asset

    def _fetch_okx_candles(self, token: str, base_url: str) -> AssetData:
        """从 OKX 抓取 30 天 4H K 线。"""
        payload = self._request_json(
            url=f"{base_url}{OKX_KLINE_ENDPOINT}",
            params={
                "instId": f"{token}-USDT",
                "bar": CANDLE_INTERVAL_OKX,
                "limit": str(CANDLE_LIMIT),
            },
        )

        if not isinstance(payload, dict):
            raise DataSourceUnavailableError("OKX 返回结构异常")

        code = str(payload.get("code", ""))
        if code and code != "0":
            message = str(payload.get("msg", "OKX 返回错误码"))
            if code == "50011":
                raise DataSourceUnavailableError("OKX 触发限流")
            if looks_like_missing_instrument(message):
                raise TokenNotFoundError(token)
            raise DataSourceUnavailableError(message)

        candles = payload.get("data", [])
        if not candles:
            raise TokenNotFoundError(token)

        return self._parse_okx_candles(token=token, candles=candles)

    def _fetch_gate_candles(self, token: str) -> AssetData:
        """从 Gate.io 抓取 30 天 4H K 线。"""
        payload = self._request_json(
            url=f"{GATE_API_BASE_URL}{GATE_KLINE_ENDPOINT}",
            params={
                "currency_pair": f"{token}_USDT",
                "interval": CANDLE_INTERVAL_GATE,
                "limit": str(CANDLE_LIMIT),
            },
        )

        if isinstance(payload, dict):
            label = str(payload.get("label", "")).upper()
            message = str(payload.get("message", "")) or str(payload.get("detail", ""))
            if label in {"INVALID_PARAM_VALUE", "NOT_FOUND", "BAD_REQUEST"}:
                raise TokenNotFoundError(token)
            raise DataSourceUnavailableError(message or "Gate.io 返回异常字典结构")

        if not isinstance(payload, list):
            raise DataSourceUnavailableError("Gate.io 返回结构异常")
        if not payload:
            raise TokenNotFoundError(token)

        return self._parse_gate_candles(token=token, candles=payload)

    def _request_json(self, url: str, params: Dict[str, str]) -> Any:
        """公共 GET 包装器，带超时、限流重试和指数退避。"""
        last_error: Optional[Exception] = None
        last_status_code: Optional[int] = None

        for attempt in range(1, REQUEST_RETRY_COUNT + 1):
            try:
                response = self.session.get(url, params=params, timeout=REQUEST_TIMEOUT_SECONDS)
                last_status_code = response.status_code

                if response.status_code in {429, 500, 502, 503, 504}:
                    raise DataSourceUnavailableError(f"HTTP {response.status_code}")

                if response.status_code in {400, 404}:
                    return response.json()

                response.raise_for_status()
                return response.json()
            except (requests.Timeout, requests.ConnectionError) as exc:
                last_error = exc
            except requests.HTTPError as exc:
                last_error = exc
            except ValueError as exc:
                last_error = exc
            except DataSourceUnavailableError as exc:
                last_error = exc

            if attempt < REQUEST_RETRY_COUNT:
                time.sleep(REQUEST_BACKOFF_SECONDS * attempt)

        if last_status_code is not None:
            raise DataSourceUnavailableError(f"请求失败，最终状态码 {last_status_code}") from last_error
        raise DataSourceUnavailableError("请求失败，网络不可用或解析异常") from last_error

    def _parse_okx_candles(self, token: str, candles: Sequence[Sequence[Any]]) -> AssetData:
        """解析 OKX 蜡烛图数据。"""
        frame = pd.DataFrame(
            candles,
            columns=[
                "timestamp_ms",
                "open",
                "high",
                "low",
                "close",
                "volume_base",
                "volume_quote",
                "volume_quote_alt",
                "confirm",
            ],
        )
        frame["timestamp_ms"] = pd.to_numeric(frame["timestamp_ms"], errors="coerce")
        frame["timestamp"] = pd.to_datetime(frame["timestamp_ms"], unit="ms", utc=True).dt.floor("4h")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame["volume"] = pd.to_numeric(frame["volume_base"], errors="coerce")
        frame = frame.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
        frame = frame.drop_duplicates(subset=["timestamp"], keep="last")

        prices = pd.Series(frame["close"].to_numpy(dtype=float), index=frame["timestamp"], name=token)
        volumes = pd.Series(frame["volume"].fillna(0.0).to_numpy(dtype=float), index=frame["timestamp"], name=token)
        is_new = prices.shape[0] < NEW_TOKEN_MIN_CANDLES

        if prices.shape[0] < 2:
            raise DataSourceUnavailableError("OKX 返回的有效 K 线数量不足")

        warnings: List[str] = []
        if is_new:
            warnings.append(f"{token} 上线时间较短，已标记为 [高风险盲盒]，单币权重上限锁定为 5%。")

        return AssetData(
            token=token,
            prices=prices,
            volumes=volumes,
            source="OKX",
            is_new=is_new,
            warnings=warnings,
        )

    def _parse_gate_candles(self, token: str, candles: Sequence[Sequence[Any]]) -> AssetData:
        """解析 Gate.io 蜡烛图数据。"""
        normalized_rows: List[Dict[str, Any]] = []
        for candle in candles:
            if isinstance(candle, dict):
                normalized_rows.append(
                    {
                        "timestamp_s": candle.get("t") or candle.get("timestamp") or candle.get("time"),
                        "quote_volume": candle.get("sum") or candle.get("quote_volume"),
                        "close": candle.get("c") or candle.get("close"),
                        "high": candle.get("h") or candle.get("high"),
                        "low": candle.get("l") or candle.get("low"),
                        "open": candle.get("o") or candle.get("open"),
                        "base_volume": candle.get("v") or candle.get("base_volume") or candle.get("volume"),
                    }
                )
                continue

            if not isinstance(candle, (list, tuple)) or len(candle) < 7:
                continue

            normalized_rows.append(
                {
                    "timestamp_s": candle[0],
                    "quote_volume": candle[1],
                    "close": candle[2],
                    "high": candle[3],
                    "low": candle[4],
                    "open": candle[5],
                    "base_volume": candle[6],
                }
            )

        frame = pd.DataFrame(normalized_rows)
        if frame.empty or "timestamp_s" not in frame.columns or "close" not in frame.columns:
            raise DataSourceUnavailableError("Gate.io 返回的 K 线字段不完整")
        frame["timestamp"] = pd.to_datetime(frame["timestamp_s"], unit="s", utc=True).dt.floor("4h")
        frame["close"] = pd.to_numeric(frame["close"], errors="coerce")
        frame["volume"] = pd.to_numeric(frame["base_volume"], errors="coerce")
        frame = frame.dropna(subset=["timestamp", "close"]).sort_values("timestamp")
        frame = frame.drop_duplicates(subset=["timestamp"], keep="last")

        prices = pd.Series(frame["close"].to_numpy(dtype=float), index=frame["timestamp"], name=token)
        volumes = pd.Series(frame["volume"].fillna(0.0).to_numpy(dtype=float), index=frame["timestamp"], name=token)
        is_new = prices.shape[0] < NEW_TOKEN_MIN_CANDLES

        if prices.shape[0] < 2:
            raise DataSourceUnavailableError("Gate.io 返回的有效 K 线数量不足")

        warnings: List[str] = []
        if is_new:
            warnings.append(f"{token} 上线时间较短，已标记为 [高风险盲盒]，单币权重上限锁定为 5%。")

        return AssetData(
            token=token,
            prices=prices,
            volumes=volumes,
            source="Gate.io",
            is_new=is_new,
            warnings=warnings,
        )

    def _build_mock_asset(self, token: str) -> AssetData:
        """当真实 API 完全不可用时，生成可重复的模拟价格路径。"""
        target_index = self._target_index()
        seed = int(hashlib.sha256(token.encode("utf-8")).hexdigest()[:8], 16)
        rng = np.random.default_rng(seed)

        base_price = float(10 ** rng.uniform(-1.0, 3.0))
        drift = float(rng.uniform(-0.0004, 0.0012))
        volatility = float(rng.uniform(0.01, 0.06))
        raw_returns = rng.normal(loc=drift, scale=volatility, size=len(target_index))
        bounded_returns = np.clip(raw_returns, -0.35, 0.35)
        prices = base_price * np.cumprod(1 + bounded_returns)
        volumes = rng.lognormal(mean=13.0, sigma=0.8, size=len(target_index))

        return AssetData(
            token=token,
            prices=pd.Series(prices, index=target_index, name=token),
            volumes=pd.Series(volumes, index=target_index, name=token),
            source="Mock",
            is_new=False,
            used_mock=True,
            warnings=[],
        )

    def _ensure_minimum_series_quality(
        self, assets: Sequence[AssetData], warnings: List[str]
    ) -> List[AssetData]:
        """样本太少时直接切换到 Mock，避免统计量完全失真。"""
        prepared_assets: List[AssetData] = []
        for asset in assets:
            if asset.prices.shape[0] >= MIN_USABLE_CANDLES:
                prepared_assets.append(asset)
                continue

            mock_asset = self._build_mock_asset(asset.token)
            mock_asset.is_new = asset.is_new or True
            warnings.append(
                f"{asset.token} 的可用 K 线少于 {MIN_USABLE_CANDLES} 根，已替换为 Mock Data 以保证统计与优化可执行。"
            )
            prepared_assets.append(mock_asset)

        return prepared_assets

    def _compute_asset_stats(self, asset: AssetData) -> AssetStats:
        """计算单资产年化收益、波动率、Sortino、Calmar 与最大回撤。"""
        returns = asset.prices.pct_change(fill_method=None).dropna()
        if returns.empty:
            annual_return = RISK_FREE_RATE
            annual_volatility = 0.0
            max_drawdown = 0.0
            sortino = 0.0
            calmar = 0.0
        else:
            annual_return = _annualize_return_from_periodic_returns(returns)
            annual_volatility = float(returns.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR))
            max_drawdown = _compute_max_drawdown(returns)
            sortino = _compute_sortino_ratio(annual_return, returns)
            calmar = _compute_calmar_ratio(annual_return, max_drawdown)

        return AssetStats(
            token=asset.token,
            expected_return=annual_return,
            volatility=annual_volatility,
            sortino=sortino,
            calmar=calmar,
            max_drawdown=max_drawdown,
            cap=get_weight_cap(asset.token, asset.is_new),
            effective_cap=get_weight_cap(asset.token, asset.is_new),
            data_points=int(asset.prices.shape[0]),
            is_new=asset.is_new,
            used_mock=asset.used_mock,
            source=asset.source,
        )

    def _build_aligned_price_frame(self, assets: Sequence[AssetData], warnings: List[str]) -> pd.DataFrame:
        """将多资产价格序列对齐到共同时间轴。"""
        series_list = []
        for asset in assets:
            series = asset.prices.copy()
            series.index = pd.to_datetime(series.index, utc=True).floor("4h")
            series = series[~series.index.duplicated(keep="last")].sort_index()
            series_list.append(series.rename(asset.token))

        inner_frame = pd.concat(series_list, axis=1, join="inner").sort_index()
        if inner_frame.shape[0] >= MIN_ALIGNMENT_POINTS:
            return inner_frame

        warnings.append("实盘时间轴交集过小，已使用统一 4 小时时间轴做保底对齐。")
        target_index = self._target_index()
        aligned_map: Dict[str, pd.Series] = {}

        for asset in assets:
            reindexed = asset.prices.reindex(target_index).ffill().bfill()
            if reindexed.isna().any() or reindexed.nunique() < 2:
                mock_asset = self._build_mock_asset(asset.token)
                reindexed = mock_asset.prices
            aligned_map[asset.token] = reindexed.rename(asset.token)

        return pd.DataFrame(aligned_map, index=target_index).sort_index()

    def _detect_overlap_pairs(
        self, correlation_matrix: pd.DataFrame, stats_map: Dict[str, AssetStats]
    ) -> Tuple[List[Tuple[str, str, float, str]], set[str]]:
        """找出高相关资产对，并标记应被惩罚的低效率资产。"""
        overlap_pairs: List[Tuple[str, str, float, str]] = []
        dominated_tokens: set[str] = set()
        columns = list(correlation_matrix.columns)

        for i, token_a in enumerate(columns):
            for j in range(i + 1, len(columns)):
                token_b = columns[j]
                correlation_value = float(correlation_matrix.iloc[i, j])
                if np.isnan(correlation_value) or correlation_value < HIGH_CORRELATION_THRESHOLD:
                    continue

                loser = self._pick_weaker_asset(stats_map[token_a], stats_map[token_b])
                overlap_pairs.append((token_a, token_b, correlation_value, loser))
                dominated_tokens.add(loser)

        return overlap_pairs, dominated_tokens

    def _pick_weaker_asset(self, left: AssetStats, right: AssetStats) -> str:
        """在高相关资产对里选出更应该被削减的一方。"""
        left_score = (left.sortino, left.calmar, -left.max_drawdown, left.expected_return)
        right_score = (right.sortino, right.calmar, -right.max_drawdown, right.expected_return)
        return right.token if left_score >= right_score else left.token

    def _effective_cap_for_asset(self, token: str, stats: AssetStats, dominated_tokens: set[str]) -> float:
        """在硬上限之内，再根据质量给低效率资产施加更保守的有效上限。"""
        effective_cap = stats.cap
        if token in dominated_tokens:
            effective_cap = min(effective_cap, DOMINATED_ASSET_MAX_WEIGHT)
        if stats.used_mock:
            effective_cap = min(effective_cap, 0.05)
        if stats.max_drawdown >= 0.70:
            effective_cap = min(effective_cap, DEEP_DRAWDOWN_MAX_WEIGHT)
        elif stats.sortino < 0 and stats.calmar < 0:
            effective_cap = min(effective_cap, VERY_LOW_QUALITY_MAX_WEIGHT)
        elif stats.sortino < 0 or stats.calmar < 0:
            effective_cap = min(effective_cap, LOW_QUALITY_MAX_WEIGHT)
        return effective_cap

    def _optimize_weights(
        self,
        risky_tokens: Sequence[str],
        returns_matrix: np.ndarray,
        adjusted_caps: np.ndarray,
        dominated_mask: np.ndarray,
        quality_penalty_vector: np.ndarray,
        drawdown_penalty_vector: np.ndarray,
        cash_symbol: str,
    ) -> OptimizationResult:
        """使用 SLSQP 求解高 Sortino 组合，并允许剩余资金停留在稳定币现金位。"""
        max_cash_buffer = max(0.0, 1.0 - float(np.sum(adjusted_caps)))
        allow_cash_buffer = bool(max_cash_buffer > NUMERICAL_EPSILON)
        asset_expected_returns = np.array(
            [_annualize_return_from_periodic_returns(pd.Series(returns_matrix[:, idx])) for idx in range(returns_matrix.shape[1])],
            dtype=float,
        )
        initial_risky, initial_cash = self._capped_priority_allocation(
            scores=np.maximum(asset_expected_returns - RISK_FREE_RATE, 0.0) + np.maximum(0.10 - quality_penalty_vector, 0.0),
            caps=adjusted_caps,
            allow_cash=allow_cash_buffer,
        )
        x0 = (
            np.concatenate([initial_risky, np.array([initial_cash], dtype=float)])
            if allow_cash_buffer
            else initial_risky
        )
        bounds = (
            [(0.0, float(cap)) for cap in adjusted_caps] + [(0.0, float(max_cash_buffer))]
            if allow_cash_buffer
            else [(0.0, float(cap)) for cap in adjusted_caps]
        )
        constraints = [{"type": "eq", "fun": lambda x: float(np.sum(x) - 1.0)}]

        def objective(weights: np.ndarray) -> float:
            risky_weights = weights[:-1] if allow_cash_buffer else weights
            cash_weight = float(weights[-1]) if allow_cash_buffer else 0.0
            portfolio_return, portfolio_volatility, portfolio_sortino, portfolio_calmar, portfolio_max_drawdown = self._portfolio_metrics(
                risky_weights=risky_weights,
                cash_weight=cash_weight,
                returns_matrix=returns_matrix,
            )
            overlap_penalty = float(np.dot(risky_weights, dominated_mask)) * OVERLAP_WEIGHT_PENALTY
            low_quality_penalty = float(np.dot(risky_weights, quality_penalty_vector)) * LOW_QUALITY_WEIGHT_PENALTY
            drawdown_penalty = float(np.dot(risky_weights, drawdown_penalty_vector)) * DRAWDOWN_WEIGHT_PENALTY
            concentration_penalty = float(np.square(risky_weights).sum()) * DIVERSIFICATION_PENALTY

            if np.isnan(portfolio_return) or np.isnan(portfolio_volatility):
                return 1e6
            return (
                -portfolio_sortino
                - 0.20 * portfolio_calmar
                + 0.25 * portfolio_max_drawdown
                + overlap_penalty
                + low_quality_penalty
                + drawdown_penalty
                + concentration_penalty
            )

        result = minimize(
            objective,
            x0=x0,
            method="SLSQP",
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 300, "ftol": 1e-9, "disp": False},
        )

        if not result.success or np.isnan(result.x).any():
            fallback_risky, fallback_cash = self._capped_priority_allocation(
                scores=np.maximum(asset_expected_returns - RISK_FREE_RATE, 0.0) + np.maximum(0.08 - quality_penalty_vector, 0.0),
                caps=adjusted_caps,
                allow_cash=allow_cash_buffer,
            )
            expected_return, volatility, sortino, calmar, max_drawdown = self._portfolio_metrics(
                risky_weights=fallback_risky,
                cash_weight=fallback_cash,
                returns_matrix=returns_matrix,
            )
            weights = {
                token: float(weight) for token, weight in zip(risky_tokens, fallback_risky)
            }
            if allow_cash_buffer:
                weights[cash_symbol] = float(fallback_cash)
            return OptimizationResult(
                weights=weights,
                risky_weights=fallback_risky,
                cash_weight=fallback_cash,
                expected_return=expected_return,
                volatility=volatility,
                sortino=sortino,
                calmar=calmar,
                max_drawdown=max_drawdown,
                used_fallback=True,
            )

        optimized_weights = np.clip(result.x, 0.0, 1.0)
        optimized_risky = optimized_weights[:-1] if allow_cash_buffer else optimized_weights
        optimized_cash = float(optimized_weights[-1]) if allow_cash_buffer else 0.0
        expected_return, volatility, sortino, calmar, max_drawdown = self._portfolio_metrics(
            risky_weights=optimized_risky,
            cash_weight=optimized_cash,
            returns_matrix=returns_matrix,
        )
        weights = {
            token: float(weight)
            for token, weight in zip(risky_tokens, optimized_risky)
        }
        if allow_cash_buffer:
            weights[cash_symbol] = optimized_cash

        return OptimizationResult(
            weights=weights,
            risky_weights=optimized_risky,
            cash_weight=optimized_cash,
            expected_return=expected_return,
            volatility=volatility,
            sortino=sortino,
            calmar=calmar,
            max_drawdown=max_drawdown,
            used_fallback=False,
        )

    def _capped_priority_allocation(
        self, scores: np.ndarray, caps: np.ndarray, allow_cash: bool = True
    ) -> Tuple[np.ndarray, float]:
        """在权重上限约束下分配初始权重，并把剩余仓位留给现金位。"""
        weights = np.zeros_like(scores, dtype=float)
        positive_scores = np.maximum(scores, 0.0)

        if positive_scores.sum() <= NUMERICAL_EPSILON:
            positive_scores = np.ones_like(scores, dtype=float)

        active_indices = {idx for idx, cap in enumerate(caps) if cap > NUMERICAL_EPSILON}
        while active_indices:
            remaining = max(0.0, 1.0 - float(weights.sum()))
            if remaining <= NUMERICAL_EPSILON:
                break

            total_score = float(sum(positive_scores[idx] for idx in active_indices))
            if total_score <= NUMERICAL_EPSILON:
                break

            distributed = 0.0
            saturated: List[int] = []
            for idx in list(active_indices):
                ideal_addition = remaining * (positive_scores[idx] / total_score)
                room = float(caps[idx] - weights[idx])
                add_weight = min(ideal_addition, room)
                if add_weight > NUMERICAL_EPSILON:
                    weights[idx] += add_weight
                    distributed += add_weight
                if caps[idx] - weights[idx] <= NUMERICAL_EPSILON:
                    saturated.append(idx)

            for idx in saturated:
                active_indices.discard(idx)

            if distributed <= NUMERICAL_EPSILON:
                break

        if not allow_cash and weights.sum() < 1.0 - NUMERICAL_EPSILON:
            residual = 1.0 - float(weights.sum())
            for idx in np.argsort(-positive_scores):
                room = float(caps[idx] - weights[idx])
                if room <= NUMERICAL_EPSILON:
                    continue
                add_weight = min(room, residual)
                weights[idx] += add_weight
                residual -= add_weight
                if residual <= NUMERICAL_EPSILON:
                    break

        cash_weight = max(0.0, 1.0 - float(weights.sum())) if allow_cash else 0.0
        return weights, cash_weight

    def _build_reference_portfolio(
        self,
        risky_tokens: Sequence[str],
        returns_matrix: np.ndarray,
        cash_symbol: str,
        current_weights: Optional[Dict[str, float]],
    ) -> ReferencePortfolio:
        """构造用于前后对比的原始参考组合。"""
        asset_count = len(risky_tokens)
        if asset_count == 0:
            return ReferencePortfolio(
                label="等权参考组合",
                weights={cash_symbol: 1.0},
                risky_weights=np.array([], dtype=float),
                cash_weight=1.0,
                expected_return=RISK_FREE_RATE,
                volatility=0.0,
                sortino=0.0,
                calmar=0.0,
                max_drawdown=0.0,
            )

        if current_weights:
            normalized_map = {
                normalize_token(token): max(float(weight), 0.0)
                for token, weight in current_weights.items()
            }
            raw_total = float(sum(normalized_map.values()))
            scale = 100.0 if raw_total > 1.5 else 1.0
            normalized_map = {token: weight / scale for token, weight in normalized_map.items()}

            risky_weights = np.array(
                [normalized_map.get(token, 0.0) for token in risky_tokens],
                dtype=float,
            )
            cash_weight = float(sum(normalized_map.get(token, 0.0) for token in STABLECOINS))
            total_weight = float(risky_weights.sum() + cash_weight)

            if total_weight > NUMERICAL_EPSILON:
                risky_weights = risky_weights / total_weight
                cash_weight = cash_weight / total_weight
                expected_return, volatility, sortino, calmar, max_drawdown = self._portfolio_metrics(
                    risky_weights=risky_weights,
                    cash_weight=cash_weight,
                    returns_matrix=returns_matrix,
                )
                weight_map = {
                    token: float(weight) for token, weight in zip(risky_tokens, risky_weights)
                }
                if cash_weight > NUMERICAL_EPSILON:
                    weight_map[cash_symbol] = cash_weight
                return ReferencePortfolio(
                    label="用户当前持仓",
                    weights=weight_map,
                    risky_weights=risky_weights,
                    cash_weight=cash_weight,
                    expected_return=expected_return,
                    volatility=volatility,
                    sortino=sortino,
                    calmar=calmar,
                    max_drawdown=max_drawdown,
                )

        equal_risky_weights = np.full(asset_count, 1.0 / asset_count, dtype=float)
        expected_return, volatility, sortino, calmar, max_drawdown = self._portfolio_metrics(
            risky_weights=equal_risky_weights,
            cash_weight=0.0,
            returns_matrix=returns_matrix,
        )
        return ReferencePortfolio(
            label="等权参考组合",
            weights={token: float(weight) for token, weight in zip(risky_tokens, equal_risky_weights)},
            risky_weights=equal_risky_weights,
            cash_weight=0.0,
            expected_return=expected_return,
            volatility=volatility,
            sortino=sortino,
            calmar=calmar,
            max_drawdown=max_drawdown,
        )

    def _portfolio_metrics(
        self,
        risky_weights: np.ndarray,
        cash_weight: float,
        returns_matrix: np.ndarray,
    ) -> Tuple[float, float, float, float, float]:
        """计算组合年化收益、波动率、Sortino、Calmar 与最大回撤。"""
        if returns_matrix.size == 0:
            return RISK_FREE_RATE, 0.0, 0.0, 0.0, 0.0

        portfolio_returns = returns_matrix @ risky_weights + cash_weight * _risk_free_period_return()
        portfolio_returns_series = pd.Series(portfolio_returns)
        annual_return = _annualize_return_from_periodic_returns(portfolio_returns_series)
        annual_volatility = float(portfolio_returns_series.std(ddof=1) * math.sqrt(PERIODS_PER_YEAR))
        max_drawdown = _compute_max_drawdown(portfolio_returns_series)
        sortino = _compute_sortino_ratio(annual_return, portfolio_returns_series)
        calmar = _compute_calmar_ratio(annual_return, max_drawdown)
        return annual_return, annual_volatility, sortino, calmar, max_drawdown

    def _format_markdown_table(
        self, headers: Sequence[str], rows: Sequence[Sequence[str]]
    ) -> List[str]:
        """生成 Markdown 表格。"""
        table_lines = [
            "| " + " | ".join(headers) + " |",
            "| " + " | ".join("---" for _ in headers) + " |",
        ]
        for row in rows:
            table_lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
        return table_lines

    def _compute_asset_portfolio_correlation(
        self,
        returns_frame: pd.DataFrame,
        risky_tokens: Sequence[str],
        risky_weights: np.ndarray,
    ) -> Dict[str, float]:
        """计算每个资产与优化后风险组合收益流的相关系数。"""
        risky_returns = returns_frame[list(risky_tokens)]
        portfolio_returns = risky_returns.mul(risky_weights, axis=1).sum(axis=1)
        if portfolio_returns.std(ddof=1) <= NUMERICAL_EPSILON:
            return {token: 0.0 for token in risky_tokens}

        correlations: Dict[str, float] = {}
        for token in risky_tokens:
            corr_value = risky_returns[token].corr(portfolio_returns)
            correlations[token] = 0.0 if pd.isna(corr_value) else float(corr_value)
        return correlations

    def _get_token_max_correlation(
        self, token: str, correlation_matrix: pd.DataFrame
    ) -> Tuple[str, float]:
        """找出某个资产与其他资产的最高相关对象。"""
        if token not in correlation_matrix.index:
            return "-", 0.0

        token_correlations = correlation_matrix.loc[token].drop(labels=[token], errors="ignore")
        if token_correlations.empty:
            return "-", 0.0

        top_peer = str(token_correlations.idxmax())
        top_corr = float(token_correlations.loc[top_peer])
        return top_peer, top_corr

    def _grade_asset(
        self,
        token: str,
        stats: AssetStats,
        weight: float,
        correlation_to_portfolio: float,
        dominated_tokens: set[str],
    ) -> Tuple[str, str]:
        """给资产做更适合客户阅读的综合评级。"""
        score = 0
        if stats.sortino >= 2:
            score += 2
        elif stats.sortino >= 1:
            score += 1
        elif stats.sortino < 0:
            score -= 1

        if stats.calmar >= 1:
            score += 2
        elif stats.calmar >= 0.3:
            score += 1
        elif stats.calmar < 0:
            score -= 1

        if stats.max_drawdown <= 0.25:
            score += 1
        elif stats.max_drawdown >= 0.55:
            score -= 1

        if correlation_to_portfolio <= 0.55:
            score += 1
        elif correlation_to_portfolio >= 0.85:
            score -= 1

        if token in dominated_tokens:
            score -= 1
        if stats.is_new:
            score -= 1
        if stats.used_mock:
            score -= 1
        if weight <= 0.001:
            score = min(score, 0)

        if score >= 3:
            return "A", "优先配置"
        if score >= 1:
            return "B", "可保留"
        if score >= 0:
            return "C", "观察"
        return "D", "建议削减"

    def _describe_weight_change(self, before_weight: float, after_weight: float) -> str:
        """把前后权重变化转成客户易懂的动作建议。"""
        delta = after_weight - before_weight
        if delta >= 0.10:
            return "明显加仓"
        if delta >= 0.03:
            return "适度加仓"
        if delta <= -0.10:
            return "明显减仓"
        if delta <= -0.03:
            return "适度减仓"
        return "基本维持"

    def _describe_portfolio_posture(
        self, risky_tokens: Sequence[str], optimization: OptimizationResult
    ) -> Tuple[str, str]:
        """根据权重结构给组合一个更容易理解的风格标签。"""
        risky_allocations = sorted(
            ((token, optimization.weights.get(token, 0.0)) for token in risky_tokens),
            key=lambda item: item[1],
            reverse=True,
        )
        top_two_weight = sum(weight for _, weight in risky_allocations[:2])

        if optimization.cash_weight >= 0.20:
            return "防守型", "模型保留了较高比例的稳定币缓冲，说明它更重视回撤控制与等待更优时机。"
        if optimization.cash_weight >= 0.08:
            return "平衡偏防守", "模型愿意保留一部分现金仓位，说明它在追求收益的同时，也在主动压低整体波动。"
        if top_two_weight >= 0.75:
            return "集中进攻型", "模型把大部分仓位集中在少数效率更高的资产上，追求更强弹性，但集中度也更高。"
        return "平衡型", "模型在核心资产与弹性资产之间做了分层配置，目标是在收益和波动之间取得更稳妥的平衡。"

    def _describe_asset_position(
        self,
        token: str,
        weight: float,
        stats: AssetStats,
        dominated_tokens: set[str],
    ) -> Tuple[str, str]:
        """把量化结果翻译成更适合客户理解的仓位角色与动作建议。"""
        if weight <= 0.001:
            role = "剔除候选"
            action = "建议暂不配置，避免资金占用在当前性价比偏低的资产上。"
        elif (stats.sortino < 0 or stats.calmar < 0) and weight >= 0.10:
            role = "策略性保留"
            action = "当前保留更多是为了分散或满足组合约束，不代表它属于高置信增配对象。"
        elif (stats.sortino < 0 or stats.calmar < 0) and weight >= 0.03:
            role = "低置信配置"
            action = "可以少量保留做结构平衡，但不适合作为主要进攻仓位。"
        elif stats.is_new and weight <= 0.05:
            role = "试错仓"
            action = "可以小仓位参与，但更适合作为高风险弹性仓，而不是主力仓位。"
        elif weight >= 0.30:
            role = "核心仓"
            action = "适合作为组合的主要收益来源，应重点跟踪其趋势与回撤。"
        elif weight >= 0.10:
            role = "重点仓"
            action = "适合作为核心资产外的第二梯队配置，承担一定增长弹性。"
        elif weight >= 0.03:
            role = "卫星仓"
            action = "适合作为弹性补充，既参与上涨，也控制单一资产冲击。"
        else:
            role = "观察仓"
            action = "仅保留小比例跟踪，等待更明确的风险收益改善后再决定是否加仓。"

        if token in dominated_tokens and weight > 0:
            action += " 由于与更高效率资产高度重叠，这个仓位需要被严格控制。"
        if stats.max_drawdown >= 0.55:
            action += " 最近样本中的最大回撤偏深，需要把底线风险放在第一位。"
        if stats.sortino < 0 or stats.calmar < 0:
            action += " 从最近样本看，它的下行性价比并不理想。"
        elif stats.sortino >= 2 and stats.calmar >= 1:
            action += " 从最近样本看，它的下行风险调整后性价比处于组合前列。"
        return role, action

    def _classify_asset_actions(
        self,
        risky_allocations: Sequence[Tuple[str, float]],
        stats_map: Dict[str, AssetStats],
        grade_map: Dict[str, str],
        dominated_tokens: set[str],
    ) -> Tuple[List[str], List[str], List[str]]:
        """统一生成高置信增配、策略性保留、控制削减三类动作。"""
        keep_tokens: List[str] = []
        strategic_hold_tokens: List[str] = []
        reduce_tokens: List[str] = []

        for token, weight in risky_allocations:
            stats = stats_map[token]
            grade = grade_map[token]
            if weight >= 0.10 and grade in {"A", "B"} and stats.sortino > 0 and stats.calmar > 0:
                keep_tokens.append(token)
                continue
            if weight > 0 and (grade == "D" or token in dominated_tokens or stats.sortino < 0 or stats.calmar < 0):
                reduce_tokens.append(token)
                continue
            if weight >= 0.03:
                strategic_hold_tokens.append(token)

        return keep_tokens[:3], strategic_hold_tokens[:3], reduce_tokens[:4]

    def _format_report(
        self,
        risky_tokens: Sequence[str],
        stats_map: Dict[str, AssetStats],
        overlap_pairs: Sequence[Tuple[str, str, float, str]],
        correlation_matrix: pd.DataFrame,
        returns_frame: pd.DataFrame,
        optimization: OptimizationResult,
        reference_portfolio: ReferencePortfolio,
        total_capital: float,
        cash_symbol: str,
        warnings: Sequence[str],
    ) -> str:
        """将量化结果格式化为专业中文 Markdown。"""
        baseline_return = reference_portfolio.expected_return
        baseline_volatility = reference_portfolio.volatility
        baseline_sortino = reference_portfolio.sortino
        baseline_calmar = reference_portfolio.calmar
        baseline_max_drawdown = reference_portfolio.max_drawdown
        dominated_tokens = {loser for _, _, _, loser in overlap_pairs}
        posture_title, posture_description = self._describe_portfolio_posture(risky_tokens, optimization)
        asset_to_portfolio_corr = self._compute_asset_portfolio_correlation(
            returns_frame=returns_frame,
            risky_tokens=risky_tokens,
            risky_weights=optimization.risky_weights,
        )
        grade_map: Dict[str, str] = {}
        grade_label_map: Dict[str, str] = {}
        for token in risky_tokens:
            grade, grade_label = self._grade_asset(
                token=token,
                stats=stats_map[token],
                weight=optimization.weights.get(token, 0.0),
                correlation_to_portfolio=asset_to_portfolio_corr.get(token, 0.0),
                dominated_tokens=dominated_tokens,
            )
            grade_map[token] = grade
            grade_label_map[token] = grade_label
        risky_allocations = sorted(
            ((token, optimization.weights.get(token, 0.0)) for token in risky_tokens),
            key=lambda item: item[1],
            reverse=True,
        )
        keep_tokens, strategic_hold_tokens, reduce_tokens = self._classify_asset_actions(
            risky_allocations=risky_allocations,
            stats_map=stats_map,
            grade_map=grade_map,
            dominated_tokens=dominated_tokens,
        )
        strongest_overlap = max(overlap_pairs, key=lambda item: item[2]) if overlap_pairs else None
        lines: List[str] = ["# Web3 持仓组合优化报告", ""]

        if warnings:
            lines.append("> 系统提示")
            for warning in warnings:
                lines.append(f"> - {warning}")
            lines.append("")

        lines.append("## 执行摘要")
        if strongest_overlap:
            left, right, corr_value, loser = strongest_overlap
            winner = right if loser == left else left
            lines.append(
                f"- 当前组合最突出的结构问题是 `{left}` 与 `{right}` 高度同涨同跌（相关系数 `{corr_value:.2f}`），"
                f" 这会让“看起来分散”的持仓在下跌时一起回撤。模型建议优先保留 `{winner}`，压缩 `{loser}`。"
            )
        else:
            lines.append("- 当前组合没有发现特别严重的相关性重叠，分散结构整体是健康的。")

        keep_text = "、".join(f"`{token}`" for token in keep_tokens) if keep_tokens else "暂无高置信增配资产"
        strategic_hold_text = (
            "、".join(f"`{token}`" for token in strategic_hold_tokens)
            if strategic_hold_tokens
            else "暂无需要单独说明的策略性保留仓位"
        )
        reduce_text = "、".join(f"`{token}`" for token in reduce_tokens) if reduce_tokens else "暂无明确需要控制的低效仓位"
        lines.append(f"- 优化后的组合风格是 `{posture_title}`。{posture_description}")
        lines.append(f"- 建议重点保留或增配：{keep_text}。")
        if strategic_hold_tokens:
            lines.append(f"- 约束下保留/低置信配置：{strategic_hold_text}。这部分仓位更多用于分散和平衡，不宜按高置信主仓理解。")
        lines.append(f"- 建议重点控制或削减：{reduce_text}。")
        if reference_portfolio.label == "用户当前持仓":
            lines.append("- 本次前后对比以你提供的原始持仓比例为基准，而不是默认等权。")
        else:
            lines.append("- 由于未提供原始持仓比例，本报告默认将“风险资产等权持有”视作原始参考方案。")
        lines.append(
            "- 通俗理解：这套方案不是单纯追涨，而是优先把资金留给“最近 30 天里更能打、且不那么重复”的资产。"
        )
        lines.append("")

        lines.append("## 1. 数据口径与方法披露")
        lines.append("- 数据窗口：最近 `30 天`、`4 小时` K 线，稳定币不进入波动率矩阵，仅作为无风险基准。")
        lines.append("- 数据来源：优先 `OKX`，必要时降级 `Gate.io`；若双源失效，会显式提示并启用 Mock Data。")
        lines.append("- 方法框架：先看相关性，再看单资产 Sortino / Calmar / 最大回撤，最后在权重上限约束下做 SLSQP 优化。")
        lines.append("- 约束规则：`BTC/ETH` 单币上限 `50%`，蓝筹资产上限 `30%`，长尾资产上限 `15%`，新币上限 `5%`。")
        lines.append("- 解释口径：Sortino 越高代表下行风险调整后的性价比越好，Calmar 越高代表收益与回撤更划算，最大回撤越低越稳。")
        lines.append("")

        lines.append("## 2. 相关性与重叠诊断")
        lines.append("### 资产两两相关系数矩阵")
        matrix_headers = ["资产"] + list(risky_tokens)
        matrix_rows: List[List[str]] = []
        for row_token in risky_tokens:
            row_cells = [f"`{row_token}`"]
            for col_token in risky_tokens:
                corr_value = float(correlation_matrix.loc[row_token, col_token])
                row_cells.append(f"{corr_value:.2f}")
            matrix_rows.append(row_cells)
        lines.extend(self._format_markdown_table(matrix_headers, matrix_rows))
        lines.append("")

        lines.append("### 单资产风险画像表")
        profile_headers = [
            "资产",
            "与组合相关性",
            "最高相关对象",
            "最高相关系数",
            "Sortino",
            "Calmar",
            "最大回撤",
            "年化收益",
            "年化波动",
            "建议权重",
            "评级",
        ]
        profile_rows: List[List[str]] = []
        for token in risky_tokens:
            stats = stats_map[token]
            max_peer, max_corr = self._get_token_max_correlation(token, correlation_matrix)
            profile_rows.append(
                [
                    f"`{token}`",
                    f"{asset_to_portfolio_corr.get(token, 0.0):.2f}",
                    f"`{max_peer}`" if max_peer != "-" else "-",
                    f"{max_corr:.2f}",
                    f"{stats.sortino:.2f}",
                    f"{stats.calmar:.2f}",
                    f"{stats.max_drawdown:.2%}",
                    f"{stats.expected_return:.2%}",
                    f"{stats.volatility:.2%}",
                    f"{optimization.weights.get(token, 0.0):.2%}",
                    f"{grade_map[token]} / {grade_label_map[token]}",
                ]
            )
        lines.extend(self._format_markdown_table(profile_headers, profile_rows))
        lines.append("")

        lines.append("### 重点重叠分析")
        if overlap_pairs:
            for left, right, corr_value, loser in overlap_pairs:
                winner = right if loser == left else left
                loser_sortino = stats_map[loser].sortino
                winner_sortino = stats_map[winner].sortino
                loser_calmar = stats_map[loser].calmar
                winner_calmar = stats_map[winner].calmar
                lines.append(
                    f"- `{left}` 与 `{right}` 的相关系数为 `{corr_value:.2f}`，已达到 [风险重叠组] 标准。"
                    f" 基于下行性价比对决，建议优先保留 `{winner}`（Sortino `{winner_sortino:.2f}` / Calmar `{winner_calmar:.2f}`），"
                    f" 将 `{loser}`（Sortino `{loser_sortino:.2f}` / Calmar `{loser_calmar:.2f}`）削减至极低权重或直接剔除。"
                    " 简单理解：这两类资产历史上经常一起涨跌，同时拿太多，分散效果会明显打折。"
                )
        else:
            lines.append(
                "- 当前输入资产之间未发现高于 `0.85` 的显著相关性重叠，组合结构相对健康。"
                " 简单理解：这些资产的走势没有严重“绑死”，分散持有更有意义。"
            )
        lines.append("")

        sorted_stats = sorted(
            stats_map.values(),
            key=lambda item: (item.sortino, item.calmar, -item.max_drawdown, item.expected_return),
        )
        best_asset = sorted_stats[-1]
        worst_asset = sorted_stats[0]
        blind_boxes = [stats.token for stats in stats_map.values() if stats.is_new]

        lines.append("## 3. 资产效率分析与综合评级")
        lines.append(
            f"- 效率最高的资产是 `{best_asset.token}`：年化预期收益约 `{best_asset.expected_return:.2%}`，"
            f" 年化波动率约 `{best_asset.volatility:.2%}`，Sortino `{best_asset.sortino:.2f}`，Calmar `{best_asset.calmar:.2f}`，最大回撤 `{best_asset.max_drawdown:.2%}`。"
        )
        lines.append(
            f"- 最拖后腿的资产是 `{worst_asset.token}`：年化预期收益约 `{worst_asset.expected_return:.2%}`，"
            f" 年化波动率约 `{worst_asset.volatility:.2%}`，Sortino `{worst_asset.sortino:.2f}`，Calmar `{worst_asset.calmar:.2f}`，最大回撤 `{worst_asset.max_drawdown:.2%}`。"
        )
        lines.append(
            f"- 通俗理解：`{best_asset.token}` 是最近这段时间里“下跌风险调整后更划算、而且回撤更能接受”的资产；"
            f" `{worst_asset.token}` 则更像是在承担回撤，但补偿还不够高。"
        )
        if blind_boxes:
            lines.append(
                f"- `[高风险盲盒]` 资产：{', '.join(f'`{token}`' for token in blind_boxes)}，"
                "这些资产因历史样本不足，已被自动施加 5% 上限。"
            )
        mock_assets = [stats.token for stats in stats_map.values() if stats.used_mock]
        if mock_assets:
            lines.append(
                f"- 使用 Mock Data 的资产：{', '.join(f'`{token}`' for token in mock_assets)}，"
                "请在真实 API 恢复后重新跑一次，结论会更稳健。"
            )
        grade_buckets: Dict[str, List[str]] = {"A": [], "B": [], "C": [], "D": []}
        for token in risky_tokens:
            grade_buckets[grade_map[token]].append(f"`{token}`({grade_label_map[token]})")
        lines.append("### 分析阶梯")
        lines.append(
            f"- `A 级` 优先配置：{('、'.join(grade_buckets['A']) if grade_buckets['A'] else '暂无')}。"
            " 这类资产通常兼具较高单位风险回报和较明确的配置价值。"
        )
        lines.append(
            f"- `B 级` 可保留：{('、'.join(grade_buckets['B']) if grade_buckets['B'] else '暂无')}。"
            " 这类资产可以保留在组合中，但优先级低于 A 级。"
        )
        lines.append(
            f"- `C 级` 观察：{('、'.join(grade_buckets['C']) if grade_buckets['C'] else '暂无')}。"
            " 这类资产暂时不建议给太多仓位，适合继续跟踪。"
        )
        lines.append(
            f"- `D 级` 建议削减：{('、'.join(grade_buckets['D']) if grade_buckets['D'] else '暂无')}。"
            " 这类资产要么效率偏弱，要么与更优资产重叠过高。"
        )
        lines.append("### 单资产逐项点评")
        for token in risky_tokens:
            weight = optimization.weights.get(token, 0.0)
            stats = stats_map[token]
            role, action = self._describe_asset_position(token, weight, stats, dominated_tokens)
            lines.append(
                f"- `{token}`：定位 `{role}`。当前建议权重 `{weight:.2%}`，年化收益 `{stats.expected_return:.2%}`，"
                f" 年化波动 `{stats.volatility:.2%}`，Sortino `{stats.sortino:.2f}`，Calmar `{stats.calmar:.2f}`，最大回撤 `{stats.max_drawdown:.2%}`，"
                f" 与优化组合相关性 `{asset_to_portfolio_corr.get(token, 0.0):.2f}`，综合评级 `{grade_map[token]} / {grade_label_map[token]}`。{action}"
            )
        lines.append("")

        lines.append("## 4. 原始参考方案 vs 优化方案")
        lines.append(f"- 说明：本节原始参考方案采用 `{reference_portfolio.label}`。")
        max_single_weight = max((optimization.weights.get(token, 0.0) for token in risky_tokens), default=0.0)
        reference_max_single_weight = max(
            (reference_portfolio.weights.get(token, 0.0) for token in risky_tokens),
            default=0.0,
        )
        comparison_headers = ["指标", "原始参考方案（等权）", "优化方案", "变化说明"]
        comparison_rows = [
            [
                "预期年化收益",
                f"{baseline_return:.2%}",
                f"{optimization.expected_return:.2%}",
                "越高越好，但需要结合波动一起看",
            ],
            [
                "年化波动率",
                f"{baseline_volatility:.2%}",
                f"{optimization.volatility:.2%}",
                "越低越稳，代表净值起伏更可控",
            ],
            [
                "Sortino",
                f"{baseline_sortino:.2f}",
                f"{optimization.sortino:.2f}",
                "越高代表下行风险调整后的性价比越好",
            ],
            [
                "Calmar",
                f"{baseline_calmar:.2f}",
                f"{optimization.calmar:.2f}",
                "越高代表收益相对最大回撤更划算",
            ],
            [
                "最大回撤",
                f"{baseline_max_drawdown:.2%}",
                f"{optimization.max_drawdown:.2%}",
                "越低越好，代表底线风险更可控",
            ],
            [
                "最大单币权重",
                f"{reference_max_single_weight:.2%}",
                f"{max_single_weight:.2%}",
                "用于观察组合集中度",
            ],
            [
                "现金缓冲",
                f"{reference_portfolio.cash_weight:.2%}",
                f"{optimization.cash_weight:.2%}",
                "用于承接波动和保留机动性",
            ],
        ]
        comparison_headers[1] = f"原始参考方案（{reference_portfolio.label}）"
        lines.extend(self._format_markdown_table(comparison_headers, comparison_rows))
        lines.append("")

        lines.append("### 权重变化对比表")
        weight_headers = ["资产", "原始参考权重", "优化权重", "变动", "金额变化", "执行动作"]
        weight_rows: List[List[str]] = []
        for token in risky_tokens:
            before_weight = reference_portfolio.weights.get(token, 0.0)
            after_weight = optimization.weights.get(token, 0.0)
            delta_weight = after_weight - before_weight
            weight_rows.append(
                [
                    f"`{token}`",
                    f"{before_weight:.2%}",
                    f"{after_weight:.2%}",
                    f"{delta_weight:+.2%}",
                    f"{delta_weight * total_capital:+,.2f} USDT",
                    self._describe_weight_change(before_weight, after_weight),
                ]
            )
        if optimization.cash_weight > 0.001:
            before_cash_weight = reference_portfolio.cash_weight
            weight_rows.append(
                [
                    f"`{cash_symbol}`",
                    f"{before_cash_weight:.2%}",
                    f"{optimization.cash_weight:.2%}",
                    f"{(optimization.cash_weight - before_cash_weight):+.2%}",
                    f"{(optimization.cash_weight - before_cash_weight) * total_capital:+,.2f} USDT",
                    "保留缓冲",
                ]
            )
        lines.extend(self._format_markdown_table(weight_headers, weight_rows))
        lines.append("")

        lines.append("## 5. 最终建议与执行路径")
        lines.append("### 目标配置表")
        target_headers = ["资产", "建议权重", "建议金额", "仓位角色", "有效上限", "数据源"]
        target_rows: List[List[str]] = []
        for token in risky_tokens:
            weight = optimization.weights.get(token, 0.0)
            amount = weight * total_capital
            stats = stats_map[token]
            role, _ = self._describe_asset_position(token, weight, stats, dominated_tokens)
            tags: List[str] = []
            if stats.is_new:
                tags.append("高风险盲盒")
            if stats.used_mock:
                tags.append("Mock")
            role_text = role if not tags else f"{role} / {' / '.join(tags)}"
            target_rows.append(
                [
                    f"`{token}`",
                    f"{weight:.2%}",
                    f"{amount:,.2f} USDT",
                    role_text,
                    f"{stats.effective_cap:.0%}",
                    stats.source,
                ]
            )
        if optimization.cash_weight > 0.001:
            cash_amount = optimization.cash_weight * total_capital
            target_rows.append(
                [
                    f"`{cash_symbol}`",
                    f"{optimization.cash_weight:.2%}",
                    f"{cash_amount:,.2f} USDT",
                    "现金缓冲",
                    "-",
                    "Risk-free anchor",
                ]
            )
        lines.extend(self._format_markdown_table(target_headers, target_rows))
        if optimization.used_fallback:
            lines.append("")
            lines.append("- 由于 SLSQP 未稳定收敛，最终仓位使用了保底分配器；约束仍被严格满足。")
        lines.append("")
        lines.append("### 建议执行顺序")
        step_lines: List[str] = []
        if reduce_tokens:
            step_lines.append(
                f"1. 先处理重叠或效率偏弱的资产：优先把 {'、'.join(f'`{token}`' for token in reduce_tokens)} 降到建议区间。"
            )
        else:
            step_lines.append("1. 先核对当前持仓与目标权重差距，确认是否需要实际调仓。")
        if keep_tokens:
            step_lines.append(
                f"2. 再把资金集中到核心建议仓位：重点向 {'、'.join(f'`{token}`' for token in keep_tokens)} 靠拢。"
            )
        else:
            step_lines.append("2. 再根据风险偏好把资金补到模型给出的主力配置。")
        if strategic_hold_tokens:
            step_lines.append(
                f"3. 对 {'、'.join(f'`{token}`' for token in strategic_hold_tokens)} 这类策略性保留资产，只保留模型建议的小中仓位，不要按高置信主仓追配。"
            )
        else:
            step_lines.append("3. 对剩余非核心资产维持纪律性仓位，避免把辅助配置误当成主攻方向。")
        if optimization.cash_weight > 0.001:
            step_lines.append(
                f"4. 最后保留 `{cash_symbol}` 现金缓冲 `{optimization.cash_weight:.2%}`，为后续波动或补仓预留机动性。"
            )
        else:
            step_lines.append("4. 组合已经接近满仓，后续重点放在仓位纪律和止盈止损执行。")
        lines.extend(step_lines)
        lines.append("")

        lines.append("## 6. 结论与优化效果")
        volatility_reduction = 0.0
        if baseline_volatility > NUMERICAL_EPSILON:
            volatility_reduction = (baseline_volatility - optimization.volatility) / baseline_volatility * 100.0
        volatility_text = (
            f"波动率降低 `{volatility_reduction:.1f}%`"
            if volatility_reduction >= 0
            else f"波动率上升 `{abs(volatility_reduction):.1f}%`"
        )
        max_drawdown_improvement = 0.0
        if baseline_max_drawdown > NUMERICAL_EPSILON:
            max_drawdown_improvement = (
                (baseline_max_drawdown - optimization.max_drawdown) / baseline_max_drawdown * 100.0
            )
        drawdown_text = (
            f"最大回撤降低 `{max_drawdown_improvement:.1f}%`"
            if max_drawdown_improvement >= 0
            else f"最大回撤上升 `{abs(max_drawdown_improvement):.1f}%`"
        )
        baseline_phrase = (
            "相比于你提供的原始持仓"
            if reference_portfolio.label == "用户当前持仓"
            else "相比于默认等权参考组合"
        )

        if abs(baseline_sortino) > NUMERICAL_EPSILON:
            sortino_improvement = (optimization.sortino - baseline_sortino) / abs(baseline_sortino) * 100.0
            calmar_improvement = (
                (optimization.calmar - baseline_calmar) / abs(baseline_calmar) * 100.0
                if abs(baseline_calmar) > NUMERICAL_EPSILON
                else 0.0
            )
            lines.append(
                f"- {baseline_phrase}，优化后的组合预期年化收益约 `{optimization.expected_return:.2%}`，"
                f" 年化波动率约 `{optimization.volatility:.2%}`，{volatility_text}，{drawdown_text}，"
                f" Sortino 提升 `{sortino_improvement:.1f}%`，Calmar 提升 `{calmar_improvement:.1f}%`。"
            )
        else:
            lines.append(
                f"- {baseline_phrase}，优化后的组合预期年化收益约 `{optimization.expected_return:.2%}`，"
                f" 年化波动率约 `{optimization.volatility:.2%}`，{volatility_text}，{drawdown_text}，"
                f" Sortino 由 `{baseline_sortino:.2f}` 变为 `{optimization.sortino:.2f}`，"
                f" Calmar 由 `{baseline_calmar:.2f}` 变为 `{optimization.calmar:.2f}`。"
            )
        if volatility_reduction >= 0:
            lines.append("- 这说明模型在当前样本下，用更合理的仓位结构换到了更高的单位风险回报。")
        else:
            lines.append("- 这说明模型接受了更高波动，去换取更好的收益弹性；适合风险承受能力更强的客户。")
        lines.append("")
        lines.append("## 7. 合规与风险揭示")
        lines.append("- 本报告属于历史数据驱动的量化分析输出，用于辅助理解组合结构，不构成保证收益的投资承诺。")
        lines.append("- 本模型未接入你的真实财务状况、负债、税务、流动性需求和适当性测评，因此不属于个性化投资顾问服务。")
        lines.append("- 报告默认基于现货视角，不包含杠杆、合约、滑点、手续费、链上冲击成本和极端流动性风险。")
        lines.append("- 对新币、样本不足资产或启用了 Mock Data 的资产，应降低信任权重，并在真实成交前做额外人工复核。")
        lines.append("- 若面向客户展示，建议同时披露样本窗口、数据来源、策略边界和“历史表现不代表未来”的风险提示。")
        lines.append("")
        lines.append("## 8. 指标说明")
        lines.append("- `相关系数`：越接近 1，说明两个币越容易一起涨跌；相关性太高时，分散效果会变差。")
        lines.append("- `与组合相关性`：越高说明该资产越像是“主导组合波动”的成员，越低则更像分散化补充。")
        lines.append("- `年化波动率`：可以理解为价格起伏的剧烈程度，越高代表净值上下波动越大。")
        lines.append("- `Sortino`：只惩罚下跌波动，越高代表下行风险调整后的性价比越好。")
        lines.append("- `Calmar`：看收益相对最大回撤是否划算，越高越说明“这份回撤换来的收益更值”。")
        lines.append("- `最大回撤`：观察从阶段高点回落最深有多大，越低越能守住底线。")
        lines.append("- 风险提示：本报告基于最近 30 天、4 小时级别历史数据，适合做结构优化参考，不代表未来收益承诺。")

        return "\n".join(lines)

    def _format_empty_report(self, total_capital: float) -> str:
        """空输入时返回一致格式的结果。"""
        return "\n".join(
            [
                "# Web3 持仓组合优化报告",
                "",
                "## 1. 风险重叠警告",
                "- 未收到有效代币列表，暂无法生成相关性矩阵。",
                "",
                "## 2. 资产效率分析",
                "- 没有可分析的资产，请至少输入一个代币代码，例如 `BTC, ETH, SOL`。",
                "",
                "## 3. 最终优化比例建议",
                f"- `USDT`(现金缓冲): `100.00%` | `{total_capital:,.2f} USDT`",
                "",
                "## 4. 💡 优化效果",
                "- 当前为占位结果；补充代币后即可生成真实优化方案。",
            ]
        )

    def _format_cash_only_report(
        self, cash_symbol: str, total_capital: float, warnings: Sequence[str]
    ) -> str:
        """只有稳定币或全部无效代币时的保底输出。"""
        lines = ["# Web3 持仓组合优化报告", ""]
        if warnings:
            lines.append("> 系统提示")
            for warning in deduplicate_messages(warnings):
                lines.append(f"> - {warning}")
            lines.append("")

        lines.extend(
            [
                "## 1. 风险重叠警告",
                "- 当前没有可参与波动率建模的风险资产，因此不存在重叠风险。",
                "",
                "## 2. 资产效率分析",
                f"- 当前仅保留 `{cash_symbol}` 作为无风险基准仓位，适合等待更好的入场窗口。",
                "",
                "## 3. 最终优化比例建议",
                f"- `{cash_symbol}`(现金缓冲): `100.00%` | `{total_capital:,.2f} USDT`",
                "",
                "## 4. 💡 优化效果",
                "- 在缺少有效风险资产样本时，保持稳定币仓位是波动率最低的保守解。",
            ]
        )
        return "\n".join(lines)

    def _build_unknown_token_warning(self, token: str) -> str:
        """构造拼写友好的提示信息。"""
        matches = difflib.get_close_matches(token, KNOWN_TOKENS, n=3, cutoff=0.6)
        if matches:
            suggestion_text = " / ".join(matches)
            return f"未查询到 `{token}` 的有效交易对，请检查拼写；你可能想输入：`{suggestion_text}`。其他资产已继续分析。"
        return f"未查询到 `{token}` 的有效交易对，请检查拼写；其他资产已继续分析。"

    def _target_index(self) -> pd.DatetimeIndex:
        """统一生成 30 天 4 小时频率的时间轴。"""
        end_time = pd.Timestamp.utcnow().floor("4h")
        return pd.date_range(end=end_time, periods=CANDLE_LIMIT, freq="4h", tz="UTC")

    def _nearest_positive_semidefinite(self, matrix: np.ndarray) -> np.ndarray:
        """将协方差矩阵投影到最近的半正定矩阵，避免数值优化报错。"""
        symmetric = (matrix + matrix.T) / 2.0
        eigenvalues, eigenvectors = np.linalg.eigh(symmetric)
        clipped = np.clip(eigenvalues, NUMERICAL_EPSILON, None)
        psd_matrix = eigenvectors @ np.diag(clipped) @ eigenvectors.T
        return (psd_matrix + psd_matrix.T) / 2.0


def optimize_web3_portfolio(
    tokens: Sequence[str],
    total_capital: float = 10000.0,
    current_weights: Optional[Dict[str, float]] = None,
) -> str:
    """对外暴露的主函数。"""
    optimizer = Web3PortfolioOptimizer()
    return optimizer.optimize(tokens=tokens, total_capital=total_capital, current_weights=current_weights)


def build_argument_parser() -> argparse.ArgumentParser:
    """命令行参数解析器。"""
    parser = argparse.ArgumentParser(description="Web3 Portfolio Optimizer")
    parser.add_argument(
        "--tokens",
        type=str,
        default="BTC,ETH,SOL,PEPE,ARB",
        help="代币列表，支持逗号或空格分隔，例如 BTC,ETH,SOL",
    )
    parser.add_argument(
        "--capital",
        type=float,
        default=10000.0,
        help="总资金，默认 10000 USDT",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="",
        help="原始持仓比例，可选，例如 BTC=40,ETH=30,USDT=30 或 我现在 40% BTC、30% ETH、30% USDT；若不传则默认风险资产等权",
    )
    return parser


def main() -> None:
    """CLI 入口。"""
    parser = build_argument_parser()
    args = parser.parse_args()
    tokens = parse_token_text(args.tokens)
    current_weights = parse_weights_text(args.weights) if args.weights else None
    report = optimize_web3_portfolio(
        tokens=tokens,
        total_capital=args.capital,
        current_weights=current_weights,
    )
    print(report)


if __name__ == "__main__":
    main()
