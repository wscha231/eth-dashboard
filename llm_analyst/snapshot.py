"""Build a compact, LLM-digestible numeric snapshot of ETH market state.

Goal
----
The analyst prompt needs a **pre-digested** view of the market, not a
142-column CSV dump. The LLM will drown in that. Instead we hand it ~30
hand-chosen metrics, each with (current value, 7d change, 30d change,
and regime flag where useful), plus short context lines like
"ETH is 12% below its 30-day high".

Date-gating for leak-safe backtests
-----------------------------------
Every snapshot is built relative to an ``as_of_date``. The master CSV is
filtered to rows with ``Date <= as_of_date`` before computation, so if
you ask for a 2024-08-01 snapshot, *nothing* after that date is visible.
This lets the backtest harness replay "what the analyst would have seen
on that day" cleanly.

Curation principle
------------------
Include things that matter for **30-day ETH direction** and are stable
signals (not noisy indicators that change hourly). Skip:
  - Pure-short-term technicals (intraday vol, 1d RSI) — LLM doesn't need
    to reason about minute-level state.
  - Features that overlap with each other (we keep 30d vol, skip 14d vol).
  - Anything derived purely from price (model's own predictions). We want
    *raw* market state, not our prior views.

Output format: single dict with nested structure so the JSON is readable
by humans too. The prompt-assembler formats it into Markdown bullets.
"""
from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

DEFAULT_MASTER_CSV = Path("lake/gold/eth_master_daily.csv")


def _safe_pct_change(current: float, prior: float) -> float | None:
    """Percent change as a fraction (0.05 = +5%). None if either side missing."""
    if pd.isna(current) or pd.isna(prior) or prior == 0:
        return None
    return float((current - prior) / prior)


def _safe_lookup(frame: pd.DataFrame, column: str, pos: int) -> float | None:
    """Return ``frame.iloc[pos][column]`` as a float, or None on absence/NaN."""
    if column not in frame.columns:
        return None
    if pos >= len(frame) or -pos > len(frame):
        return None
    value = frame.iloc[pos].get(column)
    if pd.isna(value):
        return None
    return float(value)


def _rolling_vol_annualised(close: pd.Series, window: int) -> float | None:
    """Annualised realised vol from daily log returns. None if too few rows."""
    if close.isna().all() or len(close.dropna()) < window + 1:
        return None
    log_ret = (close.astype(float).apply(lambda x: math.log(x) if x > 0 else math.nan)
               .diff())
    window_ret = log_ret.tail(window)
    if window_ret.isna().any() or len(window_ret) < window:
        return None
    return float(window_ret.std(ddof=1) * math.sqrt(365))


def _drawdown(close: pd.Series, window: int) -> float | None:
    """Return the drawdown (negative fraction) from the rolling max over ``window``."""
    if close.empty:
        return None
    tail = close.tail(window).dropna()
    if tail.empty:
        return None
    last = tail.iloc[-1]
    peak = tail.max()
    if peak == 0 or pd.isna(peak):
        return None
    return float((last / peak) - 1.0)


def _price_block(frame: pd.DataFrame) -> dict[str, Any]:
    """ETH, BTC, and ETH/BTC ratio block — core price state."""
    eth_close_series = frame["eth_close"].astype(float)
    btc_close_series = frame["btc_close"].astype(float)

    eth_now = eth_close_series.iloc[-1] if not eth_close_series.empty else None
    btc_now = btc_close_series.iloc[-1] if not btc_close_series.empty else None

    eth_vs_30d_high = _drawdown(eth_close_series, 30)
    eth_vs_90d_high = _drawdown(eth_close_series, 90)

    return {
        "eth_close": None if eth_now is None or pd.isna(eth_now) else float(eth_now),
        "btc_close": None if btc_now is None or pd.isna(btc_now) else float(btc_now),
        "eth_return_1d":  _safe_pct_change(eth_close_series.iloc[-1],  eth_close_series.iloc[-2])  if len(eth_close_series) >= 2 else None,
        "eth_return_7d":  _safe_pct_change(eth_close_series.iloc[-1],  eth_close_series.iloc[-8])  if len(eth_close_series) >= 8 else None,
        "eth_return_30d": _safe_pct_change(eth_close_series.iloc[-1],  eth_close_series.iloc[-31]) if len(eth_close_series) >= 31 else None,
        "eth_return_90d": _safe_pct_change(eth_close_series.iloc[-1],  eth_close_series.iloc[-91]) if len(eth_close_series) >= 91 else None,
        "btc_return_30d": _safe_pct_change(btc_close_series.iloc[-1],  btc_close_series.iloc[-31]) if len(btc_close_series) >= 31 else None,
        "eth_btc_ratio_change_30d": (
            None
            if eth_now is None or btc_now is None
               or len(eth_close_series) < 31 or len(btc_close_series) < 31
               or pd.isna(btc_close_series.iloc[-31]) or btc_close_series.iloc[-31] == 0
            else float(
                (eth_close_series.iloc[-1]  / btc_close_series.iloc[-1])
              / (eth_close_series.iloc[-31] / btc_close_series.iloc[-31])
              - 1.0
            )
        ),
        "eth_drawdown_from_30d_high": eth_vs_30d_high,
        "eth_drawdown_from_90d_high": eth_vs_90d_high,
    }


def _volatility_block(frame: pd.DataFrame) -> dict[str, Any]:
    eth_close = frame["eth_close"].astype(float)
    vol_30 = _rolling_vol_annualised(eth_close, 30)
    vol_90 = _rolling_vol_annualised(eth_close, 90)
    return {
        "eth_realised_vol_30d_annualised": vol_30,
        "eth_realised_vol_90d_annualised": vol_90,
        "vol_regime_30d_over_90d": (
            float(vol_30 / vol_90) if vol_30 is not None and vol_90 and vol_90 > 0 else None
        ),
        "deribit_option_iv_mean_now": _safe_lookup(frame, "deribit_eth_option_mark_iv_mean", -1),
        "deribit_option_iv_mean_30d_ago": _safe_lookup(frame, "deribit_eth_option_mark_iv_mean", -31) if len(frame) >= 31 else None,
    }


def _derivatives_block(frame: pd.DataFrame) -> dict[str, Any]:
    funding_8h = _safe_lookup(frame, "deribit_eth_funding_interest_8h", -1)
    # Annualise the 8h funding: (1 + r)^(365*3) − 1. Cap at ±100% for display sanity.
    funding_annualised = None
    if funding_8h is not None:
        try:
            funding_annualised = (1.0 + funding_8h) ** (365 * 3) - 1.0
            funding_annualised = max(min(funding_annualised, 1.0), -1.0)
        except (OverflowError, ValueError):
            funding_annualised = None

    pc_oi = _safe_lookup(frame, "deribit_eth_option_put_call_oi_ratio", -1)
    pc_oi_30d_ago = _safe_lookup(frame, "deribit_eth_option_put_call_oi_ratio", -31) if len(frame) >= 31 else None

    return {
        "deribit_funding_8h": funding_8h,
        "deribit_funding_approx_annualised": funding_annualised,
        "deribit_put_call_oi_ratio": pc_oi,
        "deribit_put_call_oi_ratio_30d_ago": pc_oi_30d_ago,
        "binance_funding_rate_mean": _safe_lookup(frame, "binance_eth_funding_rate_mean", -1),
        "binance_basis_annualised": _safe_lookup(frame, "binance_eth_basis_annualizedbasisrate", -1),
    }


def _macro_block(frame: pd.DataFrame) -> dict[str, Any]:
    vix_now = _safe_lookup(frame, "vix_close", -1)
    vix_30d_ago = _safe_lookup(frame, "vix_close", -31) if len(frame) >= 31 else None

    return {
        "vix_now": vix_now,
        "vix_30d_ago": vix_30d_ago,
        "dxy_now": _safe_lookup(frame, "dxy_close", -1),
        "dxy_30d_ago": _safe_lookup(frame, "dxy_close", -31) if len(frame) >= 31 else None,
        "treasury_10y_now":   _safe_lookup(frame, "fred_treasury_10y", -1),
        "treasury_10y_30d_ago": _safe_lookup(frame, "fred_treasury_10y", -31) if len(frame) >= 31 else None,
        "treasury_2y_now":    _safe_lookup(frame, "fred_treasury_2y", -1),
        "yield_curve_10y_2y": _safe_lookup(frame, "fred_yield_curve_10y_2y", -1),
        "hy_oas_now":         _safe_lookup(frame, "fred_hy_oas", -1),
        "hy_oas_30d_ago":     _safe_lookup(frame, "fred_hy_oas", -31) if len(frame) >= 31 else None,
        "spy_return_30d": _safe_pct_change(
            _safe_lookup(frame, "spy_close", -1) or float("nan"),
            _safe_lookup(frame, "spy_close", -31) or float("nan"),
        ) if len(frame) >= 31 else None,
    }


def _onchain_block(frame: pd.DataFrame) -> dict[str, Any]:
    """Only include on-chain metrics that are actually populated."""
    block: dict[str, Any] = {}

    def _add_if(key: str, column: str, label: str) -> None:
        current = _safe_lookup(frame, column, -1)
        if current is None:
            return
        prior = _safe_lookup(frame, column, -31) if len(frame) >= 31 else None
        block[f"{label}_now"] = current
        block[f"{label}_30d_change_pct"] = _safe_pct_change(current, prior) if prior else None

    _add_if("glassnode_active_addresses", "glassnode_eth_active_addresses", "active_addresses")
    _add_if("glassnode_sopr", "glassnode_eth_sopr", "sopr")
    _add_if("glassnode_realized_price", "glassnode_eth_realized_price_usd", "realized_price_usd")
    _add_if("etherscan_dailytx", "etherscan_eth_dailytx", "daily_tx_count")
    _add_if("etherscan_gas", "etherscan_eth_dailyavggasprice", "avg_gas_price_gwei")
    _add_if("artemis_stablecoin_supply", "artemis_eth_stablecoin_supply", "stablecoin_supply_eth")
    _add_if("defillama_tvl", "defillama_ethereum_chain_tvl_usd", "eth_chain_tvl")
    _add_if("defillama_dex_vol", "defillama_ethereum_dex_volume_usd", "eth_dex_volume")

    return block


def _sentiment_block(frame: pd.DataFrame) -> dict[str, Any]:
    fg_now = _safe_lookup(frame, "sentiment_fear_greed", -1)
    if len(frame) >= 30:
        fg_30d_avg = frame["sentiment_fear_greed"].tail(30).astype(float).mean() if "sentiment_fear_greed" in frame.columns else None
        fg_30d_avg = None if pd.isna(fg_30d_avg) else float(fg_30d_avg)
    else:
        fg_30d_avg = None
    return {
        "fear_greed_now": fg_now,
        "fear_greed_30d_avg": fg_30d_avg,
    }


@dataclass(frozen=True)
class MarketSnapshot:
    as_of_date: str              # ISO date (YYYY-MM-DD)
    price: dict[str, Any]
    volatility: dict[str, Any]
    derivatives: dict[str, Any]
    macro: dict[str, Any]
    onchain: dict[str, Any]
    sentiment: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "as_of_date": self.as_of_date,
            "price": self.price,
            "volatility": self.volatility,
            "derivatives": self.derivatives,
            "macro": self.macro,
            "onchain": self.onchain,
            "sentiment": self.sentiment,
        }


def build_snapshot(
    as_of_date: str | pd.Timestamp,
    master_csv: Path | str = DEFAULT_MASTER_CSV,
) -> MarketSnapshot:
    """Build a date-gated market snapshot.

    ``as_of_date`` is inclusive; rows strictly after are dropped before any
    aggregate computation.
    """
    frame = pd.read_csv(master_csv)
    frame["Date"] = pd.to_datetime(frame["Date"]).dt.tz_localize(None)
    cutoff = pd.to_datetime(as_of_date).tz_localize(None)
    filtered = frame[frame["Date"] <= cutoff].copy()
    if filtered.empty:
        raise ValueError(f"No master rows at or before {cutoff.date().isoformat()}.")
    filtered = filtered.sort_values("Date").reset_index(drop=True)

    return MarketSnapshot(
        as_of_date=cutoff.date().isoformat(),
        price=_price_block(filtered),
        volatility=_volatility_block(filtered),
        derivatives=_derivatives_block(filtered),
        macro=_macro_block(filtered),
        onchain=_onchain_block(filtered),
        sentiment=_sentiment_block(filtered),
    )


def _format_pct(value: float | None, *, plus: bool = True) -> str:
    if value is None:
        return "n/a"
    sign = "+" if plus and value >= 0 else ""
    return f"{sign}{value*100:.2f}%"


def _format_usd(value: float | None) -> str:
    return "n/a" if value is None else f"${value:,.0f}"


def _format_number(value: float | None, digits: int = 2) -> str:
    return "n/a" if value is None else f"{value:.{digits}f}"


def snapshot_to_markdown(snapshot: MarketSnapshot) -> str:
    """Render the snapshot as Markdown bullets suitable for embedding in a prompt.

    This is the exact form the LLM sees. Keeping it terse and ASCII so
    different LLMs (Claude / GPT) produce consistent tokenisation.
    """
    p = snapshot.price
    v = snapshot.volatility
    d = snapshot.derivatives
    m = snapshot.macro
    o = snapshot.onchain
    s = snapshot.sentiment

    lines = [
        f"As of: {snapshot.as_of_date}",
        "",
        "**Price**",
        f"- ETH: {_format_usd(p.get('eth_close'))} "
        f"(1d {_format_pct(p.get('eth_return_1d'))}, "
        f"7d {_format_pct(p.get('eth_return_7d'))}, "
        f"30d {_format_pct(p.get('eth_return_30d'))}, "
        f"90d {_format_pct(p.get('eth_return_90d'))})",
        f"- BTC: {_format_usd(p.get('btc_close'))} "
        f"(30d {_format_pct(p.get('btc_return_30d'))})",
        f"- ETH/BTC ratio 30d change: {_format_pct(p.get('eth_btc_ratio_change_30d'))}",
        f"- ETH drawdown from 30d high: {_format_pct(p.get('eth_drawdown_from_30d_high'), plus=False)}",
        f"- ETH drawdown from 90d high: {_format_pct(p.get('eth_drawdown_from_90d_high'), plus=False)}",
        "",
        "**Volatility**",
        f"- ETH realised vol 30d (annualised): {_format_pct(v.get('eth_realised_vol_30d_annualised'), plus=False)}",
        f"- ETH realised vol 90d (annualised): {_format_pct(v.get('eth_realised_vol_90d_annualised'), plus=False)}",
        f"- Vol regime (30d/90d): {_format_number(v.get('vol_regime_30d_over_90d'))} (>1 = elevated)",
        f"- Deribit option IV mean: {_format_number(v.get('deribit_option_iv_mean_now'))} "
        f"(30d ago: {_format_number(v.get('deribit_option_iv_mean_30d_ago'))})",
        "",
        "**Derivatives**",
        f"- Deribit 8h funding: {_format_pct(d.get('deribit_funding_8h'))} "
        f"(annualised ≈ {_format_pct(d.get('deribit_funding_approx_annualised'))})",
        f"- Deribit put/call OI ratio: {_format_number(d.get('deribit_put_call_oi_ratio'))} "
        f"(30d ago: {_format_number(d.get('deribit_put_call_oi_ratio_30d_ago'))})",
        f"- Binance funding rate mean: {_format_pct(d.get('binance_funding_rate_mean'))}",
        f"- Binance basis (annualised): {_format_pct(d.get('binance_basis_annualised'))}",
        "",
        "**Macro**",
        f"- VIX: {_format_number(m.get('vix_now'))} (30d ago: {_format_number(m.get('vix_30d_ago'))})",
        f"- DXY: {_format_number(m.get('dxy_now'))} (30d ago: {_format_number(m.get('dxy_30d_ago'))})",
        f"- US 10y: {_format_number(m.get('treasury_10y_now'))}% "
        f"(30d ago: {_format_number(m.get('treasury_10y_30d_ago'))}%)",
        f"- 10y-2y curve: {_format_number(m.get('yield_curve_10y_2y'))}",
        f"- HY OAS: {_format_number(m.get('hy_oas_now'))} (30d ago: {_format_number(m.get('hy_oas_30d_ago'))})",
        f"- SPY 30d return: {_format_pct(m.get('spy_return_30d'))}",
        "",
    ]

    if o:
        lines.append("**On-chain**")
        if "active_addresses_now" in o:
            lines.append(f"- Active addresses: {_format_number(o['active_addresses_now'], 0)} "
                         f"(30d change {_format_pct(o.get('active_addresses_30d_change_pct'))})")
        if "sopr_now" in o:
            lines.append(f"- SOPR: {_format_number(o['sopr_now'], 3)} "
                         f"(30d change {_format_pct(o.get('sopr_30d_change_pct'))})")
        if "realized_price_usd_now" in o:
            lines.append(f"- Realised price: {_format_usd(o['realized_price_usd_now'])} "
                         f"(30d change {_format_pct(o.get('realized_price_usd_30d_change_pct'))})")
        if "daily_tx_count_now" in o:
            lines.append(f"- Daily tx count: {_format_number(o['daily_tx_count_now'], 0)} "
                         f"(30d change {_format_pct(o.get('daily_tx_count_30d_change_pct'))})")
        if "avg_gas_price_gwei_now" in o:
            lines.append(f"- Avg gas price (gwei): {_format_number(o['avg_gas_price_gwei_now'], 1)} "
                         f"(30d change {_format_pct(o.get('avg_gas_price_gwei_30d_change_pct'))})")
        if "stablecoin_supply_eth_now" in o:
            lines.append(f"- Stablecoin supply on ETH: {_format_number(o['stablecoin_supply_eth_now'], 0)} "
                         f"(30d change {_format_pct(o.get('stablecoin_supply_eth_30d_change_pct'))})")
        if "eth_chain_tvl_now" in o:
            lines.append(f"- ETH chain TVL: {_format_usd(o['eth_chain_tvl_now'])} "
                         f"(30d change {_format_pct(o.get('eth_chain_tvl_30d_change_pct'))})")
        if "eth_dex_volume_now" in o:
            lines.append(f"- ETH DEX volume: {_format_usd(o['eth_dex_volume_now'])} "
                         f"(30d change {_format_pct(o.get('eth_dex_volume_30d_change_pct'))})")
        lines.append("")

    lines.append("**Sentiment**")
    lines.append(f"- Fear & Greed: {_format_number(s.get('fear_greed_now'), 1)} "
                 f"(30d avg {_format_number(s.get('fear_greed_30d_avg'), 1)})")

    return "\n".join(lines)


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Build a date-gated market snapshot.")
    parser.add_argument("--as-of", required=True, help="Snapshot date YYYY-MM-DD (inclusive).")
    parser.add_argument("--csv", default=str(DEFAULT_MASTER_CSV), help="Master CSV path.")
    parser.add_argument("--format", choices=("json", "markdown"), default="markdown")
    args = parser.parse_args()

    snap = build_snapshot(args.as_of, master_csv=args.csv)
    if args.format == "json":
        print(json.dumps(snap.to_dict(), indent=2, default=str))
    else:
        print(snapshot_to_markdown(snap))


if __name__ == "__main__":
    _cli()
