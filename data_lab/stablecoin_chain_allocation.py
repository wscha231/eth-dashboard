"""Research-only stablecoin chain-allocation helpers.

No function in this module is wired to MODEL selection or public forecast output.
Historical aggregate inputs may be reconstructed research. Open USD observations
are eligible only from actual receipts captured after collector deployment.
"""
from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import math
from typing import Any

import numpy as np
import pandas as pd


OPEN_STANDARD_OUSD_NAME = "Open USD"
OPEN_STANDARD_OUSD_SYMBOL = "OUSD"
OPEN_STANDARD_OUSD_PEG_TYPE = "peggedUSD"
OPEN_STANDARD_OUSD_EXPECTED_CHAINS = ("Ethereum", "Base", "Solana", "Tempo")
OPEN_STANDARD_OUSD_CONTRACTS = {
    "Base": "0xB2000000000000000000002fEb517dFeC7415344",
    "Ethereum": "0x9f6F3991D525015a6F8CaF062C83b62fD3AC4436",
    "Solana": "ousd2mJsPEckLHcSCDxyKD7NDGARZcfLbDZkKiatYHB",
    "Tempo": "0x20c0000000000000000000006a37DA5C996874BE",
}
DEFILLAMA_STABLECOINS_URL = "https://stablecoins.llama.fi/stablecoins"


class OUSDIdentityError(ValueError):
    """Raised when the source cannot be matched unambiguously to Open Standard OUSD."""


def _finite_number(value: Any, *, field: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field} must be numeric, not boolean")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field} is not numeric") from exc
    if not math.isfinite(number) or number < 0:
        raise ValueError(f"{field} must be finite and non-negative")
    return number


def _circulating_amount(value: Any, *, field: str) -> float:
    """Extract the current pegged-USD circulating amount without using prior periods."""
    if isinstance(value, dict):
        for key in ("current", "circulating", OPEN_STANDARD_OUSD_PEG_TYPE):
            if key in value:
                return _circulating_amount(value[key], field=f"{field}.{key}")
        raise ValueError(f"{field} has no current circulating amount")
    return _finite_number(value, field=field)


def _utc_iso(value: datetime | str | pd.Timestamp | None) -> str:
    ts = pd.Timestamp(value if value is not None else datetime.now(timezone.utc))
    if ts.tzinfo is None:
        ts = ts.tz_localize("UTC")
    else:
        ts = ts.tz_convert("UTC")
    return ts.isoformat()


def _chain_supply_map(value: Any) -> dict[str, float]:
    """Normalize current DefiLlama list shape and retained legacy dict shape."""
    if isinstance(value, dict):
        items = list(value.items())
    elif isinstance(value, list):
        items = []
        for index, row in enumerate(value):
            if not isinstance(row, dict):
                raise ValueError(f"chainCirculating[{index}] must be an object")
            chain = row.get("chain")
            if not isinstance(chain, str) or not chain:
                raise ValueError(f"chainCirculating[{index}].chain is missing")
            if "circulating" not in row:
                raise ValueError(f"chainCirculating[{index}].circulating is missing")
            items.append((chain, row["circulating"]))
    else:
        raise ValueError("Open Standard OUSD chainCirculating is missing")

    result: dict[str, float] = {}
    for chain, raw in items:
        chain = str(chain)
        if chain in result:
            raise OUSDIdentityError(f"duplicate OUSD chain: {chain}")
        result[chain] = _circulating_amount(raw, field=f"chainCirculating.{chain}")
    return result


def parse_open_standard_ousd_snapshot(
    payload: Any,
    *,
    received_at: datetime | str | pd.Timestamp | None = None,
) -> dict[str, Any]:
    """Parse one DefiLlama stablecoin snapshot into a fail-closed OUSD receipt.

    Identity is exact-name + exact-symbol + USD peg + the four official launch
    chains. Unknown positive chain supply blocks the receipt so a later chain
    expansion cannot silently change the research definition.

    DefiLlama does not expose the observation/publication timestamp for this
    current snapshot. For PIT use, available_at is therefore set
    conservatively to the actual EtherForecast received_at.
    """
    assets = payload.get("peggedAssets") if isinstance(payload, dict) else payload
    if not isinstance(assets, list):
        raise ValueError("DefiLlama stablecoin payload has no asset list")

    matches = [
        asset
        for asset in assets
        if isinstance(asset, dict)
        and asset.get("name") == OPEN_STANDARD_OUSD_NAME
        and asset.get("symbol") == OPEN_STANDARD_OUSD_SYMBOL
        and asset.get("pegType") == OPEN_STANDARD_OUSD_PEG_TYPE
    ]
    if len(matches) != 1:
        raise OUSDIdentityError(f"expected exactly one Open Standard OUSD asset, found {len(matches)}")
    asset = matches[0]

    chain_map = _chain_supply_map(asset.get("chainCirculating"))

    missing = [chain for chain in OPEN_STANDARD_OUSD_EXPECTED_CHAINS if chain not in chain_map]
    if missing:
        raise OUSDIdentityError("missing official OUSD chains: " + ",".join(missing))

    positive_unknown = [
        chain
        for chain, amount in chain_map.items()
        if chain not in OPEN_STANDARD_OUSD_EXPECTED_CHAINS and amount > 0
    ]
    if positive_unknown:
        raise OUSDIdentityError("unreviewed positive OUSD chain supply: " + ",".join(sorted(positive_unknown)))

    supplies = {chain: chain_map[chain] for chain in OPEN_STANDARD_OUSD_EXPECTED_CHAINS}
    chain_total = float(sum(supplies.values()))
    if chain_total <= 0:
        raise ValueError("Open Standard OUSD supply must be positive")

    reported_total = _circulating_amount(asset.get("circulating"), field="circulating")
    tolerance = max(1.0, reported_total * 0.005)
    if abs(chain_total - reported_total) > tolerance:
        raise ValueError("Open Standard OUSD chain sum conflicts with reported total")

    timestamp = _utc_iso(received_at)
    ethbase = supplies["Ethereum"] + supplies["Base"]
    raw_hash = hashlib.sha256(
        json.dumps(asset, sort_keys=True, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    ).hexdigest()

    price = asset.get("price")
    price_usd = None if price is None else _finite_number(price, field="price")
    shares = {chain: supplies[chain] / chain_total for chain in supplies}
    return {
        "received_at": timestamp,
        "available_at": timestamp,
        "source_observation_at": None,
        "source_id": "defillama_open_standard_ousd_receipt",
        "defillama_asset_id": str(asset.get("id", "")),
        "name": OPEN_STANDARD_OUSD_NAME,
        "symbol": OPEN_STANDARD_OUSD_SYMBOL,
        "peg_type": OPEN_STANDARD_OUSD_PEG_TYPE,
        "ethereum_supply_ousd": supplies["Ethereum"],
        "base_supply_ousd": supplies["Base"],
        "solana_supply_ousd": supplies["Solana"],
        "tempo_supply_ousd": supplies["Tempo"],
        "total_supply_ousd": chain_total,
        "ethbase_supply_ousd": ethbase,
        "ethbase_share": ethbase / chain_total,
        "tempo_share": shares["Tempo"],
        "solana_share": shares["Solana"],
        "chain_hhi": sum(value * value for value in shares.values()),
        "source_price_usd": price_usd,
        "raw_asset_sha256": raw_hash,
    }


def _positive(series: pd.Series) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce").astype(float)
    return numeric.where(numeric > 0)


def build_stablecoin_chain_allocation_features(
    global_supply: pd.Series,
    ethereum_supply: pd.Series,
    base_supply: pd.Series,
) -> pd.DataFrame:
    """Build the frozen 24h Center research feature family.

    Missing or non-positive inputs remain missing. No forward fill, zero fill,
    interpolation, or use of future observations is performed here.
    """
    frame = pd.concat(
        {
            "global": _positive(global_supply),
            "ethereum": _positive(ethereum_supply),
            "base": _positive(base_supply),
        },
        axis=1,
    ).sort_index()

    ethbase = (frame["ethereum"] + frame["base"]).where(
        frame["ethereum"].notna() & frame["base"].notna()
    )
    rest = (frame["global"] - ethbase).where(frame["global"].notna() & ethbase.notna())
    rest = rest.where(rest > 0)
    share = (ethbase / frame["global"]).where(
        ethbase.notna() & frame["global"].notna() & (ethbase <= frame["global"])
    )

    out = pd.DataFrame(index=frame.index)
    out["sc_global_growth_1d"] = np.log(frame["global"]).diff(1)
    out["sc_ethbase_share_delta_1d"] = share.diff(1)
    out["sc_ethbase_share_delta_7d"] = share.diff(7)
    out["sc_ethbase_vs_rest_flow_1d"] = np.log(ethbase).diff(1) - np.log(rest).diff(1)
    return out.replace([np.inf, -np.inf], np.nan)
