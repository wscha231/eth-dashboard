"""Donation tracker — scans 4 wallet chains for incoming transfers.

Reads donation wallet addresses + Etherscan API key from .env, then queries:
  - Blockstream (Bitcoin)
  - Etherscan   (Ethereum native + ERC-20 USDT / USDC / DAI)
  - Solscan     (Solana native + SPL USDT / USDC)
  - Tronscan    (Tron native + TRC-20 USDT / USDC)

Converts each incoming transfer to USD:
  - Stablecoins (USDT/USDC/DAI) = 1:1
  - Natives (BTC/ETH/SOL/TRX)   = historical CoinGecko price at tx timestamp
    (cached locally in forecast_site/private/price_cache.json to minimize
    free-tier API calls).

Writes two files:
  - forecast_site/public/donations.json       PUBLIC — only milestones_reached
                                              + next_milestone + last_updated.
                                              NO dollar amounts exposed.
  - forecast_site/private/donation_ledger.json  PRIVATE (gitignored) — full
                                              per-tx ledger with USD values
                                              for your records / tax purposes.

Milestones (kept server-side; only reached/next surfaced publicly):
  I    = $5,000
  II   = $25,000
  III  = $100,000

Designed to be called once a day from daily_pipeline.ps1 as a fail-silent step.

Run:
    python donation_tracker.py --verbose
    python donation_tracker.py --quiet
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.parse import urlencode
from urllib.request import Request, urlopen


# ---------------------------------------------------------------
# Config
# ---------------------------------------------------------------

MILESTONES = {
    "I": 5_000,
    "II": 25_000,
    "III": 100_000,
}

# ERC-20 contract addresses (Ethereum mainnet)
ERC20_TOKENS = {
    "USDT": {"contract": "0xdac17f958d2ee523a2206206994597c13d831ec7", "decimals": 6},
    "USDC": {"contract": "0xa0b86991c6218b36c1d19d4a2e9eb0ce3606eb48", "decimals": 6},
    "DAI":  {"contract": "0x6b175474e89094c44da98b954eedeac495271d0f", "decimals": 18},
}

# TRC-20 contract addresses (Tron mainnet) -- address matching is also case-sensitive
TRC20_TOKENS = {
    "USDT": {"contract": "TR7NHqjeKQxGTCi8q8ZY4pL8otSzgjLj6t", "decimals": 6},
    "USDC": {"contract": "TEkxiTehnzSmSe2XqrBj4w32RUN966rdz8", "decimals": 6},
}

# CoinGecko IDs for historical native price lookups
CG_IDS = {"BTC": "bitcoin", "ETH": "ethereum", "SOL": "solana", "TRX": "tron"}

DEFAULT_PRIVATE_LEDGER = Path("forecast_site/private/donation_ledger.json")
DEFAULT_PUBLIC_DONATIONS = Path("forecast_site/public/donations.json")
DEFAULT_PRICE_CACHE = Path("forecast_site/private/price_cache.json")

USER_AGENT = "etherforecast-donation-tracker/1.0"


def _public_wallets(addr_btc: str, addr_eth: str, addr_sol: str, addr_trx: str) -> list[dict]:
    """Return public donation destinations without exposing balances."""
    candidates = [
        ("Bitcoin", "BTC", addr_btc, "https://blockstream.info/address/{}"),
        ("Ethereum", "ETH / ERC-20", addr_eth, "https://etherscan.io/address/{}"),
        ("Solana", "SOL / SPL", addr_sol, "https://solscan.io/account/{}"),
        ("Tron", "TRX / TRC-20", addr_trx, "https://tronscan.org/#/address/{}"),
    ]
    wallets = []
    for chain, asset, address, explorer in candidates:
        if not address:
            continue
        wallets.append({
            "chain": chain,
            "asset": asset,
            "address": address,
            "explorer_url": explorer.format(address),
        })
    return wallets


# ---------------------------------------------------------------
# .env autoload (zero-dependency; same pattern as eth_data_collector)
# ---------------------------------------------------------------

def _autoload_dotenv() -> None:
    candidates = [Path(__file__).resolve().parent / ".env", Path.cwd() / ".env"]
    for env_path in candidates:
        if not env_path.exists():
            continue
        try:
            for raw_line in env_path.read_text(encoding="utf-8").splitlines():
                line = raw_line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                key = key.strip()
                value = value.strip().strip('"').strip("'")
                if key and key not in os.environ:
                    os.environ[key] = value
        except OSError:
            continue
        break


# ---------------------------------------------------------------
# HTTP helper
# ---------------------------------------------------------------

def http_get_json(url: str, timeout: int = 20):
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=timeout) as resp:
        return json.loads(resp.read().decode("utf-8"))


# ---------------------------------------------------------------
# Bitcoin (Blockstream) -- no API key needed
# ---------------------------------------------------------------

def fetch_bitcoin_txs(address: str) -> list[dict]:
    """Return incoming confirmed txs for a BTC address.

    Blockstream returns last 25 mempool + last 25 confirmed by default. For
    sites with many donations you would paginate via /txs/chain/:last_seen_txid
    -- we stop after 500 for safety; adjust if the donation load grows.
    """
    out = []
    seen: set[str] = set()
    url = f"https://blockstream.info/api/address/{address}/txs"
    for page in range(20):  # hard cap: 500 confirmed txs
        try:
            txs = http_get_json(url)
        except HTTPError as exc:
            if exc.code == 404:
                return out
            raise
        if not txs:
            break
        for tx in txs:
            txid = tx.get("txid")
            if not txid or txid in seen:
                continue
            seen.add(txid)
            ts = tx.get("status", {}).get("block_time")
            if ts is None:
                continue  # unconfirmed
            # Sum all outputs addressed to `address` in this tx
            inc_sats = 0
            for vout in tx.get("vout", []):
                if vout.get("scriptpubkey_address") == address:
                    inc_sats += int(vout.get("value", 0) or 0)
            if inc_sats <= 0:
                continue
            out.append({
                "txid": txid,
                "timestamp": int(ts),
                "amount_btc": inc_sats / 1e8,
            })
        if len(txs) < 25:
            break
        last_txid = txs[-1].get("txid")
        if not last_txid:
            break
        url = f"https://blockstream.info/api/address/{address}/txs/chain/{last_txid}"
        time.sleep(0.3)  # be polite
    return out


# ---------------------------------------------------------------
# Ethereum (Etherscan) -- native + ERC-20
# ---------------------------------------------------------------

def fetch_ethereum_native_txs(address: str, api_key: str) -> list[dict]:
    # Etherscan v2 unified API (chainid=1 = Ethereum mainnet). The legacy v1
    # endpoint was deprecated in 2024 and now returns "NOTOK" for all requests.
    params = {
        "chainid": 1,
        "module": "account", "action": "txlist", "address": address,
        "startblock": 0, "endblock": 99999999, "sort": "asc",
        "apikey": api_key,
    }
    url = f"https://api.etherscan.io/v2/api?{urlencode(params)}"
    data = http_get_json(url)
    if data.get("status") != "1":
        if "No transactions found" in str(data.get("message", "")):
            return []
        raise RuntimeError(f"Etherscan txlist error: {data.get('message')}")
    out = []
    addr_lower = address.lower()
    for tx in data.get("result", []):
        if tx.get("to", "").lower() != addr_lower:
            continue
        if tx.get("isError") == "1":
            continue
        wei = int(tx.get("value", "0") or 0)
        if wei == 0:
            continue
        out.append({
            "txhash": tx.get("hash"),
            "timestamp": int(tx.get("timeStamp", 0)),
            "amount_eth": wei / 1e18,
        })
    return out


def fetch_ethereum_token_txs(address: str, api_key: str) -> list[dict]:
    # Etherscan v2 unified API (chainid=1 = Ethereum mainnet).
    params = {
        "chainid": 1,
        "module": "account", "action": "tokentx", "address": address,
        "startblock": 0, "endblock": 99999999, "sort": "asc",
        "apikey": api_key,
    }
    url = f"https://api.etherscan.io/v2/api?{urlencode(params)}"
    data = http_get_json(url)
    if data.get("status") != "1":
        if "No transactions found" in str(data.get("message", "")):
            return []
        raise RuntimeError(f"Etherscan tokentx error: {data.get('message')}")
    out = []
    addr_lower = address.lower()
    contract_to_symbol = {v["contract"].lower(): k for k, v in ERC20_TOKENS.items()}
    for tx in data.get("result", []):
        if tx.get("to", "").lower() != addr_lower:
            continue
        contract = tx.get("contractAddress", "").lower()
        symbol = contract_to_symbol.get(contract)
        if symbol is None:
            continue  # non-stablecoin ERC-20; skip
        decimals = ERC20_TOKENS[symbol]["decimals"]
        raw_value = int(tx.get("value", "0") or 0)
        amount = raw_value / (10 ** decimals)
        if amount <= 0:
            continue
        out.append({
            "txhash": tx.get("hash"),
            "timestamp": int(tx.get("timeStamp", 0)),
            "symbol": symbol,
            "amount": amount,
        })
    return out


# ---------------------------------------------------------------
# Solana (Solscan public API)
# ---------------------------------------------------------------

def fetch_solana_txs(address: str) -> list[dict]:
    """Solana native + SPL token incoming transfers via Solscan public API."""
    out = []
    # --- Native SOL ---
    try:
        url = f"https://public-api.solscan.io/account/solTransfers?account={address}&limit=100&offset=0"
        data = http_get_json(url)
        for tx in (data.get("data") or []):
            if tx.get("dst") != address:
                continue
            lamports = int(tx.get("lamport", 0) or 0)
            if lamports <= 0:
                continue
            out.append({
                "txhash": tx.get("txHash"),
                "timestamp": int(tx.get("blockTime") or 0),
                "symbol": "SOL",
                "amount": lamports / 1e9,
            })
    except Exception:
        pass  # Solscan public API flaky; skip on fail
    # --- SPL tokens (USDT/USDC) ---
    try:
        url = f"https://public-api.solscan.io/account/splTransfers?account={address}&limit=100&offset=0"
        data = http_get_json(url)
        for tx in (data.get("data") or []):
            if tx.get("dst") != address:
                continue
            symbol = str(tx.get("symbol", "")).upper()
            if symbol not in {"USDT", "USDC"}:
                continue
            decimals = int(tx.get("decimals") or 6)
            raw = int(tx.get("changeAmount") or 0)
            amount = raw / (10 ** decimals)
            if amount <= 0:
                continue
            out.append({
                "txhash": tx.get("txHash"),
                "timestamp": int(tx.get("blockTime") or 0),
                "symbol": symbol,
                "amount": amount,
            })
    except Exception:
        pass
    return out


# ---------------------------------------------------------------
# Tron (Tronscan public API)
# ---------------------------------------------------------------

def fetch_tron_txs(address: str) -> list[dict]:
    out = []
    # --- Native TRX ---
    try:
        url = f"https://apilist.tronscan.org/api/transaction?address={address}&limit=50&start=0"
        data = http_get_json(url)
        for tx in (data.get("data") or []):
            contract_data = tx.get("contractData") or {}
            if contract_data.get("to_address") != address:
                continue
            sun = int(contract_data.get("amount", 0) or 0)  # 1 TRX = 1e6 SUN
            if sun <= 0:
                continue
            ts_ms = int(tx.get("timestamp", 0) or 0)
            out.append({
                "txhash": tx.get("hash"),
                "timestamp": ts_ms // 1000,
                "symbol": "TRX",
                "amount": sun / 1e6,
            })
    except Exception:
        pass
    # --- TRC-20 (USDT/USDC) ---
    try:
        url = f"https://apilist.tronscan.org/api/token_trc20/transfers?relatedAddress={address}&limit=50&start=0"
        data = http_get_json(url)
        for tx in (data.get("token_transfers") or []):
            if tx.get("to_address") != address:
                continue
            token_info = tx.get("tokenInfo") or {}
            symbol = str(token_info.get("tokenAbbr", "")).upper()
            if symbol not in TRC20_TOKENS:
                continue
            decimals = int(token_info.get("tokenDecimal") or 6)
            raw = int(tx.get("quant", "0") or 0)
            amount = raw / (10 ** decimals)
            if amount <= 0:
                continue
            ts_ms = int(tx.get("block_ts", 0) or 0)
            out.append({
                "txhash": tx.get("transaction_id"),
                "timestamp": ts_ms // 1000,
                "symbol": symbol,
                "amount": amount,
            })
    except Exception:
        pass
    return out


# ---------------------------------------------------------------
# Historical USD price (CoinGecko free tier + local cache)
# ---------------------------------------------------------------

def get_historical_usd_price(symbol: str, unix_ts: int, cache: dict) -> float | None:
    """Cache key is symbol+YYYY-MM-DD so the same day only hits CG once per native asset."""
    if unix_ts <= 0:
        return None
    date_ddmmyyyy = datetime.fromtimestamp(unix_ts, tz=timezone.utc).strftime("%d-%m-%Y")
    key = f"{symbol}_{date_ddmmyyyy}"
    if key in cache:
        return cache[key]
    cg_id = CG_IDS.get(symbol)
    if not cg_id:
        return None
    url = f"https://api.coingecko.com/api/v3/coins/{cg_id}/history?date={date_ddmmyyyy}&localization=false"
    try:
        data = http_get_json(url)
        price = data.get("market_data", {}).get("current_price", {}).get("usd")
        time.sleep(1.3)  # CG free tier is ~30 req/min; stay conservative
        if price is not None:
            cache[key] = float(price)
            return float(price)
    except Exception:
        return None
    return None


def tx_to_usd(tx: dict, symbol: str, cache: dict) -> float:
    # Stablecoins: already USD
    if symbol in {"USDT", "USDC", "DAI"}:
        return float(tx.get("amount", 0.0))
    # Natives: convert at tx timestamp
    if symbol == "BTC":
        amount = float(tx.get("amount_btc", 0.0))
    elif symbol == "ETH":
        amount = float(tx.get("amount_eth", 0.0))
    else:
        amount = float(tx.get("amount", 0.0))
    if amount == 0.0:
        return 0.0
    price = get_historical_usd_price(symbol, int(tx.get("timestamp", 0)), cache)
    return amount * price if price else 0.0


# ---------------------------------------------------------------
# Main
# ---------------------------------------------------------------

def _append_chain(ledger: dict, chain: str, txs: list[dict], default_symbol: str,
                  cache: dict, verbose: bool) -> None:
    for tx in txs:
        symbol = tx.get("symbol", default_symbol)
        usd = tx_to_usd(tx, symbol, cache)
        row = {**tx, "symbol": symbol, "usd": round(usd, 2)}
        ledger["by_chain"][chain]["txs"].append(row)
        ledger["by_chain"][chain]["usd_total"] += usd
        ledger["by_chain"][chain]["tx_count"] += 1
    if verbose:
        c = ledger["by_chain"][chain]
        print(f"  {chain:<9s}  {c['tx_count']:>4d} txs  ${c['usd_total']:>12,.2f}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--public-out", default=str(DEFAULT_PUBLIC_DONATIONS),
                        help="Public milestone JSON (no $ amounts).")
    parser.add_argument("--private-out", default=str(DEFAULT_PRIVATE_LEDGER),
                        help="Full per-tx ledger (kept local, gitignored).")
    parser.add_argument("--price-cache", default=str(DEFAULT_PRICE_CACHE))
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    verbose = not args.quiet

    _autoload_dotenv()

    addr_btc = os.environ.get("DONATION_BTC_ADDRESS", "").strip()
    addr_eth = os.environ.get("DONATION_ETH_ADDRESS", "").strip()
    addr_sol = os.environ.get("DONATION_SOL_ADDRESS", "").strip()
    addr_trx = os.environ.get("DONATION_TRX_ADDRESS", "").strip()
    etherscan_key = os.environ.get("ETHERSCAN_API_KEY", "").strip()

    if verbose:
        print("=" * 60)
        print("Donation tracker")
        print("=" * 60)
        print(f"  BTC:  {addr_btc or '(unset)'}")
        print(f"  ETH:  {addr_eth or '(unset)'}")
        print(f"  SOL:  {addr_sol or '(unset)'}")
        print(f"  TRX:  {addr_trx or '(unset)'}")
        print()

    # Load price cache
    cache: dict = {}
    cache_path = Path(args.price_cache)
    if cache_path.exists():
        try:
            cache = json.loads(cache_path.read_text(encoding="utf-8"))
            if not isinstance(cache, dict):
                cache = {}
        except Exception:
            cache = {}

    ledger = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "by_chain": {
            "bitcoin":  {"address": addr_btc, "tx_count": 0, "usd_total": 0.0, "txs": []},
            "ethereum": {"address": addr_eth, "tx_count": 0, "usd_total": 0.0, "txs": []},
            "solana":   {"address": addr_sol, "tx_count": 0, "usd_total": 0.0, "txs": []},
            "tron":     {"address": addr_trx, "tx_count": 0, "usd_total": 0.0, "txs": []},
        },
        "usd_total": 0.0,
        "errors": [],
    }

    # --- BTC ---
    if addr_btc:
        try:
            _append_chain(ledger, "bitcoin", fetch_bitcoin_txs(addr_btc), "BTC", cache, verbose)
        except Exception as exc:
            ledger["errors"].append(f"bitcoin: {exc}")
            if verbose:
                traceback.print_exc()

    # --- ETH (native + ERC-20) ---
    if addr_eth and etherscan_key:
        try:
            native_txs = fetch_ethereum_native_txs(addr_eth, etherscan_key)
            token_txs = fetch_ethereum_token_txs(addr_eth, etherscan_key)
            _append_chain(ledger, "ethereum", native_txs, "ETH", cache, False)
            _append_chain(ledger, "ethereum", token_txs, "ETH", cache, False)
            if verbose:
                c = ledger["by_chain"]["ethereum"]
                print(f"  {'ethereum':<9s}  {c['tx_count']:>4d} txs  ${c['usd_total']:>12,.2f}  "
                      f"(native={len(native_txs)}, tokens={len(token_txs)})")
        except Exception as exc:
            ledger["errors"].append(f"ethereum: {exc}")
            if verbose:
                traceback.print_exc()
    elif addr_eth and not etherscan_key:
        ledger["errors"].append("ethereum: ETHERSCAN_API_KEY not set")

    # --- SOL ---
    if addr_sol:
        try:
            _append_chain(ledger, "solana", fetch_solana_txs(addr_sol), "SOL", cache, verbose)
        except Exception as exc:
            ledger["errors"].append(f"solana: {exc}")
            if verbose:
                traceback.print_exc()

    # --- TRON ---
    if addr_trx:
        try:
            _append_chain(ledger, "tron", fetch_tron_txs(addr_trx), "TRX", cache, verbose)
        except Exception as exc:
            ledger["errors"].append(f"tron: {exc}")
            if verbose:
                traceback.print_exc()

    # --- Totals ---
    for chain in ledger["by_chain"].values():
        chain["usd_total"] = round(chain["usd_total"], 2)
    ledger["usd_total"] = round(
        sum(c["usd_total"] for c in ledger["by_chain"].values()), 2
    )

    reached = [ml for ml, thresh in MILESTONES.items() if ledger["usd_total"] >= thresh]
    next_ml = next((ml for ml, t in MILESTONES.items() if ledger["usd_total"] < t), None)
    ledger["milestones_reached"] = reached
    ledger["next_milestone"] = next_ml

    # --- Write PRIVATE ledger ---
    priv_path = Path(args.private_out)
    priv_path.parent.mkdir(parents=True, exist_ok=True)
    priv_path.write_text(json.dumps(ledger, indent=2, default=str), encoding="utf-8")

    # --- Write PUBLIC donations (no amounts) ---
    public_payload = {
        "generated_at_utc": ledger["generated_at_utc"],
        "milestones_reached": reached,
        "next_milestone": next_ml,
        "wallets": _public_wallets(addr_btc, addr_eth, addr_sol, addr_trx),
    }
    pub_path = Path(args.public_out)
    pub_path.parent.mkdir(parents=True, exist_ok=True)
    pub_path.write_text(json.dumps(public_payload, indent=2), encoding="utf-8")

    # --- Save price cache ---
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        cache_path.write_text(json.dumps(cache, indent=2), encoding="utf-8")
    except Exception:
        pass

    if verbose:
        print()
        print(f"Total raised:       ${ledger['usd_total']:,.2f}")
        print(f"Milestones reached: {reached if reached else '(none)'}")
        print(f"Next milestone:     {next_ml}")
        print(f"Private ledger:     {priv_path}")
        print(f"Public (no $):      {pub_path}")
        if ledger["errors"]:
            print(f"Errors: {ledger['errors']}")

    # Non-zero exit only if we couldn't reach ANY chain AND wallets were configured.
    if ledger["errors"] and ledger["by_chain"] and all(
        c["tx_count"] == 0 and c["address"] for c in ledger["by_chain"].values()
    ):
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
