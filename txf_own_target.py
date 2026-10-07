"""Frozen native TXF own-target probe: no fitted model or Python fill simulation.

Prepare performs metadata-only native resolution and freezes bytes. Pilot/export
produce native features; analysis applies the preregistered quote-markout contract.
All non-prepare stages must execute the frozen copy and verify exact receipts.
"""

from __future__ import annotations

import sys as _astra_sys
from pathlib import Path as _AstraPath

_astra_repo_root = next(parent.parent for parent in _AstraPath(__file__).resolve().parents if parent.name == "AstraResearch")
_astra_sys.path.insert(0, str(_astra_repo_root))

import argparse
import calendar
import csv
import hashlib
import json
import math
import platform
import re
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.csv as pacsv
import yaml
import zstandard

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT.parent))
sys.path.insert(0, str(HERE))

from txf_reporting import reporting_supplement

from AstraResearch.code_guard import source_files
from AstraResearch.engine_identity import identity
from AstraResearch.io import ContractError, file_hash, lock, read_yaml, write_yaml
from AstraResearch.native_contract import sample_summary, validate_captured_origins, validate_receipt
from AstraResearch.store import ExposureLedger

FEATURES = {"touch_imbalance": "BookStructure.0.imbalance_touch.0", "trade_flow_5s": "TradeFlowState.0.flow_quantity_imbalance0.0"}
MID = "OriginMidTicks"
KEYS = ["SampleTime", "SampleBookTime", "SampleBookSeq"]
BOOK_PRICES = [f"{side}Price{i}" for side in ["Bid", "Ask"] for i in range(1, 6)]
BOOK_SIZES = [f"{side}Vol{i}" for side in ["Bid", "Ask"] for i in range(1, 6)]
COLUMNS = [
    "Type",
    "Seq",
    "Timestamp",
    "Symbol",
    "ExchangeTime",
    "StatusMask",
    "Side",
    "Price",
    "TradeVolume",
    "TotalVolume",
    "Turnover",
    "BidDepth",
    "AskDepth",
    *BOOK_PRICES,
    *BOOK_SIZES,
]
FLOATS = ["Price", "Turnover", *BOOK_PRICES]
INTS = ["Seq", "Timestamp", "ExchangeTime", "Side", "TradeVolume", "TotalVolume", "BidDepth", "AskDepth", *BOOK_SIZES]
PROTOCOL = ROOT / "Experiments/e2e_20260926/txf_own_target.yaml"


def require(condition, reason):
    if not condition:
        raise ContractError(reason)


def local_us(day, hhmmss):
    return int(pd.Timestamp(f"{day} {hhmmss[:2]}:{hhmmss[2:4]}:{hhmmss[4:]}", tz="Asia/Taipei").value // 1000)


def grid(day, config):
    session = config["session"]
    return np.arange(local_us(day, session["decision_start"]), local_us(day, session["decision_until_inclusive"]) + 1, 1_000_000, dtype=np.int64)


def checked_protocol(path):
    config = read_yaml(path)
    require(config["data"]["protected_from"] == 20260828 and config["data"]["research_validation"][-1][-1] == 20260826, "Protected date boundary changed")
    require(config["session"]["decision_start"] == "085000" and config["session"]["decision_until_inclusive"] == "134000", "Fixed session changed")
    require(config["quote_markout"]["primary_receive_delay_ms"] == 50 and config["quote_markout"]["robustness_receive_delay_ms"] == 250, "Fixed delays changed")
    require(config["contract"]["tick_size_points"] == 1 and config["contract"]["point_value_twd"] == 200, "TXF coordinates changed")
    require(
        config["contract"]["quantity_contracts"] == 1 and config["costs"]["broker_commission_twd_per_leg"] == 50 and config["costs"]["tax_rate_notional_per_leg"] == 0.00002,
        "Fixed economic assumptions changed",
    )
    require(config["costs"]["fee_stress"] == "gross_cash - 1.2 * (commission_both_legs + tax_both_legs)", "Fixed fee stress changed")
    return config


def python_environment():
    return {"python": sys.version, "platform": platform.platform(), "packages": {m.__name__: m.__version__ for m in [np, pd, pa, yaml, zstandard]}}


def freeze_native_sources(config, output):
    manifest = json.loads(Path(config["runtime"]["source_manifest"]).read_text())
    require(
        manifest["binary_sha256"] == config["native"]["proposed_binary_sha256"] and file_hash(Path(config["runtime"]["source_patch"])) == manifest["source_patch_sha256"],
        "Native binary/source manifest mismatch",
    )
    sources = [
        "src/oms/api/taifex_symbol_resolve.cpp",
        "src/oms/api/time_api.h",
        "src/oms/api/time_api.cpp",
        "src/oms/model/quote.cpp",
        "src/oms/model/quote.h",
        "src/oms/model/contract.h",
        "src/oms/modules/tw/taifex_info/taifex_info.cpp",
        "src/oms/modules/tw/taifex_filter/taifex_filter.cpp",
        "src/oms/modules/md/trade_book_md/trade_book_md.cpp",
        "src/oms/modules/feature/microstructure/book/current_book/current_book.cpp",
        "src/oms/modules/feature/microstructure/book/current_book/current_book.h",
        "src/oms/modules/feature/microstructure/book/book_structure/book_structure.cpp",
        "src/oms/modules/feature/microstructure/book/book_structure/book_structure.h",
        "src/oms/modules/feature/order_flow/trade_flow/trade_flow_state/trade_flow_state.cpp",
        "src/oms/modules/feature/order_flow/trade_flow/trade_flow_state/trade_flow_state.h",
        "src/oms/modules/writer/dataset_writer/dataset_writer.cpp",
        "src/oms/modules/writer/dataset_writer/value_parquet_sink.cpp",
        "src/oms/modules/writer/md_writer/md_writer.cpp",
        "src/oms/labelers/return_labeler/return_labeler.cpp",
        "src/oms/training_label_state.hpp",
        "src/oms/training_label_scaffold.hpp",
        "src/msg/md_msg.h",
        "src/sdk/math/compare.hpp",
        "src/sdk/trading/price_cmp.hpp",
        "src/sdk/trading/tick_calculator.hpp",
        "src/sdk/coco_type.h",
        "src/marketdata/parsers/taifex_parser.cpp",
    ]
    result = {}
    for relative in sources:
        source = ROOT.parent / relative
        expected = manifest["source_files"].get(relative)
        if expected is None:
            original = subprocess.run(["git", "show", manifest["base_revision"] + ":" + relative], cwd=ROOT.parent, check=True, capture_output=True).stdout
            expected = hashlib.sha256(original).hexdigest()
        require(file_hash(source) == expected, "Current study source differs from frozen binary lineage: " + relative)
        target = output / "native-source" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)
        result[relative] = expected
    write_yaml(output / "native-study-sources.yaml", {"base_revision": manifest["base_revision"], "files": result})


def decl(name, spec):
    return {"Desc": name, "Spec": spec}


def metadata_config(config, work):
    return {
        "Modules": [
            {
                "Gid": "",
                "Decl": [
                    decl("TaifexInfo.0", {"Product": "TXF", "TickSize": 1, "PointValue": 200, "BasicInfo": config["data"]["contract_metadata"]}),
                    decl("MarketInfoWriter.0", {"OutputPath": str(work / "contracts.yaml"), "Contracts": ["TXF@1"]}),
                ],
            }
        ]
    }


def native_config(config, row, work, inputs):
    symbol = row["contract"]
    dep = {"Book": ["TaifexFilter.0@TXF"], "Trade": ["TaifexFilter.0@TXF"]}
    continuous = config["native"]["feature_status_filter"].split(" for ")[0]
    features = [decl("TaifexFilter.0", {"Subscribe": [{"Book": [symbol], "Trade": [symbol]}]})]
    features += [decl("CurrentBook.0", {"Dep": dep, "StatusFilter": continuous}), decl("BookStructure.0", {"Dep": {"Book": dep["Book"]}, "StatusFilter": continuous})]
    features.append(decl("TradeFlowState.0", {"Dep": {"Trade": dep["Trade"]}, "Window": {"Time": ["5s"]}, "MinKnownSideShare": 0.8, "StatusFilter": continuous}))
    features.append(
        decl(
            "DatasetWriter.0",
            {
                "Subscribe": [{"Book": [symbol]}],
                "Exports": [v + "@TXF" for v in FEATURES.values()],
                "MetadataExports": [{"Feature": "CurrentBook.0.book_mid_price.0@TXF", "Name": MID}],
                "Labelers": [
                    {
                        "Type": "ForwardReturnRateLabeler",
                        "Spec": {
                            "Dep": {"Book": ["CurrentBook.0@TXF"]},
                            "Labels": [{"Y": "CurrentBook.0.book_mid_price.0@TXF", "Horizon": f"{h}s", "Name": "mid_return_bps", "Scale": 10000} for h in [10, 60]],
                        },
                    }
                ],
                "PeriodicSampler": {"StartTime": config["session"]["decision_start"], "UntilTime": config["session"]["decision_until_inclusive"], "SampleInterval": "1s"},
                "OutputPath": str(work / "grid/values.parquet"),
                "Format": "parquet",
                "UseTmp": True,
                "EmitSampleContext": True,
            },
        )
    )
    return {
        "Modules": [
            {
                "Gid": "",
                "Decl": [
                    decl("TaifexInfo.0", {"Product": "TXF", "TickSize": 1, "PointValue": 200, "BasicInfo": config["data"]["contract_metadata"]}),
                    decl("TradeBookMd.0", {"Dirs": [str(inputs)]}),
                    decl("MdWriter.0", {"Filter": "^" + symbol + "$", "OutputDir": str(work / "tape"), "Source": "txf-probe", "FallbackExchange": "TAIFEX"}),
                ],
            },
            {"Gid": "TXF", "Decl": features},
        ]
    }


def run_native(config, day, work, native_path, timeout=900):
    work.mkdir(parents=True, exist_ok=False)
    command = [
        config["runtime"]["binary"],
        "-d",
        day,
        "-C",
        str(work),
        "--quiet",
        "--run-status-dir",
        str(work / "status"),
        "--log-dir",
        str(work / "logs"),
        "--trading-calendar",
        config["data"]["calendar"],
        str(native_path),
    ]
    started = time.monotonic()
    with (work / "process.log").open("w") as stream:
        result = subprocess.run(command, stdout=stream, stderr=subprocess.STDOUT, timeout=timeout, check=False, env={**__import__("os").environ, "TZ": "Asia/Taipei"})
    require(result.returncode == 0, f"Native failed, preserve {work}")
    validate_receipt(read_yaml(work / "status" / f"{day}.yaml"), day)
    return {"command": command, "elapsed_seconds": time.monotonic() - started}


def expected_contract(day, listed, trading_days):
    year, month = int(day[:4]), int(day[4:6])
    third = [w[2] for w in calendar.monthcalendar(year, month) if w[2]][2]
    expiry = f"{year:04d}{month:02d}{third:02d}"
    early = max(d for d in trading_days if d < expiry)
    ranked = []
    for symbol in listed:
        if re.fullmatch(r"TXF[A-L][0-9]", symbol):
            delivery_year = year // 10 * 10 + int(symbol[-1])
            delivery_year += 10 if delivery_year < year - 2 else -10 if delivery_year > year + 8 else 0
            delivery = delivery_year * 100 + ord(symbol[-2]) - 64
            require(delivery >= year * 100 + month, f"Stale listed contract {day}/{symbol}")
            if day in {early, expiry} and delivery == year * 100 + month:
                continue
            require(not (day > expiry and delivery == year * 100 + month), f"Expired listed contract {day}/{symbol}")
            ranked.append((delivery, symbol))
    require(bool(ranked), f"No ordinary monthly contracts: {day}")
    return min(ranked)[1]


def prepare(protocol, output):
    config = checked_protocol(protocol)
    require(not output.exists(), "Immutable output already exists; preserve and register a new revision")
    require(str(output) == config["runtime"]["output"], "Output differs from protocol")
    require(file_hash(Path(config["runtime"]["binary"])) == config["native"]["proposed_binary_sha256"], "Frozen binary changed")
    require(file_hash(Path(config["data"]["calendar"])) == config["data"]["calendar_sha256"], "Calendar changed")
    inventory = json.loads(Path(config["runtime"]["inventory"]).read_text())
    days = [r["day"] for r in inventory["rows"]]
    calendar_doc = read_yaml(Path(config["data"]["calendar"]))
    trading = sorted(set(map(str, calendar_doc["trading_days"])) - set(map(str, calendar_doc.get("unscheduled_closures", []))))
    require(days == [d for d in trading if "20260601" <= d <= "20260826"] and len(days) == 61, "Planned calendar population changed")
    output.mkdir(parents=True)
    write_yaml(output / "protocol.yaml", config)
    # Claim even failed attempts before opening any raw recording bytes for hashes.
    ExposureLedger(ROOT / "runs/.cache/dataset").claim("TAIFEX:TXF", days, "development", config["hypothesis_family"])
    write_yaml(output / "exposure.yaml", {"scope": "TAIFEX:TXF", "days": days, "role": "development", "owner": config["hypothesis_family"]})
    for source in source_files(ROOT):
        target = output / "source/AstraResearch" / source.relative_to(ROOT)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
    for source in [Path(__file__), HERE / "txf_reporting.py", ROOT / "engine_identity.py", ROOT / "tests/test_txf_own_target.py", ROOT / "tests/test_txf_reporting.py"]:
        shutil.copy2(source, output / "source" / source.name)
    for key in ["source_manifest", "source_patch", "inventory"]:
        shutil.copy2(config["runtime"][key], output / Path(config["runtime"][key]).name)
    freeze_native_sources(config, output)
    oracle = Path(config["runtime"]["independent_oracle_amendment"])
    require(file_hash(oracle / "amendment.json") == config["runtime"]["independent_oracle_amendment_sha256"], "Independent oracle amendment changed")
    amendment = json.loads((oracle / "amendment.json").read_text())
    for name, key in [("txf_feature_oracle.py", "revised_helper_sha256"), ("test_txf_feature_oracle.py", "revised_tests_sha256")]:
        require(file_hash(oracle / name) == amendment[key], "Independent oracle source changed: " + name)
        shutil.copy2(oracle / name, output / "source" / name)
    shutil.copy2(oracle / "amendment.json", output / "source/amendment.json")
    write_yaml(
        output / "environment.yaml",
        {
            **python_environment(),
            "native": identity(Path(config["runtime"]["binary"])),
        },
    )
    external = {config["data"]["calendar"]: config["data"]["calendar_sha256"]}
    rows = []
    for day in days:
        meta_path = Path(config["data"]["contract_metadata"]) / (day + ".csv")
        external[str(meta_path)] = file_hash(meta_path)
        with meta_path.open(encoding="utf-8-sig", newline="") as stream:
            listed = {r["symbol"]: r for r in csv.DictReader(stream)}
        expected = expected_contract(day, listed, trading)
        work = output / "metadata" / day
        conf = output / "configs/metadata" / (day + ".yaml")
        write_yaml(conf, metadata_config(config, work))
        run_native(config, day, work, conf, timeout=60)
        native = yaml.load((work / "contracts.yaml").read_text(), Loader=yaml.BaseLoader)["contracts"]["TXF@1"]
        require(
            native["available"] == "true" and native["symbol"] == expected and float(native["trading_unit"]) == 200 and float(native["tick_size"]) == 1,
            f"Native metadata mismatch {day}",
        )
        record = listed[expected]
        require(record["exchange"] == "TAIFEX" and record["state"] == "n" and record["unit"] == "1" and record["day_trade"] == "Yes", f"Metadata contract ineligible {day}")
        raw = Path(config["runtime"]["raw_root"]) / expected / (day + ".csv.zst")
        raw_sha = file_hash(raw) if raw.is_file() else None
        external[str(raw)] = raw_sha
        row = {
            "day": day,
            "contract": expected,
            "raw_path": str(raw),
            "raw_sha256": raw_sha,
            "native_contract": native,
            "status": "available" if raw_sha else "recording_unavailable",
        }
        rows.append(row)
        write_yaml(output / "configs/native" / (day + ".yaml"), native_config(config, row, output / "native" / day, output / "inputs" / day))
        if len(rows) % 10 == 0 or len(rows) == len(days):
            print(f"metadata {len(rows)}/{len(days)}", flush=True)
    require([r["day"] for r in rows if not r["raw_sha256"]] == ["20260602", "20260729"], "Metadata-only missing-file inventory changed")
    write_yaml(output / "inventory.yaml", rows)
    files = {str(p.relative_to(output)): file_hash(p) for p in output.rglob("*") if p.is_file()}
    write_yaml(output / "freeze.yaml", {"files": files, "external": external, "stage": "before_this_revision_market_replay_and_economic_analysis"})
    print(json.dumps({"prepared": str(output), "freeze_sha256": file_hash(output / "freeze.yaml"), "days": len(rows), "available": 59}), flush=True)


def verify_freeze(output):
    require(Path(__file__).resolve() == (output / "source/txf_own_target.py").resolve(), "Execute the frozen source copy")
    for name in ["AstraResearch.io", "AstraResearch.store", "AstraResearch.native_contract", "engine_identity", "txf_reporting"]:
        require(Path(sys.modules[name].__file__).resolve().is_relative_to(output / "source"), "Unfrozen Python dependency: " + name)
    frozen = read_yaml(output / "freeze.yaml")
    for name, expected in frozen["files"].items():
        require(file_hash(output / name) == expected, "Frozen file changed: " + name)
    for name, expected in frozen["external"].items():
        path = Path(name)
        require((file_hash(path) if path.is_file() else None) == expected, "External source changed: " + name)
    config = checked_protocol(output / "protocol.yaml")
    environment = read_yaml(output / "environment.yaml")
    require(all(environment[key] == value for key, value in python_environment().items()), "Python environment changed")
    require(identity(Path(config["runtime"]["binary"])) == environment["native"], "Native runtime changed")
    return config


def read_tape(path, symbol, *, native=False):
    types = {c: pa.float64() for c in FLOATS} | {c: pa.int64() for c in INTS} | {c: pa.string() for c in ["Type", "Symbol", "StatusMask"]}
    with path.open("rb") as stream, zstandard.ZstdDecompressor().stream_reader(stream) as decoded:
        table = pacsv.read_csv(
            decoded, read_options=pacsv.ReadOptions(use_threads=False), convert_options=pacsv.ConvertOptions(include_columns=COLUMNS, column_types=types, null_values=[])
        )
    frame = table.to_pandas()
    require(not frame.empty and frame.Type.isin(["B", "T"]).all() and frame.Symbol.eq(symbol).all(), "Invalid raw type/symbol/empty tape")
    require(frame.StatusMask.str.fullmatch(r"[0-9a-fA-F]+").all(), "Invalid raw status field")
    status = frame.StatusMask.map(lambda v: int(v, 16)).to_numpy(np.int64)
    require(((status & ~7) == 0).all(), "Unknown status bits")
    frame["status"] = status
    for name in ["TradeVolume", "TotalVolume", *BOOK_SIZES]:
        require(frame[name].between(-(2**31), 2**31 - 1).all(), "Native Qty int32 overflow: " + name)
    require(frame.Side.between(0, 255).all(), "Native Side uint8 overflow")
    both_touches = frame.Type.eq("B").to_numpy(copy=True)
    for side in ["Bid", "Ask"]:
        both_touches &= (frame[side + "Depth"].to_numpy() > 0) & (frame[side + "Price1"].to_numpy() > 0) & (frame[side + "Vol1"].to_numpy() > 0)
    for side in ["Bid", "Ask"]:
        depth = frame[side + "Depth"].to_numpy()
        require(((depth >= 0) & (depth <= 5)).all(), "Book depth outside native capacity: " + side)
        # BookStructure evaluates to_tick(int32) on each active level of a side
        # whose touch is valid. Protect that conversion, not the economic population.
        safe_side = frame.Type.eq("B").to_numpy() & (frame.status.to_numpy() == 0)
        safe_side &= (depth > 0) & (frame[side + "Price1"].to_numpy() > 0) & (frame[side + "Vol1"].to_numpy() > 0)
        min_ticks = np.full(len(frame), np.inf)
        max_ticks = np.full(len(frame), -np.inf)
        for level in range(1, 6):
            prices = frame[side + f"Price{level}"].to_numpy(float)
            active = (safe_side & (depth >= level)) | (both_touches if level == 1 else False)
            require((~active | (np.isfinite(prices) & (prices + 0.5 >= -(2**31)) & (prices + 0.5 <= 2**31 - 1))).all(), "Unsafe native tick conversion: " + side + str(level))
            level_active = safe_side & (depth >= level)
            ticks = np.trunc(prices[level_active] + 0.5)
            min_ticks[level_active] = np.minimum(min_ticks[level_active], ticks)
            max_ticks[level_active] = np.maximum(max_ticks[level_active], ticks)
        # BookStructure subtracts Tick(int32) values before converting to double.
        # Check the range in float64, where every possible int32 difference is exact.
        require((max_ticks[safe_side] - min_ticks[safe_side] <= 2**31 - 1).all(), "Unsafe native tick subtraction: " + side)
    original_time = frame.Timestamp.to_numpy(np.int64)
    require((original_time > 0).all(), "Invalid raw receive time")
    frame["raw_timestamp"] = original_time
    frame.Timestamp = np.maximum.accumulate(original_time)
    if native:
        require(np.array_equal(frame.Timestamp, original_time), "Native publication times not monotonic")
    # Preserve rejected raw status records for the stricter economic path audit.
    # Native-feature and publication oracles use only the independently marked published subset.
    valid_trade = np.isfinite(frame.Price) & (frame.Price > 0) & (frame.TradeVolume > 0)
    frame["published"] = frame.Type.eq("B") | valid_trade
    frame["reject_reason"] = np.where(frame.published, "published", np.where(~np.isfinite(frame.Price) | (frame.Price <= 0), "invalid_trade_price", "nonpositive_trade_quantity"))
    if native:
        require(frame.published.all(), "Native tape contains a filtered trade")
    frame.reset_index(drop=True, inplace=True)
    return frame


def publication_parity(raw, native):
    ignored = int((~raw.published).sum())
    rejects = raw.loc[~raw.published].groupby(["reject_reason", "status"], dropna=False).size().reset_index(name="count").to_dict("records")
    raw_count = len(raw)
    raw = raw.loc[raw.published].reset_index(drop=True)
    require(len(raw) == len(native), "Raw/native publication count differs")
    for kind in ["B", "T"]:
        a, b = raw.loc[raw.Type.eq(kind)], native.loc[native.Type.eq(kind)]
        require(len(a) == len(b) and np.array_equal(b.Seq, np.arange(1, len(b) + 1)), "Native per-feed sequence mismatch")
        cols = ["Timestamp", "ExchangeTime", "status"] + (
            ["BidDepth", "AskDepth", *BOOK_PRICES, *BOOK_SIZES] if kind == "B" else ["Side", "Price", "TradeVolume", "TotalVolume", "Turnover"]
        )
        for col in cols:
            require(np.array_equal(a[col].to_numpy(), b[col].to_numpy(), equal_nan=True), f"Raw/native {kind}/{col} differs")
    require(np.array_equal(raw.Type, native.Type) and np.array_equal(raw.Timestamp, native.Timestamp), "Raw/native publication order differs")
    return {
        "passed": True,
        "records": len(raw),
        "books": int(raw.Type.eq("B").sum()),
        "noncontinuous": int((raw.status != 0).sum()),
        "invalid_books": int((raw.Type.eq("B") & ~valid_book(raw)).sum()),
        "timestamp_clamps": int((raw.raw_timestamp != raw.Timestamp).sum()),
        "raw_records": raw_count,
        "filtered_trades": ignored,
        "filtered_trade_reasons": rejects,
    }


def valid_book(frame):
    return (
        (frame.status == 0)
        & (frame.BidDepth >= 1)
        & (frame.AskDepth >= 1)
        & np.isfinite(frame.BidPrice1)
        & np.isfinite(frame.AskPrice1)
        & (frame.BidPrice1 > 0)
        & (frame.AskPrice1 > frame.BidPrice1)
        & (frame.BidVol1 >= 1)
        & (frame.AskVol1 >= 1)
    )


def quote_state(frame):
    books = frame.loc[frame.Type.eq("B")].copy()
    good = valid_book(books).to_numpy(bool)
    bad = (frame.status != 0) | (frame.Type.eq("B") & ~valid_book(frame))
    return {
        "time": books.Timestamp.to_numpy(np.int64),
        "bid": books.BidPrice1.to_numpy(float),
        "ask": books.AskPrice1.to_numpy(float),
        "good": good,
        "position": np.flatnonzero(frame.Type.eq("B")),
        "bad_position": np.flatnonzero(bad),
        "event_time": frame.Timestamp.to_numpy(np.int64),
        "bad_time": frame.loc[bad, "Timestamp"].to_numpy(np.int64),
        "coverage_end": int(books.loc[good, "Timestamp"].iloc[-1]) if good.any() else -1,
    }


def snapshot(tape, boundaries, age_us=1_000_000, *, tie_unknown=True):
    boundaries = np.asarray(boundaries, dtype=np.int64)
    index = np.searchsorted(tape["time"], boundaries, side="left") - 1
    safe = np.clip(index, 0, max(0, len(tape["time"]) - 1))
    reason = np.full(len(boundaries), "resolved", dtype=object)
    if not len(tape["time"]):
        return np.full(len(boundaries), np.nan), np.full(len(boundaries), np.nan), np.full(len(boundaries), "no_prior_book", dtype=object)
    times = tape["time"][safe]
    last_bad = np.searchsorted(tape["bad_time"], boundaries, side="left") - 1
    bad_position = np.full(len(boundaries), -1, dtype=np.int64)
    has_bad = last_bad >= 0
    bad_position[has_bad] = tape["bad_position"][last_bad[has_bad]]
    reason[~tape["good"][safe] | (bad_position >= tape["position"][safe])] = "invalid_or_unreseeded_book"
    reason[boundaries - times > age_us] = "stale_book"
    reason[index < 0] = "no_prior_book"
    if tie_unknown:
        ties = np.searchsorted(tape["event_time"], boundaries, side="right") != np.searchsorted(tape["event_time"], boundaries, side="left")
        reason[ties] = "endpoint_tie_unknown"
    return tape["bid"][safe], tape["ask"][safe], reason


def markouts(tape, origins, sides, horizon, delay_ms):
    origins = np.asarray(origins, dtype=np.int64)
    sides = np.asarray(sides, dtype=np.int64)
    enter, leave = origins + delay_ms * 1000, origins + (horizon * 1000 + delay_ms) * 1000
    eb, ea, er = snapshot(tape, enter)
    xb, xa, xr = snapshot(tape, leave)
    reason = np.where(er != "resolved", "entry_" + er, np.where(xr != "resolved", "exit_" + xr, "resolved"))
    bad_count = np.searchsorted(tape["bad_time"], leave, side="right") - np.searchsorted(tape["bad_time"], origins, side="left")
    reason[(reason == "resolved") & (bad_count > 0)] = "path_invalidated"
    reason[tape["coverage_end"] <= leave] = "future_coverage_unknown"
    known = reason == "resolved"
    result = pd.DataFrame(
        {"SampleTime": origins, "side": sides, "horizon": horizon, "delay_ms": delay_ms, "status": reason, "entry_bid": eb, "entry_ask": ea, "exit_bid": xb, "exit_ask": xa}
    )
    for name, direction in [("actual", sides), ("long", np.ones(len(sides))), ("short", -np.ones(len(sides)))]:
        entry = np.where(direction > 0, ea, eb)
        exit_price = np.where(direction > 0, xb, xa)
        gross = 200 * direction * (exit_price - entry)
        tax = 0.00002 * 200 * (entry + exit_price)
        fee = 100 + tax
        mid_gross = 200 * direction * ((xb + xa - eb - ea) / 2)
        for field, value in [
            ("gross", gross),
            ("tax", tax),
            ("fees", fee),
            ("net", gross - fee),
            ("stress", gross - 1.2 * fee),
            ("mid_gross", mid_gross),
            ("spread_cost", mid_gross - gross),
        ]:
            result[f"{name}_{field}"] = np.where(known, value, np.nan)
    result["expected50_net"] = (result.long_net + result.short_net) / 2
    result["directional_increment"] = result.actual_net - result.expected50_net
    return result


def causal_schedule(times, scores, available, threshold, horizon):
    statuses = np.full(len(times), "unavailable", dtype=object)
    sides = np.zeros(len(times), dtype=np.int8)
    next_time = -1
    for i, (stamp, score, known) in enumerate(zip(times, scores, available, strict=True)):
        if not known or not math.isfinite(score):
            continue
        if score == 0 or abs(score) < threshold:
            statuses[i] = "neutral"
        elif stamp < next_time:
            statuses[i] = "cadence_reserved"
        else:
            statuses[i] = "accepted"
            sides[i] = 1 if score > 0 else -1
            next_time = int(stamp) + (horizon + 1) * 1_000_000
    return statuses, sides


def side_corroboration(frame):
    """Observable quote-rule equivalent; never overwrite the archived native input."""
    frame = frame.loc[frame.published].reset_index(drop=True)
    is_book = frame.Type.eq("B").to_numpy()
    positions = np.arange(len(frame))
    prior = np.maximum.accumulate(np.where(is_book, positions, -1))
    prior = np.r_[-1, prior[:-1]]
    trades = np.flatnonzero(~is_book)
    chosen = prior[trades]
    safe = np.maximum(chosen, 0)
    signable = (chosen >= 0) & (frame.status.to_numpy()[safe] == 0) & (frame.status.to_numpy()[trades] == 0)
    price = frame.Price.to_numpy()[trades]
    bid, ask = frame.BidPrice1.to_numpy()[safe], frame.AskPrice1.to_numpy()[safe]
    bid_depth, ask_depth = frame.BidDepth.to_numpy()[safe], frame.AskDepth.to_numpy()[safe]
    # Native parser tests SELL first and uses price_better_or_equal's 1e-8 epsilon.
    expected = np.where(signable & (bid_depth > 0) & (bid - price >= -1e-8), 2, np.where(signable & (ask_depth > 0) & (ask - price <= 1e-8), 1, 0))
    actual = frame.Side.to_numpy()[trades]
    require(np.isin(actual, [0, 1, 2]).all(), "Unknown trade side code")
    mismatches = actual != expected
    return {
        "trades": len(trades),
        "known_sides": int((actual != 0).sum()),
        "mismatches": int(mismatches.sum()),
        "passed": not mismatches.any(),
        "interpretation": "Exact observed causal quote-rule equivalence only; private parser stale state and original producer revision unrecovered",
        "examples": [
            {"row": int(trades[i]), "time": int(frame.Timestamp.iloc[trades[i]]), "stored_side": int(actual[i]), "causal_side": int(expected[i])}
            for i in np.flatnonzero(mismatches)[:10]
        ],
    }


def native_mid_updates(frame):
    """Independent CurrentBook continuous-filter/contract-halt state oracle."""
    frame = frame.loc[frame.published].reset_index(drop=True)
    status = frame.status.to_numpy(np.int64)
    halted = (status & 5) != 0  # Contract HALT covers trial/suspend; auction alone does not emit halt.
    halt_transition = halted & ~np.r_[True, halted[:-1]]  # TradeBookMd starts contract HALT.
    book = frame.Type.eq("B").to_numpy() & (status == 0)
    update = book | halt_transition
    bid_valid = (frame.BidDepth.to_numpy() > 0) & (frame.BidPrice1.to_numpy() > 0) & (frame.BidVol1.to_numpy() > 0)
    ask_valid = (frame.AskDepth.to_numpy() > 0) & (frame.AskPrice1.to_numpy() > 0) & (frame.AskVol1.to_numpy() > 0)
    bid, ask = frame.BidPrice1.to_numpy(float), frame.AskPrice1.to_numpy(float)
    mid = np.where(bid_valid & ask_valid, (bid + ask) / 2, np.where(bid_valid, bid, np.where(ask_valid, ask, np.nan)))
    mid[~book] = -np.inf
    return frame.Timestamp.to_numpy(np.int64)[update], mid[update]


def asof_values(times, values, query, inclusive, *, initial=np.nan):
    index = np.searchsorted(times, query, side="right" if inclusive else "left") - 1
    result = np.full(len(query), initial)
    good = index >= 0
    result[good] = values[index[good]]
    return result


def native_capture_grid(raw, day, config):
    """Periodic timers run only through the last event published to the native clock."""
    planned = grid(day, config)
    published = raw.loc[raw.published, "Timestamp"]
    return planned[planned <= int(published.iloc[-1])] if len(published) else planned[:0]


def read_native_frame(work, raw, day, config):
    """An absent file is valid only for a proved completed, empty capture prefix."""
    frame_path = work / "grid/values.parquet"
    summary = sample_summary(work / "grid/sample_summary.yaml")
    expected = native_capture_grid(raw, day, config)
    if frame_path.is_file():
        validate_captured_origins(frame_path)
        frame = pd.read_parquet(frame_path)
    else:
        validate_receipt(read_yaml(work / "status" / f"{day}.yaml"), day)
        require(
            not len(expected) and all(summary[name] == 0 for name in ["sampled_rows", "emitted_rows", "pending_rows", "emitted_unresolved_rows", "unresolved_label_cells"]),
            "Missing native Parquet without completed zero-capture prefix",
        )
        columns = [*KEYS, *FEATURES.values(), MID, "mid_return_bps[10s]", "mid_return_bps[60s]"]
        frame = pd.DataFrame({name: pd.Series(dtype="int64" if name in KEYS else "float64") for name in columns})
    require(np.array_equal(frame.SampleTime, expected), "Native captured grid differs from exact published-clock prefix")
    return frame, summary


def verify_native_frame(frame, raw, day, config):
    expected_cols = {*KEYS, *FEATURES.values(), MID, "mid_return_bps[10s]", "mid_return_bps[60s]"}
    require(set(frame) == expected_cols and not frame[KEYS].duplicated().any(), "Unexpected native schema/keys")
    planned_grid = grid(day, config)
    expected_grid = native_capture_grid(raw, day, config)
    require(np.array_equal(frame.SampleTime, expected_grid), "Native captured grid differs from exact published-clock prefix")
    books = raw.loc[raw.Type.eq("B")]
    book_times = books.Timestamp.to_numpy(np.int64)
    index = np.searchsorted(book_times, expected_grid, side="left") - 1
    expected_time = np.zeros(len(expected_grid), dtype=np.int64)
    has_book = index >= 0
    expected_time[has_book] = book_times[index[has_book]]
    require(np.array_equal(frame.SampleBookTime, expected_time) and np.array_equal(frame.SampleBookSeq, index + 1), "Native capture book identity differs")
    updates, values = native_mid_updates(raw)
    y0 = asof_values(updates, values, expected_grid, False, initial=-np.inf)
    require(np.allclose(frame[MID], y0, rtol=0, atol=1e-10, equal_nan=True), "Native origin mid parity failed")
    result = {"planned_origins": len(planned_grid), "origin_rows": len(frame), "uncaptured_origins": len(planned_grid) - len(frame), "labels": {}}
    for horizon in [10, 60]:
        deadline = expected_grid + horizon * 1_000_000
        y1 = asof_values(updates, values, deadline, True)
        with np.errstate(invalid="ignore"):
            expected = (y1 - y0) / y0 * 10000
        expected[np.isnan(y0) | np.isnan(y1) | np.isneginf(y0) | np.isneginf(y1)] = np.nan
        expected[deadline >= updates[-1] if len(updates) else np.ones(len(deadline), dtype=bool)] = np.nan
        actual = frame[f"mid_return_bps[{horizon}s]"].to_numpy(float)
        mismatch = ~np.isclose(actual, expected, rtol=0, atol=1e-10, equal_nan=True)
        result["labels"][str(horizon)] = {"known": int(np.isfinite(actual).sum()), "mismatches": int(mismatch.sum())}
        require(not mismatch.any(), f"Native {horizon}s label endpoint parity failed")
    return result


def verify_receipt(work, expected_sources):
    receipt = read_yaml(work / "receipt.yaml")
    require(receipt["sources"] == expected_sources, "Resumed source identity changed")
    actual_files = {str(p.relative_to(work)) for p in work.rglob("*") if p.is_file() and p.name != "receipt.yaml"}
    require(actual_files == set(receipt["files"]), "Missing/foreign native output files")
    for name, expected in receipt["files"].items():
        require(file_hash(work / name) == expected, "Native output changed: " + name)
    return receipt


def native_tape_path(work, day, symbol):
    # Non-default Exchange::TAIFEX formats as lowercase; FallbackExchange is unused.
    return work / "tape/taifex/txf-probe/trade_book/csv" / symbol / (day + ".csv.zst")


def export_day(row, config, output):
    if not row["raw_sha256"]:
        return {"day": row["day"], "status": "recording_unavailable"}
    day, symbol = row["day"], row["contract"]
    work = output / "native" / day
    sources = {row["raw_path"]: row["raw_sha256"], "native_config": file_hash(output / "configs/native" / (day + ".yaml"))}
    raw_path = Path(row["raw_path"])
    require(file_hash(raw_path) == row["raw_sha256"], "Raw input changed")
    input_path = output / "inputs" / day / symbol / (day + ".csv.zst")
    input_path.parent.mkdir(parents=True, exist_ok=True)
    if not input_path.exists():
        input_path.symlink_to(raw_path)
    require(input_path.is_symlink() and input_path.resolve() == raw_path.resolve(), "Private CSV source changed")
    if (work / "receipt.yaml").is_file():
        return verify_receipt(work, sources)
    require(not work.exists(), "Incomplete native attempt preserved: " + str(work))
    # Validate stored values before native reader silently normalizes a malformed status.
    raw = read_tape(raw_path, symbol)
    command = run_native(config, day, work, output / "configs/native" / (day + ".yaml"))
    tape = read_tape(native_tape_path(work, day, symbol), symbol, native=True)
    parity = publication_parity(raw, tape)
    frame, summary = read_native_frame(work, raw, day, config)
    native_parity = verify_native_frame(frame, raw, day, config)
    sides = side_corroboration(raw)
    for name, report in [("publication-parity", parity), ("label-parity", native_parity), ("side-corroboration", sides)]:
        write_yaml(work / (name + ".yaml"), report)
    receipt = {
        "day": day,
        "status": "complete",
        "sources": sources,
        "native": command,
        "sample_summary": summary,
        "publication_parity": parity,
        "native_label_parity": native_parity,
        "side_corroboration": sides,
        "files": {str(p.relative_to(work)): file_hash(p) for p in work.rglob("*") if p.is_file()},
    }
    write_yaml(work / "receipt.yaml", receipt)
    return receipt


def export(output, pilot):
    config = verify_freeze(output)
    inventory = read_yaml(output / "inventory.yaml")
    if pilot:
        dates = set(map(str, config["execution_budget"]["pilot_dates"]))
        rows = [r for r in inventory if r["day"] in dates]
        require(len(rows) == 3, "Wrong structural pilot date set")
    else:
        pilot_report = read_yaml(output / "pilot.yaml")
        require(pilot_report["passed"] is True, "Technical pilot must pass before full export")
        independent = read_yaml(output / "independent-feature-oracle.yaml")
        require(independent["passed"] is True and independent["pilot_receipt_sha256"] == pilot_report["receipt_hashes"], "Independent native-feature oracle must match this pilot")
        rows = inventory
    receipts = []
    with lock(output / "export.lock", blocking=False), ThreadPoolExecutor(max_workers=config["execution_budget"]["workers_max"]) as pool:
        for row, receipt in zip(rows, pool.map(lambda r: export_day(r, config, output), rows), strict=True):
            receipts.append(receipt)
            print(f"{'pilot' if pilot else 'export'} {len(receipts)}/{len(rows)} {row['day']} {receipt['status']}", flush=True)
    report = {
        "passed": all(r["status"] in {"complete", "recording_unavailable"} for r in receipts),
        "days": [r["day"] for r in rows],
        "receipt_hashes": {r["day"]: file_hash(output / "native" / r["day"] / "receipt.yaml") for r in receipts if r["status"] == "complete"},
        "scope": "Technical coverage and parity only; no economic aggregation",
    }
    write_yaml(output / ("pilot.yaml" if pilot else "export.yaml"), report)


def verify_export(output, inventory):
    report = read_yaml(output / "export.yaml")
    require(report["passed"] and report["days"] == [r["day"] for r in inventory], "Incomplete/foreign export population")
    expected_days = {r["day"] for r in inventory if r["raw_sha256"]}
    require(set(report["receipt_hashes"]) == expected_days, "Wrong exported receipt set")
    require({p.name for p in (output / "native").iterdir()} == expected_days, "Foreign/partial native directories")
    for row in inventory:
        if row["raw_sha256"]:
            work = output / "native" / row["day"]
            link = output / "inputs" / row["day"] / row["contract"] / (row["day"] + ".csv.zst")
            require(link.is_symlink() and link.resolve() == Path(row["raw_path"]).resolve(), "Private CSV source changed")
            require(file_hash(work / "receipt.yaml") == report["receipt_hashes"][row["day"]], "Export receipt changed")
            verify_receipt(work, {row["raw_path"]: row["raw_sha256"], "native_config": file_hash(output / "configs/native" / (row["day"] + ".yaml"))})


def origin_panel(row, config, output):
    times = grid(row["day"], config)
    panel = pd.DataFrame({"day": row["day"], "contract": row["contract"], "SampleTime": times})
    seconds = (times // 1_000_000 + 8 * 3600) % 86400
    panel["session"] = np.where(seconds < 9 * 3600 + 10 * 60, "08:50-09:10", np.where(seconds < 11 * 3600, "09:10-11:00", "11:00-13:40"))
    if not row["raw_sha256"]:
        panel["origin_status"] = "recording_unavailable"
        panel["native_captured"] = False
        for name in KEYS[1:]:
            panel[name] = pd.Series(pd.NA, index=panel.index, dtype="Int64")
        for name in FEATURES:
            panel[name] = np.nan
        for horizon in [10, 60]:
            panel[f"native_label_{horizon}s"] = np.nan
        return panel, None
    raw = read_tape(Path(row["raw_path"]), row["contract"])
    frame, _ = read_native_frame(output / "native" / row["day"], raw, row["day"], config)
    tape = quote_state(raw)
    _, _, status = snapshot(tape, times, tie_unknown=False)
    panel["origin_status"] = status
    panel["native_captured"] = panel.SampleTime.isin(frame.SampleTime)
    renames = {column: name for name, column in FEATURES.items()}
    renames.update({f"mid_return_bps[{horizon}s]": f"native_label_{horizon}s" for horizon in [10, 60]})
    captured = frame[[*KEYS, *renames]].rename(columns=renames).astype({name: "Int64" for name in KEYS[1:]})
    panel = panel.merge(captured, on="SampleTime", how="left", validate="one_to_one", sort=False)
    panel.loc[~panel.native_captured, "origin_status"] = "native_capture_unavailable"
    return panel, tape


def calibrate(output):
    config = verify_freeze(output)
    inventory = read_yaml(output / "inventory.yaml")
    verify_export(output, inventory)
    require(not (output / "calibration.yaml").exists(), "Calibration already frozen")
    rows = [r for r in inventory if r["day"].startswith("202606")]
    panels = [origin_panel(r, config, output)[0] for r in rows]
    frame = pd.concat(panels, ignore_index=True)
    thresholds = {}
    for name in FEATURES:
        known = frame.origin_status.eq("resolved") & np.isfinite(frame[name])
        values = frame.loc[known, name].abs().to_numpy(float)
        value = float(np.quantile(values, 0.95, method="linear")) if len(values) else None
        thresholds[name] = {
            "q95": value if value and value > 0 else None,
            "finite_eligible_rows": int(known.sum()),
            "master_rows": len(frame),
            "absolute_threshold_ties": int((values == value).sum()) if value else 0,
        }
    path = output / "calibration-origins.parquet"
    frame.to_parquet(path, index=False)
    write_yaml(
        output / "calibration.yaml",
        {
            "thresholds": thresholds,
            "freeze_sha256": file_hash(output / "freeze.yaml"),
            "origins_sha256": file_hash(path),
            "native_export_sha256": file_hash(output / "export.yaml"),
            "uses": "June causal score population only; no labels or future quotes",
        },
    )
    print(json.dumps({"calibration_sha256": file_hash(output / "calibration.yaml"), "thresholds": thresholds}), flush=True)


def daily_stats(events, day_origin, day, arm, delay):
    known = events.status.eq("resolved")
    count, resolved = len(events), int(known.sum())
    fully_observed = day_origin.available.all()
    mean = float(events.loc[known, "actual_net"].mean()) if resolved else (0.0 if count == 0 and fully_observed else np.nan)
    row = {
        "day": day,
        "month": day[:6],
        "arm": arm,
        "delay_ms": delay,
        "planned_origins": len(day_origin),
        "available_origins": int(day_origin.available.sum()),
        "accepted": count,
        "resolved": resolved,
        "fully_origin_observed": bool(fully_observed),
        "observed_event_mean": mean,
        "unconditional_daily_known": bool(fully_observed and resolved == count),
    }
    for field in [
        "actual_net",
        "actual_gross",
        "actual_fees",
        "actual_tax",
        "actual_stress",
        "actual_mid_gross",
        "actual_spread_cost",
        "long_net",
        "short_net",
        "expected50_net",
        "directional_increment",
    ]:
        row[field] = float(events.loc[known, field].sum()) if resolved else (0.0 if count == 0 and fully_observed else np.nan)
    row["unknown_intent_tipping_net_cash"] = -row["actual_net"] / (count - resolved) if count > resolved and np.isfinite(row["actual_net"]) else np.nan
    return row


def block_bounds(days, replicates=10000, seed=20260927):
    """Fixed planned-date blocks retain unknown rows; all reported means are conditional."""
    days = days.sort_values("day").reset_index(drop=True)
    rng = np.random.default_rng(seed)
    event, equal_day = np.full(replicates, np.nan), np.full(replicates, np.nan)
    groups = [g.index.to_numpy() for _, g in days.groupby("month", sort=True)]
    require(all(len(g) >= 3 for g in groups), "Too few planned dates for block bounds")
    for i in range(replicates):
        selected = []
        for indexes in groups:
            starts = rng.integers(0, len(indexes) - 2, size=math.ceil(len(indexes) / 3))
            selected.extend(indexes[(starts[:, None] + np.arange(3)).ravel()[: len(indexes)]])
        sample = days.iloc[selected]
        count = int(sample.resolved.sum())
        if count:
            event[i] = sample.actual_net.sum(min_count=1) / count
        if sample.observed_event_mean.notna().any():
            equal_day[i] = sample.observed_event_mean.mean()
    result = {"planned_days": len(days), "known_day_means": int(days.observed_event_mean.notna().sum()), "replicates": replicates, "seed": seed}
    for name, values in [("event", event), ("equal_day", equal_day)]:
        finite = np.isfinite(values)
        result[name] = {
            "undefined_fraction": float((~finite).mean()),
            "lower97_5": float(np.quantile(values[finite], 0.025, method="linear")) if finite.any() else None,
            "lower95": float(np.quantile(values[finite], 0.05, method="linear")) if finite.any() else None,
        }
    result["supported"] = result["known_day_means"] >= 20 and all(result[n]["undefined_fraction"] <= 0.01 for n in ["event", "equal_day"])
    return result


def arm_summary(daily, events, lineage_ok):
    primary = daily.loc[(daily.delay_ms == 50) & daily.month.isin(["202607", "202608"])].copy()
    observed = events.loc[events.day.str[:6].isin(["202607", "202608"])]
    monthly = []
    for month, group in primary.groupby("month", sort=True):
        n, accepted = int(group.resolved.sum()), int(group.accepted.sum())
        monthly.append(
            {
                "month": month,
                "master_origins": int(group.planned_origins.sum()),
                "score_origin_availability": float(group.available_origins.sum() / group.planned_origins.sum()),
                "accepted": accepted,
                "resolved": n,
                "endpoint_support": n / accepted if accepted else 0,
                "event_mean_net": float(group.actual_net.sum(min_count=1) / n) if n else None,
                "equal_day_mean_net": float(group.observed_event_mean.mean()) if group.observed_event_mean.notna().any() else None,
                "directional_increment": float(group.directional_increment.sum(min_count=1) / n) if n else None,
            }
        )
    bounds = block_bounds(primary)
    net, stress = primary.actual_net.sum(min_count=1), primary.actual_stress.sum(min_count=1)
    positive_sum = primary.actual_net.clip(lower=0).sum()
    share = float(primary.actual_net.max() / positive_sum) if positive_sum > 0 else None
    keys = ["day", "SampleTime"]
    first = observed.loc[(observed.delay_ms == 50) & observed.status.eq("resolved")]
    second = observed.loc[(observed.delay_ms == 250) & observed.status.eq("resolved")]
    common = first[keys].merge(second, on=keys, how="inner", validate="one_to_one")
    delayed = {
        "common_count": len(common),
        "net": float(common.actual_net.sum()),
        "stress": float(common.actual_stress.sum()),
        "monthly_net": {month: float(common.loc[common.day.str.startswith(month), "actual_net"].sum()) for month in ["202607", "202608"]},
    }
    gates = {
        "side_lineage": bool(lineage_ok),
        "months_complete": len(monthly) == 2,
        "monthly_support": all(m["score_origin_availability"] >= 0.9 and m["endpoint_support"] >= 0.9 and m["resolved"] >= 100 for m in monthly),
        "monthly_net": all(m["event_mean_net"] is not None and m["event_mean_net"] > 0 and m["equal_day_mean_net"] is not None and m["equal_day_mean_net"] > 0 for m in monthly),
        "monthly_directional_increment": all(m["directional_increment"] is not None and m["directional_increment"] > 0 for m in monthly),
        "fee_stress": bool(stress > 0),
        "constant_side_comparison": bool(net > max(primary.long_net.sum(), primary.short_net.sum())),
        "bounds": bool(bounds["supported"] and all(bounds[n]["lower97_5"] is not None and bounds[n]["lower97_5"] > 0 for n in ["event", "equal_day"])),
        "leave_best_day": bool(net - primary.actual_net.max() > 0),
        "positive_day_concentration": bool(share is not None and share <= 0.3),
        "delayed": bool(delayed["net"] > 0 and delayed["stress"] > 0 and all(v > 0 for v in delayed["monthly_net"].values())),
    }
    return {
        "monthly": monthly,
        "conditional_pooled_net": float(net),
        "conditional_pooled_stress": float(stress),
        "maximum_positive_day_share": share,
        "leave_best_day_conditional_net": float(net - primary.actual_net.max()),
        "block_bounds": bounds,
        "delay250_common_known": delayed,
        "gates": gates,
        "advances": bool(all(gates.values())),
        "status": "evaluated" if gates["months_complete"] and gates["monthly_support"] and lineage_ok and bounds["supported"] else "inconclusive",
    }


def reporting_tables(analysis, origins, events):
    origins = origins.assign(month=origins.day.str[:6])
    events = events.assign(month=events.day.str[:6])
    origins.groupby(["arm", "month", "day", "session", "origin_status", "available", "decision"], dropna=False).size().rename("count").reset_index().to_csv(
        analysis / "all-origin-support-reasons.csv", index=False
    )
    fields = [
        "actual_mid_gross",
        "actual_gross",
        "actual_spread_cost",
        "actual_fees",
        "actual_tax",
        "actual_net",
        "actual_stress",
        "long_net",
        "short_net",
        "expected50_net",
        "directional_increment",
    ]
    dimensions = ["arm", "delay_ms", "month", "side", "session", "contract"]
    grouped = events.groupby(dimensions, dropna=False)
    counts = grouped.size().rename("all_accepted")
    known = grouped.actual_net.count().rename("resolved")
    means = grouped[fields].mean().add_suffix("_conditional_mean")
    sums = grouped[fields].sum(min_count=1).add_suffix("_conditional_sum")
    pd.concat([counts, known, means, sums], axis=1).reset_index().to_csv(analysis / "side-session-delivery-economics.csv", index=False)
    label_rows = []
    for (arm, month), group in origins.groupby(["arm", "month"], sort=True):
        horizon = 10 if arm.endswith("_10s") else 60
        label = f"native_label_{horizon}s"
        mask = group.available & np.isfinite(group[label])
        observed = group.loc[mask]
        enough = len(observed) > 1 and observed.score.nunique() > 1 and observed[label].nunique() > 1
        label_rows.append(
            {
                "arm": arm,
                "month": month,
                "master_origins": len(group),
                "available_score_origins": int(group.available.sum()),
                "own_label_metric_origins": len(observed),
                "pearson": float(observed.score.corr(observed[label])) if enough else np.nan,
                "spearman": float(observed.score.rank().corr(observed[label].rank())) if enough else np.nan,
                "mean_signed_native_return_bps": float((np.sign(observed.score) * observed[label]).mean()) if len(observed) else np.nan,
                "metric_population": "conditional on own-horizon known native label; never a decision/calibration mask",
            }
        )
    pd.DataFrame(label_rows).to_csv(analysis / "native-label-association.csv", index=False)


def analyze(output):
    config = verify_freeze(output)
    inventory = read_yaml(output / "inventory.yaml")
    verify_export(output, inventory)
    calibration = read_yaml(output / "calibration.yaml")
    require(
        calibration["freeze_sha256"] == file_hash(output / "freeze.yaml")
        and calibration["native_export_sha256"] == file_hash(output / "export.yaml")
        and calibration["origins_sha256"] == file_hash(output / "calibration-origins.parquet"),
        "Frozen calibration inputs changed",
    )
    analysis = output / "analysis"
    require(not analysis.exists(), "Preserve existing/partial analysis; no silent overwrite")
    analysis.mkdir()
    all_events, all_days, all_origins = [], [], []
    side_receipts = [read_yaml(output / "native" / r["day"] / "side-corroboration.yaml") for r in inventory if r["raw_sha256"]]
    flow_lineage_ok = all(r["passed"] for r in side_receipts)
    for row in inventory:
        panel, tape = origin_panel(row, config, output)
        for feature in FEATURES:
            values = panel[feature].to_numpy(float)
            available = panel.origin_status.eq("resolved").to_numpy() & np.isfinite(values)
            threshold = calibration["thresholds"][feature]["q95"]
            if threshold is None:
                available[:] = False
            for horizon in [10, 60]:
                arm = feature + f"_{horizon}s"
                statuses, sides = causal_schedule(panel.SampleTime.to_numpy(np.int64), values, available, threshold or math.inf, horizon)
                origins = panel[["day", "contract", *KEYS, "native_captured", "origin_status", "session", "native_label_10s", "native_label_60s"]].copy()
                origins["arm"], origins["score"], origins["available"], origins["decision"], origins["side"] = arm, values, available, statuses, sides
                all_origins.append(origins)
                accepted = statuses == "accepted"
                for delay in [50, 250]:
                    if tape is not None:
                        events = markouts(tape, panel.SampleTime.to_numpy(np.int64)[accepted], sides[accepted], horizon, delay)
                    else:
                        # No invented actions on a missing day; typed empty quote result for reporting.
                        empty_tape = {"time": np.array([], np.int64), "bad_time": np.array([], np.int64), "coverage_end": -1}
                        events = markouts(empty_tape, [], [], horizon, delay)
                    events["day"], events["contract"], events["arm"] = row["day"], row["contract"], arm
                    events["session"] = panel.loc[accepted, "session"].to_numpy()
                    events["score"] = values[accepted]
                    all_events.append(events)
                    all_days.append(daily_stats(events, origins, row["day"], arm, delay))
        print(f"analyzed frozen intents {row['day']}", flush=True)
    events, days, origins = pd.concat(all_events, ignore_index=True), pd.DataFrame(all_days), pd.concat(all_origins, ignore_index=True)
    events.to_parquet(analysis / "all-accepted-intents.parquet", index=False)
    origins.to_parquet(analysis / "all-master-origins.parquet", index=False)
    days.to_csv(analysis / "all-day-arm-delay.csv", index=False)
    events.groupby(["arm", "delay_ms", "day", "side", "status"], dropna=False).size().rename("count").reset_index().to_csv(analysis / "all-support-reasons.csv", index=False)
    reporting_tables(analysis, origins, events)
    summary = {
        "scope": "Conditional research-validation quote markouts; no fills, final OOS or promotion",
        "flow_lineage_passed": flow_lineage_ok,
        "flow_lineage": side_receipts,
        "calibration_sha256": file_hash(output / "calibration.yaml"),
        "arms": {},
    }
    for arm in sorted(days.arm.unique()):
        record = arm_summary(days.loc[days.arm == arm], events.loc[events.arm == arm], flow_lineage_ok if arm.startswith("trade_flow") else True)
        record["primary"] = arm.endswith("_10s")
        record["advances"] = record["advances"] and record["primary"]
        summary["arms"][arm] = record
    write_yaml(analysis / "summary.yaml", summary)
    pd.DataFrame(
        [
            {
                "arm": arm,
                "primary": record["primary"],
                "status": record["status"],
                "advances": record["advances"],
                **{f"gate_{key}": value for key, value in record["gates"].items()},
            }
            for arm, record in summary["arms"].items()
        ]
    ).to_csv(analysis / "all-arm-gates.csv", index=False)
    reporting_supplement(analysis, origins, events, days, summary)
    write_yaml(
        analysis / "receipt.yaml",
        {
            "freeze_sha256": file_hash(output / "freeze.yaml"),
            "calibration_sha256": file_hash(output / "calibration.yaml"),
            "files": {str(p.relative_to(analysis)): file_hash(p) for p in analysis.iterdir() if p.is_file()},
        },
    )
    print(json.dumps({"analysis": str(analysis), "primary_advances": {k: v["advances"] for k, v in summary["arms"].items() if v["primary"]}}), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["prepare", "pilot", "export", "calibrate", "analyze"])
    parser.add_argument("--protocol", type=Path, default=PROTOCOL)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    output = (args.output or Path(checked_protocol(args.protocol)["runtime"]["output"])).resolve()
    if args.stage == "prepare":
        prepare(args.protocol.resolve(), output)
    elif args.stage in {"pilot", "export"}:
        export(output, args.stage == "pilot")
    elif args.stage == "calibrate":
        calibrate(output)
    else:
        analyze(output)


if __name__ == "__main__":
    main()
