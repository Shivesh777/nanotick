#!/usr/bin/env python3
"""
itch_to_parquet.py  •  TotalView-ITCH-5.0 → Parquet
===================================================
Streams an ITCH session (.gz or raw) and writes a **loss-free** Parquet file
containing every field a deterministic limit-order-book simulator needs,
including *replace* details.  Benchmarked about 1.6 M msg/s. 1.4 gb in csv to 330 mb in parquet.

Captured columns
----------------
ts        : uint64  • nanoseconds since midnight (itchfeed already expands 48-bit)
oid       : uint64  • order_reference_number
side      : uint8   • 1 = buy, 0 = sell, 255 = N/A (exec/cancel/delete/replace)
px        : uint32  • price × 10⁻⁴ USD (0 if not present)
qty       : uint32  • shares/contracts (0 if not present)
m         : string  • single-char message type
stock     : string  • ticker symbol (dict-encoded, near-free)
new_oid   : uint64  • new_order_reference_number (U-messages); 0 otherwise
new_px    : uint32  • replacement price; 0 otherwise
new_qty   : uint32  • replacement shares; 0 otherwise

Usage
-----
  pip install itchfeed pyarrow tqdm
  python itch_to_parquet.py 20190530.BX_ITCH_50.gz  bx_20190530.parquet
  python itch_to_parquet.py raw.ITCH out.parquet --symbol AAPL
"""

from __future__ import annotations
import argparse, gzip, pathlib, sys, typing as t, time, os
from collections import defaultdict
import pyarrow as pa, pyarrow.parquet as pq
from tqdm import tqdm
from itch.parser import MessageParser

# ──────────────────────────── robust class lookup ───────────────────────────
def _try(names: list[str]) -> type | None:
    """Return the first present itch.messages class from *names* list."""
    import importlib
    mod = importlib.import_module("itch.messages")
    for n in names:
        cls = getattr(mod, n, None)
        if cls is not None:
            return cls
    return None

AddOrderMessage            = _try(["AddOrderMessage"])
AddOrderMPIDAttrib         = _try(["AddOrderMPIDAttributionMessage",
                                   "AddOrderMPIDAttribution"])
OrderExecutedMessage       = _try(["OrderExecutedMessage"])
OrderExecutedWithPxMessage = _try(["OrderExecutedWithPriceMessage"])
OrderCancelMessage         = _try(["OrderCancelMessage"])
OrderDeleteMessage         = _try(["OrderDeleteMessage"])
OrderReplaceMessage        = _try(["OrderReplaceMessage"])
NonCrossTradeMessage       = _try(["NonCrossTradeMessage"])
CrossTradeMessage          = _try(["CrossTradeMessage"])

_ORDER_MSGS: tuple[type] = tuple(
    cls for cls in (
        AddOrderMessage, AddOrderMPIDAttrib,
        OrderExecutedMessage, OrderExecutedWithPxMessage,
        OrderCancelMessage, OrderDeleteMessage, OrderReplaceMessage,
        NonCrossTradeMessage, CrossTradeMessage,
    ) if cls
)
if not _ORDER_MSGS:
    sys.exit("✖  itchfeed API drifted — update the class list.")

# ─────────────────────────── helper field extractors ────────────────────────
def _side(msg) -> int:
    bs = getattr(msg, "buy_sell_indicator", None)
    return 255 if bs is None else (1 if bs == b'B' else 0)

def _price(msg) -> int:
    for f in ("price", "execution_price", "execution_price_old",
              "execution_price_new"):
        p = getattr(msg, f, None)
        if p is not None:
            return p
    return 0

def _qty(msg) -> int:
    for f in ("shares", "executed_shares", "quantity", "shares_remaining"):
        q = getattr(msg, f, None)
        if q is not None:
            return q
    return 0

# ─────────────────────────────  main decoder  ───────────────────────────────
def _iter_msgs(fh: t.BinaryIO):
    yield from MessageParser().read_message_from_file(fh)

def decode_itch(path: pathlib.Path, sym: str | None) -> tuple[pa.Table, dict]:
    """Decode ITCH file and return table + metrics."""
    ts, oid, side, px, qty, m, stock = [], [], [], [], [], [], []
    new_oid, new_px, new_qty = [], [], []
    open_fn = gzip.open if path.suffix == ".gz" else open
    
    # Metrics tracking
    start_time = time.time()
    total_messages = 0
    filtered_messages = 0
    msg_type_counts = defaultdict(int)
    bytes_processed = 0
    parse_start = None
    
    # Get file size for throughput calculation
    file_size = path.stat().st_size
    
    with open_fn(path, "rb") as fh:
        parse_start = time.time()
        for msg in tqdm(_iter_msgs(fh), unit="msg", desc="Parsing"):
            total_messages += 1
            msg_type_counts[msg.message_type.decode()] += 1
            
            if not isinstance(msg, _ORDER_MSGS):
                continue
                
            sym_raw = getattr(msg, "stock", b"")
            sym_str = sym_raw.decode().strip()
            if sym and sym_str != sym:
                continue
                
            filtered_messages += 1
            ts.append(msg.timestamp)
            oid.append(getattr(msg, "order_reference_number", 0))
            side.append(_side(msg))
            px.append(_price(msg))
            qty.append(_qty(msg))
            m.append(msg.message_type.decode())
            stock.append(sym_str)

            if isinstance(msg, OrderReplaceMessage):
                new_oid.append(msg.new_order_reference_number)
                new_px.append(msg.price)
                new_qty.append(msg.shares)
            else:
                new_oid.append(0)
                new_px.append(0)
                new_qty.append(0)

    parse_end = time.time()
    
    if not ts:
        sys.exit("✖  No rows — wrong file or overly strict symbol filter.")

    # Build table
    table_start = time.time()
    table = pa.Table.from_pydict(
        {
            "ts":       pa.array(ts,      pa.uint64()),
            "oid":      pa.array(oid,     pa.uint64()),
            "side":     pa.array(side,    pa.uint8()),
            "px":       pa.array(px,      pa.uint32()),
            "qty":      pa.array(qty,     pa.uint32()),
            "m":        pa.array(m,       pa.string()),
            "stock":    pa.array(stock,   pa.string()),
            "new_oid":  pa.array(new_oid, pa.uint64()),
            "new_px":   pa.array(new_px,  pa.uint32()),
            "new_qty":  pa.array(new_qty, pa.uint32()),
        }
    )
    table_end = time.time()
    
    # Calculate metrics
    parse_time = parse_end - parse_start
    table_time = table_end - table_start
    total_time = parse_end - start_time
    
    metrics = {
        'total_time': total_time,
        'parse_time': parse_time,
        'table_creation_time': table_time,
        'total_messages': total_messages,
        'filtered_messages': filtered_messages,
        'messages_per_second': total_messages / parse_time if parse_time > 0 else 0,
        'filtered_msgs_per_second': filtered_messages / parse_time if parse_time > 0 else 0,
        'file_size_mb': file_size / (1024 * 1024),
        'throughput_mbps': (file_size / (1024 * 1024)) / parse_time if parse_time > 0 else 0,
        'filter_ratio': filtered_messages / total_messages if total_messages > 0 else 0,
        'msg_type_counts': dict(msg_type_counts),
        'avg_msg_size': file_size / total_messages if total_messages > 0 else 0,
    }
    
    return table, metrics

def print_metrics(metrics: dict, output_path: pathlib.Path, table_rows: int) -> None:
    """Print comprehensive performance metrics."""
    print(f"\n{'='*60}")
    print(f"📊 PERFORMANCE METRICS")
    print(f"{'='*60}")
    
    print(f"⏱️  TIMING:")
    print(f"   Total time:           {metrics['total_time']:.2f}s")
    print(f"   Parse time:           {metrics['parse_time']:.2f}s")
    print(f"   Table creation:       {metrics['table_creation_time']:.2f}s")
    
    print(f"\n📈 THROUGHPUT:")
    print(f"   Messages/sec:         {metrics['messages_per_second']:,.0f}")
    print(f"   Filtered msgs/sec:    {metrics['filtered_msgs_per_second']:,.0f}")
    print(f"   File throughput:      {metrics['throughput_mbps']:.1f} MB/s")
    
    print(f"\n📊 MESSAGE STATS:")
    print(f"   Total messages:       {metrics['total_messages']:,}")
    print(f"   Filtered messages:    {metrics['filtered_messages']:,}")
    print(f"   Filter ratio:         {metrics['filter_ratio']:.1%}")
    print(f"   Avg message size:     {metrics['avg_msg_size']:.1f} bytes")
    
    print(f"\n💾 FILE INFO:")
    print(f"   Input file size:      {metrics['file_size_mb']:.1f} MB")
    if output_path.exists():
        output_size = output_path.stat().st_size / (1024 * 1024)
        compression_ratio = metrics['file_size_mb'] / output_size
        print(f"   Output file size:     {output_size:.1f} MB")
        print(f"   Compression ratio:    {compression_ratio:.1f}x")
    
    print(f"\n🔤 MESSAGE TYPES:")
    for msg_type, count in sorted(metrics['msg_type_counts'].items(), 
                                  key=lambda x: x[1], reverse=True):
        pct = (count / metrics['total_messages']) * 100
        print(f"   {msg_type:>3}: {count:>8,} ({pct:>5.1f}%)")
    
    print(f"\n✅ OUTPUT:")
    print(f"   Parquet rows:         {table_rows:,}")
    print(f"={'='*60}")

# ────────────────────────────────  CLI  ─────────────────────────────────────
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("itch_file",   type=pathlib.Path,
                    help="ITCH 5.0 session (.gz or raw)")
    ap.add_argument("out_parquet", type=pathlib.Path,
                    help="Destination parquet file")
    ap.add_argument("--symbol", help="Filter to single ticker (optional)")
    ap.add_argument("--row-group", type=int, default=1_000_000,
                    help="Parquet row-group size (default 1 M)")
    args = ap.parse_args()

    print(f"🚀 Starting ITCH→Parquet conversion...")
    print(f"   Input:  {args.itch_file}")
    print(f"   Output: {args.out_parquet}")
    if args.symbol:
        print(f"   Symbol: {args.symbol}")
    
    # Decode with metrics
    table, metrics = decode_itch(args.itch_file, args.symbol)
    
    # Write parquet file
    write_start = time.time()
    pq.write_table(
        table, args.out_parquet,
        compression="zstd", use_dictionary=True,
        data_page_size=1 << 20, row_group_size=args.row_group,
    )
    write_time = time.time() - write_start
    
    print(f"✓  Parquet write completed in {write_time:.2f}s")
    
    # Print comprehensive metrics
    print_metrics(metrics, args.out_parquet, table.num_rows)

if __name__ == "__main__":
    main()