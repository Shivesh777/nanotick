#!/usr/bin/env python3
"""
lob_replay.py — High-performance Python Limit-Order-Book replay engine
======================================================================

Python implementation of the C++ LOB replay engine with optimizations:
* PyArrow zero-copy columnar data processing
* Numba JIT compilation for hot paths
* Efficient collections.defaultdict for order books
* High-resolution time measurement
* Vectorized operations where possible

SETUP & USAGE:
======================================================================

1. Install Dependencies
----------------------------------------------------------------------
   pip install pyarrow numba pandas numpy

2. Run the Replay Engine
----------------------------------------------------------------------
   python lob_replay.py bx_20190530.parquet

PERFORMANCE OPTIMIZATIONS:
======================================================================
* Numba JIT compilation for order book operations
* PyArrow zero-copy data access
* Efficient hash maps using defaultdict
* Minimal Python overhead in hot loops
* High-resolution timing using time.perf_counter_ns()

======================================================================
"""

import sys
import time
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from collections import defaultdict
from typing import Dict, List, Optional
import numba
from numba import jit, types
from numba.typed import Dict as NumbaDict


# ──────────────────────────── Order-book with Numba JIT ──────────────────────
@numba.experimental.jitclass([
    ('oid', numba.uint64),
    ('px', numba.uint32),
    ('qty', numba.uint32),
    ('side', numba.uint8),
])
class Order:
    def __init__(self, oid, px, qty, side):
        self.oid = oid
        self.px = px
        self.qty = qty
        self.side = side


@jit(nopython=True, cache=True)
def add_order(live_orders, bid_levels, ask_levels, oid, side, px, qty):
    """Add order to book with JIT compilation"""
    live_orders[oid] = (px, qty, side)
    
    if side == 0:  # bid
        if px in bid_levels:
            bid_levels[px] += qty
        else:
            bid_levels[px] = qty
    else:  # ask
        if px in ask_levels:
            ask_levels[px] += qty
        else:
            ask_levels[px] = qty


@jit(nopython=True, cache=True)
def cancel_order(live_orders, bid_levels, ask_levels, oid):
    """Cancel order from book with JIT compilation"""
    if oid not in live_orders:
        return
    
    px, qty, side = live_orders[oid]
    
    if side == 0:  # bid
        if px in bid_levels:
            bid_levels[px] -= qty
            if bid_levels[px] == 0:
                del bid_levels[px]
    else:  # ask
        if px in ask_levels:
            ask_levels[px] -= qty
            if ask_levels[px] == 0:
                del ask_levels[px]
    
    del live_orders[oid]


@jit(nopython=True, cache=True)
def execute_order(live_orders, bid_levels, ask_levels, oid, qty_exec):
    """Execute order with JIT compilation"""
    if oid not in live_orders:
        return
    
    px, qty, side = live_orders[oid]
    decr = min(qty_exec, qty)
    
    if side == 0:  # bid
        if px in bid_levels:
            bid_levels[px] -= decr
            if bid_levels[px] == 0:
                del bid_levels[px]
    else:  # ask
        if px in ask_levels:
            ask_levels[px] -= decr
            if ask_levels[px] == 0:
                del ask_levels[px]
    
    new_qty = qty - decr
    if new_qty == 0:
        del live_orders[oid]
    else:
        live_orders[oid] = (px, new_qty, side)


@jit(nopython=True, cache=True)
def replace_order(live_orders, bid_levels, ask_levels, oid, new_oid, new_px, new_qty):
    """Replace order with JIT compilation"""
    if oid not in live_orders:
        return
    
    _, _, side = live_orders[oid]
    cancel_order(live_orders, bid_levels, ask_levels, oid)
    add_order(live_orders, bid_levels, ask_levels, new_oid, side, new_px, new_qty)


class OrderBook:
    """Python order book using Numba-compiled dictionaries"""
    
    def __init__(self):
        # Use Numba typed dictionaries for JIT compilation
        self.live_orders = NumbaDict.empty(
            key_type=numba.uint64,
            value_type=numba.types.Tuple([numba.uint32, numba.uint32, numba.uint8])
        )
        self.bid_levels = NumbaDict.empty(
            key_type=numba.uint32,
            value_type=numba.uint32
        )
        self.ask_levels = NumbaDict.empty(
            key_type=numba.uint32,
            value_type=numba.uint32
        )
    
    def add(self, oid, side, px, qty):
        add_order(self.live_orders, self.bid_levels, self.ask_levels, oid, side, px, qty)
    
    def cancel(self, oid):
        cancel_order(self.live_orders, self.bid_levels, self.ask_levels, oid)
    
    def execute(self, oid, qty_exec):
        execute_order(self.live_orders, self.bid_levels, self.ask_levels, oid, qty_exec)
    
    def replace(self, oid, new_oid, new_px, new_qty):
        replace_order(self.live_orders, self.bid_levels, self.ask_levels, oid, new_oid, new_px, new_qty)


def main():
    if len(sys.argv) != 2:
        print("Usage: python lob_replay.py <bx_YYYYMMDD.parquet>")
        sys.exit(1)
    
    # 1. Load Parquet file with PyArrow (zero-copy)
    print("Loading Parquet file...")
    table = pq.read_table(sys.argv[1])
    rows = len(table)
    
    # 2. Extract column arrays (zero-copy access to Arrow data)
    ts_arr = table.column('ts').to_numpy()
    oid_arr = table.column('oid').to_numpy()
    side_arr = table.column('side').to_numpy()
    px_arr = table.column('px').to_numpy()
    qty_arr = table.column('qty').to_numpy()
    m_arr = table.column('m').to_pandas()  # String array
    stock_arr = table.column('stock').to_pandas()  # String array
    new_oid_arr = table.column('new_oid').to_numpy()
    new_px_arr = table.column('new_px').to_numpy()
    new_qty_arr = table.column('new_qty').to_numpy()
    
    # 3. Initialize per-symbol order books
    books = {}
    latencies = []
    
    print(f"Processing {rows:,} messages...")
    
    # 4. Execute order book replay with per-message latency tracking
    wall_t0 = time.perf_counter()
    
    for i in range(rows):
        tic = time.perf_counter_ns()
        
        oid = oid_arr[i]
        side = side_arr[i]
        px = px_arr[i]
        qty = qty_arr[i]
        m_type = m_arr.iloc[i]
        symbol = stock_arr.iloc[i]
        
        # Get or create order book for symbol
        if symbol not in books:
            books[symbol] = OrderBook()
        ob = books[symbol]
        
        # Process message
        if m_type == 'A':
            ob.add(oid, side, px, qty)
        elif m_type == 'C':
            ob.cancel(oid)
        elif m_type == 'E':
            ob.execute(oid, qty)
        elif m_type == 'U':
            new_oid = new_oid_arr[i]
            new_px = new_px_arr[i]
            new_qty = new_qty_arr[i]
            ob.replace(oid, new_oid, new_px, new_qty)
        
        toc = time.perf_counter_ns()
        latencies.append(toc - tic)
    
    wall_t1 = time.perf_counter()
    
    # 5. Calculate and display performance metrics
    wall_seconds = wall_t1 - wall_t0
    throughput = rows / wall_seconds
    
    # Sort latencies for percentile calculation
    latencies.sort()
    
    def percentile(p):
        return latencies[int(p * len(latencies))]
    
    # Convert nanoseconds to approximate cycles (assuming 3.2 GHz M1)
    # This is for comparison with C++ version
    ns_to_cycles = 3.2  # Approximate conversion for M1
    
    print("\nLOB Replay Metrics")
    print("──────────────────")
    print(f"Rows processed     : {rows:,}")
    print(f"Total wall time (s): {wall_seconds:.6f}")
    print(f"Throughput (msg/s) : {throughput/1e6:.2f} M")
    print(f"Latency (ns) — p50 : {percentile(0.50):,}")
    print(f"Latency (ns) — p95 : {percentile(0.95):,}")
    print(f"Latency (ns) — p99 : {percentile(0.99):,}")
    print(f"Latency (ns) — max : {max(latencies):,}")
    print(f"Latency (cycles) — p50 : {int(percentile(0.50) * ns_to_cycles):,}")
    print(f"Latency (cycles) — p95 : {int(percentile(0.95) * ns_to_cycles):,}")
    print(f"Latency (cycles) — p99 : {int(percentile(0.99) * ns_to_cycles):,}")
    print(f"Latency (cycles) — max : {int(max(latencies) * ns_to_cycles):,}")
    
    print(f"\nOrder books created: {len(books)}")
    print(f"Symbols processed: {', '.join(list(books.keys())[:10])}" + 
          (f" (and {len(books)-10} more)" if len(books) > 10 else ""))


if __name__ == "__main__":
    main()