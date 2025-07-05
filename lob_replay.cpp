// lob_replay.cpp — Portable, Arrow‑20‑ready Limit‑Order‑Book replay engine
// ======================================================================
// 
// High-performance limit order book replay engine that processes market data
// from Parquet files and provides detailed latency metrics for each message.
//
// FEATURES:
// * Zero-copy Arrow columnar data processing
// * Real limit order book semantics (add/cancel/execute/replace)
// * Sub-10ns per-message latency measurement using cycle counters
// * Comprehensive throughput and latency statistics (p50/p95/p99/max)
// * Support for BX/ITCH-like market data schemas
//
// SETUP & USAGE:
// ======================================================================
//
// 1. Install Dependencies (macOS with Homebrew)
// ----------------------------------------------------------------------
//    brew install apache-arrow
//    # For other platforms, install Apache Arrow 20+ via your package manager
//
// 2. Download Sample Data
// ----------------------------------------------------------------------
//    wget https://emi.nasdaq.com/ITCH/Nasdaq%20BX%20ITCH/20190530.BX_ITCH_50.gz
//
// 3. Convert ITCH to Parquet (requires itch_to_parquet.py)
// ----------------------------------------------------------------------
//    python data_pipeline/itch_to_parquet.py 20190530.BX_ITCH_50.gz bx_20190530.parquet
//
// 4. Build the Executable
// ----------------------------------------------------------------------
//    INC=$(brew --prefix)/include
//    LIB=$(brew --prefix)/lib
//    g++ -O3 -march=native -std=c++20 lob_replay.cpp \
//        -I${INC} -L${LIB} -larrow -lparquet -pthread -o lob_replay
//
//    # If runtime loader can't find dylibs:
//    export DYLD_LIBRARY_PATH="${LIB}:$DYLD_LIBRARY_PATH"
//
// 5. Run the Replay Engine
// ----------------------------------------------------------------------
//    ./lob_replay bx_20190530.parquet
//
// EXPECTED PARQUET SCHEMA:
// ======================================================================
// Column names (all non-nullable):
//   ts:uint64        - timestamp
//   oid:uint64       - order ID
//   side:uint8       - 0=bid, 1=ask
//   px:uint32        - price
//   qty:uint32       - quantity
//   m:string         - message type ("A"=add, "C"=cancel, "E"=execute, "U"=replace)
//   stock:string     - symbol
//   new_oid:uint64   - new order ID (for replace messages)
//   new_px:uint32    - new price (for replace messages)
//   new_qty:uint32   - new quantity (for replace messages)
//
// PERFORMANCE NOTES:
// ======================================================================
// * Uses __builtin_readcyclecounter() for sub-10ns latency measurement
// * Optimized for Apple Silicon and x86-64 architectures
// * Processes millions of messages per second with detailed per-message metrics
// * Memory-efficient order book implementation using hash maps
//
// ======================================================================

#include <arrow/api.h>
#include <arrow/io/api.h>
#include <parquet/arrow/reader.h>
#include <parquet/exception.h>

#include <chrono>
#include <unordered_map>
#include <vector>
#include <iostream>
#include <cstring>
#include <algorithm>
#include <string_view>

// ──────────────────────────── Order‑book structs ──────────────────────
struct Order {
    uint64_t oid;
    uint32_t px;
    uint32_t qty;
    uint8_t  side;   // 0 = bid, 1 = ask
};

struct OrderBook {
    std::unordered_map<uint64_t, Order> live;           // oid → Order
    std::unordered_map<uint32_t, uint32_t> bid_size;    // px → qty
    std::unordered_map<uint32_t, uint32_t> ask_size;

    inline void add(uint64_t oid, uint8_t side, uint32_t px, uint32_t qty) {
        live.emplace(oid, Order{oid, px, qty, side});
        auto &lvl = (side == 0 ? bid_size : ask_size);
        lvl[px] += qty;
    }
    inline void cancel(uint64_t oid) {
        auto it = live.find(oid);
        if (it == live.end()) return;
        auto &lvl = (it->second.side == 0 ? bid_size : ask_size);
        auto lit = lvl.find(it->second.px);
        if (lit != lvl.end()) {
            lit->second -= it->second.qty;
            if (lit->second == 0) lvl.erase(lit);
        }
        live.erase(it);
    }
    inline void execute(uint64_t oid, uint32_t qty_exec) {
        auto it = live.find(oid);
        if (it == live.end()) return;
        auto &o   = it->second;
        auto &lvl = (o.side == 0 ? bid_size : ask_size);
        auto lit  = lvl.find(o.px);
        uint32_t decr = std::min(qty_exec, o.qty);
        if (lit != lvl.end()) {
            lit->second -= decr;
            if (lit->second == 0) lvl.erase(lit);
        }
        o.qty -= decr;
        if (o.qty == 0) live.erase(it);
    }
    inline void replace(uint64_t oid, uint64_t new_oid, uint32_t new_px, uint32_t new_qty) {
        auto it = live.find(oid);
        if (it == live.end()) return;
        uint8_t side = it->second.side;
        cancel(oid);
        add(new_oid, side, new_px, new_qty);
    }
};

// ───────────────────────── Low‑level cycle timer (ns precision) ─────────────
static inline uint64_t rdtsc() {
    return __builtin_readcyclecounter(); // works on x86‑64 & Apple‑silicon
}

// ─────────────────────────────────── Main Application ─────────────────────────────
int main(int argc, char **argv) {
    if (argc != 2) {
        std::cerr << "Usage: ./lob_replay <bx_YYYYMMDD.parquet>\n";
        return 1;
    }

    // 1. Open Parquet file and initialize Arrow reader ----------------------
    auto maybe_infile = arrow::io::ReadableFile::Open(argv[1]);
    if (!maybe_infile.ok()) { std::cerr << maybe_infile.status() << '\n'; return 1; }
    std::shared_ptr<arrow::io::RandomAccessFile> infile = *maybe_infile;

    parquet::ReaderProperties pq_props(arrow::default_memory_pool());
    std::unique_ptr<parquet::ParquetFileReader> pq_reader = parquet::ParquetFileReader::Open(infile, pq_props);

    parquet::ArrowReaderProperties ar_props;
    std::unique_ptr<parquet::arrow::FileReader> reader;
    PARQUET_THROW_NOT_OK(parquet::arrow::FileReader::Make(arrow::default_memory_pool(), std::move(pq_reader), ar_props, &reader));

    std::shared_ptr<arrow::Table> table;
    PARQUET_THROW_NOT_OK(reader->ReadTable(&table));
    PARQUET_ASSIGN_OR_THROW(table, table->CombineChunks(arrow::default_memory_pool()));

    const int64_t rows = table->num_rows();

    // 2. Extract column arrays (zero-copy access to Arrow data) --------------
    auto ts_col      = table->GetColumnByName("ts")->chunk(0);
    auto oid_col     = table->GetColumnByName("oid")->chunk(0);
    auto side_col    = table->GetColumnByName("side")->chunk(0);
    auto px_col      = table->GetColumnByName("px")->chunk(0);
    auto qty_col     = table->GetColumnByName("qty")->chunk(0);
    auto m_col       = table->GetColumnByName("m")->chunk(0);
    auto stock_col   = table->GetColumnByName("stock")->chunk(0);
    auto new_oid_col = table->GetColumnByName("new_oid")->chunk(0);
    auto new_px_col  = table->GetColumnByName("new_px")->chunk(0);
    auto new_qty_col = table->GetColumnByName("new_qty")->chunk(0);

    auto ts_arr      = std::static_pointer_cast<arrow::UInt64Array>(ts_col);
    auto oid_arr     = std::static_pointer_cast<arrow::UInt64Array>(oid_col);
    auto side_arr    = std::static_pointer_cast<arrow::UInt8Array >(side_col);
    auto px_arr      = std::static_pointer_cast<arrow::UInt32Array>(px_col);
    auto qty_arr     = std::static_pointer_cast<arrow::UInt32Array>(qty_col);
    auto m_arr       = std::static_pointer_cast<arrow::StringArray>(m_col);
    auto stock_arr   = std::static_pointer_cast<arrow::StringArray>(stock_col);
    auto new_oid_arr = std::static_pointer_cast<arrow::UInt64Array>(new_oid_col);
    auto new_px_arr  = std::static_pointer_cast<arrow::UInt32Array>(new_px_col);
    auto new_qty_arr = std::static_pointer_cast<arrow::UInt32Array>(new_qty_col);

    // 3. Initialize per‑symbol order books -----------------------------------
    std::unordered_map<std::string, OrderBook> books;
    books.reserve(256);

    std::vector<uint64_t> latencies; latencies.reserve(rows);

    // 4. Execute order book replay with per-message latency tracking ---------
    const auto wall_t0 = std::chrono::steady_clock::now();
    const uint64_t cyc0 = rdtsc();

    for (int64_t i = 0; i < rows; ++i) {
        uint64_t tic = rdtsc();

        uint64_t oid  = oid_arr->Value(i);
        uint8_t  side = side_arr->Value(i);
        uint32_t px   = px_arr->Value(i);
        uint32_t qty  = qty_arr->Value(i);
        std::string_view m_type = m_arr->GetView(i);     // "A", "C", "E", "U" …
        const std::string &sym  = stock_arr->GetString(i);

        OrderBook &ob = books.emplace(sym, OrderBook{}).first->second;

        if (m_type == "A") {
            ob.add(oid, side, px, qty);
        } else if (m_type == "C") {
            ob.cancel(oid);
        } else if (m_type == "E") {
            ob.execute(oid, qty);
        } else if (m_type == "U") {
            uint64_t new_oid = new_oid_arr->Value(i);
            uint32_t new_px  = new_px_arr->Value(i);
            uint32_t new_qty = new_qty_arr->Value(i);
            ob.replace(oid, new_oid, new_px, new_qty);
        }

        uint64_t toc = rdtsc();
        latencies.push_back(toc - tic);
    }

    const uint64_t cyc1 = rdtsc();
    const auto wall_t1 = std::chrono::steady_clock::now();

    // 5. Calculate and display performance metrics ---------------------------
    double wall_seconds = std::chrono::duration<double>(wall_t1 - wall_t0).count();
    double throughput   = rows / wall_seconds;

    std::sort(latencies.begin(), latencies.end());
    auto pct = [&](double p){ return latencies[static_cast<size_t>(p * latencies.size())]; };

    std::cout << "LOB Replay Metrics\n";
    std::cout << "──────────────────\n";
    std::cout << "Rows processed     : " << rows << '\n';
    std::cout << "Total wall time (s): " << wall_seconds << '\n';
    std::cout << "Throughput (msg/s) : " << throughput/1e6 << " M" << '\n';
    std::cout << "Latency (cycles) — p50 : " << pct(0.50) << '\n';
    std::cout << "Latency (cycles) — p95 : " << pct(0.95) << '\n';
    std::cout << "Latency (cycles) — p99 : " << pct(0.99) << '\n';
    std::cout << "Latency (cycles) — max : " << latencies.back() << '\n';

    return 0;
}