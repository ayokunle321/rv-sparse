# rv-sparse Benchmarking Setup

This covers what you need installed before `bench/run_eval.py` will run. The
script's `preflight()` checks all of this automatically and will tell you
exactly what's missing — so once you've got things installed, just run the
eval once and let it warn you if a path is wrong.

## Prerequisites

### 1. RISC-V cross toolchain

You need `riscv64-unknown-linux-gnu-gcc` on disk somewhere. **Use the
prebuilt release** — you do not need to build from source.

**Prebuilt (do this):**

```bash
wget https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/<tag>/riscv64-glibc-ubuntu-<ver>-<date>-nightly.tar.gz
mkdir -p ~/tools/riscv
tar -xzf riscv64-glibc-ubuntu-*.tar.gz -C ~/tools/riscv --strip-components=1
~/tools/riscv/bin/riscv64-unknown-linux-gnu-gcc --version
```

Grab the tag from the
[releases page](https://github.com/riscv-collab/riscv-gnu-toolchain/releases).
Make sure it's a `riscv64-glibc-...` tarball, not `elf-newlib` — you need
glibc since the harness links `-static` against a linux-gnu target. Make
sure it's gcc 16.

**Building from source (only if the prebuilt doesn't fit your needs — e.g.
you need a specific glibc config):**

```bash
git clone https://github.com/riscv-collab/riscv-gnu-toolchain.git
cd riscv-gnu-toolchain
git submodule update --init --recursive

sudo apt-get install autoconf automake autotools-dev curl python3 \
  python3-pip libmpc-dev libmpfr-dev libgmp-dev gawk build-essential \
  bison flex texinfo gperf libtool patchutils bc zlib1g-dev libexpat-dev \
  ninja-build git cmake libglib2.0-dev libslirp-dev

./configure --prefix=~/tools/riscv --with-arch=rv64gcv --with-abi=lp64d
make linux -j$(nproc)
```

Use `make linux` (glibc-based), not plain `make` (that builds the
newlib/bare-metal toolchain instead). This takes 30-60+ minutes depending
on the machine — another reason to just grab the prebuilt.

### 2. gem5, built for RISC-V

```bash
git clone https://github.com/gem5/gem5.git
cd gem5
scons build/RISCV/gem5.opt -j$(nproc)
```

This is a full ISA build via scons — takes a while, budget time for it.

### 3. se.py config

Comes free with the gem5 clone, at:
```
gem5/configs/deprecated/example/se.py
```
This path lives under `deprecated/` and has moved before between gem5
versions. If your clone doesn't have it at that exact path, search the repo
for `se.py` and update your local `experiments.json` accordingly.

### 4. m5ops / libm5

Built from inside the gem5 tree:
```bash
cd gem5/util/m5
scons build/riscv/out/m5
```
This gives you:
- `gem5/include` — the m5ops header (`m5_inc`)
- `gem5/util/m5/build/riscv/out/libm5.a` — the lib (`m5_lib`)

Exact scons target may vary slightly by gem5 version — check your build log
if this path doesn't exist afterward.

## Configuring your local experiments.json

Copy `bench/experiments.json` and edit the `paths` block at the top to point
at wherever *you* installed the above.

```json
"paths": {
  "cc": "~/tools/riscv/bin/riscv64-unknown-linux-gnu-gcc",
  "gem5": "~/gem5/build/RISCV/gem5.opt",
  "se_config": "~/gem5/configs/deprecated/example/se.py",
  "m5_inc": "~/gem5/include",
  "m5_lib": "~/gem5/util/m5/build/riscv/out"
}
```

`~` and relative paths are both fine — the script expands and resolves them.

## Sanity check

Once installed, run:
```bash
./run_eval.py --matrices wiki-Vote --kernels i8
```
This runs `preflight()` first, which checks every path in the config exists
before doing any real work — so a bad setup fails in seconds, not after a
48-hour sweep.

## Turning results into eval CSVs

`run_eval.py` writes one JSON record per run into `results/runs/`, plus a
flattened `results.csv`. For the per-build breakdown used in writeups
(speedup of `gcv` over `gc`, matrix property stats, etc.), run the scraper
on top:

```bash
./bench/scrape_results.py \
    --runs-dir results/runs \
    --matrix-dir matrices \
    --out-dir eval
```

This produces, in `eval/`:

**`gc.csv`** — performance rows for the scalar (`rv64gc`) build.

**`gcv.csv`** — performance rows for the vectorized (`rv64gcv`) build, plus
a `speedup_over_gc` column, computed by joining against `gc.csv`'s cycle
counts for each matrix/kernel pair.

**`matrix_stats.csv`** — static matrix properties, computed by re-parsing
the `.mtx` files in `matrices/`:
- number of rows
- number of nonzeros (`nnz`)
- density
- average nonzeros per row
- maximum nonzeros per row
- maximum-to-median nonzeros-per-row ratio

Notes:
- It only reads records with `status == "ok"` — failed/timeout runs from
  `run_eval.py` are silently skipped here, so check `run_eval.py`'s own
  exit/verification output first if a matrix seems to be missing from the
  CSVs.
- Needs `pandas` installed (`pip install pandas`) — not required by
  `run_eval.py` itself, only by this script.
- Re-run it any time after adding more completed runs; it's read-only
  against `results/runs/` and just regenerates the `eval/` CSVs from
  scratch each time.

## Notes

- `matrices/` and `results/` are gitignored — populate `matrices/` yourself
  before running `./run_eval.py` by running `./matrices/getResources.sh`.