# gem5 Benchmarking Setup

This covers what you need installed before `bench/run_eval.py` will run.

`run_eval.py` runs a `preflight()` check first and tells you exactly what's
missing or misconfigured, so once you've installed everything below, just run
it once to verify your setup.

## Prerequisites

### 1. RISC-V cross toolchain

You need `riscv64-unknown-linux-gnu-gcc` somewhere on your machine. **Use the
prebuilt release**—there's no need to build it from source unless it doesn't
work for you.

**Prebuilt (recommended):**

```bash
wget https://github.com/riscv-collab/riscv-gnu-toolchain/releases/download/<tag>/riscv64-glibc-ubuntu-<ver>-<date>-nightly.tar.gz
mkdir -p ~/tools/riscv
tar -xzf riscv64-glibc-ubuntu-*.tar.gz -C ~/tools/riscv --strip-components=1
~/tools/riscv/bin/riscv64-unknown-linux-gnu-gcc --version
```

Grab the latest tag from the
[releases page](https://github.com/riscv-collab/riscv-gnu-toolchain/releases).

Make sure it's a `riscv64-glibc-...` tarball, not `elf-newlib`—the harness
links `-static` against a Linux GNU target.

**Building from source (only if the prebuilt doesn't work):**

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

Use `make linux` (glibc-based), not plain `make`, which builds the
newlib/bare-metal toolchain instead. This takes 30–60+ minutes depending on
your machine.

### 2. gem5, built for RISC-V

```bash
git clone https://github.com/gem5/gem5.git
cd gem5
scons build/RISCV/gem5.opt -j$(nproc)
```

This is a full RISC-V build via SCons. It takes a while.

### 3. `se.py` config

This comes with the gem5 repository:

```
gem5/configs/deprecated/example/se.py
```

This path has moved between gem5 versions before. If your checkout doesn't
have it there, search the repo for `se.py` and update
`bench/experiments.json` accordingly.

### 4. m5ops / libm5

Build this from inside the gem5 tree:

```bash
cd gem5/util/m5
scons build/riscv/out/m5
```

This gives you:

- `gem5/include` — the m5ops headers (`m5_inc`)
- `gem5/util/m5/build/riscv/out/libm5.a` — the library (`m5_lib`)

The exact SCons target may vary slightly between gem5 versions, so check the
build output if these paths don't exist afterward.

## Configuring `experiments.json`

Edit the `paths` block near the top of `bench/experiments.json` to point at
wherever you installed the above.

```json
"paths": {
  "cc": "~/tools/riscv/bin/riscv64-unknown-linux-gnu-gcc",
  "gem5": "~/gem5/build/RISCV/gem5.opt",
  "se_config": "~/gem5/configs/deprecated/example/se.py",
  "m5_inc": "~/gem5/include",
  "m5_lib": "~/gem5/util/m5/build/riscv/out"
}
```

`~` and relative paths are both fine—`run_eval.py` expands and resolves them.

## Sanity check

Once installed, run:

```bash
./run_eval.py --matrices wiki-Vote --kernels i8
```

`preflight()` checks that every configured path exists before running any
benchmarks.

## Turning results into evaluation CSVs

`run_eval.py` writes one JSON record per run into `results/runs/`, plus a
flattened `results.csv`.

For the per-build breakdown (speedups, matrix stats, etc.), run:

```bash
./bench/scrape_results.py \
    --runs-dir results/runs \
    --matrix-dir matrices \
    --out-dir eval
```

This produces, in `eval/`:

**`gc.csv`** — performance rows for the scalar (`rv64gc`) build.

**`gcv.csv`** — performance rows for the vectorized (`rv64gcv`) build, plus
a `speedup_over_gc` column computed by joining against `gc.csv`'s cycle
counts for each matrix/kernel pair.

**`matrix_stats.csv`** — static matrix properties computed by re-parsing the
`.mtx` files in `matrices/`:

- number of rows
- number of nonzeros (`nnz`)
- density
- average nonzeros per row
- maximum nonzeros per row
- maximum-to-median nonzeros-per-row ratio

Notes:

- Only records with `status == "ok"` are processed. Failed or timed-out runs
  are skipped, so if a matrix is missing from the CSVs, first check
  `run_eval.py`'s output.
- `scrape_results.py` requires `pandas` (`pip install pandas`). It isn't
  needed by `run_eval.py`.
- Re-run the scraper any time after adding more completed runs. It reads
  `results/runs/` and regenerates the CSVs from scratch.

## Notes

- `matrices/` and `results/` are gitignored.
- Before running `./run_eval.py`, populate `matrices/` by running:

```bash
./matrices/getResources.sh
```