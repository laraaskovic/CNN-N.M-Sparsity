# CNN N:M Sparsity Accelerator with a Confidence Auditor and Early Exit

An INT8 CNN inference accelerator in SystemVerilog, built around three ideas that
compound:

1. **2:4 structured sparsity** — exactly 2 of every 4 weights along the reduction
   axis are non-zero, so the multiplier array can be half the size for the same
   throughput.
2. **A cheap confidence auditor** — per-channel L1 statistics tapped off the
   activation stream, fed to a tiny INT8 MLP that predicts whether an early
   prediction is trustworthy.
3. **Early exit** — when the auditor is confident, the tail of the network never
   runs, so latency becomes a tunable knob rather than a fixed number.

Everything is verified **bit-exact** against an integer-only NumPy golden model.
No "close enough" comparisons anywhere in the flow.

---

## The reference network

CIFAR-10, 4 conv + 2 FC, no batchnorm in the deployed graph (BN is folded into
the conv weights before quantization, so the hardware never sees it).

| Layer | Shape | Output | MACs | Pruned? |
|---|---|---|---|---|
| conv1 | 3→16, 3×3, pad 1 | 16×32×32 | 442,368 | no (C_in=3) |
| conv2 | 16→32, 3×3, pad 1, +pool2 | 32×16×16 | 4,718,592 | yes |
| **exit tap** | per-channel accumulate | 32 features | ~8K adds | — |
| conv3 | 32→64, 3×3, pad 1, +pool2 | 64×8×8 | 4,718,592 | yes |
| conv4 | 64→128, 3×3, pad 1, +pool2 | 128×4×4 | 4,718,592 | yes |
| fc1 | 2048→64 | 64 | 131,072 | yes |
| fc2 | 64→10 | 10 | 640 | no (tiny) |
| **total** | | | **14,729,856** | |

The three conv stages are deliberately sized to **exactly 4,718,592 MACs each**.
That makes cycle counts trivial to reason about and makes the early-exit story
clean: the head (conv1+conv2) is 35% of the work, the tail is 65%.

Auxiliary heads, both fed from the same 32 per-channel accumulators:

| Block | Shape | MACs |
|---|---|---|
| exit classifier | 32→10 | 320 |
| auditor MLP | 32→32→16→1 | 1,552 |

Combined: **0.013% of the network**. The confidence machinery is free.

### The trick that makes the tap free

The exit tap sits **after a ReLU**, so every activation is non-negative and
`|x| == x`. A per-channel L1 norm and a per-channel global average pool are
therefore *the same accumulator*. One adder tree and 32 INT32 registers, running
concurrently with the activation stream, feed both the early classifier and the
auditor. Zero extra cycles, zero extra multipliers.

---

## Configurations to build and measure

The PE array is parameterized by multiplier count `P`. The sparse unit consumes
`2P` weights' worth of work per cycle because half of them are structurally zero.

| Config | P | DSPs | Cycles/inference (ideal) | Story |
|---|---|---|---|---|
| dense | 16 | 16 | ~920K | baseline |
| sparse, iso-latency | 8 | 8 | ~920K | **half the area** |
| sparse, iso-area | 16 | 16 | ~474K | **~2× the speed** |
| sparse + early exit | 16 | 16 | sweeps with threshold | latency as a knob |

One parameterized design produces both sparse rows. Being able to report
iso-area *and* iso-latency comparisons from the same RTL is a better answer than
either number alone.

---

## Repo layout

```
model/      PyTorch: train, BN-fold, 2:4 prune, quantize, export, auditor
golden/     integer-only NumPy reference + test-vector generation  <-- source of truth
rtl/
  common/   axis pkg, requant, mac_lane, adder_tree
  dense/    dense PE array
  sparse/   2:4 PE array, index decode
  mem/      line buffers, weight/activation buffers, ping-pong
  ctrl/     layer FSM, address generator, CSR block
  audit/    L1 accumulator, auditor MLP, exit control
  top/      accel_top
tb/         cocotb tests (Verilator)
syn/        Vivado scripts, constraints, captured reports
vectors/    golden .npy test vectors (generated, not committed)
results/    plots and tables for the writeup
docs/       planning and learning material
```

## Docs

- [`docs/BACKLOG.md`](docs/BACKLOG.md) — epics, stories, tasks, acceptance criteria
- [`docs/HARDWARE_PRIMER.md`](docs/HARDWARE_PRIMER.md) — the hardware half, explained for someone who knows PyTorch
- [`docs/BUILD_GUIDE.md`](docs/BUILD_GUIDE.md) — step-by-step build order, de-risked
- [`docs/INTERVIEW_NOTES.md`](docs/INTERVIEW_NOTES.md) — what interviewers will ask about this, and what your own numbers answer

## Status

Planning. No code yet.
