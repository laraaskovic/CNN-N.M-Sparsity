# Interview Notes

What this project lets you claim, what you'll be asked, and where the traps are.

---

## What roles this maps to

| Role | What they'll care about here |
|---|---|
| **Design Verification** (most hiring) | The golden model, bit-exactness, cocotb, randomized backpressure, SVA. EPIC-3's verification story *is* the job. |
| **RTL Design** | The datapath, the FSM, pipelining for Fmax, the DSP/LUT trade |
| **ML Hardware / Accelerator Architecture** | The sparsity trade-off, dataflow choice, the early-exit control question, MAC utilization |
| **HW/SW Co-design, Quantization** | Symmetric-vs-asymmetric, per-channel requant, BN folding, why the golden model has to be integer-only |

Companies where this is directly on-topic: NVIDIA (2:4 sparsity is theirs —
Ampere sparse tensor cores), AMD/Xilinx, Intel, Qualcomm, Apple, Arm, Tenstorrent,
SiMa.ai, Axelera, Untether, plus every automotive and edge-inference group.

---

## Questions you will get, and what your project answers

**"Walk me through what happens to one image."**
The whole project in 90 seconds. Practice it. INT8 image streams in → line buffer
→ 16-lane MAC array → INT32 accumulate → requantize to INT8 → ReLU → pool →
next layer. At conv2's output a per-channel accumulator taps the stream for free,
a tiny MLP scores confidence, and the FSM either emits early or continues.

**"Why 2:4 and not just prune 50% of the weights?"**
Unstructured sparsity is more accurate at the same ratio but the hardware can't
use it — variable-length encoding, dynamic scheduling, load imbalance across
lanes. 2:4 guarantees every group takes identical time. Fixed latency, fixed
bandwidth, trivial control. You trade accuracy for regularity.

**"What did sparsity actually cost you?"**
Your EPIC-4 table. DSPs roughly halve, LUTs go up from muxes and index decode,
Fmax may drop because the mux sits in front of the multiplier, and weights carry
25% index overhead. Having measured all four is the answer.

**"How do you know the RTL is correct?"**
Integer-only golden model written first, per-layer comparison of both the INT32
accumulators and the INT8 outputs, 20 images, all layers, randomized
backpressure, zero mismatches. Plus the bugs it caught.

**"Why symmetric quantization?"**
Zero-points introduce `−z_x·Σw` correction terms, one of which is data-dependent.
Symmetric makes the datapath `Σw·x` and deletes a hardware block, for a fraction
of a percent of accuracy. A software choice made for a hardware reason.

**"What was your critical path?"**
Almost certainly the requantizer's multiply, or the sparse mux. Know the number
and know how you pipelined it.

**"What's your MAC utilization?"**
Have the number. If it's 70%, know where the other 30% went. This question
separates people who measured from people who assumed.

**"Early exit makes latency data-dependent. What breaks?"**
Fixed-latency testbenches. Downstream consumers that assume a cycle count.
Worst-case timing budgets in a real-time system — your *average* improved but
your worst case didn't, which matters enormously in automotive. And the
prefetch-vs-stall question: do you speculatively fetch the tail weights during
the auditor's decision?

**"Did the L1 auditor beat softmax?"**
Whatever your number is, say it. If it lost, the answer is: comparable detection
from features that exist before any classifier head runs, using no exp/log
hardware, reusing the existing MAC array. And then give the AUROC anyway.

---

## Traps

**Claiming 2× speedup from halving the array.** Halving the array is an *area*
win at constant latency. Report the configuration you actually built, and have
both P=8 and P=16 numbers so you can discuss iso-area and iso-latency separately.

**Saying "50% fewer multiplies" without the index overhead.** 2 bits per 8-bit
weight is 25% more weight memory traffic. Mention it before they do.

**Confusing latency and throughput.** Pipelining increases latency and increases
throughput. Be precise; people notice.

**Overstating the auditor's novelty.** Early exit and confidence-based gating are
well-studied (BranchyNet, SDN, and the whole selective-prediction literature).
Your contribution is the *hardware* framing: a confidence signal that is free
because it reuses a post-ReLU accumulator you'd build anyway. Frame it that way
and you sound informed rather than unaware.

**Not knowing your own numbers.** Cycles, DSPs, LUTs, Fmax, top-1 loss, AUROC,
utilization. Have a one-page cheat sheet and know them cold. Nothing deflates a
project conversation faster than "I'd have to check."

---

## The one-line version

> "An INT8 CNN accelerator with 2:4 structured sparsity and a confidence-gated
> early exit, verified bit-exact against an integer-only golden model. Half the
> DSPs at equal latency from sparsity, and a further 30% average cycle reduction
> from early exit, with accuracy-versus-latency as a runtime-tunable threshold."

Fill in the real numbers once you have them. Don't ship a number you haven't
measured.
