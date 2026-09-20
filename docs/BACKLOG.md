# Backlog: Epics → Stories → Tasks

Estimates are in **days of focused work**, not calendar days. The original
10-week plan assumes you already write RTL fluently; the estimates below assume
you do not, and total roughly **14–16 weeks part-time**. Phase 3 is where that
difference lives. See `BUILD_GUIDE.md` for how to de-risk it.

Labels used: `model` `golden` `rtl` `dv` `synth` `docs` `risk:high`

---

## EPIC-0 — Baseline and scope lock
*Goal: a frozen reference you can never argue with later.*
**Exit criteria:** checkpoint on disk, test-set top-1 number written down, per-layer MAC table committed.
**Estimate: 3 days**

### S0.1 — Repo and tooling skeleton
| Task | Detail | Est |
|---|---|---|
| T0.1.1 | Create the directory layout from the README; `.gitignore` for `*.pth`, `vectors/`, Vivado junk | 0.5d |
| T0.1.2 | `requirements.txt` (torch, numpy, matplotlib, cocotb, cocotb-test, pytest) | 0.2d |
| T0.1.3 | Install Verilator + cocotb, run a 10-line blinky testbench to prove the toolchain works | 0.5d |
| T0.1.4 | GitHub Actions workflow: lint RTL (`verilator --lint-only`) + run cocotb tests | 0.5d |

**Acceptance:** `make test` runs a trivial cocotb test green, locally and in CI.
Do this *first*. Discovering your simulator is broken in week 5 is the classic way this project dies.

### S0.2 — Train the reference network
| Task | Detail | Est |
|---|---|---|
| T0.2.1 | `model/net.py`: the 4-conv/2-FC network from the README, BN after every conv | 0.3d |
| T0.2.2 | `model/train.py`: CIFAR-10, standard aug (crop+flip), SGD+cosine, ~40 epochs | 0.5d |
| T0.2.3 | Train to convergence on Colab; save `checkpoints/fp32_baseline.pth` | 0.5d |
| T0.2.4 | `model/fold_bn.py`: fold BN into conv weights/bias, verify FP32 accuracy is unchanged to <0.05% | 0.5d |
| T0.2.5 | `model/mac_count.py`: emit the per-layer MAC table as markdown; commit it | 0.3d |

**Acceptance:** BN-folded FP32 model matches the BN model's top-1 within noise.
Expect ~85–88% top-1. Accuracy is not the point — a *fixed* number is.

> **Why fold BN now:** BN at inference is an affine per-channel scale and shift.
> Folded into the preceding conv it disappears entirely. Leave it in and your
> hardware needs a second requantization stage per layer for no benefit.

---

## EPIC-1 — Compression: 2:4 pruning and INT8 quantization
*Goal: an integer-only model that a fixed-point datapath can execute exactly.*
**Exit criteria:** integer-only NumPy inference within 0.3% of quantized PyTorch; top-1 loss vs FP32 stated; activation dumps for 20 images saved.
**Estimate: 8 days** · `risk:high` on T1.3.2

### S1.1 — 2:4 structured pruning
| Task | Detail | Est |
|---|---|---|
| T1.1.1 | `model/prune_24.py`: reshape each weight tensor so the **reduction axis** is contiguous, group in 4s, keep top-2 by magnitude | 1d |
| T1.1.2 | Assert every group has exactly 2 non-zeros; unit-test the mask generator | 0.3d |
| T1.1.3 | Fine-tune with the mask re-applied after every optimizer step, ~10 epochs at 0.1× LR | 1d |
| T1.1.4 | Record top-1 before/after pruning | 0.2d |

> **The gotcha that ruins this project if you get it wrong:** the groups of 4
> must be along the axis the hardware *reduces over*, in the order the hardware
> *reads them*. For a conv weight `[C_out, C_in, kh, kw]` the reduction axis is
> `C_in × kh × kw` flattened. If your hardware iterates `C_in` innermost, group
> along `C_in`. Prune along `C_out` instead and you have a sparse model your
> datapath cannot exploit at all — it will still be structurally sparse, still
> lose accuracy, and buy you exactly nothing.
>
> Consequences: `C_in` must be a multiple of 4 for every pruned layer (16, 32,
> 64 — check), and **conv1 stays dense** because `C_in=3`. That's standard;
> NVIDIA leaves the first layer dense too.

### S1.2 — INT8 quantization
| Task | Detail | Est |
|---|---|---|
| T1.2.1 | Per-**channel symmetric** weight scales (one `s_w` per output channel, zero-point 0) | 0.5d |
| T1.2.2 | Per-**tensor symmetric** activation scales from calibration on ~512 train images (percentile clipping, e.g. 99.9%) | 0.5d |
| T1.2.3 | Compute per-channel requant params: `M = s_w[c]·s_x / s_y`, decompose into `M0 ∈ [2^30, 2^31)` and right-shift `n` such that `M ≈ M0·2^-n` | 0.5d |
| T1.2.4 | Fake-quant evaluation in PyTorch; record top-1 | 0.5d |

> **Choose symmetric everywhere, and know why.** Asymmetric (zero-point ≠ 0)
> quantization gives slightly better activation range utilization, but it forces
> the accumulator to compute `Σ(w)(x − z_x) = Σwx − z_x·Σw`, which means an extra
> per-output-channel correction term and the logic to apply it. Symmetric makes
> the datapath literally `Σ w·x` and nothing else. This is a hardware/software
> co-design decision and a very good thing to be able to explain: *you gave up a
> fraction of a percent of accuracy to delete a hardware block.*

### S1.3 — The golden model
| Task | Detail | Est |
|---|---|---|
| T1.3.1 | `golden/int_ref.py`: integer-only conv/FC — INT8×INT8 → INT32 accumulate, no floats anywhere in the inner loop | 1.5d |
| T1.3.2 | Requantize exactly: `(acc · M0 + (1 << (n-1))) >> n`, then clamp to [−128, 127]. Pick **round-half-up** and write it down | 1d · `risk:high` |
| T1.3.3 | Prove accumulator width: max reduction length is fc1 at 2048; `2048 × 127 × 127 = 33,032,192` fits in 25 magnitude bits, so 26 bits including sign → INT32 leaves 6 bits of headroom. Put this in the docs | 0.3d |
| T1.3.4 | Assert no float ops: run the reference with `numpy` integer dtypes only and a guard that raises on any float array | 0.3d |
| T1.3.5 | Evaluate full test set with the integer reference; compare to fake-quant PyTorch | 0.5d |

> **This is the single most important deliverable in the project.** PyTorch's
> fake-quant rounds in floating point and then rounds again; your hardware does
> an integer multiply and an arithmetic right shift. Those are *not the same
> function*. If the golden model is "approximately" the hardware, you will chase
> off-by-one mismatches in RTL for a week and never close them. Write the
> integer reference first, make PyTorch agree with *it*, not the other way round.

### S1.4 — Weight packing
| Task | Detail | Est |
|---|---|---|
| T1.4.1 | `golden/pack.py`: for each group of 4, emit 2× INT8 values + 2× 2-bit indices; pack into a byte stream the RTL can consume | 0.5d |
| T1.4.2 | Matching unpacker + round-trip property test (random tensors, 1000 iterations) | 0.5d |
| T1.4.3 | Document the exact bit layout in `docs/FORMATS.md` — byte order, index encoding, padding rules | 0.3d |

> Index overhead: 2 bits per stored weight on an 8-bit weight is 25% metadata,
> in exchange for skipping 50% of the multiplies. That ratio is why 2:4 is the
> sweet spot and why 1:4 (75% sparse) is not obviously better — know this number.

### S1.5 — Test vectors
| Task | Detail | Est |
|---|---|---|
| T1.5.1 | `golden/gen_vectors.py`: dump per-layer INT8 inputs, INT8 outputs, **and INT32 pre-requant accumulators** for 20 test images | 0.5d |
| T1.5.2 | Save as `.npy` with a documented naming scheme; add a regeneration make target | 0.3d |

> Dump the INT32 accumulators too, not just the INT8 layer outputs. When RTL
> mismatches, that tells you instantly whether the bug is in the MAC array or in
> the requantizer. Without it you are bisecting blind.

---

## EPIC-2 — The auditor, in software
*Goal: evidence that the confidence signal actually works, with honest baselines.*
**Exit criteria:** AUROC table, risk–coverage plot, a chosen threshold with a written justification.
**Estimate: 5 days**

### S2.1 — Exit head and features
| Task | Detail | Est |
|---|---|---|
| T2.1.1 | Add the exit classifier: per-channel sum over conv2's post-ReLU output (32 values) → FC 32→10 | 0.5d |
| T2.1.2 | Train the exit head with the backbone frozen | 0.5d |
| T2.1.3 | Record early-head top-1 on the full test set (expect well below the full net — that's fine and expected) | 0.2d |

> Because the tap is post-ReLU, the per-channel L1 norm and the per-channel
> global-average-pool are **the same 32 numbers**. One accumulator block serves
> the exit classifier and the auditor. Say this out loud in an interview.

### S2.2 — Train the auditor
| Task | Detail | Est |
|---|---|---|
| T2.2.1 | Decide the target label. Recommended: **"the early head's prediction is correct"** | 0.2d |
| T2.2.2 | `model/auditor.py`: MLP 32→32→16→1, sigmoid, BCE loss, trained on train-set features | 0.5d |
| T2.2.3 | Hold out a calibration split for threshold selection — never pick the threshold on the test set | 0.3d |

> The original plan said the auditor predicts whether *the full network* is
> correct. For early exit that's the wrong target: what you need to know at the
> exit point is whether **the early prediction** can be trusted, because that is
> the one you would emit. Predicting the full net's correctness tells you
> nothing about the accuracy you'd actually ship. Fix this now, before you
> generate any plots.

### S2.3 — Baselines and measurement
| Task | Detail | Est |
|---|---|---|
| T2.3.1 | Implement baselines on the **early head's** logits: max-softmax-probability, max-logit, predictive entropy, energy (logsumexp) | 0.5d |
| T2.3.2 | AUROC for each method (error detection) | 0.5d |
| T2.3.3 | Risk–coverage curve: overall accuracy = early head on exited samples + full net on retained ones, swept over threshold | 0.5d |
| T2.3.4 | Cycle-savings axis on the same sweep: `avg_MACs = c·0.35 + (1−c)·1.0` | 0.3d |
| T2.3.5 | Write the cost argument honestly (see below) | 0.3d |

> **Be rigorous about the cost claim.** "L1 is cheaper than softmax" is only
> half true — max-*logit* needs no exponentials either, and it is the strongest
> cheap baseline. What is actually defensible:
> 1. L1 features exist **before any classifier head runs** — available mid-stream.
> 2. They are a 32-dimensional signal, so a learned auditor can beat any
>    single-scalar statistic derived from 10 logits.
> 3. Softmax, entropy and energy need `exp`/`log` — a CORDIC block or a LUT you
>    would otherwise never build. L1 + a tiny INT8 MLP reuses the MAC array.
>
> If L1 loses on AUROC, report it. "I measured it, here's the number, here's why
> I'd still build it this way" is a far better interview answer than a plot that
> conveniently omits the baseline.
>
> **Stretch:** feed the auditor both the 32 L1 features *and* the 10 early
> logits. Keep the pure-L1 number for the comparison story.

### S2.4 — Quantize the auditor
| Task | Detail | Est |
|---|---|---|
| T2.4.1 | INT8-quantize the exit head and auditor MLP with the same scheme as EPIC-1 | 0.5d |
| T2.4.2 | Right-shift the 32 INT32 channel sums by a fixed amount to land in INT8 range (N=256 elements/channel → `>> 8` is a natural normalization) | 0.3d |
| T2.4.3 | Re-run AUROC and risk–coverage on the quantized auditor; confirm degradation is negligible | 0.3d |
| T2.4.4 | Dump auditor test vectors alongside the layer vectors | 0.2d |

---

## EPIC-3 — Dense INT8 RTL baseline
*Goal: a working, bit-exact, synthesizable accelerator. The long pole.*
**Exit criteria:** bit-exact on all 20 images, all layers; measured cycles/inference; Vivado report with LUT/FF/DSP/BRAM and Fmax.
**Estimate: 25 days** · `risk:high`

### S3.1 — Arithmetic primitives
| Task | Detail | Est |
|---|---|---|
| T3.1.1 | `rtl/common/requant.sv`: INT32 × INT32 → round → arithmetic shift → clamp to INT8 | 1d |
| T3.1.2 | cocotb test: sweep `requant` against `golden/int_ref.py` over 100k random (acc, M0, n) triples — **must be 100% exact** | 1d |
| T3.1.3 | `rtl/common/mac_lane.sv`: signed 8×8 → 16, accumulate into 32 | 0.5d |
| T3.1.4 | `rtl/common/adder_tree.sv`: parameterized P-input signed tree, registered at each level | 1d |
| T3.1.5 | cocotb tests for both against NumPy | 0.5d |

> Do these **before** anything with an FSM in it. They are small, fully
> testable, and every later bug you don't have is one of these you already
> proved. The requantizer in particular is where bit-exactness is won or lost.

### S3.2 — AXI-Stream plumbing
| Task | Detail | Est |
|---|---|---|
| T3.2.1 | `rtl/common/axis_pkg.sv`: interface definition, `tdata`/`tvalid`/`tready`/`tlast` | 0.5d |
| T3.2.2 | A skid buffer (2-deep) so `tready` never combinationally feeds `tvalid` | 1d |
| T3.2.3 | SVA assertions: valid stable until handshake, no data change while stalled, `tlast` framing | 1d |
| T3.2.4 | cocotb stream driver/monitor with randomized backpressure | 1d |

> **Randomized backpressure is not optional.** A design that only works when the
> sink is always ready is a design that will fail the first time it meets real
> hardware, and "how did you test backpressure" is a standard DV interview
> question. Make your monitor deassert `tready` randomly 30% of the time from
> day one.

### S3.3 — Memory and address generation
| Task | Detail | Est |
|---|---|---|
| T3.3.1 | `rtl/mem/line_buffer.sv`: row buffers for 3×3 convolution, with padding handling | 3d |
| T3.3.2 | `rtl/ctrl/addr_gen.sv`: nested counters implementing the `oc/oh/ow/ic/kh/kw` loop nest | 2d |
| T3.3.3 | Ping-pong weight buffer: load tile N+1 while computing tile N | 2d |
| T3.3.4 | cocotb test: address sequence matches a Python generator producing the same nest | 1d |

> Memory is the hard part, not the multipliers. Expect more than half your RTL
> debugging time here. If you internalize one thing from this project for
> interviews, make it this.

### S3.4 — Compute core and layer FSM
| Task | Detail | Est |
|---|---|---|
| T3.4.1 | `rtl/dense/pe_array_dense.sv`: P=16 lanes → adder tree → INT32 accumulator | 2d |
| T3.4.2 | `rtl/ctrl/layer_fsm.sv`: IDLE → LOAD_W → COMPUTE → DRAIN → REQUANT → NEXT_LAYER → DONE | 3d |
| T3.4.3 | `rtl/ctrl/csr.sv`: AXI4-Lite registers — start, done, layer config, base addresses, threshold | 1.5d |
| T3.4.4 | ReLU and 2×2 max-pool in the output path | 1d |
| T3.4.5 | `rtl/top/accel_top.sv` wiring it all together | 1d |

### S3.5 — Bit-exact verification
| Task | Detail | Est |
|---|---|---|
| T3.5.1 | Per-layer cocotb tests: drive layer inputs, compare INT32 accumulators **and** INT8 outputs | 2d |
| T3.5.2 | Full-network test across all 20 images | 1d |
| T3.5.3 | Cycle counter in the testbench; record cycles/inference and MAC utilization | 0.5d |

**Acceptance:** zero mismatches. Not "0.1% off" — zero. If you cannot close the
last mismatch, the bug is almost always rounding mode, sign extension on the
INT8 loads, or the clamp bound (−128 vs −127).

### S3.6 — Synthesis
| Task | Detail | Est |
|---|---|---|
| T3.6.1 | Vivado project scripted in Tcl (no GUI-only state), target `xc7z020clg400-1` | 1d |
| T3.6.2 | Timing constraints; iterate on Fmax, pipeline the critical path | 2d |
| T3.6.3 | Capture LUT/FF/DSP/BRAM + Fmax into `syn/reports/dense.md` | 0.5d |

> Expect the requantizer's multiplier to be your critical path. Pipelining it
> into 2–3 stages is the standard fix and a good thing to have done deliberately
> rather than accidentally.

---

## EPIC-4 — Sparse 2:4 datapath
*Goal: the same results with half the multipliers.*
**Exit criteria:** bit-exact against the same vectors; dense-vs-sparse comparison table.
**Estimate: 10 days**

### S4.1 — Index decode and select
| Task | Detail | Est |
|---|---|---|
| T4.1.1 | `rtl/sparse/index_decode.sv`: unpack 2-bit indices from the weight stream | 1d |
| T4.1.2 | Two 4:1 muxes per group select the activations matching the indices | 1.5d |
| T4.1.3 | cocotb test against `golden/pack.py`'s unpacker | 1d |

### S4.2 — Sparse PE array
| Task | Detail | Est |
|---|---|---|
| T4.2.1 | `rtl/sparse/pe_array_sparse.sv`: P multipliers consuming 2P weights' worth of work per cycle | 2d |
| T4.2.2 | Parameterize so P=8 (iso-latency) and P=16 (iso-area) both build | 1d |
| T4.2.3 | Route conv1 and fc2 through the dense path — they are unpruned | 1d |
| T4.2.4 | FSM changes: different weight-stream bandwidth, different tile counts | 1.5d |

### S4.3 — Verify and compare
| Task | Detail | Est |
|---|---|---|
| T4.3.1 | Bit-exact against the **same** 20-image vectors | 1d |
| T4.3.2 | Synthesize both P=8 and P=16 sparse builds | 1d |
| T4.3.3 | Comparison table: cycles, DSP, LUT, FF, BRAM, Fmax across dense P=16 / sparse P=8 / sparse P=16 | 0.5d |

> **Expect LUT count to go up.** The muxes, index decode and wider weight
> fetch all cost logic, and you are trading DSPs for LUTs. Predicting that
> trade-off before you measure it, then showing the measurement, is one of the
> two or three strongest things you will have to talk about. Also watch Fmax —
> the mux stage adds combinational delay in front of the multiplier.

---

## EPIC-5 — Auditor in hardware and early exit
*Goal: the control logic that makes latency data-dependent.*
**Exit criteria:** hardware exit decisions match the software auditor on every test image; average cycles/inference reported.
**Estimate: 10 days** · `risk:high` on S5.3

### S5.1 — L1 accumulator
| Task | Detail | Est |
|---|---|---|
| T5.1.1 | `rtl/audit/l1_accum.sv`: 32 INT32 accumulators, one per channel, tapped off the conv2 output stream | 1.5d |
| T5.1.2 | Confirm it runs fully concurrently — assert cycle count is identical with the tap enabled and disabled | 0.5d |
| T5.1.3 | Fixed right-shift normalization to INT8 | 0.5d |
| T5.1.4 | cocotb test against the software features | 0.5d |

> Since the tap is post-ReLU, there is no absolute-value stage at all — the
> inputs are already non-negative. If you ever move the tap before the ReLU you
> need `abs()`, which for INT8 is an XOR-and-increment, and you need to handle
> `−128` having no positive counterpart. Mention this; it shows you thought
> about the boundary.

### S5.2 — Auditor MLP and exit head in hardware
| Task | Detail | Est |
|---|---|---|
| T5.2.1 | Decide: reuse the main PE array, or a dedicated 4-MAC unit. **Recommend dedicated** — 1,872 MACs is nothing, and sharing the array entangles the FSM | 0.5d |
| T5.2.2 | `rtl/audit/auditor_mlp.sv`: 32→32→16→1 with the same requantizer | 2d |
| T5.2.3 | Exit classifier 32→10 sharing the same unit | 1d |
| T5.2.4 | cocotb test against quantized software auditor, bit-exact | 1d |

> The "reuse the array" option sounds more impressive and is the wrong call. It
> costs you ~1,900 MACs of savings and buys a large FSM complication right where
> your control logic is already hardest. Being able to explain *why you didn't*
> is better engineering than doing it.

### S5.3 — Exit control
| Task | Detail | Est |
|---|---|---|
| T5.3.1 | `rtl/audit/exit_ctrl.sv`: compare auditor output to the CSR threshold | 0.5d |
| T5.3.2 | FSM branch: emit early result with `tlast`, or continue into conv3 | 2d · `risk:high` |
| T5.3.3 | Handle the weight-prefetch decision — conv3 weights may already be streaming when the exit fires. Choose: stall until decided, or speculate and discard | 1.5d |
| T5.3.4 | Output framing so a downstream consumer tolerates variable latency | 1d |
| T5.3.5 | cocotb test: decisions match software on all test images; measure average cycles across the set | 1d |

> **T5.3.3 is the most interesting hardware question in the whole project.**
> Weights for conv3 come from off-chip. If you wait for the auditor before
> starting that fetch, every non-exiting inference pays the full memory latency
> as a bubble. If you prefetch speculatively, exiting inferences waste memory
> bandwidth and power. There is no free answer — there is a *measured* answer.
> Build the stall version first because it is simpler and correct, measure the
> bubble, then decide whether the prefetch is worth it. An interviewer who hears
> "I measured the bubble at N cycles and it was 4% of average latency so I left
> it alone" will take you seriously.

---

## EPIC-6 — Measurement and writeup
*Goal: turn a pile of RTL into a portfolio project.*
**Exit criteria:** README with plots, reproducible test command, all tables filled.
**Estimate: 5 days**

| Task | Detail | Est |
|---|---|---|
| T6.1 | Average cycles/inference: dense, sparse, sparse + early exit | 0.5d |
| T6.2 | **Headline plot:** accuracy vs coverage vs average cycles, swept over threshold | 1d |
| T6.3 | Resource + Fmax table across all configurations | 0.5d |
| T6.4 | Bit-exactness statement: what was compared, how many vectors, what "exact" means here | 0.5d |
| T6.5 | Auditor vs MSP / max-logit / entropy / energy, with the cost argument | 0.5d |
| T6.6 | Architecture diagram (block diagram + dataflow) | 0.5d |
| T6.7 | `make reproduce` — one command from checkout to all results | 1d |
| T6.8 | Rewrite the README as the project's front page | 0.5d |

> The headline plot is the one that makes this a project rather than an
> exercise: it shows latency is a **tunable knob**, not a fixed number. Put it
> at the top of the README and put the README link on your resume.

---

## EPIC-7 — Stretch (only after EPIC-6 is done)

| Story | Why it's worth it | Why it's stretch |
|---|---|---|
| S7.1 — Deploy on a Pynq-Z2, Python host driver | An actual demo beats any report | Board bring-up is its own multi-week project |
| S7.2 — Power estimate from Vivado + switching activity | Completes the efficiency story | Estimates without silicon are soft numbers |
| S7.3 — Sweep N:M (1:4, 2:8) | Shows the design space, not one point | Requires retraining and repacking each variant |
| S7.4 — Formal property check on the AXI-Stream interfaces | Formal on your resume is rare and valuable | New toolchain, new mental model |

Resist these until EPIC-6 is closed. A finished project with three configurations
beats an unfinished one with eight.
