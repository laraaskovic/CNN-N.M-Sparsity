# Build Guide: The Order To Actually Do This In

The backlog says *what*. This says *what order*, *what to do when stuck*, and
where the project is most likely to die.

---

## The three rules

**1. The golden model comes before the RTL, always.**
Every RTL module gets a Python reference and a cocotb test written *before or
alongside* the module. Not after. The moment you write RTL without a checker for
it, you're accumulating unverified code, and unverified code is where projects go
to die — you'll have 2,000 lines that "should work" and no way to find the one
that doesn't.

**2. Bit-exact or not done.**
"99.8% of outputs match" is not a partial pass. It is a bug you haven't found
yet, and it will be a rounding mode, a sign extension, or a clamp bound. Close
it before moving on. Every time.

**3. Build the smallest thing that can be verified, then grow it.**
Do not write a conv engine as your first RTL. See the staircase below.

---

## The staircase (this is the important part)

EPIC-3 is where this project fails if it's going to. The reason is always the
same: people try to build a convolution accelerator as their first real RTL.
Convolution combines an address generator, a line buffer, a multi-cycle
accumulate, a requantizer and an FSM — five hard things at once, and when it
doesn't work you cannot tell which one is wrong.

Climb instead. Each step is verified bit-exact before the next begins:

| Step | What you build | New hard thing | Est |
|---|---|---|---|
| **3a** | `requant.sv` alone. INT32 in, INT8 out. No clock needed at first. | Fixed-point rounding | 2d |
| **3b** | One MAC lane. INT8 × INT8 → INT32 accumulate over N cycles. | Sequential accumulation | 1d |
| **3c** | P=16 lanes + adder tree → one dot product. Still no memory. | Parallelism, pipelining | 2d |
| **3d** | Wrap 3c in AXI-Stream with randomized backpressure. | Handshake, skid buffer | 3d |
| **3e** | **An FC layer.** fc2 (64→10). Weights in a small BRAM, simple counters. | Weight storage, a real FSM | 4d |
| **3f** | fc1 (2048→64). Same structure, needs tiling. | Tiling, ping-pong buffers | 3d |
| **3g** | **1×1 "conv"** — i.e. conv with kh=kw=1. Spatial loop, no line buffer. | Spatial address generation | 3d |
| **3h** | 3×3 conv with a line buffer and padding. | Line buffer, halo/padding | 5d |
| **3i** | ReLU + 2×2 max-pool in the output path. | Output-side processing | 1d |
| **3j** | Multi-layer sequencing FSM, full network. | Layer orchestration | 4d |

**Step 3e is the milestone that matters.** A working, bit-exact, synthesizable
fully-connected layer with an AXI-Stream interface is *already* a legitimate
portfolio artifact. If everything after it goes wrong, you still have something
real to show and talk about. Get there before you do anything else.

Note that 3a–3e is ~12 days of the 25-day EPIC-3 estimate, and it's the part
where you learn the most per hour.

---

## Week-by-week, realistically

The original plan's 10 weeks assumes RTL fluency. Here's a de-risked schedule
assuming you're learning the hardware as you go. Adjust freely — the *order*
matters more than the dates.

| Weeks | Focus | Deliverable you could show someone |
|---|---|---|
| 1 | EPIC-0: repo, toolchain, train baseline, BN fold, MAC table | A reproducible baseline |
| 2–3 | EPIC-1: prune, quantize, **golden model**, packing, vectors | Integer-only NumPy inference matching PyTorch |
| 3–4 | EPIC-2: exit head, auditor, AUROC, risk–coverage | The risk–coverage plot |
| 4–5 | Steps 3a–3d: arithmetic primitives + AXI-Stream | Bit-exact requantizer and dot product |
| 6–7 | Steps 3e–3f: **FC layers working end to end** | **A real accelerator, however small** |
| 8–9 | Steps 3g–3i: convolution | Bit-exact conv layers |
| 10 | Step 3j + EPIC-3 synthesis | Dense baseline, full numbers |
| 11–12 | EPIC-4: sparse datapath, both P configurations | The dense-vs-sparse table |
| 13–14 | EPIC-5: L1 tap, auditor MLP, exit control | Variable-latency inference |
| 15 | EPIC-6: measurement, plots, README | The finished project |

Run EPIC-2 in gaps and evenings — it's Python, it's the part you're already good
at, and it doesn't block the RTL. Do **not** run EPIC-1 in parallel with RTL:
the golden model must be frozen before RTL verification starts, or you'll be
chasing a moving target.

---

## Checkpoints where you should stop and be honest

**After EPIC-1.** Can your integer-only NumPy model classify a CIFAR-10 image
with no floating-point operation anywhere in the forward pass, matching
quantized PyTorch to within 0.3%? If not, do not start RTL. Everything
downstream is verified against this.

**After step 3e.** Is the FC layer bit-exact under randomized backpressure? If
you're fighting it, the problem is probably in your understanding of the AXI
handshake or your address-generator-vs-BRAM-latency alignment. Both are worth
slowing down for. They recur in every later module.

**After EPIC-3.** Do you have a synthesis report? If you've never run Vivado,
budget two full days for the first one. Scripted from Tcl, not the GUI — you
will need to re-run it four more times for the other configurations.

**After EPIC-4.** Did LUT count go up while DSP count went down? If DSPs did not
roughly halve, your sparse array isn't actually exploiting the sparsity — check
that the pruning axis matches the reduction axis (T1.1.1). This is the single
most likely conceptual error in the whole project.

---

## Failure modes, and what they actually mean

| Symptom | Almost always |
|---|---|
| RTL output is off by ±1 from golden | Rounding mode. You rounded half-up, the reference rounds half-even, or the `+ (1 << (n-1))` is missing/misplaced |
| Everything is shifted by one pixel | Address generator vs BRAM read latency misalignment |
| Large negative values wrong, positives fine | Sign extension on the INT8 load, or `>>` instead of `>>>` |
| First and last row/column wrong | Padding handling in the line buffer |
| Works standalone, fails when connected | Backpressure. Your module only works when `tready` is always 1 |
| Simulation hangs | FSM stuck in a state waiting for a condition that can't occur. Add a timeout and a state-trace to the testbench |
| Vivado Fmax much lower than expected | Critical path through the requant multiplier, or through the sparse mux. Pipeline it |
| `-128` produces wrong results | INT8 min has no positive counterpart. Your `abs()` or negation overflows |
| Sparse gives no DSP saving | Pruning axis ≠ reduction axis |

Add a `TROUBLESHOOTING.md` as you go and record each real bug with its symptom
and root cause. That file is genuinely useful in an interview — "here are the
nine bugs I found and how I found them" is a concrete demonstration of debugging
skill, which is most of what the job is.

---

## Things the original plan gets subtly wrong, fixed here

**The auditor's target.** The plan says predict whether *the full network* is
correct. For early exit, the useful question is whether **the early head's
prediction** is correct, because that's the prediction you'd emit. Fix in S2.2.

**The exit head is missing.** "Emit the early prediction" requires something that
produces a prediction at the exit point. The plan doesn't mention training one.
Added in S2.1 — and it's nearly free, because it shares the L1 accumulator.

**"Halve the array" needs a clearer claim.** If you halve the array, cycle count
stays the *same* as dense and DSP count halves. That's an **area** win, not a
latency win, and stating it as "2× faster" would be wrong. Parameterize by `P`
and report both configurations (EPIC-4) so you can say "iso-latency at half the
area, or iso-area at 2× throughput" — which is the honest and more impressive
framing.

**Exit-point placement changes the whole story.** Exit after conv3 in a
conventional CIFAR CNN and you'd have already done ~85% of the work, so exiting
saves almost nothing. The network in the README is deliberately sized so the
three conv stages have identical MAC counts and the exit lands at 35%. Choose the
architecture to make the mechanism meaningful — that itself is a design decision
worth talking about.

**"Run the auditor on the same array."** Tempting, wrong. 1,552 MACs is 0.01% of
the network; a dedicated unit costs almost nothing and keeps the FSM tractable
right where control is already hardest. Choosing simplicity deliberately is a
better answer than the complicated one.

---

## Keeping the repo interview-ready as you go

Someone will open this repo for ninety seconds before deciding whether to talk to
you. Optimize for that.

- **README front page:** the headline plot (accuracy vs coverage vs cycles), the
  resource table, and one sentence on bit-exactness. Above the fold.
- **`make reproduce`** from a clean checkout. If a reviewer can't run it, they
  assume it doesn't work.
- **CI badge that means something.** Green = the cocotb tests pass.
- **Commit history that tells a story.** Small commits with real messages. A
  single "initial commit" dump of 5,000 lines reads as copied.
- **Write down the numbers you didn't like.** AUROC lower than max-softmax? Put
  it in the table with your reasoning. Utilization only 70%? Say so and explain
  where the stalls are. Nothing signals competence faster than reporting a
  number that isn't flattering, and nothing signals the opposite faster than a
  results section where everything conveniently won.
