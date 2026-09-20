# The Hardware Half, for Someone Who Knows PyTorch

You already understand the network. This document is the other side: what RTL
actually is, what changes when you leave Python, and what each hardware block in
this project is really doing. Read it once before EPIC-3, and again during it.

---

## 1. The single mental shift

In PyTorch you write:

```python
y = conv(x)
```

and the framework decides everything else — how many multiplies happen at once,
what order data is read in, where intermediates live.

In RTL, **you decide all of that, and the math is the easy part.** The
convolution is the same convolution. What you are designing is:

- **How many multipliers physically exist.** Not "how many can run" — how many
  are etched into the device. In this project: 8 or 16. That's it.
- **What order data arrives in.** There is no random access to a tensor. There
  is a stream, and a small local memory, and you get one or two reads per cycle.
- **Where partial sums live**, and for how long.
- **When each thing happens**, cycle by cycle.

A useful reframing: your PyTorch layer is a six-deep loop nest.

```python
for oc in range(C_out):
  for oh in range(H):
    for ow in range(W):
      acc = 0
      for ic in range(C_in):
        for kh in range(3):
          for kw in range(3):
            acc += w[oc,ic,kh,kw] * x[ic, oh+kh-1, ow+kw-1]
      y[oc,oh,ow] = requantize(acc)
```

**Hardware design is choosing which of those loops to unroll into parallel
hardware, which to turn into counters, and how to keep the operands arriving.**
Unroll `ic` by 16 → 16 multipliers. Turn `oc/oh/ow` into counters → that's your
address generator. The rest is making sure the data shows up on time.

---

## 2. What RTL actually is

Verilog/SystemVerilog is **not** a programming language that runs. It *describes
structure*. Two kinds:

**Combinational logic** — gates. Exists continuously, no memory. `assign y = a & b;`
Whenever `a` or `b` changes, `y` follows after a propagation delay.

**Sequential logic** — registers (flip-flops). Sample their input on a clock edge
and hold it until the next one.

```systemverilog
always_ff @(posedge clk) begin
  if (!rst_n) acc <= '0;
  else if (en) acc <= acc + product;
end
```

That is one physical bank of flip-flops, updating once per clock tick. Not a
loop, not a statement that executes — a thing that exists.

The whole design is **registers separated by clouds of combinational logic.**
Every clock cycle, signals leave one register bank, propagate through logic, and
must be stable at the next register bank before the next edge.

### Why Fmax exists

If the slowest path between two registers takes 8 ns, the clock period must be
at least 8 ns → **125 MHz maximum**. That path is the **critical path**. Timing
closure means finding it and shortening it.

The standard fix is **pipelining**: insert a register in the middle of the long
path. Now you have two 4 ns paths instead of one 8 ns path → 250 MHz. You pay
one extra cycle of latency, and you have to make sure everything else in the
design accounts for that extra cycle. This is why a "1 cycle" block becomes a
"3 cycle" block and why your FSM suddenly has off-by-one bugs.

> In this project, the requantizer's 32×32 multiplier will almost certainly be
> your critical path. Plan to pipeline it.

---

## 3. Numbers: why INT8

An FP32 multiplier is a large, slow piece of logic — mantissa multiply,
exponent add, normalize, round. An INT8×INT8 multiplier is tiny and fast.

On a Xilinx 7-series FPGA the relevant resource is the **DSP48 slice**: a
hardened block that does a 25×18 signed multiply plus an accumulate. An INT8
multiply fits inside one with room to spare. A Zynq-7020 has 220 of them. That
number is your budget, and it is why the array is 16 wide and not 1024 wide.

Resources you will report:
- **LUT** — look-up tables, generic combinational logic
- **FF** — flip-flops, your registers
- **DSP** — hard multipliers
- **BRAM** — block RAM, 36 Kbit each, on-chip memory

Trading one for another is the core craft. 2:4 sparsity trades **DSPs for LUTs**:
fewer multipliers, more mux and index logic. That's the trade-off you'll measure
in EPIC-4.

---

## 4. Quantization as the hardware sees it

Software quantization is `x_int = round(x_float / s)`. Hardware never sees `s`
at all during compute. The datapath is:

```
INT8 weight  ─┐
              ├─► signed multiply ─► INT16 product ─┐
INT8 activ.  ─┘                                     │
                                                    ├─► adder tree ─► INT32 accumulator
                            (P lanes in parallel)   │
                                                    ┘
```

Then, once per output value, **requantization** brings INT32 back to INT8:

```
acc (INT32) ──► × M0 (INT32) ──► + (1 << (n-1)) ──► >>> n ──► clamp[-128,127] ──► INT8
```

### Where M0 and n come from

Mathematically you need to multiply the accumulator by
`M = (s_w · s_x) / s_y`, a real number, typically in (0, 1). Hardware has no
floats. So you decompose:

```
M = M0 · 2^-n    with M0 an INT32 in [2^30, 2^31)
```

Normalizing `M0` into that top range keeps maximum precision. Then multiplying
by `M` is an integer multiply followed by an arithmetic right shift by `n`.
Adding `1 << (n-1)` before the shift implements round-half-up.

**Per-channel weight scales** mean each output channel has its own `(M0, n)`
pair. Store them in a small ROM/BRAM indexed by output channel. This is exactly
why per-channel weights are cheap in hardware but per-channel *activations*
would be expensive — activation scale is a property of the whole tensor flowing
through, and making it per-channel means a scale lookup on every input, not
every output.

### Why symmetric quantization matters to you

With zero-points, the accumulator must compute:

```
Σ (w - z_w)(x - z_x) = Σ w·x  −  z_x·Σw  −  z_w·Σx  +  N·z_w·z_x
```

Three correction terms, one of which (`Σx`) depends on the input data and must
be computed at runtime. With symmetric quantization (`z = 0`) all of it
vanishes and the datapath is `Σ w·x`. You delete a hardware block by making a
software choice. **This is what hardware/software co-design means**, and it is
the kind of thing an interviewer is actually probing for.

---

## 5. AXI-Stream: the interface you must get right

Nearly every FPGA data path uses AXI-Stream. Four signals:

| Signal | Direction | Meaning |
|---|---|---|
| `tdata` | source → sink | the payload |
| `tvalid` | source → sink | "the data on `tdata` is real" |
| `tready` | sink → source | "I can accept data this cycle" |
| `tlast` | source → sink | "this is the final beat of the packet" |

**A transfer happens on a rising clock edge when `tvalid` AND `tready` are both
high.** That's the whole protocol. Two rules that people get wrong:

1. **`tvalid` must not depend combinationally on `tready`.** If it does, you can
   create a combinational loop between two connected modules, and the design
   either won't meet timing or won't simulate deterministically. The fix is a
   **skid buffer**: a 2-deep register stage that absorbs one beat when
   downstream stalls, letting `tvalid` be registered.
2. **Once `tvalid` is asserted it must stay asserted, with `tdata` unchanged,
   until the handshake completes.** You cannot offer data and then withdraw it.

> "Explain the AXI-Stream handshake and why `tvalid` can't depend on `tready`"
> is close to a guaranteed interview question for any FPGA or DV role. Being
> able to also say "so I put a skid buffer here" is the answer they want.

You'll also use **AXI4-Lite** for control: a simple address/data register
interface where software writes `start`, reads `done`, and sets configuration —
including your exit threshold. That register block is the CSR.

---

## 6. Memory is the real problem

This surprises people coming from software, so internalize it early: **the
multipliers are not the hard part.** In a 16-MAC array you need 16 activations
and 16 weights every single cycle. Getting them there is the design.

**BRAM** is dual-port and has ~1 cycle of read latency (2 with the output
register, which you often need for timing). So a read issued in cycle `t` gives
data in cycle `t+1` or `t+2`. Your address generator must run *ahead* of your
compute by exactly that much. Off-by-one here is the most common source of
"everything is shifted by one pixel" bugs.

**Line buffers.** A 3×3 conv at output row `r` needs input rows `r-1, r, r+1`.
You do not re-read them from external memory three times — you keep 2–3 rows on
chip in a shift-register-of-rows and slide a 3×3 window across. This is the
single most important structure in a conv accelerator, and building one is most
of T3.3.1.

**Ping-pong buffering.** Two weight buffers. Compute from buffer A while loading
buffer B from memory. Swap. Without this, compute stalls every time you switch
tiles, and your MAC utilization — the number you report — collapses.

**MAC utilization** is your honesty metric: `MACs / (cycles × P)`. If you have 16
multipliers and 920K theoretical cycles but the design takes 1.4M cycles, your
utilization is 66% and the missing third is memory stalls. Reporting that number
rather than hiding it is what separates a credible project from a hopeful one.

---

## 7. The FSM: the loop nest in silicon

Your layer sequencer is a state machine:

```
IDLE ──start──► LOAD_WEIGHTS ──► COMPUTE ──┬──► DRAIN ──► REQUANT ──► WRITE_OUT
                     ▲                     │
                     └──── next tile ──────┘
                                                    │
                                     more layers? ──┴──► DONE
```

The nested counters inside `COMPUTE` **are** the `oc/oh/ow/ic/kh/kw` loops. A
counter that wraps and increments the next one up is a `for` loop. There is no
call stack, no recursion, no dynamic allocation — just counters and comparators.

Debugging an FSM means looking at waveforms and asking "what state was I in when
this went wrong, and what was the counter?" This is the skill that takes the
longest to acquire and it is the one that makes you employable.

---

## 8. How 2:4 sparsity works in hardware

The weights are pruned so that in every group of 4 consecutive weights along the
reduction axis, exactly 2 are non-zero. You store:

- 2 × INT8 values
- 2 × 2-bit indices saying which of the 4 slots they came from

```
Dense group:   w = [  0, -12,   0,  45 ]   4 multipliers, 2 of them × 0
Sparse group:  vals = [-12, 45], idx = [1, 3]   2 multipliers, both useful
```

The hardware fetches **all 4 activations** for that group (you can't skip
activations — you don't know in advance which ones you'll need), then two 4:1
muxes select `x[idx[0]]` and `x[idx[1]]`, and two multipliers do the work that
four did.

```
x[0] x[1] x[2] x[3]
  │    │    │    │
  └────┴────┴────┴──► 4:1 mux ──(idx0)──► ×  vals[0] ─┐
  └────┴────┴────┴──► 4:1 mux ──(idx1)──► ×  vals[1] ─┴─► adder tree
```

**What you gain:** half the multipliers for the same work, or twice the work
from the same multipliers.

**What you pay:**
- 2 bits of index per stored weight — on an 8-bit weight that's **25% metadata
  overhead** to skip 50% of the multiplies. That ratio is why 2:4 is the chosen
  point in the design space.
- Mux logic and index decode → **more LUTs**.
- The mux sits in front of the multiplier → **added combinational delay**,
  so watch Fmax.
- Activation fetch bandwidth is unchanged — you still read all 4.

**Why *structured* sparsity at all?** Unstructured sparsity (prune any 50% of
weights) gives better accuracy at the same sparsity level, but the hardware
cannot exploit it: you'd need variable-length encodings, dynamic scheduling, and
load-balancing across lanes, because one lane might get 5 non-zeros and another
0. 2:4 guarantees **every group takes exactly the same time**. Fixed latency,
fixed bandwidth, no load balancing, trivial control. You give up some accuracy
to make the hardware regular. That is the entire argument, and it's the argument
NVIDIA made when they put 2:4 sparse tensor cores in Ampere.

---

## 9. The auditor and early exit in hardware

**The L1 tap.** As conv2's output streams past, accumulate a running sum per
channel — 32 INT32 registers and an adder. It runs *concurrently* with the
stream, so it costs **zero extra cycles**. Because the tap is after a ReLU,
every value is non-negative, so the L1 norm and the sum are identical, and the
same 32 numbers feed both the early classifier and the auditor. One block, two
consumers.

**The MLP.** 32→32→16→1, about 1,552 MACs. Run it on a small dedicated unit with
the same requantizer. Compare the output to a threshold held in a CSR.

**The interesting part is the control.** When the auditor says "confident", the
FSM must:

1. emit the early prediction with `tlast` and return to IDLE, **or**
2. continue into conv3.

And here is the real engineering question, the one worth thinking hard about:
**conv3's weights come from off-chip memory, and that fetch has latency.** So:

- **Stall until the auditor decides** → simple and correct, but every
  non-exiting inference eats a memory-latency bubble.
- **Speculatively prefetch conv3's weights** during the auditor's few cycles →
  no bubble, but every exiting inference wasted bandwidth and power.

There is no free answer. There is a *measured* answer. Build the stall version
first, measure the bubble, then decide whether the prefetch earns its complexity.

**And a consequence you must handle:** latency is now **data-dependent**. Two
different images take different numbers of cycles. That breaks the assumption of
every fixed-latency testbench you've written, and it means downstream consumers
must be framing-aware (`tlast`, not a cycle count). It also makes verification
harder — you can no longer say "the answer appears at cycle 920,000". Say this
to an interviewer; it shows you understand that variable latency is a system
property, not just a local optimization.

---

## 10. Verification: the part that is actually the job

In software you write a test and it passes or fails. In hardware verification,
the discipline is:

1. **A golden model** — an independent implementation, ideally written from the
   spec rather than translated from the RTL, that produces the exact expected
   output. Yours is `golden/int_ref.py`.
2. **Stimulus** — including randomized and adversarial cases, not just the happy
   path. Random backpressure. Minimum and maximum values. `-128`, which has no
   positive counterpart in INT8.
3. **Checkers** — automatic comparison, never eyeballing waveforms as the pass
   criterion. Waveforms are for *debugging* a known failure.
4. **Assertions (SVA)** — properties that must hold at all times, checked
   continuously during simulation. "`tvalid` never drops before a handshake."
5. **Coverage** — did you actually exercise every state, every layer
   configuration, every boundary?

**Bit-exact** means every single output value matches, on every vector. Not
99.9%. If you accept 99.9% you have accepted that you don't know what's wrong,
and the remaining 0.1% is always a real bug — a rounding mode, a sign extension,
a clamp bound.

This discipline *is* the job description for a Design Verification engineer, and
DV is where most of the semiconductor hiring is. A project where you can say "I
built a golden model, verified bit-exactness across 20 images and 6 layers with
randomized backpressure, and here are the three bugs it caught" is worth more in
an interview than a larger design with hand-waved testing.

---

## 11. Tooling

| Need | Tool | Note |
|---|---|---|
| RTL | SystemVerilog | Not VHDL — SV dominates in US/industry ML-hardware roles |
| Simulation | **Verilator** | Free, compiles to C++, fast enough for full-network sims |
| Testbench | **cocotb** | Python testbenches → your NumPy golden model is directly importable. This is the single biggest reason this project is tractable for you |
| Lint | `verilator --lint-only`, Verible | Catches the width-mismatch class of bug before simulation |
| Waveforms | GTKWave / Surfer | For debugging failures you already detected |
| Synthesis | **Vivado** (free edition) | Target `xc7z020clg400-1` (Pynq-Z2) |
| CI | GitHub Actions | Verilator + cocotb install via apt; run on every push |

You do **not** need a physical board. Synthesis reports give you LUT/FF/DSP/BRAM
and Fmax, which is all the resource story needs. A board is a nice stretch goal
(EPIC-7), not a requirement.

> **cocotb is the key enabler here.** Your golden model is NumPy. Your testbench
> is Python. You `import` the golden model directly into the testbench and
> compare arrays with `np.array_equal`. Without cocotb you'd be writing SystemVerilog
> testbenches and dumping hex files, and this project would take twice as long.

---

## 12. Vocabulary you'll be expected to use correctly

| Term | Meaning |
|---|---|
| RTL | Register Transfer Level — describing logic as registers + the logic between them |
| Critical path | The slowest combinational path; sets Fmax |
| Setup / hold | Data must be stable before (setup) and after (hold) the clock edge |
| Pipelining | Adding registers to shorten paths; more throughput, more latency |
| Latency vs throughput | Cycles for one result vs results per cycle. **Different things.** |
| Backpressure | A downstream block saying "stop, I'm full" |
| Skid buffer | 2-deep buffer that lets `tvalid` be registered under backpressure |
| Systolic array | Grid of MACs passing operands neighbour-to-neighbour |
| Weight-stationary | Dataflow where weights stay put and activations move |
| Output-stationary | Accumulator stays put, weights and activations stream |
| CDC | Clock domain crossing — moving signals between clocks safely |
| Utilization | Fraction of cycles your multipliers do useful work |
| Golden model | Independent reference implementation used as truth |
| Bit-exact | Every output bit matches the reference |
