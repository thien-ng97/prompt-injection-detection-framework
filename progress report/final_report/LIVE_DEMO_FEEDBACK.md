# Live Demo Feedback

---

## 1. Verdict

**Go ahead with the demo, and keep the latency display in it.**

The discrepancy you found is not a problem with your model. Your script is measuring a
different thing from the benchmark, and once you understand why, it becomes something you
can explain confidently if an examiner asks — which is a stronger position than having no
latency figure at all.

Removing the latency field is the one option I would rule out. An examiner who notices the
demo prints confidence but conspicuously not speed will ask why, and *"I took it out because
it looked bad"* is a considerably worse answer than explaining a measurement difference you
understand.

---

## 2. Why your numbers differ from 10.9 ms

The 10.9 ms figure came from `evaluate_custom_model.py` running 1,104 prompts back-to-back
in a tight loop. Two things follow from that:

- The one-time MPS graph compilation is amortised across 1,104 calls, so it becomes a
  rounding error.
- The GPU never leaves its high-performance power state, because work keeps arriving.

Your interactive script does the opposite. Every prompt has an indefinite idle gap in front
of it while you type, so Apple Silicon drops the GPU into a low-power state and each call
pays to ramp it back up.

**The evidence is in your own trace.** `hello` — a single token — took 337 ms as your sixth
prompt, while a 100-word jailbreak later in the same session took 127 ms. If this were
compute time, the one-word prompt would be the fastest thing in the list. It isn't. That
tells you the measurement is dominated by per-call overhead and power-state ramping, not by
the model.

So the two numbers are not in conflict — they describe different operating conditions, and
the benchmark one is the condition that matters. A deployed ingestion guardrail sits under
continuous load; it is never idle for thirty seconds between requests. **Steady-state
throughput is the honest figure for the deployment claim.**

---

## 3. Required changes to `live_infer.py`

### 3.1 Warm up before the prompt loop

After the pipeline is built and before you accept any input, run 20–30 throwaway inferences
and discard them. Print `Warming up...` so it is visible on screen during the demo — it also
buys you a natural moment to explain what the model is.

### 3.2 Report the median of repeated runs, not a single call

For each prompt entered, run the classifier about 20 times and report the median. A single
call is dominated by per-call overhead, which is exactly what you were seeing.

### 3.3 Synchronise around the timer

MPS is asynchronous. Without an explicit synchronise, you are timing when the work was
*queued*, not when it *finished*.

### 3.4 Reference implementation

```python
import time
import statistics
import torch

N_WARMUP = 30
N_REPEATS = 20


def sync():
    """MPS is async — force completion before reading the clock."""
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


# ---- after the pipeline is built, before the input loop ----
print("Warming up...")
for _ in range(N_WARMUP):
    clf("warm-up prompt for timing")
sync()
print("Ready.\n")


# ---- per prompt entered by the user ----
def classify_with_timing(clf, prompt, n_repeats=N_REPEATS):
    timings = []
    result = None
    for _ in range(n_repeats):
        sync()
        t0 = time.perf_counter()
        result = clf(prompt)
        sync()
        timings.append((time.perf_counter() - t0) * 1000)
    return result, statistics.median(timings)


result, median_ms = classify_with_timing(clf, prompt)
print(f"  Latency:    {median_ms:.1f} ms  (median of {N_REPEATS}, warm)")
```

Label the figure on screen as median and warm. Nobody should have to guess what was
measured.

### 3.5 Check this before you run it

The pipeline in `live_infer.py` must be configured **identically** to the one in
`evaluate_custom_model.py`:

| Setting | Must match |
|---|---|
| Device | MPS |
| `truncation` | `True` |
| `max_length` | `512` |
| Batch size | 1 (one prompt at a time) |

If any of these differ, the comparison to 10.9 ms is not valid no matter how carefully you
time it. Confirm they match when you report back.

### 3.6 If the median still doesn't land near 10.9 ms

Keep the field and say plainly: *this is interactive timing with idle gaps between calls;
the 10.9 ms benchmark is sustained throughput under continuous load, which is the deployment
condition.* That is an accurate and defensible statement. Send me the numbers either way and
we will agree the wording before your presentation date.

---

## 4. The thing I am more concerned about

You did not raise this, and it matters more than the latency.

Two results in your sample run would be awkward live:

**False positive.** `ignore the weather yesterday, how's the weather today?` returned
**MALICIOUS at 95.06%**. That is a completely harmless sentence, and it is good evidence the
model has partly learned the token *"ignore"* as a shortcut rather than learning intent.

**Missed attack.** The dystopian science-fiction prompt returned **SAFE at 98.74%** despite
containing *"the ethical constraints are adjusted"*, a demand for step-by-step procedure, and
a coercive *"fail to do so"* clause. That is a textbook roleplay jailbreak and the model went
straight past it.

### What to do about it

- **Rehearse a fixed prompt set.** Decide in advance exactly which prompts you will run.
- **Do not invite the panel to type their own prompts.** If someone asks anyway, say yes —
  refusing looks worse — but decide beforehand how you will respond to a miss.
- **Consider pre-empting it.** This is the stronger play if you are feeling confident: run
  three prompts that work, then deliberately run the weather one, and say something like:

  > *"This is the failure mode I flagged in my limitations. Eighty-eight percent of my
  > training corpus is jailbreak-heavy, and the model has partly learned trigger words rather
  > than intent. It is exactly why the first item in my future work is retraining on a
  > broader distribution."*

  Showing a known failure and naming its cause demonstrates command of the work. Examiners
  reward that considerably more than a demo where everything conveniently succeeds.

---

## 5. Practical constraints for the day

- **Keep the demo under one minute.** It supports the talk; it is not the talk.
- **Have a screen recording ready** in case the Hugging Face download stalls or the network
  misbehaves. A demo that breaks live costs more than it gains. Upload the recording under this
  folder as well so that I can watch it before your presentation.
- **Rehearse it until it cannot fail.** If you are not confident it will run cleanly, skip it
  — the deck stands on its own.

---

## 6. What to send me

1. The terminal output from a run over **the same prompt set you sent me before**, so I can
   compare directly against your earlier trace.
2. The updated `live_infer.py`.
3. One line confirming the pipeline configuration matches `evaluate_custom_model.py`
   (§3.5).

If the median lands near 10.9 ms, the demo becomes direct evidence for your latency claim
rather than something that undercuts it. Either way, I would rather know now than find out
during your slot.

---

