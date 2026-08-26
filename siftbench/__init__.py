"""SiftBench — a small, scored benchmark for Sift's analysis pipeline.

See ``cases.py`` for what
"scored" means here and ``runner.py`` for how a case is executed and
graded.

Honest scope note
------------------
"SiftBench" as originally imagined would grade a live frontier model:
hand it a natural-language research question, let it drive
``submit_script`` (and everything else) autonomously the way a real
session would, and score what it eventually reports back to the
researcher. That is a genuine, valuable thing to build, and this seed
deliberately leaves room for it — every :class:`~siftbench.cases.BenchCase`
already carries a ``prompt`` field written as if it were being handed
to a model, not just a human maintainer.

It is NOT what this seed runs today, for two honest reasons:

1. This development environment has no live model connection (no
   network egress to Anthropic/OpenAI/a local server) to drive that
   loop, so the deterministic suite does not claim to grade a live model.
2. A live-model eval is inherently non-deterministic and expensive
   (API cost, run time, flakiness) — not something that belongs in a
   test suite that has to stay fast and free to run on every change.

What this seed DOES run, on every ``pytest`` invocation
(``tests/test_siftbench.py``) and via ``python -m siftbench``: each
case's REFERENCE script — hand-written, not model-written — through
Sift's actual, unmodified pipeline (the real sandboxed executor, the
real disclosure-control sanitizer, the real result store), against a
synthetic dataset with a KNOWN ground-truth answer baked in by
construction (fixed RNG seed). The score checks Sift's own numeric
and disclosure-control output against that known answer.

That is real, valuable, and honest: it is a regression benchmark for
"does Sift's pipeline compute and disclose the right thing", scored
against ground truth, runnable with zero API cost. It is a narrower
claim than "grades a model's research judgment" — this module says so
plainly rather than calling itself something it isn't. Wiring an
actual live-model driver in later is a matter of adding a second
runner that sends each case's ``prompt`` through a real
``ProviderSession`` instead of running ``reference_script`` directly,
then scoring whatever the model actually submitted the same way.
"""

from __future__ import annotations
