# Fable's reply (claude-fable-5-1 via Agent tool, 0 tool uses, brief in fable_brief.md)

**Bottom line:** the instruction variants changed belief, which isn't the bottleneck. A reframe needs three
things at once: a failure the reader can't explain away, a view of where the received categories and the
unit's behaviour disagree, and a rival reading in view. None of the setups provide these.

**What the leap is:** a move from describing the inputs in the observer's categories to finding the variable
that cuts across them. The evidence for it is the leftover cases:
- same category, opposite response (fox vs pug);
- different category, same response (fox vs convertible);
- a coherent opposite pole (round against pointed).
It is restructuring after an impasse. The models saw that pugs suppress the unit and filed it as a footnote;
the evidence was perceived but never integrated.

**Why the interventions did nothing:**
- Telling is not triggering.
- The model never reaches an impasse. "Unrelated" and "noise" work as stopping rules.
- The search confirmed rather than tested (the 2-4-6 task), and the interface built the received categories
  into the only action.
- The top-k presentation reflects the dataset's makeup.
- The model is the expert (the averaged literature), not the fresh mind.

**Triggers:**
1. Commit to predictions on random held-out items, then see the failures.
2. Present contrast pairs.
3. Ask for the dimension that runs from most negative to most positive, or for what change would flip an input.
4. Price the "two unrelated jobs" exit: it has to be earned once single-variable rivals fail.
5. Sample several independent rivals and select by held-out prediction.
6. Give a track record across units with feedback.

**Diagnose first:**
- Perception: outside the neuron framing, does it describe shapes?
- Recognising vs generating: offer V-vs-O among distractors.
- Integration: the negatives result suggests this is the break point.
- Contamination: use units with no published reading.
- The text tests had no received reading to overturn. Attach the paper's reading to the dot unit.

**Triggered vs imposed:**
- Evidence ablation: remove the decisive items and the unity should vanish.
- A confusion matrix over a mixed battery.
- Sharpness, on items where the readings disagree, scored against the unit, not Nick's label.
- Risky predictions.
- Transfer to non-ML data.

**Expectation:** prompting moves disposition only weakly; the structure of the situation moves it more. Seeing
its own chance-level accuracy and still not saying "cannot tell" would point to a training limit. Test
against a base or less chat-tuned model, framed as a lab notebook.
