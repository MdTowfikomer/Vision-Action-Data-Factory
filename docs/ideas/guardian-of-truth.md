# Guardian of Truth — Hallucination Detector

## Problem Statement
How might we automatically detect contextual hallucinations in LLM agent responses, given only the input context and output text, with no access to model internals?

---

## Recommended Direction
A two-stage hybrid pipeline: **Laya as the fast primary judge** (runs fully on A100, free, 7ms/question batched) escalating uncertain cases to **Gemini API** for deep reasoning. The pipeline parses structured agent transcripts, extracts verifiable claims from the response, asks Laya typed questions against the context, and only calls Gemini when Laya's confidence is below a tunable threshold.

This approach is fast enough to handle large test sets within the 30-min limit, costs near-zero on API, and is genuinely competitive because most submissions will be naive LLM-judge-only approaches.

---

## Architecture

```
┌─────────────────────────────────────────────────────┐
│                    Input CSV                        │
│              (id, prompt, response)                 │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│              Transcript Parser                      │
│  Extracts: SYSTEM, USER, TOOL_RESULTs, ASSISTANT   │
│  response sections into structured dict             │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│              Claim Extractor                        │
│  Splits response into checkable claims:             │
│  - Text sentences from ⟦ASSISTANT⟧                 │
│  - Tool call arguments (name + args JSON)           │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│           Laya Decision Engine  (A100)              │
│  Per claim, asks typed questions:                   │
│  Q1: "Does this claim contradict the context?"     │
│  Q2: "Is this claim unsupported by the context?"   │
│  Returns: probability score per question            │
└────────────┬──────────────────┬─────────────────────┘
             │                  │
     confident enough      uncertain (score
     (score > threshold)    near 0.5)
             │                  │
             ▼                  ▼
      label = 0 or 1    ┌──────────────────┐
                        │   Gemini API     │
                        │  (deep judge)    │
                        │  returns 0 or 1  │
                        └────────┬─────────┘
                                 │
                                 ▼
┌─────────────────────────────────────────────────────┐
│           Response-level Aggregator                 │
│  If ANY claim = 1 → label = 1                       │
│  All claims = 0   → label = 0                       │
└───────────────────┬─────────────────────────────────┘
                    │
                    ▼
┌─────────────────────────────────────────────────────┐
│           Output CSV (id, label)                    │
└─────────────────────────────────────────────────────┘
```

---

## Components

| Component | What it does | Tech |
|---|---|---|
| Transcript Parser | Splits prompt/response by `⟦...⟧` markers into structured sections | Python, regex |
| Claim Extractor | Breaks response into atomic checkable units | Python, simple sentence splitting + JSON parsing |
| Laya Engine | Asks typed questions, returns calibrated probabilities | `convaiinnovations/laya` from HuggingFace, PyTorch |
| Gemini Escalator | Deep judge for uncertain cases | `google-generativeai` SDK |
| Aggregator | Collapses claim-level results to response-level binary | Python |
| predict.py | Entry point CLI script | argparse, pandas |

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Primary judge | `convaiinnovations/laya` (HuggingFace) | Open weights, 7ms/question batched, runs on A100, free |
| Escalation judge | Gemini API (`gemini-1.5-flash`) | 1M token context, fast, free tier available |
| Fallback judge | Groq free tier | Backup if Gemini quota runs out |
| Runtime | Python 3.11, PyTorch | Standard ML stack |
| Packaging | Docker + pyproject.toml | Competition requirement |
| Data | pandas | CSV I/O |

---

## Constraints
- Runtime: ≤ 30 minutes on 1× NVIDIA A100 GPU
- Submission size: < 40 GB
- Output: CSV with `id`, `label` columns only
- No access to model weights/internals of the agent being checked
- Docker-based submission
- Max 10 successful submissions per team

---

## Key Assumptions to Validate
- [ ] **Laya handles long inputs** — agent transcripts can be long; verify Laya's max token limit and truncation behavior
- [ ] **Dataset size** — download public dataset first, count rows, check avg prompt length
- [ ] **Label distribution** — check class balance (50/50 vs skewed) before assuming F1 strategy
- [ ] **Gemini context window** — long transcripts need to fit; `gemini-1.5-flash` has 1M token context, should be fine
- [ ] **Laya confidence threshold** — 0.7 is a starting guess; needs tuning against public labeled data
- [ ] **Outbound API calls allowed** — verify the competition Docker environment allows outbound HTTPS (for Gemini API)

---

## Not Doing (and Why)
- **Fine-tuning any model** — unfamiliar territory, 1-week constraint, public dataset size unknown
- **Using Jev** — behind early-access waitlist; Laya is open-weight and equivalent for this task
- **Calling Gemini for every example** — too slow, too expensive, risks hitting the 30-min runtime limit
- **Building a UI or explainability layer** — competition only needs `id, label` CSV output
- **Multi-model ensemble** — good for top-1% leaderboard, overkill for week-1 submission
- **Clef** — no credible decision model by this name exists in this space

---

## Open Questions
- How many rows is the private test set? (determines batching strategy)
- Does the public dataset have enough examples to tune the Laya threshold meaningfully?
- Will the competition Docker environment allow outbound API calls to Gemini?
- Does the team have access to the competition platform to download the dataset?

---

## Project Structure
```
GuardianofTruth/
├── README.md
├── pyproject.toml
├── Dockerfile
├── model/
│   └── laya/               ← downloaded HuggingFace weights
├── src/
│   └── guardian/
│       ├── __init__.py
│       ├── parser.py        ← transcript parser
│       ├── extractor.py     ← claim extractor
│       ├── laya_judge.py    ← Laya integration
│       ├── gemini_judge.py  ← Gemini escalator
│       └── pipeline.py      ← wires everything together
├── scripts/
│   └── predict.py           ← entry point (competition requirement)
├── notebooks/
│   └── eda.ipynb            ← Day 1 data exploration
└── data/
    └── public_test.csv      ← public dataset (not committed to git)
```

---

## Day-by-Day Implementation Plan

### Day 1 — Understand the Data
**Owner: Both teammates**
- [ ] Download the public dataset from the competition platform
- [ ] Write `notebooks/eda.ipynb` — count rows, check label distribution, print sample examples
- [ ] Manually read 10 examples — understand what hallucinations actually look like in this data
- [ ] Note: avg prompt length, types of tool calls present, how long responses are
- [ ] Check if Laya's token limit is sufficient for average prompt length

**Goal:** Be able to describe a hallucination example in plain English before writing any pipeline code.

---

### Day 2 — Build the Transcript Parser + Claim Extractor
**Owner: Teammate A (or solo)**
- [ ] Write `src/guardian/parser.py` — splits `⟦MARKER⟧` blocks into a structured dict
- [ ] Write `src/guardian/extractor.py` — extracts claims from the response section
- [ ] Test on 10 real examples, print the output, verify it looks sane
- [ ] Handle edge cases: missing markers, empty tool results, multi-turn conversations, JSON tool call args

**You'll learn:** Regex parsing, how the data is structured, Python dict manipulation.

---

### Day 3 — Integrate Laya
**Owner: Teammate B (or solo)**
- [ ] Load `convaiinnovations/laya` from HuggingFace
- [ ] Write `src/guardian/laya_judge.py` — takes (context, claim) → returns probability score
- [ ] Design the typed questions carefully (this matters a lot for accuracy)
- [ ] Run on 50 examples from the public dataset, compare predictions to labels
- [ ] Tune the confidence threshold (start at 0.7, adjust based on F1)

**You'll learn:** HuggingFace model loading, calibrated probabilities, threshold tuning.

---

### Day 4 — Integrate Gemini Escalator
**Owner: Teammate A**
- [ ] Write `src/guardian/gemini_judge.py` — takes full (prompt, response) → returns 0/1
- [ ] Design the Gemini prompt carefully — be explicit about both hallucination types (contradiction + unsupported)
- [ ] Wire into pipeline: Laya uncertain → call Gemini
- [ ] Test full pipeline end-to-end on the public dataset
- [ ] Measure: what % of cases escalate to Gemini? Adjust threshold if too many

**You'll learn:** Gemini API usage, prompt engineering for classification, structured output parsing.

---

### Day 5 — Evaluate + Tune
**Owner: Both teammates**
- [ ] Run full pipeline on the public labeled dataset
- [ ] Compute F1, precision, recall using `sklearn.metrics`
- [ ] Identify failure patterns — where is it wrong? Which hallucination types does it miss?
- [ ] Iterate on: Laya question wording, Gemini prompt, escalation threshold
- [ ] Stretch: compare Laya-only vs Laya+Gemini — is Gemini actually improving F1?

**You'll learn:** Evaluating a binary classifier, F1 vs precision vs recall trade-offs, what "tuning" means in practice.

---

### Day 6 — Docker + predict.py
**Owner: Teammate B (or whoever is more comfortable with CLI/infra)**
- [ ] Write `scripts/predict.py` with argparse (`--input`, `--output`)
- [ ] Write `pyproject.toml` with all dependencies pinned to exact versions
- [ ] Write `Dockerfile`
- [ ] Test: `docker build` → `docker run` → verify output CSV format matches spec (`id`, `label`)
- [ ] Check submission size < 40 GB

**You'll learn:** Docker basics, Python packaging with pyproject.toml, CLI script structure.

---

### Day 7 — Submit + Iterate
**Owner: Both teammates**
- [ ] Submit to public leaderboard
- [ ] Compare F1 to leaderboard — understand the gap
- [ ] If time: try swapping Gemini for Groq (faster, different reasoning style)
- [ ] Document what worked and what didn't for the next submission iteration

---

## Teammate Division Suggestion

| Area | Teammate A | Teammate B |
|---|---|---|
| Data exploration (Day 1) | ✓ | ✓ |
| Transcript parser | ✓ | |
| Claim extractor | ✓ | |
| Laya integration | | ✓ |
| Gemini escalator | ✓ | |
| Evaluation + tuning | ✓ | ✓ |
| Docker + packaging | | ✓ |
| predict.py entry point | | ✓ |

---

## Key Concepts to Understand as You Build

| Concept | Where it appears | Why it matters |
|---|---|---|
| NLI (Natural Language Inference) | Laya's core task | Understanding entailment vs contradiction |
| Calibrated probability | Laya output | A score of 0.7 means "70% confident" — not just a rank |
| F1 score | Evaluation metric | Harmonic mean of precision and recall — why not just accuracy |
| Precision vs Recall trade-off | Threshold tuning | Higher threshold = higher precision, lower recall |
| System 1 vs System 2 models | Laya vs Gemini | Fast reflexive judgment vs slow deliberate reasoning |
| Binary classification | The whole task | Output is always 0 or 1, not a score |
