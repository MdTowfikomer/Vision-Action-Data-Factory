# Guardian of Truth — Product Requirements Document (PRD)
## Version 2.0 — Updated after EDA

---

## Problem Statement
How might we automatically detect contextual hallucinations in LLM agent responses, given only the input context and output text, with no access to model internals?

---

## Key Findings from EDA (valid.parquet)

| Metric | Value |
|---|---|
| Total rows | 46 |
| Label distribution | 23 hallucination / 23 clean (50/50) |
| Avg prompt length | ~42,000 chars |
| Max prompt length | ~233,000 chars |
| Avg response length | ~519 chars |
| Domains | airline (12), telecom (12), banking_knowledge (11), retail (11) |
| Baseline always-1 F1 | 0.667 |

**Critical insight:** Hallucinations in this dataset are primarily **tool call argument violations** — wrong field names, missing required parameters, values contradicting policy or previous tool results. The response to check is usually one short tool call, not paragraphs of text.

---

## Recommended Architecture

A rule-based schema/value checker as primary signal, with Laya + Gemini as fallback for ambiguous cases.

```
Input: (prompt, response)
        ↓
   Unit 1: Transcript Parser
   - Extract ⟦SYSTEM⟧, ⟦USER⟧, ⟦ASSISTANT⟧ blocks
   - Extract [AVAILABLE TOOLS] schema
   - Extract ← TOOL_RESPONSE results
   - Extract tool calls from response
        ↓
   Unit 2: Schema Checker          Unit 3: Value Checker
   - Tool name exists?              - IDs referenced exist in context?
   - Required fields present?       - Numeric values match tool responses?
   - Field names correct?           - Policy numeric constraints met?
   - Enum values valid?
        ↓                                ↓
   If EITHER detects violation → label = 1 (HIGH confidence)
   If both pass but uncertain → Unit 4: LLM Judge
        ↓
   Unit 4: Laya (fast, on A100) → if uncertain → Gemini API fallback
        ↓
   Unit 5: Pipeline + CLI
   - Aggregation: any claim = 1 → label = 1
   - scripts/predict.py --input --output
        ↓
   Output: id, label CSV
```

---

## Components

| Unit | File | What it does |
|---|---|---|
| 1 — Parser | `src/guardian/parser.py` | Splits transcripts into structured objects |
| 2 — Schema Checker | `src/guardian/schema_checker.py` | Validates tool call args against [AVAILABLE TOOLS] schema |
| 3 — Value Checker | `src/guardian/value_checker.py` | Validates tool call values against tool response data + policy |
| 4 — LLM Judge | `src/guardian/llm_judge.py` | Laya + Gemini for ambiguous cases |
| 5 — Pipeline + CLI | `src/guardian/pipeline.py` + `scripts/predict.py` | End-to-end orchestration |

---

## Tech Stack

| Layer | Choice | Why |
|---|---|---|
| Primary judge | Rule-based (Units 2+3) | Catches field name / schema violations deterministically |
| Secondary judge | `convaiinnovations/laya` (HuggingFace) | 7ms/question batched, runs on A100, free |
| Tertiary judge | Gemini API (`gemini-1.5-flash`) | Deep reasoning for ambiguous text claims |
| Runtime | Python 3.11, PyTorch | Standard ML stack |
| Packaging | Docker + pyproject.toml | Competition requirement |
| Data | pandas + pyarrow | CSV/parquet I/O |

---

## Task Breakdown

### Unit 1 — Transcript Parser
- [ ] 1.1 Parse `⟦SYSTEM⟧`, `⟦USER⟧`, `⟦ASSISTANT⟧` blocks
- [ ] 1.2 Extract `[AVAILABLE TOOLS]` section as raw text
- [ ] 1.3 Extract `<policy>` section as raw text
- [ ] 1.4 Parse `→ TOOL_CALL name: {json}` patterns
- [ ] 1.5 Parse `← TOOL_RESPONSE name: content` patterns
- [ ] 1.6 Parse `⟦ASSISTANT_TOOL_CALL name="X"⟧{json}` style
- [ ] 1.7 `extract_response_claims()` → list of checkable claims

**Test:** Run on all 46 rows, zero crashes, >0 claims per row.

### Unit 2 — Tool Schema Checker
- [ ] 2.1 Parse available tools section into structured dict
- [ ] 2.2 Check: tool name exists in available tools?
- [ ] 2.3 Check: all required parameters present?
- [ ] 2.4 Check: parameter names match schema exactly?
- [ ] 2.5 Check: enum values are valid?
- [ ] 2.6 Return `{is_hallucination, reason, confidence}`

**Test:** `airline__23::t10` (wrong field names) must be caught.

### Unit 3 — Value Checker
- [ ] 3.1 Extract IDs and values from last 2 tool responses
- [ ] 3.2 Check: IDs in tool call args exist in context?
- [ ] 3.3 Check: numeric values contradict tool response data?
- [ ] 3.4 Check: policy numeric constraints violated?
- [ ] 3.5 Return `{is_hallucination, reason, confidence}`

**Test:** `airline__21::t7` (wrong nonfree_baggages) must be caught.

### Unit 4 — LLM Judge
- [ ] 4.1 Build focused context string (max 4000 tokens)
- [ ] 4.2 Laya integration: load model, define typed questions
- [ ] 4.3 Gemini fallback: call if Laya confidence < 0.65
- [ ] 4.4 Tune escalation threshold against valid.parquet
- [ ] 4.5 Return `{label, confidence, source}`

**Test:** Run on 10 examples, compare to ground truth.

### Unit 5 — Pipeline + CLI
- [ ] 5.1 Orchestrate all units in `pipeline.py`
- [ ] 5.2 Aggregation logic: any claim = 1 → label = 1
- [ ] 5.3 `scripts/predict.py` with `--input` / `--output` args
- [ ] 5.4 Batch processing with tqdm progress bar
- [ ] 5.5 Evaluate on valid.parquet, print F1/P/R
- [ ] 5.6 Dockerfile + pinned pyproject.toml

**Test:** `python scripts/predict.py --input data/valid.parquet --output predictions.csv` produces valid CSV.

---

## Submission Strategy (10 submissions total)

### Submission 1 — Rule-only baseline (Units 1+2+3)
- Pure schema + value checking, no LLM
- Fast to build, fully deterministic
- Goal: get on leaderboard, see where we stand
- Expected F1: ~0.5–0.7 depending on how many hallucinations are schema violations

### Submissions 2–4 — Add LLM judge, tune threshold
- Add Laya, tune confidence threshold (0.5, 0.6, 0.7)
- Each submission tests a different threshold

### Submissions 5–7 — Improve context extraction
- Better truncation strategy for long prompts
- Better claim extraction from text responses

### Submissions 8–10 — Reserved for late improvements
- Fine-tune threshold based on leaderboard feedback
- Try Gemini-only vs hybrid

---

## Constraints
- Runtime: ≤ 30 minutes on 1× NVIDIA A100 GPU
- Submission size: < 40 GB
- Output: CSV with `id`, `label` columns only
- Max 10 successful submissions per team

---

## Not Doing
- Fine-tuning any model (unfamiliar territory, 1-week constraint)
- Building a UI or explainability layer
- Multi-model ensemble (overkill for week 1)

---

## Open Questions
- Does the private test set have the same domain distribution?
- Does the competition Docker environment allow outbound HTTPS (Gemini API)?
- What is the size of the private test set?

---

## Project Structure
```
GuardianofTruth/
├── docs/
│   ├── PRD.md                   ← this file
│   └── ideas/guardian-of-truth.md
├── src/guardian/
│   ├── __init__.py
│   ├── parser.py                ← Unit 1
│   ├── schema_checker.py        ← Unit 2
│   ├── value_checker.py         ← Unit 3
│   ├── llm_judge.py             ← Unit 4
│   └── pipeline.py              ← Unit 5
├── scripts/
│   └── predict.py               ← competition entry point
├── notebooks/
│   └── eda.py
├── model/laya/                  ← downloaded HuggingFace weights
├── data/
│   └── valid.parquet
├── pyproject.toml
└── Dockerfile
```
