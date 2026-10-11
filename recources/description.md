## Guardian of Truth

Create new methods for detecting hallucinations in LLM models

### Task Description

LLM agents are increasingly being integrated into real product and business processes: they answer user questions, work with documents and databases, call tools, and produce final conclusions. Along with their benefits, such agents carry a specific risk — hallucinations: statements and actions that are not grounded in the context the agent actually received as input.

Especially critical are contextual hallucinations — cases where the agent has all the necessary context (instructions, data, tool results), but its final response contradicts that context or does not follow from it. For business, this is a direct source of financial and reputational risk: false data can end up in a report, in a reply to a customer, or in a decision being made.

The participants' task is to build a guardrail that automatically detects the presence of a contextual hallucination in an LLM agent's response. The guardrail operates in black-box mode: it only has access to the text of the input context and the text of the agent's response — without access to the weights, internal states, or hidden reasoning of the model being checked. This approach is applicable to any agent regardless of its internal architecture.

## Problem Statement

Given: the context the agent saw and the agent's response. The goal is to return a single binary flag — whether the response contains a contextual hallucination.

This is a response-level binary classification task: 1 — the response contains at least one hallucination, 0 — no hallucinations.

### What Counts as a Hallucination

A hallucination is a statement or action in the response that is not grounded in the context. We distinguish two mechanisms:

- Contradiction — the response contradicts the context (the data, the instructions, the tool results) in a way that makes the outcome incorrect. Typical cases: a number / date / value / name in the response does not match the data from the context; a reference to an entity, field, or value that does not exist in the provided data, or a distortion of them; a violation of a mandatory rule from the instructions that distorts the result.

- Unsupported — a statement in the response cannot be derived from the context (a fabricated fact, number, reason, entity, or condition), even though it does not directly contradict anything.

Both types of contextual hallucinations are considered within the task — contradiction and unsupported. The presence of either of them means a hallucination is present (label 1).

### What Does NOT Count as a Hallucination

Cosmetic and stylistic issues that clearly do not affect the result are not hallucinations:

- correct paraphrasing, aggregation, filtering, and sorting of data from the context;
- rounding, formatting, rephrasing without a change of meaning;
- imprecise wording in the agent's internal reasoning when the final result is correct (the outcome of the action is evaluated, not the choice of words); this allowance is revoked if the result itself is incorrect;
- service / secondary output fields that never reach the user and do not affect the result under the chosen control flag.

Litmus test for a borderline case: "Will this change the outcome or the data the user sees?" If yes or potentially yes — it is a hallucination. If obviously never — it is not.

### Input Data Format

Each example consists of two text fields:

prompt — the full context the agent saw before its response: the system instruction (rules and available data/metadata), the user's request, as well as previous tool calls and their results;

response — the agent's response being checked: its text and/or the tool calls it made in this turn (a turn may contain both).

Inside these blocks, messages and actions are recorded as a single text transcript with special bracketed markers ⟦…⟧:


Marker	Meaning
---
⟦SYSTEM⟧	the agent's system message (rules + context/metadata)
---
⟦USER⟧	user message
---
⟦ASSISTANT⟧	agent message (reasoning / text)
---
⟦ASSISTANT_TOOL_CALL name="X"⟧	a call of tool X by the agent; its arguments follow below
---
⟦USER_TOOL_CALL name="X"⟧	a call of tool X by the user (if the user has their own tools)
---
⟦TOOL_RESULT name="X" requestor="assistant"⟧	the result of tool X; the requestor attribute indicates who called it (assistant or user)
---

The substantive part of the response should be looked for in the ⟦ASSISTANT⟧ text, in the tool call itself (its type/name), and in its arguments.

### Example structure:
```

prompt:
⟦SYSTEM⟧
… agent instructions and available data/metadata …
⟦USER⟧
… user request …
⟦ASSISTANT⟧
… agent response …
⟦ASSISTANT_TOOL_CALL name="get_user_details"⟧
{"user_id": "noah_muller_9847"}
⟦TOOL_RESULT name="get_user_details" requestor="assistant"⟧
{"user_id": "noah_muller_9847", "name": {...}, ...}

response:
⟦ASSISTANT⟧
… text of the response being checked …
⟦ASSISTANT_TOOL_CALL name="cancel_reservation"⟧
{"reservation_id": "ABC123"}
```

### Data

Participants are provided with two datasets:

- Public test dataset — for the public leaderboard;
- Private test dataset — for the private leaderboard.

Each example is described by the following fields:

Column   	Type	Description
---
`id`	    str  request identifier
---
`prompt`	str	 input context (system instruction, user request, tool calls and their results)
---
`response`	str	 the model's response given the context
---
`label`	    int	 response-level binary classification label: 0 — no hallucination, 1 — hallucination present


The public dataset additionally contains the field:

Column	        Type	Description
--- 
`explanation`	str	a textual explanation of why the response was assigned this label
---

### Solution Format

The solution is submitted as a Docker submission — an archive with the following structure:

```
root/
├── README.md            (optional)
├── pyproject.toml
├── model/               (weights, configs, tokenizer vocabularies, etc.)
├── src/
│   └── your_module/
│       └── ...
└── scripts/
    └── predict.py       (entry point: inference on the test dataset)
```

How the platform runs the solution:

```
pip install .
python scripts/predict.py --input <path_to_test.csv> --output <path_to_predictions.csv>
```


The output is a file with columns `id`, `label` (the predicted binary flag for each response).

The solution must fit within the established time and resource limits (see the "Terms and Constraints" section).

### Evaluation Metric

Evaluation is performed at the response level by comparing the predicted binary flag with the reference label:

- Response F1-score — the primary metric (higher is better);
- Response Precision — higher is better;
- Response Recall — higher is better.

The final ranking is based on Response F1-score.

### Leaderboard

- Public leaderboard — visible to participants throughout the competition and shows their intermediate standing.
- Private leaderboard — revealed at the end of the competition and determines the winners

### Terms and Constraints

    Task type: Docker-based competition.
    Maximum number of submissions: 10 per team for the entire competition (only successful submissions count; failed ones do not consume the limit).
    Script runtime limit: 30 minutes.
    Hardware: 1× NVIDIA A100 GPU.
    User submission size: strictly less than 40 GB.

### Regulatory documentation

Contest Rules(https://gitverse.ru/api/repos/gitverse/AIJ/raw/branch/master/AIJourney2026-rules-en.pdf)
Consent to the processing of my personal data (https://gitverse.ru/api/repos/gitverse/AIJ/raw/branch/master/consent2026_1_en.pdf)
Sberbank's Personal Data Processing Policy (https://www.sberbank.ru/en/individualclients/privacy_policy#gdpr)
