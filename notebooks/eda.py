# Guardian of Truth — Day 1 EDA
# Run this cell by cell to understand the dataset

# ============================================================
# CELL 1: Basic stats (you already ran this, keeping for reference)
# ============================================================
import pandas as pd
import sys
sys.path.insert(0, '..')

df = pd.read_parquet("../data/valid.parquet")

print("=== BASIC STATS ===")
print(f"Rows: {len(df)}")
print(f"Label distribution:\n{df['label'].value_counts()}")
print(f"\n=== TEXT LENGTH STATS ===")
df['prompt_len'] = df['prompt'].str.len()
df['response_len'] = df['response'].str.len()
print(f"Prompt length - mean: {df['prompt_len'].mean():.0f}, max: {df['prompt_len'].max()}, min: {df['prompt_len'].min()}")
print(f"Response length - mean: {df['response_len'].mean():.0f}, max: {df['response_len'].max()}")
print(f"\n=== DOMAIN DISTRIBUTION ===")
df['domain'] = df['id'].str.split('__').str[0]
print(df['domain'].value_counts())

# ============================================================
# CELL 2: Hallucination type analysis
# What kinds of hallucinations are in this dataset?
# ============================================================
print("\n\n=== HALLUCINATION EXPLANATION SAMPLES ===")
hallucinations = df[df['label'] == 1].copy()
print(f"Total hallucination examples: {len(hallucinations)}")
print(f"\nFull explanations:\n")
for i, row in hallucinations.iterrows():
    print(f"--- ID: {row['id']} ---")
    print(row['explanation'][:500])
    print()

# ============================================================
# CELL 3: Response analysis — are hallucinations in text or tool calls?
# ============================================================
import re

def classify_response(response: str) -> str:
    """Is the response primarily text, a tool call, or both?"""
    has_tool_call = bool(re.search(r'→\s*TOOL_CALL|⟦ASSISTANT_TOOL_CALL', response))
    has_assistant_text = bool(re.search(r'⟦ASSISTANT⟧', response))
    
    # Get text content length
    text_only = re.sub(r'→\s*TOOL_CALL.*?(?=\n⟦|\Z)', '', response, flags=re.DOTALL)
    text_only = re.sub(r'⟦[^\]]+⟧', '', text_only).strip()
    
    if has_tool_call and len(text_only) > 50:
        return "both"
    elif has_tool_call:
        return "tool_call_only"
    else:
        return "text_only"

df['response_type'] = df['response'].apply(classify_response)

print("\n=== RESPONSE TYPE BY LABEL ===")
print(pd.crosstab(df['label'], df['response_type']))

print("\n=== HALLUCINATIONS BY RESPONSE TYPE ===")
hall_types = df[df['label'] == 1]['response_type'].value_counts()
print(hall_types)

# ============================================================
# CELL 4: Parse a few examples with our parser
# ============================================================
from guardian.parser import parse_transcript, extract_response_claims

print("\n=== PARSER TEST ON FIRST 3 EXAMPLES ===")
for i, row in df.head(3).iterrows():
    print(f"\n--- {row['id']} (label={row['label']}) ---")
    
    parsed = parse_transcript(row['prompt'])
    claims = extract_response_claims(row['response'])
    
    print(f"System length: {len(parsed.system)} chars")
    print(f"Available tools: {len(parsed.available_tools_text)} chars")
    print(f"Policy text: {len(parsed.policy_text)} chars")
    print(f"Turns: {len(parsed.turns)}")
    print(f"Tool calls in prompt: {len(parsed.tool_calls)}")
    print(f"Tool responses in prompt: {len(parsed.tool_responses)}")
    print(f"Claims in response: {len(claims)}")
    for c in claims:
        print(f"  [{c['type']}] {c['content'][:120]}")

# ============================================================
# CELL 5: Check if tool call args are the main hallucination signal
# For hallucination examples with tool_call responses,
# look at what's wrong in the tool call args vs. the context
# ============================================================
print("\n=== TOOL CALL HALLUCINATION DEEP DIVE ===")
hall_with_tc = df[(df['label'] == 1) & (df['response_type'].isin(['tool_call_only', 'both']))]
print(f"Hallucinations involving tool calls: {len(hall_with_tc)} / {len(df[df['label']==1])}")

for _, row in hall_with_tc.head(2).iterrows():
    print(f"\n--- {row['id']} ---")
    print(f"RESPONSE:\n{row['response'][:500]}")
    print(f"\nEXPLANATION:\n{row['explanation'][:300]}")

# ============================================================
# CELL 6: Baseline — what F1 do we get with a naive "always predict 1" strategy?
# ============================================================
from sklearn.metrics import f1_score, precision_score, recall_score

y_true = df['label'].values

# Naive baselines
always_1 = [1] * len(y_true)
always_0 = [0] * len(y_true)

print("\n=== BASELINE SCORES (on valid.parquet) ===")
print(f"Always predict 1: F1={f1_score(y_true, always_1):.3f}, P={precision_score(y_true, always_1):.3f}, R={recall_score(y_true, always_1):.3f}")
print(f"Always predict 0: F1={f1_score(y_true, always_0):.3f}, P={precision_score(y_true, always_0, zero_division=0):.3f}, R={recall_score(y_true, always_0, zero_division=0):.3f}")
print(f"\nNote: Dataset is perfectly balanced (50/50), so:")
print(f"  Always-1 F1 = {f1_score(y_true, always_1):.3f}")
print(f"  We need F1 > 0.667 to beat a trivial classifier on balanced data")

# ============================================================
# CELL 7: Look at the AVAILABLE TOOLS section in detail
# This is what we check tool call arguments against
# ============================================================
print("\n=== AVAILABLE TOOLS SECTION ANALYSIS ===")
for i, row in df.head(2).iterrows():
    parsed = parse_transcript(row['prompt'])
    print(f"\n--- {row['id']} ---")
    print(f"Available tools text (first 800 chars):")
    print(parsed.available_tools_text[:800])
    print("...")

# ============================================================
# CELL 8: Save key findings summary
# ============================================================
summary = {
    "total_rows": len(df),
    "label_0": int(df['label'].value_counts()[0]),
    "label_1": int(df['label'].value_counts()[1]),
    "domains": df['domain'].value_counts().to_dict(),
    "avg_prompt_len": int(df['prompt_len'].mean()),
    "max_prompt_len": int(df['prompt_len'].max()),
    "avg_response_len": int(df['response_len'].mean()),
    "response_types": df['response_type'].value_counts().to_dict(),
    "hallucinations_with_tool_calls": int(len(hall_with_tc)),
    "baseline_always1_f1": round(float(f1_score(y_true, always_1)), 3),
}

import json
with open("../data/eda_summary.json", "w") as f:
    json.dump(summary, f, indent=2)

print("\n=== EDA SUMMARY ===")
print(json.dumps(summary, indent=2))
print("\n✅ Saved to data/eda_summary.json")
