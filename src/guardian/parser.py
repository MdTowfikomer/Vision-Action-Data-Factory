"""
Transcript parser for Guardian of Truth.

Handles both marker styles observed in the dataset:
- ⟦SYSTEM⟧, ⟦USER⟧, ⟦ASSISTANT⟧ style
- → TOOL_CALL name: {args} / ← TOOL_RESPONSE name: result style
"""

import re
import json
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class ToolCall:
    name: str
    arguments: dict
    raw_arguments: str  # original string before JSON parsing


@dataclass
class ToolResponse:
    name: str
    content: str


@dataclass
class ParsedTranscript:
    """Structured representation of a prompt or response."""
    system: str = ""
    available_tools_text: str = ""  # raw [AVAILABLE TOOLS] section
    policy_text: str = ""           # raw <policy> section
    turns: list = field(default_factory=list)  # list of dicts with role/content
    tool_calls: list = field(default_factory=list)    # ToolCall objects from prompt
    tool_responses: list = field(default_factory=list)  # ToolResponse objects from prompt


def parse_transcript(text: str) -> ParsedTranscript:
    """
    Parse a prompt or response string into structured components.
    
    Handles the ⟦MARKER⟧ style and the → TOOL_CALL / ← TOOL_RESPONSE style.
    """
    result = ParsedTranscript()
    
    # --- Extract [AVAILABLE TOOLS] section from SYSTEM block ---
    available_tools_match = re.search(
        r'\[AVAILABLE TOOLS\](.*?)(?=\[|\Z)',
        text,
        re.DOTALL
    )
    if available_tools_match:
        result.available_tools_text = available_tools_match.group(1).strip()
    
    # --- Extract <policy> section ---
    policy_match = re.search(r'<policy>(.*?)</policy>', text, re.DOTALL)
    if policy_match:
        result.policy_text = policy_match.group(1).strip()
    
    # --- Split by ⟦MARKER⟧ blocks ---
    # Pattern: ⟦MARKER⟧ or ⟦MARKER name="X"⟧
    block_pattern = re.compile(r'⟦([^\]]+)⟧')
    parts = block_pattern.split(text)
    
    if len(parts) > 1:
        # parts[0] is text before first marker (usually empty)
        # Then alternating: marker_text, content, marker_text, content...
        i = 1
        while i < len(parts):
            marker = parts[i].strip()
            content = parts[i + 1].strip() if i + 1 < len(parts) else ""
            i += 2
            
            if marker == "SYSTEM":
                result.system = content
            elif marker == "USER":
                result.turns.append({"role": "user", "content": content})
            elif marker.startswith("ASSISTANT"):
                # Extract tool calls embedded in ASSISTANT content
                turn_content = content
                tool_calls_in_turn = _extract_arrow_tool_calls(content)
                result.tool_calls.extend(tool_calls_in_turn)
                result.turns.append({"role": "assistant", "content": turn_content})
            elif marker.startswith("ASSISTANT_TOOL_CALL"):
                name_match = re.search(r'name="([^"]+)"', marker)
                name = name_match.group(1) if name_match else "unknown"
                tc = _parse_tool_call(name, content)
                result.tool_calls.append(tc)
            elif marker.startswith("TOOL_RESULT") or marker.startswith("USER_TOOL_CALL"):
                name_match = re.search(r'name="([^"]+)"', marker)
                name = name_match.group(1) if name_match else "unknown"
                result.tool_responses.append(ToolResponse(name=name, content=content))
    else:
        # No ⟦⟧ markers — try → TOOL_CALL / ← TOOL_RESPONSE style
        result.system = ""
        tool_calls = _extract_arrow_tool_calls(text)
        result.tool_calls.extend(tool_calls)
        tool_responses = _extract_arrow_tool_responses(text)
        result.tool_responses.extend(tool_responses)
        # Remaining text as assistant turn
        clean = _remove_arrow_markers(text).strip()
        if clean:
            result.turns.append({"role": "assistant", "content": clean})
    
    return result


def _parse_tool_call(name: str, raw_args: str) -> ToolCall:
    """Parse a tool call with its JSON arguments."""
    raw_args = raw_args.strip()
    try:
        args = json.loads(raw_args)
    except json.JSONDecodeError:
        args = {"_raw": raw_args}
    return ToolCall(name=name, arguments=args, raw_arguments=raw_args)


def _extract_arrow_tool_calls(text: str) -> list:
    """
    Extract → TOOL_CALL name: {json} patterns.
    Also handles → TOOL_CALL name: {...} multiline.
    """
    tool_calls = []
    # Match: → TOOL_CALL <name>: <json_or_text>
    pattern = re.compile(
        r'→\s*TOOL_CALL\s+(\w+)\s*:\s*(\{.*?\})',
        re.DOTALL
    )
    for match in pattern.finditer(text):
        name = match.group(1)
        raw_args = match.group(2).strip()
        tool_calls.append(_parse_tool_call(name, raw_args))
    return tool_calls


def _extract_arrow_tool_responses(text: str) -> list:
    """Extract ← TOOL_RESPONSE <name>: <content> patterns."""
    responses = []
    pattern = re.compile(
        r'←\s*TOOL_RESPONSE\s+(\w+)\s*:\s*(.*?)(?=→\s*TOOL_CALL|←\s*TOOL_RESPONSE|\Z)',
        re.DOTALL
    )
    for match in pattern.finditer(text):
        name = match.group(1)
        content = match.group(2).strip()
        responses.append(ToolResponse(name=name, content=content))
    return responses


def _remove_arrow_markers(text: str) -> str:
    """Remove → TOOL_CALL and ← TOOL_RESPONSE lines from text."""
    lines = []
    skip = False
    for line in text.split('\n'):
        if re.match(r'\s*[→←]', line):
            skip = True
        elif skip and line.strip() and not line.strip().startswith('{') and not line.strip().startswith('}'):
            skip = False
        if not skip:
            lines.append(line)
    return '\n'.join(lines)


def extract_response_claims(response_text: str) -> list:
    """
    Extract checkable claims from a response string.
    
    Returns a list of dicts with:
    - type: 'text' or 'tool_call'
    - content: the claim text
    - tool_name: (for tool_calls) the tool being called
    - tool_args: (for tool_calls) parsed arguments dict
    """
    claims = []
    
    # Extract ⟦ASSISTANT⟧ text portion
    assistant_text = ""
    assistant_match = re.search(r'⟦ASSISTANT[^⟧]*⟧(.*?)(?=⟦|\→|\Z)', response_text, re.DOTALL)
    if assistant_match:
        assistant_text = assistant_match.group(1).strip()
    else:
        # Try to extract text before any tool call markers
        clean = re.split(r'[→←]', response_text)[0].strip()
        if clean:
            assistant_text = clean
    
    # Add non-empty text sentences as claims
    if assistant_text:
        sentences = re.split(r'(?<=[.!?])\s+', assistant_text)
        for s in sentences:
            s = s.strip()
            if len(s) > 20:  # skip very short fragments
                claims.append({
                    "type": "text",
                    "content": s
                })
    
    # Extract tool calls from response
    tool_calls = _extract_arrow_tool_calls(response_text)
    
    # Also check ⟦ASSISTANT_TOOL_CALL⟧ style
    tc_pattern = re.compile(r'⟦ASSISTANT_TOOL_CALL\s+name="([^"]+)"⟧(.*?)(?=⟦|\Z)', re.DOTALL)
    for m in tc_pattern.finditer(response_text):
        name = m.group(1)
        raw = m.group(2).strip()
        tc = _parse_tool_call(name, raw)
        tool_calls.append(tc)
    
    for tc in tool_calls:
        claims.append({
            "type": "tool_call",
            "content": f"TOOL_CALL {tc.name}: {tc.raw_arguments}",
            "tool_name": tc.name,
            "tool_args": tc.arguments
        })
    
    return claims


if __name__ == "__main__":
    # Quick test
    import pandas as pd
    df = pd.read_parquet("data/valid.parquet")
    
    row = df.iloc[0]
    print(f"ID: {row['id']}")
    print(f"Label: {row['label']}")
    
    parsed = parse_transcript(row['prompt'])
    print(f"\nSystem length: {len(parsed.system)}")
    print(f"Available tools section length: {len(parsed.available_tools_text)}")
    print(f"Policy section length: {len(parsed.policy_text)}")
    print(f"Number of turns: {len(parsed.turns)}")
    print(f"Tool calls in prompt: {len(parsed.tool_calls)}")
    print(f"Tool responses in prompt: {len(parsed.tool_responses)}")
    
    claims = extract_response_claims(row['response'])
    print(f"\nClaims in response: {len(claims)}")
    for c in claims:
        print(f"  [{c['type']}] {c['content'][:100]}")
