# MLX Model Prompt Tuning

## Problem: Chatty Models Adding Explanatory Text

Some LLMs (especially instruct-tuned models like Llama) tend to be "helpful" by adding explanations before/after JSON output.

### Example Problematic Response
```
Note: The confidence level is subjective and may vary based on individual perspectives.
The provided confidence level is based on a general assessment of the email content.

Also, note that the email [TRUNCATED - ran out of tokens]
```

### Root Causes

1. **Instruct-tuned behavior**: Models trained to be helpful/verbose
2. **Insufficient tokens**: Response cut off before JSON appears
3. **Weak instructions**: Model ignores "no explanations" request

## Solution: Chat Templates

### The Fix: Use Proper Chat Template (`mlx_backend.py`, `agents.py`, `langgraph_subgraphs.py`)

**Root Cause:** Llama instruct models need proper system/user message formatting to follow instructions.

**Before:** Plain string prompts → Model generates 1049 chars of step-by-step analysis
**After:** Chat template with system message → Model generates 42 chars of perfect JSON

### Implementation

```python
# mlx_backend.py - Apply chat template
messages = [
    {
        "role": "system",
        "content": "You are a JSON-only API. You MUST respond with ONLY valid JSON. "
        "No explanations, no reasoning, no markdown - just the JSON object."
    },
    {
        "role": "user",
        "content": f"""Classify this email...
Output format: {{"is_marketing": true, "confidence": 0.95}}"""
    }
]

prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
```

Key changes:
- ✅ Use `tokenizer.apply_chat_template()` for instruct models
- ✅ System message enforces JSON-only output
- ✅ User message contains classification task
- ✅ Removed `json_schema` parameter (no longer needed)
- ✅ Increased `max_tokens=200` (generous buffer)

### 3. Robust JSON Extraction (`agents.py:404-424`)

Even if model adds text, we extract JSON:

```python
# Remove markdown fences
if response_text.startswith("```"):
    # Strip opening/closing fences

# Find first { and last }
json_start = response_text.find("{")
json_end = response_text.rfind("}") + 1

if json_start != -1 and json_end > json_start:
    json_text = response_text[json_start:json_end]
    parsed = json.loads(json_text)
```

This handles:
- ✅ Text before JSON
- ✅ Text after JSON
- ✅ Markdown wrapping
- ✅ Multiple paragraphs

## Results

### Before Chat Template Fix
- ❌ Plain string prompts: 1049 chars of step-by-step analysis
- ❌ 50 tokens: responses cut off mid-sentence
- ❌ Full schema in prompt: model echoed schema back
- ❌ Weak instructions: model ignored them
- ❌ Empty responses: model confusion

### After Chat Template Fix
- ✅ **42 chars: Perfect JSON only** `{"is_marketing": true, "confidence": 0.95}`
- ✅ **100% success rate** across all test emails
- ✅ System message: Model respects JSON-only instruction
- ✅ 200 tokens: Generous buffer (though only ~42 needed)
- ✅ Robust extraction: Handles edge cases gracefully

### Test Results
Tested with 5 marketing emails:
```
✓ email_213.eml [42 chars] marketing=True, conf=0.95
✓ email_214.eml [42 chars] marketing=True, conf=0.95
✓ email_215.eml [42 chars] marketing=True, conf=0.95
✓ email_216.eml [42 chars] marketing=True, conf=0.95
✓ email_217.eml [42 chars] marketing=True, conf=0.95
```

## Safe Defaults on Error

If JSON extraction still fails:
```python
return EmailDecision(
    email=email,
    decision=Decision.KEEP,  # Safe default
    reason=FilterReason.AI_CLASSIFIED,
    confidence=0.0,
)
```

This prevents accidental deletions when AI fails.

## Model-Specific Notes

### Llama 3.2 3B Instruct
- ✅ **Works perfectly with chat template**
- **Before:** 1049 chars of step-by-step analysis
- **After:** 42 chars of pure JSON
- **Chat template is essential** - plain prompts don't work

### Alternative Models for JSON Output

If you want to try other MLX models:

1. **`mlx-community/Llama-3.2-1B-Instruct-4bit`** (smaller, faster)
   - Same architecture, more constrained = faster
   - Also needs chat template

2. **`mlx-community/gemma-2-2b-it-4bit`** (Google)
   - Good at following format instructions
   - 2B params - sweet spot for speed/quality

3. **`mlx-community/Qwen2.5-3B-Instruct-4bit`** (Alibaba)
   - Excellent instruction following
   - Known for structured output compliance

**All instruct models benefit from chat templates.**

## Testing

Run MLX JSON parsing tests:
```bash
uv run pytest tests/test_mlx_json_parsing.py -v
```

All 24 tests validate different response formats including:
- Plain JSON
- JSON with markdown
- JSON with surrounding text
- Incomplete responses
- Invalid JSON (safe defaults)
