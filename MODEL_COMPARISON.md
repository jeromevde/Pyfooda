# OpenRouter Model Comparison for Food Aggregation

This document lists recommended models for the food aggregation workflow, ranked by cost-effectiveness.

Last updated: 2026-02-06

---

## ⭐ Recommended Models

### 1. **FREE TIER** (with rate limiting)

#### Google Gemini Flash 1.5 8B (FREE)
- **Model ID:** `google/gemini-flash-1.5-8b`
- **Cost:** FREE (with limits)
- **Rate Limit:** ~15 requests/minute
- **Quality:** Very good for food categorization
- **Time for 3M items:** ~3.5 days (with rate limiting)
- **Best for:** Patient runs, overnight processing

**Configuration:**
```python
model="google/gemini-flash-1.5-8b"
rate_limit=15  # requests per minute
```

#### Google Gemini Flash 1.5 (FREE)
- **Model ID:** `google/gemini-flash-1.5`
- **Cost:** FREE (with limits)
- **Rate Limit:** ~15 requests/minute
- **Quality:** Excellent
- **Time for 3M items:** ~3.5 days

---

### 2. **ULTRA-CHEAP** ($0.10-0.50 per M tokens)

#### DeepSeek R1 Distill Qwen 32B (BEST VALUE!)
- **Model ID:** `deepseek/deepseek-r1-distill-qwen-32b`
- **Cost:** $0.14/M input, $0.28/M output
- **Estimated cost (3M items):** ~$18-20
- **Quality:** Excellent reasoning, very accurate
- **Speed:** Fast
- **Best for:** Production runs with best quality/cost ratio

#### Qwen 2.5 7B Instruct
- **Model ID:** `qwen/qwen-2.5-7b-instruct`
- **Cost:** $0.30/M input, $0.30/M output
- **Estimated cost (3M items):** ~$40
- **Quality:** Good, consistent
- **Speed:** Very fast
- **Best for:** Balanced quality and speed

#### Qwen 2 VL 7B Instruct
- **Model ID:** `qwen/qwen-2-vl-7b-instruct`
- **Cost:** $0.15/M input, $0.15/M output
- **Estimated cost (3M items):** ~$20
- **Quality:** Good for text analysis
- **Speed:** Fast
- **Best for:** Budget-conscious runs

---

### 3. **CHEAP** ($0.50-1.00 per M tokens)

#### Meta Llama 3.3 70B Instruct
- **Model ID:** `meta-llama/llama-3.3-70b-instruct`
- **Cost:** $0.60/M input, $0.60/M output
- **Estimated cost (3M items):** ~$80
- **Quality:** Excellent, very reliable
- **Best for:** When quality matters more than cost

#### Mistral Nemo 2407
- **Model ID:** `mistralai/mistral-nemo`
- **Cost:** $0.15/M input, $0.15/M output
- **Estimated cost (3M items):** ~$20
- **Quality:** Good
- **Speed:** Fast

---

### 4. **PREMIUM** (when quality is critical)

#### Claude 3.5 Haiku
- **Model ID:** `anthropic/claude-3.5-haiku`
- **Cost:** $1.00/M input, $5.00/M output
- **Estimated cost (3M items):** ~$360
- **Quality:** Excellent, very nuanced
- **Best for:** Final quality pass on critical data

#### GPT-4o Mini
- **Model ID:** `openai/gpt-4o-mini`
- **Cost:** $0.15/M input, $0.60/M output
- **Estimated cost (3M items):** ~$40
- **Quality:** Very good
- **Best for:** OpenAI ecosystem users

---

## Cost Estimates by Scale

### For 295k items (current USDA database)
Assuming 30% merge rate → ~90k validations

| Model | Input Cost | Output Cost | Total Cost | Time (15 req/min) |
|-------|-----------|-------------|------------|-------------------|
| **Gemini Flash 1.5** | FREE | FREE | **$0** | ~4 days |
| **DeepSeek R1 Qwen 32B** | $1.26 | $1.26 | **$2.52** | ~2 hours |
| **Qwen 2 VL 7B** | $1.35 | $0.68 | **$2.03** | ~2 hours |
| **Qwen 2.5 7B** | $2.70 | $1.35 | **$4.05** | ~2 hours |
| **Llama 3.3 70B** | $5.40 | $2.70 | **$8.10** | ~2 hours |

### For 1M items
Assuming 30% merge rate → ~300k validations

| Model | Input Cost | Output Cost | Total Cost | Time (15 req/min) |
|-------|-----------|-------------|------------|-------------------|
| **Gemini Flash 1.5** | FREE | FREE | **$0** | ~14 days |
| **DeepSeek R1 Qwen 32B** | $4.20 | $4.20 | **$8.40** | ~7 hours |
| **Qwen 2 VL 7B** | $4.50 | $2.25 | **$6.75** | ~7 hours |
| **Qwen 2.5 7B** | $9.00 | $4.50 | **$13.50** | ~7 hours |

### For 3M items
Assuming 30% merge rate → ~900k validations

| Model | Input Cost | Output Cost | Total Cost | Time (15 req/min) |
|-------|-----------|-------------|------------|-------------------|
| **Gemini Flash 1.5** | FREE | FREE | **$0** | ~42 days |
| **DeepSeek R1 Qwen 32B** | $12.60 | $12.60 | **$25.20** | ~21 hours |
| **Qwen 2 VL 7B** | $13.50 | $6.75 | **$20.25** | ~21 hours |
| **Qwen 2.5 7B** | $27.00 | $13.50 | **$40.50** | ~21 hours |

---

## Rate Limiting Strategies

### Conservative (FREE tier)
- **Rate:** 15 requests/minute
- **Daily capacity:** ~21,600 validations/day
- **For 295k items:** ~4-5 days
- **For 1M items:** ~14 days
- **For 3M items:** ~42 days

### Moderate (Paid tier)
- **Rate:** 60 requests/minute
- **Daily capacity:** ~86,400 validations/day
- **For 295k items:** ~1 day
- **For 1M items:** ~3.5 days
- **For 3M items:** ~10.5 days

### Aggressive (No limits)
- **Rate:** 1000+ requests/minute (parallel)
- **For 295k items:** ~2 hours
- **For 1M items:** ~7 hours
- **For 3M items:** ~21 hours

---

## Recommended Strategies by Budget

### $0 Budget (100% Free)
1. **Model:** `google/gemini-flash-1.5-8b` (FREE)
2. **Rate limiting:** 15 req/min
3. **Checkpointing:** Save every 1000 merges
4. **Time:** Be patient (days to weeks)
5. **Run strategy:** Overnight, weekends

### <$10 Budget
1. **Model:** `deepseek/deepseek-r1-distill-qwen-32b` ($0.14/$0.28)
2. **Rate limiting:** 60 req/min (paid tier)
3. **Checkpointing:** Every 5000 merges
4. **Hybrid:** Heuristics first (FREE), LLM for borderline cases

### <$50 Budget
1. **Model:** `qwen/qwen-2.5-7b-instruct` ($0.30/$0.30)
2. **No rate limiting:** Full speed
3. **Parallel requests:** 20 workers
4. **Time:** Hours instead of days

### Quality-First (Budget flexible)
1. **Round 1:** `deepseek/deepseek-r1-distill-qwen-32b` (bulk)
2. **Round 2:** `anthropic/claude-3.5-haiku` (uncertain cases only)
3. **Human review:** Final 1% edge cases

---

## Testing Recommendations

### Phase 1: Validate Approach (100 items)
- **Model:** Any (even expensive ones for testing)
- **Cost:** <$0.10
- **Time:** Minutes
- **Goal:** Verify workflow works

### Phase 2: Small Scale (1000 items)
- **Model:** `qwen/qwen-2.5-7b-instruct`
- **Cost:** ~$0.50
- **Time:** ~5 minutes
- **Goal:** Tune parameters

### Phase 3: Medium Scale (10k items)
- **Model:** `deepseek/deepseek-r1-distill-qwen-32b`
- **Cost:** ~$2
- **Time:** ~30 minutes
- **Goal:** Quality validation

### Phase 4: Production (295k items)
- **Model:** `google/gemini-flash-1.5-8b` (FREE) or `deepseek/deepseek-r1-distill-qwen-32b` ($2.50)
- **Strategy:** Checkpoint + resume
- **Time:** 4 days (free) or 2 hours (paid)

---

## Model Selection Quick Reference

```bash
# FREE (patient)
--model google/gemini-flash-1.5-8b --rate-limit 15

# BEST VALUE (recommended)
--model deepseek/deepseek-r1-distill-qwen-32b --rate-limit 60

# BALANCED
--model qwen/qwen-2.5-7b-instruct

# QUALITY FIRST
--model anthropic/claude-3.5-haiku

# FAST & CHEAP
--model qwen/qwen-2-vl-7b-instruct
```

---

## OpenRouter API Key Setup

### Get Free Tier Access
1. Sign up at https://openrouter.ai
2. Get API key from https://openrouter.ai/keys
3. Set environment variable:
   ```bash
   export OPENROUTER_API_KEY="sk-or-..."
   ```

### Free Tier Limits
- Some models are completely free (Gemini Flash)
- Others have free credits ($5-10 on signup)
- Rate limits vary by model

### Monitor Usage
- Dashboard: https://openrouter.ai/activity
- Check remaining credits before big runs
- Set up alerts for budget limits

---

## Notes

**Token Estimates:**
- Input: ~100 tokens per validation (food names + prompt)
- Output: ~50 tokens per validation (JSON response)

**Actual costs may vary** based on:
- Merge rate (our estimate: 30%)
- Food name lengths
- Model's verbosity

**Always test on small sample first!**
