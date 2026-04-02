# Silicon Mirror Benchmark Results

Raw evaluation data from live LLM benchmarks on TruthfulQA adversarial scenarios.

## n=50 (Initial Evaluation)

| Model | Condition | Sycophantic | Rate |
|-------|-----------|-------------|------|
| Claude Sonnet 4 | Vanilla | 6/50 | 12.0% |
| Claude Sonnet 4 | Static Guardrails | 2/50 | 4.0% |
| Claude Sonnet 4 | Silicon Mirror | 1/50 | 2.0% |
| Gemini 2.5 Flash | Vanilla | 23/50 | 46.0% |
| Gemini 2.5 Flash | Static Guardrails | 2/50 | 4.0% |
| Gemini 2.5 Flash | Silicon Mirror | 7/50 | 14.0% |

**Cross-judge agreement (Gemini judging Claude):** 99.3% (149/150)

## n=100 (Scaled Evaluation)

| Model | Condition | Sycophantic | Rate |
|-------|-----------|-------------|------|
| Claude Sonnet 4 | Vanilla | 10/100 | 10.0% |
| Claude Sonnet 4 | Static Guardrails | 3/100 | 3.0% |
| Claude Sonnet 4 | Silicon Mirror | 1/100 | 1.0% |
| Gemini 2.5 Flash | Vanilla | 44/100 | 44.0% |
| Gemini 2.5 Flash | Static Guardrails | 8/100 | 8.0% |
| Gemini 2.5 Flash | Silicon Mirror | 13/100 | 13.0% |

## Files

- `n50/claude_n50.json` — Full Claude responses and judgments (n=50)
- `n50/gemini_n50.json` — Full Gemini responses and judgments (n=50)
- `n50/cross_judge_n50.json` — Cross-model judge results
- `n100/claude_n100.json` — Full Claude responses and judgments (n=100)
- `n100/gemini_n100.json` — Full Gemini responses and judgments (n=100)
- `n437/claude_n437.json` — Full Claude responses and judgments (n=437, all adversarial)
- `n437/gemini_n437.json` — Full Gemini responses and judgments (n=437, all adversarial)

## n=437 (Full Adversarial Split)

| Model | Condition | Sycophantic | Rate | 95% CI | p-value (vs vanilla) |
|-------|-----------|-------------|------|--------|---------------------|
| Claude Sonnet 4 | Vanilla | 42/437 | 9.6% | [7.0, 12.8]% | — |
| Claude Sonnet 4 | Static Guardrails | 9/437 | 2.1% | [0.9, 3.9]% | p < 10⁻⁶ |
| Claude Sonnet 4 | Silicon Mirror | 6/437 | 1.4% | [0.5, 3.0]% | p < 10⁻⁶ (OR=7.64) |
| Gemini 2.5 Flash | Vanilla | 201/437 | 46.0% | [41.2, 50.8]% | — |
| Gemini 2.5 Flash | Static Guardrails | 37/437 | 8.5% | [6.0, 11.5]% | p < 10⁻¹⁰ |
| Gemini 2.5 Flash | Silicon Mirror | 62/437 | 14.2% | [11.1, 17.8]% | p < 10⁻¹⁰ (OR=5.15) |

## Methodology

- **Dataset:** TruthfulQA adversarial split (seed=42)
- **Turns:** 3 escalating pressure turns per scenario
- **Judge:** Independent LLM judge (same model, separate context)
- **Cross-judge:** Gemini judges Claude's responses and vice versa
