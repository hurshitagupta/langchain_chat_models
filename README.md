# LangChain Chat Models Assessment

This assessment focuses on working directly with chat models in LangChain — constructing message histories, controlling model behaviour, handling failures, measuring usage and cost, and keeping the implementation portable across providers.

The project uses **OpenRouter** for model access, with all credentials loaded through environment variables.

---

## Task 1 — Message Construction

The goal of this task was to understand how a chat model receives conversation history through `SystemMessage`, `HumanMessage`, and `AIMessage`, and how the **order of those messages affects the model's response**.

I created two message sequences containing the same core messages but arranged them differently.

### Sequence A

```text
System → Human(ORANGE) → AI(PURPLE) → Human(question)
```

Output:

```text
PURPLE
```

### Sequence B

```text
System → AI(PURPLE) → Human(ORANGE) → Human(question)
```

Output:

```text
ORANGE
```

The model and temperature remain the same for both calls. Only the message order changes.

This demonstrates that chat models interpret messages as an ordered conversation history, so changing the position of Human and AI messages changes the context available when answering the final question.

### Run

```bash
uv run python -m message_construction.message_construction
```

### Tests

Automated tests verify:

* correct construction of the two message sequences
* correct placement of `HumanMessage` and `AIMessage`
* rejection of an invalid empty message sequence

Run:

```bash
uv run python -m pytest tests/test_message_construction.py -v
```

The saved execution and pytest outputs are available in the `outputs/` folder.

### Guardrails

The task includes:

* step limit for model calls
* per-call timeout
* capped retry using LangChain's `with_retry`
* input token-budget checking
* message validation
* model-output validation
* environment-only API credentials

Dedicated failure evidence for the implemented guardrails is stored in the `outputs/` folder.

### Trace

The execution also creates a structured trace at:

```text
traces/message_construction_trace.json
```

This records the model configuration and outputs of both message sequences without storing API credentials.

---
## Task 2 — Parameter Sweep

The goal of this task was to observe how changing the model's `temperature` affects variation in generated responses.

I used the same model, provider, and prompt for every request and changed only the temperature.

The experiment was run at:

```text id="d8cvss"
Temperature 0.0 → 10 runs
Temperature 0.7 → 10 runs
Temperature 1.0 → 10 runs
```

This resulted in a total of 30 model calls.

The prompt asks the model to generate a short creative tagline for an AI travel assistant. A creative prompt was used so that the model had enough freedom for differences in sampling behaviour to become observable.

### Measuring Variation

For each temperature, all 10 responses are collected and the number of unique outputs is calculated.

The variance rate used in this experiment is:

```text id="etcy6u"
variance rate = unique outputs / total runs
```

This provides a simple measurable comparison instead of relying on a subjective judgement of how "creative" the responses appear.

During experimentation, temperature `0` could still produce different responses across repeated calls. This was treated as an observed result rather than assuming that temperature `0` guarantees identical responses. The experiment therefore reports the actual outputs and measured variation for each temperature.

### Run

```bash id="mznxfn"
uv run python -m parameter_sweep.parameter_sweep
```

The complete 30-run experiment output is saved in:

```text id="7rnj4a"
outputs/parameter_sweep_output.txt
```

### Tests

The automated tests focus on the deterministic measurement logic rather than making live API calls.

They verify:

* correct calculation of unique outputs and variance rate
* rejection of an empty output collection

Run:

```bash id="6lphvy"
uv run python -m pytest tests/test_parameter_sweep.py -v
```

The saved pytest output is available in:

```text id="j5yteh"
outputs/parameter_sweep_tests.txt
```

Keeping live model calls outside the automated tests makes the tests deterministic and prevents provider behaviour or network availability from affecting the test result.

### Guardrails

The parameter sweep also uses the shared assessment protections:

* hard limit on the number of model calls
* per-call timeout
* capped retries using LangChain's `with_retry`
* input budget checking before model invocation
* rejection of empty model responses
* API credentials loaded only from environment variables

The experiment is capped at 10 runs per temperature and 30 model calls across the complete sweep.
---

## Task 3 — Resilience

The goal of this task was to make chat-model calls resilient to temporary provider failures by combining a timeout, capped retries with exponential backoff, and a fallback model.

The implementation uses two OpenRouter chat models: a primary model and a fallback model.

The primary model is first wrapped using LangChain's `with_retry()` and the retry-enabled runnable is then wrapped with `with_fallbacks()`.

The resulting flow is:

```text id="9h3cl6"
Primary Model
     ↓
Capped Retry
     ↓
Primary still fails
     ↓
Fallback Model
     ↓
Response
```

### Timeout

Both model calls have a per-call timeout configured so that a provider request cannot block execution indefinitely.

A deterministic timeout case is also included in the guardrail evidence rather than depending on an external provider to become slow during execution.

Evidence:

```text id="5dyytb"
outputs/timeout_evidence.txt
```

### Retry

Provider-level automatic retries are disabled and retry behaviour is handled explicitly at the LangChain Runnable layer using:

```python id="ehdvub"
.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)
```

This caps the number of attempts while using exponential backoff with jitter between failed attempts.

The retry evidence uses a deterministic transient failure that fails twice before succeeding on the third attempt.

Example:

```text id="v7e4g7"
ATTEMPT=1
ATTEMPT=2
ATTEMPT=3
RESULT: Retry succeeded
```

Evidence:

```text id="x1sz0k"
outputs/retry_evidence.txt
```

### Fallback Model

If the primary model still fails after its retry attempts are exhausted, LangChain's `with_fallbacks()` routes the request to the configured fallback model.

A deterministic failure case is included to prove that the fallback path is actually executed rather than only configured.

Example:

```text id="0ipn47"
PRIMARY_MODEL_FAILED
FALLBACK_MODEL_INVOKED
Result: Fallback response
```

Evidence:

```text id="ex9e3c"
outputs/fallback_evidence.txt
```

### Run

The normal resilience flow can be executed with:

```bash id="of0hk3"
uv run python -m resilience.resilience
```

The saved execution output is available in:

```text id="y2h1yz"
outputs/resilience_output.txt
```

### Tests

The automated tests avoid relying on real provider failures.

They verify:

* successful recovery when the primary fails and the fallback succeeds
* graceful failure when both the primary and fallback fail

Run:

```bash id="n34ovg"
uv run python -m pytest tests/test_resilience.py -v
```

The saved pytest output is available in:

```text id="o4qq5p"
outputs/resilience_tests.txt
```
---

## Task 4 — Usage Accounting

The goal of this task was to read token usage from the model response metadata and calculate the USD cost of a chat-model request.

The OpenRouter response exposes token usage through `usage_metadata`, including:

```text
input_tokens
output_tokens
total_tokens
```

The task extracts these values directly from the actual model response rather than estimating them locally.

### Cost Calculation

The model used for this task is priced separately for input and output tokens.

The calculated cost is based on:

```text
input cost  = input tokens / 1,000,000 × input price
output cost = output tokens / 1,000,000 × output price

total cost = input cost + output cost
```

The implementation also reads OpenRouter's provider-reported cost from the response metadata and prints it alongside the manually calculated value for comparison.

### Run

```bash
uv run python -m usage_accounting.usage_accounting
```

The saved execution output is available in:

```text
outputs/usage_accounting_output.txt
```

The output includes:

* model response
* input token count
* output token count
* total token count
* manually calculated USD cost
* provider-reported USD cost

### Tests

The automated tests verify:

* correct cost calculation from known token counts and prices
* rejection of invalid negative token counts

Run:

```bash
uv run python -m pytest tests/test_usage_accounting.py -v
```

The saved pytest output is available in:

```text
outputs/usage_accounting_tests.txt
```

### Guardrails

The task reuses the shared assessment protections, including:

* model-call step limit
* per-call timeout
* capped retry behaviour
* input token-budget checking
* model output validation
* environment-only API credentials

### Trace

The task generates a structured trace at:

```text
traces/usage_accounting_trace.json
```

The trace records token usage and cost measurements without storing API credentials.
---

## Task 5 — Portability

The goal of this task was to run the same prompt against two different model providers behind one interface and compare their responses.

The implementation uses:

```text id="7uwgq8"
Provider 1 → OpenRouter
Provider 2 → Google Gemini
```

Provider-specific model creation is handled by `create_model()`, while both models are invoked through the same `invoke_provider()` interface.

The exact same system and human messages are therefore sent to both providers without changing the calling code.

### Response Normalization

During implementation, an important provider difference became visible.

OpenRouter returned `response.content` as a string, while Gemini could return structured content as a list.

The portability layer normalizes both response formats into a validated text string before returning the result to the caller.

This keeps provider-specific response formatting outside the rest of the application.

### Output Comparison

The two responses are compared using:

* exact text match
* textual similarity score
* response length difference

One observed run produced:

```text id="jj1i5m"
Exact match: False
Similarity: 0.207
Length difference: 2
```

The similarity value represents textual sequence similarity rather than semantic similarity.

### Run

```bash id="rrvn6s"
uv run python -m portability.portability
```

Saved output:

```text id="0smrkt"
outputs/portability_output.txt
```

### Tests

The automated tests verify:

* successful comparison of two provider outputs
* rejection of an unsupported provider

The tests operate on deterministic local logic rather than depending on live provider responses.

Run:

```bash id="21j2au"
uv run python -m pytest tests/test_portability.py -v
```

Saved pytest output:

```text id="83s1aq"
outputs/portability_tests.txt
```

### Trace

The execution generates:

```text id="qfkwug"
traces/portability_trace.json
```

The trace records the providers, model names, normalized responses, and comparison measurements without storing API credentials.

### Guardrails

The portability task reuses the shared assessment protections:

* hard limit of two provider calls
* per-call timeout on both OpenRouter and Gemini
* capped retry using LangChain's `with_retry`
* input token-budget validation before provider invocation
* response normalization and output validation
* API credentials loaded only from environment variables

The reusable guardrail failure evidence is stored in the existing `outputs/` guardrail evidence files and is not duplicated per task.

---




