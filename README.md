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

