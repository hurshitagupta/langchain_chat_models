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

