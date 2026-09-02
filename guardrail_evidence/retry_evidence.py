from langchain_core.runnables import RunnableLambda


attempts = {"count": 0}


def flaky_call(value):
    attempts["count"] += 1
    print(f"ATTEMPT={attempts['count']}")

    if attempts["count"] < 3:
        raise ConnectionError("Simulated transient failure.")

    return "Retry succeeded"


runnable = RunnableLambda(flaky_call).with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)

result = runnable.invoke("test")

print("RESULT:", result)