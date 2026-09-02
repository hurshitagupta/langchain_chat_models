from langchain_core.runnables import RunnableLambda


def failing_primary(messages):
    print("PRIMARY_MODEL_FAILED")
    raise ConnectionError("Simulated primary provider failure.")


def working_fallback(messages):
    print("FALLBACK_MODEL_INVOKED")
    return "Fallback response"


primary = RunnableLambda(failing_primary)

fallback = RunnableLambda(working_fallback)

resilient = primary.with_fallbacks([fallback])

result = resilient.invoke("test")

print("Result:", result)