import os

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage

load_dotenv()


primary_model = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model=os.environ["MODEL_NAME"],
    base_url=os.environ["BASE_URL"],
    temperature=0,
    timeout=20_000,
    max_retries=0,
)


fallback_model = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model=os.environ["FALLBACK_MODEL_NAME"],
    base_url=os.environ["BASE_URL"],
    temperature=0,
    timeout=20_000,
    max_retries=0,
)

primary_with_retry = primary_model.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)

resilient_model = primary_with_retry.with_fallbacks(
    [fallback_model]
)

def build_messages():
    return [
        SystemMessage(
            content="Answer briefly and clearly."
        ),
        HumanMessage(
            content="What is the capital of France?"
        ),
    ]

def run_resilient_model():
    messages = build_messages()

    response = resilient_model.invoke(messages)

    return response

if __name__ == "__main__":
    response = run_resilient_model()
    print("Response:", response.content)