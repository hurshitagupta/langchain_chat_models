import os

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage

import json
from pathlib import Path

load_dotenv()

INPUT_PRICE_PER_MILLION = 0.03
OUTPUT_PRICE_PER_MILLION = 0.17

model = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model=os.environ["MODEL_NAME"],
    base_url=os.environ["BASE_URL"],
    temperature=0,
    timeout=20_000,
    max_retries=0,
)


messages = [
    SystemMessage(
        content="Answer briefly and clearly."
    ),
    HumanMessage(
        content="Explain retrieval augmented generation in one sentence."
    ),
]

def calculate_cost(input_tokens, output_tokens, input_price, output_price):
    if input_tokens < 0 or output_tokens < 0:
        raise ValueError("Token counts cannot be negative.")

    input_cost = (input_tokens / 1_000_000) * input_price
    output_cost = (output_tokens / 1_000_000) * output_price

    return input_cost + output_cost

def extract_usage(response):
    usage = response.usage_metadata or {}

    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    total_tokens = usage.get("total_tokens", 0)

    return input_tokens, output_tokens, total_tokens

def main():
    response = model.invoke(messages)

    input_tokens, output_tokens, total_tokens = extract_usage(response)

    calculated_cost = calculate_cost(
        input_tokens=input_tokens,
        output_tokens=output_tokens,
        input_price=INPUT_PRICE_PER_MILLION,
        output_price=OUTPUT_PRICE_PER_MILLION,
    )

    provider_cost = response.response_metadata.get("cost", 0)

    trace = {
        "task": "usage_accounting",
        "model": os.environ["MODEL_NAME"],
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
        "calculated_cost_usd": calculated_cost,
        "provider_cost_usd": provider_cost,
    }
    
    Path("traces").mkdir(exist_ok=True)
    
    with open("traces/usage_accounting_trace.json", "w") as file:
        json.dump(trace, file, indent=4)

    print("Response:", response.content)
    print("Input tokens:", input_tokens)
    print("Output tokens:", output_tokens)
    print("Total tokens:", total_tokens)
    print("Calculated cost USD:", calculated_cost)
    print("Provider reported cost USD:", provider_cost)


if __name__ == "__main__":
    main()