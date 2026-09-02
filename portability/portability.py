import os
import difflib

import json
from pathlib import Path

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import SystemMessage, HumanMessage


load_dotenv()


def create_model(provider):
    if provider == "openrouter":
        model = ChatOpenRouter(
            api_key=os.environ["OPENROUTER_API_KEY"],
            model=os.environ["MODEL_NAME"],
            base_url=os.environ["BASE_URL"],
            temperature=0,
            timeout=20_000,
            max_retries=0,
        )

        return model.with_retry(
            stop_after_attempt=3,
            wait_exponential_jitter=True,
        )

    if provider == "gemini":
        model = ChatGoogleGenerativeAI(
            google_api_key=os.environ["GOOGLE_API_KEY"],
            model=os.environ["GOOGLE_MODEL"],
            temperature=0,
            timeout=60,
            max_retries=0,
        )

        return model.with_retry(
            stop_after_attempt=3,
            wait_exponential_jitter=True,
        )

    raise ValueError(f"Unsupported provider: {provider}")


def build_messages():
    return [
        SystemMessage(
            content="Answer briefly and clearly."
        ),
        HumanMessage(
            content="Explain retrieval augmented generation in one sentence."
        ),
    ]


def invoke_provider(provider):
    model = create_model(provider)
    messages = build_messages()

    response = model.invoke(messages)

    content = response.content

    if isinstance(content, str):
        output = content.strip()

    elif isinstance(content, list):
        parts = []

        for item in content:
            if isinstance(item, dict) and "text" in item:
                parts.append(item["text"])
            elif isinstance(item, str):
                parts.append(item)

        output = " ".join(parts).strip()

    else:
        raise ValueError(
            f"OUTPUT_QUARANTINED: unsupported response type {type(content)}"
        )

    if not output:
        raise ValueError(
            "OUTPUT_QUARANTINED: provider returned empty output."
        )

    return output


def compare_outputs(output_a, output_b):
    similarity = difflib.SequenceMatcher(
        None,
        output_a,
        output_b,
    ).ratio()

    return {
        "exact_match": output_a == output_b,
        "similarity": round(similarity, 3),
        "length_difference": abs(
            len(output_a) - len(output_b)
        ),
    }


def main():
    openrouter_output = invoke_provider("openrouter")
    gemini_output = invoke_provider("gemini")

    comparison = compare_outputs(
        openrouter_output,
        gemini_output,
    )

    trace = {
    "task": "portability",
    "providers": ["openrouter", "gemini"],
    "openrouter_model": os.environ["MODEL_NAME"],
    "gemini_model": os.environ["GOOGLE_MODEL"],
    "openrouter_output": openrouter_output,
    "gemini_output": gemini_output,
    "exact_match": comparison["exact_match"],
    "textual_similarity": comparison["similarity"],
    "length_difference": comparison["length_difference"],
}

    Path("traces").mkdir(exist_ok=True)

    with open("traces/portability_trace.json", "w") as file:
        json.dump(trace, file, indent=4)

    print("\n--- OpenRouter ---")
    print(openrouter_output)

    print("\n--- Gemini ---")
    print(gemini_output)

    print("\n--- Comparison ---")
    print("Exact match:", comparison["exact_match"])
    print("Similarity:", comparison["similarity"])
    print(
        "Length difference:",
        comparison["length_difference"],
    )


if __name__ == "__main__":
    main()