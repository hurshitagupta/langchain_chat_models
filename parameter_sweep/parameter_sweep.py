import os

from dotenv import load_dotenv
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage

from common.guards import StepCounter, check_token_budget, validate_output


load_dotenv()


def create_model(temperature):
    model = ChatOpenRouter(
        api_key=os.environ["OPENROUTER_API_KEY"],
        model=os.environ["MODEL_NAME"],
        base_url=os.environ["BASE_URL"],
        temperature=temperature,
        timeout=20_000,
        max_retries=0,
    )

    return model.with_retry(
        stop_after_attempt=3,
        wait_exponential_jitter=True,
    )


def build_messages():
    return [
        SystemMessage(
            content="You are a creative naming assistant."
        ),
        HumanMessage(
            content=(
                "Suggest one short creative tagline for an AI travel assistant. "
                "Use 3 to 6 words. Return only the tagline."
            )
        )
    ]


def run_temperature(temperature, step_counter, runs=10):
    if runs > 10:
        raise ValueError("Maximum 10 runs allowed per temperature.")

    model = create_model(temperature)
    messages = build_messages()

    # Token budget guard
    token_count = len(" ".join(message.content for message in messages).split())
    check_token_budget(token_count, max_tokens=100)

    outputs = []

    for run in range(runs):

        step_counter.check()

        response = model.invoke(messages)

        output = response.content.strip()

        if not output:
            raise ValueError(
                "OUTPUT_QUARANTINED: model returned an empty response."
            )

        outputs.append(output)

        print(
            f"Temperature={temperature} | "
            f"Run={run + 1} | "
            f"Output={output}"
        )

    return outputs


def measure_variance(outputs):
    if not outputs:
        raise ValueError("Outputs cannot be empty.")

    unique_outputs = len(set(outputs))
    total_runs = len(outputs)
    variance_rate = unique_outputs / total_runs

    return unique_outputs, variance_rate


def main():
    temperatures = [0.0, 0.7, 1.0]

    # 3 temperatures × 10 runs = maximum 30 model calls
    step_counter = StepCounter(max_steps=30)

    for temperature in temperatures:
        print(f"\n--- Temperature {temperature} ---")

        outputs = run_temperature(
            temperature,
            step_counter,
        )

        unique_outputs, variance_rate = measure_variance(outputs)

        print(
            f"Summary | "
            f"Temperature={temperature} | "
            f"Runs={len(outputs)} | "
            f"Unique Outputs={unique_outputs} | "
            f"Variance Rate={variance_rate:.2f}"
        )


if __name__ == "__main__":
    main()