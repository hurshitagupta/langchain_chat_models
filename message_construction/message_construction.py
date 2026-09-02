import os
import json
from langchain_openrouter import ChatOpenRouter
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from common.guards import validate_output, StepCounter, check_token_budget

from dotenv import load_dotenv

load_dotenv()

model = ChatOpenRouter(
    api_key=os.environ["OPENROUTER_API_KEY"],
    model=os.environ["MODEL_NAME"],
    base_url=os.environ["BASE_URL"],
    temperature=0,
    timeout=20_000
)

model_with_retry = model.with_retry(
    stop_after_attempt=3,
    wait_exponential_jitter=True,
)

def build_messages():
    messages_a = [
        SystemMessage(
            content="Return only the content of the message immediately before the final human message."
        ),
        HumanMessage(content="ORANGE"),
        AIMessage(content="PURPLE"),
        HumanMessage(content="What was the message immediately before this one?"),
    ]

    messages_b = [
        SystemMessage(
            content="Return only the content of the message immediately before the final human message."
        ),
        AIMessage(content="PURPLE"),
        HumanMessage(content="ORANGE"),
        HumanMessage(content="What was the message immediately before this one?"),
    ]

    return messages_a, messages_b


def run_experiment():

    sequence_a, sequence_b = build_messages()

    validate_messages(sequence_a)
    validate_messages(sequence_b)

    step_counter = StepCounter(max_steps=2)

    token_count_a = model.get_num_tokens_from_messages(sequence_a)
    token_count_b = model.get_num_tokens_from_messages(sequence_b)

    check_token_budget(token_count_a, max_tokens=100)
    check_token_budget(token_count_b, max_tokens=100)

    step_counter.check()
    response_a = model_with_retry.invoke(sequence_a)

    step_counter.check()
    response_b = model_with_retry.invoke(sequence_b)

    output_a = validate_output(response_a.content, "PURPLE")
    output_b = validate_output(response_b.content, "ORANGE")

    return output_a, output_b

def validate_messages(messages):
    if not messages:
        raise ValueError("Message sequence cannot be empty.")

    return True

output_a, output_b = run_experiment()

print("Sequence A: ", output_a)
print("Sequence B: ", output_b)

trace = {
    "task": "message_construction",
    "model": os.environ["MODEL_NAME"],
    "temperature": 0,
    "sequence_a_output": output_a,
    "sequence_b_output": output_b,
    "outputs_equal": output_a == output_b,
}

with open("traces/message_construction_trace.json", "w") as file:
    json.dump(trace, file, indent=4)