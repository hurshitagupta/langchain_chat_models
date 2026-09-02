import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError


def slow_call():
    print("CALL_STARTED")
    time.sleep(3)
    return "finished"


with ThreadPoolExecutor(max_workers=1) as executor:
    future = executor.submit(slow_call)

    try:
        result = future.result(timeout=1)
        print("RESULT:", result)

    except TimeoutError:
        print("MODEL_TIMEOUT: call exceeded 1 second.")