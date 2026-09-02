def validate_output(content, expected):

    if not isinstance(content,str):
        raise ValueError("Model output is not a valid text")

    cleaned = content.strip()

    if cleaned != expected:
        raise ValueError(f"Output validation failed. Expected {expected}, got {cleaned}")

    return cleaned


class StepLimitError(Exception):
    pass


class StepCounter:
    def __init__(self, max_steps):
        self.max_steps = max_steps
        self.steps = 0

    def check(self):
        if self.steps >= self.max_steps:
            raise StepLimitError(
                f"STEP_LIMIT_REACHED: maximum {self.max_steps} model calls allowed."
            )

        self.steps += 1


class TokenBudgetError(Exception):
    pass


def check_token_budget(token_count:int, max_tokens: int):
    if token_count > max_tokens:
        raise TokenBudgetError(
            f"TOKEN_BUDGET_EXCEEDED: {token_count} tokens exceeds "
            f"the maximum budget of {max_tokens}."
        )

    return True
    

    
