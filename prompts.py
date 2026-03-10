LLMs prompts: qwen,codellama,deepseek

SYSTEM_PROMPT = (
    "You are a secure code reviewer. Your job is to decide whether the given "
    "function contains a vulnerability. Reply with only one word: VULNERABLE or SAFE."
)

USER_PROMPT = (
    "Analyze the following function and determine whether it is vulnerable.\n"
    "Return only one word: VULNERABLE or SAFE.\n\n"
    "```c\n{code}\n```"
)

LABEL_STR = {1: "VULNERABLE", 0: "SAFE"}


def format_messages(code: str, label: int | None = None):
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": USER_PROMPT.format(code=code)},
    ]
    if label is not None:
        messages.append({"role": "assistant", "content": LABEL_STR[int(label)]})
    return messages


def parse_label(text: str):
    if not text:
        return None
    t = text.strip().upper()
    if "VULNERABLE" in t:
        return 1
    if "SAFE" in t:
        return 0
    return None
