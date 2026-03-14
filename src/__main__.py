import sys
from .arg_validator import ArgValidator
from .config_parser import ConfigParser, ParserError
from .json_creator import get_function_name, get_parameters
from pydantic import ValidationError
from typing import Dict
from llm_sdk import Small_LLM_Model
import numpy as np
import json


def build_vocab_index(model) -> Dict:
    vocab_path = model.get_path_to_vocab_file()
    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
        vocab_size = len(vocab_json)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: could not read vocab file: {e}, using fallback vocab size")
        vocab_size = 151936

    all_tokens: Dict[int, str] = {
        i: model.decode([i]) for i in range(vocab_size)
    }

    def _find_exact_token(target: str) -> int:
        matches = [i for i, s in all_tokens.items() if s == target]
        if not matches:
            raise ValueError(f"Token '{target}' not found in vocab")
        return matches[0]

    def _is_numeric_token(s: str) -> bool:
        return len(s) > 0 and all(c in "0123456789.-" for c in s)

    quote_id = _find_exact_token('"')
    comma_id = _find_exact_token(',')
    rbrace_id = _find_exact_token('}')

    # ── Fix 1: separate base (no terminators) from full (with terminators)
    _numeric_base = [i for i, s in all_tokens.items() if _is_numeric_token(s)]
    numeric_base_ids = np.array(_numeric_base, dtype=np.int64)
    numeric_ids = np.array(_numeric_base + [comma_id, rbrace_id], dtype=np.int64)

    bool_ids = np.array(
        [
            i for i, s in all_tokens.items()
            if "true".startswith(s.lower()) or "false".startswith(s.lower())
        ],
        dtype=np.int64
    )

    # ── Fix 3: exclude structurally dangerous characters from strings
    _UNSAFE_CHARS = {'{', '}', '[', ']', '\n', '\r', '\t'}
    str_ids = np.array(
        [
            i for i, s in all_tokens.items()
            if not any(c in _UNSAFE_CHARS for c in s)
            and not ('"' in s and len(s) > 1)  # exclude multi-char tokens containing quote
        ],
        dtype=np.int64
    )

    return {
        "all_tokens":       all_tokens,
        "vocab_size":       vocab_size,
        "quote_id":         quote_id,
        "comma_id":         comma_id,
        "rbrace_id":        rbrace_id,
        "numeric_base_ids": numeric_base_ids,   # digits only
        "numeric_ids":      numeric_ids,         # digits + terminators
        "bool_ids":         bool_ids,
        "str_ids":          str_ids,
    }


def main() -> None:
    config_files = {
                "--functions_definition": "data/input/functions_definition."""
                "json",
                "--input": "data/input/function_calling_tests.json",
                "--output": "data/output/function_calls.json"
    }
    try:
        arg_validator = ArgValidator(args=sys.argv, config_files=config_files)
        config_files = arg_validator.config_files
        parser = ConfigParser(**config_files)
    except ValidationError as e:
        print(e.errors()[0]["msg"].removeprefix("Value error, "))
    try:
        functions = parser.load_functions()
        prompts = parser.load_prompts()
        print(f"FUNCTIONS - {functions}\n")
        print(f"PROMPTS - {prompts}")
    except ParserError as e:
        print(e)
    model = Small_LLM_Model()
    jsons = []
    vocab = build_vocab_index(model)
    for prompt in prompts:
        function = get_function_name(model, prompt.prompt, functions)
        result = "{"
        result += f'"prompt": "{prompt.prompt}", "name": "{function.name}", '
        result += '"parameters": '
        params = get_parameters(model, function, prompt.prompt, vocab)
        result += f"{json.dumps(params)}"
        result += "}"
        jsons.append(result)
        output = "[\n" + ",\n".join(jsons) + "\n]"
    print(output)


if __name__ == "__main__":
    main()
