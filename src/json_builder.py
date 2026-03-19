from .parser import Function, Prompt
from typing import List, Dict, Any, Tuple
from llm_sdk import Small_LLM_Model
import numpy as np
import json


class JsonBuildingError(Exception):
    pass


def build_vocab_index(model) -> Dict:
    vocab_path = model.get_path_to_vocab_file()
    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_json = json.load(f)
        vocab_size = len(vocab_json)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: could not read vocab file: {e}, "
              f"using fallback vocab size")
        vocab_size = 151936

    all_tokens: Dict[int, str] = {
        i: model.decode([i]) for i in range(vocab_size)
    }
    # print(str(vocab_json)[:200])

    def _find_exact_token(target: str) -> int:
        matches = [i for i, s in all_tokens.items() if s == target]
        if not matches:
            raise ValueError(f"Token '{target}' not found in vocab")
        return matches[0]

    def _is_numeric_token(s: str) -> bool:
        # The + sign was intentionally left out
        # because in the context of JSON number values, '+' is not valid.
        if len(s) == 0:
            return False
        if not all(c in "0123456789.-" for c in s):
            return False
        not_numeric = ["..", "--", ".-", "-.", "- "]
        for not_num in not_numeric:
            if not_num in s:
                return False
        return True

    _numeric_base = [i for i, s in all_tokens.items() if _is_numeric_token(s)]
    numeric_base_ids = np.array(_numeric_base, dtype=np.int64)

    # The purpose of this section is to find the token IDs for
    # JSON structural terminators — characters that signal
    # "this value is finished, move to the next one"
    quote_id = _find_exact_token('"')
    comma_id = _find_exact_token(',')
    rbrace_id = _find_exact_token('}')
    negative_sign = _find_exact_token('-')
    space_minus = _find_exact_token(' -')

    numeric_ids = np.array(_numeric_base + [comma_id, rbrace_id],
                           dtype=np.int64)

    # exclude structurally dangerous characters from strings
    _unsafe_chars = {'{', '}', '\n', '\r', '\t'}
    ids = []
    for id, token in all_tokens.items():
        if not token:
            continue
        if token == '"':
            ids.append(id)
        elif '"' in token:
            continue
        else:
            if not any(c in _unsafe_chars for c in token):
                ids.append(id)
    str_ids = np.array(ids, dtype=np.int64)

    return {
        "all_tokens":       all_tokens,
        "vocab_size":       vocab_size,
        "quote_id":         quote_id,
        "comma_id":         comma_id,
        "rbrace_id":        rbrace_id,
        "numeric_base_ids": numeric_base_ids,   # digits only
        "numeric_ids":      numeric_ids,        # digits + terminators
        "str_ids":          str_ids,
        "negative_sign":    negative_sign,
        "space_minus":      space_minus
    }


def find_function_name(model: Small_LLM_Model,
                       prefix: str,
                       prompt: str,
                       func_name_tokens: Dict[str, List[int]],
                       ) -> str:
    prefix += f'\nUser prompt: {prompt}\n'
    prefix += "{"
    prefix += f'"prompt": "{prompt}", "name": "'
    generated_tokens = []
    prefix_input_ids = model.encode(prefix).tolist()[0]
    max_tokens_length = len(max(func_name_tokens.values(),
                            key=lambda tokens: len(tokens)))
    for _ in range(max_tokens_length):
        allowed = set()
        for tokens in func_name_tokens.values():
            generated_len = len(generated_tokens)
            if tokens[:generated_len] == generated_tokens:
                if len(tokens) > generated_len:
                    allowed.add(tokens[generated_len])
        logits = model.get_logits_from_input_ids(prefix_input_ids)
        next_token = max(allowed, key=lambda i: logits[i])
        generated_tokens.append(next_token)
        prefix_input_ids.append(next_token)
        for name, tokens in func_name_tokens.items():
            if tokens == generated_tokens:
                return name
    raise RuntimeError()


def _masked_argmax(logits: List[float], allowed_ids: np.ndarray) -> int:
    """Mask all tokens except allowed_ids and
    return the highest-logit token id."""
    arr = np.array(logits, dtype=np.float32)
    mask = np.full(len(arr), -np.inf, dtype=np.float32)
    mask[allowed_ids] = arr[allowed_ids]
    return int(np.argmax(mask))


def clean_and_cast(raw: str, param_type: str) -> Any:
    raw = raw.strip()
    if param_type in ("int", "float", "number"):
        try:
            return int(raw) if "." not in raw else float(raw)
        except ValueError:
            cleaned = ''.join(c for c in raw if c in "0123456789.-")
            try:
                return int(cleaned) if "." not in cleaned else float(cleaned)
            except ValueError:
                return None
    return raw


def get_parameters(
    model,
    function: Function,
    prompt: str,
    vocab: dict,
    max_str_tokens: int = 80,
    max_num_tokens: int = 20
) -> Dict[str, Any]:

    all_tokens = vocab["all_tokens"]
    quote_id = vocab["quote_id"]
    numeric_ids = vocab["numeric_ids"]
    numeric_base_ids = vocab["numeric_base_ids"]
    negative_sign_id = vocab["negative_sign"]
    space_minus = vocab["space_minus"]
    str_ids = vocab["str_ids"]

    parameters: Dict[str, Any] = {}
    # if you want a literal '{' or '}' character in the output,
    # you must double it
    prefix_str = (
        f'Available functions:\n{function.name}: {function.description}\n'
        f'\nUser prompt: {prompt}\n'
        f'{{ "prompt": "{prompt}", '
        f'"name": "{function.name}", '
        f'"parameters": {{'
    )

    input_ids: List[int] = model.encode(prefix_str).tolist()[0]
    param_items = list(function.parameters.items())

    # prompt_token_ids = np.array(model.encode(prompt).tolist()[0])

    for idx, (name, param) in enumerate(param_items):

        is_last = (idx == len(param_items) - 1)
        ptype = param.type

        key_str = f'"{name}": '
        if ptype == "string":
            key_str += '"'

        input_ids.extend(model.encode(key_str).tolist()[0])

        val_ids: List[int] = []
        while True:
            val_ids_len = len(val_ids)
            logits = model.get_logits_from_input_ids(input_ids)

            # ---------- numeric ----------
            if ptype in ("int", "float", "number"):
                if val_ids_len >= max_num_tokens:
                    break
                if val_ids_len == 0:
                    allowed_ids = np.union1d(numeric_base_ids, np.array([space_minus], dtype=np.int64))
                    if '-' in prompt and '-' != prompt[len(prompt) - 1] and prompt[prompt.find('-') + 1] in "0123456789":
                        logits[space_minus] += 4.0
                        logits[negative_sign_id] += 4.0
                else:
                    allowed_ids = numeric_ids

            # ---------- string ----------
            else:
                if val_ids_len >= max_str_tokens:
                    allowed_ids = np.array([quote_id], dtype=np.int64)
                # elif name == "regex":
                #     prompt_based = np.intersect1d(str_ids, prompt_token_ids)
                #     allowed_ids = np.union1d(
                #         prompt_based,
                #         np.array([quote_id], dtype=np.int64)
                #     )
                else:
                    allowed_ids = str_ids

            next_token = _masked_argmax(logits, allowed_ids)
            t_str = all_tokens[next_token]

            # Repetition guard — stop if any token pattern repeats back-to-back
            stop = False
            if ptype == "string" and val_ids_len >= 6:
                for window in (2, 3, 4, 5, 6):
                    if val_ids_len >= window * 2:
                        if val_ids[-window:] == val_ids[-window*2:-window]:
                            val_ids = val_ids[:-window]
                            stop = True
                            break
                if stop is True:
                    break

            # ---------- termination ----------
            is_done = (
                (ptype == "string" and t_str == '"')
                or (ptype in ("int", "float", "number")
                    and t_str in (",", "}"))
            )

            if is_done:
                break

            val_ids.append(next_token)
            input_ids.append(next_token)

        raw_val = model.decode(val_ids).strip()
        if name == "regex" and ".*" in raw_val:
            raw_val = raw_val.split(".*")[0]
        parameters[name] = clean_and_cast(raw_val, ptype)

        # ---------- append separators ----------
        if ptype == "string":
            sep = '"' + (", " if not is_last else "")
        else:
            sep = ", " if not is_last else ""

        if sep:
            input_ids.extend(model.encode(sep).tolist()[0])

    return parameters


def create_output(
        functions: List[Function],
        prompts: List[Prompt],
        model: Small_LLM_Model
        ) -> str:
    vocab = build_vocab_index(model)
    prefix = "Available functions:\n"
    name_function_dict = {}
    func_name_tokens = {}
    for func in functions:
        prefix += f"{func.name}: {func.description}\n"
        func_name_tokens[func.name] = model.encode(func.name).tolist()[0]
        name_function_dict[func.name] = func
    jsons = []
    for prompt in prompts:
        try:
            function_name = find_function_name(model,
                                               prefix, prompt.prompt,
                                               func_name_tokens)
        except RuntimeError:
            raise JsonBuildingError("No function name found")
        function = name_function_dict[function_name]
        parameters = get_parameters(model, function, prompt.prompt, vocab)
        result = {
            "prompt": prompt.prompt,
            "name": function.name,
            "parameters": parameters
            }
        jsons.append(result)
    try:
        return json.dumps(jsons, indent=2)
    except Exception as e:
        raise JsonBuildingError(f"json.dumps operation failed, "
                                f"{e}")


def get_top_logits(logits: List[float],
                   allowed_ids: np.ndarray,
                   el_count: int,
                   all_tokens: Dict[int, str]) -> None:
    ind_logit_tuples = [(ind, value) for ind, value in enumerate(logits)
                        if ind in allowed_ids]
    sorted_logits = sorted(ind_logit_tuples, key=(lambda tup: tup[1]),
                           reverse=True)
    top = sorted_logits[:el_count]
    print("Top biggest logits:")
    for id, value in top:
        print(f"'{all_tokens[id]}' : {value}")
