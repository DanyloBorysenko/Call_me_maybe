from .config_parser import Function
from typing import List, Dict, Any
from llm_sdk import Small_LLM_Model
import numpy as np


def get_function_name(model: Small_LLM_Model,
                      prompt: str,
                      functions: List[Function]) -> Function:
    function_names = [funct.name for funct in functions]
    name_func_dict = {func.name: func for func in functions}
    funct_name_tokens = {}
    for name in function_names:
        funct_name_tokens[name] = model.encode(name).tolist()[0]
    prefix = "Available functions:\n"
    for func in functions:
        prefix += f"{func.name}: {func.description}\n"
    prefix += f'\nUser prompt: {prompt}\n'
    prefix += "{"
    prefix += f'"prompt": "{prompt}", "name": "'
    generated_tokens = []
    prefix_input_ids = model.encode(prefix).tolist()[0]
    max_tokens_length = len(max(funct_name_tokens.values(),
                            key=lambda tokens: len(tokens)))
    for _ in range(max_tokens_length):
        allowed = set()
        for name, tokens in funct_name_tokens.items():
            generated_len = len(generated_tokens)
            if tokens[:generated_len] == generated_tokens:
                if len(tokens) > generated_len:
                    allowed.add(tokens[generated_len])
        if not allowed:
            raise RuntimeError("No valid tokens available")
        logits = model.get_logits_from_input_ids(prefix_input_ids)
        next_token = max(allowed, key=lambda i: logits[i])
        generated_tokens.append(next_token)
        prefix_input_ids.append(next_token)
        for name, tokens in funct_name_tokens.items():
            if tokens == generated_tokens:
                return name_func_dict[name]
    raise RuntimeError("No function name found within token limit")


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
                    allowed_ids = numeric_base_ids
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
