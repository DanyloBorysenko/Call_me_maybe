from .parser import Function, Prompt
from typing import List, Dict, Any
from llm_sdk import Small_LLM_Model
import numpy as np
import json


def build_vocab_index(model: Small_LLM_Model) -> Dict:
    """Builds vocabulary index and token groups for constrained decoding.
    Extracts all tokens from the model and categorizes them into:
    - numeric tokens (with and without terminators)
    - string-safe tokens
    - regex-safe tokens
    - special structural tokens (quotes, commas, braces, etc.)
    Args:
        model: LLM model instance.
    Returns:
        Dictionary containing token mappings and categorized token ID arrays.
    """
    vocab_path = model.get_path_to_vocab_file()
    try:
        with open(vocab_path, "r", encoding="utf-8") as f:
            vocab_json: Dict[str, int] = json.load(f)
        vocab_size = len(vocab_json)
    except (FileNotFoundError, json.JSONDecodeError) as e:
        print(f"Warning: could not read vocab file: {e}, "
              f"using fallback vocab size")
        exit()

    all_tokens: Dict[int, str] = {
        i: model.decode([i]) for i in range(vocab_size)
    }
    # print(str(vocab_json)[:200])
    # print("\n", str(all_tokens)[:200])

    def _find_exact_token(target: str) -> int:
        matches = [i for i, s in all_tokens.items() if s == target]
        if not matches:
            print(f"Token '{target}' not found in vocab")
            exit()
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
    close_parenth = _find_exact_token(')')
    close_sq_bracket = _find_exact_token(']')
    slash = _find_exact_token('/')
    space_slash = _find_exact_token(' /')

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
            if _unsafe_chars.isdisjoint(token):
                ids.append(id)
    str_ids = np.array(ids, dtype=np.int64)
    reg_ids = []
    for id, token in all_tokens.items():
        if not token:
            continue
        if token == ")" or token == "]":
            reg_ids.append(id)
        elif ")" in token or "]" in token:
            continue
        else:
            reg_ids.append(id)
    regex_ids = np.array(reg_ids, dtype=np.int64)

    return {
        "all_tokens":       all_tokens,
        "quote_id":         quote_id,
        "comma_id":         comma_id,
        "rbrace_id":        rbrace_id,
        "numeric_base_ids": numeric_base_ids,   # digits only
        "numeric_ids":      numeric_ids,        # digits + terminators
        "str_ids":          str_ids,
        "negative_sign":    negative_sign,
        "space_minus":      space_minus,
        "close_parenth":    close_parenth,
        "close_sq_bracket": close_sq_bracket,
        "regex_ids":        regex_ids,
        "slash":            slash,
        "space_slash":      space_slash
    }


def find_function_name(model: Small_LLM_Model,
                       prefix: str,
                       prompt: str,
                       func_name_tokens: Dict[str, List[int]],
                       ) -> str:
    """Selects the most likely function name using constrained decoding.

    Iteratively generates tokens while restricting choices to valid
    function name prefixes.

    Args:
        model: LLM model instance.
        prefix: Prompt prefix containing available functions.
        prompt: User input prompt.
        func_name_tokens: Mapping of function names to token sequences.

    Returns:
        The selected function name.
    """
    prefix += f'\nUser prompt: {prompt}\n'
    prefix += "{"
    prefix += f'"prompt": "{prompt}", "name": "'
    generated_tokens: List[int] = []
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
        # print(", ".join([
        #     f"{model.decode([id])}" for id in prefix_input_ids]))
        # exit()
        for name, tokens in func_name_tokens.items():
            if tokens == generated_tokens:
                return name
    return ""


def _masked_argmax(logits: List[float], allowed_ids: np.ndarray) -> int:
    """Returns the index of the highest logit among allowed tokens.

    Args:
        logits: Model output logits.
        allowed_ids: Token IDs allowed for selection.

    Returns:
        Token ID with the highest allowed logit.
    """
    arr = np.array(logits, dtype=np.float32)
    mask = np.full(len(arr), -np.inf, dtype=np.float32)
    mask[allowed_ids] = arr[allowed_ids]
    return int(np.argmax(mask))


def clean_and_cast(raw: str, param_type: str, param_name: str) -> Any:
    """Cleans and converts raw generated values to target types.

    Handles:
    - numeric casting (int/float)
    - regex post-processing (closing brackets, trimming patterns)

    Args:
        raw: Raw decoded string value.
        param_type: Expected parameter type.
        param_name: Parameter name (used for special handling).

    Returns:
        Converted or cleaned value.
    """
    raw = raw.strip()
    if param_type in ("float", "number"):
        try:
            return float(raw)
        except ValueError:
            print(f"float({raw}) operation failed")
    elif param_type == "integer":
        try:
            return int(raw)
        except ValueError:
            print(f"int({raw}) operation failed")
    elif param_name == "regex":
        if ".*" in raw:
            raw = raw.split(".*")[0]
        open_parens = raw.count("(") - raw.count(")")
        open_brackets = raw.count("[") - raw.count("]")
        raw += "]" * open_brackets
        raw += ")" * open_parens
    return raw


def get_parameters(
    model: Small_LLM_Model,
    function: Function,
    prompt: str,
    vocab: dict,
    max_str_tokens: int = 80,
    max_num_tokens: int = 50
) -> Dict[str, Any]:
    """Generates function parameters using constrained token decoding.

    Applies different token constraints based on parameter type:
    - numeric: digits + optional sign + terminators
    - string: safe tokens with quote termination
    - regex: restricted tokens + structural fixes

    Includes safeguards:
    - max token limits
    - repetition detection
    - termination conditions

    Args:
        model: LLM model instance.
        function: Selected function definition.
        prompt: User input prompt.
        vocab: Precomputed vocabulary index.
        max_str_tokens: Maximum tokens for string values.
        max_num_tokens: Maximum tokens for numeric values.

    Returns:
        Dictionary of generated parameters.
    """

    all_tokens = vocab["all_tokens"]
    quote_id = vocab["quote_id"]
    numeric_ids = vocab["numeric_ids"]
    numeric_base_ids = vocab["numeric_base_ids"]
    negative_sign_id = vocab["negative_sign"]
    space_minus = vocab["space_minus"]
    str_ids = vocab["str_ids"]
    regex_ids = vocab["regex_ids"]
    close_parenth = vocab["close_parenth"]
    close_sq_bracket = vocab["close_sq_bracket"]
    slash = vocab["slash"]
    space_slash = vocab["space_slash"]

    parameters: Dict[str, Any] = {}
    # if you want a literal '{' or '}' character in the output,
    # you must double it
    prefix_str = (
        f'Available function:\n{function.name}: {function.description}\n'
        f'\nUser prompt: {prompt}\n'
        f'{{ "prompt": "{prompt}", '
        f'"name": "{function.name}", '
        f'"parameters": {{'
    )

    input_ids: List[int] = model.encode(prefix_str).tolist()[0]
    param_items = list(function.parameters.items())

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
            if ptype in ("int", "float", "number", "integer"):
                if val_ids_len >= max_num_tokens:
                    break
                if val_ids_len == 0:
                    allowed_ids = np.union1d(numeric_base_ids,
                                             np.array([space_minus],
                                                      dtype=np.int64))
                    if (
                        '-' in prompt
                        and '-' != prompt[len(prompt) - 1]
                        and prompt[prompt.find('-') + 1].isdigit()
                    ):
                        logits[space_minus] += 4.0
                        logits[negative_sign_id] += 4.0
                else:
                    allowed_ids = numeric_ids

            # ---------- string ----------
            else:
                if val_ids_len >= max_str_tokens:
                    allowed_ids = np.array([quote_id], dtype=np.int64)
                elif name == "regex":
                    if (
                        val_ids
                        and (all_tokens[val_ids[-1]])[-1] in ("+", "?", "*")
                    ):
                        allowed_ids = np.array([close_parenth,
                                                close_sq_bracket],
                                               dtype=np.int64)
                    else:
                        allowed_ids = np.intersect1d(regex_ids, str_ids)
                else:
                    allowed_ids = str_ids
                    if name == "replacement" and val_ids_len != 0:
                        logits[quote_id] += 1
                    elif (val_ids_len == 0 and "/" in prompt):
                        logits[slash] += 1
                        logits[space_slash] += 1
            # if name == "regex":
            #     get_top_logits(logits, allowed_ids, 10, all_tokens)
            next_token = _masked_argmax(logits, allowed_ids)
            t_str = all_tokens[next_token]

            # ---------- termination ----------
            is_done = (
                (ptype == "string" and t_str == '"')
                or (ptype in ("int", "float", "number", "integer")
                    and t_str in (",", "}"))
                or (name == "regex" and val_ids and val_ids[-1] in (
                    close_parenth, close_sq_bracket))
            )

            if is_done:
                break

            val_ids.append(next_token)
            input_ids.append(next_token)
        raw_val = model.decode(val_ids)
        parameters[name] = clean_and_cast(raw_val, ptype, name)

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
    """Generates JSON output for all prompts.

    For each prompt:
    - selects a function
    - generates parameters
    - builds structured JSON result

    Args:
        functions: List of available functions.
        prompts: List of input prompts.
        model: LLM model instance.

    Returns:
        JSON string with formatted results.

    Raises:
        RuntimeError: If function selection or JSON serialization fails.
    """
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
        function_name = find_function_name(model,
                                           prefix, prompt.prompt,
                                           func_name_tokens)
        if not function_name:
            raise RuntimeError("json_builder error: No function name found")
        function = name_function_dict[function_name]
        parameters = get_parameters(model, function, prompt.prompt, vocab)
        result = {
            "prompt": prompt.prompt,
            "name": function.name,
            "parameters": parameters
            }
        jsons.append(result)
    try:
        return json.dumps(jsons, indent=4)
    except Exception as e:
        raise RuntimeError(f"json_builder error: json.dumps operation failed, "
                           f"{e}")


def get_top_logits(logits: List[float],
                   allowed_ids: np.ndarray,
                   el_count: int,
                   all_tokens: Dict[int, str]) -> None:
    """Prints top-N logits for debugging token selection.

    Args:
        logits: Model logits.
        allowed_ids: Allowed token IDs.
        el_count: Number of top elements to display.
        all_tokens: Mapping of token IDs to strings.
    """
    ind_logit_tuples = [(ind, value) for ind, value in enumerate(logits)
                        if ind in allowed_ids]
    sorted_logits = sorted(ind_logit_tuples, key=(lambda tup: tup[1]),
                           reverse=True)
    top = sorted_logits[:el_count]
    print("Top biggest logits:")
    for id, value in top:
        print(f"'{all_tokens[id]}' : {value}")
