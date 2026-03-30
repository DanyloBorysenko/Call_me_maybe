from llm_sdk import Small_LLM_Model
from typing import Dict
import numpy as np
import json


class VocabIndex():
    def __init__(self, model: Small_LLM_Model):
        self.model = model
        self._build_vocab_index()

    def _build_vocab_index(self) -> None:
        """Builds vocabulary index and token groups for constrained decoding.
        Extracts all tokens from the model and categorizes them into:
        - numeric tokens (with and without terminators)
        - string-safe tokens
        - regex-safe tokens
        - special structural tokens (quotes, commas, braces, etc.)
        Args:
            model: LLM model instance.
        Returns:
            Dictionary containing token mappings
            and categorized token ID arrays.
        """
        vocab_path = self.model.get_path_to_vocab_file()
        try:
            with open(vocab_path, "r", encoding="utf-8") as f:
                vocab_json: Dict[str, int] = json.load(f)
            vocab_size = len(vocab_json)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            print(f"Warning: could not read vocab file: {e}, "
                  f"using fallback vocab size")
            exit()

        all_tokens: Dict[int, str] = {
            i: self.model.decode([i]) for i in range(vocab_size)
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

        _numeric_base = [i for i, s in all_tokens.items()
                         if _is_numeric_token(s)]
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
        self.all_tokens = all_tokens
        self.quote_id = quote_id
        self.comma_id = comma_id
        self.rbrace_id = rbrace_id
        self.numeric_base_ids = numeric_base_ids   # digits only
        self.numeric_ids = numeric_ids        # digits + terminators
        self.str_ids = str_ids
        self.negative_sign = negative_sign
        self.space_minus = space_minus
        self.close_parenth = close_parenth
        self.close_sq_bracket = close_sq_bracket
        self.regex_ids = regex_ids
        self.slash = slash
        self.space_slash = space_slash
