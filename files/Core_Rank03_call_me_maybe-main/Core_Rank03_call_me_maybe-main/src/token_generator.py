from typing import List, Set, Callable, Tuple, Final, FrozenSet
import numpy as np
import re
from dataclasses import dataclass
from llm_sdk import Small_LLM_Model
from .helper_functions import is_valid_num
from .parser import FnInfo


class ConstantParams:
    """Immutable configuration constants for constrained token generation.

    All bias values are additive adjustments applied to raw logits
    before argmax selection. Higher values increase the probability
    that the model picks tokens from the biased group.

    Terminators mark the end of a generated value:
    - STR_TERMINATOR: closing quote ends a string argument.
    - NUM_TERMINATOR: comma ends a numeric argument.

    Character sets define which vocabulary tokens receive extra bias
    for special argument types (regex patterns, symbol replacements).
    """

    # --- Terminators ---
    STR_TERMINATOR: Final[str] = '"'
    NUM_TERMINATOR: Final[str] = ','

    # --- Logit bias values ---
    REGEX_BIAS: Final[int] = 5
    SYMBOL_BIAS: Final[int] = 5
    PROMPT_BIAS: Final[int] = 5
    TERMINATOR_BIAS: Final[int] = 5
    NEGATIVE_SIGN_BIAS: Final[int] = 6

    # --- Generation limit ---
    TOKEN_LIMITS: Final[int] = 20

    # --- Character sets for special argument types ---
    REGEX_CHAR: Final[FrozenSet[str]] = frozenset({
        '\\', '|', '[', ']', '+', '*', '?', '^', '$',
        '(', ')', '.', ' ', '-', '\\b', '\\d', '\\s',
        'a-z', 'A-Z', '"',
    })
    SYMBOL_CHAR: Final[FrozenSet[str]] = frozenset({
        '*', '#', ' ', '_', '$', '@',
    })


@dataclass
class GenerationCost:
    """Tracks resource usage for a single generation pass.

    Attributes:
        token_used: Total number of LLM forward passes performed.
        time_taken_seconds: Wall-clock time for the full generation.

    Example:
        cost = GenerationCost(token_used=42, time_taken_seconds=3.5)
        print(cost.avg_time)  # 0.0833
    """

    token_used: int = 0
    time_taken_seconds: float = 0.0

    @property
    def avg_time(self) -> float:
        """Return average seconds per token, or 0.0 if no tokens used."""
        if self.token_used == 0:
            return 0.0
        return round(self.time_taken_seconds / self.token_used, 4)


class TokenGenerator:
    """Generates token sequences for function names and argument values
    using constrained decoding over a language model's logits.

    At each generation step, raw logits from the LLM are modified
    before token selection — either by hard-masking invalid tokens
    to -1e9 (guaranteeing they are never chosen) or by soft-biasing
    preferred tokens upward (nudging the model without forcing it).

    Two strategies are used depending on context:
    - Hard mask: used for function names — only exact valid tokens allowed.
    - Soft bias: used for string arguments — prompt tokens are preferred
      but the model retains some freedom.

    Attributes:
        prompt_tokens: The running token sequence fed to the LLM.
        tkn_limits: Maximum tokens to generate per argument or name.
        tokens_spend: Total LLM forward passes used since last reset.

    Example:
        generator = TokenGenerator(llm, vocab_tokens, encode, decode)
        generator.add_to_prompt(system_prompt_tokens)
        fn_tokens = generator.generate_function_name(functions)
        arg_tokens, _ = generator.generate_args_val(
            allowed_tokens, 'name', 'string', prompt, soft_bias=2)
    """

    def __init__(
            self,
            llm: Small_LLM_Model,
            token_set: List[str],
            encode: Callable[[str], List[int]],
            decode: Callable[[int], str],
            token_limit: int = ConstantParams.TOKEN_LIMITS
            ) -> None:
        """Initialise with model, vocabulary, and codec functions."""
        self.prompt_tokens: List[int] = []
        self.llm = llm
        self.encode = encode
        self.decode = decode
        self.tkn_limits = token_limit
        self.token_set = token_set
        self.tokens_spend = 0

        # Precomputed at init to avoid recomputing on every generation step
        self._regex = self.get_str_to_matching_tokens(
            ConstantParams.REGEX_CHAR)
        self._symbols = self.get_str_to_matching_tokens(
            ConstantParams.SYMBOL_CHAR)

    def get_prompt(self) -> List[int]:
        """Return the current prompt token sequence."""
        return self.prompt_tokens

    def set_token_limit(self, limit: int) -> None:
        """Override the maximum tokens to generate per value."""
        self.tkn_limits = limit

    def slice_prompt_tokens(self, start: int, end: int) -> None:
        """Truncate the prompt to the given slice range."""
        self.prompt_tokens = self.prompt_tokens[start: end]

    def add_to_prompt(self, tokens: List[int]) -> int:
        """Append token IDs to the running prompt and
        return new length."""
        self.prompt_tokens.extend(tokens)
        return len(self.prompt_tokens)

    def re_initialize_prompt_token(self) -> None:
        """Reset prompt tokens and spend counter for a
        new generation pass."""
        self.prompt_tokens = []
        self.tokens_spend = 0

    def get_total_token_spend(self) -> int:
        """Return total LLM forward passes since last reset."""
        return self.tokens_spend

    def get_str_to_matching_tokens(
            self, char_set: FrozenSet[str]) -> List[int]:
        """Return token IDs whose string representation starts with
        any member of char_set."""
        return [
            idx for idx, tok in enumerate(self.token_set)
            for pattern in char_set if tok.startswith(pattern)
        ]

    def generate_function_name(
            self, allowed_token: List[FnInfo]) -> List[int]:
        """Generate a function name token by token using position-aware
        trie-based constrained decoding."""
        complete_fn_tokens: List[int] = []
        token: int | float = float("-inf")
        terminating_token = self.encode(ConstantParams.STR_TERMINATOR)

        while (token != terminating_token[0] and
               len(complete_fn_tokens) < self.tkn_limits):
            logits = self.llm.get_logits_from_input_ids(self.prompt_tokens)
            next_allowed_tokens: Set[int] = set()
            complete_fn_len = len(complete_fn_tokens)

            for tkn in allowed_token:
                if len(tkn.fn_name_token) > complete_fn_len:
                    next_allowed_tokens.add(
                        tkn.fn_name_token[complete_fn_len])
                else:
                    next_allowed_tokens.add(terminating_token[0])

            token = self.get_next_fn_token(logits, next_allowed_tokens)
            complete_fn_tokens.append(token)
            self.prompt_tokens.append(token)
            self.tokens_spend += 1

        return complete_fn_tokens

    def get_matching_word(
            self,
            sub_str: str,
            prompt: str,
            arg_type: str
            ) -> str | None:
        """Return the unique word in prompt that starts with sub_str,
        or None if no unique match exists."""
        cleaned_token = sub_str.strip()
        if (len(cleaned_token) == 0 or
            cleaned_token == ConstantParams.STR_TERMINATOR or
                cleaned_token == ConstantParams.NUM_TERMINATOR):
            return None

        num_type = {"float", "int", "num", "number"}
        if arg_type in num_type:
            pattern = re.escape(sub_str) + "[0-9.]*"
        else:
            pattern = r"\b" + re.escape(sub_str) + r"\w*"

        match = re.findall(pattern, prompt)
        unique_matches: List[str] = list(set(match))

        if len(unique_matches) == 1:
            return unique_matches[0]

        return None

    def generate_args_val(
            self,
            allowed_tokens: List[int],
            arg_name: str,
            arg_type: str,
            prompt: str,
            soft_bias: int
            ) -> Tuple[List[int], List[int]]:
        """Generate the token sequence for a single argument value,
        dispatching between numeric and string strategies by arg_type."""
        complete_arg_tokens: List[int] = []
        token: int | float = float("-inf")

        if arg_type in {"float", "int", "number"}:
            terminating_token = self.encode(ConstantParams.NUM_TERMINATOR)
        else:
            terminating_token = self.encode(ConstantParams.STR_TERMINATOR)

        if arg_type == 'bool':
            allowed_tokens.extend(self.encode('True'))
            allowed_tokens.extend(self.encode('False'))

        allowed_tokens.extend(terminating_token)

        token_counter = 0
        sub_str = ""

        while (token != terminating_token[0] and
               len(complete_arg_tokens) < self.tkn_limits):

            logits = self.llm.get_logits_from_input_ids(self.prompt_tokens)
            self.tokens_spend += 1
            token_counter += 1

            if arg_type in {"float", "int", "number"}:
                token = self.get_next_numeric_token(
                    logits, set(allowed_tokens))
                str_val = self.decode(token)

                if " " in str_val:
                    str_val = str_val.strip()
                    token = self.encode(str_val)[0]

                sub_str += str_val
                matching_word = self.get_matching_word(
                    sub_str, prompt, arg_type)

                if matching_word is not None:
                    for _ in range(token_counter - 1):
                        self.prompt_tokens.pop()
                        complete_arg_tokens.pop()
                    if is_valid_num(matching_word):
                        self.prompt_tokens.extend(self.encode(matching_word))
                        complete_arg_tokens.extend(self.encode(matching_word))
                        sub_str = ""
                        token_counter = 0

                elif str_val.isdigit() or str_val in {"-", "."}:
                    complete_arg_tokens.append(token)
                    self.prompt_tokens.append(token)

            else:
                token = self.get_next_str_token(
                    logits, set(allowed_tokens), arg_name, soft_bias)
                str_val = self.decode(token)
                sub_str += str_val
                matching_word = self.get_matching_word(
                    sub_str, prompt, arg_type)

                if matching_word is not None:
                    for _ in range(token_counter - 1):
                        self.prompt_tokens.pop()
                        complete_arg_tokens.pop()
                    self.prompt_tokens.extend(self.encode(matching_word))
                    complete_arg_tokens.extend(self.encode(matching_word))
                    sub_str = ""
                    token_counter = 0
                else:
                    complete_arg_tokens.append(token)
                    self.prompt_tokens.append(token)

                if '"' in str_val:
                    idx = str_val.index('"')
                    self.prompt_tokens.pop()
                    self.prompt_tokens.extend(
                        self.encode(str_val[:idx + 1]))
                    break

        return complete_arg_tokens, allowed_tokens

    def get_next_fn_token(
            self,
            logits: List[float],
            allowed_idx: Set[int]
            ) -> int:
        """Select the highest-logit token after hard-masking all tokens
        outside allowed_idx to -1e9."""
        logits_np = np.array(logits)
        mask = np.full_like(logits_np, -1e9)
        mask[list(allowed_idx)] = 0
        return int(np.argmax(logits_np + mask))

    def get_next_str_token(
            self,
            logits: List[float],
            allowed_idx: Set[int],
            arg_name: str,
            soft_bias: int = 5
            ) -> int:
        """Select the next string token after soft-biasing allowed tokens
        and applying special biases for regex and replacement arguments."""
        logits_np = np.array(logits)
        logits_np[list(allowed_idx)] += soft_bias

        if arg_name == 'regex':
            logits_np[self._regex] += ConstantParams.REGEX_BIAS

        if arg_name == 'replacement':
            logits_np[self._symbols] += ConstantParams.SYMBOL_BIAS

        return int(np.argmax(logits_np))

    def get_next_numeric_token(
            self,
            logits: List[float],
            allowed_idx: Set[int]
            ) -> int:
        """Select the next numeric token using hard masking with extra
        bias toward the negative sign character."""
        logits_np = np.array(logits)
        mask = np.full_like(logits_np, -1e9)
        mask_idx = list(allowed_idx)
        mask[mask_idx] = 0
        self.create_token_biasing(
            mask, mask_idx, "-", ConstantParams.NEGATIVE_SIGN_BIAS)
        return int(np.argmax(logits_np + mask))

    def create_token_biasing(
            self,
            mask: np.ndarray,
            allowed_idx: List[int],
            bias_string: str,
            soft_bias: int = 5
            ) -> None:
        """Apply an extra logit bonus in-place to allowed tokens whose
        decoded string appears in bias_string."""
        boosted: List[int] = []
        for token_idx in allowed_idx:
            token_str = self.decode(token_idx).strip()
            if token_str and token_str in bias_string:
                boosted.append(token_idx)
        mask[boosted] += soft_bias
