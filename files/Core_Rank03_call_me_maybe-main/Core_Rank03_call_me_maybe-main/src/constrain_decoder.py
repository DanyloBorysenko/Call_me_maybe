import time
from typing import List, Callable, Any
import re
from llm_sdk import Small_LLM_Model
from .token_generator import TokenGenerator
from .parser import FnInfo, Prompts, Output


def build_initial_prompt(prompt: str, functions: List[FnInfo],
                         encode: Callable) -> List[int]:
    """Build and encode the system prompt for function selection.

    Constructs a prompt that describes available functions to the
    LLM, giving it the context needed to select the right function
    and extract the right arguments.

    Args:
        prompt: The user's natural language prompt
        functions: Available functions with descriptions
        encode: Tokenizer encode function

    Returns:
        Token IDs for the complete system prompt
    """
    pre_prompt = ""
    pre_prompt += "Available functions with description about the function:\n"
    for fn in functions:
        pre_prompt += f"{fn.fn_name}: {fn.description}\n"
        # pre_prompt += f"{fn.fn_name}("
        # arg_with_types = ""
        # for a, t in fn.args_types.items():
        #     arg_with_types += f"{a}: {t}, "
        # pre_prompt += f"{arg_with_types[:-2]})"
        # pre_prompt += f" -> {fn.return_type}\n\n"

    pre_prompt += (
        "Example: 'Question': 'Greet Sukanta' -> "
        "'fn_name': 'fn_greet', 'args': {'name': 'Sukanta'}\n"
        )

    tokens: List[int] = encode(pre_prompt)
    return tokens


class ConstrainDecoder:
    """Translates natural language prompts into structured function calls.

    Uses constrained decoding to guarantee:
    - 100% valid JSON output
    - Function name always matches a known function
    - Argument types always match the function schema

    Attributes:
        functions: Available functions the model can call

    Example:
        decoder = ConstrainedDecoder(llm, functions,
        token_set, encode, decode)
        result = decoder.process(prompt)
        print(result.name, result.parameters)
    """
    def __init__(self, llm: Small_LLM_Model,
                 functions: List[FnInfo],
                 token_set: List[str],
                 encode: Callable, decode: Callable) -> None:
        """Initialise the decoder with model and function registry.

        Args:
            llm: The language model to use for generation
            functions: Available functions with schema info
            token_set: Full vocabulary token list for pattern matching
            encode: Tokenizer encode function
            decode: Tokenizer decode function
        """
        self.llm = llm
        self.functions = functions
        self.tkn_generator = TokenGenerator(llm, token_set, encode, decode)
        self.encode = encode
        self.decode = decode

    def add_str_to_prompt(self, string: str) -> int:
        token_patch = self.encode(string)
        return self.tkn_generator.add_to_prompt(token_patch)

    def generate(self, prompt: str, out: Output) -> None:
        """Process a single natural language prompt.

        Args:
            prompt: Natural language description of function to call

        Returns:
            FunctionCall with name and typed parameters
        """
        starting_prompt = f'Question: "{prompt}",\n'
        self.add_str_to_prompt(starting_prompt)

        self.add_str_to_prompt('{"name": "')
        # print(f"prompt: {self.tokenizer.decode(
        #     self.constrain_decoder.prompt_tokens)}")
        final_fn = self.tkn_generator.\
            generate_function_name(self.functions)

        final_fn_name = self.decode(final_fn[: -1])
        # self.tkn_generator.slice_prompt_tokens(0, before_fn_pos)
        for fn in self.functions:
            if final_fn_name == fn.fn_name:
                out.name = final_fn_name
                self.handle_arguments(prompt, fn, out)
        self.add_str_to_prompt("},\n},")

    def get_all_allowed_token(self, prompt: str) -> List[int]:
        """Encode the prompt also split the prompt into quoted
        and individual words and tokenize them

        Args:
            prompt used

        Returns:
            List of token ides
        """
        tokens = self.encode(prompt)
        words = re.findall(r"'[^']+'|\"([^\"]+)\"|\b\w+\b", prompt)
        # print(words)
        for word in words:
            tokens.extend(self.encode(word))
        unique_tokens = list(set(tokens))
        # print([self.decode(tkn) for tkn in unique_tokens])
        return unique_tokens

    def handle_arguments(self, prompt: str,
                         fn: FnInfo, out: Output) -> None:
        """Generate all arguments for the selected function.

        Args:
            prompt: Original prompt for token extraction
            fn: Selected function with schema info
            result: FunctionCall to populate with arguments
        """
        original_prompt = prompt

        prompt_tokens = self.get_all_allowed_token(prompt)
        initial_arg_token = ',\n"parameters": {'
        self.add_str_to_prompt(initial_arg_token)
        total_args = len(fn.args_names)

        for i, arg in enumerate(fn.args_names):
            arg_type = fn.args_types[arg]
            if (arg_type == "float" or arg_type == "int" or
               arg_type == "number"):
                self.add_str_to_prompt(f'"{arg}": ')
            else:
                self.add_str_to_prompt(f'"{arg}": "')
            arg_val_token, prompt_tokens = self.tkn_generator.\
                generate_args_val(
                    prompt_tokens, arg, arg_type, original_prompt, 2)
            arg_val_str = self.decode(arg_val_token)
            self._store_arguments(arg, arg_val_str, arg_type, out)

            if i < total_args - 1:
                self.add_str_to_prompt(", ")

    def _store_arguments(self, key: str, val: Any,
                         arg_type: Any, out: Output) -> None:
        """Parse raw generated string into the correct Python type.

        Args:
            key: Parameter name for error messages
            val: Raw string from token generation
            arg_type: Expected type from function schema
            out: Output instance for storing the arguments

        Returns:
            Typed value matching the schema"""
        if arg_type == 'number':
            try:
                if "." in val:
                    out.parameters[key] = float(val)
                else:
                    out.parameters[key] = int(val)
            except ValueError:
                out.parameters[key] = ""
                print(f"{key} has not numeric value: {val}")
        elif arg_type == 'bool':
            if 'right' in val.lower() or 'true' in val.lower():
                out.parameters[key] = True
            elif 'wrong' in val.lower() or 'false' in val.lower():
                out.parameters[key] = False
        else:
            str_in_args = re.findall("[^\",}]", val)
            clean_val = "".join(str_in_args)
            if len(clean_val) > 0 and len(clean_val.strip()) == 0:
                clean_val = " "
            else:
                clean_val = clean_val.strip()
            # if key == "replacement":
            #     symbol_prompt = (
            #         "\nWhat is the symbolic representation of"
            #         f" the string '{clean_val}'?\nAnswer: \"")
            #     self.add_str_to_prompt(symbol_prompt)
            #     # print(regex_prompt)
            #     self.tkn_generator.generate_args_val(
            #         [], key, val, symbol_prompt, 0)
            out.parameters[key] = clean_val

    def generate_for_all_prompts(self, prompts: List[Prompts]) -> List[Output]:
        """Process a list of prompts sequentially.
        Args:
            prompts: List of Prompt objects to process

        Returns:
            List of FunctionCall results, one per prompt
        """
        outputs: List[Output] = []
        for prompt in prompts:
            start = time.time()

            out = Output()
            out.prompt = prompt.prompt
            self.tkn_generator.re_initialize_prompt_token()
            initial_token = build_initial_prompt(
                prompt.prompt, self.functions, self.encode)
            # print(f"prompt: {self.tokenizer.decode(initial_token)}")

            self.tkn_generator.add_to_prompt(initial_token)
            self.generate(prompt.prompt, out)
            final_token = self.tkn_generator.get_prompt()
            print(self.decode(final_token[len(initial_token):]))

            outputs.append(out)

            end = time.time()
            time_taken = round((end - start), 3)
            tokens_spend = self.tkn_generator.get_total_token_spend()
            avg_cost = round(time_taken / tokens_spend, 3)
            print(f"\033[92mTokens Used: {tokens_spend},"
                  f" Time: {time_taken}s, Cost/Token: {avg_cost}s\033[0m")
        return outputs
