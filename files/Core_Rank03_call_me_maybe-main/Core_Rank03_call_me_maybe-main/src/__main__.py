import time
import sys
# from src.helper_functions import load_json
# from src.ConstrainDecoder import ConstrainDecoder
from typing import List, Tuple
from pydantic import TypeAdapter
from llm_sdk import Small_LLM_Model
from .parser import (
    CLI_Parser, FnInfo,
    FunctionLoader, PromptLoader)
from .constrain_decoder import (
    ConstrainDecoder, Output,
    build_initial_prompt)
from .token_generator import GenerationCost
from .tokenizer import Tokenizer
from .custom_errors import SourceError


# def custom_prompts() -> List[str]:
#     prompts = [
#             # "Replace all alphabets in 'hi1245 assuu152' with digit 0",
#             "Replace all the alphabets in 'hi1245 assuu152' with digit 0",
#             "Replace all numbers in \"Hello 34 I'm "
#             "233 years old\" with NUMBERS",
#             "Substitute the digits in the string 'Hello"
#             " 34 I'm 233 years old' with 'NUMBERS'",
#             "Substitute all alphabet between a-z in the "
#             "string 'Hello 34 I'm 233 years old' with digits",
#             # "Replace all '[aeiou]' in 'Programming is fun' with '*'",
#             # "Replace all vowels in 'Programming is fun' with asterisks",
#             "Replace consonants in 'Programming is fun' with hash",
#             "Substitute the word 'cat' with 'dog' in "
#             "'The cat sat on the mat with another cat'",
#             "Replace the 'space' in 'Programming is fun' with 'underscore'",
#             "Replace the spaces in 'Programming is fun' with underscore",
#             "Replace all spaces in 'Programming   is fun' with underscore",
#             # "replace the user in email id user@email.com with user123",
#             # "replace the 'user' in email id 'user@email.com'"
#               " with 'user123'",
#             # "replace all a in the word hella warld with o",
#             # "replace all 'a' in the word 'hella warld' with 'o'",

#             # "what is the sum of -21567850.25 and 30.5545665427?",
#             # "what is the sum of -520, -2.5 and -30?",

#             # "reverse sukanta das",
#             # "reverse 'sukanta das'",
#             # "'reverse' 'sukanta'",
#             # "reverse sukanta",
#             # "write sukanta from end to start",
#             # "write 'sukanta' from end to start",

#             # "Greet sukanta das",
#             # "Greet 'sukanta das'",
#             # "Greet mr. das",
#             # "hi mr. unnamed",
#             # "'Greet mr. unnamed",
#             # "Greet ram",
#             # "Greet shrek",
#             # "Greet Shrek",
#             # "Greet 'Shrek 1234'",
#             # "'hello' 'Shrek'",
#             # "'Greet John'",
#             # "'Welcome John'",
#             # "'Hello' 'John'",
#             # "let us all welcome shreak to the party",
#             # "what is your name? ",

#             # "'Mango is a fruit' is a true statement",
#             # "'Mango is a fruit' is a correct statement",
#             # "'Mango is a fruit' is a right answer",
#             # "'Mango is a vegetable' is a wrong answer",
#             # "Mango is not a vegetable",
#             # "Mango is a vegetable. Wrong",
#         ]
#     return prompts


def initialize_pipeline(
        device: str = 'cpu'
        ) -> Tuple[ConstrainDecoder, List[FnInfo]]:
    """Load model, tokenizer, and function registry.

    Args:
        paths: Validated resource file paths
        device: Compute device ('cpu', 'cuda', 'mps')

    Returns:
        Tuple of (decoder, functions) ready for generation
    """
    llm = Small_LLM_Model(device=device)
    tokenizer = Tokenizer(path=llm.get_path_to_vocab_file())
    paths = CLI_Parser.parse_cli_arguments(sys.argv)
    functions = FunctionLoader.load_json(
        paths.function_def, tokenizer.encode)
    token_set = tokenizer.get_all_tokes()

    decoder = ConstrainDecoder(
        llm, functions, token_set, tokenizer.encode, tokenizer.decode)

    return decoder, functions


def function_generator(
        prompt: str, decoder: ConstrainDecoder
        ) -> Tuple[Output, GenerationCost]:
    """Generate a single function call from a prompt."""
    start = time.time()
    out = Output()
    cost = GenerationCost()
    out.prompt = prompt
    decoder.tkn_generator.re_initialize_prompt_token()

    initial_token = build_initial_prompt(
        prompt, decoder.functions, decoder.encode)
    decoder.tkn_generator.add_to_prompt(initial_token)
    decoder.generate(prompt, out)

    end = time.time()
    cost.time_taken_seconds = round((end - start), 3)
    cost.token_used = decoder.tkn_generator.get_total_token_spend()
    return out, cost


def main() -> None:
    try:
        start = time.time()
        llm = Small_LLM_Model(device='cpu')
        token_path = llm.get_path_to_vocab_file()
        tokenizer = Tokenizer(path=token_path)
        encode = tokenizer.encode
        decode = tokenizer.decode
        token_set = tokenizer.get_all_tokes()
        # encode = llm._encode
        # decode = llm._decode
        mid = time.time()
        print(f"LLM loaded in {(mid - start):.3f}s")

        paths = CLI_Parser.parse_cli_arguments(sys.argv)

        functions = FunctionLoader.load_json(
            paths.function_def, encode)

        prompts = PromptLoader.load_json(paths.inputs)

        # prompts = custom_prompts()
        # prompts = parser.load_prompts(prompts)

        prompt_generator = ConstrainDecoder(
            llm, functions, token_set,
            encode, decode)

        outputs = prompt_generator.generate_for_all_prompts(prompts)
        end = time.time()
        print(f"function generation time {(end - mid):.3f}s")

        # Saving the outputs
        adapter = TypeAdapter(List[Output])
        json_data = adapter.dump_json(outputs, indent=4).decode('utf-8')
        with open(paths.outputs, 'w') as fl:
            fl.write(json_data)

    except SourceError as e:
        print("Configuration error: ", e)
    except KeyboardInterrupt:
        print("Interrupted by user")
    except Exception as e:
        print(f"Unexpected error: {e}")


if __name__ == "__main__":
    main()
