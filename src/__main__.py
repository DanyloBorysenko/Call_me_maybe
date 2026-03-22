from .arg_validator import ArgValidator
from .parser import ConfigParser
from .json_builder import create_output
from .output_writer import write_output
from pydantic import ValidationError
from llm_sdk import Small_LLM_Model
import sys


def main() -> None:
    """Runs the function-calling pipeline using an LLM model.

    This function orchestrates the full workflow:
    - Validates arguments and resolves configuration file paths.
    - Parses function definitions and input prompts.
    - Initializes the language model.
    - Generates structured function call outputs.
    - Writes results to the specified output file.

    The configuration includes:
        --functions_definition: Path to JSON file with function schemas.
        --input: Path to JSON file with user prompts.
        --output: Path where generated results will be saved.

    Raises:
        SystemExit: If argument validation or parsing fails.

    Side Effects:
        - Prints validation or runtime errors to stdout.
        - Writes output JSON to disk.
    """

    config_files = {
                "--functions_definition": "data/input/functions_definition."
                "json",
                "--input": "data/input/function_calling_tests.json",
                "--output": "data/output/function_calling_results.json"
    }
    try:
        arg_validator = ArgValidator(args=sys.argv, config_files=config_files)
        config_files = arg_validator.config_files
        parser = ConfigParser(**config_files)
    except ValidationError as e:
        print(e.errors()[0]["msg"].removeprefix("Value error, "))
        exit()
    try:
        functions = parser.load_functions()
        prompts = parser.load_prompts()
        model = Small_LLM_Model()
        output = create_output(functions, prompts, model)
        write_output(output, config_files["--output"])
    except RuntimeError as e:
        print(f"{e}")
        exit()


if __name__ == "__main__":
    main()
    # import re
    # prompt = "Hello 34 I'm 233 years old"
    # prompt2 = "ProgrAmming is fun"
    # regex = "(\\d+)"
    # regex2 = "([aeiouAEIOU])"
    # match_result = re.findall(regex, prompt)
    # if match_result:
    #     print(f"Result is - {match_result}")
