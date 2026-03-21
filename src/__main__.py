from .arg_validator import ArgValidator
from .parser import ConfigParser
from .json_builder import create_output
from .output_writer import write_output
from pydantic import ValidationError
from llm_sdk import Small_LLM_Model
import sys


def main() -> None:
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
