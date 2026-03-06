import sys
from .arg_validator import ArgValidator
from .config_parser import ConfigParser, ParserError, Function
from pydantic import ValidationError


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
    except ParserError as e:
        print(e)


if __name__ == "__main__":
    main()
