import sys
from .arg_validator import ArgValidator
from pydantic import ValidationError

config_files = {
    "--functions_definition": "data/input/functions_definition.json",
    "--input": "data/input/function_calling_tests.json",
    "--output": "data/output/function_calls.json"
    }
print(f"CONFIG FILES BEFORE INPUT {config_files}")
try:
    arg_validator = ArgValidator(args=sys.argv, config_files=config_files)
    config_files = arg_validator.config_files
except ValidationError as e:
    print(e.errors()[0]["msg"].removeprefix("Value error, "))
