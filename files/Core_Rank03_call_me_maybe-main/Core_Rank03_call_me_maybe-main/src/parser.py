import json
from pathlib import Path
import argparse
from dataclasses import dataclass
from enum import StrEnum
from typing import Dict, List, Callable, Any, Set, ClassVar
from pydantic import (
    BaseModel, Field, model_validator, Json, field_validator)
from src.custom_errors import SourceError


class Flags(StrEnum):
    """Flags to indicate different type of file paths"""
    INPUT = "--input"
    FUNCTION = "--functions_definition"
    OUTPUT = "--output"


@dataclass(frozen=True)
class ResourcePaths:
    """Immutable file paths for the pipeline.

    Attributes:
        function_def: Path to function definitions JSON
        inputs: Path to input prompts JSON
        outputs: Path to write results JSON
    """
    function_def: Path
    inputs: Path
    outputs: Path

    # Defaults live here as class constants
    DEFAULT_FUNCTIONS: ClassVar[str] = \
        "data/input/functions_definition.json"
    DEFAULT_INPUTS: ClassVar[str] = \
        "data/input/function_calling_tests.json"
    DEFAULT_OUTPUTS: ClassVar[str] = \
        "data/output/function_calling_results.json"

    # test paths
    # function_def: str = "data/input/test_functions.json"
    # inputs: str = "data/input/test_prompt.json"
    # outputs: str = "data/output/test_results.json"


class CLI_Parser:
    """Class to handle commandline arguments and verify the the
    provided paths"""
    @staticmethod
    def parse_cli_arguments(argv: List[str]) -> ResourcePaths:
        """Parse command line arguments and return validated resource paths.

        Args:
            argv: Command line arguments, typically sys.argv

        Returns:
            ResourcePaths with all paths validated

        Raises:
            SystemExit: If arguments are invalid (argparse handles this)
            SourceError: If paths don't exist or have wrong format
        """
        parser = argparse.ArgumentParser()

        parser.add_argument(
            Flags.FUNCTION,
            type=str,
            default=ResourcePaths.DEFAULT_FUNCTIONS,
            help="Path to function definitions JSON"
        )
        parser.add_argument(
            Flags.INPUT,
            type=str,
            default=ResourcePaths.DEFAULT_INPUTS,
            help="Path to input prompts JSON"
        )
        parser.add_argument(
            Flags.OUTPUT,
            type=str,
            default=ResourcePaths.DEFAULT_OUTPUTS,
            help="Path to output results JSON"
        )

        args = parser.parse_args(argv[1:])

        paths = ResourcePaths(
            function_def=Path(args.functions_definition),
            inputs=Path(args.input),
            outputs=Path(args.output)
        )
        return CLI_Parser._validate_paths(paths)

    @staticmethod
    def _validate_paths(paths: ResourcePaths) -> ResourcePaths:
        """Validate all resource paths exist and have correct formats.
        Args:
            paths: ResourcePaths to validate
        Returns:
            The same paths if all valid
        Raises:
            SourceError: With specific message about which path failed
        """
        CLI_Parser._validate_input_path(
            paths.function_def, "Function definition")
        CLI_Parser._validate_input_path(
            paths.inputs, "Input prompts")
        CLI_Parser._ensure_output_path(
            paths.outputs)
        return paths

    @staticmethod
    def _validate_input_path(path: Path, name: str) -> None:
        """Validate a single input path exists and is JSON."""
        if not path.exists():
            raise SourceError(
                f"{name} file not found: '{path}'\n"
                f"Current working directory: {Path.cwd()}"
            )
        if path.suffix != '.json':
            raise SourceError(
                f"{name} must be a JSON file, got: '{path.suffix}'"
            )

    @staticmethod
    def _ensure_output_path(path: Path) -> None:
        """Create output file and parent directories if they don't exist."""
        try:
            path.parent.mkdir(parents=True, exist_ok=True)
            if not path.exists():
                path.touch()
        except PermissionError:
            raise SourceError(
                f"Cannot create output file at '{path}': permission denied"
            )
        except OSError as e:
            raise SourceError(f"Cannot create output path: {e}")


class FnInfo(BaseModel):
    """Metadata for a callable function including name, args, and types.

    Attributes:
        name: Function identifier, must start with 'fn_'
        description: Human readable description of what the function does
        parameters: Raw parameter definitions from JSON
        returns: Raw return type definition from JSON
        name_tokens: Tokenized function name for constrained decoding
        arg_names: Ordered list of argument names
        arg_types: Mapping of argument name to its ArgType
        return_type: The return type of the function
    """

    VALID_TYPES: ClassVar[Set[str]] = {"number", "string", "bool"}

    fn_name: str = Field(min_length=2, alias="name")
    description: str = Field(default="", alias="description")
    parameters: Dict = Field(alias="parameters")
    returns: Dict = Field(alias="returns")

    # Field populated by validator
    fn_name_token: List[int] = Field(default_factory=list)
    args_names: List[str] = Field(default_factory=list)
    args_types: Dict[str, str] = Field(default_factory=dict)
    return_type: List[str] = Field(default_factory=list)

    @field_validator('fn_name')
    @classmethod
    def validate_name_format(cls, v: str) -> str:
        """Ensure function name follows fn_ convention."""
        if not v.startswith('fn_'):
            raise ValueError(
                f"Function name must start with 'fn_', got: '{v}'. "
                f"All functions must follow the fn_<action> naming convention."
            )
        return v

    @model_validator(mode='after')
    def extract_parameters(self) -> 'FnInfo':
        """Extract and validate parameter names and types from raw dict."""
        for arg, val in self.parameters.items():
            if 'type' not in val:
                raise ValueError(
                    f"Parameter '{arg}' is missing 'type' field. "
                    f"Got: {val}"
                )
            arg_type = val['type']
            if arg_type not in self.VALID_TYPES:
                raise ValueError(
                    f"parameter: {arg} has type unknown "
                    f"'{arg_type}' types."
                )
            self.args_names.append(arg)
            self.args_types[arg] = arg_type

        return self

    @model_validator(mode='after')
    def extract_return_type(self) -> 'FnInfo':
        """Extract and validate return type from raw dict."""
        if 'type' not in self.returns:
            raise ValueError(
                f"'returns' field is missing 'type'. Got: {self.returns}"
            )

        return_type = self.returns['type']
        if return_type not in self.VALID_TYPES:
            raise ValueError(
                f"Unknown return type '{return_type}'. "
                f"Valid types are: {self.VALID_TYPES}"
            )

        self.return_type = return_type
        return self


class Prompts(BaseModel):
    """A single natural language prompt to be processed.

    Attributes:
        prompt: The raw natural language input text
    """
    prompt: str = Field(min_length=1, alias="prompt")

    @field_validator('prompt')
    @classmethod
    def clean_prompt(cls, value: str) -> str:
        """Strip whitespace and validate prompt is not just whitespace."""
        cleaned = value.strip()
        if not cleaned:
            raise ValueError(
                "Prompt can not be empty"
            )
        return cleaned


class Output(BaseModel):
    prompt: str = Field(default="", description="Original Prompt")
    name: str = Field(default="", description="Function name")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Function arguments"
    )

    def get_json_str(self) -> Json:
        return self.model_dump_json(indent=2)


class FunctionLoader():
    """Loads and tokenizes function definitions from JSON."""
    @staticmethod
    def load_json(path: Path, encode: Callable) -> List[FnInfo]:
        """Load function definitions from a JSON file.

        Args:
            path: Path to the function definitions JSON
            encode: Tokenizer encode function for pre-tokenizing names

        Returns:
            List of validated FnInfo objects

        Raises:
            SourceError: If file is missing, malformed, or has invalid schema
        """
        try:
            with open(path, "r") as fl:
                data = json.load(fl)
        except json.JSONDecodeError as e:
            raise SourceError(
                f"Function definition file is not valid JSON: {e}"
            )
        except FileNotFoundError:
            raise SourceError(
                f"Function definition file not found: '{path}'"
            )

        if not isinstance(data, list):
            raise SourceError(
                "Function definition file must contain a JSON array"
            )

        functions = []
        for i, info in enumerate(data):
            try:
                info['fn_name_token'] = encode(info.get('name', ''))
                functions.append(FnInfo.model_validate(info))
            except Exception as e:
                raise SourceError(
                    f"Invalid function definition at index {i}: {e}"
                )
        return functions


class PromptLoader:
    """Loads prompts from JSON files or raw string lists."""

    @staticmethod
    def load_json(path: Path) -> List[Prompts]:
        """Load prompts from a JSON file."""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
        except json.JSONDecodeError as e:
            raise SourceError(
                f"Prompt file is not valid JSON: {e}"
            )

        if not isinstance(data, list):
            raise SourceError("Prompt file must contain a JSON array")

        prompts = []
        for i, item in enumerate(data):
            try:
                prompts.append(Prompts.model_validate(item))
            except Exception as e:
                raise SourceError(
                    f"Invalid prompt at index {i}: {e}"
                )
        return prompts

    @staticmethod
    def from_strings(raw: List[str]) -> List[Prompts]:
        """Create prompts from a list of raw strings."""
        return [Prompts(prompt=s) for s in raw if s.strip()]


if __name__ == "__main__":
    pass
