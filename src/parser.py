"""Configuration parsing and validation module.

This module defines data models and utilities for:
- Validating function definitions and prompts using Pydantic.
- Loading and parsing JSON configuration files.
- Normalizing CLI-style arguments into internal configuration.

It ensures strict schema validation and provides consistent error handling
for invalid input data.
"""

from pydantic import BaseModel, model_validator, field_validator, Field
from typing import Dict, List
import json
from pydantic import ValidationError


class Parameter(BaseModel):
    """Represents a function parameter definition.

    Attributes:
        type: The data type of the parameter (e.g., "string", "number").
    """
    type: str = Field(min_length=1)


class Return(BaseModel):
    """Represents a function return type.

    Attributes:
        type: The data type of the return value.
    """
    type: str = Field(min_length=1)


class Function(BaseModel):
    """Represents a callable function definition.

    This model describes a function that can be selected and invoked
    by the system, including its parameters and return type.

    Attributes:
        name: Unique function name.
        description: Human-readable description of the function.
        parameters: Mapping of parameter names to their definitions.
        returns: Return type definition.

    Raises:
        ValueError: If any parameter name is empty.
    """
    model_config = {"frozen": True}  # This makes the class hashable
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: Dict[str, Parameter]
    returns: Return

    @field_validator("parameters")
    def check_params(cls,
                     params: Dict[str, Parameter]) -> Dict[str, Parameter]:
        for param_name in params:
            if param_name.strip() == "":
                raise ValueError("parameter's name can not be empty")
        return params


class Prompt(BaseModel):
    """Represents a user input prompt.

    Attributes:
        prompt: The input string provided by the user.
    """
    prompt: str = Field(min_length=1)


class ConfigParser(BaseModel):
    """Parses and validates configuration files for the pipeline.

    This class:
    - Normalizes CLI-style arguments (e.g., "--input" → "input").
    - Loads function definitions and prompts from JSON files.
    - Validates structure and schema using Pydantic models.

    Attributes:
        functions_definition: Path to JSON file with function schemas.
        input: Path to JSON file with prompts.
        output: Path to output file.

    Raises:
        RuntimeError: If files are missing, malformed, or invalid.
    """
    functions_definition: str = Field(min_length=6)
    input: str = Field(min_length=6)
    output: str = Field(min_length=6)

    @model_validator(mode="before")
    def preprocess_args(cls, raw_data: Dict[str, str]) -> Dict[str, str]:
        """Normalizes CLI-style keys by removing leading '--'.

        Args:
            raw_data: Raw input dictionary with CLI-style keys.

        Returns:
            A dictionary with normalized keys.
        """
        new_data = {}
        for key in raw_data.keys():
            new_key = key.removeprefix("--")
            new_data[new_key] = raw_data[key]
        return new_data

    def load_functions(self) -> List[Function]:
        """Loads and validates function definitions from a JSON file.

        Returns:
            A list of validated Function objects.

        Raises:
            RuntimeError: If the file is missing, malformed, or invalid.
        """
        try:
            with open(self.functions_definition, "r", encoding="utf-8") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"ParserError: File {self.functions_definition}"
                               " not found")
        except json.JSONDecodeError:
            raise RuntimeError("ParserError: Invalid JSON format "
                               "in functions definition")
        if not isinstance(data, list):
            raise RuntimeError("ParserError: Functions definition "
                               "must be a list")
        for item in data:
            if not isinstance(item, dict):
                raise RuntimeError("Each function definition "
                                   "must be an object")
        try:
            functions = [Function(**item) for item in data]
            if len(functions) != len(set(func.name for func in functions)):
                raise RuntimeError("ParserError: Functions can not have "
                                   "duplicate name")
            return functions
        except ValidationError as e:
            raise_parser_error(e)
        except Exception as e:
            raise RuntimeError(f"{e}")

    def load_prompts(self) -> List[Prompt]:
        """Loads and validates prompts from a JSON file.

        Returns:
            A list of validated Prompt objects.

        Raises:
            RuntimeError: If the file is missing, malformed, or invalid.
        """
        try:
            with open(self.input, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise RuntimeError(f"ParserError: File {self.input} not found")
        except json.JSONDecodeError:
            raise RuntimeError("ParserError: Invalid JSON format in prompts "
                               "definition")
        if not isinstance(data, list):
            raise RuntimeError("ParserError: Prompt definition must be a list")
        for item in data:
            if not isinstance(item, dict):
                raise RuntimeError("ParserError: Each prompt definition "
                                   "must be an object")
        try:
            return [Prompt(**item) for item in data]
        except ValidationError as e:
            raise_parser_error(e)
        except Exception as e:
            raise RuntimeError(f"ParserError: {e.__class__.__name__} - {e}")
        return []


def raise_parser_error(error: ValidationError) -> None:
    """Converts Pydantic validation errors into RuntimeError.

    Args:
        error: The ValidationError raised by Pydantic.

    Raises:
        RuntimeError: With a simplified and user-friendly message.
    """
    err = error.errors()[0]
    field = ".".join(map(str, err["loc"]))
    msg = err["msg"].removeprefix("Value error, ")
    raise RuntimeError(f"ParserError: Invalid definition: '{field}' "
                       f"→ {msg}")
