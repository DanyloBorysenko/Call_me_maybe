from pydantic import BaseModel, model_validator, field_validator, Field
from typing import Dict, List
import json
from pydantic import ValidationError


class ParserError(Exception):
    pass


class Parameter(BaseModel):
    type: str = Field(min_length=1)


class Return(BaseModel):
    type: str = Field(min_length=1)


class Function(BaseModel):
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
    prompt: str = Field(min_length=1)


class ConfigParser(BaseModel):
    functions_definition: str = Field(min_length=6)
    input: str = Field(min_length=6)
    output: str = Field(min_length=6)

    @model_validator(mode="before")
    def preprocess_args(cls, row_data: Dict[str, str]) -> Dict[str, str]:
        new_data = {}
        for key in row_data.keys():
            new_key = key.removeprefix("--")
            new_data[new_key] = row_data[key]
        return new_data

    def load_functions(self) -> List[Function]:
        try:
            with open(self.functions_definition, "r") as f:
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
                                   "duplacate name")
            return functions
        except ValidationError as e:
            raise_parser_error(e)
        except Exception as e:
            raise RuntimeError(f"{e}")
        return []

    def load_prompts(self) -> List[Prompt]:
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
    err = error.errors()[0]
    field = ".".join(map(str, err["loc"]))
    msg = err["msg"].removeprefix("Value error, ")
    raise RuntimeError(f"ParserError: Invalid definition: '{field}' "
                       f"→ {msg}")
