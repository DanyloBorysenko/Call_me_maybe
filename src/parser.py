from pydantic import BaseModel, model_validator, field_validator, Field
from typing import Dict, List, Any
import json
from pydantic import ValidationError


class ParserError(Exception):
    pass


class Parameter(BaseModel):
    type: str = Field(min_length=1)


class Return(BaseModel):
    type: str = Field(min_length=1)


class Function(BaseModel):
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
        for arg in row_data.keys():
            new_arg = arg.removeprefix("--")
            new_data[new_arg] = row_data[arg]
        return new_data

    def load_functions(self) -> List[Function]:
        try:
            with open(self.functions_definition, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise ParserError(f"File {self.functions_definition} not found")
        except json.JSONDecodeError:
            raise ParserError("Invalid JSON format in functions definition")
        if not isinstance(data, list):
            raise ParserError("Functions definition must be a list")
        for item in data:
            if not isinstance(item, dict):
                raise ParserError("Each function definition must be an object")
        try:
            functions = [Function(**item) for item in data]
            if len(functions) != len(set(func.name for func in functions)):
                raise ParserError("Functions can not have duplacate name")
            return functions
        except ValidationError as e:
            raise_parser_error(e)
        except Exception as e:
            raise ParserError(f"{e}")
        return []

    def load_prompts(self) -> List[Prompt]:
        try:
            with open(self.input, "r") as f:
                data = json.load(f)
        except FileNotFoundError:
            raise ParserError(f"File {self.input} not found")
        except json.JSONDecodeError:
            raise ParserError("Invalid JSON format in prompts definition")
        if not isinstance(data, list):
            raise ParserError("Prompt definition must be a list")
        for item in data:
            if not isinstance(item, dict):
                raise ParserError("Each prompt definition must be an object")
        try:
            return [Prompt(**item) for item in data]
        except ValidationError as e:
            raise_parser_error(e)
        except Exception as e:
            raise ParserError(f"{e}")
        return []


def raise_parser_error(error: ValidationError) -> None:
    err = error.errors()[0]
    field = ".".join(map(str, err["loc"]))
    msg = err["msg"].removeprefix("Value error, ")
    raise ParserError(f"Invalid function definition: '{field}' "
                      f"→ {msg}")
