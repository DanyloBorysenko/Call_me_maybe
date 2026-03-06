from pydantic import BaseModel, model_validator, Field
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
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: Dict[str, Parameter]
    returns: Return

    # @model_validator(mode="after")
    # def check_params(self) -> Self:
    #     if len(self.parameters.keys()) == 0:
    #         raise ValueError("At least one parameter is required")
    #     for param_name in self.parameters.keys():
    #         if param_name == "":
    #             raise ValueError("parameter's name can not be empty")
    #     return self


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
            if not isinstance(item, Dict):
                raise ParserError("Each function definition must be an object")
        try:
            functions = [Function(**item) for item in data]
            return functions
        except ValidationError as e:
            err = e.errors()[0]
            field = ".".join(map(str, err["loc"]))
            msg = err["msg"]
            raise ParserError(f"Invalid function definition: '{field}' "
                              f"→ {msg}")
        except Exception as e:
            raise ParserError(f"{e}")
