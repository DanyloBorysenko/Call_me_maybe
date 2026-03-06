from pydantic import BaseModel, model_validator, Field
from typing import Dict, List
import json
from pathlib import Path
from pydantic import ValidationError


class ParserError(Exception):
    pass


class Function(BaseModel):
    name: str = Field(min_length=1)
    description: str = Field(min_length=1)
    parameters: Dict[str, Dict[str, str]]
    returns: Dict[str, str]


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
            with open(Path(self.functions_definition), "r") as f:
                data = json.load(f)
                if not isinstance(data, list):
                    raise ParserError("Functions definition must be "
                                      "a list")
                functions = [Function(**item) for item in data]
                return functions
        except FileNotFoundError:
            raise ParserError(f"File {self.functions_definition} not found")
        except ValidationError as e:
            # error = e.errors()[0]
            # field = ".".join(map(str, error["loc"]))
            # msg = error["msg"]
            # raise ParserError(f"{field}: {msg}")
            field = e.errors()[0]["loc"]
            raise ParserError(f"Field: '{field[0]}',"
                              f" msg: {e.errors()[0]['msg']}")
        except Exception as e:
            raise ParserError(f"{e}")
