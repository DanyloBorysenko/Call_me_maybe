from pydantic import BaseModel, ValidationError, Field, field_validator


class OutputWriter(BaseModel):
    data: str = Field(min_length=2)

    @field_validator("data")
    def check_data(cls, data: str) -> str:
        if (
            data.startswith("[") and
            data.endswith("]")
        ):
            return data
        raise ValidationError(
            ("Output data must start with '['"
             "and end with ']")
        )
    
    def write_to_file(filename: str) -> bool:
        