from pydantic import BaseModel,  model_validator
from typing import List, Dict, Self


class ArgValidator(BaseModel):
    """
    Validate CLI arguments and update configuration file paths.

    The class expects arguments in flag-path pairs (e.g.``--input file.json``).
    It verifies that flags are known, not duplicated, and that provided paths
    end with ``.json``. Valid paths override the corresponding defaults in
    ``config_files``.

    Attributes:
        args (List[str]): Raw command-line arguments (usually ``sys.argv``).
        config_files (Dict[str, str]): Mapping of CLI flags to JSON file paths.

    Raises:
        ValueError: If arguments are malformed, duplicated, unknown, or
        if a file path does not end with ``.json``.
    """

    args: List[str]
    config_files: Dict[str, str]

    @model_validator(mode="after")
    def check_args(self) -> Self:
        """
        Validate CLI arguments and apply user-provided config paths.

        Returns:
            Self: The validated model instance.

        Raises:
            ValueError: If arguments are malformed or unsupported.
        """
        self.args = self.args[1:]
        flags = self.args[::2]
        paths = self.args[1::2]
        flags_len = len(flags)
        paths_len = len(paths)
        max_arg_count = len(self.config_files)
        if flags_len != paths_len:
            raise ValueError("Missed argument or file path.")
        if flags_len > max_arg_count:
            raise ValueError(f"Too many arguments. Max is {max_arg_count}")
        if flags_len != len(set(flags)):
            raise ValueError("Duplicate flags detected")
        if paths_len != len(set(paths)):
            raise ValueError("Duplicate paths detected")
        input_config_files: Dict[str, str] = dict(zip(flags, paths))
        for flag, file_path in input_config_files.items():
            if not self.config_files.get(flag, None):
                raise ValueError(f"Unknown argument: '{flag}'. Correct args"
                                 f": {', '.join(self.config_files.keys())}")
            if not file_path.endswith(".json"):
                raise ValueError(f"File path {file_path} doesn't end with "
                                 "'.json' suffix")
            self.config_files[flag] = file_path
        return self
