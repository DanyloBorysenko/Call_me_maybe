import time
import json
from pathlib import Path
from typing import Dict, List, Final
# from memory_profiler import profile


class Tokenizer:
    """Tokenizer built from a vocabulary JSON file.

    Implements greedy longest-match encoding and direct
    index-based decoding. Vocabulary is loaded once at
    construction and cached for the lifetime of the object.

    Attributes:
        vocab_path: Path to the vocabulary JSON file
        vocabulary_size: Number of tokens in the vocabulary
        max_token_length: Length of the longest token string

    Example:
        tokenizer = Tokenizer("path/to/vocab.json")
        ids = tokenizer.encode("Hello world")
        text = tokenizer.decode(ids)
    """

    _UNICODE_MAP: Final[Dict[str, str]] = {
        "Ġ": " ",    # space prefix (BPE artifact)
        "Ċ": "\n",   # newline
        "ĉ": "\t",   # tab
    }

    def __init__(self, path: str) -> None:
        """Load and index vocabulary from a JSON file.

        Args:
            path: Path to vocabulary JSON mapping token strings to IDs

        Raises:
            FileNotFoundError: If vocabulary file doesn't exist
            ValueError: If vocabulary file is malformed
        """
        self._vocab_path: Path = Path(path)
        self._decoder_list: List[str] = []
        self._encoder_dict: Dict[str, int] = {}
        self._max_token_len: int = 0
        self._create_encoder_decoder()

    def _refine_key(self, key: str) -> str:
        """Convert unicode artifacts to standard characters.
        """
        for k, v in self._UNICODE_MAP.items():
            key = key.replace(k, v)
        # new_key = key.replace("Ġ", " ").replace("Ċ", "\n").replace("ĉ", "\t")
        return key

    def _create_encoder_decoder(self) -> None:
        """Load vocabulary from JSON and build encoder/decoder indexes."""
        if not self._vocab_path.exists():
            raise FileNotFoundError(
                f"Vocabulary file not found: '{self._vocab_path}'"
            )

        with open(self._vocab_path, 'r') as fl:
            data = json.load(fl)

        if not data:
            raise ValueError("Vocabulary file is empty")

        max_id = max(data.values())
        self._decoder_list = [""] * (max_id + 1)

        for key, val in data.items():
            new_key = self._refine_key(key)
            self._encoder_dict[new_key] = val
            # self._decoder_list[val] = new_key
            self._decoder_list[val] = new_key
        del data
        self._reset_max_token_len()

    def _reset_max_token_len(self) -> None:
        self._max_token_len = \
            self.get_longest_str_in_list(self._decoder_list)

    @staticmethod
    def get_longest_str_in_list(str_list: List[str]) -> int:
        max_len: int = 0
        for s in str_list:
            if len(s) > max_len:
                max_len = len(s)
        return max_len

    def get_all_tokes(self) -> List[str]:
        return self._decoder_list.copy()

    def encode(self, string: str) -> List[int]:
        """Scan from right to left an ake the longest matching token

        Args:
            string (str): String to encode

        Returns:
            List[int]: List of integer token IDs
        """
        tokens: List[int] = []
        str_len = len(string)
        left_ptr = 0
        if str_len > self._max_token_len:
            right_ptr = left_ptr + self._max_token_len
        else:
            right_ptr = str_len
        while left_ptr < right_ptr:
            sub_str = string[left_ptr: right_ptr]
            token = self._encoder_dict.get(sub_str)
            if token is not None:
                # print(f"Match found: {sub_str}, {left_ptr}, {right_ptr}")
                tokens.append(token)
                left_ptr = right_ptr
                if left_ptr < str_len:
                    if (str_len - left_ptr) < self._max_token_len:
                        right_ptr = str_len
                    else:
                        right_ptr = left_ptr + self._max_token_len
            else:
                right_ptr -= 1
        return tokens

    def decode(self, tokens: List[int] | int) -> str:
        """Decode a list of token IDs back to a string.

        Args:
            token_ids: List of integer token IDs to decode

        Returns:
            Decoded string

        Raises:
            TokenDecodeError: If any token ID is out of range
        """
        string = ""
        if isinstance(tokens, List):
            for token in tokens:
                try:
                    string += self._decoder_list[token]
                except IndexError as e:
                    string = ""
                    raise IndexError(f"Token ID {token}, not found: {e}")
        elif isinstance(tokens, int):
            try:
                string += self._decoder_list[tokens]
            except IndexError as e:
                string = ""
                raise IndexError(f"Token ID {tokens}, not found: {e}")
        else:
            print(f"Error: {tokens} can not be decoded")
        return string


def test_toke_encoder() -> None:
    start = time.time()
    from llm_sdk import Small_LLM_Model
    llm = Small_LLM_Model()
    token_path = llm.get_path_to_vocab_file()

    tokenizer = Tokenizer(path=token_path)
    # print(len(encoder_decoder._decoder_list))
    msg = "Greet shrek"
    start_1 = time.time()
    tokens1 = llm.encode(msg).tolist()[0]
    print(f"Built-in encoder: {tokens1}")
    end_1 = time.time()
    print(f"Time taken: {(end_1 - start_1):.3f}s")
    start_1 = time.time()
    tokens2 = tokenizer.encode(msg)
    print(f"Custom encoder: {tokens2}")
    end_1 = time.time()
    print(f"Time taken: {(end_1 - start_1):.3f}s")

    start_1 = time.time()
    print(f"Built-in decoder: {[llm.decode(token) for token in tokens1]}")
    end_1 = time.time()
    print(f"Time taken: {(end_1 - start_1):.3f}s")
    start_1 = time.time()
    print("Custom decoder: "
          f"{[tokenizer.decode([token]) for token in tokens2]}")
    end_1 = time.time()
    print(f"Time taken: {(end_1 - start_1):.3f}s")

    end = time.time()
    print(f"Time taken: {(end - start):.3f}s")


if __name__ == "__main__":
    # test()
    test_toke_encoder()
