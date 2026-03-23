_This project has been created as part of the 42 curriculum by **danborys**_

# call me maybe

## Description

This project implements a **function calling system** for Large Language Models (LLMs),
built as part of the 42 school curriculum. The goal is to bridge the gap between
natural language user requests and structured, machine-executable function calls.

### Goal

Given a natural language prompt such as:

> "What is the sum of 2 and 3?"

the system identifies the correct function to call and extracts its arguments,
producing a structured JSON output:
```json
{
  "prompt": "What is the sum of 2 and 3?",
  "name": "fn_add_numbers",
  "parameters": {
    "a": 2.0,
    "b": 3.0
  }
}
```

### Overview

The system is built around **constrained decoding** — a technique that guides the
LLM's token generation process by restricting which tokens are valid at each step.
Rather than prompting the model and hoping for structured output, the system
intervenes directly in the generation pipeline:

1. **Function selection** — the model is constrained to generate only valid function
   names from the provided definitions, token by token.
2. **Parameter extraction** — each parameter value is generated under type-aware
   constraints: numeric tokens only for numbers and safe vocabulary tokens for strings.
3. **JSON assembly** — results are validated and serialized into a schema-compliant
   JSON output file.

The system runs on **Qwen3-0.6B**, a 500 million parameter model. Despite its small
size, constrained decoding allows it to achieve reliability comparable to much larger
models by structurally guiding every token it produces

## Instructions

### Requirements

- Python 3.13 or higher
- [uv](https://docs.astral.sh/uv/) package manager

### Installation

Clone the repository and install all dependencies using `uv`:
```bash
git clone <repository_url>
cd call-me-maybe
uv sync
```

This will automatically install all required dependencies:
- `numpy >= 2.4.2`
- `pydantic >= 2.12.5`
- `llm_sdk` (local workspace package)


### Input Files

Place your input files in the `data/input/` directory before running:
```
data/
  input/
    functions_definition.json   # available function definitions
    function_calling_tests.json # natural language prompts to process
  output/
    function_calls.json         # generated output (created automatically)
```

### Execution

Run the project with:
```bash
uv run python3 -m src
```

Or using the Makefile:
```bash
make run
```

### Output

Results are written to `data/output/function_calls.json` as a JSON array:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  }
]
```

### Additional Makefile Commands

| Command | Description |
|---|---|
| `make install` | Install dependencies only |
| `make run` | Install and run the project |
| `make debug` | Run with Python debugger (`pdb`) |
| `make clean` | Remove `__pycache__` and `.mypy_cache` directories |
| `make lint` | Run `flake8` and `mypy` static analysis |

### Notes

- The first run will download the **Qwen3-0.6B** model weights from Hugging Face,
  which requires an internet connection and approximately 1.5GB of disk space.
  Subsequent runs use the cached model and are significantly faster.
- Processing time depends on hardware. On standard CPU hardware, all prompts
  should complete within 5 minutes.

### 💾 Storage & Cache Management (42 / Linux)
If you are working in an environment with limited home directory storage (like the 42 network), you must redirect heavy caches and virtual environments to a secondary storage partition (e.g., /sgoinfre).

1. Create External Directories
First, create the necessary folders on the high-capacity storage drive:
```
Bash
mkdir -p /sgoinfre/$USER/envs
mkdir -p /sgoinfre/$USER/uv_cache
mkdir -p /sgoinfre/$USER/huggingface
```
2. Configure Environment Variables
Add the following lines to your **~/.zshrc (or ~/.bashrc)** to redirect uv and Hugging Face traffic away from your home quota:
```
export UV_CACHE_DIR="/sgoinfre/$USER/uv_cache"
export HF_HOME="/sgoinfre/$USER/huggingface"
```

3. Setup Virtual Environment via Symlink
To keep your project "standard" while storing the heavy library files externally, create the environment in sgoinfre and link it back to your project folder:
```
uv venv /sgoinfre/$USER/envs/llm_env
```
Activate the environment
```
source /sgoinfre/$USER/envs/llm_env/bin/activate
```
Create a symbolic link in your project root
```
ln -s /sgoinfre/$USER/envs/llm_env .venv
```
This allows VS Code and other tools to 'see' the .venv locally


## 💡 Why this works
UV_CACHE_DIR: Prevents uv from filling up your home disk with downloaded .whl files.

HF_HOME: Redirects the large LLM model weights (e.g., Qwen's 1.5GB) to the secondary drive.

Symlink (ln -s): Acts as a "portal." Your project directory remains lightweight, but your IDE still recognizes the local .venv for IntelliSense and debugging.

## Algorithm

The system uses **constrained decoding** to guide a 600M parameter LLM into
producing structured JSON output with 100% schema compliance. Instead of prompting
the model and parsing its free-form response, the system intervenes at every single
token generation step — restricting which tokens the model is allowed to choose.

The core idea:
```
Normal LLM:      logits → argmax → any token
Constrained LLM: logits → mask → argmax → only valid tokens
```

The masking mechanism at every generation step:
```python
arr = np.array(logits, dtype=np.float32)
mask = np.full(len(arr), -np.inf, dtype=np.float32)  # block everything
mask[allowed_ids] = arr[allowed_ids]                  # unblock valid tokens
return int(np.argmax(mask))                           # pick best valid token
```

Setting invalid tokens to `-inf` ensures they can never win the `argmax` —
the model is structurally incapable of generating an invalid token at any step.

The pipeline runs in four stages:

**1. Vocabulary pre-computation** — every token in the vocabulary (~150,000) is
decoded once and categorized into pre-built numpy arrays:

| Token set | Contents | Used for |
|---|---|---|
| `numeric_base_ids` | digits, `.`, `-` | first token of a number |
| `numeric_ids` | digits + `,` + `}` | subsequent number tokens |
| `str_ids` | all tokens except structural chars and embedded `"` | string values |
| `regex_ids` | `str_ids` minus tokens containing `)` or `]` | regex generation |

**2. Function name selection** — the model is constrained to generate only tokens
that continue a valid function name prefix. At each step, allowed tokens are
computed by checking which function names are still reachable:
```python
for name, tokens in func_name_tokens.items():
    if tokens[:generated_len] == generated_tokens:
        allowed.add(tokens[generated_len])
```

**3. Parameter extraction** — each parameter is generated under type-specific
constraints:
- `numeric` — digits only for first token, digits + terminators for subsequent
- `string` — safe vocabulary tokens, terminates on `"`
- `regex` — restricted token set with forced group closing after quantifiers

**4. JSON assembly** — parameters are cast to correct Python types and serialized
via `json.dumps` which handles all escaping and formatting automatically.

---

## Design Decisions

**numpy arrays over Python lists for token masking**
The vocabulary has ~150,000 tokens. A Python loop over all tokens to apply a mask
runs in O(n) with Python overhead. numpy vectorized indexing does the same
operation at C speed:
```python
mask[allowed_ids] = arr[allowed_ids]  # one C-level operation
```
Token sets are pre-computed once at startup so the expensive O(vocab_size) decode
loop runs exactly once per program execution regardless of how many prompts are
processed.

**Append-only `input_ids`**
Rather than re-encoding the full context string on every token generation step,
a running `input_ids` list is maintained and tokens are appended one by one.
Re-encoding from scratch would be O(n²) in the number of generated tokens.

**Two numeric token sets**
`numeric_base_ids` (digits only) is used for the first token of a number,
while `numeric_ids` (digits + `,` + `}`) is used for subsequent tokens. This
prevents the model from immediately emitting a terminator before generating
any digits — which would produce empty numeric values.

**Explicit punctuation control**
The system never relies on model-emitted delimiters. Separators (`, `) and
closing quotes are appended explicitly after each parameter value. This
eliminates the double-comma bug that occurs when the model emits a terminator
that is also appended manually.

**Regex post-processing**
Rather than tracking bracket balance during generation, unclosed groups are
repaired after generation:
```python
raw += "]" * (raw.count("[") - raw.count("]"))
raw += ")" * (raw.count("(") - raw.count(")"))
```
This is simpler and more reliable than attempting to force closing tokens
during the generation loop.

**Negative number boosting**
When the prompt contains a negative number, logits for `-` and ` -` tokens
are boosted by `+4.0` before the first numeric token is selected. This corrects
the model's bias toward positive numbers from JSON context without restricting
the token set.

---
## Performance Analysis

**Accuracy**

| Category | Result |
|---|---|
| Function name selection | 100% correct on all test cases |
| Numeric extraction | ~90% correct (negative numbers occasionally missed) |
| String extraction | 100% correct (for simple prompts)|
| Regex generation | ~90% semantically correct, 100% structurally valid after post-processing |
| Boolean extraction | 100% correct |

**Speed**

On standard CPU hardware the full test suite completes within the 5-minute
requirement. The vocabulary pre-computation adds ~30-60 seconds once at startup.

**Reliability**

- 100% valid JSON output on every run — structural constraints make malformed
  JSON impossible
- No infinite loops — `max_str_tokens` (80) and `max_num_tokens` (50) provide
  hard upper bounds on all generation loops
- Repetition guard prevents the most common failure mode in string generation

---

## Challenges Faced

**String generation loops**
The model would sometimes enter repetition loops generating patterns like
`vowels in all vowels in all vowels in all...`. Solved with a repetition guard
that detects when the last N tokens repeat the N tokens before them across
window sizes 2-6, then trims the repeated suffix.

**Regex termination**
Early versions used `)` and `]` as termination signals, which caused the model
to terminate before closing groups — producing broken patterns like `(\d+`
instead of `(\d+)`. Solved by removing `)` and `]` from termination conditions
and instead using post-processing to close any unclosed groups after generation.

**Negative numbers**
The JSON context prefix biased the model toward positive numbers even when the
prompt contained negative values. The model consistently picked `2` instead of
`-2` for prompts like `"sum of -2 and 3"`. Solved by detecting the presence of
a negative number in the prompt and boosting the logits of `-` and ` -` tokens
before the first numeric token is selected.

---

## Testing Strategy

Testing was performed by running the system against a fixed set of prompts and
manually verifying the output JSON after each change.

## Example Usage

### Basic execution
```bash
make run
```

### Custom input files
```bash
uv run python3 -m src \
  --functions_definition data/input/functions_definition.json \
  --input data/input/function_calling_tests.json \
  --output data/output/function_calls.json
```

### Input format

`functions_definition.json`:
```json
[
  {
    "name": "fn_add_numbers",
    "description": "Add two numbers together and return their sum.",
    "parameters": {
      "a": { "type": "number" },
      "b": { "type": "number" }
    },
    "returns": { "type": "number" }
  }
]
```

`function_calling_tests.json`:
```json
[
  { "prompt": "What is the sum of 2 and 3?" },
  { "prompt": "Greet shrek" },
  { "prompt": "Reverse the string 'hello'" }
]
```

### Output format

`function_calls.json`:
```json
[
  {
    "prompt": "What is the sum of 2 and 3?",
    "name": "fn_add_numbers",
    "parameters": {
      "a": 2.0,
      "b": 3.0
    }
  },
  {
    "prompt": "Greet shrek",
    "name": "fn_greet",
    "parameters": {
      "name": "shrek"
    }
  },
  {
    "prompt": "Reverse the string 'hello'",
    "name": "fn_reverse_string",
    "parameters": {
      "s": "hello"
    }
  }
]
```

## Resources

### Documentation & References

- [Hugging Face](https://huggingface.co/) — model hub and transformers documentation,
  used to access and understand the Qwen3-0.6B model and its tokenizer structure
- [Qwen3-0.6B Model Card](https://huggingface.co/Qwen/Qwen3-0.6B) — model
  architecture details and tokenizer specifications
- [Pydantic Documentation](https://docs.pydantic.dev/) — data validation and
  settings management using Python type annotations
- [NumPy Documentation](https://numpy.org/doc/) — array operations used for
  efficient logit masking during constrained decoding
- [JSON Specification (RFC 8259)](https://datatracker.ietf.org/doc/html/rfc8259) —
  reference for valid JSON structure and string encoding rules
- [Python `re` module](https://docs.python.org/3/library/re.html) — regular
  expression syntax reference used for regex parameter validation
- [Byte Pair Encoding (BPE)](https://huggingface.co/learn/nlp-course/en/chapter6/5)
  — explanation of the tokenization algorithm used by Qwen3-0.6B

### AI Usage

AI assistance was used in the following parts of this project:

| Task | Tool |
|---|---|
| Writing docstrings for functions and classes | ChatGPT |
| Designing and iterating on `build_vocab_index()` — vocabulary pre-computation, token set construction, and numpy index arrays | Claude AI |
| Writing this README file | Claude AI |
