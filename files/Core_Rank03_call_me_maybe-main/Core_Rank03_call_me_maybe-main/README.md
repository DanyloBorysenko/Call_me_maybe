# call_me_maybe

*This project has been created as part of the 42 curriculum by sudas*

---

## Description

Small language models are resource-efficient but inherently unreliable at generating
structured output. They lose context awareness quickly and when prompted to produce
JSON, a 0.6B parameter model succeeds only around 30% of the time without guidance.

This project implements **constrained decoding** — a technique that intervenes in the
model's token-by-token generation process to guarantee 100% structurally valid JSON
output, even with the small Qwen3-0.6B model. Instead of answering a question
directly, the model is guided to predict which function to call and what arguments
to pass, producing a structured function call as output.

**Example:**

```
Input:  "What is the sum of 40 and 2?"

Output: {
    "name": "fn_add_numbers",
    "parameters": {"a": 40, "b": 2}
}
```

This project extends beyond the base requirements with two additional features:

- A **custom BPE tokenizer** built from scratch that encodes and decodes tokens
  directly from the model vocabulary file, without relying on the HuggingFace
  tokenizer.
- A **Streamlit web interface** that provides an interactive browser-based UI for
  running the model and exploring results in real time.

---

## Instructions

### Requirements

- Python 3.10 or later
- [uv](https://github.com/astral-sh/uv) package manager
- ~2GB disk space for model weights

### Installation

```bash
git clone <git@github.com:sukanta0011/Core_Rank03_call_me_maybe.git>
cd call_me_maybe
uv sync
```

### Running

```bash
# Run with default input/output paths
make run

# Or directly
uv run python3 -m src

# Run with custom paths
uv run python3 -m src \
    --functions_definition data/input/functions_definition.json \
    --input data/input/function_calling_tests.json \
    --output data/output/function_calling_results.json

# Run the interactive web interface
source .venv/bin/activate # Activate the virtual environment if not active
make visual
# Or
streamlit run visualizer.py
```

By default the program reads function definitions and prompts from `data/input/`
and writes results to `data/output/`. Use the `--input`, `--functions_definition`,
and `--output` flags to set custom paths.

### Running on a 42 School Computer

The model weights require approximately 2GB of storage. The default home directory
on 42 machines does not have enough space, so you must redirect the cache to
`goinfre` or `sgoinfre` before running.

```bash
# Create cache directories
mkdir -p /goinfre/$USER/uv_cache
mkdir -p /goinfre/$USER/hf_cache

# Set environment variables
export UV_CACHE_DIR="/goinfre/$USER/uv_cache"

```

> **Important:** Also copy your project into `goinfre` or `sgoinfre`. Running from
> your home directory while the cache is in `goinfre` may cause issues.

---

## Algorithm Explanation

### Overview

The pipeline has three stages for each prompt:

```
System Prompt (function descriptions)
        │
        ▼
┌─────────────────────┐
│  Function Selection │  ← Hard-masked constrained decoding
│                     │    Only valid fn_name tokens allowed
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│  Argument Extraction│  ← Soft-biased generation
│                     │    Prompt tokens preferred
└──────────┬──────────┘
           │
           ▼
  Pydantic JSON Output
  (100% schema-valid)
```

### System Prompt

Before generation begins, the model receives a system prompt listing all available
functions with their names and descriptions. This gives the model the context it
needs to select the correct function.

### Constrained Decoding — Function Name

Function name generation uses **hard masking** — a trie-based approach where at each
token position, only tokens that are valid continuations of a known function name
are allowed. All other tokens are masked to `-1e9` (effectively impossible to select).

This guarantees the output is always a valid function name from the provided list,
never a hallucination.

### Constrained Decoding — Arguments

Argument generation uses **soft biasing** rather than hard masking. Tokens that
appear in the original prompt receive a logit bonus, nudging the model to extract
values from the prompt rather than generating them freely. Two special cases apply:

- `regex` arguments: tokens starting with regex metacharacters (`\`, `[`, `+`, etc.)
  receive an additional bias to help the model generate valid patterns.
- `replacement` arguments: tokens starting with symbol characters (`*`, `#`, `_`, etc.)
  receive an additional bias to help the model pick symbols rather than their word
  descriptions.

### Word Anchoring

To reduce token usage and improve accuracy, a regex search runs at each generation
step to check whether the partially generated string uniquely matches a word or
number in the original prompt. If a unique match is found, the model jumps directly
to the complete value instead of generating character by character. This is
particularly effective for long numbers — instead of spending 9 tokens on a
9-digit number, it takes only 1-2 tokens to identify the match.

### Output Construction

Rather than relying on the model to produce valid JSON on its own, all values are
stored separately during generation and assembled into the final output using
Pydantic's JSON serialization. This guarantees 100% valid, schema-compliant output
regardless of what the model generates internally.

---

## Design Decisions

### Function Name Selection — Hard Mask

Function names are generated using strict trie-based token masking. At each position,
only the token that the correct function name expects at that exact character position
is allowed. This makes wrong function names structurally impossible to generate.

### Argument Extraction — Soft Bias

Arguments use soft biasing instead of hard masking because argument values must come
from the prompt — which is open-ended text. Hard masking would be too restrictive.
The soft bias approach nudges the model toward prompt tokens while still allowing
some freedom for cases like regex patterns that do not appear verbatim in the prompt.

### Output Serialisation

The model never writes JSON directly. Each generated value is extracted, type-converted,
and stored in a Pydantic model. The final JSON is produced by Pydantic's `dump_json`
— this is what guarantees structural validity unconditionally.

---

## Performance Analysis

**Speed:** The LLM SDK provided for this project does not support KV caching, meaning
every new token requires a full forward pass over the entire prompt from scratch. This
makes generation time scale as O(N²) with prompt length. Keeping the system prompt
concise is therefore important for performance.

**Function selection accuracy:** The model reliably identifies the correct function
when the prompt is clear and unambiguous.

**Number extraction:** Numbers are extracted accurately. Negative numbers occasionally
cause issues because the model assigns lower probability to the `-` sign than to the
digit that follows — addressed with a negative sign bias of 6.

**String extraction:** Strings with clear boundaries (quoted in the prompt) are
extracted reliably. Unquoted strings with ambiguous boundaries occasionally fail.

**Regex generation:** Simple regex patterns (e.g. `\d+`, `[aeiou]`) succeed
intermittently with the bias approach. Complex patterns fail consistently.
This is a known limitation of 0.6B parameter models — regex generation requires
a level of structural reasoning that exceeds the model's capacity without a
dedicated second-stage generation pass.

---

## Challenges and Solutions

**Challenge 1 — Stopping generation at the right point**

Without intervention, the model generates indefinitely. The solution is to use
a terminating token: the model is primed with an opening `"` for string values
or nothing for numeric values, and generation stops the moment the model produces
the corresponding closing `"` or `,` terminator.

**Challenge 2 — Extracting unquoted strings**

When argument values are not surrounded by quotes in the prompt, the model struggles
to identify their boundaries. The solution was to add few-shot examples to the system
prompt showing the expected input-output format, which significantly improved accuracy
on unquoted values.

**Challenge 3 — Negative number extraction**

The model assigns low probability to the `-` sign even when a negative number is
clearly indicated in the prompt. The solution was to apply a soft bias of +6 to the
`-` token within the allowed numeric token set.

**Challenge 4 — Regex and symbol generation**

The model generates word descriptions of patterns instead of the patterns themselves
(e.g. `vowels` instead of `[aeiouAEIOU]`). The solution was to precompute token ID
lists for known regex metacharacters and symbol characters and apply an extra logit
bias to those tokens during generation. This improves performance on simple patterns
but complex regex remains unreliable at this model scale.

---

## Testing Strategy

The project was validated manually against the provided test cases in
`function_calling_tests.json`. Each prompt was run and the output was inspected for:

- Correct function name selection
- Correct argument extraction with proper types
- 100% valid JSON structure in all outputs

Automated testing was not implemented in this version. A `pytest` test suite
covering function selection accuracy and argument type correctness is a planned
improvement.

---

## Example Usage

**CLI:**
```bash
uv run python3 -m src \
    --input data/input/function_calling_tests.json \
    --functions_definition data/input/functions_definition.json \
    --output data/output/results.json
```

**Interactive mode:**
```bash
streamlit run visualizer.py
```
Enter any natural language prompt in the text box and click **Generate Function Call**.
The available functions are shown on the right side of the interface so you know
what kinds of prompts the model can handle.

**Example prompts to try:**
```
What is the sum of 265 and 345?
Greet john
Reverse the string 'hello'
Calculate the square root of 144
Replace all vowels in 'Programming is fun' with asterisks
Substitute the word 'cat' with 'dog' in 'The cat sat on the mat'
```

---

## Resources

- [uv — Python package and environment manager](https://github.com/astral-sh/uv)
- [Pydantic documentation](https://docs.pydantic.dev/latest/)
- [Qwen3 model on HuggingFace](https://huggingface.co/Qwen/Qwen3-0.6B)
- [Constrained decoding — overview](https://huggingface.co/blog/constrained-beam-search)
- [BPE tokenization explained](https://huggingface.co/learn/nlp-course/en/chapter6/5)
- [Streamlit documentation](https://docs.streamlit.io)
- [Python memory usage](https://code.tutsplus.com/understand-how-much-memory-your-python-objects-use--cms-25609t)

---

## AI Usage

AI assistance (Claude, claude.ai) was used for the following:

- Understanding how LLMs work, what constrained decoding is, and how soft biasing
  and hard masking differ conceptually.
- Validating algorithmic ideas and discussing tradeoffs before implementation.
- Writing docstrings for the codebase.
- Reviewing code structure and discussing how a senior developer would approach
  specific design decisions.
- Partial assistance with the Streamlit visualizer interface.
- Improving the writing and structure of this README.

All core logic — the constrained decoding pipeline, custom tokenizer, argument
extraction strategy, and word anchoring optimisation — was designed and implemented
independently. AI was used as a discussion partner and reviewer, not as a code
generator.