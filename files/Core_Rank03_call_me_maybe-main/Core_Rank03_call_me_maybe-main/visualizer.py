import streamlit as st
from typing import Tuple, List
from src.__main__ import initialize_pipeline, function_generator
from src.parser import FnInfo
from src.constrain_decoder import ConstrainDecoder


@st.cache_resource
def load_pipeline() -> Tuple[ConstrainDecoder, List[FnInfo]]:
    """Loads once, stays cached across all user interactions."""
    with st.spinner("Loading model..."):
        # st.write("Loading model... (only happens once)"
        pass
    decoder, functions = initialize_pipeline(device='cpu')
    return decoder, functions


st.title("🤙 Call Me Maybe")
st.caption("Natural language [Model: Qwen/Qwen3-0.6B] → "
           "structured function calls "
           "using constrained decoding")

# Load pipeline once
decoder, functions = load_pipeline()

col1, col2 = st.columns(2)

with col1:
    prompt = st.text_area(
        "Your prompt",
        placeholder="What is the sum of 40 and 2?")
    run = st.button("Generate Function Call")

with col2:
    # Show available functions so user knows what to ask
    all_func = ""
    for fn in functions:
        args = ", ".join(
            f"{k}: {v}" for k, v in fn.args_types.items())
        all_func += f"{fn.fn_name} ({args})\n"
    st.text_area(
        f"Available functions ({len(functions)}):",
        value=all_func,
        height=200,
    )

if run and prompt:
    with st.spinner("Generating..."):
        try:
            result, cost = function_generator(prompt, decoder)
            st.success("Done!")

            # Main result
            st.subheader("Result")
            st.json({
                "prompt": result.prompt,
                "name": result.name,
                "parameters": result.parameters
            })
            st.markdown(f"Tokens Used: :green[{cost.token_used}],"
                        f" Time: :green[{cost.time_taken_seconds}]s, "
                        f"Cost/Token: :green[{cost.avg_time}]s")

        except Exception as e:
            st.error(f"Error: {e}")
