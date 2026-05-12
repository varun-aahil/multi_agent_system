import streamlit as st
from multi_agent_system import create_llm, build_graph, build_initial_state, get_available_models

st.set_page_config(page_title="Multi-Agent Travel Planner", page_icon="✈️", layout="wide")

st.title("✈️ Intelligent Travel Planner")
st.markdown("**Powered by LangChain, LangGraph, and local Ollama models**")

st.write(
    "Enter a description of your ideal trip, including your destination, duration, "
    "and budget, and our multi-agent system will generate a complete travel guide for you."
)

available_models = get_available_models()
model_choice = st.selectbox(
    "Select Ollama Model",
    options=available_models,
    index=0 if "gemma3:12b" not in available_models else available_models.index("gemma3:12b"),
    help="Select a model available on your local Ollama instance."
)

user_request = st.text_area(
    "Your Travel Request",
    value="Plan a 5-day trip to Japan with a budget of $2500",
    height=100
)

if st.button("Generate Travel Guide", type="primary"):
    if not user_request.strip():
        st.error("Please enter a travel request before proceeding.")
    else:
        status_placeholder = st.empty()
        with st.spinner("Our AI agents are working..."):
            try:
                # Initialize the system
                status_placeholder.info("🤖 Initializing AI Agents...")
                llm = create_llm(model_name=model_choice)
                graph = build_graph(llm)
                initial_state = build_initial_state(user_request)
                
                # Run the LangGraph workflow
                status_placeholder.info("🔍 Validating request and planning")
                final_state = graph.invoke(initial_state)
                
                final_output = final_state.get("final_output", "").strip()
                status_placeholder.empty()
                
                if final_output:
                    st.success("Your Travel Guide is ready!")
                    st.markdown("---")
                    # Display the generated markdown output
                    st.markdown(final_output)
                else:
                    st.warning("The system completed the workflow, but no output was generated.")
                
                # Display any system logs or errors if they occurred
                if final_state.get("errors"):
                    st.markdown("---")
                    st.subheader("System Trace / Execution Logs")
                    for error in final_state["errors"]:
                        st.info(error)
                        
            except Exception as e:
                st.error(f"An unexpected error occurred during execution: {e}")