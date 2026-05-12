"""
multi_agent_system.py

Production-ready Multi-Agent Travel Planner built with LangChain + LangGraph
and powered by a local Ollama model via ChatOllama.
"""

from __future__ import annotations

import json
import re
import sys
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, TypedDict
from tenacity import retry, stop_after_attempt, wait_exponential


def _import_dependencies() -> tuple[Any, Any, Any, Any, Any]:
    try:
        from langchain_core.messages import HumanMessage, SystemMessage
        from langgraph.graph import END, START, StateGraph
        try:
            from langchain_ollama import ChatOllama
        except ImportError:
            from langchain_community.chat_models import ChatOllama
        return ChatOllama, HumanMessage, SystemMessage, StateGraph, (START, END)
    except ImportError as exc:
        raise RuntimeError(f"Missing dependencies: {exc}") from exc


ChatOllama, HumanMessage, SystemMessage, StateGraph, GRAPH_SENTINELS = _import_dependencies()
START, END = GRAPH_SENTINELS


class TravelPlannerState(TypedDict):
    original_request: str
    destination: str
    number_of_days: int
    budget: str
    budget_amount: float
    budget_currency: str
    requirements_summary: str
    itinerary: str
    budget_breakdown: str
    final_output: str
    errors: List[str]
    is_valid: bool
    validation_message: str


def get_available_models() -> List[str]:
    """
    Fetch the list of models currently available in the local Ollama instance,
    filtering out embedding models and other non-chat models.
    """
    import requests
    try:
        response = requests.get("http://localhost:11434/api/tags", timeout=2)
        if response.status_code == 200:
            all_models = response.json().get("models", [])
            chat_models = []
            
            # Heuristics to filter out embedding models
            # 1. Known embedding/encoder families
            excluded_families = {"bert", "nomic-bert"}
            # 2. Keywords in name that indicate non-chat purpose
            excluded_keywords = {"embed", "rerank", "clip"}

            for m in all_models:
                name = m.get("name", "").lower()
                details = m.get("details", {})
                family = str(details.get("family", "")).lower()
                families = [f.lower() for f in details.get("families", []) if f]

                # Filter by family
                if family in excluded_families or any(f in excluded_families for f in families):
                    continue
                
                # Filter by keywords in name
                if any(kw in name for kw in excluded_keywords):
                    continue
                
                chat_models.append(m["name"])
            
            return chat_models if chat_models else ["gemma3:12b"]
    except Exception:
        pass
    return ["gemma3:12b"]


def create_llm(model_name: str = "gemma3:12b") -> Any:
    return ChatOllama(
        model=model_name,
        temperature=0,
        num_predict=4000,
    )


@retry(
    stop=stop_after_attempt(2),
    wait=wait_exponential(multiplier=1, min=1, max=4),
    reraise=True
)
def invoke_llm(llm: Any, system_prompt: str, user_prompt: str) -> str:
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=user_prompt),
        ])
        content = getattr(response, "content", "")
        if isinstance(content, list):
            content = "\n".join(str(part) for part in content if part is not None)
        return str(content).strip()
    except Exception:
        raise


def extract_json_object(text: str) -> Optional[Dict[str, Any]]:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        parsed = json.loads(cleaned)
        if isinstance(parsed, dict): return parsed
    except: pass
    match = re.search(r"\{.*\}", cleaned, re.DOTALL)
    if match:
        try:
            parsed = json.loads(match.group(0))
            if isinstance(parsed, dict): return parsed
        except: pass
    return None


def append_error(existing_errors: List[str], message: str) -> List[str]:
    return existing_errors + [message]


def analyzer_agent(state: TravelPlannerState, llm: Any) -> Dict[str, Any]:
    """Universal Extraction Agent."""
    system_prompt = """
You are a Travel Analysis Agent. 
Extract the following from the user request:
1. is_travel_related (boolean): Is this request about travel/planning?
2. destination (string): City/Country.
3. number_of_days (integer): If not specified, default to 3.
4. budget (string): Extract if present, otherwise set to "Not Specified".

Return ONLY JSON: {"is_travel_related": bool, "destination": "str", "number_of_days": int, "budget": "str"}
""".strip()
    try:
        output = invoke_llm(llm, system_prompt, state["original_request"])
        parsed = extract_json_object(output)
        
        if not parsed or not parsed.get("is_travel_related", True):
             # Only block if it's definitely not travel related
             if any(kw in state["original_request"].lower() for kw in ["trip", "travel", "plan", "visit", "days", "budget"]):
                 pass # Carry on, looks like travel
             else:
                 return {"is_valid": False, "validation_message": "Request does not appear to be travel-related."}
        
        # Robust extraction with fallbacks
        dest = parsed.get("destination") or "the requested location"
        days = parsed.get("number_of_days") or parsed.get("days")
        if not days:
            match = re.search(r"(\d+)\s*day", state["original_request"].lower())
            days = int(match.group(1)) if match else 3
        
        budget = parsed.get("budget") or "Not Specified"
        if budget == "Not Specified":
            match = re.search(r"(\$\d+[\d,]*|\d+[\d,]*\s*usd)", state["original_request"].lower())
            if match: budget = match.group(0)

        return {
            "is_valid": True,
            "destination": dest,
            "number_of_days": int(days),
            "budget": budget,
            "requirements_summary": f"Trip to {dest} for {days} days. Budget: {budget}."
        }
    except Exception as e:
        return {"errors": append_error(state["errors"], f"Analysis failed: {e}")}


def itinerary_planner_agent(state: TravelPlannerState, llm: Any) -> Dict[str, Any]:
    if state.get("errors"): return {}
    days = state.get("number_of_days", 3)
    system_prompt = f"""
You are an Expert Itinerary Planner.
Task: Generate a day-by-day markdown itinerary for EXACTLY {days} days.

RULES:
1. Start each day with "## Day [X]: [Location]".
2. Use EXACTLY three time-stamped bullets per day:
   - 09:00 AM: [Morning Activity]
   - 01:00 PM: [Afternoon Activity]
   - 07:00 PM: [Evening Activity]
3. EACH bullet MUST be on its own NEW LINE.
4. DO NOT ask questions.
5. DO NOT provide an introduction or conclusion.
6. Plan for all requested cities/regions: {state['destination']}.
7. Be concise but specific.
""".strip()
    user_prompt = f"Destination/Request: {state['original_request']}\nDuration: {days} days."
    try:
        itinerary = invoke_llm(llm, system_prompt, user_prompt)
        if len(itinerary) < 100 or "Day 1" not in itinerary:
             raise ValueError("Itinerary response too short or missing Day 1.")
        return {"itinerary": itinerary}
    except Exception as e:
        return {"errors": append_error(state["errors"], f"Itinerary failed: {e}")}


def budget_estimator_agent(state: TravelPlannerState, llm: Any) -> Dict[str, Any]:
    if state.get("errors"): return {}
    
    budget_context = f"Target Budget: {state['budget']}" if state['budget'] != "Not Specified" else "No budget provided. Suggest a realistic one."
    
    system_prompt = f"""
You are a Budget Estimator Agent.
Task: Generate a markdown budget table for a {state['number_of_days']} day trip.

RULES:
1. Output ONLY the markdown table.
2. Columns: Category, Estimated Cost (USD), Notes.
3. Rows: Flights, Accommodation, Food, Activities, Local Transport, Total.
4. DO NOT ask questions.
5. DO NOT provide commentary, intros, or outros.
{budget_context}
""".strip()
    user_prompt = f"Destination: {state['destination']}, Days: {state['number_of_days']}, Itinerary Context:\n{state['itinerary']}"
    try:
        breakdown = invoke_llm(llm, system_prompt, user_prompt)
        if "|" not in breakdown:
             raise ValueError("Budget breakdown missing markdown table.")
        return {"budget_breakdown": breakdown}
    except Exception as e:
        return {"errors": append_error(state["errors"], f"Budget failed: {e}")}


def summary_formatter_agent(state: TravelPlannerState, llm: Any) -> Dict[str, Any]:
    if state.get("errors"):
        return {"final_output": f"# Planning Issues Encountered\n\n{state['errors'][-1]}\n\nPartial Data:\n\n{state.get('itinerary', '')}\n\n{state.get('budget_breakdown', '')}"}
    
    if not state.get("is_valid", True):
        return {"final_output": f"# Request Blocked\n\n{state.get('validation_message')}"}

    system_prompt = """
You are a Summary Formatter.
Your ONLY job is to combine the provided ITINERARY and BUDGET into a single, polished markdown travel guide.

Rules:
1. Start with a # Title.
2. Include the ITINERARY exactly.
3. Include the BUDGET BREAKDOWN exactly.
4. Output ONLY the markdown content.
5. NO greetings, NO meta-commentary, NO questions.
""".strip()
    user_prompt = f"ITINERARY:\n{state['itinerary']}\n\nBUDGET BREAKDOWN:\n{state['budget_breakdown']}"
    try:
        final = invoke_llm(llm, system_prompt, user_prompt)
        # Ensure both parts are actually there. If not, fallback to manual joining.
        if "Day 1" not in final or "|" not in final:
             final = f"# Travel Guide: {state['destination']}\n\n## Itinerary\n\n{state['itinerary']}\n\n## Budget Breakdown\n\n{state['budget_breakdown']}"
        return {"final_output": final}
    except Exception:
        return {"final_output": f"# Travel Guide: {state['destination']}\n\n{state['itinerary']}\n\n{state['budget_breakdown']}"}


def build_graph(llm: Any) -> Any:
    workflow = StateGraph(TravelPlannerState)
    workflow.add_node("Analyzer", lambda s: analyzer_agent(s, llm))
    workflow.add_node("Planner", lambda s: itinerary_planner_agent(s, llm))
    workflow.add_node("Budget", lambda s: budget_estimator_agent(s, llm))
    workflow.add_node("Formatter", lambda s: summary_formatter_agent(s, llm))

    workflow.add_edge(START, "Analyzer")
    workflow.add_conditional_edges("Analyzer", lambda s: "Formatter" if s.get("errors") or not s.get("is_valid", True) else "Planner")
    workflow.add_edge("Planner", "Budget")
    workflow.add_edge("Budget", "Formatter")
    workflow.add_edge("Formatter", END)
    return workflow.compile()


def build_initial_state(user_request: str) -> TravelPlannerState:
    return {
        "original_request": user_request, "destination": "", "number_of_days": 0,
        "budget": "", "budget_amount": 0.0, "budget_currency": "USD",
        "requirements_summary": "", "itinerary": "", "budget_breakdown": "",
        "final_output": "", "errors": [], "is_valid": True, "validation_message": ""
    }


def print_banner() -> None:
    print("=" * 80)
    print("Intelligent Travel Planner - Multi-Agent System")
    print("Powered by LangChain + LangGraph + Ollama")
    print("=" * 80)


def main() -> None:
    """
    Console entry point.
    """
    print_banner()

    try:
        user_request = input(
            "\nEnter your travel request (example: Plan a 5-day trip to Japan with a budget of $2500):\n> "
        ).strip()
    except KeyboardInterrupt:
        print("\n\nInput cancelled by user.")
        return

    if not user_request:
        print("\nA non-empty travel request is required.")
        return

    try:
        # Detect available models and pick the best one
        available = get_available_models()
        model_name = "gemma3:12b" if "gemma3:12b" in available else available[0]
        print(f"\nUsing model: {model_name}")
        
        llm = create_llm(model_name=model_name)
        graph = build_graph(llm)
        initial_state = build_initial_state(user_request)
        
        print("🤖 AI agents are planning your trip (this may take a moment)...")
        final_state = graph.invoke(initial_state)
    except Exception as exc:
        print(f"\nUnexpected error: {exc}")
        return

    final_output = final_state.get("final_output", "").strip()
    if not final_output:
        print("\nThe system completed, but no final output was generated.")
        return

    print("\n" + "=" * 80)
    print("FINAL TRAVEL GUIDE")
    print("=" * 80 + "\n")
    print(final_output)

    if final_state.get("errors"):
        print("\n" + "=" * 80)
        print("SYSTEM LOGS")
        print("=" * 80)
        for error in final_state["errors"]:
            print(f"- {error}")


if __name__ == "__main__":
    main()
