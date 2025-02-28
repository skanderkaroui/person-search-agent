import os
from dotenv import load_dotenv
from typing import List, Dict, Any
from pydantic import BaseModel, Field

from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# Load environment variables
load_dotenv()

# Check if OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY environment variable not set.")
    print("Please set it in your .env file or export it in your terminal.")
    print("Example: export OPENAI_API_KEY=your-api-key")
    exit(1)

# Initialize the LLM
llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# Define schema
class Analyst(BaseModel):
    name: str = Field(description="Name of the analyst.")
    role: str = Field(description="Role of the analyst.")
    description: str = Field(description="Description of the analyst's focus.")

class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(description="List of analysts.")

# Define state
class AnalystState(dict):
    """State for the analyst graph."""
    topic: str
    max_analysts: int
    analysts: List[Analyst] = []
    messages: List = []
    current_analyst_index: int = 0
    reports: List[str] = []
    final_report: str = ""

# Define nodes
def create_analysts(state: AnalystState) -> Dict[str, Any]:
    """Create analyst personas based on the topic."""
    topic = state["topic"]
    max_analysts = state["max_analysts"]

    # System prompt
    system_prompt = f"""
    You are tasked with creating {max_analysts} analyst personas for the topic: {topic}.

    For each analyst:
    1. Assign a name
    2. Define their role
    3. Describe their focus and expertise

    Each analyst should focus on a different aspect of the topic.
    """

    # Get structured output
    structured_llm = llm.with_structured_output(Perspectives)
    result = structured_llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Create {max_analysts} analyst personas for the topic: {topic}")
    ])

    return {"analysts": result.analysts}

def interview(state: AnalystState) -> Dict[str, Any]:
    """Conduct an interview with the current analyst."""
    analysts = state["analysts"]
    current_index = state["current_analyst_index"]
    topic = state["topic"]

    # Get current analyst
    if current_index >= len(analysts):
        return {"reports": state["reports"]}

    analyst = analysts[current_index]

    # System prompt for the analyst
    analyst_prompt = f"""
    You are {analyst.name}, {analyst.role}.
    Your focus is: {analyst.description}

    You are interviewing an expert about: {topic}

    Ask 2 insightful questions about this topic from your perspective.
    """

    # Get analyst questions
    analyst_message = llm.invoke([
        SystemMessage(content=analyst_prompt),
        HumanMessage(content=f"I'd like to discuss {topic} with you.")
    ])

    # System prompt for the expert
    expert_prompt = f"""
    You are an expert on {topic}.

    You are being interviewed by {analyst.name}, who is a {analyst.role}.
    Their focus is: {analyst.description}

    Provide detailed, informative responses to their questions.
    """

    # Get expert answers
    expert_message = llm.invoke([
        SystemMessage(content=expert_prompt),
        HumanMessage(content=analyst_message.content)
    ])

    # Generate a report
    report_prompt = f"""
    You are a technical writer creating a report based on an interview about {topic}.

    The interview was conducted by {analyst.name}, {analyst.role}, whose focus is: {analyst.description}

    The interview transcript is:

    Analyst: {analyst_message.content}

    Expert: {expert_message.content}

    Create a concise report (300-500 words) with:
    1. A title
    2. Key insights from the interview
    3. Conclusions

    Use markdown formatting.
    """

    report = llm.invoke([
        SystemMessage(content=report_prompt),
        HumanMessage(content="Write a report based on this interview.")
    ])

    # Add report to list
    reports = state.get("reports", [])
    reports.append(report.content)

    # Move to next analyst
    return {
        "reports": reports,
        "current_analyst_index": current_index + 1
    }

def compile_final_report(state: AnalystState) -> Dict[str, Any]:
    """Compile all analyst reports into a final report."""
    topic = state["topic"]
    reports = state["reports"]

    # Join all reports
    all_reports = "\n\n---\n\n".join(reports)

    # System prompt
    system_prompt = f"""
    You are a technical writer creating a comprehensive report on {topic}.

    You have received individual reports from different analysts, each focusing on a different aspect of the topic.

    Your task is to compile these reports into a cohesive final report that:
    1. Has an engaging title
    2. Includes an introduction that sets the context
    3. Synthesizes the key insights from all analysts
    4. Provides a conclusion with overall implications

    Use markdown formatting.

    Here are the individual reports:

    {all_reports}
    """

    # Generate final report
    final_report = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content="Compile a final comprehensive report.")
    ])

    return {"final_report": final_report.content}

# Define the graph
def build_analyst_graph():
    """Build the analyst graph."""
    # Create graph
    graph = StateGraph(AnalystState)

    # Add nodes
    graph.add_node("create_analysts", create_analysts)
    graph.add_node("interview", interview)
    graph.add_node("compile_final_report", compile_final_report)

    # Add edges
    graph.add_edge(START, "create_analysts")
    graph.add_edge("create_analysts", "interview")

    # Conditional edge: if we've processed all analysts, move to final report
    def should_continue(state):
        if state["current_analyst_index"] >= len(state["analysts"]):
            return "compile_final_report"
        return "interview"

    graph.add_conditional_edges("interview", should_continue, ["interview", "compile_final_report"])
    graph.add_edge("compile_final_report", END)

    # Compile graph
    return graph.compile()

# Run the graph
def run_analyst_research(topic: str, max_analysts: int = 2):
    """Run the analyst research process."""
    # Build graph
    graph = build_analyst_graph()

    # Initial state
    initial_state = {
        "topic": topic,
        "max_analysts": max_analysts,
        "analysts": [],
        "messages": [],
        "current_analyst_index": 0,
        "reports": [],
        "final_report": ""
    }

    # Run graph
    result = graph.invoke(initial_state)

    # Return final report
    return result["final_report"]

if __name__ == "__main__":
    print("Running analyst research...")
    topic = "The impact of artificial intelligence on healthcare"
    final_report = run_analyst_research(topic, max_analysts=2)
    print("\nFinal Report:\n")
    print(final_report)
    print("\nResearch completed successfully!")