import os
from dotenv import load_dotenv
from typing import List

from langchain_core.messages import HumanMessage, SystemMessage
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

# Define a simple state type
class SimpleState(dict):
    """A simple state for our graph."""
    messages: List
    response: str = ""

# Define nodes
def ask_question(state: SimpleState):
    """Node to ask a question to the LLM."""
    messages = state["messages"]

    # Add a system message
    system_message = SystemMessage(content="You are a helpful assistant.")

    # Get response from LLM
    response = llm.invoke([system_message] + messages)

    # Return updated state
    return {"response": response.content}

# Build the graph
def build_graph():
    """Build a simple graph with one node."""
    # Create a new graph
    graph = StateGraph(SimpleState)

    # Add nodes
    graph.add_node("ask_question", ask_question)

    # Add edges
    graph.add_edge(START, "ask_question")
    graph.add_edge("ask_question", END)

    # Compile the graph
    return graph.compile()

# Run the graph
def run_simple_test(question: str):
    """Run a simple test with the graph."""
    # Build the graph
    graph = build_graph()

    # Initial state
    initial_state = {
        "messages": [HumanMessage(content=question)],
        "response": ""
    }

    # Run the graph
    result = graph.invoke(initial_state)

    # Return the response
    return result["response"]

if __name__ == "__main__":
    print("Testing LangGraph installation...")
    question = "What is LangGraph and how does it relate to LangChain?"
    response = run_simple_test(question)
    print("\nQuestion:", question)
    print("\nResponse:", response)
    print("\nTest completed successfully!")