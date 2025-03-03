from pydantic import BaseModel, Field
import os
import random
from IPython.display import Image, display
from dotenv import load_dotenv
from typing import List, Dict, Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, START, END

# Load environment variables
load_dotenv()

if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY environment variable not set.")
    print("Please set it in your .env file or export it in your terminal.")
    print("Example: export OPENAI_API_KEY=your-api-key")
    exit(1)

# Initialize LLM with higher temperature for creativity
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.8)

class State(BaseModel):
    topic: str = Field(default="The topic of the Joke")
    joke: str = Field(default="")
    improved_joke: str = Field(default="")
    final_joke: str = Field(default="")

# List of potential topics for random selection
TOPICS = [
    "productivity",
    "leadership",
    "networking",
    "work-life balance",
    "entrepreneurship",
    "innovation",
    "meetings",
    "remote work",
    "corporate culture",
    "personal branding",
    "mentorship",
    "career growth",
    "team building",
    "success mindset",
    "morning routines",
    "email management",
    "office politics",
    "performance reviews",
    "business travel",
    "professional development"
]

def get_random_topic() -> str:
    """Select a random topic from the predefined list."""
    return random.choice(TOPICS)

def create_joke(state: State) -> Dict[str, Any]:
    """Create an absurdist LinkedIn-style joke based on the topic."""
    system_prompt = """You are a comedy writer who specializes in absurdist, deadpan LinkedIn humor with unexpected twists.

    Your jokes have these exact characteristics:
    1. First-person narrative with a confident, matter-of-fact tone
    2. Short paragraphs (1-3 sentences each) that build on each other
    3. Start with a seemingly normal premise that gradually becomes absurd
    4. Include unexpected twists or non-sequiturs
    5. Often involve false equivalencies or deliberate misunderstandings
    6. End with a punchline that's either deadpan or escalates the absurdity

    IMPORTANT STYLE GUIDELINES:
    - Use simple, everyday language (avoid big words or jargon)
    - Write short, clear sentences
    - Use words that a 12-year-old would understand
    - Keep the structure of short paragraphs with line breaks
    - Maintain the absurdist humor but with simpler vocabulary
    - Still sound confident and matter-of-fact"""

    user_prompt = f"""Write a joke about {state.topic} in the exact style of these examples, but with simpler language:

    Example 1:
    I put Widowed on every form I fill out.

    Applications, surveys, prize draws.

    All my ex-spouses are alive.

    Why do I do it?

    Because Widowed sounds more powerful than Divorced.

    When you think of a divorced man you think of a loser.

    When you think of a widowed man, you think this man is vulnerable and the sex is probably really good.

    I don't make the rules.

    But I do break them.

    By lying about my alive ex-wives.

    Since doing this, I've won so many more prize draws.

    Example 2:
    How do you stop your workers clocking off early?

    Some say it's by being strict and dishing out punishments.

    Some say it's by fostering a good work environment so they don't want to.

    Both are wrong. It's snakes.

    I put 10-15 snakes out in the elevator corridor.

    They are varying degrees of venomous. As the boss, only I have the antidote.

    When employees hit their KPIs, snakes are gone.

    Human beings are simple. They aren't most motivated by fame, money or power.

    Instead, they just want one thing: being around zero snakes.

    Now write a joke about {state.topic} with the same structure and style, but using simpler words and language. Use short paragraphs with line breaks between them. Make it gradually escalate in absurdity while maintaining a deadpan, confident tone."""

    msg = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    return {"joke": msg.content}


def improve_joke(state: State) -> Dict[str, Any]:
    """Improve the joke by adding more absurdist elements."""
    system_prompt = """You are improving an absurdist LinkedIn-style joke. Your job is to:
    1. Make it more absurd while keeping the deadpan delivery
    2. Ensure it has the short paragraph structure with line breaks
    3. Add 1-2 unexpected twists
    4. Make sure it has a strong ending
    5. Use simple, everyday language that's easy to understand
    6. Avoid big words, jargon, or complex vocabulary

    Keep the same topic and premise, but enhance the absurdity and impact while using simpler language."""

    user_prompt = f"""Improve this joke by making it more absurd while maintaining its deadpan delivery:

    {state.joke}

    Make sure it follows the structure of short paragraphs with line breaks between them, has a strong ending, and uses simple, everyday language that anyone can understand."""

    msg = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    return {"improved_joke": msg.content}


def create_final_joke(state: State) -> Dict[str, Any]:
    """Finalize the joke with a stronger punchline."""
    system_prompt = """You are finalizing an absurdist LinkedIn-style joke. Your job is to:
    1. Ensure it has the perfect structure of short paragraphs with line breaks
    2. Make sure it has a strong ending that either escalates the absurdity or delivers a deadpan conclusion
    3. Maintain the confident, matter-of-fact tone throughout
    4. Use simple, everyday language that's easy to understand
    5. Replace any complex words with simpler alternatives
    6. Keep sentences short and clear

    The final joke should feel complete and have the exact style of the example jokes but with simpler vocabulary."""

    user_prompt = f"""Finalize this joke, ensuring it has the perfect structure and impact:

    {state.improved_joke}

    Make sure it has short paragraphs with line breaks between them, ends with a strong punchline or absurd conclusion, and uses simple language that anyone can understand."""

    msg = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=user_prompt)
    ])
    return {"final_joke": msg.content}

def build_joke_graph():
    """Build the joke graph."""
    graph = StateGraph(State)

    graph.add_node("create_joke", create_joke)
    graph.add_node("improve_joke", improve_joke)
    graph.add_node("create_final_joke", create_final_joke)

    graph.add_edge(START, "create_joke")
    graph.add_edge("create_joke", "improve_joke")
    graph.add_edge("improve_joke", "create_final_joke")
    graph.add_edge("create_final_joke", END)

    return graph.compile()

if __name__ == "__main__":
    # Get a random topic
    topic = get_random_topic()
    print(f"Generating a joke about: {topic}\n")

    # Build and run the joke graph
    chain = build_joke_graph()
    state = chain.invoke({"topic": topic})

    # Print the final joke
    print(state["final_joke"])
