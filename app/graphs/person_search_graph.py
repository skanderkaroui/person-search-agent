from typing import Dict, List, Optional, TypedDict, Any, Annotated
import operator
from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END, START
from langgraph.constants import Send
from app.utils.sources import SOURCE_MAP
from app.core.config import settings
from app.models.search import SourceInfo

# Define state schema
class SearchState(TypedDict):
    """State for the person search graph."""
    person: str  # Name of the person to search for
    sources: Optional[List[str]]  # List of sources to search
    is_deep_dive: bool  # Whether this is a deep dive search
    topic: Optional[str]  # Topic for deep dive
    retrieved_data: Dict[str, List[SourceInfo]]  # Data retrieved from sources
    summary: str  # AI-generated summary
    error: Optional[str]  # Error message, if any

# Define models for structured output
class SourceSelection(BaseModel):
    """Model for source selection."""
    sources: List[str] = Field(..., description="List of sources to search")
    reasoning: str = Field(..., description="Reasoning behind the source selection")

class SearchQuery(BaseModel):
    """Model for search query generation."""
    query: str = Field(..., description="Search query to use")

class PersonSearchGraph:
    """LangGraph implementation for person search."""

    def __init__(self):
        """Initialize the person search graph."""
        # Initialize LLM
        self.llm = ChatOpenAI(
            api_key=settings.OPENAI_API_KEY,
            model="gpt-4",
            temperature=0
        )

        # Build the graph
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        """Build the LangGraph for person search."""
        # Create the graph
        graph = StateGraph(SearchState)

        # Add nodes
        graph.add_node("select_sources", self._select_sources)
        graph.add_node("generate_query", self._generate_query)
        graph.add_node("retrieve_data", self._retrieve_data)
        graph.add_node("generate_summary", self._generate_summary)

        # Add edges
        graph.add_edge(START, "select_sources")
        graph.add_edge("select_sources", "generate_query")
        graph.add_edge("generate_query", "retrieve_data")
        graph.add_edge("retrieve_data", "generate_summary")
        graph.add_edge("generate_summary", END)

        # Compile the graph
        return graph.compile()

    async def execute(
        self,
        person: str,
        sources: Optional[List[str]] = None,
        is_deep_dive: bool = False,
        topic: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Execute the person search graph.

        Args:
            person: Name of the person to search for.
            sources: Optional list of sources to search.
            is_deep_dive: Whether this is a deep dive search.
            topic: Topic for deep dive.

        Returns:
            Dict[str, Any]: Result of the search.
        """
        # Initialize state
        state = {
            "person": person,
            "sources": sources,
            "is_deep_dive": is_deep_dive,
            "topic": topic,
            "retrieved_data": {},
            "summary": "",
            "error": None
        }

        # Execute the graph
        result = await self.graph.ainvoke(state)

        return result

    async def _select_sources(self, state: SearchState) -> SearchState:
        """
        Select the sources to search based on the person and topic.

        Args:
            state: Current state.

        Returns:
            SearchState: Updated state with selected sources.
        """
        # If sources are already provided, use them
        if state["sources"]:
            return state

        # Use LLM to select sources
        structured_llm = self.llm.with_structured_output(SourceSelection)

        # Create prompt
        if state["is_deep_dive"]:
            prompt = f"""
            You are an AI assistant tasked with selecting the best sources to search for information about {state["person"]}
            regarding the topic: {state["topic"]}.

            Available sources:
            - Twitter: Good for recent activities and public statements
            - Google: Good for news articles and diverse sources

            Select the most appropriate sources for this search.
            """
        else:
            prompt = f"""
            You are an AI assistant tasked with selecting the best sources to search for information about {state["person"]}.

            Available sources:
            - Twitter: Good for recent activities and public statements
            - Google: Good for news articles and diverse sources

            Select the most appropriate sources for this search.
            """

        # Get source selection
        source_selection = await structured_llm.ainvoke([SystemMessage(content=prompt)])

        # Update state
        state["sources"] = source_selection.sources

        return state

    async def _generate_query(self, state: SearchState) -> SearchState:
        """
        Generate search queries for each source.

        Args:
            state: Current state.

        Returns:
            SearchState: Updated state with search queries.
        """
        # Use LLM to generate search query
        structured_llm = self.llm.with_structured_output(SearchQuery)

        # Create prompt
        if state["is_deep_dive"]:
            prompt = f"""
            Generate a search query to find information about {state["person"]} regarding the topic: {state["topic"]}.
            The query should be specific and focused on retrieving relevant information.
            """
        else:
            prompt = f"""
            Generate a search query to find general information about {state["person"]}.
            The query should be comprehensive to retrieve a broad overview of the person.
            """

        # Get search query
        search_query = await structured_llm.ainvoke([SystemMessage(content=prompt)])

        # Store query in state
        state["query"] = search_query.query

        return state

    async def _retrieve_data(self, state: SearchState) -> SearchState:
        """
        Retrieve data from selected sources.

        Args:
            state: Current state.

        Returns:
            SearchState: Updated state with retrieved data.
        """
        # Initialize retrieved data
        retrieved_data = {}

        # Get query
        query = state["query"]

        # Search each source
        for source_name in state["sources"]:
            if source_name in SOURCE_MAP:
                source_class = SOURCE_MAP[source_name]
                source_results = await source_class.search(query)
                retrieved_data[source_name] = source_results

        # Update state
        state["retrieved_data"] = retrieved_data

        return state

    async def _generate_summary(self, state: SearchState) -> SearchState:
        """
        Generate a summary of the retrieved information.

        Args:
            state: Current state.

        Returns:
            SearchState: Updated state with summary.
        """
        # Format retrieved data for the LLM
        formatted_data = ""
        for source_name, source_results in state["retrieved_data"].items():
            formatted_data += f"\n\n## {source_name} Results:\n"
            for i, result in enumerate(source_results):
                formatted_data += f"\n{i+1}. {result.title or 'No title'}\n"
                formatted_data += f"   URL: {result.url}\n"
                formatted_data += f"   Snippet: {result.snippet or 'No snippet'}\n"

        # Create prompt
        if state["is_deep_dive"]:
            prompt = f"""
            You are an AI assistant tasked with generating a comprehensive summary about {state["person"]}
            regarding the topic: {state["topic"]}.

            Use the following information retrieved from various sources:

            {formatted_data}

            Generate a detailed summary that focuses specifically on the requested topic.
            Include relevant facts, achievements, and context. Be objective and informative.
            """
        else:
            prompt = f"""
            You are an AI assistant tasked with generating a comprehensive summary about {state["person"]}.

            Use the following information retrieved from various sources:

            {formatted_data}

            Generate a detailed summary that covers the person's background, career, achievements, and notable facts.
            Be objective and informative. Structure the summary in a logical way.
            """

        # Generate summary
        summary_message = await self.llm.ainvoke([SystemMessage(content=prompt)])

        # Update state
        state["summary"] = summary_message.content

        return state