import operator
from pydantic import BaseModel, Field
from typing import Annotated, List
from typing_extensions import TypedDict

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, get_buffer_string
from langchain_openai import ChatOpenAI

from langgraph.constants import Send
from langgraph.graph import END, MessagesState, START, StateGraph

# Initialize the LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0)

# Define the schema
class Analyst(BaseModel):
    name: str = Field(description="Name of the analyst.")
    role: str = Field(description="Role of the analyst in the context of the topic.")
    description: str = Field(description="Description of the analyst focus, concerns, and motives.")

    @property
    def persona(self) -> str:
        return f"Name: {self.name}\nRole: {self.role}\nDescription: {self.description}\n"

class Perspectives(BaseModel):
    analysts: List[Analyst] = Field(
        description="Comprehensive list of analysts with their roles and affiliations.",
    )

class GenerateAnalystsState(TypedDict):
    topic: str  # Research topic
    max_analysts: int  # Number of analysts
    analysts: List[Analyst]  # Analyst asking questions

class InterviewState(MessagesState):
    max_num_turns: int  # Number turns of conversation
    analyst: Analyst  # Analyst asking questions
    interview: str  # Interview transcript

class ResearchGraphState(TypedDict):
    topic: str  # Research topic
    max_analysts: int  # Number of analysts
    analysts: List[Analyst]  # Analyst asking questions
    final_report: str  # Final report

# Define the nodes
analyst_instructions = """You are tasked with creating a set of AI analyst personas. Follow these instructions carefully:

1. First, review the research topic:
{topic}

2. Determine the most interesting themes based upon the topic.

3. Pick the top {max_analysts} themes.

4. Assign one analyst to each theme."""

def create_analysts(state: GenerateAnalystsState):
    """ Create analysts """

    topic = state['topic']
    max_analysts = state['max_analysts']

    # Enforce structured output
    structured_llm = llm.with_structured_output(Perspectives)

    # System message
    system_message = analyst_instructions.format(
        topic=topic,
        max_analysts=max_analysts
    )

    # Generate analysts
    analysts = structured_llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content="Generate the set of analysts.")
    ])

    # Write the list of analysts to state
    return {"analysts": analysts.analysts}

# Generate analyst question
question_instructions = """You are an analyst tasked with interviewing an expert to learn about a specific topic.

Your goal is to gather interesting and specific insights related to your topic.

Here is your topic of focus and set of goals: {goals}

Begin by introducing yourself using a name that fits your persona, and then ask your question.

Continue to ask questions to drill down and refine your understanding of the topic.

When you are satisfied with your understanding, complete the interview with: "Thank you so much for your help!"

Remember to stay in character throughout your response, reflecting the persona and goals provided to you."""

def generate_question(state: InterviewState):
    """ Node to generate a question """

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]

    # Generate question
    system_message = question_instructions.format(goals=analyst.persona)
    question = llm.invoke([SystemMessage(content=system_message)] + messages)

    # Write messages to state
    return {"messages": [question]}

# Generate expert answer
answer_instructions = """You are an expert being interviewed by an analyst.

Here is the analyst's area of focus: {goals}.

Your goal is to answer questions posed by the interviewer.

When answering questions, follow these guidelines:

1. Provide detailed, informative responses based on factual information.

2. If you don't know something, acknowledge it rather than making up information.

3. Try to be helpful and address the specific questions being asked."""

def generate_answer(state: InterviewState):
    """ Node to answer a question """

    # Get state
    analyst = state["analyst"]
    messages = state["messages"]

    # Answer question
    system_message = answer_instructions.format(goals=analyst.persona)
    answer = llm.invoke([SystemMessage(content=system_message)] + messages)

    # Name the message as coming from the expert
    answer.name = "expert"

    # Append it to state
    return {"messages": [answer]}

def save_interview(state: InterviewState):
    """ Save interviews """

    # Get messages
    messages = state["messages"]

    # Convert interview to a string
    interview = get_buffer_string(messages)

    # Save to interviews key
    return {"interview": interview}

def route_messages(state: InterviewState, name: str = "expert"):
    """ Route between question and answer """

    # Get messages
    messages = state["messages"]
    max_num_turns = state.get('max_num_turns', 2)

    # Check the number of expert answers
    num_responses = len(
        [m for m in messages if isinstance(m, AIMessage) and m.name == name]
    )

    # End if expert has answered more than the max turns
    if num_responses >= max_num_turns:
        return 'save_interview'

    # This router is run after each question - answer pair
    # Get the last question asked to check if it signals the end of discussion
    last_question = messages[-2]

    if "Thank you so much for your help" in last_question.content:
        return 'save_interview'
    return "ask_question"

# Write a report based on the interviews
report_writer_instructions = """You are a technical writer creating a report on this overall topic:

{topic}

You have conducted an interview with an expert on this topic. Your task is to:

1. Analyze the interview transcript provided.
2. Extract the key insights and information.
3. Create a well-structured report that summarizes these findings.

To format your report:

1. Use markdown formatting.
2. Start with a compelling title using # header.
3. Include an ## Introduction section.
4. Include a ## Key Findings section.
5. Include a ## Conclusion section.
6. Keep your report concise but informative, around 500 words.

Here is the interview transcript to build your report from:

{interview}"""

def write_report(state: InterviewState):
    """ Node to write a report based on the interview """

    # Get interview and topic
    interview = state["interview"]
    analyst = state["analyst"]

    # Generate report
    system_message = report_writer_instructions.format(
        topic=analyst.description,
        interview=interview
    )

    report = llm.invoke([
        SystemMessage(content=system_message),
        HumanMessage(content="Write a report based on this interview.")
    ])

    return {"final_report": report.content}

# Build the interview graph
def build_interview_graph():
    interview_builder = StateGraph(InterviewState)

    interview_builder.add_node("ask_question", generate_question)
    interview_builder.add_node("answer_question", generate_answer)
    interview_builder.add_node("save_interview", save_interview)
    interview_builder.add_node("write_report", write_report)

    # Flow
    interview_builder.add_edge(START, "ask_question")
    interview_builder.add_edge("ask_question", "answer_question")
    interview_builder.add_conditional_edges(
        "answer_question",
        route_messages,
        ['ask_question', 'save_interview']
    )
    interview_builder.add_edge("save_interview", "write_report")
    interview_builder.add_edge("write_report", END)

    return interview_builder.compile()

# Function to run the interview graph
def run_interview(topic, analyst, max_turns=2):
    """Run an interview with a single analyst"""
    interview_graph = build_interview_graph()

    # Initial state
    initial_state = {
        "analyst": analyst,
        "max_num_turns": max_turns,
        "messages": [
            HumanMessage(content=f"I'd like to discuss {topic} with you.")
        ]
    }

    # Run the graph
    result = interview_graph.invoke(initial_state)
    return result.get("final_report", "No report generated")

# Function to run the full research process
def run_research(topic, max_analysts=2, max_turns=2):
    """Run the full research process with multiple analysts"""
    # Step 1: Generate analysts
    initial_state = {
        "topic": topic,
        "max_analysts": max_analysts
    }

    analysts_result = create_analysts(initial_state)
    analysts = analysts_result.get("analysts", [])

    if not analysts:
        return "Failed to generate analysts"

    # Step 2: Run interviews with each analyst
    reports = []
    for analyst in analysts:
        report = run_interview(topic, analyst, max_turns)
        reports.append(report)

    # Step 3: Combine reports
    combined_report = "\n\n---\n\n".join(reports)

    return combined_report

# Example usage
if __name__ == "__main__":
    topic = "The impact of artificial intelligence on healthcare"
    report = run_research(topic, max_analysts=2, max_turns=2)
    print(report)