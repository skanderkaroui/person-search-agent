from pydantic import BaseModel, Field
import os
import json
from datetime import datetime
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv

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

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.7)

# Define data models
class UserProfile(BaseModel):
    name: str = Field(description="User's full name")
    company_name: str = Field(description="User's company name")
    work_function: str = Field(description="User's work function (e.g., Engineering, Marketing)")
    persona_description: str = Field(description="Detailed description of the user's persona")
    communication_style: str = Field(description="User's preferred communication style")

class Message(BaseModel):
    role: str = Field(description="Role of the message sender (human or ai)")
    content: str = Field(description="Content of the message")
    timestamp: str = Field(description="Timestamp when the message was sent")

class Memory(BaseModel):
    key: str = Field(description="Key identifier for this memory")
    content: str = Field(description="Content of the memory")
    importance: int = Field(description="Importance score from 1-10")
    last_accessed: str = Field(description="When this memory was last accessed")
    created_at: str = Field(description="When this memory was created")

class ChatbotState(dict):
    """State for the conversational chatbot."""
    user_profile: UserProfile = None
    messages: List[Message] = []
    short_term_memory: List[Message] = []  # Recent conversation context
    long_term_memory: List[Memory] = []    # Important facts and context
    current_context: str = ""              # Current conversation context
    system_prompt: str = ""                # System prompt for the agent
    response: str = ""                     # Generated response

# Define nodes for the chatbot graph
def process_input(state: ChatbotState) -> Dict[str, Any]:
    """Process the user message and update the state."""
    messages = state["messages"]

    # If there are no messages, this is the first interaction
    if not messages:
        return state

    # Get the latest message
    latest_message = messages[-1]

    # Update short-term memory (keep last 10 messages)
    short_term = state.get("short_term_memory", [])
    short_term.append(latest_message)
    if len(short_term) > 10:
        short_term = short_term[-10:]

    return {"short_term_memory": short_term}

def update_long_term_memory(state: ChatbotState) -> Dict[str, Any]:
    """Extract important information and update long-term memory."""
    messages = state["messages"]
    long_term_memory = state.get("long_term_memory", [])
    user_profile = state.get("user_profile")

    if len(messages) < 1 or not user_profile:
        return {"long_term_memory": long_term_memory}

    # Get the conversation history as context
    conversation_history = "\n".join([f"{msg.role}: {msg.content}" for msg in messages[-5:]])

    # System prompt for memory extraction - focus on user preferences
    system_prompt = """
    You are a memory extraction system for a conversational chatbot. Your job is to identify important information from the conversation
    that should be remembered for future reference. Focus ONLY on:

    1. User preferences for LinkedIn response style
    2. User communication preferences
    3. Important context about the user that would help personalize future responses
    4. Specific feedback on what works well or needs improvement

    For each important piece of information, extract:
    1. A key (short descriptor)
    2. The content (the actual information)
    3. An importance score (1-10)

    Only extract truly important information that will help personalize future responses.
    If nothing important is found, return an empty list.
    Be very selective - only store information that will be useful for future interactions.
    """

    # Define a Pydantic model for memory extraction
    class MemoryItem(BaseModel):
        key: str = Field(description="Short descriptor for this memory")
        content: str = Field(description="The actual information to remember")
        importance: int = Field(description="Importance score from 1-10")

    class MemoryExtraction(BaseModel):
        memories: List[MemoryItem] = Field(description="List of extracted memories")

    # Use structured output for memory extraction
    structured_llm = llm.with_structured_output(MemoryExtraction)

    try:
        memory_extraction_result = structured_llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content=f"Extract important information from this conversation:\n{conversation_history}")
        ])

        # Convert extracted memories to Memory objects and add to long-term memory
        current_time = datetime.now().isoformat()

        for mem in memory_extraction_result.memories:
            # Check if this memory already exists (by key)
            existing_memory = next((m for m in long_term_memory if m.key == mem.key), None)

            if existing_memory:
                # Update existing memory
                existing_memory.content = mem.content
                existing_memory.importance = mem.importance
                existing_memory.last_accessed = current_time
            else:
                # Create new memory
                new_memory = Memory(
                    key=mem.key,
                    content=mem.content,
                    importance=mem.importance,
                    last_accessed=current_time,
                    created_at=current_time
                )
                long_term_memory.append(new_memory)

        # Sort memories by importance
        long_term_memory.sort(key=lambda x: x.importance, reverse=True)

        # Limit the number of memories to prevent unnecessary storage
        if len(long_term_memory) > 20:
            long_term_memory = long_term_memory[:20]

        # Save to file
        save_memories_to_file(user_profile.name, long_term_memory)

    except Exception as e:
        print(f"Error extracting memories: {e}")

    return {"long_term_memory": long_term_memory}

def prepare_context(state: ChatbotState) -> Dict[str, Any]:
    """Prepare the context for the chatbot's response."""
    short_term = state.get("short_term_memory", [])
    long_term = state.get("long_term_memory", [])
    user_profile = state.get("user_profile")

    if not user_profile:
        return {"current_context": ""}

    # Prepare short-term context (recent messages)
    short_term_context = "\n".join([f"{msg.role}: {msg.content}" for msg in short_term])

    # Prepare long-term context (important memories)
    # Sort by importance and recency
    sorted_memories = sorted(
        long_term,
        key=lambda x: (x.importance, x.last_accessed),
        reverse=True
    )

    # Take top 5 most important/recent memories
    top_memories = sorted_memories[:5]
    long_term_context = "\n".join([f"- {mem.key}: {mem.content}" for mem in top_memories])

    # Combine contexts
    current_context = f"""
    USER PROFILE:
    Name: {user_profile.name}
    Company: {user_profile.company_name}
    Work Function: {user_profile.work_function}
    Persona: {user_profile.persona_description}
    Communication Style: {user_profile.communication_style}

    RECENT CONVERSATION:
    {short_term_context}

    IMPORTANT CONTEXT:
    {long_term_context}
    """

    return {"current_context": current_context}

def generate_response(state: ChatbotState) -> Dict[str, Any]:
    """Generate a response based on the user profile and context."""
    current_context = state["current_context"]
    user_profile = state.get("user_profile")

    if not user_profile:
        return {"response": "Unable to generate response: missing user profile."}

    # System prompt for response generation
    system_prompt = f"""
    You are a conversational chatbot for {user_profile.name}, who works at {user_profile.company_name} in {user_profile.work_function}.

    Your task is to engage in a natural, helpful conversation that aligns with {user_profile.name}'s persona and communication style.

    The response should:
    1. Be authentic and sound like it was written by {user_profile.name}
    2. Reflect their professional persona: {user_profile.persona_description}
    3. Use their preferred communication style: {user_profile.communication_style}
    4. Be contextually relevant to the conversation
    5. Be engaging and include follow-up questions when appropriate
    6. Avoid generic platitudes and provide specific value

    DO NOT mention that you are an AI or that this response was generated. Write as if you are {user_profile.name}.
    """

    # Generate response
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=f"Please craft a response based on the provided context:\n\n{current_context}")
    ])

    # Create a new message
    current_time = datetime.now().isoformat()
    ai_message = Message(
        role="ai",
        content=response.content,
        timestamp=current_time
    )

    # Add to messages
    updated_messages = state.get("messages", []).copy()
    updated_messages.append(ai_message)

    return {
        "messages": updated_messages,
        "response": response.content
    }

# Helper function to save memories to a file
def save_memories_to_file(user_name: str, memories: List[Memory]):
    """Save memories to a file."""
    filename = f"{user_name.lower().replace(' ', '_')}_memories.json"

    try:
        # Convert memories to dict for JSON serialization
        memories_dict = [memory.model_dump() for memory in memories]

        with open(filename, 'w') as f:
            json.dump(memories_dict, f, indent=2)

        print(f"Memories saved to {filename}")
    except Exception as e:
        print(f"Error saving memories to file: {e}")

# Helper function to load memories from a file
def load_memories_from_file(user_name: str) -> List[Memory]:
    """Load memories from a file."""
    filename = f"{user_name.lower().replace(' ', '_')}_memories.json"
    memories = []

    try:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                memories_dict = json.load(f)

            # Convert dict back to Memory objects
            memories = [Memory(**memory_data) for memory_data in memories_dict]
            print(f"Loaded {len(memories)} memories from {filename}")
        else:
            print(f"No existing memories file found at {filename}")
    except Exception as e:
        print(f"Error loading memories from file: {e}")

    return memories

# Define the chatbot graph
def build_chatbot_graph():
    """Build the conversational chatbot graph."""
    # Create graph
    graph = StateGraph(ChatbotState)

    # Add nodes
    graph.add_node("process_input", process_input)
    graph.add_node("update_long_term_memory", update_long_term_memory)
    graph.add_node("prepare_context", prepare_context)
    graph.add_node("generate_response", generate_response)

    # Add edges
    graph.add_edge(START, "process_input")
    graph.add_edge("process_input", "update_long_term_memory")
    graph.add_edge("update_long_term_memory", "prepare_context")
    graph.add_edge("prepare_context", "generate_response")
    graph.add_edge("generate_response", END)

    # Compile graph
    return graph.compile()

class ConversationalChatbot:
    def __init__(self, user_profile: Optional[UserProfile] = None):
        """Initialize the conversational chatbot."""
        self.graph = build_chatbot_graph()
        self.state = {
            "user_profile": user_profile,
            "messages": [],
            "short_term_memory": [],
            "long_term_memory": [],
            "current_context": "",
            "response": ""
        }

        # Conversation state flags
        self.is_conversation_active = False
        self.awaiting_feedback = False
        self.last_linkedin_response = ""

        # Load existing memories if user profile is provided
        if user_profile:
            self.state["long_term_memory"] = load_memories_from_file(user_profile.name)

    def set_user_profile(self, user_profile: UserProfile):
        """Set or update the user profile."""
        self.state["user_profile"] = user_profile
        # Load existing memories for this user
        self.state["long_term_memory"] = load_memories_from_file(user_profile.name)

    def generate_linkedin_response(self, post_content: str, post_author: str = "LinkedIn User") -> str:
        """Generate a response to a LinkedIn post and start a conversation about it."""
        if not self.state["user_profile"]:
            return "Error: User profile not set. Please set a user profile before generating responses."

        # Create a new message for the LinkedIn post
        current_time = datetime.now().isoformat()
        post_message = Message(
            role="human",
            content=f"LinkedIn Post from {post_author}: {post_content}",
            timestamp=current_time
        )

        # Add to messages
        messages = self.state.get("messages", [])
        messages.append(post_message)
        self.state["messages"] = messages

        # Run the graph to generate a response
        result = self.graph.invoke(self.state)

        # Update state
        self.state = result

        # Get the generated response
        response = result.get("response", "No response generated.")

        # Check if user prefers emoji in responses
        long_term_memory = self.state.get("long_term_memory", [])
        emoji_preference = next((mem for mem in long_term_memory if "emoji" in mem.key.lower()), None)

        # Add monkey emoji if user prefers it
        if emoji_preference and "monkey" in emoji_preference.content.lower():
            response = f"{response} 🐒"

        # Store the LinkedIn response for potential feedback
        self.last_linkedin_response = response
        self.is_conversation_active = True
        self.awaiting_feedback = True

        # Add a system message prompting for feedback (won't be shown to user)
        feedback_prompt = Message(
            role="system",
            content="What do you think of this response? Would you like me to adjust it in any way?",
            timestamp=datetime.now().isoformat()
        )

        messages = self.state.get("messages", [])
        messages.append(feedback_prompt)
        self.state["messages"] = messages

        return response

    def chat(self, user_message: str) -> str:
        """Process a user message and generate a response."""
        if not self.state["user_profile"]:
            return "Error: User profile not set. Please set a user profile before chatting."

        # Check if this is a conversation ending message
        if self._is_conversation_ending(user_message):
            self.is_conversation_active = False
            return "You're welcome!"

        # Create a new message for the user input
        current_time = datetime.now().isoformat()
        user_message_obj = Message(
            role="human",
            content=user_message,
            timestamp=current_time
        )

        # Add to messages
        messages = self.state.get("messages", [])
        messages.append(user_message_obj)
        self.state["messages"] = messages

        # If awaiting feedback on a LinkedIn response, process it as feedback
        if self.awaiting_feedback and self.last_linkedin_response:
            feedback_response = self._process_feedback(user_message, self.last_linkedin_response)
            self.awaiting_feedback = False
            return feedback_response

        # Run the graph for normal conversation
        result = self.graph.invoke(self.state)

        # Update state
        self.state = result

        # Return the generated response
        return result.get("response", "No response generated.")

    def _is_conversation_ending(self, message: str) -> bool:
        """Check if the message indicates the end of the conversation."""
        ending_phrases = [
            "thank you", "thanks", "thx",
            "goodbye", "good bye", "bye",
            "see you", "talk to you later",
            "that's all", "that is all"
        ]

        message_lower = message.lower()
        return any(phrase in message_lower for phrase in ending_phrases)

    def _process_feedback(self, feedback: str, original_response: str) -> str:
        """Process user feedback on a LinkedIn response and learn from it."""
        # Analyze sentiment of feedback
        positive_indicators = ["good", "great", "excellent", "perfect", "like", "love"]
        negative_indicators = ["bad", "poor", "change", "adjust", "don't like", "not good", "improve"]

        feedback_lower = feedback.lower()

        # Check if feedback contains positive sentiment
        is_positive = any(indicator in feedback_lower for indicator in positive_indicators)

        # Check if feedback contains negative sentiment
        is_negative = any(indicator in feedback_lower for indicator in negative_indicators)

        # Create a memory about this feedback
        current_time = datetime.now().isoformat()

        if is_positive:
            # Store positive feedback as a memory - only store essential information
            new_memory = Memory(
                key="Positive Response Style",
                content=f"User prefers responses with this style and tone. Feedback: '{feedback}'",
                importance=8,
                last_accessed=current_time,
                created_at=current_time
            )

            self.state["long_term_memory"].append(new_memory)
            save_memories_to_file(self.state["user_profile"].name, self.state["long_term_memory"])

            # Return just the original response since it was liked
            return original_response

        elif is_negative:
            # Generate an improved response based on feedback
            system_prompt = f"""
            You are helping to improve a LinkedIn response based on user feedback.

            Original response: "{original_response}"

            User feedback: "{feedback}"

            Please generate an improved version of the LinkedIn response that addresses the user's feedback.
            Keep the same general tone and style, but make adjustments based on the specific feedback provided.
            """

            improved_response = llm.invoke([
                SystemMessage(content=system_prompt),
                HumanMessage(content="Please improve the LinkedIn response based on the feedback.")
            ])

            # Extract key feedback points for memory - only store essential information
            feedback_points = []
            if "too long" in feedback_lower or "shorter" in feedback_lower:
                feedback_points.append("Keep responses more concise")
            if "too formal" in feedback_lower:
                feedback_points.append("Use less formal tone")
            if "too casual" in feedback_lower:
                feedback_points.append("Use more professional tone")
            if "more specific" in feedback_lower:
                feedback_points.append("Be more specific in responses")

            # If no specific points were identified, create a generic one
            if not feedback_points:
                feedback_points = ["Adjust response style based on feedback"]

            # Store feedback as a memory
            for point in feedback_points:
                new_memory = Memory(
                    key=f"Response Style Preference",
                    content=point,
                    importance=9,
                    last_accessed=current_time,
                    created_at=current_time
                )

                self.state["long_term_memory"].append(new_memory)

            save_memories_to_file(self.state["user_profile"].name, self.state["long_term_memory"])

            # Add the improved response to messages
            improved_message = Message(
                role="ai",
                content=improved_response.content,
                timestamp=current_time
            )

            messages = self.state.get("messages", [])
            messages.append(improved_message)
            self.state["messages"] = messages

            # Return just the improved response
            return improved_response.content

        else:
            # For neutral feedback, just return the original response
            return original_response

    def get_conversation_history(self) -> List[Dict[str, str]]:
        """Get the conversation history."""
        return [{"role": msg.role, "content": msg.content} for msg in self.state.get("messages", [])]

    def get_long_term_memories(self) -> List[Dict[str, Any]]:
        """Get the long-term memories."""
        return [memory.model_dump() for memory in self.state.get("long_term_memory", [])]

    def is_active(self) -> bool:
        """Check if the conversation is still active."""
        return self.is_conversation_active

# For backward compatibility
LinkedInResponseAgent = ConversationalChatbot
LinkedInResponseState = ChatbotState