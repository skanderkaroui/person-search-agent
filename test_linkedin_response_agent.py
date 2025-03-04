import os
import json
from dotenv import load_dotenv
from linkedin_response_agent import ConversationalChatbot, UserProfile, Memory
from datetime import datetime
from typing import List, Dict, Any, Optional

# Sample user profile
DEFAULT_USER_PROFILE = {
    "name": "Alex Johnson",
    "company_name": "TechInnovate Solutions",
    "work_function": "Engineering",
    "persona_description": "A technical leader who values innovation and collaboration. Passionate about solving complex problems with elegant solutions. Believes in mentoring junior engineers and creating inclusive team environments.",
    "communication_style": "Professional but approachable. Uses technical terminology appropriately but explains complex concepts clearly. Occasionally uses humor to make points more relatable."
}

# Load environment variables
load_dotenv()

# Check if OpenAI API key is set
if not os.getenv("OPENAI_API_KEY"):
    print("Warning: OPENAI_API_KEY environment variable not set.")
    print("Please set it in your .env file or export it in your terminal.")
    print("Example: export OPENAI_API_KEY=your-api-key")
    exit(1)

# Sample memories for testing
def create_sample_memories(user_name: str):
    """Create sample memories for testing."""
    current_time = datetime.now().isoformat()

    sample_memories = [
        Memory(
            key="Communication Style Preference",
            content="Prefers concise responses with technical depth but explained clearly.",
            importance=8,
            last_accessed=current_time,
            created_at=current_time
        ),
        Memory(
            key="Response Format",
            content="Likes responses that start with a clear position and then provide supporting evidence.",
            importance=7,
            last_accessed=current_time,
            created_at=current_time
        ),
        Memory(
            key="Emoji Usage",
            content="Appreciates occasional use of relevant emojis, especially the monkey emoji 🐒.",
            importance=5,
            last_accessed=current_time,
            created_at=current_time
        ),
        Memory(
            key="Technical Interests",
            content="Particularly interested in AI, machine learning, and software architecture.",
            importance=6,
            last_accessed=current_time,
            created_at=current_time
        )
    ]

    # Save to file
    filename = f"{user_name.lower().replace(' ', '_')}_memories.json"
    memories_dict = [memory.model_dump() for memory in sample_memories]

    with open(filename, 'w') as f:
        json.dump(memories_dict, f, indent=2)

    print(f"Created {len(sample_memories)} sample memories for {user_name} in {filename}")
    return sample_memories

def load_sample_post(file_path="sample_linkedin_post.json"):
    """Load the sample LinkedIn post from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            return json.load(f)
    except Exception as e:
        print(f"Error loading sample post: {e}")
        return {
            "author": "Default Author",
            "content": "Default LinkedIn post content for testing."
        }

def main():
    print("=" * 50)
    print("LinkedIn Response Chatbot")
    print("=" * 50)

    # Create user profile
    print("\nCreating user profile...")
    user_profile = UserProfile(
        name=DEFAULT_USER_PROFILE["name"],
        company_name=DEFAULT_USER_PROFILE["company_name"],
        work_function=DEFAULT_USER_PROFILE["work_function"],
        persona_description=DEFAULT_USER_PROFILE["persona_description"],
        communication_style=DEFAULT_USER_PROFILE["communication_style"]
    )

    # Initialize the chatbot
    chatbot = ConversationalChatbot(user_profile=user_profile)

    # The chatbot will automatically load memories from alex_johnson_memories.json
    # since we're using the default file-based memory system

    print(f"\nChatbot initialized for {user_profile.name}")

    # Check if memory file exists
    memory_file = f"{user_profile.name.lower().replace(' ', '_')}_memories.json"
    if os.path.exists(memory_file):
        print(f"Found memory file: {memory_file}")
    else:
        print(f"Memory file not found: {memory_file}")

    # Load the sample LinkedIn post
    sample_post = load_sample_post()

    # Main interaction loop
    while True:
        print("\nOptions:")
        print("1. Generate response to the sample LinkedIn post")
        print("2. View conversation history")
        print("3. View user profile")
        print("4. View memory")
        print("5. Manage memories")
        print("6. Exit")

        choice = input("\nEnter your choice (1-6): ")

        if choice == "1":
            # Use the sample LinkedIn post
            post_author = sample_post["author"]
            post_content = sample_post["content"]

            # Generate response to the LinkedIn post
            print("\nGenerating response...")
            response = chatbot.generate_linkedin_response(post_content, post_author)

            print("\n" + "=" * 50)
            print("LinkedIn Post:")
            print(f"Author: {post_author}")
            print(f"Content: {post_content}")
            print("\nGenerated Response:")
            print(response)
            print("=" * 50)

            # Start conversation for feedback
            print("\nYou can now provide feedback on the response.")
            print("Type 'thank you' or similar phrases to end the conversation when you're done.")

            # Conversation loop
            while chatbot.is_active():
                user_input = input("\nYou: ")
                if not user_input.strip():
                    continue

                bot_response = chatbot.chat(user_input)
                print(f"\nResponse: {bot_response}")

                if not chatbot.is_active():
                    print("\nConversation ended.")
                    break

        elif choice == "2":
            # View conversation history
            history = chatbot.get_conversation_history()

            print("\n" + "=" * 50)
            print("Conversation History:")

            if history:
                for msg in history:
                    if msg["role"] == "human":
                        print(f"\nYou: {msg['content']}")
                    elif msg["role"] == "ai":
                        print(f"\nResponse: {msg['content']}")
                    elif msg["role"] == "system":
                        print(f"\n[System: {msg['content']}]")
            else:
                print("No conversation history yet.")

            print("=" * 50)

        elif choice == "3":
            # View user profile
            profile = chatbot.state.get("user_profile")
            if profile:
                print("\n" + "=" * 50)
                print("User Profile:")
                print(f"Name: {profile.name}")
                print(f"Company: {profile.company_name}")
                print(f"Function: {profile.work_function}")
                print(f"Persona: {profile.persona_description}")
                print(f"Communication Style: {profile.communication_style}")
                print("=" * 50)
            else:
                print("No user profile set.")

        elif choice == "4":
            # View memory
            memories = chatbot.get_long_term_memories()

            print("\n" + "=" * 50)
            print("Chatbot's Memories:")

            if memories:
                for memory in memories:
                    print(f"\n• {memory['key']} (Importance: {memory['importance']})")
                    print(f"  {memory['content']}")
                    print(f"  Last accessed: {memory['last_accessed']}")
            else:
                print("No memories stored yet.")

            print("=" * 50)

        elif choice == "5":
            # Manage memories
            print("\n" + "=" * 50)
            print("Memory Management:")
            print("1. Create sample memories")
            print("2. Back to main menu")

            mem_choice = input("\nEnter your choice (1-2): ")

            if mem_choice == "1":
                # Create sample memories
                confirm = input("This will create sample memories. Continue? (y/n): ")
                if confirm.lower() == 'y':
                    create_sample_memories(user_profile.name)
                    print("Sample memories created. Restart the chatbot to use them.")

            elif mem_choice == "2":
                # Back to main menu
                continue

            else:
                print("Invalid choice.")

        elif choice == "6":
            print("\nExiting Chatbot. Goodbye!")
            break

        else:
            print("\nInvalid choice. Please enter a number between 1 and 6.")

if __name__ == "__main__":
    main()