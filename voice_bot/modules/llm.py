# LLM:-
from openai import OpenAI
from config import cloudllm

# Defined Main Class for GPT 4O Model Reply:-
class LLMService:
    # Defined API Key and Model:-
    def __init__(self):
        self.client = OpenAI(api_key=cloudllm.OPENAI_API_KEY)
        self.model = cloudllm.GPT_MODEL

    # Defined Streaming Response Form Model:-
    def stream_response(self, user_query: str) -> str:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system",
                 "content": "You are an AI assistant. Keep all responses under 100 tokens, concise and clear."},
                {"role": "user", "content": user_query}
            ],
        )
        return response.choices[0].message.content


# Check For Main Object Calling Function of LLM:-
if __name__ == "__main__":
    llm = LLMService()
    reply = llm.stream_response("What is the Capital of India")
    print(reply)
    
