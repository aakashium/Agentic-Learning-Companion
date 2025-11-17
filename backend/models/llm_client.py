import google.generativeai as genai
from backend.config import settings

genai.configure(api_key=settings.GEMINI_API_KEY)

class LLMClient:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-flash-2.0")

    def generate(self, prompt: str):
        try:
            response = self.model.generate_content(prompt)
            return response.text
        except Exception as e:
            return f"LLM error: {e}"
