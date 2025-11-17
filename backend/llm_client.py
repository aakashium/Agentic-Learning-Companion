import google.generativeai as genai
from backend.config import settings

genai.configure(api_key=settings.GOOGLE_API_KEY)

class LLMClient:
    def __init__(self):
        self.model = genai.GenerativeModel("gemini-2.0-flash")

    def generate(self, prompt: str):
        res = self.model.generate_content(prompt)
        return res.text
