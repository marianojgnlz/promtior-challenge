from langchain_ollama import ChatOllama
from langchain_core.documents import Document
import json
import re
from .prompts import QUERY_ANALYZER_PROMPT
import os
class QueryAnalyzer:
    def __init__(self, model: str = "deepseek-r1:1.5b"):
        self.chat_model = ChatOllama(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL")
        )
        self.prompt = QUERY_ANALYZER_PROMPT
        self.model = model

    def update_model(self, model: str):
        """Update the model used by the analyzer"""
        self.model = model
        self.chat_model = ChatOllama(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL")
        )

    async def analyze_query(self, user_question: str) -> dict:
        try:
            clean_question = re.sub(r'@https?://\S+', '', user_question).strip()
            
            prompt = self.prompt.format(question=clean_question)
            response = await self.chat_model.ainvoke(prompt)
            
            try:
                content = response.content.strip()
                content = re.sub(r'^```json\s*|\s*```$', '', content)
                analysis = json.loads(content)
                
                return analysis
                
            except json.JSONDecodeError as e:
                return self._fallback_analysis(clean_question, e, response.content)
                
        except Exception as e:
            return self._error_analysis(clean_question, e)

    def _fallback_analysis(self, question: str, error: Exception, raw_response: str) -> dict:
        return {
            "analysis": "Analyzing section request",
            "queries": [
                question,
                "" + re.sub(r'what is .* content of the (.*?) section.*', r'\1', question, flags=re.I),
                "content " + re.sub(r'what is .* content of the (.*?) section.*', r'\1', question, flags=re.I)
            ]
        }

    def _error_analysis(self, question: str, error: Exception) -> dict:
        return {
            "analysis": "Using direct search",
            "queries": [question]
        }

