from langchain_ollama import ChatOllama
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
import json
import re
from .prompts import QUERY_ANALYZER_PROMPT
import os
from .logger import Logger

logger = Logger.get_logger('query_analyzer')

class QueryAnalyzer:
    def __init__(self, model: str = "deepseek-r1:7b"):
        logger.info(f"Initializing QueryAnalyzer with model: {model}")
        self.model = model
        self._setup_chat_model(model)
        self.prompt = QUERY_ANALYZER_PROMPT

    def _setup_chat_model(self, model: str):
        logger.info(f"Setting up chat model: {model}")
        if model in ["gpt-4", "gpt-3.5-turbo"]:
            self.chat_model = ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=model
            )
            logger.info(f"OpenAI chat model {model} configured")
        else:
            self.chat_model = ChatOllama(
                model=model,
                base_url=os.getenv("OLLAMA_BASE_URL")
            )
            logger.info(f"Ollama chat model configured for {model}")

    def update_model(self, model: str):
        logger.info(f"Updating model to: {model}")
        self.model = model
        self._setup_chat_model(model)

    async def analyze_query(self, user_question: str) -> dict:
        try:
            logger.info(f"Analyzing query: {user_question}")
            clean_question = re.sub(r'@https?://\S+', '', user_question).strip()
            logger.info(f"Cleaned question: {clean_question}")
            
            prompt = self.prompt.format(question=clean_question)
            response = await self.chat_model.ainvoke(prompt)
            
            try:
                content = response.content.strip()
                content = re.sub(r'^```json\s*|\s*```$', '', content)
                analysis = json.loads(content)
                logger.info(f"Query analysis completed successfully: {analysis}")
                return analysis
                
            except json.JSONDecodeError as e:
                logger.warning(f"JSON decode error, using fallback analysis: {str(e)}")
                return self._fallback_analysis(clean_question, e, response.content)
                
        except Exception as e:
            logger.error(f"Error analyzing query: {str(e)}")
            return self._error_analysis(clean_question, e)

    def _fallback_analysis(self, question: str, error: Exception, raw_response: str) -> dict:
        logger.info("Using fallback analysis")
        return {
            "analysis": "Analyzing section request",
            "queries": [
                question,
                "" + re.sub(r'what is .* content of the (.*?) section.*', r'\1', question, flags=re.I),
                "content " + re.sub(r'what is .* content of the (.*?) section.*', r'\1', question, flags=re.I)
            ]
        }

    def _error_analysis(self, question: str, error: Exception) -> dict:
        logger.warning(f"Error in analysis, using direct search: {str(error)}")
        return {
            "analysis": "Using direct search",
            "queries": [question]
        }

