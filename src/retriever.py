from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from typing import List
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from .logger import Logger

logger = Logger.get_logger('retriever')

ROOT_DIR = Path(__file__).resolve().parent.parent
PERSIST_DIR = ROOT_DIR / "chroma_db"
PERSIST_DIR.mkdir(exist_ok=True)

class DocumentRetriever:
    def __init__(self, embeddings, model: str = "deepseek-r1:7b", vectorstore=None):
        logger.info(f"Initializing DocumentRetriever with model: {model}")
        self.embeddings = embeddings
        self.model = model
        self._setup_llm(model)
        
        # Initialize vectorstore from disk if it exists
        if vectorstore:
            self.vectorstore = vectorstore
        else:
            try:
                self.vectorstore = Chroma(
                    persist_directory=str(PERSIST_DIR),
                    embedding_function=embeddings
                )
                logger.info("Loaded existing vectorstore from disk")
            except Exception as e:
                logger.info("No existing vectorstore found, will create new one when documents are added")
                self.vectorstore = None

        self.query_prompt = PromptTemplate(
            input_variables=["question"],
            template="""You are an expert at analyzing questions and creating search queries.
            For the following question, generate 3 different search queries that would help find relevant information.
            Make the queries diverse to capture different aspects of the question.
            
            Original question: {question}
            
            Generate queries that:
            1. Look for direct answers
            2. Search for related concepts
            3. Find supporting information
            
            DO NOT include any prefixes or explanations. Return ONLY the search queries, one per line."""
        )

    def _setup_llm(self, model: str):
        logger.info(f"Setting up LLM for model: {model}")
        if model in ["gpt-4o", "gpt-4o-mini"]:
            self.llm = ChatOpenAI(
                api_key=os.getenv("OPENAI_API_KEY"),
                model=model,
                callbacks=[]
            )
            logger.info(f"OpenAI LLM {model} configured")
        else:
            self.llm = ChatOllama(
                model=model,
                base_url=os.getenv("OLLAMA_BASE_URL"),
                callbacks=[]
            )
            logger.info(f"Ollama LLM configured for {model}")

    def update_model(self, model: str):
        logger.info(f"Updating model to: {model}")
        self.model = model
        self._setup_llm(model)

    def create_vectorstore(self, documents: List[Document]) -> None:
        if not documents:
            logger.warning("No documents provided for vectorstore creation")
            return

        logger.info(f"Creating vectorstore with {len(documents)} documents")
        if not self.vectorstore:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings,
                persist_directory=str(PERSIST_DIR)
            )
            self.vectorstore.persist()
            logger.info("New vectorstore created and persisted")
        else:
            self.vectorstore.add_documents(documents)
            self.vectorstore.persist()
            logger.info("Documents added to existing vectorstore and persisted")

    async def get_relevant_documents(self, query: str, k: int = 4) -> List[Document]:
        if not self.vectorstore:
            logger.warning("No vectorstore available for document retrieval")
            return []

        logger.info(f"Retrieving documents for query: {query} with k={k}")
        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k*2
            }
        )
        
        docs = await retriever.ainvoke(query)
        logger.info(f"Retrieved {len(docs)} initial documents")
        
        seen = set()
        unique_docs = []
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

        final_docs = unique_docs[:k]
        logger.info(f"Returning {len(final_docs)} unique documents")
        return final_docs