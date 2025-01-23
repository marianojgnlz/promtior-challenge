from langchain.retrievers.multi_query import MultiQueryRetriever
from langchain_community.vectorstores import Chroma
from langchain_core.documents import Document
from langchain_core.runnables import RunnableConfig
from langchain_ollama import ChatOllama
from typing import List
import os
from pathlib import Path
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = ROOT_DIR / '.env'
load_dotenv(dotenv_path=ENV_PATH)

class DocumentRetriever:
    def __init__(self, embeddings, model: str = "deepseek-r1:1.5b", vectorstore=None):
        self.embeddings = embeddings
        self.vectorstore = vectorstore
        self.model = model
        self.llm = ChatOllama(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL"),
            callbacks=[]
        )
        
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

    def update_model(self, model: str):
        """Update the model used by the retriever"""
        self.model = model
        self.llm = ChatOllama(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL"),
            callbacks=[]
        )

    def create_vectorstore(self, documents: List[Document]) -> None:
        """Create or update the vector store with documents."""
        if not documents:
            return

        if not self.vectorstore:
            self.vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=self.embeddings
            )
        else:
            self.vectorstore.add_documents(documents)

    async def get_relevant_documents(self, query: str, k: int = 4) -> List[Document]:
        """Get relevant documents using MultiQueryRetriever."""
        if not self.vectorstore:
            return []

        retriever = self.vectorstore.as_retriever(
            search_type="similarity",
            search_kwargs={
                "k": k*2
            }
        )
        
        docs = await retriever.ainvoke(query)
        
        seen = set()
        unique_docs = []
        for doc in docs:
            if doc.page_content not in seen:
                seen.add(doc.page_content)
                unique_docs.append(doc)

        return unique_docs[:k]