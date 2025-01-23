from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document
from typing import List, Union, Tuple
from pathlib import Path
from .document_loader import DocumentLoader
from .query_analyzer import QueryAnalyzer
from .prompts import RAG_PROMPT
from .source_processor import SourceProcessor
from langsmith import Client, RunTree
from dotenv import load_dotenv
import os
from .retriever import DocumentRetriever

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = ROOT_DIR / '.env'

load_dotenv(dotenv_path=ENV_PATH)

class DocumentProcessor:
    def __init__(self, model: str = "deepseek-r1:1.5b"):
        self.source_processor = SourceProcessor()
        self.document_loader = DocumentLoader()
        self.query_analyzer = QueryAnalyzer(model)
        self.embeddings = OllamaEmbeddings(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL"),
        )
        self.retriever = DocumentRetriever(self.embeddings, model)
        self.vectorstore = None
        self.rag_prompt = RAG_PROMPT
        self.langsmith_client = Client()
        self.model = model

    def update_model(self, model: str):
        """Update the model across all components"""
        self.model = model
        self.query_analyzer.update_model(model)
        self.embeddings = OllamaEmbeddings(
            model=model,
            base_url=os.getenv("OLLAMA_BASE_URL"),
        )
        self.retriever.update_model(model)

    async def load_and_split_documents(self, sources: List[Union[str, Path]]) -> List[Document]:
        processed_sources = self.source_processor.process_sources(sources)
        web_docs = await self.document_loader.load_web_documents(processed_sources['urls'])
        splits = await self.document_loader.process_documents(web_docs)
        
        if splits:
            self.retriever.create_vectorstore(splits)
            self.vectorstore = self.retriever.vectorstore
            
        return splits

    async def get_relevant_context_async(self, query: str, k: int = 4) -> Tuple[str, dict]:
        if not self.vectorstore:
            return "", {}
            
        parent_run = RunTree(
            name="Get Relevant Context",
            inputs={"query": query},
            project_name="promtior-rag",
        )
        parent_run.post()
        
        try:
            analysis_run = RunTree(
                name="analyze_query",
                inputs={"query": query},
                parent_run=parent_run,
                project_name="promtior-rag"
            )
            analysis_run.post()
            
            try:
                analysis = await self.query_analyzer.analyze_query(query)
                analysis_run.end(outputs=analysis)
                analysis_run.post()
            except Exception as e:
                analysis_run.end(error=str(e))
                analysis_run.post()
                raise
            
            search_run = RunTree(
                name="search_documents",
                inputs={"query": query},
                parent_run=parent_run,
                project_name="promtior-rag"
            )
            search_run.post()
            
            try:
                relevant_docs = await self.retriever.get_relevant_documents(query, k=k)
                search_run.end(outputs={
                    "docs_found": len(relevant_docs),
                    "total_context_length": sum(len(doc.page_content) for doc in relevant_docs)
                })
                search_run.post()
            except Exception as e:
                search_run.end(error=str(e))
                search_run.post()
                raise
        
            context = "\n\n".join(doc.page_content for doc in relevant_docs)
            
            context = "\n\n".join(doc.page_content for doc in relevant_docs)
            parent_run.end(outputs={"context_length": len(context)})
            parent_run.post()
            
            return context, analysis
            
        except Exception as e:
            parent_run.end(error=str(e))
            parent_run.post()
            raise

    def get_rag_prompt(self, question: str, context: str) -> str:
        return self.rag_prompt.format(context=context, question=question)

    def process_documents(self, documents: List[Document]) -> List[Document]:
        processed_docs = []
        
        for doc in documents:
            if doc.metadata.get('type') == 'social_media' and 'linkedin.com' in doc.metadata.get('source', ''):
                sections = doc.page_content.split('\n\n')
                for section in sections:
                    if section.strip():
                        processed_docs.append(
                            Document(
                                page_content=section,
                                metadata={
                                    **doc.metadata,
                                    'section_type': self._identify_linkedin_section(section)
                                }
                            )
                        )
            else:
                processed_docs.append(doc)
                
        return processed_docs
        
    def _identify_linkedin_section(self, content: str) -> str:
        """Identify the type of LinkedIn content section."""
        content_lower = content.lower()
        
        if 'company:' in content_lower:
            return 'company_info'
        elif 'about:' in content_lower:
            return 'about'
        elif 'post:' in content_lower:
            return 'post'
        elif 'content from' in content_lower:
            return 'additional_content'
        else:
            return 'general' 