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
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from .logger import Logger

logger = Logger.get_logger('processor')

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
ENV_PATH = ROOT_DIR / '.env'

load_dotenv(dotenv_path=ENV_PATH)

class DocumentProcessor:
    def __init__(self, model: str = "deepseek-r1:7b"):
        logger.info(f"Initializing DocumentProcessor with model: {model}")
        self.source_processor = SourceProcessor()
        self.document_loader = DocumentLoader()
        self.query_analyzer = QueryAnalyzer(model)
        self.model = model
        self._setup_embeddings(model)
        self.retriever = DocumentRetriever(self.embeddings, model)
        self.vectorstore = None
        self.rag_prompt = RAG_PROMPT
        self.langsmith_client = Client()

    def _setup_embeddings(self, model: str):
        logger.info(f"Setting up embeddings for model: {model}")
        if model in ["gpt-4o", "gpt-4o-mini"]:
            self.embeddings = OpenAIEmbeddings(
                api_key=os.getenv("OPENAI_API_KEY"),
            )
            logger.info("OpenAI embeddings configured")
        else:
            self.embeddings = OllamaEmbeddings(
                model=model,
                base_url=os.getenv("OLLAMA_BASE_URL"),
            )
            logger.info(f"Ollama embeddings configured for model {model}")

    def update_model(self, model: str):
        logger.info(f"Updating model to: {model}")
        self.model = model
        self.query_analyzer.update_model(model)
        self._setup_embeddings(model)
        self.retriever.update_model(model)

    async def load_and_split_documents(self, sources: List[Union[str, Path]]) -> List[Document]:
        logger.info(f"Processing {len(sources)} sources")
        processed_sources = self.source_processor.process_sources(sources)
        logger.info(f"Found {len(processed_sources['urls'])} URLs and {len(processed_sources['files'])} files")
        
        web_docs = await self.document_loader.load_web_documents(processed_sources['urls'])
        logger.info(f"Loaded {len(web_docs)} web documents")
        
        splits = await self.document_loader.process_documents(web_docs)
        logger.info(f"Created {len(splits)} document splits")
        
        if splits:
            self.retriever.create_vectorstore(splits)
            self.vectorstore = self.retriever.vectorstore
            logger.info("Vectorstore created and updated")
            
        return splits

    async def get_relevant_context_async(self, query: str, k: int = 4) -> Tuple[str, dict]:
        if not self.vectorstore:
            logger.warning("No vectorstore available for context retrieval")
            return "", {}
            
        logger.info(f"Getting relevant context for query: {query}")
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
                logger.info(f"Query analysis completed: {analysis}")
                analysis_run.end(outputs=analysis)
                analysis_run.post()
            except Exception as e:
                logger.error(f"Error in query analysis: {str(e)}")
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
                logger.info(f"Retrieved {len(relevant_docs)} relevant documents")
                search_run.end(outputs={
                    "docs_found": len(relevant_docs),
                    "total_context_length": sum(len(doc.page_content) for doc in relevant_docs)
                })
                search_run.post()
            except Exception as e:
                logger.error(f"Error retrieving documents: {str(e)}")
                search_run.end(error=str(e))
                search_run.post()
                raise
        
            context = "\n\n".join(doc.page_content for doc in relevant_docs)
            logger.info(f"Generated context with length: {len(context)}")
            
            parent_run.end(outputs={"context_length": len(context)})
            parent_run.post()
            
            return context, analysis
            
        except Exception as e:
            logger.error(f"Error in context retrieval: {str(e)}")
            parent_run.end(error=str(e))
            parent_run.post()
            raise

    def get_rag_prompt(self, question: str, context: str) -> str:
        logger.info("Generating RAG prompt")
        return self.rag_prompt.format(context=context, question=question)

    def process_documents(self, documents: List[Document]) -> List[Document]:
        logger.info(f"Processing {len(documents)} documents")
        processed_docs = []
        
        for doc in documents:
            if doc.metadata.get('type') == 'social_media' and 'linkedin.com' in doc.metadata.get('source', ''):
                logger.info("Processing LinkedIn document")
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
                
        logger.info(f"Document processing complete. Generated {len(processed_docs)} processed documents")
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