from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from langchain_ollama import ChatOllama
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from src.processor import DocumentProcessor
from langsmith import Client
from langsmith.run_trees import RunTree
from dotenv import load_dotenv
import json
from pathlib import Path
from typing import List
import re
import asyncio
import os
from langchain.callbacks.manager import CallbackManager
from langchain.callbacks.base import BaseCallbackHandler
from src.logger import Logger

logger = Logger.get_logger('main')
load_dotenv()

logger.info("Starting Promptior Chatbot application")
langsmith_client = Client()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)
logger.info(f"Upload directory created at {UPLOAD_DIR}")

ENVIRONMENT = os.getenv("ENVIRONMENT", "development")
logger.info(f"Running in {ENVIRONMENT} environment")

class ChatMessage(BaseModel):
    message: str
    model: str = "deepseek-r1:7b"  

document_processor = DocumentProcessor()
processed_documents = []

@app.get("/environment")
async def get_environment():
    """Get the current environment status"""
    logger.info(f"Environment status requested: {ENVIRONMENT}")
    return {"environment": ENVIRONMENT}

@app.get("/check-api-key")
async def check_api_key():
    """Check if OpenAI API key is configured and valid"""
    api_key = os.getenv("OPENAI_API_KEY")
    is_valid = bool(api_key and len(api_key) > 20)
    logger.info(f"API key check requested. Valid: {is_valid}")
    return {"valid": is_valid}

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        logger.info(f"Processing PDF upload: {file.filename}")
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        logger.info(f"PDF saved to {file_path}")
        docs = await document_processor.load_and_split_documents([file_path])
        processed_documents.extend(docs)
        
        logger.info(f"Successfully processed {file.filename} into {len(docs)} chunks")
        return {"message": f"Successfully processed {file.filename}", "chunks": len(docs)}
    except Exception as e:
        logger.error(f"Error processing PDF upload: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def get_documents():
    logger.info(f"Retrieving document list. Total chunks: {len(processed_documents)}")
    return {
        "total_chunks": len(processed_documents),
        "documents": [doc.page_content for doc in processed_documents]
    }

@app.post("/chat")
async def chat_endpoint(chat_message: ChatMessage):
    try:
        # Check environment and API key in production
        if ENVIRONMENT == "production":
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key or len(api_key) <= 20:
                logger.error("Attempted to use chat without valid API key in production")
                raise HTTPException(
                    status_code=403,
                    detail="OpenAI API key not configured. Please contact administrator."
                )
            
            # Check if trying to use local model
            if chat_message.model in ["llama3.2", "deepseek-r1:7b"]:
                logger.warning(f"Attempted to use local model {chat_message.model} in production")
                raise HTTPException(
                    status_code=400,
                    detail="Local models are not available in production environment"
                )
            
        logger.info(f"Received chat request with model: {chat_message.model}")
        run_tree = RunTree(
            name="Chat Request",
            inputs={"message": chat_message.message, "model": chat_message.model},
            project_name="promtior-rag",
        )

        class LangSmithCallback(BaseCallbackHandler):
            def __init__(self, run_tree: RunTree):
                super().__init__()
                self.run_tree = run_tree
                
            def on_llm_start(self, *args, **kwargs):
                if not self.run_tree.start_time:
                    self.run_tree.post()
                logger.info("LLM processing started")
                
            def on_llm_end(self, *args, **kwargs):
                if not self.run_tree.end_time:
                    self.run_tree.end()
                    self.run_tree.post()
                logger.info("LLM processing completed")
                
            def on_llm_error(self, error, *args, **kwargs):
                if not self.run_tree.end_time:
                    self.run_tree.end(error=str(error))
                    self.run_tree.post()
                logger.error(f"LLM error occurred: {str(error)}")

        callback_manager = CallbackManager([LangSmithCallback(run_tree)])
        
        # Initialize appropriate chat model based on selection
        if chat_message.model in ["gpt-4", "gpt-3.5-turbo"]:
            logger.info(f"Using OpenAI model: {chat_message.model}")
            chat_model = ChatOpenAI(
                model=chat_message.model,
                api_key=os.getenv("OPENAI_API_KEY"),
                streaming=True,
                callback_manager=callback_manager
            )
        else:
            logger.info(f"Using Ollama model: {chat_message.model}")
            chat_model = ChatOllama(
                model=chat_message.model,
                base_url=os.getenv("OLLAMA_BASE_URL"),
                streaming=True,
                callback_manager=callback_manager
            )

        messages = []
        url_match = re.search(r'@(https?://[^\s]+)', chat_message.message)
        
        async def process_stream():
            if url_match:
                url = url_match.group(1)
                try:
                    logger.info(f"Processing URL: {url}")
                    yield f"data: {json.dumps({'status': 'processing'})}\n\n"
                    
                    docs = await document_processor.load_and_split_documents([url])
                    question = chat_message.message.replace(url_match.group(0), "").strip()
                    logger.info(f"URL processed into {len(docs)} document chunks")
                    
                    if docs and document_processor.vectorstore:
                        context, analysis = await document_processor.get_relevant_context_async(question)
                        logger.info(f"Retrieved context with analysis: {analysis}")
                        rag_prompt = document_processor.get_rag_prompt(
                            question=question, 
                            context=context
                        )
                        messages.append(SystemMessage(content=rag_prompt))
                    else:
                        logger.warning("No documents processed from URL")
                        messages.append(SystemMessage(content="Failed to process the URL. Please try again."))
                except Exception as e:
                    logger.error(f"Error processing URL: {str(e)}")
                    messages.append(SystemMessage(content=f"Error processing URL: {str(e)}"))
                    if not run_tree.end_time:
                        run_tree.end(error=str(e))
                        run_tree.post()
                    return
            else:
                if document_processor.vectorstore:
                    logger.info("Processing question with existing vectorstore")
                    context, analysis = await document_processor.get_relevant_context_async(chat_message.message)
                    logger.info(f"Retrieved context with analysis: {analysis}")
                    rag_prompt = document_processor.get_rag_prompt(
                        question=chat_message.message,
                        context=context
                    )
                    messages.append(SystemMessage(content=rag_prompt))
                else:
                    logger.warning("No vectorstore available for query")
                    messages.append(SystemMessage(content="Please include a URL with your question using @url format."))

            messages.append(HumanMessage(content=chat_message.message))

            try:
                logger.info("Starting chat stream")
                async for chunk in chat_model.astream(messages):
                    if chunk.content:
                        yield f"data: {json.dumps({'content': chunk.content})}\n\n"
                        await asyncio.sleep(0)
                logger.info("Chat stream completed")
            except Exception as e:
                logger.error(f"Error in chat stream: {str(e)}")
                if not run_tree.end_time:
                    run_tree.end(error=str(e))
                    run_tree.post()
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(process_stream(), media_type="text/event-stream")
        
    except Exception as e:
        logger.error(f"Error in chat endpoint: {str(e)}")
        if 'run_tree' in locals() and not run_tree.end_time:
            run_tree.end(error=str(e))
            run_tree.post()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    logger.info("Serving index.html")
    return FileResponse('static/index.html') 