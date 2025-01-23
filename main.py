from fastapi import FastAPI, HTTPException, UploadFile, File
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import StreamingResponse, FileResponse
from pydantic import BaseModel
from langchain_ollama import ChatOllama
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

load_dotenv()

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

class ChatMessage(BaseModel):
    message: str
    model: str = "deepseek-r1:1.5b"  

document_processor = DocumentProcessor()
processed_documents = []

@app.post("/upload-pdf")
async def upload_pdf(file: UploadFile = File(...)):
    try:
        file_path = UPLOAD_DIR / file.filename
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        docs = await document_processor.load_and_split_documents([file_path])
        processed_documents.extend(docs)
        
        return {"message": f"Successfully processed {file.filename}", "chunks": len(docs)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/documents")
async def get_documents():
    return {
        "total_chunks": len(processed_documents),
        "documents": [doc.page_content for doc in processed_documents]
    }

@app.post("/chat")
async def chat_endpoint(chat_message: ChatMessage):
    try:
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
                
            def on_llm_end(self, *args, **kwargs):
                if not self.run_tree.end_time:
                    self.run_tree.end()
                    self.run_tree.post()
                
            def on_llm_error(self, error, *args, **kwargs):
                if not self.run_tree.end_time:
                    self.run_tree.end(error=str(error))
                    self.run_tree.post()

        callback_manager = CallbackManager([LangSmithCallback(run_tree)])
        
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
                    yield f"data: {json.dumps({'status': 'processing'})}\n\n"
                    
                    docs = await document_processor.load_and_split_documents([url])
                    question = chat_message.message.replace(url_match.group(0), "").strip()
                    
                    if docs and document_processor.vectorstore:
                        context, analysis = await document_processor.get_relevant_context_async(question)
                        rag_prompt = document_processor.get_rag_prompt(
                            question=question, 
                            context=context
                        )
                        messages.append(SystemMessage(content=rag_prompt))
                    else:
                        messages.append(SystemMessage(content="Failed to process the URL. Please try again."))
                except Exception as e:
                    messages.append(SystemMessage(content=f"Error processing URL: {str(e)}"))
                    if not run_tree.end_time:
                        run_tree.end(error=str(e))
                        run_tree.post()
                    return
            else:
                if document_processor.vectorstore:
                    context, analysis = await document_processor.get_relevant_context_async(chat_message.message)
                    rag_prompt = document_processor.get_rag_prompt(
                        question=chat_message.message,
                        context=context
                    )
                    messages.append(SystemMessage(content=rag_prompt))
                else:
                    messages.append(SystemMessage(content="Please include a URL with your question using @url format."))

            messages.append(HumanMessage(content=chat_message.message))

            try:
                async for chunk in chat_model.astream(messages):
                    if chunk.content:
                        yield f"data: {json.dumps({'content': chunk.content})}\n\n"
                        await asyncio.sleep(0)
            except Exception as e:
                if not run_tree.end_time:
                    run_tree.end(error=str(e))
                    run_tree.post()
                yield f"data: {json.dumps({'error': str(e)})}\n\n"

        return StreamingResponse(process_stream(), media_type="text/event-stream")
        
    except Exception as e:
        if 'run_tree' in locals() and not run_tree.end_time:
            run_tree.end(error=str(e))
            run_tree.post()
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/")
async def read_root():
    return FileResponse('static/index.html') 