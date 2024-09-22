# Import necessary libraries
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import chainlit as cl
from chainlit.types import AskResponse
from typing import List, Optional, Any
import pandas as pd
import lancedb
import os
from llama_index.core import VectorStoreIndex
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.llms.huggingface import HuggingFaceLLM
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.react import ReActAgent
from llama_index.core.prompts import PromptTemplate

# Import the IndexHandler class from your existing code
from Vdb_handler import IndexHandler  # Replace 'your_module' with the actual module name

# Initialize FastAPI app
app = FastAPI()

# Initialize IndexHandler
metadata = pd.DataFrame()  # Replace with your actual metadata
index_handler = IndexHandler(metadata)

# Create query engine
agent = index_handler.create_query_engine("vector_index")  # Replace "vector_index" with your actual table name

class Query(BaseModel):
    text: str

@app.post("/query")
async def query(query: Query):
    try:
        response = await agent.aquery(query.text)
        return {"response": str(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# Chainlit interface
@cl.on_chat_start
async def start():
    cl.user_session.set("agent", agent)

@cl.on_message
async def main(message: str):
    agent = cl.user_session.get("agent")
    response = await agent.aquery(message)
    await cl.Message(content=str(response)).send()

# Run the Chainlit app
if __name__ == "__main__":
    cl.run()