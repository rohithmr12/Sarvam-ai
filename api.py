from fastapi import FastAPI, File, UploadFile, HTTPException
from pydantic import BaseModel
import uvicorn
import os
from tempfile import NamedTemporaryFile
from main import process_document
from Indexer import IndexHandler
import pandas as pd

app = FastAPI()

class Query(BaseModel):
    query: str
index_handler = process_document("iesc111.pdf")

@app.post("/upload")
async def upload_file():
    try:
        # index_handler = process_document("iesc111.pdf")
        if index_handler is None:
            raise HTTPException(status_code=400, detail="Failed to process document")
        return {"message": "File processed successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error processing document: {str(e)}")

@app.post("/query")
async def query_document(query: Query):
    try:
        db_url = "./.lancedb"
        # index_handler = process_document("iesc111.pdf", db_url)
        if index_handler is None:
            raise HTTPException(status_code=400, detail="Failed to process document")
        
        agent = index_handler.create_query_engine()
        if agent is None:
            raise HTTPException(status_code=500, detail="Failed to create query engine")
        
        response = agent.chat(query.query)
        return {"response": str(response)}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error querying document: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)