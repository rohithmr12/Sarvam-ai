from ast import Tuple
from typing import List, Optional, Any
import pandas as pd
import os
import json
import lancedb
import pyarrow as pa
from pydantic import BaseModel
import uuid
from llama_index.core import VectorStoreIndex
from llama_index.core.schema import Document, IndexNode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core import StorageContext
# from llama_index.llms.huggingface import HuggingFaceLLM
# from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.core.agent.react import ReActAgent
from llama_index.core.prompts import PromptTemplate
from sentence_transformers import SentenceTransformer


os.environ['HF_TOKEN']="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"  
class Schema(BaseModel):
    id: str
    text: str
    metadata: dict 
    vector: List[float]


class IndexHandler:
    def __init__(self, metadata: pd.DataFrame, chunk_size=2000, db_url="./.lancedb",
                 vector_table_name="vector_index"):
        self.metadata = metadata
        self.chunk_size = chunk_size
        self.db_url = db_url
        self.vector_table_name = vector_table_name
        self.embeddings = SentenceTransformer("all-MiniLM-L6-v2")
        self.db = lancedb.connect(self.db_url)

    @staticmethod
    def chunk_text(df: pd.DataFrame, text_column: str, max_chunk_size: int = 4000):
        current_chunk = ""
        current_indices = []
        chunks=[]
        for index, row in df.iterrows():
            text = row[text_column]

            if len(current_chunk) + len(text) <= max_chunk_size:
                current_chunk += text + " "
                current_indices.append(index)
            else:
                # If adding the new text exceeds the max_chunk_size
                if current_chunk:
                    chunks.append((current_chunk.strip(), current_indices))

                # Start a new chunk
                current_chunk = text + " "
                current_indices = [index]

            # Check if the current_chunk is now over the limit
            while len(current_chunk) > max_chunk_size:
                # Find the last space before max_chunk_size
                split_index = current_chunk.rfind(' ', 0, max_chunk_size)
                if split_index == -1:
                    split_index = max_chunk_size

                # Add the chunk up to the split point
                chunks.append((current_chunk[:split_index].strip(), current_indices[:-1]))

                # Keep the remainder for the next chunk
                current_chunk = current_chunk[split_index:].strip() + " "
                current_indices = [current_indices[-1]]

        # Add the last chunk if there's any remaining text
        if current_chunk:
            chunks.append((current_chunk.strip(), current_indices))

        return chunks

    def add_documents_to_lancedb_table(self, df: pd.DataFrame) -> None:
        # Get metadata from the first row of the DataFrame
        first_row = df.iloc[0]
        doc_id = str(uuid.uuid4())  # Generate a single doc_id for all chunks

        # Create chunks
        chunks = self.chunk_text(df, 'text', self.chunk_size)

        # Create a new DataFrame from the chunks
        chunked_data = []
        for chunk_text, chunk_indices in chunks:
            chunked_data.append({
                'doc_id': doc_id,  # Use the same doc_id for all chunks
                'id': str(uuid.uuid4()),  # Unique id for each chunk
                'text': chunk_text,
                'coordinates': first_row.get('coordinates', ''),
                'filename': first_row['filename'],
                'page_number': first_row['page_number'],
                'type': first_row['type']
            })

        chunked_df = pd.DataFrame(chunked_data)

        # Generate embeddings for the chunks
        chunked_df['vector'] = chunked_df['text'].apply(lambda x: self.embeddings.encode(x).tolist())

        # Create metadata column
        chunked_df['metadata'] = chunked_df.apply(lambda row: {
            'coordinates': json.dumps(row['coordinates']) if isinstance(row['coordinates'], dict) else str(row['coordinates']),
            'filename': row['filename'],
            'page_number': int(row['page_number']) if pd.notna(row['page_number']) else -1,  # Use -1 for NaN values
            'type': row['type']
        }, axis=1)

        # Define schema and add to LanceDB table
        schema_vector = pa.schema([
            pa.field("doc_id", pa.string()),
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), 384)),
            pa.field("metadata", pa.struct([
                pa.field("coordinates", pa.string()),
                pa.field("filename", pa.string()),
                pa.field("page_number", pa.int64()),
                pa.field("type", pa.string())
            ]))
        ])

        try:
            vector_table = self.db.open_table(self.vector_table_name)
            print(f"Opened existing table '{self.vector_table_name}'.")
        except Exception:
            print(f"Table '{self.vector_table_name}' not found. Creating new table.")
            vector_table = self.db.create_table(
                self.vector_table_name,
                data=chunked_df[['doc_id', 'id', 'text', 'metadata', 'vector']],
                schema=schema_vector
            )
            vector_table.create_fts_index('text')
            print(f"FTS index created on the 'text' field.")
        else:
            vector_table.add(chunked_df[['doc_id', 'id', 'text', 'metadata', 'vector']])

        print(f"Data added to table '{self.vector_table_name}'.")


    def create_query_engine(self):
        vector_table = self.db.open_table(self.vector_table_name)

        def metadata_to_dict(metadata_str):
            try:
                return json.loads(metadata_str)
            except json.JSONDecodeError:
                return {"raw_metadata": metadata_str}

        # Setting up LanceDB vector store and index
        vector_store = LanceDBVectorStore(table=vector_table, metadata_decoder=metadata_to_dict)
        vector_index = VectorStoreIndex.from_vector_store(vector_store)
        vector_query_engine = vector_index.as_query_engine(
            similarity_top_k=5,
        )

        query_engine_tools = [
            QueryEngineTool(
                query_engine=vector_query_engine,
                metadata=ToolMetadata(
                    name="vector_tool_sound",
                    description="Only access this if there is a requirement for the context of the topic 'sound'."
                ),
            )

        ]

        # Setting up the OpenAI LLM
        from llama_index.llms.openai import OpenAI
        from llama_index.agent.openai import OpenAIAgent
        function_llm = OpenAI(model="gpt-3.5-turbo")

        # Create the agent using the tools and LLM
        agent = OpenAIAgent.from_tools(
            [query_engine_tools],
            llm=function_llm,
            verbose=True,
        )

        return agent

