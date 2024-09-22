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
                 vector_table_name="vector_index", use_ollama=False):
        self.metadata = metadata
        self.chunk_size = chunk_size
        self.db_url = db_url
        self.vector_table_name = vector_table_name
        self.embeddings = SentanceTransformer("all-MiniLM-L6-v2")
        self.db = lancedb.connect(self.db_url)
        
        # if use_ollama:
        self.llm = Ollama(model="llama2")
        # else:
        # self.llm = HuggingFaceLLM(
        #     model_name="google/flan-t5-large",
        #     tokenizer_name="google/flan-t5-large",
        #     query_wrapper_prompt="Answer the following question: {query}",
        #     context_window=512,
        #     max_new_tokens=256,
        # )

    def concatenate_and_chunk_text(self, df):
        concatenated_text = " ".join(df['text'].astype(str))
        chunks = [concatenated_text[i:i+self.chunk_size] for i in range(0, len(concatenated_text), self.chunk_size)]
        return chunks


    def add_documents_to_lancedb_table(self, df: pd.DataFrame) -> None:
        # Ensure 'id' field exists
        if 'id' not in df.columns:
            df['doc_id'] = [str(uuid.uuid4()) for _ in range(len(df))]
            df['id'] = [str(uuid.uuid4()) for _ in range(len(df))]

        # Generate embeddings
        df['text'] = df['text'].astype(str)
        df['vector'] = df['text'].apply(lambda x: self.embeddings.get_text_embedding(x) if pd.notna(x) else [0.0] * 384)

        required_fields = ["doc_id","id", "text", "coordinates", "filename", "page_number", "type", "vector"]
        for field in required_fields:
            if field not in df.columns:
                if field == "page_number":
                    df[field] = 0
                elif field == "vector":
                    df[field] = [[0.0] * 384 for _ in range(len(df))]
                else:
                    df[field] = ""

        # Ensure 'page_number' is of type int64
        df['page_number'] = pd.to_numeric(df['page_number'], errors='coerce').fillna(0).astype('int64')


        # Convert metadata fields to dictionaries directly
        def create_metadata(row):
            return {
                'coordinates': json.dumps(row['coordinates']) if isinstance(row['coordinates'], dict) else str(row['coordinates']),
                'filename': row['filename'],
                'page_number': int(row['page_number']),
                'type': row['type']
            }

        df['metadata'] = df.apply(create_metadata, axis=1)

        # Select only the columns that match our Schema
        df_to_insert = df[['doc_id','id', 'text', 'metadata', 'vector']]
        vector_dim = len(df['vector'].iloc[0])
        df['vector'] = df['vector'].apply(lambda x: x if len(x) == vector_dim else [0.0] * vector_dim)

        # Define schemas
        schema_vector = pa.schema([
            pa.field("doc_id", pa.string()),
            pa.field("id", pa.string()),
            pa.field("text", pa.string()),
            pa.field("vector", pa.list_(pa.float32(), vector_dim)),  # Use list_ with fixed length
            pa.field("metadata", pa.struct([
                pa.field("coordinates", pa.string()),
                pa.field("filename", pa.string()),
                pa.field("page_number", pa.int64()),
                pa.field("type", pa.string())
            ]))
        ])

        # Create or open the vector table
        def check_table_exists(db, table_name):
            try:
                db.open_table(table_name)  # Try to open the table
                return True
            except Exception as e:
                print(f"Table '{table_name}' does not exist or cannot be opened: {e}")
                return False

        try:
            if check_table_exists(self.db, self.vector_table_name):
                # Open the existing table
                vector_table = self.db.open_table(self.vector_table_name)
                print(f"Opened existing table '{self.vector_table_name}'.")
            else:
                # Create the table with the embedding function and schema
                print(f"Table '{self.vector_table_name}' not found. Creating new table.")
                vector_table = self.db.create_table(
                    self.vector_table_name,
                    data=df_to_insert,  # Data to insert
                    schema=schema_vector  # Schema or embedding function
                )
                vector_table.create_fts_index('text')

                print(f"FTS index created on the 'text' field.")
                print(f"Table '{self.vector_table_name}' created successfully.")

            # Add data to the table

            vector_table.add(df_to_insert)
            print(f"Data added to table '{self.vector_table_name}'.")
            df_to_insert.to_csv("df_to_insert.csv")

            # Create full-text search (FTS) index on the 'text' field
        except Exception as e:
            print(f"Error handling the vector table: {e}")



    def create_query_engine(self, vector_table_name):
        vector_table = self.db.open_table(vector_table_name)
        def metadata_to_dict(metadata_str):
            try:
                return json.loads(metadata_str)
            except json.JSONDecodeError:
                return {"raw_metadata": metadata_str}

        vector_store = LanceDBVectorStore(table=vector_table, metadata_decoder=metadata_to_dict)
        vector_index = VectorStoreIndex.from_vector_store(vector_store)
        vector_query_engine = vector_index.as_query_engine(
            similarity_top_k=20,
        )

        query_engine_tool = QueryEngineTool(
            query_engine=vector_query_engine,
            metadata=ToolMetadata(
                name="vector_tool",
                description=f"Useful for retrieving general and specific information about the topic sound"
            ),
        )

        tools = [query_engine_tool]

        template = '''Answer the following questions as best you can. You have access to the following tools:

        {tools}

        Your task is to provide thorough and accurate responses to user queries.

        Use the following format:

        Question: the input question you must answer
        Thought: you should always think about what to do
        Action: the action to take, should be one of [{tool_names}]
        Action Input: the input to the action
        Observation: the result of the action
        ... (this Thought/Action/Action Input/Observation can repeat 5 times)
        Final Thought: I now know the final answer
        Final Answer: the final answer to the original input question.
        Begin!

        Question: {input}
        Thought:{agent_scratchpad}'''

        prompt = PromptTemplate.from_template(template)

        agent = ReActAgent.from_tools(
            tools,
            llm=self.llm,
            verbose=True,
            react_chat_prompt=prompt
        )

        return agent