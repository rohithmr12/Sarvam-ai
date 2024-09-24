from typing import List, Optional,Any
import pandas as pd
import os
import json
import lancedb
import pyarrow as pa
from lancedb.pydantic import LanceModel, Vector
from llama_index.core import VectorStoreIndex, SummaryIndex
from llama_index.core.schema import Document, IndexNode
from llama_index.core.tools import QueryEngineTool, ToolMetadata
from llama_index.agent.openai import OpenAIAgent
from llama_index.llms.openai import OpenAI
from llama_index.vector_stores.lancedb import LanceDBVectorStore
from llama_index.core import StorageContext
from unstructured.partition.pdf import partition_pdf
from langchain_core.prompts import PromptTemplate
from tqdm import tqdm
from llama_index.core.llms import ChatMessage
import shutil
import uuid
import base64
from pydantic import BaseModel, Field
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import fitz
import base64
import pandas as pd
import base64
from io import BytesIO
from PIL import Image
from llama_index.core.tools import QueryEngineTool, ToolMetadata
import shutil
import uuid
import base64
from pydantic import BaseModel, Field
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from pdf2image import convert_from_path
from huggingface_hub import InferenceClient


class DocumentBuilder:
    def __init__(self, metadata: pd.DataFrame):
        self.metadata = metadata
        self.data_model = {
            "filetype": "first",
            "last_modified": "first",
            "type": "first",
            "page_number": "first",
            "coordinates": "first",
            "text_as_html": "sum",
        }
        self.client = InferenceClient("meta-llama/Meta-Llama-3-8B")

    def segregate_metadata(self):
        text_df = self.metadata[(self.metadata["type"] != "Table") & (self.metadata["type"] != "TableChunk") & (self.metadata["type"] != "Image")]
        table_df = self.metadata[(self.metadata["type"] == "Table") | (self.metadata["type"] == "TableChunk")]
        image_df = self.metadata[self.metadata["type"] == "Image"]

        filename_column = table_df["filename"].copy()
        table_df = table_df.drop(columns=["filename"])

        table_df = table_df.groupby(filename_column).agg(self.data_model).reset_index()

        return table_df, text_df, image_df

    def build(self, pdf_path):
        table_df, text_df, image_df = self.segregate_metadata()

        table_df = self.table_summarization(table_df=table_df)

        text_df = text_df.loc[:, ~text_df.columns.duplicated()]
        table_df = table_df.loc[:, ~table_df.columns.duplicated()]

        df = pd.concat([text_df, table_df, image_df], axis=0, ignore_index=True)

        if "label" not in df.columns:
            df["label"] = pd.NA

        # Ensure page_number is a valid integer or NaN
        df['page_number'] = pd.to_numeric(df['page_number'], errors='coerce')

        return df

    def table_summarization(self, table_df):
        table_summarization_template = """
        Summarize the following HTML table. Try to create key-value mappings. Respond in bullet points.
        Do not miss any information. Do not generate text in bold.
        html_table: {text_as_html}
        """
        for i, row in table_df.iterrows():
            formatted_prompt = table_summarization_template.format(text_as_html=row["text_as_html"])
            summary = self.client.text_generation(formatted_prompt, max_new_tokens=150)
            if 'text' not in table_df.columns:
                table_df['text'] = None
            table_df.at[i, 'text'] = summary
        return table_df
