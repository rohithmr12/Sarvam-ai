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

class DocumentExtractor:
    def __init__(self, filename, output_dir="/doc/"):
        self.filename = filename
        self.output_dir = output_dir
        self.filetype = filename.split('.')[-1].lower()
        self.meta_columns = ["type", "text", "filename", "page_number", "coordinates", "text_as_html"]

    def extract_metadata(self):
        elements = self.extract_elements()
        if not elements:
            return pd.DataFrame(columns=self.meta_columns)

        records = self.convert_to_dict(elements=elements)
        records_df = pd.DataFrame(records)

        for col in self.meta_columns:
            if col not in records_df.columns:
                records_df[col] = None

        metadata = [ele.metadata.to_dict() for ele in elements if hasattr(ele, 'metadata')]
        metadata_df = pd.DataFrame(metadata)

        df = pd.concat([records_df, metadata_df], axis=1)
        df = df.loc[:, ~df.columns.duplicated()]

        return df

    def extract_elements(self):
        if self.filetype == 'pdf':
            return self.extract_pdf()
        else:
            print(f"Unsupported filetype: {self.filetype}")
            return []

    def extract_pdf(self):
        try:
            return partition_pdf(
                filename=self.filename,
                strategy="ocr_only",
                max_characters=2000,
                new_after_n_chars=1500,
                infer_table_structure=True,
                extract_images_in_pdf=False,
            )
        except Exception as e:
            print(f"Error during PDF extraction: {e}")
            return []

    @staticmethod
    def convert_to_dict(elements):
        from unstructured.staging.base import convert_to_dict
        return convert_to_dict(elements=elements)