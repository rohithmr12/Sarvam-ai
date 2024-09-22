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
        self.client= InferenceClient(
                    "meta-llama/Meta-Llama-3.1-8B-Instruct",
                    token="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",
                    )



    def segregate_metadata(self):
        text_df = self.metadata[(self.metadata["type"] != "Table") & (self.metadata["type"] != "TableChunk") & (self.metadata["type"] != "Image")]
        table_df = self.metadata[(self.metadata["type"] == "Table") | (self.metadata["type"] == "TableChunk")]
        image_df = self.metadata[self.metadata["type"] == "Image"]

        # Drop filename before aggregation to avoid conflicts
        filename_column = table_df["filename"].copy()
        table_df = table_df.drop(columns=["filename"])

        # Perform aggregation and then reattach the filename
        table_df = table_df.groupby(filename_column).agg(self.data_model).reset_index()

        return table_df, text_df, image_df





    @staticmethod
    def convert_pdf_page_to_image(pdf_document, page_number):
        # Load the page
        page = pdf_document.load_page(page_number)

        # Convert the page to a Pixmap
        pixmap = page.get_pixmap()

        # Convert the Pixmap to a PIL Image
        image = Image.open(BytesIO(pixmap.tobytes("png")))

        # Save the PIL Image to a BytesIO buffer in PNG format
        buffer = BytesIO()
        image.save(buffer, format="PNG")

        # Encode the image as base64
        base64_image = base64.b64encode(buffer.getvalue()).decode('utf-8')
        return base64_image

    def build(self, files):
        table_df, text_df, image_df = self.segregate_metadata()
        pdf_document = fitz.open(files)
        # Summarize tables only once
        table_df = self.table_summarization(table_df=table_df)

        # Load PDF document
        # Initialize your PDF document object here

        # Summarize images
        # image_df = self.image_summarization(image_df=image_df, pdf_document=pdf_document)

        # Remove any duplicate columns
        text_df = text_df.loc[:, ~text_df.columns.duplicated()]
        table_df = table_df.loc[:, ~table_df.columns.duplicated()]
        image_df = image_df.loc[:, ~image_df.columns.duplicated()]

        # Concatenate dataframes
        df = pd.concat([text_df, table_df], axis=0)
        df = pd.concat([df, image_df], axis=0, ignore_index=True)

        # Ensure "label" column exists without duplication
        if "label" not in df.columns:
            df["label"] = pd.NA

        return None, df

    @staticmethod
    def table_summarization(table_df):
        table_summarization_template = """
        Summarize the following HTML table. Try to create key-value mappings. Respond in bullet points.
        Do not miss any information. Do not generate text in bold.
        html_table: {text_as_html}
        """
        for i, row in table_df.iterrows():
            formatted_prompt = table_summarization_template.format(text_as_html=row["text_as_html"])
            summary = DocumentBuilder.llama_text(message=formatted_prompt)
            if 'text' not in table_df.columns:
                table_df['text'] = None  # Ensure the column exists
            table_df.at[i, 'text'] = summary
        return table_df

    @staticmethod
    def image_summarization(image_df, pdf_document):
        unique_pages = set()  # Use a set to track unique pages
        rows_to_remove = []  # List to store indices of rows to remove

        for i, row in image_df.iterrows():
            page_number = row.get('page_number')

            if page_number not in unique_pages:
                unique_pages.add(page_number)  # Add page number to the set
                base64_image = DocumentBuilder.convert_pdf_page_to_image(pdf_document, page_number - 1)
                image_summary = DocumentBuilder.call_openai_image(base64_image)
                image_df.at[i, 'text'] = image_summary
            else:
                rows_to_remove.append(i)  # Mark row for removal if page number is a duplicate

        # Remove the marked rows from the DataFrame
        image_df = image_df.drop(rows_to_remove).reset_index(drop=True)

        return image_df

    def llama_text(self,message):
        messages=self.client.chat_completion(
            messages=[{"role": "user", "content":message}],
            stream=True,
        )
        return messages.choices[0].delta.content

    @staticmethod
    def convert_base64(image):
        with open(image, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode('utf-8')

    @staticmethod
    def call_openai_image(base64_image):
        from openai import OpenAI
        client = OpenAI()
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Summarize all the images graphs and charts in the  this page. If the image is a graph or a chart, create a summary in bullet points."},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            },
                        },
                    ],
                }
            ],
            max_tokens=250,
        )
        return response.choices[0].message.content