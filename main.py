from Extractor import DocumentExtractor
from Builder import DocumentBuilder
from Indexer import IndexHandler
import pandas as pd

def process_document(filename, db_url="./.lancedb"):
    # extractor = DocumentExtractor(filename=filename)
    # df_metadata = extractor.extract_metadata()

    # if df_metadata.empty:
    #     print(f"No valid metadata extracted from {filename}")
    #     return None

    # builder = DocumentBuilder(metadata=df_metadata)
    # df_enriched = builder.build(filename)
    # df_enriched = df_enriched.sort_values(by="page_number")
    # df_enriched.to_csv("df_enriched.csv")
    df_enriched=pd.read_csv("df_enriched.csv")
    index_handler = IndexHandler(metadata=df_enriched, db_url=db_url)
    index_handler.add_documents_to_lancedb_table(df_enriched)

    return index_handler