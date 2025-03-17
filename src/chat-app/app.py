import sys, os
from langchain.schema.runnable.config import RunnableConfig
from typing import cast

from dotenv import load_dotenv
load_dotenv()

import chainlit as cl
from chainlit.types import AskFileResponse

# Add the its_a_rag module to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './its_a_rag')))

from assistant import Assistant
from rag_assistant import RagAssistant

# Custom Libraries
from its_a_rag.doc_intelligence import AzureAIDocumentIntelligenceLoader, AzureAIDocumentIntelligenceParser
from its_a_rag.ingestion import *


from azure.ai.documentintelligence.models import DocumentAnalysisFeature, AnalyzeDocumentRequest
from azure.ai.documentintelligence.models import DocumentContentFormat

key_credential = os.environ["AZURE_SEARCH_ADMIN_KEY"] if len(os.environ["AZURE_SEARCH_ADMIN_KEY"]) > 0 else None



@cl.set_starters
async def set_starters():
    return [
        cl.Starter(
            label="Google revenue 2000",
            message="What are the revenues of GOOGLE in the year 2000?",
            ),
        cl.Starter(
            label="Revenue and operating margins",
            message="What are the revenues and the operative margins of ALPHABET Inc. in 2022 and how it compares with the previous year?",
            ),
        cl.Starter(
            label="FY23 highlights",
            message="Can you give me the Fiscal Year 2023 Highlights for APPLE, MICROSOFT, NVIDIA and GOOGLE?",
            ),
        cl.Starter(
            label="Stocks on 23/07/2024",
            message="What was the price of APPLE, NVIDIA and MICROSOFT stock in 23/07/2024?",
            )
        ]

@cl.on_chat_start
async def on_chat_start():
    cl.user_session.set("assistant", Assistant())
    cl.user_session.set("ragassistant", RagAssistant())

@cl.on_message
async def on_message(message: cl.Message):

    if not message.elements:
        await cl.Message(content="No file attached").send()

        # print('cast assistant')
        # assistant = cast(Assistant, cl.user_session.get("assistant"))  # type: Assistant

        # msg = cl.Message(content="")

        # async for chunk in assistant.astream(
        #     message.content,
        #     config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        # ):
        #     await msg.stream_token(chunk)

        # await msg.send()
        # return
    

        print('cast RAG assistant')
        assistant = cast(RagAssistant, cl.user_session.get("ragassistant"))  # type: Assistant

        msg = cl.Message(content="")

        async for chunk in assistant.astream(
            message.content,
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk)

        await msg.send()    


    # Filter and process images
    files = [file for file in message.elements]
    if not files:
        await cl.Message(content="No file attached").send()
        return
    else:
        print('files')
        print(files)


        print(os.getenv("AZURE_SEARCH_INDEX"))
        print(os.getenv("AZURE_OPENAI_API_KEY"))
        print(os.getenv("AZURE_OPENAI_ENDPOINT"))
        print(os.getenv("AZURE_OPENAI_API_VERSION"))
        print(os.getenv("AZURE_OPENAI_EMBEDDING"))
        print(os.getenv("AZURE_SEARCH_ENDPOINT"))

        # Create the index for Azure Search store and Embedding
        vector_store_multimodal, aoai_embeddings = create_multimodal_vector_store(os.getenv("AZURE_SEARCH_INDEX"), 
                                                                                os.getenv("AZURE_OPENAI_API_KEY"), 
                                                                                os.getenv("AZURE_OPENAI_ENDPOINT"),
                                                                                os.getenv("AZURE_OPENAI_API_VERSION"),
                                                                                os.getenv("AZURE_OPENAI_EMBEDDING"), 
                                                                                os.getenv("AZURE_SEARCH_ENDPOINT"), 
                                                                                key_credential)

        # For each file
        for file in files:
            # Get the file name
            pdf_file_name = file.path
            # Index : Load the file and create a document
            print("Processing: ", file.path)
            loader = AzureAIDocumentIntelligenceLoader(file_path=file.path, 
                                                api_key = os.getenv("DOCUMENT_INTELLIGENCE_API_KEY"), 
                                                api_endpoint = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT"),
                                                api_model="prebuilt-layout",
                                                api_version=os.getenv("DOCUMENT_INTELLIGENCE_VERSION"),
                                                analysis_features = [DocumentAnalysisFeature.OCR_HIGH_RESOLUTION])
            docs = loader.load()
            # Index : Split
            print('go for TEXT SPLITTER')
            docs = advanced_text_splitter(docs,pdf_file_name)
            # Index : Store
            print(docs)
            print('STORE DOC')
            vector_store_multimodal.add_documents(documents=docs)
            print('store DONE')



        print('cast RAG assistant')
        assistant = cast(RagAssistant, cl.user_session.get("ragassistant"))  # type: Assistant

        msg = cl.Message(content="")

        async for chunk in assistant.astream(
            message.content,
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk)

        await msg.send()
