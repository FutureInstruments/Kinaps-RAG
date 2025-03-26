import sys, os
from langchain.schema.runnable.config import RunnableConfig
from typing import cast, Optional
import asyncio
import psycopg2
from urllib.parse import urlparse


from dotenv import load_dotenv
load_dotenv()

import chainlit as cl
from chainlit.types import AskFileResponse
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

from chainlit.data.storage_clients.azure_blob import AzureBlobStorageClient

PASSWORD=os.getenv("AZURE_SQL_ACCESS")
DATABASE=os.getenv("AZURE_SQL_DATABASE")
SERVER=os.getenv("AZURE_SQL_SERVER")
USERNAME=os.getenv("AZURE_SQL_USER")


from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.data.storage_clients.azure_blob import AzureBlobStorageClient
from chainlit.input_widget import Select, Switch, Slider
from chainlit import make_async


from pathlib import Path

from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

async def async_create_data_layer():
    conninfo = os.getenv("AZURE_POSTGRESQL_CONNECTION_STRING")
    print(conninfo)
    # Create async engine
    engine = create_async_engine(conninfo)

    # Execute initialization statements
    # Ref: https://docs.chainlit.io/data-persistence/custom#sql-alchemy-data-layer
    async with engine.begin() as conn:
        await conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS User (
                    "id" UUID PRIMARY KEY,
                    "identifier" TEXT NOT NULL UNIQUE,
                    "metadata" JSONB NOT NULL,
                    "createdAt" TEXT
                );
        """
            )
        )


        await conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS threads (
                    "id" UUID PRIMARY KEY,
                    "createdAt" TEXT,
                    "name" TEXT,
                    "userId" UUID,
                    "userIdentifier" TEXT,
                    "tags" TEXT[],
                    "metadata" JSONB,
                    FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
                );
        """
            )
        )

        await conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS Thread (
                    "id" UUID PRIMARY KEY,
                    "createdAt" TEXT,
                    "name" TEXT,
                    "userId" UUID,
                    "userIdentifier" TEXT,
                    "tags" TEXT[],
                    "metadata" JSONB,
                    FOREIGN KEY ("userId") REFERENCES users("id") ON DELETE CASCADE
                );
        """
            )
        )

        await conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS steps (
                    "id" UUID PRIMARY KEY,
                    "name" TEXT NOT NULL,
                    "type" TEXT NOT NULL,
                    "threadId" UUID NOT NULL,
                    "parentId" UUID,
                    "disableFeedback" BOOLEAN NOT NULL,
                    "streaming" BOOLEAN NOT NULL,
                    "waitForAnswer" BOOLEAN,
                    "isError" BOOLEAN,
                    "metadata" JSONB,
                    "tags" TEXT[],
                    "input" TEXT,
                    "output" TEXT,
                    "createdAt" TEXT,
                    "start" TEXT,
                    "end" TEXT,
                    "generation" JSONB,
                    "showInput" TEXT,
                    "language" TEXT,
                    "indent" INT
                );
        """
            )
        )

        await conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS Step (
                    "id" UUID PRIMARY KEY,
                    "name" TEXT NOT NULL,
                    "type" TEXT NOT NULL,
                    "threadId" UUID NOT NULL,
                    "parentId" UUID,
                    "disableFeedback" BOOLEAN NOT NULL,
                    "streaming" BOOLEAN NOT NULL,
                    "waitForAnswer" BOOLEAN,
                    "isError" BOOLEAN,
                    "metadata" JSONB,
                    "tags" TEXT[],
                    "input" TEXT,
                    "output" TEXT,
                    "createdAt" TEXT,
                    "start" TEXT,
                    "end" TEXT,
                    "generation" JSONB,
                    "showInput" TEXT,
                    "language" TEXT,
                    "indent" INT
                );
        """
            )
        )

        await conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS elements (
                    "id" UUID PRIMARY KEY,
                    "threadId" UUID,
                    "type" TEXT,
                    "url" TEXT,
                    "chainlitKey" TEXT,
                    "name" TEXT NOT NULL,
                    "display" TEXT,
                    "objectKey" TEXT,
                    "size" TEXT,
                    "page" INT,
                    "language" TEXT,
                    "forId" UUID,
                    "mime" TEXT
                );
        """
            )
        )

        await conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS Element (
                    "id" UUID PRIMARY KEY,
                    "threadId" UUID,
                    "type" TEXT,
                    "url" TEXT,
                    "chainlitKey" TEXT,
                    "name" TEXT NOT NULL,
                    "display" TEXT,
                    "objectKey" TEXT,
                    "size" TEXT,
                    "page" INT,
                    "language" TEXT,
                    "forId" UUID,
                    "mime" TEXT
                );
        """
            )
        )

        await conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS feedbacks (
                    "id" UUID PRIMARY KEY,
                    "forId" UUID NOT NULL,
                    "threadId" UUID NOT NULL,
                    "value" INT NOT NULL,
                    "comment" TEXT
                );
        """
            )
        )
        
        await conn.execute(
            text(
                """
                CREATE TABLE IF NOT EXISTS Feedback (
                    "id" UUID PRIMARY KEY,
                    "forId" UUID NOT NULL,
                    "threadId" UUID NOT NULL,
                    "value" INT NOT NULL,
                    "comment" TEXT
                );
        """
            )
        )


def create_data_layer():
    result = urlparse(os.getenv("AZURE_POSTGRESQL_CONNECTION_STRING"))
    username = result.username
    password = result.password
    database = result.path[1:]
    hostname = result.hostname
    port = result.port
    conn = None
    try:
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(
            host = hostname,
            dbname = database,
            user = username,
            password = password,
            port = port
        )
        conn.set_session(autocommit=True)
        # Creating a cursor with name cur.
        cur = conn.cursor()
        print('Connected to the PostgreSQL database')
        
        # Execute a query:
        # To display the PostgreSQL 
        # database server version
        cur.execute(open("schema.sql", "r").read())
        
        # Close the connection
        cur.close()
        
    except(Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')    

def update_user_setting():
    result = urlparse(os.getenv("AZURE_POSTGRESQL_CONNECTION_STRING"))
    username = result.username
    password = result.password
    database = result.path[1:]
    hostname = result.hostname
    port = result.port
    app_user = cl.user_session.get("user")
    settings = cl.user_session.get("settings")
    metadata_json = json.loads(json.dumps(app_user.metadata))
    metadata_json["per_thread_indexes"] = settings["per_thread_indexes"]
    metadata = json.dumps(metadata_json)

    conn = None
    try:
        # connect to the PostgreSQL server
        print('Connecting to the PostgreSQL database...')
        conn = psycopg2.connect(
            host = hostname,
            dbname = database,
            user = username,
            password = password,
            port = port
        )
        cursor = conn.cursor()
        cursor.execute("""
        UPDATE "User" SET metadata=%s WHERE id = %s
        """, 
        (metadata,app_user.id));
        conn.commit()

    except(Exception, psycopg2.DatabaseError) as error:
        print(error)
    finally:
        if conn is not None:
            conn.close()
            print('Database connection closed.')    


print('CREATE DATA LAYER')
print(os.getenv("DATABASE_URL"))
print(os.getenv("AZURE_POSTGRESQL_CONNECTION_STRING"))
create_data_layer()
print('CREATE DATA LAYER DONE')

# storage_client = AzureBlobStorageClient(
#     container_name=os.getenv("AZURE_BLOB_CONTAINER_NAME"),
#     storage_account=os.getenv("AZURE_STORAGE_ACCOUNT"),
#     storage_key=os.getenv("AZURE_BLOB_STORAGE_KEY"),
# )

# @cl.data_layer
# def get_data_layer():
#     print('data layer')
#     print(os.getenv("ASYNC_DATABASE_URL"))
#     return SQLAlchemyDataLayer(conninfo=os.getenv("ASYNC_DATABASE_URL"), storage_provider=storage_client)

# Add the its_a_rag module to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './its_a_rag')))

# from assistant import Assistant
from rag_assistant import RagAssistant

# Custom Libraries
from its_a_rag.doc_intelligence import AzureAIDocumentIntelligenceLoader, AzureAIDocumentIntelligenceParser
from its_a_rag.ingestion import *

from azure.ai.documentintelligence.models import DocumentAnalysisFeature, AnalyzeDocumentRequest
from azure.ai.documentintelligence.models import DocumentContentFormat
import time

key_credential = os.environ["AZURE_SEARCH_ADMIN_KEY"] if len(os.environ["AZURE_SEARCH_ADMIN_KEY"]) > 0 else None

user_id = ''
index_prefix = 'first-'

def get_current_chainlit_thread_id() -> str:
    return cl.context.session.thread_id


@cl.password_auth_callback
def auth_callback(username: str, password: str)-> Optional[cl.User]:
    # Fetch the user matching username from your database
    # and compare the hashed password with the value stored in the database
    if (username, password) == ("admin", "adminBRAIN2025"):
            print(f"Authenticated user {username}")
            return cl.User(
            identifier="admin", metadata={"role": "admin", "provider": "credentials"})
    else:
        if (username, password) == ("user1", "user1BRAIN2025"):
                print(f"Authenticated user {username}")
                return cl.User(
                identifier="user1", metadata={"role": "user", "provider": "credentials"})
        else:
            if (username, password) == ("remy", "remyBRAIN2025"):
                    print(f"Authenticated user {username}")
                    return cl.User(
                    identifier="remy", metadata={"role": "user", "provider": "credentials", "per_thread_indexes": False})
            else:
                return None
        
    


@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("settings", settings)
    update_user_setting()

@cl.on_chat_resume
async def on_chat_resume(thread):
    global index_prefix
    global user_id

    print('Define SETTING')
    
    app_user = cl.user_session.get("user")
    print('APP USER  ----')
    print(app_user)
    metadata_json = json.loads(json.dumps(app_user.metadata))
    print('METADATA')
    print(metadata_json)
    if "per_thread_indexes" in metadata_json:
        settings = await cl.ChatSettings(
        [
                Switch(id="per_thread_indexes", label="Per Thread indexes", initial=metadata_json["per_thread_indexes"]),
        ]
        ).send()

    print('THREAD')
    chainlit_thread_id = thread.get("id")
    print(chainlit_thread_id)
    app_user = cl.user_session.get("user")
    user_id = app_user.identifier    
    print('APP USER  ----')
    print(app_user)
    metadata_json = json.loads(json.dumps(app_user.metadata))
    if metadata_json["per_thread_indexes"] == True:
        index_prefix = user_id+'-'+chainlit_thread_id+'-'
    else:
        index_prefix = user_id+'-'
    print('set RagAssistant in SESSION')
    cl.user_session.set("ragassistant", RagAssistant(index_prefix))
    print('set RagAssistant in SESSION DONE')
    pass

@cl.on_chat_start
async def on_chat_start():
    global user_id 
    global index_prefix

    print('Define SETTING')
    
    app_user = cl.user_session.get("user")
    print('APP USER  ----')
    print(app_user)
    metadata_json = json.loads(json.dumps(app_user.metadata))
    print('METADATA')
    print(metadata_json)
    if "per_thread_indexes" in metadata_json:
        settings = await cl.ChatSettings(
        [
                Switch(id="per_thread_indexes", label="Per Thread indexes", initial=metadata_json["per_thread_indexes"]),
        ]
        ).send()

    cl.user_session.set("settings", settings)
    set = cl.user_session.get("settings")
    print('Setting SETTING')
    print(set)

    print("The chat session has started!")
    await cl.Message(f"Hello {app_user.identifier}").send()
    user_id = app_user.identifier


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



@cl.on_message
async def on_message(message: cl.Message):

    global user_id 
    global index_prefix
    print('STORE ASSISTANT IN SESSION')
    chainlit_thread_id = get_current_chainlit_thread_id()
    print('Thread ID')
    print(chainlit_thread_id)
    app_user = cl.user_session.get("user")
    print('APP USER  ----')
    print(app_user)
    metadata_json = json.loads(json.dumps(app_user.metadata))
    if metadata_json["per_thread_indexes"] == True:
        index_prefix = user_id+'-'+chainlit_thread_id+'-'
    else:
        index_prefix = user_id+'-'
    print('set RagAssistant in SESSION')
    cl.user_session.set("ragassistant", RagAssistant(index_prefix))
    print('set RagAssistant in SESSION DONE')

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
        assistant = cast(RagAssistant, cl.user_session.get("ragassistant"))  # type: RagAssistant

        msg = cl.Message(content="")

        async for chunk in assistant.astream(
            message.content,
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk)

        await msg.send()

    else:
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
        vector_store_multimodal, aoai_embeddings = create_multimodal_vector_store(index_prefix+os.getenv("AZURE_SEARCH_INDEX"), 
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
            print('LOADER init')
            start_time = time.time()
            loader = AzureAIDocumentIntelligenceLoader(file_path=file.path, 
                                                api_key = os.getenv("DOCUMENT_INTELLIGENCE_API_KEY"), 
                                                api_endpoint = os.getenv("DOCUMENT_INTELLIGENCE_ENDPOINT"),
                                                api_model="prebuilt-layout",
                                                api_version=os.getenv("DOCUMENT_INTELLIGENCE_VERSION"),
                                                analysis_features = [DocumentAnalysisFeature.OCR_HIGH_RESOLUTION])
            print('END OF LOADER init')
            print("--- %s seconds ---" % (time.time() - start_time))
            
            print('LOADER load')
            start_time = time.time()
            # docs = loader.load()
            async_function = make_async(load_doc)

            docs = await async_function(loader)
            print('END OF LOADER load')
            print("--- %s seconds ---" % (time.time() - start_time))

            # Index : Split
            print('go for TEXT SPLITTER')
            start_time = time.time()        
            docs = advanced_text_splitter(docs,pdf_file_name)
            print('END OF TEXT SPLITTER')
            print("--- %s seconds ---" % (time.time() - start_time))   

            # Index : Store
            # print(docs)
            print('STORE DOC')
            start_time = time.time()
            vector_store_multimodal.add_documents(documents=docs)
            print("--- %s seconds ---" % (time.time() - start_time))

            print('store DONE')

        print('cast RAG assistant')
        assistant = cast(RagAssistant, cl.user_session.get("ragassistant"))  # type: RagAssistant

        msg = cl.Message(content="")

        async for chunk in assistant.astream(
            message.content,
            config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        ):
            await msg.stream_token(chunk)

        await msg.send()


def load_doc(loader):

    docs = loader.load()

    return docs