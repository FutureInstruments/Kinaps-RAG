
import base64
import sys, os
import uuid

from pathlib import Path

# Add the its_a_rag module to the Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), './its_a_rag')))

from dotenv import load_dotenv
load_dotenv()

from langchain.schema.runnable.config import RunnableConfig
from langchain_core.runnables import Runnable, RunnablePassthrough, RunnableLambda
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema.output_parser import StrOutputParser
from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI

from langchain.memory import ConversationBufferMemory

from operator import itemgetter

from typing import cast, Optional
import asyncio
from asyncio import sleep

import psycopg2
from urllib.parse import urlparse
from tqdm import tqdm



import chainlit as cl
from chainlit.types import AskFileResponse
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer

from chainlit.data.storage_clients.azure_blob import AzureBlobStorageClient
from chainlit.data.sql_alchemy import SQLAlchemyDataLayer
from chainlit.data.storage_clients.azure_blob import AzureBlobStorageClient
from chainlit.input_widget import Select, Switch, Slider
from chainlit import make_async
from chainlit.types import ThreadDict


from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# from assistant import Assistant
from rag_assistant import RagAssistant

# Custom Libraries
from its_a_rag.doc_intelligence import AzureAIDocumentIntelligenceLoader, AzureAIDocumentIntelligenceParser
from its_a_rag.ingestion import *

from azure.ai.documentintelligence.models import DocumentAnalysisFeature, AnalyzeDocumentRequest
from azure.ai.documentintelligence.models import DocumentContentFormat
import time

from azure.search.documents import SearchIndexingBufferedSender  
from azure.core.exceptions import HttpResponseError

PASSWORD=os.getenv("AZURE_SQL_ACCESS")
DATABASE=os.getenv("AZURE_SQL_DATABASE")
SERVER=os.getenv("AZURE_SQL_SERVER")
USERNAME=os.getenv("AZURE_SQL_USER")


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

max_doc_upload = int(os.getenv("AZURE_MAX_DOC_UPLOAD"))

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



key_credential = os.environ["AZURE_SEARCH_ADMIN_KEY"] if len(os.environ["AZURE_SEARCH_ADMIN_KEY"]) > 0 else None

user_id = ''
index_prefix = 'first-'

print('Init LLM')
model = AzureChatOpenAI(
    azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
    api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
    temperature=0,
    max_retries=2,
    streaming=True,
    max_tokens=None,
    logprobs=True
)

print('init embendding')
print(os.getenv("AZURE_OPENAI_EMBEDDING"))
print(os.getenv("AZURE_OPENAI_API_VERSION"))
print(os.getenv("AZURE_OPENAI_ENDPOINT"))
print(os.getenv("AZURE_OPENAI_API_KEY"))

embeddings = AzureOpenAIEmbeddings(
    model = os.getenv("AZURE_OPENAI_EMBEDDING"),
    openai_api_version = os.getenv("AZURE_OPENAI_API_VERSION"),
    azure_endpoint = os.getenv("AZURE_OPENAI_ENDPOINT"),
    api_key = os.getenv("AZURE_OPENAI_API_KEY")
)

# Create Additional Fields for the Azure Search Index    
embedding_function = embeddings.embed_query
fields = [
    SimpleField(
        name="id",
        type=SearchFieldDataType.String,
        key=True,
        filterable=True,
    ),
    SearchableField(
        name="content",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    SearchField(
        name="content_vector",
        type=SearchFieldDataType.Collection(SearchFieldDataType.Single),
        searchable=True,
        vector_search_dimensions=len(embedding_function("Text")),
        vector_search_profile_name="myHnswProfile",
    ),
    SearchableField(
        name="metadata",
        type=SearchFieldDataType.String,
        searchable=False,
    ),
    # Additional field to store the title
    SearchableField(
        name="header",
        type=SearchFieldDataType.String,
        searchable=True,
    ),
    # Additional field for filtering on document source
    SimpleField(
        name="source",
        type=SearchFieldDataType.String,
        filterable=True,
        searchable=False,
    ),
    # Additional field for filtering on document source
    SimpleField(
        name="image",
        type=SearchFieldDataType.String,
        filterable=False,
        searchable=False,
    ),
]

def setup_runnable():
    global model
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory
    # model = ChatOpenAI(streaming=True)
    print("prepare PROMPT")
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "You are an assistant for question-answering tasks. Use only and exclusively the following pieces of retrieved context to answer the question. You need to analyze the context to be sure it's relevant. You can use history provided to refine the answer by using the history provided. If the answer cannot be deduced only from the retrieved context say that you don't know. if the context is empty or if the context is not relevant for the question say that the context is not good, forget the context and just say that you don't know. Use seven sentences maximum and keep the answer concise. At the end you can repeat the context you received if and only if the human ask for it in the question but don't propose to give the context."), # You are a helpful chatbot. You need to use the following pieces of retrieved context to answer. At the end you need to repeat the context you received."),
            MessagesPlaceholder(variable_name="history"),
            ("system", "context: {context}"),
            ("human", "{question}"),
        ]
    )
    print("Prepare Runnable")
    runnable = (
        RunnablePassthrough.assign(
            history=RunnableLambda(memory.load_memory_variables) | itemgetter("history")
        )
        | prompt
        | model
        | StrOutputParser()
    )
    cl.user_session.set("runnable", runnable)


def get_current_chainlit_thread_id() -> str:
    return cl.context.session.thread_id

async def send_animated_message(
    base_msg: str,
    end_msg: str,
    frames: List[str],
    interval: float = 0.8
) -> None:
    """Display animated message with minimal resource usage"""
    msg = cl.Message(content=base_msg)
    await msg.send()
    
    progress = 0
    bar_length = 10  # Optimal length for progress bar
    
    try:
        while True:
            # Efficient progress calculation
            current_frame = frames[progress % len(frames)]
            progress_bar = ("▣" * (progress % bar_length)).ljust(bar_length, "▢")
            
            # Single update operation
            msg.content = f"{current_frame} {base_msg}\n{progress_bar}"
            await msg.update()
            
            progress += 1
            await asyncio.sleep(interval)
    except asyncio.CancelledError:
        msg.content = end_msg
        await msg.update()  # Final static message

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
                if (username, password) == ("uspi", "uspiBRAIN2025"):
                        print(f"Authenticated user {username}")
                        return cl.User(
                        identifier="uspi", metadata={"role": "user", "provider": "credentials"})
                else:
                    if (username, password) == ("gastrovd", "gastrovdBRAIN2025"):
                            print(f"Authenticated user {username}")
                            return cl.User(
                            identifier="gastrovd", metadata={"role": "user", "provider": "credentials"})
                    else:
                        if (username, password) == ("cifer", "ciferBRAIN2025"):
                                print(f"Authenticated user {username}")
                                return cl.User(
                                identifier="cifer", metadata={"role": "user", "provider": "credentials"})
                        else:
                            if (username, password) == ("dho", "dhoBRAIN2025"):
                                    print(f"Authenticated user {username}")
                                    return cl.User(
                                    identifier="dho", metadata={"role": "user", "provider": "credentials"})
                            else:
                                if (username, password) == ("hevs", "hevsBRAIN2025"):
                                        print(f"Authenticated user {username}")
                                        return cl.User(
                                        identifier="hevs", metadata={"role": "user", "provider": "credentials"})
                                else:
                                    if (username, password) == ("dta.hevs", "vKam4yqEj+etP@2D"):
                                            print(f"Authenticated user {username}")
                                            return cl.User(
                                            identifier="dta.hevs", metadata={"role": "user", "provider": "credentials"})
                                    else:
                                        if (username, password) == ("marian.hevs", "J@=7g!3w%CV&ncZe"):
                                                print(f"Authenticated user {username}")
                                                return cl.User(
                                                identifier="marian.hevs", metadata={"role": "user", "provider": "credentials"})
                                        else:
                                            if (username, password) == ("loic.hevs", "yhBuj&7W46bfQDF3"):
                                                    print(f"Authenticated user {username}")
                                                    return cl.User(
                                                    identifier="loic.hevs", metadata={"role": "user", "provider": "credentials"})
                                            else:
                                                if (username, password) == ("filipe.hevs", "hDVF@2u?gf&8=+QS"):
                                                        print(f"Authenticated user {username}")
                                                        return cl.User(
                                                        identifier="filipe.hevs", metadata={"role": "user", "provider": "credentials"})
                                                else:
                                                    if (username, password) == ("antoine.hevs", "C+&2=rN?jB7SneVZ"):
                                                            print(f"Authenticated user {username}")
                                                            return cl.User(
                                                            identifier="antoine.hevs", metadata={"role": "user", "provider": "credentials"})
                                                    else:
                                                        return None

@cl.on_settings_update
async def setup_agent(settings):
    cl.user_session.set("settings", settings)
    update_user_setting()

@cl.on_chat_resume
async def on_chat_resume(thread: ThreadDict):
    global index_prefix
    global user_id

    memory = ConversationBufferMemory(return_messages=True)

    print('GETTING THREAD MESSAGE')
    print('----')
    print(thread)
    print('-----------------')
    print('Define SETTING')
    
    app_user = cl.user_session.get("user")
    print('APP USER  ----')
    print(app_user)
    user_id = str(app_user.identifier).replace('.','-')
    metadata_json = json.loads(json.dumps(app_user.metadata))
    print('METADATA')
    print(metadata_json)
    if "per_thread_indexes" in metadata_json:
        settings = await cl.ChatSettings(
        [
                Switch(id="per_thread_indexes", label="Per Thread indexes", initial=metadata_json["per_thread_indexes"]),
        ]
        ).send()
        if metadata_json["per_thread_indexes"] == True:
            index_prefix = user_id+'-'+chainlit_thread_id+'-'
        else:
            index_prefix = user_id+'-'
    else:
        index_prefix = user_id+'-'


    print('THREAD')
    chainlit_thread_id = thread.get("id")
    print(chainlit_thread_id)
    app_user = cl.user_session.get("user")
    user_id = str(app_user.identifier).replace('.','-')
    print('APP USER  ----')
    print(app_user)

    print("loading up memory")
    root_messages = [m for m in thread["steps"] if m["parentId"] == None]
    for message in root_messages:
        if message["type"] == "user_message":
            memory.chat_memory.add_user_message(message["output"])
        else:
            memory.chat_memory.add_ai_message(message["output"])

    cl.user_session.set("memory", memory)
    setup_runnable()

    print('set RagAssistant in SESSION + memory')
    cl.user_session.set("ragassistant", RagAssistant(index_prefix, memory))
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

    print("The chat session has started!")

    # image = cl.Image(path="./cat.jpg", name="cat image", display="inline")

    # await cl.Message(
    #     # Notice that the name of the image is NOT referenced in the message content
    #     content="Hello! this is a cat",
    #     elements=[image],
    # ).send()

    #     # Sending a pdf with the local file path
    # pdfelement = [
    #   cl.Pdf(name="pdf1", display="inline", path="./pdfname.pdf", page=1)
    # ]

    # await cl.Message(content="Look at this local pdf!", elements=pdfelement).send()

    print("setting memory in session")
    cl.user_session.set("memory", ConversationBufferMemory(return_messages=True))

    await cl.Message(f"Hello {app_user.identifier}").send()
    user_id = str(app_user.identifier).replace('.','-')
    setup_runnable()



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
    global animation_task
    global user_id 
    global index_prefix
    print('STORE ASSISTANT IN SESSION')
    chainlit_thread_id = get_current_chainlit_thread_id()
    print('Thread ID')
    print(chainlit_thread_id)
    app_user = cl.user_session.get("user")
    user_id = str(app_user.identifier).replace('.','-')

    print('APP USER  ----')
    print(app_user)
    metadata_json = json.loads(json.dumps(app_user.metadata))
    if "per_thread_indexes" in metadata_json:
        if metadata_json["per_thread_indexes"] == True:
            index_prefix = user_id+'-'+chainlit_thread_id+'-'
        else:
            index_prefix = user_id+'-'
    else:
        index_prefix = user_id+'-'

    print("get memory and runnable from session")
    memory = cl.user_session.get("memory")  # type: ConversationBufferMemory

    runnable = cl.user_session.get("runnable")  # type: Runnable

    print('set RagAssistant in SESSION')
    cl.user_session.set("ragassistant", RagAssistant(index_prefix,memory))
    print('set RagAssistant in SESSION DONE')

    print(os.getenv("AZURE_SEARCH_INDEX"))
    print(os.getenv("AZURE_OPENAI_API_KEY"))
    print(os.getenv("AZURE_OPENAI_ENDPOINT"))
    print(os.getenv("AZURE_OPENAI_API_VERSION"))
    print(os.getenv("AZURE_OPENAI_EMBEDDING"))
    print(os.getenv("AZURE_SEARCH_ENDPOINT"))

    # Create the index for Azure Search store and Embedding
    # vector_store_multimodal, aoai_embeddings = create_multimodal_vector_store(index_prefix+os.getenv("AZURE_SEARCH_INDEX"), 
    #                                                                         os.getenv("AZURE_OPENAI_API_KEY"), 
    #                                                                         os.getenv("AZURE_OPENAI_ENDPOINT"),
    #                                                                         os.getenv("AZURE_OPENAI_API_VERSION"),
    #                                                                         os.getenv("AZURE_OPENAI_EMBEDDING"), 
    #                                                                         os.getenv("AZURE_SEARCH_ENDPOINT"), 
    #                                                                         key_credential)
    
    print('Init AzureSearch')
    print(index_prefix)
    vector_store = AzureSearch (
        azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
        azure_search_key=key_credential,
        index_name=index_prefix+os.getenv("AZURE_SEARCH_INDEX"),
        embedding_function=embeddings.embed_query,
        fields=fields,
        # Configure max retries for the Azure client
        additional_search_client_options={"retry_total": 7},
    )

    print('Init Retriever')
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold", 
        search_kwargs={"score_threshold": 0.5}
    )

    if not message.elements:
        print("Doing question without context")
        # await cl.Message(content="No file attached").send()

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
    

        # print('cast RAG assistant')
        # assistant = cast(RagAssistant, cl.user_session.get("ragassistant"))  # type: RagAssistant

        # msg = cl.Message(content="")
        # # resultinvoke = assistant.invoke(message.content)
        # print('DEBUG DEBUG')
        # # print(resultinvoke)
        # async for chunk in assistant.astream(
        #     message.content,
        #     config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()])

        # ):
        #     await msg.stream_token(chunk)

        # await msg.send()

        print('create annimation task')
        animation_task = asyncio.create_task(
            send_animated_message(
                "Processing ...",
                "Processing Done",
                ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"],
                0.8
            )
        )
        await wait_it(1)
        print("START ANIMATION")
        await animation_task


    else:
        # Filter and process images
        files = [file for file in message.elements]
        if not files:
            await cl.Message(content="No file attached").send()
            return
        else:
            print('files')
            print(files)

        # For each file
        for file in files:
            # Get the file name
            pdf_file_name = file.path
            # Index : Load the file and create a document
            print("Processing: ", file.path)
            print('LOADER init')
            
            print('create annimation task')
            animation_task = asyncio.create_task(
                send_animated_message(
                    "Processing ...",
                    "Processing Done",
                    ["🌑", "🌒", "🌓", "🌔", "🌕", "🌖", "🌗", "🌘"],
                    0.8
                )
            )

            # docs = loader.load()
            async_function = make_async(load_doc)
            docs = await async_function(vector_store, embeddings, pdf_file_name, file)
            
            print("START ANIMATION")   
            await animation_task

            print('store DONE')

        # print('cast RAG assistant with files')
        # assistant = cast(RagAssistant, cl.user_session.get("ragassistant"))  # type: RagAssistant

        # msg = cl.Message(content="")
        # # resultinvoke = assistant.invoke(message.content)
        # print('DEBUG DEBUG')
        # # print(resultinvoke)
        # async for chunk in assistant.astream(
        #     message.content,
        #     config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
        # ):
        #     await msg.stream_token(chunk)

        # await msg.send()

    res = cl.Message(content="")
    retrieve_data = retriever.invoke(input=message.content)
    context = format_docs(retrieve_data)
    print("CONTEXT ---")
    print(context)
    print("---------")

    async for chunk in runnable.astream(
        {"question": message.content, "context": context},
        config=RunnableConfig(callbacks=[cl.LangchainCallbackHandler()]),
    ):
        await res.stream_token(chunk)

    await res.send()

    print("store new message / answer in memory")
    memory.chat_memory.add_user_message(message.content)
    memory.chat_memory.add_ai_message(res.content)
    
    cl.user_session.set("memory", memory)

async def wait_it(second):
    await asyncio.sleep(second)
    print('- END ANIMATION')
    animation_task.cancel()

def format_docs(docs):
    print('JOIN PAGE CONTENT')
    print(doc.page_content for doc in docs)
    print('--- END JOIN ---')
    return "début du context"+"\n\n".join(doc.page_content for doc in docs)+"\n\nfin du context"

def upload_documents(vector_store: AzureSearch, embedder: AzureOpenAIEmbeddings,docs: list[Document]):
    fields = [x.name for x in vector_store.fields]
    print('fields')
    print(fields)
    for i in range(0, len(docs), max_doc_upload):
        print('embedding doc')
        embeddings_batch = embedder.embed_documents([x.page_content for x in docs[i:i+max_doc_upload]])
        docs_batch = []
        for j, doc in enumerate(docs[i:i+max_doc_upload]):
            key = str(uuid.uuid4())
            doc_id = base64.urlsafe_b64encode(bytes(key, "utf-8")).decode("ascii")
            doc_dict = {
                "id": doc_id,
                "content": doc.page_content,
                "content_vector": embeddings_batch[j],
                "metadata": json.dumps(doc.metadata)
            }
            for k, v in doc.metadata.items():
                if k in fields:
                    doc_dict[k] = v
            docs_batch.append(doc_dict)
        print('upload doc iteration '+str(i))
        print(len(docs_batch))
        print(sys.getsizeof(docs_batch))
        print(sys.getsizeof(json.dumps(docs_batch)))
        vector_store.client.upload_documents(docs_batch)
    print("end Uploading ---")

def load_doc(vector_store_multimodal, embedder, pdf_file_name, file):

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
    docs = loader.load()
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
    print('DOCS before upload')
    print(len(docs))
    print(sys.getsizeof(docs))
    docs_dict = [{"page_content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    print(sys.getsizeof(json.dumps(docs_dict)))
    print('STORE DOC')
    start_time = time.time()
    # vector_store_multimodal.add_documents(documents=docs)
    print("start upload documents")
    upload_documents(vector_store_multimodal, embedder, docs)
    print("--- %s seconds ---" % (time.time() - start_time))
    print('END ANIMATION')
    animation_task.cancel()

    return docs
