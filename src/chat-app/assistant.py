import os
from dotenv import load_dotenv
load_dotenv()



from langchain_openai import AzureOpenAIEmbeddings, AzureChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_community.vectorstores.azuresearch import AzureSearch

# Custom Libraries
from its_a_rag.doc_intelligence import AzureAIDocumentIntelligenceLoader, AzureAIDocumentIntelligenceParser
from its_a_rag.ingestion import *

key_credential = os.environ["AZURE_SEARCH_ADMIN_KEY"] if len(os.environ["AZURE_SEARCH_ADMIN_KEY"]) > 0 else None


class Assistant:
    def __init__(self):
        model = AzureChatOpenAI(
                streaming=True,
                api_version=os.getenv('AZURE_OPENAI_API_VERSION'),
                deployment_name=os.getenv('AZURE_OPENAI_DEPLOYMENT_NAME')
                )
        
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


        vector_store = AzureSearch (
            azure_search_endpoint=os.getenv("AZURE_SEARCH_ENDPOINT"),
            azure_search_key=key_credential,
            index_name=os.getenv("AZURE_SEARCH_INDEX"),
            embedding_function=embeddings.embed_query,
            fields=fields,
            # Configure max retries for the Azure client
            additional_search_client_options={"retry_total": 4},
        )
        retriever = vector_store.as_retriever(
            search_type="similarity_score_threshold", 
            search_kwargs={"score_threshold": 0.5}
        )
        llm = AzureChatOpenAI(
            azure_deployment=os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME"),
            api_key=os.getenv("AZURE_OPENAI_API_KEY"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            temperature=0,
            max_retries=2
        )

        prompt = ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    "You're a very knowledgeable financial analyst who provides accurate and eloquent answers to financial questions.",
                ),
                ("human", "{question}"),
            ]
        )
        self.runnable = prompt | model | StrOutputParser()



        # def format_docs(docs):
        #     print('JOIN PAGE CONTENT')
        #     return "\n\n".join(doc.page_content for doc in docs)
        
        # # Use the ChatPromptTemplate to define the prompt that will be sent to the model (Human) remember to include the question and the context
        # prompt = ChatPromptTemplate.from_messages([
        #     ("system", "You are an assistant for question-answering tasks. Use the following pieces of retrieved context to answer the question. If you don't know the answer, just say that you don't know. Use three sentences maximum and keep the answer concise. "),
        #     ("human", "Question: {question}"),
        #     ("system", "Context: {context} Answer:"),
        #     ]
        # )
        # # Define the Chain to get the answer
        # self.runnable = (
        #     {'context': retriever | format_docs }
        #     | prompt
        #     | llm
        #     | StrOutputParser()
        # )
        
    def astream(self, content, config):
        return self.runnable.astream({ "question": content }, config)

    def invoke(self, content):
        return self.runnable.invoke({ "question": content })