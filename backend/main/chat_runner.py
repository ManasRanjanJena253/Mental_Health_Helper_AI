import os
import chromadb
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Setting the verbose = True, to see the thinking of LLMs
#set_verbose(True)

class RunModel:
    def __init__(self, db_name: str, api_key = os.getenv("GOOGLE_API_KEY"), model_name: str = "gemini-2.5-flash", temperature: int = 0.3):
        """
        Initializes the chat model running class.
        :param api_key: The api key of the model being used.
        :param model_name: The name of the model being used.
        :param temperature: The temperature of the model (It controls the creativity of model, too high value will result in gibberish).
        :param db_name: The name of the chromadb memory_collection based on the user.
        """
        # Loading the llm
        self.__api_key = api_key
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model = "gemini-embedding-001", google_api_key = self.__api_key
        )
        # Creating chroma client.
        self.chroma_memory_client = chromadb.PersistentClient(path ="../chroma")
        self.memory_collection = self.chroma_memory_client.get_or_create_collection(name = db_name.replace(" ", "").lower())  # Handling the spaces in names.

        self.chroma_taboo_client = chromadb.PersistentClient(path = "backend/vector_embeddings/Mental_Health_Taboos_In_India")
        self.taboo_collection = self.chroma_memory_client.get_or_create_collection(name ="mentalhealthtaboosinindia")

        self.indian_taboo_vector_store = Chroma(collection_name = "mentalhealthtaboosinindia",
                                                persist_directory = "backend/vector_embeddings/Mental_Health_Taboos_In_India",
                                                embedding_function = self.embeddings)

        self.remedies_vector_store = Chroma(collection_name = "mentalhealthremedies",
                                            persist_directory = "backend/vector_embeddings/Mental_Health_Remedies",
                                            embedding_function = self.embeddings)

        self.remedies_client = chromadb.PersistentClient(path = "backend/vector_embeddings/Mental_Health_Remedies")
        self.remedies_collection = self.remedies_client.get_or_create_collection(name = "mentalhealthremedies")

        try:
            self.llm = ChatGoogleGenerativeAI(model = model_name,
                                         google_api_key = self.__api_key,
                                         verbose = True,
                                         temperature = temperature)
        except Exception as e:
            print(f"ERROR : {e}")

    def initiate_run(self, user_prompt: str, session_id: str, user_name: str):
        """
        Used to initiate the first run to create the user memory.
        :param user_prompt: The first prompt given by the user.
        :param session_id: The session of the chat.
        :param user_name: The unique name of the user.
        :return: The first output and the memory.
        """

        remedies_context = self.remedies_vector_store.max_marginal_relevance_search(query = user_prompt,
                                                                                    k = 5,
                                                                                    fetch_k = 25)

        taboo_details = self.indian_taboo_vector_store.max_marginal_relevance_search(query = user_prompt,
                                                                                     k = 5,
                                                                                     fetch_k = 25)
        system_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI Therapist in INDIA and you are trying to help your client using your knowledge about the issue that the client is currently facing.\n"
                       "You have to make the client feel accepted and by also debunking the taboos that might be associated regarding the client's mental state or situation\n"
                       "In India. You will provided with the following resources to help you i.e 1. The taboos regarding the clients state in India. 2. The general remedies or"
                       "methods to help on the client based on the knowledge of an actual therapist."
                       "1. The taboos : \n"
                       "{taboo_details}\n\n"
                       "2. The knowledge from the therapist : \n"
                       "{remedies_context}\n\n"
                       "Remember no need to address any of the following except Your past convo directly to the patient all these are for your help and for you to perform better. "
                       "If the user asks anything outside the scope of a therapist, simply just DENY it."),
            ("user", "{query}")
        ])

        main_chain = system_prompt | self.llm | StrOutputParser()

        # Running the full pipeline
        output = main_chain.invoke({
                "taboo_details": taboo_details,
                "remedies_context": remedies_context,
                "query": user_prompt
            })

        # Checking if the model produced any output or not.
        if not output:
            raise ValueError("UNABLE TO GENERATE A REPLY !!! \n Plz Try Again Later.")

        # Adding the session data into the chromadb
        self.memory_collection.add(
            documents=[
                f"User feeling: {user_prompt}\n"
                f"Therapist final response: {output}"
            ],
            metadatas=[{"session_id": session_id, "user_name": user_name}],
            ids=[f"{session_id}"]
        )

        return output

    def run(self, user_prompt, session_id: str, user_name: str):
        """
        The main function for running the whole LLMChain and using it via frontend for the user.
        :param user_prompt: The prompt given by the user.
        :param session_id: The unique id of a chat session.
        :param user_name: The unique name of the user.
        :return: The final answer to the users question or chat discussion.
        """
        retrieved_memory = self.memory_collection.query(
            query_texts = [user_prompt],
            n_results = 5,
            where = {"session_id": {"$eq": session_id}}
        )

        remedies_context = self.remedies_vector_store.max_marginal_relevance_search(query = user_prompt,
                                                                                    k = 5,
                                                                                    fetch_k = 25)

        taboo_details = self.indian_taboo_vector_store.max_marginal_relevance_search(query = user_prompt,
                                                                                     k = 5,
                                                                                     fetch_k = 25)

        # filtering by user_id
        filtered_docs = [
            (doc, meta) for doc, meta in zip(retrieved_memory["documents"][0], retrieved_memory["metadatas"][0])
            if meta.get("user_name") == user_name
        ]

        # Checking if there are any data about previous sessions of the user.
        if not filtered_docs:
            self.initiate_run(user_prompt = user_prompt, session_id = session_id, user_name = user_name)

        system_prompt = ChatPromptTemplate.from_messages([
            ("system", "You are an AI Therapist in INDIA and you are trying to help your client using your knowledge about the issue that the client is currently facing.\n"
                       "You have to make the client feel accepted and by also debunking the taboos that might be associated regarding the client's mental state or situation\n"
                       "In India. You will provided with the following resources to help you i.e 1. The taboos regarding the clients state in India. 2. The general remedies or"
                       "methods to help on the client based on the knowledge of an actual therapist. 3. The details regarding your past conversations with the client for you to refer to them.\n"
                       "1. The taboos : \n"
                       "{taboo_details}\n\n"
                       "2. The knowledge from the therapist : \n"
                       "{remedies_context}\n\n"
                       "3. Your past conversation with the patient : \n"
                       "{retrieved_memory}\n\n"
                       "Remember no need to address any of the following except Your past convo directly to the patient all these are for your help and for you to perform better. "
                       "If the user asks anything outside the scope of a therapist, simply just DENY it."),
            ("user", "{query}")
        ])

        context_str = "\n".join([doc for doc, _ in filtered_docs])

        main_chain = system_prompt | self.llm | StrOutputParser()

        try:
            output = main_chain.invoke({
                "taboo_details": taboo_details,
                "remedies_context": remedies_context,
                "retrieved_memory": context_str,
                "query": user_prompt
            })

        except Exception as e:
            return f"Error : {e}"

        # Checking if the model produced any output or not.
        if not output:
            raise ValueError("UNABLE TO GENERATE A REPLY !!! \n Plz Try Again Later.")

        self.memory_collection.add(
            documents=[
                f"User feeling: {user_prompt}\n"
                f"Therapist final response: {output}"
            ],
            metadatas=[{"session_id": session_id, "user_name": user_name}],
            ids=[f"{session_id}"]
        )

        return output