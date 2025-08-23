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

        self.indian_taboo_vector_store = Chroma(collection_name = "mentalhealthtaboosinindia",
                                                persist_directory = "Mental_Health_Taboos_in_India",
                                                embedding_function = self.embeddings)

        # self.indian_taboo_vector_client = chromadb.PersistentClient(path = "Mental_Health_Taboos_In_India")
        # self.indian_taboo_vector_store = self.indian_taboo_vector_client.get_collection(name = "mentalhealthtaboosinindia")


        self.remedies_vector_store = Chroma(collection_name = "mentalhealthremedies",
                                            persist_directory = "Mental_Health_Remedies",
                                            embedding_function = self.embeddings)

        # self.remedies_vector_client = chromadb.PersistentClient(path = "Mental_Health_Remedies")
        # self.remedies_vector_store = self.remedies_vector_client.get_collection(name = "mentalhealthremedies")

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

        remedies_retrieved = self.remedies_vector_store.max_marginal_relevance_search(query = user_prompt,
                                                                                    k = 3,
                                                                                    fetch_k = 25)
        remedies_context = "\n\n".join(doc.page_content for doc in remedies_retrieved)

        # remedies_context = self.remedies_vector_store.query(query_texts = [user_prompt],
        #                                                     n_results = 5)

        taboo_retrieved = self.indian_taboo_vector_store.max_marginal_relevance_search(query = user_prompt,
                                                                                     k = 3,
                                                                                     fetch_k = 25)

        taboo_details = "\n\n".join(doc.page_content for doc in taboo_retrieved)

        # taboo_details = self.indian_taboo_vector_store.query(query_texts = [user_prompt],
        #                                                      n_results = 5)

        system_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "Everytime you get a query warmly greet with Namaste"
             "You are a compassionate therapist in INDIA. "
             "Your role is to support the client with empathy, understanding, and practical advice. "
             "You have access to background knowledge about: "
             "1) Common taboos around mental health in India. "
             "2) Remedies and therapeutic practices. "
             "Use these only as background to inform your answers — do not list them directly. "

             "When responding: "
             "- Speak warmly and conversationally, like in a private session. "
             "- Validate feelings sometimes, but vary your words so it doesn’t sound repetitive. "
             "- Offer small, practical suggestions the client can try. "
             "- If relevant, gently mention cultural taboos ,The taboos:\n{taboo_details}\n\n"
             "to reassure the client they are not alone — but keep it subtle. "
             "- Balance empathy, encouragement, and curiosity; don’t always end with a question. "
             "- If asked something outside the scope of therapy, politely refuse. "

             "Background context:\n"

             "1. The therapist’s knowledge:\n{remedies_context}\n\n"
             "All these contexts are only for making your answers more reliable, you still need to change your vocab to be soothing and not just be a bookish informatic model."),
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
                                                                                    k = 3,
                                                                                    fetch_k = 25)

        # remedies_context = self.remedies_vector_store.query(query_texts = [user_prompt],
        #                                                     n_results = 5)

        taboo_details = self.indian_taboo_vector_store.max_marginal_relevance_search(query = user_prompt,
                                                                                     k = 3,
                                                                                     fetch_k = 25)

        # taboo_details = self.indian_taboo_vector_store.query(query_texts = [user_prompt],
        #                                                      n_results = 5)
        # filtering by user_id
        filtered_docs = [
            (doc, meta) for doc, meta in zip(retrieved_memory["documents"][0], retrieved_memory["metadatas"][0])
            if meta.get("user_name") == user_name
        ]

        # Checking if there are any data about previous sessions of the user.
        if not filtered_docs:
            self.initiate_run(user_prompt = user_prompt, session_id = session_id, user_name = user_name)

        system_prompt = ChatPromptTemplate.from_messages([
            ("system",
             "You are a compassionate therapist in India. "
             "Your role is to guide the client with empathy, practical advice, and cultural awareness. "
             "You have access to background knowledge about: "
             "1) Common taboos around mental health in India. "
             "2) Remedies and therapeutic practices. "
             "3) Past conversations with this client. "
             "Use these only as background — do not copy them directly. "

             "When responding: "
             "- Speak in a warm, conversational tone. "
             "- Sometimes validate the client’s feelings, but use varied wording. "
             "- Offer small, actionable suggestions that fit naturally into the conversation. "
             "- If relevant, bring up cultural taboos gently once or twice, The taboos:\n{taboo_details}\n\n"
             "to help the client feel understood — but keep it subtle, not repetitive. "
             "- Balance empathy, encouragement, and curiosity. Do not end every reply with a question. "
             "- Keep responses concise and human-like. "
             "- If asked something unrelated to therapy, politely refuse. "

             "Background context:\n"
             "1. The therapist’s knowledge:\n{remedies_context}\n\n"
             "2. Past conversation with the client:\n{retrieved_memory}\n\n"
             "All these contexts are only for making your answers more reliable, you still need to change your vocab to be soothing and not just be a bookish informatic model."),

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

model_runner = RunModel(db_name = "vector")
print(model_runner.initiate_run("I am feeling really depressed", session_id = "68790798v", user_name = "Raj"))