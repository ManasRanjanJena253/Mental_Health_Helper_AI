import os
import chromadb
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma

load_dotenv()

# Setting the verbose = True, to see the thinking of LLMs
#set_verbose(True)

class RunModel:
    def __init__(self, db_name: str, api_key = os.getenv("GOOGLE_API_KEY"), model_name: str = "gemini-2.5-flash", temperature: int = 0.5):
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
                                                                                    fetch_k = 10)
        remedies_context = "\n\n".join(doc.page_content for doc in remedies_retrieved)

        # remedies_context = self.remedies_vector_store.query(query_texts = [user_prompt],
        #                                                     n_results = 5)

        taboo_retrieved = self.indian_taboo_vector_store.max_marginal_relevance_search(query = user_prompt,
                                                                                     k = 3,
                                                                                     fetch_k = 10)

        taboo_details = "\n\n".join(doc.page_content for doc in taboo_retrieved)

        # taboo_details = self.indian_taboo_vector_store.query(query_texts = [user_prompt],
        #                                                      n_results = 5)

        system_prompt = ChatPromptTemplate.from_messages([
            ("system",
                         """Start every reply with a warm "Namaste".
            
            You are a compassionate therapist based in INDIA. Role: support the client with empathy, practical steps, and culturally grounded understanding. Use the following background only internally (do not list them): 
            - Indian mental-health taboos: {taboo_details}
            - Indian remedies & therapeutic practices: {remedies_context}
            
            When responding:
            - Speak like a private session: warm, conversational, gentle.
            - Validate feelings (vary wording), mix empathy + encouragement + curiosity.
            - Offer 1–3 small, doable suggestions tied to Indian life (family dynamics, food/routine, festivals, local help lines, simple home practices).
            - If relevant, subtly name cultural taboos to normalize the client’s experience — keep it reassuring and brief.
            - Ask 1–2 proactive prompts so the user can vent and feel heard (don’t always end with a question).
            - Use Indian touchstones, idioms or short Hindi/regional phrases when natural; keep language soothing, not clinical.
            - Politely refuse requests outside therapeutic scope and signpost safer alternatives.
            
            Always make the answer feel specifically INDIAN and culturally sensitive and concise."""
        ),
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
                                                                                    fetch_k = 10)

        # remedies_context = self.remedies_vector_store.query(query_texts = [user_prompt],
        #                                                     n_results = 5)

        taboo_details = self.indian_taboo_vector_store.max_marginal_relevance_search(query = user_prompt,
                                                                                     k = 3,
                                                                                     fetch_k = 10)

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
                         """You are a compassionate therapist in INDIA. Guide the client with empathy, practical steps, and cultural sensitivity. Use the following only as internal context (do not quote): 
            - Indian mental-health taboos: {taboo_details}
            - Indian remedies & therapeutic practices: {remedies_context}
            - Past conversation: {retrieved_memory}
            
            Reply style:
            - Warm, private-session tone; sprinkle short Hindi/regional phrases and Indian touchstones (family, festivals, food, local resources) when natural.
            - Validate feelings (vary wording), blend empathy + encouragement + gentle curiosity.
            - Give 1–3 simple, actionable suggestions that fit Indian life (home routines, family conversations, rituals, local support).
            - If relevant, gently normalize experience by referencing cultural taboos once or twice — brief and reassuring.
            - Keep replies concise, human, and not overly clinical; don’t always end with a question.
            - Politely refuse requests outside therapy and signpost safer alternatives.
            
            Always make the answer feel specifically INDIAN and culturally grounded and concise."""),

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

# model_runner = RunModel(db_name = "vector")
# print(model_runner.initiate_run("I am feeling really depressed", session_id = "68790798v", user_name = "Raj"))