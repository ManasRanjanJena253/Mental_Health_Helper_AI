import os
import chromadb
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_google_genai import ChatGoogleGenerativeAI

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
        :param db_name: The name of the chromadb collection based on the user.
        """
        # Loading the llm
        self.__api_key = api_key

        # Creating chroma client.
        self.chroma_client = chromadb.PersistentClient(path ="../chroma")
        self.collection = self.chroma_client.get_or_create_collection(name = db_name.replace(" ", "").lower())  # Handling the spaces in names.

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

        # Creating various prompts for chaining, to avoid the model from hallucinating and having better reasoning capabilities

        prompt_1 = ChatPromptTemplate.from_messages([
            ("system", "You are a psychologist and you need to identify the potential mental issues regarding the query made by the user.\
            Provide the identified issues as a list and formulate the procedure by which the therapist should address these for resolving these.\
            You are in INDIA where mental health is stigmatised and frowned upon,\
            so make sure the procedures to be used by therapist to destigmatize and comfort the patient simultaneously."),
            ("user", "{feeling}")
        ])

        prompt_2 = ChatPromptTemplate.from_messages([
            ("system", """You are a therapist in INDIA and your task is to address the mental issues of your client by asking progressive questions and listening patiently.\
                       You will be provided with the procedures provided by the doctor to be taken were learn from to resolve those issues. You also need to identify the \
                       cultural or traditional stigma that might be the root cause of distress of the client. Don't address the doctor's remedies directly to the user,\
                       and no need to tell the patient about any of your postural changes you do, just try to convey those kindness and tonality change through your language.\
                       You need to WAIT for the users reply after asking a question, this is a LIVE session NOT a play.\
                        You need to WAIT for the users reply after asking a question, this is a LIVE session NOT a play.\n""
                        Anything asked out of the scope of a therapist you should strictly deny it. 
                        Try to keep your responses more interactive and longer."""),
            ("user", "plz help !!!, the suggestions given by the doctor are : {remedies}"),
        ])

        # Creating the chain
        # chain1 : Identifying the remedies.
        chain_1 = (
                {"feeling": RunnablePassthrough()}
                | prompt_1
                | self.llm
                | StrOutputParser()
        )

        # chain2
        chain_2 = (
                {"remedies": chain_1}
                | prompt_2
                | self.llm
                | StrOutputParser()
        )

        # Running the full pipeline
        output = chain_2.invoke({"feeling": user_prompt})

        # Checking if the model produced any output or not.
        if not output:
            raise ValueError("UNABLE TO GENERATE A REPLY !!! \n Plz Try Again Later.")

        # Adding the session data into the chromadb
        self.collection.add(
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
        results = self.collection.query(
            query_texts = [user_prompt],
            n_results = 5,
            where = {"session_id": {"$eq": session_id}}
        )

        # filtering by user_id
        filtered_docs = [
            (doc, meta) for doc, meta in zip(results["documents"][0], results["metadatas"][0])
            if meta.get("user_name") == user_name
        ]

        # Checking if there are any data about previous sessions of the user.
        if not filtered_docs:
            self.initiate_run(user_prompt = user_prompt, session_id = session_id, user_name = user_name)

        assistant_prompt = ChatPromptTemplate.from_messages([
            ("system", """ You are an assistant responsible for keeping in check that the ai therapist is keeping him on the right track in helping the issue of the patient.\n
                        TO do this you need to strictly and clearly tell the therapist what his next steps will be, based on the info you will be provided on the past interaction 
                        with the patient. Make sure that the issues like societal and traditional bounds are addressed correctly as you are in INDIA. In every step try to destigmatize the 
                        traditional orthodox beliefs related to their culture and may be the cause of distress in patients.                  
                        These are the qualities you need to enforce on the therapist AI: \n
                        Step 1 --> Validating the patient’s feelings without judgment.\n
                        Step 2 --> Identifying cognitive distortions and gently educating the patient.\n
                        Step 3 --> Providing practical strategies for challenging negative thinking.\n
                        Step 4 --> Encouraging self-awareness and insight into underlying beliefs.\n
                        Step 5 --> Setting a collaborative, supportive tone for ongoing progress.\n
                       Therapist shouldn't ask only questions it have to also try to comfort the patient on their given answers and make them feel safe. For example : \n
                         Patient: I feel like I’m failing at everything.\n
                        
                        Therapist: That’s tough. Can you give one example? (Validation & Active Listening)\n
                        
                        Patient: I messed up a presentation and skipped my workout.\n
                        
                        Therapist: What about something you did well recently? (Cognitive Reframing)\n
                        
                        Patient: I helped a colleague solve a problem yesterday.\n
                        
                        Therapist: Good! Your mind is focusing on negatives—this is all-or-nothing thinking. (Psychoeducation)\n
                        
                        Patient: It feels real, though.\n
                        
                        Therapist: When it arises, try: “I failed here, but I succeeded there.” (Behavioral / Cognitive Strategy)\n
                        
                        Patient: I can try that.\n
                        
                        Therapist: Also, consider if your expectations are realistic. (Guided Discovery / Goal Setting)\n
                        Anything asked out of the scope of a therapist you should strictly deny it and say NO. 
                        Try to keep your responses more interactive and longer."""),
            ("user", "Previous_Context : {context}")
        ])
        main_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a therapist in INDIA and your task is to address the mental issues of your client by asking progressive questions and listening patiently.\
                       You have already had some conversation with the patient and this is a continuation.Your previous conversation : {context}. You also need to identify the \
                       cultural or traditional stigma that might be the root cause of distress of the client. Don't address the doctor's remedies directly to the user,\
                       and no need to tell the patient about any of your postural changes you do, just try to convey those kindness and tonality change through your language.\
                       You need to WAIT for the users reply after asking a question, this is a LIVE session NOT a play.\n""
                       "\n\n \
                        Anything asked out of the scope of a therapist you should strictly deny it. 
                        Try to keep your responses more interactive and longer.
                        You will be constantly given instructions on how to handle the conversation and therapy session. You need to follow those instructions STRICTLY. """),
            ("user", "{response}"),
        ])

        context_str = "\n".join([doc for doc, _ in filtered_docs])

        assist_chain = ({"context": RunnablePassthrough()}
                    | assistant_prompt
                    | self.llm
                    | StrOutputParser())

        instructions = assist_chain.invoke({"context": context_str})

        main_chain = ({"response": RunnablePassthrough(), "context": RunnablePassthrough(), "instructions": RunnablePassthrough()}
                 | main_prompt
                 | self.llm
                 | StrOutputParser())

        try:
            output = main_chain.invoke({
                    "response": user_prompt,
                    "context": context_str,
                    "instructions": instructions
                })

        except Exception as e:
            return f"Error : {e}"

        # Checking if the model produced any output or not.
        if not output:
            raise ValueError("UNABLE TO GENERATE A REPLY !!! \n Plz Try Again Later.")

        self.collection.add(
            documents=[
                f"User feeling: {user_prompt}\n"
                f"Therapist final response: {output}"
            ],
            metadatas=[{"session_id": session_id, "user_name": user_name}],
            ids=[f"{session_id}"]
        )

        return output