import os
from uuid import uuid4

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI
from PyPDF2 import PdfReader
import chromadb


class CreateEmbeddings:
    def __init__(self, db_name: str, file_path: str, chroma_path: str):
        self.db_name = db_name
        self.file_path = file_path
        self.chroma_client = chromadb.PersistentClient(path = chroma_path)
        self.collection = self.chroma_client.get_or_create_collection(
            name=db_name.replace(" ", "").lower())  # Handling the spaces in names.

    def process_pdf(self):
        """
        Processes the text from a given pdf and dumps it into the vector db.
        :return: None
        """
        try:
            with open(self.file_path, "rb+") as f:
                text = ""
                pdf_reader = PdfReader(f)
                for page in pdf_reader.pages:
                    text += page.extract_text()

        except Exception as e:
            raise ValueError(e)

        self.collection.add(documents = text,
                            ids = str(uuid4()))

    def test_llm(self, llm, user_prompt):
        """
        Used for testing the llm output using the vector db created by dumping the provided pdf.
        :param llm: Any Chat generative AI model.
        :param user_prompt: The user query.
        :return: The response of the model.
        """

        self.process_pdf()

        results = self.collection.query(
            query_texts=[user_prompt],
            n_results=5
        )

        docs = results.get("documents", [])
        context = "\n".join([doc for sublist in docs for doc in sublist])

        prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant.
            Use the following context to answer the question.
            No need to mention the source from where you are getting the context from.
            Only refer to the points which you deem important to answer the users query.
            Context is provided only for your guidance and not the sole determiner of the type of answer you will be giving.
            Remember that the user is from INDIA, so, make sure to answer the queries from the perspective of indian
            heritage, culture, taboos, and societal rules.

            Context:
            {context}

            Question: {question}
            """
        )

        chain = (
                {"context": RunnablePassthrough(), "question": RunnablePassthrough()}
                | prompt
                | llm
                | StrOutputParser()
        )

        try:
            output = chain.invoke({"context": context, "question": user_prompt})
            return output
        except Exception as e:
            return str(e)

load_dotenv()
api_key = os.getenv("GOOGLE_API_KEY")

llm = ChatGoogleGenerativeAI(model = "gemini-2.5-flash",
                                  google_api_key = api_key,
                                  verbose = True,
                                  temperature = 0.5)
obj = CreateEmbeddings(db_name = "mentalhealthtaboosinindia", chroma_path = "Mental_Health_Taboos_In_India", file_path ="Mental_Health_Concerns_In_Indian_Population.pdf")
print(obj.test_llm(llm = llm, user_prompt = "I am feeling really depressed and lonely."))

# db_names :
# 1. guidelinesforpreventionofmentalhealth
# 2. mentalhealthremedies
# 3. yogaformentalhealth
# 4. mentalhealthtaboosinindia