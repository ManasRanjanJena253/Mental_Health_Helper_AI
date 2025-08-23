import os
import hashlib
from typing import List, Tuple

from dotenv import load_dotenv
from PyPDF2 import PdfReader

from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
import chromadb


class RAGIndex:
    def __init__(
        self,
        collection_name: str,
        persist_dir: str,
        embedding_model: str = "gemini-embedding-001",  # safe default for langchain_google_genai
        chunk_size: int = 1000,
        chunk_overlap: int = 150,
    ):
        load_dotenv()
        self.google_api_key = os.getenv("GOOGLE_API_KEY")
        if not self.google_api_key:
            raise RuntimeError("GOOGLE_API_KEY not set.")

        # Embeddings + Vector store
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model=embedding_model, google_api_key = self.google_api_key
        )

        # Using LangChain's Chroma wrapper so we can use MMR, filters, etc.
        self.collection_name = collection_name.replace(" ", "").lower()
        self.persist_dir = persist_dir

        self._client = chromadb.PersistentClient(path = self.persist_dir)
        # Create the LC vectorstore (creates/loads the same memory_collection)
        self.vs = Chroma(
            collection_name = self.collection_name,
            persist_directory = self.persist_dir,
            embedding_function = self.embeddings,
        )

        # Chunking
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size = chunk_size,
            chunk_overlap = chunk_overlap,
            add_start_index = True,
            separators = ["\n\n", "\n", " ", ""],
        )

        # LLM for final answer
        self.llm = ChatGoogleGenerativeAI(
            model = "gemini-2.5-flash",
            google_api_key = self.google_api_key,
            temperature = 0.4,
            verbose = True,
        )

        self.prompt = ChatPromptTemplate.from_template(
            """You are a careful, supportive assistant for mental health topics.
Use the provided context (quotes from authoritative sources) to answer the user's question.
If something is unclear or potentially high risk, say what you can and encourage seeking professional help.
Be concise, kind, and practical. You will be provided some remedies after learning from which you need to help
the user. Don't mention the retrieval or remedies directly or where you got the remedies from. Your sole
responsibility is to help the user.

# Remedies
{context}

# Question
{question}

# Helpful, careful answer"""
        )
        self.chain = self.prompt | self.llm | StrOutputParser()

    @staticmethod
    def _hash_id(source: str, page: int, start: int, text: str) -> str:
        h = hashlib.sha1()
        h.update(f"{source}|{page}|{start}|{text}".encode("utf-8", errors="ignore"))
        return h.hexdigest()

    def _pdf_to_documents(self, pdf_path: str) -> List[Document]:
        docs: List[Document] = []
        with open(pdf_path, "rb") as f:
            reader = PdfReader(f)
            for p_idx, page in enumerate(reader.pages):
                try:
                    raw = page.extract_text() or ""
                except Exception:
                    raw = ""
                raw = raw.strip()
                if not raw:
                    continue
                chunks = self.splitter.split_text(raw)
                # add_start_index=True gave us start offsets in splitter? For RCTextSplitter, it returns strings only.
                # We'll compute start offsets approximately by cumulative indexing.
                cursor = 0
                for chunk in chunks:
                    start = raw.find(chunk, cursor)
                    if start < 0:
                        start = cursor
                    cursor = start + len(chunk)

                    doc = Document(
                        page_content=chunk,
                        metadata={
                            "source": os.path.abspath(pdf_path),
                            "page": p_idx + 1,
                            "start": int(start),
                        },
                    )
                    docs.append(doc)
        return docs

    def index_pdf(self, pdf_path: str, rebuild_collection: bool = False) -> int:
        """
        Ingest a PDF as chunked documents with stable IDs.
        If rebuild_collection=True, drop and recreate the memory_collection cleanly.
        Returns number of chunks indexed.
        """
        if rebuild_collection:
            # safest: delete memory_collection & recreate
            self._client.delete_collection(name=self.collection_name)
            self.vs = Chroma(
                collection_name=self.collection_name,
                persist_directory=self.persist_dir,
                embedding_function=self.embeddings,
            )

        docs = self._pdf_to_documents(pdf_path)
        if not docs:
            return 0

        # Stable IDs -> upserts are predictable. We use LC's add_documents() with ids.
        ids = [
            self._hash_id(
                doc.metadata["source"], doc.metadata["page"], doc.metadata["start"], doc.page_content
            )
            for doc in docs
        ]

        # Add (duplicates are skipped by Chroma if IDs already exist)
        # If you want a true "upsert", use vectorstore._collection.update(...) as needed.
        self.vs.add_documents(documents=docs, ids=ids)
        return len(docs)

    # ---------- Retrieval + Answering ----------
    def retrieve(self, query: str, k: int = 6, fetch_k: int = 24) -> List[Document]:
        """
        Using MMR to balance relevance + diversity. Great for long PDFs that repeat phrases.
        """
        return self.vs.max_marginal_relevance_search(query, k = k, fetch_k=fetch_k)

    def answer(self, query: str, k: int = 6) -> str:
        ctx_docs = self.retrieve(query, k=k)
        context = "\n\n".join(
            [f"(p.{d.metadata.get('page', '?')}) {d.page_content.strip()}" for d in ctx_docs]
        )
        return self.chain.invoke({"context": context, "question": query})


# -------- Example usage --------
if __name__ == "__main__":
    """
    One-time (or occasional) indexing:
        python optimized_rag_chroma.py  # runs the example below

    At runtime for chat:
        rag = RAGIndex("mentalhealthtaboosinindia", persist_dir="./chroma_store")
        print(rag.answer("I am feeling really depressed and lonely."))
    """
    load_dotenv()

    rag = RAGIndex(
        collection_name = "mentalhealthtaboosinindia",
        persist_dir ="../main/Mental_Health_Taboos_in_India",
        # For latest Google embeddings via LangChain wrapper, "gemini-embedding-001" is a stable choice.
        # If your langchain_google_genai supports it, you can try: embedding_model="text-embedding-004"
        chunk_size = 1500,
        chunk_overlap = 150,
    )

    # Index (set rebuild_collection=True the first time or when PDFs change substantially)
    n = rag.index_pdf("Mental_Health_in_India_Taboos_Report_2025.pdf", rebuild_collection = True)
    print(f"Indexed {n} chunks.")

    # Ask something
    print(rag.answer("I am feeling really depressed and lonely."))

# db_names :
# 1. guidelinesforpreventionofmentalhealth
# 2. mentalhealthremedies
# 3. yogaformentalhealth
# 4. mentalhealthtaboosinindia