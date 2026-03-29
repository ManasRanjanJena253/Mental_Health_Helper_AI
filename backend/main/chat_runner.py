"""
chat_runner.py
~~~~~~~~~~~~~~
Async-capable wrapper around the LLM + ChromaDB RAG pipeline.

  1. All vector-store queries run in a thread-pool via asyncio.run_in_executor
     so they never block the FastAPI event loop.
  2. `astream()` yields LLM tokens as they arrive (SSE-ready).
  3. `arun()` is the non-streaming async equivalent of the old `run()`.
  4. ChromaDB + LangChain objects are still created synchronously in __init__
     (they are not async) but instantiation is cheap enough to do once per user.
  5. Memory writes are fire-and-forget (asyncio.create_task) so the response
     is not delayed by the ChromaDB write.
"""

import asyncio
import os
from functools import partial
from typing import AsyncIterator

import chromadb
from dotenv import load_dotenv
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

load_dotenv()


# Prompt templates  (defined once at module level — no per-request allocation)
_INIT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Start every reply with a warm "Namaste".

You are a compassionate therapist based in INDIA. Role: support the client with empathy,
practical steps, and culturally grounded understanding.
Use the following background only internally (do not list them):
- Indian mental-health taboos: {taboo_details}
- Indian remedies & therapeutic practices: {remedies_context}

When responding:
- Speak like a private session: warm, conversational, gentle.
- Validate feelings (vary wording), mix empathy + encouragement + curiosity.
- Offer 1–3 small, doable suggestions tied to Indian life (family dynamics, food/routine,
  festivals, local help lines, simple home practices).
- If relevant, subtly name cultural taboos to normalise the client's experience — keep it
  reassuring and brief.
- Ask 1–2 proactive prompts so the user can vent and feel heard (don't always end with a
  question).
- Use Indian touchstones, idioms or short Hindi/regional phrases when natural; keep language
  soothing, not clinical.
- Politely refuse requests outside therapeutic scope and signpost safer alternatives.

Always make the answer feel specifically INDIAN and culturally sensitive and concise.""",
    ),
    ("user", "{query}"),
])

_CHAT_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a compassionate therapist in INDIA. Guide the client with empathy,
practical steps, and cultural sensitivity.
Use the following only as internal context (do not quote):
- Indian mental-health taboos: {taboo_details}
- Indian remedies & therapeutic practices: {remedies_context}
- Past conversation: {retrieved_memory}

Reply style:
- Warm, private-session tone; sprinkle short Hindi/regional phrases and Indian touchstones
  (family, festivals, food, local resources) when natural.
- Validate feelings (vary wording), blend empathy + encouragement + gentle curiosity.
- Give 1–3 simple, actionable suggestions that fit Indian life (home routines, family
  conversations, rituals, local support).
- If relevant, gently normalise experience by referencing cultural taboos once or twice —
  brief and reassuring.
- Keep replies concise, human, and not overly clinical; don't always end with a question.
- Politely refuse requests outside therapy and signpost safer alternatives.

Always make the answer feel specifically INDIAN and culturally grounded and concise.""",
    ),
    ("user", "{query}"),
])


# RunModel
class RunModel:
    def __init__(
        self,
        db_name: str,
        api_key: str = None,
        model_name: str = "gemini-2.5-flash",
        temperature: float = 0.5,
    ):
        self.__api_key = api_key or os.getenv("GOOGLE_API_KEY")
        if not self.__api_key:
            raise RuntimeError("GOOGLE_API_KEY not set.")

        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="gemini-embedding-001",
            google_api_key=self.__api_key,
        )

        # Per-user conversation memory (ChromaDB)
        self._chroma_memory_client = chromadb.PersistentClient(path="../chroma")
        self.memory_collection = self._chroma_memory_client.get_or_create_collection(
            name=db_name.replace(" ", "").lower()
        )

        # Shared RAG knowledge bases
        self.indian_taboo_vs = Chroma(
            collection_name="mentalhealthtaboosinindia",
            persist_directory="Mental_Health_Taboos_in_India",
            embedding_function=self.embeddings,
        )
        self.remedies_vs = Chroma(
            collection_name="mentalhealthremedies",
            persist_directory="Mental_Health_Remedies",
            embedding_function=self.embeddings,
        )

        self.llm = ChatGoogleGenerativeAI(
            model=model_name,
            google_api_key=self.__api_key,
            temperature=temperature,
        )

        # Pre-build chains (prompt | llm | parser) — reused across calls
        self._init_chain = _INIT_PROMPT | self.llm | StrOutputParser()
        self._chat_chain = _CHAT_PROMPT | self.llm | StrOutputParser()

        # For streaming we attach the parser separately
        self._init_chain_raw = _INIT_PROMPT | self.llm
        self._chat_chain_raw = _CHAT_PROMPT | self.llm


    # Internal: RAG retrieval (sync — called via run_in_executor)

    def _retrieve_context(self, query: str) -> tuple[str, str]:
        """Returns (remedies_context, taboo_details) as plain strings."""
        remedies_docs = self.remedies_vs.max_marginal_relevance_search(
            query=query, k=3, fetch_k=10
        )
        taboo_docs = self.indian_taboo_vs.max_marginal_relevance_search(
            query=query, k=3, fetch_k=10
        )
        remedies_context = "\n\n".join(d.page_content for d in remedies_docs)
        taboo_details = "\n\n".join(d.page_content for d in taboo_docs)
        return remedies_context, taboo_details

    def _retrieve_memory(self, query: str, session_id: str, user_name: str) -> str:
        """Fetch relevant past turns for this session from ChromaDB."""
        result = self.memory_collection.query(
            query_texts=[query],
            n_results=5,
            where={"session_id": {"$eq": session_id}},
        )
        if not result["documents"] or not result["documents"][0]:
            return ""
        filtered = [
            doc
            for doc, meta in zip(
                result["documents"][0], result["metadatas"][0]
            )
            if meta.get("user_name") == user_name
        ]
        return "\n".join(filtered)

    def _write_memory(self, user_prompt: str, output: str, session_id: str, user_name: str) -> None:
        """Persist a turn to the per-user ChromaDB collection (sync)."""
        import uuid
        self.memory_collection.add(
            documents=[f"User feeling: {user_prompt}\nTherapist final response: {output}"],
            metadatas=[{"session_id": session_id, "user_name": user_name}],
            ids=[str(uuid.uuid4())],   # unique ID per message, not session-level
        )

    # Async non-streaming  (drop-in replacement for old `run()`)
    async def arun(self, user_prompt: str, session_id: str, user_name: str) -> str:
        """
        Full async pipeline: retrieve → invoke LLM → persist memory.
        Returns the complete response string.
        """
        loop = asyncio.get_event_loop()

        # Run all sync blocking calls in the thread pool
        remedies_context, taboo_details = await loop.run_in_executor(
            None, self._retrieve_context, user_prompt
        )
        memory_str = await loop.run_in_executor(
            None, partial(self._retrieve_memory, user_prompt, session_id, user_name)
        )

        is_first_turn = not memory_str.strip()

        if is_first_turn:
            output = await loop.run_in_executor(
                None,
                partial(
                    self._init_chain.invoke,
                    {
                        "taboo_details": taboo_details,
                        "remedies_context": remedies_context,
                        "query": user_prompt,
                    },
                ),
            )
        else:
            output = await loop.run_in_executor(
                None,
                partial(
                    self._chat_chain.invoke,
                    {
                        "taboo_details": taboo_details,
                        "remedies_context": remedies_context,
                        "retrieved_memory": memory_str,
                        "query": user_prompt,
                    },
                ),
            )

        if not output:
            raise ValueError("LLM returned an empty response.")

        # Fire-and-forget memory write — doesn't delay the response
        asyncio.get_event_loop().run_in_executor(
            None, partial(self._write_memory, user_prompt, output, session_id, user_name)
        )

        return output

    # Async streaming  (yields str tokens as they arrive from the LLM)

    async def astream(
        self, user_prompt: str, session_id: str, user_name: str
    ) -> AsyncIterator[str]:
        """
        Async generator that yields LLM response tokens one by one.
        Designed to be consumed by a FastAPI StreamingResponse (SSE).

        Usage in endpoint:
            async def event_generator():
                full = []
                async for token in runner.astream(prompt, sid, uname):
                    full.append(token)
                    yield f"data: {token}\\n\\n"
                yield "data: [DONE]\\n\\n"
                # memory write happens after streaming completes
        """
        loop = asyncio.get_event_loop()

        # Retrieve context concurrently
        remedies_context, taboo_details = await loop.run_in_executor(
            None, self._retrieve_context, user_prompt
        )
        memory_str = await loop.run_in_executor(
            None, partial(self._retrieve_memory, user_prompt, session_id, user_name)
        )

        is_first_turn = not memory_str.strip()
        chain_input = (
            {
                "taboo_details": taboo_details,
                "remedies_context": remedies_context,
                "query": user_prompt,
            }
            if is_first_turn
            else {
                "taboo_details": taboo_details,
                "remedies_context": remedies_context,
                "retrieved_memory": memory_str,
                "query": user_prompt,
            }
        )
        chain = self._init_chain_raw if is_first_turn else self._chat_chain_raw

        full_response_parts: list[str] = []

        # LangChain's astream() is truly async — no executor needed
        async for chunk in chain.astream(chain_input):
            # chunk is an AIMessageChunk; extract text
            token: str = chunk.content if hasattr(chunk, "content") else str(chunk)
            if token:
                full_response_parts.append(token)
                yield token

        full_response = "".join(full_response_parts)

        # Persist memory without blocking the caller
        if full_response:
            asyncio.get_event_loop().run_in_executor(
                None,
                partial(self._write_memory, user_prompt, full_response, session_id, user_name),
            )