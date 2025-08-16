import chromadb


# Example usecase :
# session_id = "user_123"   # You can use auth ID or UUID
# turn_id = "turn_1"        # Increment each conversation step
#
# collection.add(
#     documents=[
#         f"User feeling: {user_prompt}\n"
#         f"Identified symptoms: {symptoms}\n"
#         f"Doctor's remedies: {remedies}\n"
#         f"Therapist final response: {therapist_response}"
#     ],
#     metadatas=[{"session_id": session_id, "turn_id": turn_id}],
#     ids=[f"{session_id}_{turn_id}"]
# )

# Retrieving Past context :
# results = collection.query(
#     query_texts=["patient feels anxious about exams"],  # new query
#     n_results=3,                                        # top-k matches
#     where={"session_id": session_id}                   # optional filter Can be used if the user is chatting in some previous chat.
# )
#
# for doc, meta in zip(results["documents"][0], results["metadatas"][0]):
#     print(meta["turn_id"], ":", doc)

