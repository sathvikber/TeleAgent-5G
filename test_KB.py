from teleagent.rag.faiss_vector_store import FaissVectorStore

store = FaissVectorStore.load("/workspace/teleagent/data/knowledge_base")

print("Stored chunks:", store.size)

for chunk in store.chunks[:5]:
    print("=" * 70)
    print("ID       :", chunk.id)
    print("Metadata :", chunk.metadata)
    print("Content  :", chunk.content[:300])
