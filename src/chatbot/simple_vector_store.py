"""
Simple vector store using LangChain's native pgvector integration.
Much cleaner than our custom implementation!
"""

from typing import List, Optional
from langchain_postgres import PGVector
from langchain_openai import OpenAIEmbeddings
from langchain_core.documents import Document
from dotenv import load_dotenv

load_dotenv()


class SimpleVectorStore:
    """Simple wrapper around LangChain's pgvector integration."""

    def __init__(self):
        """Initialize with LangChain's PGVector."""
        connection_string = "postgresql://chatbot_user:chatbot_password@localhost:5432/conversation_store"

        self.embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

        self.vector_store = PGVector(
            embeddings=self.embeddings,
            connection=connection_string,
            collection_name="conversations",
        )

    def store_conversation(self, summary: str) -> None:
        """Store a conversation summary."""
        # Create a simple document
        doc = Document(page_content=summary)

        # Add to vector store
        self.vector_store.add_documents([doc])
        print(f"✅ Stored conversation summary")

    def search_conversations(
        self, query: str, limit: int = 2, distance_threshold: float = 0.8
    ) -> List[str]:
        """Search for similar conversations with distance filtering (lower distance = more similar)."""
        try:
            # Use LangChain's similarity search with scores (returns distances)
            results = self.vector_store.similarity_search_with_score(query, k=limit)

            if not results:
                return []

            # Filter by distance threshold (lower scores = more similar)
            filtered_results = [
                doc for doc, score in results if score <= distance_threshold
            ]

            if not filtered_results:
                print(
                    f"🔍 Found {len(results)} results but none below distance threshold {distance_threshold}"
                )
                return []

            # Just return the summaries
            summaries = [doc.page_content[:150] + "..." for doc in filtered_results]
            print(
                f"🔍 Found {len(filtered_results)} similar conversations (distance ≤ {distance_threshold})"
            )
            return summaries

        except Exception as e:
            print(f"❌ Search error: {e}")
            return []


if __name__ == "__main__":
    """Test the simple vector store."""
    print("🧪 Testing Simple Vector Store...")

    vs = SimpleVectorStore()

    # Test storing a conversation
    vs.store_conversation(
        summary="User asked about Python web frameworks and chose FastAPI for their project. They were building a REST API and needed something fast and modern."
    )

    # Test searching
    results = vs.search_conversations("web development", limit=2)
    for result in results:
        print(f"📄 {result}")
