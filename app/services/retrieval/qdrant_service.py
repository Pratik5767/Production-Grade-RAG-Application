import logfire
from qdrant_client import QdrantClient
from qdrant_client.http import models
from app.config import settings
from app.services.retrieval.embeddings import embed_query


# Initialize Qdrant Client
client = QdrantClient(url=settings.QDRANT_URL, api_key=settings.QDRANT_API_KEY)


def search_enterprise_knowledge(query: str, limit: int = 8):
    """
    Performs a high-precision search in the enterprise knowledge base.
    Uses the modern query_points interface.
    """

    try:
        # Convert query text into a vector embedding
        query_vector = embed_query(query)

        # Search Qdrant for the closest matching vectors
        response = client.query_points(
            collection_name=settings.QDRANT_COLLECTION,
            query=query_vector,
            limit=limit,
            with_payload=True # Include stored metadata (JSON) in results
        )

        results = []
        # Extract content, source, and score from each matched point
        for res in response.points:
            results.append({
                "content": res.payload.get("text", ""),         # Original text chunk
                "source": res.payload.get("source", "Unknown"), # Document/source name
                "score": res.score                              # Similarity score
            })

        return results
    except Exception as e:
        logfire.error(f"❌ Qdrant Search Failed: {e}")
        return []
