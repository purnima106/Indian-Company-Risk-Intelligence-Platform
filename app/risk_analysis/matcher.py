from app.embeddings.embedder import RiskEmbedder
from app.embeddings.vector_store import InMemoryVectorStore
from app.risk_analysis.risk_extractor import RiskRecord

class RiskMatcher:
    def __init__(self, embedder: RiskEmbedder | None = None,):
        self.embedder = embedder if embedder is not None else RiskEmbedder()
        self.vector_store = InMemoryVectorStore()

    def match(
            self,previous_risks:list[RiskRecord], current_risks:list[RiskRecord], top_k:int = 5,
    ) -> list[dict]:
        if not previous_risks or not current_risks:
            return []

        previous_texts = [risk.text for risk in previous_risks]
        current_texts = [risk.text for risk in current_risks]
        previous_embeddings = self.embedder.embed(previous_texts)
        current_embeddings = self.embedder.embed(current_texts)
        vector_store = InMemoryVectorStore()

        previous_metadata = [
            {
                "risk_id": risk.risk_id,
                "source_year": risk.source_year,
                "category": risk.category,
                "title": risk.title,
                "text": risk.text,
                "page": risk.page,
                "source_document": risk.source_document,
            }
            for risk in previous_risks
        ]

        vector_store.add(previous_embeddings, previous_metadata)
    
        matches = []
        for risk, embedding in zip(
            current_risks,
            current_embeddings,
        ):
            candidates = vector_store.search(
                embedding,
                top_k=top_k,
            )

            matches.append(
                {
                    "current_risk": {
                        "risk_id": risk.risk_id,
                        "source_year": risk.source_year,
                        "title": risk.title,
                        "text": risk.text,
                        "page": risk.page,
                    },
                    "candidates": candidates,
                }
            )

        return matches