from fastapi.testclient import TestClient

import rag_agent.api as rag_api


class StubAgent:
    def health(self):
        return {"status": "ok", "llm_configured": False}

    def answer_with_sources(self, question):
        return {
            "answer": f"answer:{question}",
            "sources": [
                {
                    "source": "knowledge.md",
                    "section": "6.1 最大回撤",
                    "chunk_id": "knowledge.md:0",
                    "hybrid_score": 0.01,
                }
            ],
            "error": "",
        }


def test_health_and_query_endpoints(monkeypatch):
    monkeypatch.setattr(rag_api, "get_agent", lambda: StubAgent())
    client = TestClient(rag_api.app)

    health = client.get("/health")
    response = client.post("/query", json={"query": "最大回撤是什么？"})

    assert health.status_code == 200
    assert health.json()["status"] == "ok"
    assert response.status_code == 200
    assert response.json()["sources"][0]["section"] == "6.1 最大回撤"


def test_query_endpoint_rejects_empty_and_overlong_input():
    client = TestClient(rag_api.app)

    assert client.post("/query", json={"query": ""}).status_code == 422
    assert client.post("/query", json={"query": "x" * 401}).status_code == 422
