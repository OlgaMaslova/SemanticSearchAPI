from fastapi.testclient import TestClient

from main import app

client = TestClient(app)


def test_read_main():
    response = client.get("/")
    assert response.status_code == 200
    assert response.json() == {"message": "Welcome to Semantic Search. Please upload your text."}


def test_upload_vocabulary():
    response = client.post(
        "/upload-vocabulary/",
        json={"text": "The lack of a formal link between neural network structure and its emergent function "
                            "has hampered our understanding of how the brain processes information. We have now come "
                            "closer to describing such a link by taking the direction of synaptic transmission into "
                            "account, constructing graphs of a network that reflect the direction of information "
                            "flow, and analyzing these directed graphs using algebraic topology."},
    )
    assert response.status_code == 200
    assert response.json() == {
        "message": "Successful upload"
    }
