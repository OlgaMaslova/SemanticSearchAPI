from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import uuid
from semantic.utils import create_vocabulary, create_embeddings_for_sentences, \
    create_embedding_for_sentence, semantic_search, is_valid_query

app = FastAPI(
    title="SemanticSearch API",
    description="SemanticSearch API allows you to upload a long-form text with multiple sentences and search "
                "for similar sentences using a query sentence. Users might be searching for similar claims "
                "in a scientific text",
    version="0.0.1"
)
vocabularies = {} # {vocabulary_id: sentence_list}
embeddings = {} # {vocabulary_id: embeddings}
origins = [
    "http://localhost",
    "http://localhost:4200",
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class Message(BaseModel):
    message: str


class Vocabulary(BaseModel):
    text: str

@app.get("/")
async def root():
    return {"message": "Welcome to Semantic Search. Please upload your text."}


@app.post("/upload-vocabulary/", status_code=200, summary="Uploads text",
          description="Loads text and splits it in sentences.",
          responses={
              200:  {
                   "description": "Upload vocabulary",
                   "content": {
                        "application/json": {
                             "example": {"id": "vocabulary id"}
                       }
                   },
              },
          }
          )
async def load_vocabulary(voc: Vocabulary):
    global vocabularies
    sentence_list = create_vocabulary(voc.text)
    voc_id = str(uuid.uuid4())
    vocabularies[voc_id] = sentence_list
    return JSONResponse(status_code=200, content={"id": voc_id})


@app.get("/embeddings", status_code=200, summary="Creates embeddings",
         description="Returns embeddings for sentences.",
         responses={
             404: {"model": Message, "description": "The text was not uploaded"},
             200: {
                 "description": "List of embeddings per sentence",
                 "content": {
                     "application/json": {
                         "example": [{"sentence": "sentence text processed", "embeddings": [0.596, -0.16]}]
                     }
                 },
             },
         },
         )
async def get_embeddings(vocabulary_id: str):
    global vocabularies, embeddings
    if vocabulary_id not in vocabularies:
        return JSONResponse(status_code=404, content={"message": "No sentences found, upload your text first"})
    if vocabulary_id in embeddings:
        sentence_embeddings = embeddings[vocabulary_id]
    else:
        sentence_list = vocabularies[vocabulary_id]
        sentence_embeddings = create_embeddings_for_sentences(sentence_list)
        embeddings[vocabulary_id] = sentence_embeddings.tolist()
    response = []
    for i, embeddings in enumerate(sentence_embeddings):
        response.append(
            {"sentence": sentence_list[i], "embeddings": embeddings.tolist()}
        )
    return response


@app.post("/query", status_code=200, summary="Finds similar sentence",
          description="Searches for similar sentences using a query sentence. Returns the top result with its score.",
          responses={
              400: {"model": Message, "description": "Bad input (empty, only special caracters...)"},
              404: {"model": Message, "description": "The text was not uploaded"},
              200: {
                  "description": "Top matched sentence",
                  "content": {
                      "application/json": {
                          "example": [
                              {"top_answer": "some text", "score": 0.95}]
                      }
                  },
              },
          },
          )
async def semantic_query(vocabulary_id: str, query: str):
    global vocabularies, embeddings
    if vocabulary_id not in vocabularies:
        return JSONResponse(status_code=404, content={"message": "No sentences found, upload your text first"})
    if vocabulary_id not in embeddings:
        sentence_list = vocabularies[vocabulary_id]
        sentence_embeddings = create_embeddings_for_sentences(sentence_list)
    if not is_valid_query(query):
        return JSONResponse(status_code=400,
                            content={"message": "Query must no be empty and should contain letters or numbers"})
    query_embedding = create_embedding_for_sentence(query)
    answer_meta = semantic_search(query_embedding, sentence_embeddings)
    response = {"top_answer": "", "score": 0}
    for answer in answer_meta[0]:
        s_id = answer["corpus_id"]
        score = answer["score"]
        response["top_answer"] = sentence_list[s_id]
        response["score"] = score
    return response


@app.get("/reset", status_code=200, summary="Resets data",
         description="Cleans vocabulary and embeddings for a given vocabulary id.",
         responses={
             200: {"model": Message},
         }
         )
async def reset(vocabulary_id: str):
    global vocabularies, embeddings
    if vocabulary_id in vocabularies:
        del vocabularies[vocabulary_id]
    if vocabulary_id in embeddings:
        del embeddings[vocabulary_id]
    return JSONResponse(status_code=200, content={"message": "Reset successful"})
