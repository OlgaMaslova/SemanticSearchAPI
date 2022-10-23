from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from semantic.utils import create_vocabulary, create_embeddings_for_sentences, \
    create_embedding_for_sentence, semantic_search, is_valid_query

app = FastAPI(
    title="SemanticSearch API",
    description="SemanticSearch API allows you to upload a long-form text with multiple sentences and search "
                "for similar sentences using a query sentence. Users might be searching for similar claims "
                "in a scientific text",
    version="0.0.1"
)
sentence_list = None
sentence_embeddings = None
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
              200: {"model": Message},
          }
          )
async def load_vocabulary(voc: Vocabulary):
    global sentence_list
    sentence_list = create_vocabulary(voc.text)
    return JSONResponse(status_code=200, content={"message": "Successful upload"})


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
async def get_embeddings():
    global sentence_list, sentence_embeddings
    if not sentence_list:
        return JSONResponse(status_code=404, content={"message": "No sentences found, upload your text first"})
    sentence_embeddings = create_embeddings_for_sentences(sentence_list)
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
async def semantic_query(query: str):
    global sentence_list, sentence_embeddings
    if not sentence_list:
        return JSONResponse(status_code=404, content={"message": "No sentences found, upload your text first"})
    if sentence_embeddings is None:
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
         description="Cleans vocabulary and embeddings.",
         responses={
             200: {"model": Message, "description": "The text was not uploaded"},
         }
         )
async def reset():
    global sentence_list, sentence_embeddings
    sentence_list = None
    sentence_embeddings = None
    return JSONResponse(status_code=200, content={"message": "Reset successful"})
