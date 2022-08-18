import os
from flask import Flask, request
from flask_cors import CORS
from sentence_transformers import SentenceTransformer


app = Flask(__name__)

CORS(app)

app.logger.info("Starting app.")

# Download the embedding model.
embedding_model = SentenceTransformer('./embedding_model')

app.logger.info("Loaded model.")

embedding_api_key = "c0f2c08f-cf76-4422-8f0e-e5743639572a"


@app.route("/")
def hello_world():
    return "Hello World!"


@app.route("/query", methods=['POST'])
def embed():
    body = request.get_json()

    if 'text' not in body or 'api_key' not in body:
        return {"error": "missing text"}, 400

    api_key = body['api_key']

    if api_key != embedding_api_key:
        return {"error": "invalid api key"}, 400

    result = embedding_model.encode(body['text'])

    return {"result": result.tolist()}


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0",
            port=int(os.environ.get("PORT", 8080)))
