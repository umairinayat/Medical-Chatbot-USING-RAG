from flask import Flask, render_template, jsonify, request

from langchain_community.vectorstores import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_community.embeddings import HuggingFaceEmbeddings


from langchain.prompts import PromptTemplate
from langchain.llms import CTransformers
from langchain.chains import RetrievalQA
from dotenv import load_dotenv
from src.prompt import *
from src.helper import *
import os

app=Flask(__name__)

load_dotenv()

PINECONE_API_KEY = os.environ.get('PINECONE_API_KEY')
PINECONE_API_ENV = os.environ.get('PINECONE_API_ENV')
os.environ['PINECONE_API_KEY'] = PINECONE_API_KEY




# Download HuggingFace embeddings
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

index_name="medical-chatbot"

docsearch=PineconeVectorStore.from_existing_index(index_name, embeddings)

# Define the prompt template
PROMPT = PromptTemplate(template=prompt_template, input_variables=["context", "question"])

chain_type_kwargs = {"prompt": PROMPT}

# Configure the LLM
llm1=CTransformers(model="model\llama-2-7b-chat.ggmlv3.q4_0.bin",
                    model_type="llama",
                    config={'max_new_tokens':512,
                            'temperature':0.8})

retriever1 = docsearch.as_retriever(search_type="similarity", search_kwargs={'k': 3})

qa=RetrievalQA.from_chain_type(
    llm=llm1,
    chain_type="stuff",
    retriever=retriever1,
    return_source_documents=True,
    chain_type_kwargs=chain_type_kwargs)

@app.route("/")
def index():
    return render_template('chat.html')

@app.route("/get", methods=["GET", "POST"])
def chat():
    msg = request.form["msg"]
    input = msg
    print(input)
    result = qa.invoke({"query": input})
    print("Response:", result["result"])
    return str(result["result"])

if __name__ == '__main__':
    app.run(host="0.0.0.0", port=8080, debug=True)
