import os
import sys
from dotenv import load_dotenv
from flask import Flask, request, jsonify
from flask_cors import CORS

from rich.markdown import Markdown
from langchain.document_loaders import PyMuPDFLoader
from langchain.vectorstores import Chroma
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
# import chromadb
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import AIMessage, HumanMessage
from langchain.chains import ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from langchain.prompts.prompt import PromptTemplate

app = Flask(__name__)
# app.debug = True
# app.config["MAX_CONTENT_LENGTH"] = 5 * 1024 * 1024
CORS(app)
load_dotenv()
OPENAI_API_KEY=os.getenv("OPENAI_API_KEY")
llm = ChatOpenAI(model_name="gpt-4-1106-preview", openai_api_key=OPENAI_API_KEY, temperature=0)


def load_chunk_persist_pdf() -> Chroma:
    pdf_folder_path = "data/"
    documents = []
    pdf_paths = []
    for root, dirs, files in os.walk(pdf_folder_path):
        for file in files:
            if file.lower().endswith('.pdf'):
                pdf_path = os.path.join(root, file)
                loader = PyMuPDFLoader(pdf_path)
                pdf_paths.append(pdf_path)
                documents.extend(loader.load())
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    
    chunked_documents = text_splitter.split_documents(documents)
    pdf_ids = ['pdf' + str(i) for i in range(len(chunked_documents))]
    # print("--->", pdf_paths)
    # new_client = chromadb.EphemeralClient()
    # client = chromadb.Client()
    # if client.list_collections():
    #     consent_collection = client.create_collection("consent_collection")
    # else:
    #     print("Collection already exists")
    # new_client = chromadb.EphemeralClient()
    vectordb = Chroma.from_documents(
        documents=chunked_documents,
        embedding=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY),
        # client=new_client,
        collection_name="openai_collection",
        persist_directory="chroma_db"
        # ids=pdf_ids,
    )
    # vectordb.persist()
    return vectordb

custom_template = """"Given the following conversation and a follow up question, rephrase the follow up question to be a standalone question. 

Chat History: 
{chat_history} 
Follow Up Input: {question} 

Standalone question:"""

CONDENSE_QUESTION_PROMPT = PromptTemplate.from_template(custom_template)

# vectorstore = load_chunk_persist_pdf()
vectorstore = Chroma(persist_directory="./chroma_db", embedding_function=OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY), collection_name="openai_collection")

retriever = vectorstore.as_retriever(search_kwargs={"k":100})
# create the chain for allowing us to chat with the document

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)

rag_chain = ConversationalRetrievalChain.from_llm(
    llm, retriever, condense_question_prompt=CONDENSE_QUESTION_PROMPT, return_source_documents=True
)

chat_history = []

@app.route('/api/proprietary-assistant', methods = ['POST'])
def proprietary_assistant():
    chat_history = []
    query = request.json["prompt"]
    print(query)
    if query == "exit" or query == "quit" or query == "q":
        print('Exiting')
        sys.exit()
    # we pass in the query to the LLM, and print out the response. As well as
    # our query, the context of semantically relevant information from our
    # vector store will be passed in, as well as list of our chat history
    
    ai_msg = rag_chain({'question': query, 'chat_history': chat_history})

    print(ai_msg)

    source_details = ai_msg['source_documents']
    # print('---->', source_details)

    # chat_history.extend([HumanMessage(content=query), ai_msg])
    chat_history = [(query, ai_msg["answer"])]
    
    # client.create_message(prompt)
    # client.create_run()
    # gpt_output = client.output()
    # conversation.append(gpt_output)
    # markdown = Markdown(ai_msg, code_theme="one-dark")
    # print('-------------------MARK-----------------', markdown)
    
    return jsonify({'result': ai_msg['answer']})

@app.route('/', methods = ['GET'])
def index():
    return "Hello"

if __name__ == "__main__":
    print("kkk")
    
    app.run(host='0.0.0.0', port=5099)