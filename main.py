import streamlit as st
import boto3
import os
import uuid
from dotenv import load_dotenv
from langchain_aws import BedrockEmbeddings
from langchain_community.chat_models import BedrockChat
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA

# Load environment variables from .env file
load_dotenv()

# Retrieve AWS credentials and other environment variables
aws_access_key_id = os.getenv("AWS_ACCESS_KEY_ID")
aws_secret_access_key = os.getenv("AWS_SECRET_ACCESS_KEY")
region = os.getenv("region")
BUCKET_NAME = os.getenv("BUCKET_NAME")

s3_client = boto3.client(
    "s3",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region
)

bedrock_client = boto3.client(
    service_name="bedrock-runtime",
    aws_access_key_id=aws_access_key_id,
    aws_secret_access_key=aws_secret_access_key,
    region_name=region
)

bedrock_embedding = BedrockEmbeddings(model_id="amazon.titan-embed-image-v1", client=bedrock_client)

folder_path = "/tmp/"

def get_unique_id():
    return str(uuid.uuid4())

def split_text(pages, chunk_size, chunk_overlap):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    docs = text_splitter.split_documents(pages)
    return docs

def create_vector_store(request_id, documents):
    vectorstore_faiss = FAISS.from_documents(documents, bedrock_embedding)
    file_name = "my_faiss"
    folder_path = "/tmp/"
    vectorstore_faiss.save_local(index_name=file_name, folder_path=folder_path)

    faiss_path = f"{folder_path}{file_name}.faiss"
    pkl_path = f"{folder_path}{file_name}.pkl"
    
    if os.path.exists(faiss_path) and os.path.exists(pkl_path):
        s3_client.upload_file(Filename=faiss_path, Bucket=BUCKET_NAME, Key="my_faiss.faiss")
        s3_client.upload_file(Filename=pkl_path, Bucket=BUCKET_NAME, Key="my_faiss.pkl")
        return True
    else:
        raise FileNotFoundError("One or more files not found for upload.")

def load_index():
    os.makedirs(folder_path, exist_ok=True)
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.faiss", Filename=f"{folder_path}my_faiss.faiss")
    s3_client.download_file(Bucket=BUCKET_NAME, Key="my_faiss.pkl", Filename=f"{folder_path}my_faiss.pkl")
    vectorstore_faiss = FAISS.load_local(index_name="my_faiss", folder_path=folder_path, embeddings=bedrock_embedding, allow_dangerous_deserialization=True)
    return vectorstore_faiss

def get_llm():
    llm = BedrockChat(model_id="anthropic.claude-3-haiku-20240307-v1:0", client=bedrock_client)
    return llm

def get_response(llm, vectorstore, question):
    prompt_template = """
    Human: Please use the given context to provide a concise answer to the question.
    If you don't know the answer, just say that you don't know, don't try to make up an answer.
    <context>
    {context}
    </context>

    Question: {question}

    Assistant:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=vectorstore.as_retriever(
            search_type="similarity", search_kwargs={"k": 5}
        ),
        return_source_documents=True,
        chain_type_kwargs={"prompt": PROMPT}
    )
    answer = qa({"query": question, "messages": [{"role": "user", "content": question}]})
    return answer['result']

def main():
    st.sidebar.title("Navigation")
    selection = st.sidebar.radio("Go to", ["Upload PDF", "Ask Question"])

    if selection == "Upload PDF":
        st.title("Admin Site")
        st.write("Upload a PDF file for processing.")

        uploaded_file = st.file_uploader("Choose a PDF file", type="pdf")

        if uploaded_file is not None:
            request_id = get_unique_id()
            st.write(f"Request Id: {request_id}")

            saved_file_name = f"{request_id}.pdf"
            with open(saved_file_name, mode="wb") as w:
                w.write(uploaded_file.getvalue())

            loader = PyPDFLoader(saved_file_name)
            pages = loader.load_and_split()

            st.write(f"Total Pages: {len(pages)}")

            splitted_docs = split_text(pages, 1000, 200)
            st.write(f"Splitted Docs Length: {len(splitted_docs)}")
            st.write("================")
            st.write(splitted_docs[0])
            st.write("================")
            st.write(splitted_docs[1])

            st.write("Creating the Vector Store")
            try:
                result = create_vector_store(request_id, splitted_docs)
            except Exception as e:
                st.write(f"An error occurred: {e}")
                result = False

            if result:
                st.write("Hurray!! PDF processed successfully")

    elif selection == "Ask Question":
        st.title("Client Site")

        try:
            vectorstore = load_index()
            st.write("FAISS index loaded successfully.")
        except Exception as e:
            st.write(f"Error loading FAISS index: {e}")
            return

        faiss_index = FAISS.load_local(
            index_name="my_faiss",
            folder_path=folder_path,
            embeddings=bedrock_embedding,
            allow_dangerous_deserialization=True
        )

        st.write("INDEX IS READY")
        question = st.text_input("Please ask your question")

        if st.button("Ask Question"):
            with st.spinner("Querying. . ."):
                llm = get_llm()
                st.write(get_response(llm, faiss_index, question))
                st.success("Done")

if __name__ == "__main__":
    main()
