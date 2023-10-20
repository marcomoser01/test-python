import streamlit as st
import os
from my_package.openai import Openai
from my_package.vectorstore import Vectorstore
from my_package.pdf2chunks import Pdf2Chunks
from dotenv import load_dotenv
from my_package.htmlTemplates import css, bot_template, user_template

pinecone_index_name = "pdf-index"


def get_vectorstore(texts):
    vectorestore = Vectorstore(
        os.getenv("OPENAI_API_KEY"),
        os.getenv("PINECONE_API_KEY"),
        os.getenv("PINECONE_API_ENV"),
        pinecone_index_name,
    )
    return vectorestore.get_vectorstore(texts)


def handle_userinput(user_question):
    response = st.session_state.conversation({"question": user_question})
    st.session_state.chat_history = response["chat_history"]

    for i, message in enumerate(st.session_state.chat_history):
        if i % 2 == 0:
            st.write(
                user_template.replace("{{MSG}}", message.content),
                unsafe_allow_html=True,
            )
        else:
            st.write(
                bot_template.replace("{{MSG}}", message.content), unsafe_allow_html=True
            )


def load_pdfs(pdfs):
    for pdf in pdfs:
        # Salva il file temporaneamente su disco
        with open(pdf.name, "wb") as f:
            f.write(pdf.read())

        # Ottieni l'URL del file salvato
        file_url = f.name
        text_chunks = Pdf2Chunks.getChunks(file_url)

        # create vector store
        return get_vectorstore(text_chunks)


def main():
    load_dotenv()
    st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
    st.write(css, unsafe_allow_html=True)

    if "conversation" not in st.session_state:
        st.session_state.conversation = None
    if "chat_history" not in st.session_state:
        st.session_state.chat_history = None

    st.header("Chat with multiple PDFs :books:")
    user_question = st.text_input("Ask a question about your documents:")
    if user_question:
        handle_userinput(user_question)

    with st.sidebar:
        st.subheader("Your documents")
        pdf_docs = st.file_uploader(
            "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
        )
        if st.button("Process"):
            with st.spinner("Processing"):
                if pdf_docs:
                    vectorstore = load_pdfs(pdf_docs)

                if vectorstore:
                    # create conversation chain
                    st.session_state.conversation = Openai.get_conversation_chain(
                        vectorstore
                    )


if __name__ == "__main__":
    main()
