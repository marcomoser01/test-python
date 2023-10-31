import streamlit as st
from my_package.PDF_ChatBot import PDF_ChatBot
from my_package.LLM import LLM
from my_package.pdf2chunks import Pdf2Chunks
from my_package.htmlTemplates import css, bot_template, user_template

class Frontend(object):

    @staticmethod
    def load_pdfs(pdfs, pdf_ChatBot: PDF_ChatBot):
        for pdf in pdfs:
            # Salva il file temporaneamente su disco
            with open(pdf.name, "wb") as f:
                f.write(pdf.read())

            # Ottieni l'URL del file salvato
            file_url = f.name
            text_chunks = Pdf2Chunks.getChunks(file_url)

            # create vector store
            return pdf_ChatBot.vecs_upload_data(text_chunks, file_url)

    @staticmethod
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

    @staticmethod
    def main():
        pdf_ChatBot = PDF_ChatBot()
        st.set_page_config(page_title="Chat with multiple PDFs", page_icon=":books:")
        st.write(css, unsafe_allow_html=True)

        if "conversation" not in st.session_state:
            st.session_state.conversation = None
        if "chat_history" not in st.session_state:
            st.session_state.chat_history = None

        st.header("Chat with multiple PDFs :books:")
        user_question = st.text_input("Ask a question about your documents:")
        if user_question:
            Frontend.handle_userinput(user_question)

        with st.sidebar:
            st.subheader("Your documents")
            pdf_docs = st.file_uploader(
                "Upload your PDFs here and click on 'Process'", accept_multiple_files=True
            )
            if st.button("Process"):
                with st.spinner("Processing"):
                    if pdf_docs:
                        vectorstore = Frontend.load_pdfs(pdf_docs, pdf_ChatBot)

                    if vectorstore:
                        # create conversation chain
                        st.session_state.conversation = LLM.get_conversation_chain(
                            vectorstore
                        )

if __name__ == "__main__":
    Frontend.main()
