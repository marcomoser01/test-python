from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.embeddings import OpenAIEmbeddings


class LLM(object):
    
    @staticmethod
    def getEmbeddings(apiKey):
        return OpenAIEmbeddings(openai_api_key=apiKey)

    @staticmethod
    def get_conversation_chain(vectorstore) -> ConversationalRetrievalChain:
        llm = ChatOpenAI()

        memory = ConversationBufferMemory(
            memory_key="chat_history", return_messages=True
        )
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm, retriever=vectorstore.as_retriever(), memory=memory
        )
        return conversation_chain
    
    def __init__(self, vecs):
        retriever = vecs.as_retriever()
        llm = ChatOpenAI()
        self.qa = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

    def do_query(self, query):
        try:
            llm_response = self.qa(query)
            return llm_response["result"]
        except Exception as err:
            print('Exception occurred. Please try again', str(err))