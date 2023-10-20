from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.embeddings import OpenAIEmbeddings


class LLM(object):
    
    @staticmethod
    def getEmbeddings(apiKey):
        return OpenAIEmbeddings(openai_api_key=apiKey)
    
    @staticmethod
    def get_conversation_chain(vectorstore):
        llm = ChatOpenAI()

        memory = ConversationBufferMemory(
            memory_key='chat_history',
            return_messages=True
            )
        conversation_chain = ConversationalRetrievalChain.from_llm(
            llm=llm,
            retriever=vectorstore.as_retriever(),
            memory=memory
        )
        return conversation_chain
