from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationalRetrievalChain, RetrievalQA
from langchain.embeddings import OpenAIEmbeddings
from langchain.schema import Document

from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
import json


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


    
    def __init__(self, api_key, vecs, chain_type="stuff", return_source_documents=True):
        # retriever = vecs.as_retriever()
        # llm = ChatOpenAI()
        # self.qa = RetrievalQA.from_chain_type(llm=llm, chain_type=chain_type, retriever=retriever, return_source_documents=return_source_documents)
        llm = OpenAI(temperature=0, openai_api_key=api_key)
        self.chain = load_qa_chain(llm, chain_type=chain_type)
        self.vecs = vecs

    def do_query(self, query: str) -> (str, [str]):
        # #ToDo controlla che la sorgente sia corretta
        # try:
        #     llm_response = self.qa({"query": query})
        #     sources = []
        #     for item in llm_response['source_documents']:
        #         source = Document.parse_obj(item).metadata.get('source')
        #         if source not in sources:
        #             sources.append(source)
        #     return llm_response["result"], sources
        # except Exception as err:
        #     print('Exception occurred. Please try again', str(err))
        sources = []
        docs = self.vecs.similarity_search(query)
        for document in docs:
            source = json.loads(document.json())['metadata']['source']
            if source not in sources:
                sources.append(source)
        return self.chain.run(input_documents=docs, question=query), sources
