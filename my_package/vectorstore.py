import os
from typing import List
import pinecone
from langchain.schema import Document
from my_package.LLM import LLM
from langchain.vectorstores import Pinecone


class Vectorstore(object):
    def __init__(
        self, openai_api_key, pinecone_api_key, pinecone_api_env, pinecone_index_name
    ):
        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_api_env = pinecone_api_env
        self.pinecone_index_name = pinecone_index_name

    def _pineconeConfig(self, apiKey: str, env: str) -> pinecone:
        # initialize pinecone
        return pinecone.init(
            api_key=apiKey,  # find at app.pinecone.io
            environment=env,  # next to api key in console
        )

    @staticmethod
    def genDocs(data: List[str], source: str) -> []:
        docs = []
        for row in data:
            doc = Document(
                page_content=row,
                metadata={"source": source},
            )
            docs.append(doc)
        return docs

    def get_index(self, index_name: str):
        '''
            Recupera l'indice e nel caso in cui non esista restituisce vectorstore None
        '''
        self._pineconeConfig(self.pinecone_api_key, self.pinecone_api_env)        
        embeddings = LLM.getEmbeddings(self.openai_api_key)
        if index_name in pinecone.list_indexes():
            index = pinecone.Index(index_name=index_name)
            vectorstore = Pinecone(index, embeddings, "text")
        else:
            vectorstore = None
            
        return vectorstore, embeddings

    def upload_data(self, texts, file_name) -> bool:
        vectorstore, embeddings = self.get_index(self.pinecone_index_name)
        docs = Vectorstore.genDocs(texts, file_name)
        
        if not vectorstore:
            # Crea l'indice solo se non esiste
            pinecone.create_index(self.pinecone_index_name, metric='cosine', dimension=1536)

        try:
            vectorstore = Pinecone.from_documents(
                docs, embeddings, index_name=self.pinecone_index_name
            )
            return True
        except pinecone.exceptions.PineconeException as e:
            print(f"Si è verificato un errore di Pinecone: {e}")
            return False