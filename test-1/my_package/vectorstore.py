import pinecone
from my_package.openai import Openai
from langchain.vectorstores import Pinecone


class Vectorstore(object):

    def __init__(self, openai_api_key, pinecone_api_key, pinecone_api_env, pinecone_index_name):
        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_api_env = pinecone_api_env
        self.pinecone_index_name = pinecone_index_name

    def get_vectorstore(self, texts):
        embeddings = Openai.getEmbeddings(self.openai_api_key)

        self._pineconeConfig(self.pinecone_api_key, self.pinecone_api_env)

        return Pinecone.from_texts([t for t in texts], embeddings, index_name=self.pinecone_index_name)

    def _pineconeConfig(self, apiKey: str, env: str) -> None:
        # initialize pinecone
        pinecone.init(
            api_key=apiKey,  # find at app.pinecone.io
            environment=env  # next to api key in console
        )
