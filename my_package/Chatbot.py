import json
import os
from typing import List
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
from my_package.vectorstore import Vectorstore
from dotenv import load_dotenv


class Chatbot:
    def __init__(self, data_retriever):
        self._get_env()
        self.chat_memory_vec = Vectorstore(self._pinecone_index_name)
        self.chat_memory_vec.create_index()
        settings_llm = {
            'temperature': self._openai_temperature, 
            'top_p': self._openai_top_p,
            'presence_penalty': self._openai_presence_penalty,
            'frequency_penalty': self._openai_frequency_penalty
        }

        PROMPT = PromptTemplate(
            template=self._openai_prompt_template, 
            input_variables=["context", "history", "question"]
        )

        memory = ConversationBufferMemory(memory_key="history", input_key="question")

        chain_type_kwargs = {"verbose": True, "prompt": PROMPT, "memory": memory}

        self.qa = RetrievalQA.from_chain_type(
            llm=OpenAI(**settings_llm),
            chain_type="stuff",
            retriever=data_retriever,
            verbose=True,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs,
        )
        print()

    def _get_env(self):
        load_dotenv()
        self._openai_api_key = os.getenv("OPENAI_API_KEY")
        self._openai_model_name = os.getenv("OPENAI_MODEL_NAME")
        self._openai_temperature = float(os.getenv("OPENAI_TEMPERATURE"))
        self._openai_top_p = float(os.getenv("OPENAI_TOP_P"))
        self._openai_presence_penalty = float(os.getenv("OPENAI_PRESENCE_PENALTY"))
        self._openai_frequency_penalty = float(os.getenv("OPENAI_FREQUENCY_PENALTY"))
        self._openai_prompt_template = os.getenv("OPENAI_PROMPT_TEMPLATE")
        self._pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")

    @staticmethod
    def remove_duplicates(data: List[str]) -> List[str]:
        """Prese in ingresso una lista di stringhe restituisce una lista di stringhe, rimuovendo tutti i duplicati

        Args:
            sources (List[str]): Lista sulla quale si vuole lavorare

        Returns:
            List[str]: Lista di partenza, senza la presenza di elementi duplicati
        """
        result = []
        for item in data:
            item = json.loads(item.json())["metadata"]["source"]
            if item not in result:
                result.append(item)
        return result

    def chat(self, query: str) -> (str, [str], str):
        response = ""
        sources = []
        result = self.qa({"query": query})
        self.chat_memory_vec.upload_data([query], "chat_memory")
        if "non ho dati a riguardo" not in str(result["result"]).lower():
            sources = Chatbot.remove_duplicates(result["source_documents"])
            response = result["result"]
            response += "\nQuesti dati sono stati trovati nelle seguenti sorgenti: "
        else:
            response = "Non sono stati caricati dati a riguardo.\n"
        return response, sources, self._openai_model_name
