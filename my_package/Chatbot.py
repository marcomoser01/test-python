import json
from typing import List
from langchain.llms import OpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate


class Chatbot:
    def __init__(self, openai_api_key, openai_model_name, data_retriever):
        # Recupera i dati da Pinecone
        self.data = data_retriever

        # Inizializza OpenAI
        self.openai_api_key = openai_api_key
        self.openai_model_name = openai_model_name
        
        prompt_template = '''Te sei un assistente e ti viene chiesto di interpretare il contenuto di PDF.
        L'utente ti farà domande riferite ai PDF e te dovrai rispondere in modo efficace, utilizza soltanto i dati che ti vengono caricati nel contesto.
        Nel caso in cui non ci fossero informazioni per rispondere, rispondi dicendomi 'non ho dati a riguardo'
        
        {context}

        Question: {question}
        Answer in Italian:     
        '''
        
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )
        
        chain_type_kwargs = {"prompt": PROMPT}

        
        
        llm = OpenAI(
            temperature=0, top_p=0.2, presence_penalty=0.4, frequency_penalty=0.2
        )

        self.qa = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=data_retriever,
            return_source_documents=True,
            chain_type_kwargs=chain_type_kwargs
        )
        print()


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
            item = json.loads(item.json())['metadata']['source']
            if item not in result:
                result.append(item)
        return result


    def chat(self, query: str) -> str:
        response = ""
        result = self.qa({'query': query})
        if 'non ho dati a riguardo' not in str(result['result']).lower():
            sources = Chatbot.remove_duplicates(result['source_documents'])
            response = result['result']
            response += "\nQuesti dati sono stati trovati nelle seguenti sorgenti: "
            response += ", ".join(sources)
        else:
            response = "Non sono stati caricati dati a riguardo.\n"
        return response
