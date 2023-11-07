import json
import os
from flask import Flask, request, jsonify
import langchain
from my_package.Chatbot import Chatbot
from my_package.vectorstore import Vectorstore
from my_package.pdf2chunks import Pdf2Chunks
from dotenv import load_dotenv



class PDF_ChatBot(object):
    _index_name = ""
    _openai_api_key = ""
    _pinecone_api_key = ""
    _pinecone_api_env = ""
    _chatbot = None

#TODO togliere tutti i get and setter

    def get_openai_api_key(self):
        return self._openai_api_key


    def __init__(self, index_name):
        load_dotenv()
        self._index_name = index_name
        self._openai_api_key = os.getenv("OPENAI_API_KEY")
        self._pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self._pinecone_api_env = os.getenv("PINECONE_API_ENV")

        self.vectorstore = Vectorstore(
            openai_api_key=self._openai_api_key,
            pinecone_api_key=self._pinecone_api_key,
            pinecone_api_env=self._pinecone_api_env,
            pinecone_index_name=self._index_name,
        )

    def delete_index(self) -> bool:
        return self.vectorstore.delete_index(self._index_name)

    def get_vectorstore(self) -> langchain:
        return self.vectorstore.get_index(self._index_name)

    def delete_data_vectorstore(self, filename: []):
        return self.vectorstore.delete_data(self._index_name, filename)

    def get_sources(self) -> []:
        return self.vectorstore.get_all_source(self._index_name)

    def vecs_upload_data(self, data, source: str) -> bool:
        success = self.vectorstore.upload_data(data, source)
        return success

    def upload_pdfs(self, pdfs: [str], history_file: str) -> ([str], [str], bool):
        """
        Carica il testo da file PDF in un sistema di indicizzazione e restituisce una lista dei nomi dei file che non è riuscito a caricare con successo.
        Infine salva i titoli dei file nel file json passato

        parameters:
        - pdfs (List[str]): Una lista di percorsi dei file PDF da elaborare.
        - hsitory_file (str): Path del file sul quale salvare la storia dei file caricati

        returns:
        - Una tupla contenente due liste:
            - La prima lista contiene i percorsi dei file PDF che sono stati caricati con successo.
            - La seconda lista contiene i percorsi dei file PDF che non sono stati elaborati con successo.
        """
        uploaded = []
        not_uploaded = []

        for pdf in pdfs:
            abs_path = os.path.abspath(pdf)  # Ottiene la path assoluta del file
            file_name = os.path.basename(abs_path)  # Estrae il nome del file
            print(f"Cerco di caricare il file {abs_path}")

            text_chunks = Pdf2Chunks.getChunks(
                abs_path
            )  # Estrae il testo suddiviso in chunks

            success = self.vecs_upload_data(
                text_chunks, file_name
            )  # Carica i dati in un sistema di indicizzazione
            if success:
                print(f"Il file {file_name} è stato caricato correttamente")
                uploaded.append(abs_path)
            else:
                not_uploaded.append(
                    abs_path
                )  # Aggiunge il nome del file alla lista dei risultati se l'operazione non ha avuto successo

        history_state = PDF_ChatBot.save_in_json_file(uploaded, history_file)

        return uploaded, not_uploaded, history_state

    def init_ChatBot(self, openai_model_name, vecs: langchain) -> bool:
        settings = {
            'openai_api_key' : self._openai_api_key , 
            'openai_model_name' : openai_model_name, 
            'data_retriever' : vecs.as_retriever()
        }
        self._chatbot = Chatbot(**settings)
        if self._chatbot:
            return True
        else:
            return False

    def interact_with_chatbot(self, query: str) -> str:
        return self._chatbot.chat(query)

    @staticmethod
    def save_in_json_file(data: [str], file_path: str) -> bool:
        """
        Salva un array di stringhe in un file JSON, aggiungendo i dati a quelli esistenti se il file esiste.

        Parameters:
        - data (List[str]): Un array di stringhe da salvare nel file JSON.
        - file_path (str): Il percorso del file JSON in cui salvare i dati.

        Returns:
        - True se il salvataggio ha avuto successo, altrimenti False.
        """
        try:
            # Leggi i dati esistenti se il file esiste
            existing_data = []
            if os.path.exists(file_path):
                with open(file_path, "r") as json_file:
                    existing_data = json.load(json_file)
            
            # Aggiungi i nuovi dati all'elenco esistente
            existing_data.extend(data)
            
            # Scrivi l'elenco aggiornato nel file JSON
            with open(file_path, "w") as json_file:
                json.dump(existing_data, json_file, indent=4)
            
            return True
        except Exception as e:
            print(f"Si è verificato un errore durante il salvataggio del file JSON: {str(e)}")
            return False

