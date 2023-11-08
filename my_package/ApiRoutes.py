import json
import os
from typing import Optional, Tuple
from dotenv import load_dotenv
import langchain

from my_package.chatbot import Chatbot
from my_package.pdf2chunks import Pdf2Chunks
from my_package.vectorstore import Vectorstore

class ApiRoutes(object):
    
    def __init__(self):
        self._get_env()
        self.vectorstore = Vectorstore(self._pinecone_index_name)
    
    def _get_env(self):
        load_dotenv()
        self._openai_api_key = os.getenv("OPENAI_API_KEY")
        self._pinecone_api_key = os.getenv("PINECONE_API_KEY")
        self._pinecone_api_env = os.getenv("PINECONE_API_ENV")
        self._history_file = os.getenv("HISTORY_FILE")
        self._pinecone_index_name_chat_memory = os.getenv("PINECONE_INDEX_NAME_CHAT_MEMORY")
        self._model_openai = os.getenv("MODEL_OPENAI")
        self._pinecone_index_name = os.getenv("PINECONE_INDEX_NAME")
    
    def create_index(self) -> bool:
        return self.vectorstore.create_index()
    
    def get_sources(self) -> []:
        return self.vectorstore.get_all_source()

    def delete_source(self, source: [str]) -> ([str], [str], [str]):
        return self.vectorstore.delete_data(source)
    
    def delete_all(self) -> ([str], [str], [str]):
        return self.delete_source(self.get_sources())
    
    def delete_index(self) -> bool:
        return self.vectorstore.delete_index()
    
    def upload_pdfs(self, pdfs: [str]) -> ([str], [str], [str]):
        """
        Carica il testo da file PDF in un sistema di indicizzazione e restituisce una lista dei nomi dei file che non è riuscito a caricare con successo.
        Infine salva i percorsi dei file che sono stati caricati con successo in un file JSON.

        Parameters:
        - pdfs (List[str]): Una lista di percorsi dei file PDF da elaborare.

        Returns:
        - Una tupla contenente tre liste:
            - La prima lista contiene i percorsi dei file PDF che sono stati caricati con successo.
            - La seconda lista contiene i percorsi dei file PDF che non sono stati elaborati con successo.
            - La terza lista contiene i percorsi dei file PDF che non hanno percorsi assoluti.
        """
        uploaded = []
        not_uploaded = []
        wrong_path = []
        data2save = []

        for file_path in pdfs:
            if os.path.isabs(file_path) and os.path.exists(file_path):
                file_name = os.path.basename(file_path)  # Estrae il nome del file

                text_chunks = Pdf2Chunks.getChunks(file_path)  # Estrae il testo suddiviso in chunks

                success = self.vecs_upload_data(text_chunks, file_name)  # Carica i dati in un sistema di indicizzazione
                if success:
                    uploaded.append(file_path)
                    data2save.append({
                        "path": file_path,
                        "state": "success"
                    })
                else:
                    not_uploaded.append(file_path)  # Aggiunge il nome del file alla lista dei risultati se l'operazione non ha avuto successo
                    data2save.append({
                        "path": file_path,
                        "state": "error"
                    })
            else:
                wrong_path.append(file_path)
                data2save.append({
                        "path": file_path,
                        "state": "wrong"
                    })

        self.save_in_json_file(data2save, self._history_file)

        return uploaded, not_uploaded, wrong_path
    
    def vecs_upload_data(self, data: [str], source: str) -> bool:
        success = self.vectorstore.upload_data(data, source)
        return success

    def get_vectorstore(self) -> Tuple[Optional[langchain.vectorstores], langchain.embeddings.openai.OpenAIEmbeddings]:
        return self.vectorstore.get_index()

    def init_ChatBot(self) -> bool:
        vecs, _ = self.get_vectorstore()
        settings = {
            'data_retriever' : vecs.as_retriever()
        }
        self._chatbot = Chatbot(**settings)
        if self._chatbot:
            return True
        else:
            return False

    def chat(self, query: str) -> (str, [str], str):
        return self._chatbot.chat(query)
    
    
    
    def save_in_json_file(self, data: [], file_path: str) -> bool:
        """
        Salva un array di stringhe in un file JSON, aggiungendo i dati a quelli esistenti se il file esiste.

        Parameters:
        - data (List[]): Un array di stringhe da salvare nel file JSON.
        - file_path (str): Il percorso del file JSON in cui salvare i dati.

        Returns:
        - True se il salvataggio ha avuto successo, altrimenti False.
        """
        existing_data = []
        # Verifica se il file esiste
        if os.path.exists(file_path):
            # Se il file esiste, leggi i dati esistenti
            with open(file_path, "r") as json_file:
                existing_data = json.load(json_file)
        
        existing_data.extend(data)
        try:
                
            # Scrivi l'elenco aggiornato nel file JSON
            with open(file_path, "w") as json_file:
                json.dump(existing_data, json_file, indent=4)
            
            return True
        except Exception:
            return False

