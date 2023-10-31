import os
from my_package.LLM import LLM
from my_package.vectorstore import Vectorstore
from my_package.pdf2chunks import Pdf2Chunks
from dotenv import load_dotenv

class PDF_ChatBot(object):
    _index_name = ""
    def get_index_name(self):
        return self._index_name

    def set_index_name(self, index_name):
        self._index_name = index_name
    
    def __init__(self, index_name):
        load_dotenv()
        self._index_name = index_name
        self.openai_api_key = os.getenv("OPENAI_API_KEY")
        self.vectorstore = Vectorstore(
            os.getenv("OPENAI_API_KEY"),
            os.getenv("PINECONE_API_KEY"),
            os.getenv("PINECONE_API_ENV"),
            self._index_name,
        )

    def delete_index(self) -> bool:
        return self.vectorstore.delete_index(self._index_name)

    def get_vectorstore(self):
        return self.vectorstore.get_index(self._index_name)

    def delete_data_vectorstore(self, filename: []):
        return self.vectorstore.delete_data(self._index_name, filename)

    def get_sources(self) -> []:
        return self.vectorstore.get_all_source(self._index_name)

    def vecs_upload_data(self, data, source: str) -> bool:
        success = self.vectorstore.upload_data(data, source)
        return success

    def load_pdfs(self, pdfs: [str]) -> ([str], [str]):
        """
        Carica il testo da file PDF in un sistema di indicizzazione e restituisce una lista dei nomi dei file che non è riuscito a caricare con successo.

        parameters:
        - pdfs (List[str]): Una lista di percorsi dei file PDF da elaborare.
        
        returns:
        - Una tupla contenente due liste:
            - La prima lista contiene i percorsi dei file PDF che sono stati caricati con successo.
            - La seconda lista contiene i percorsi dei file PDF che non sono stati elaborati con successo.
        """
        uploaded = []
        not_uploaded = []

        for pdf in pdfs:
            abs_path = os.path.abspath(pdf) # Ottiene la path assoluta del file
            file_name = os.path.basename(abs_path)  # Estrae il nome del file
            print(f"Cerco di caricare il file {abs_path}")
            
            text_chunks = Pdf2Chunks.getChunks(abs_path)  # Estrae il testo suddiviso in chunks

            success = self.vecs_upload_data(text_chunks, file_name)  # Carica i dati in un sistema di indicizzazione
            if success:
                print(f"Il file {file_name} è stato caricato correttamente")
                uploaded.append(abs_path)
            else:
                not_uploaded.append(abs_path)  # Aggiunge il nome del file alla lista dei risultati se l'operazione non ha avuto successo

        return uploaded, not_uploaded

    def init_LLM(self, vecs):
        return LLM(self.openai_api_key, vecs)
