import os
from my_package.vectorstore import Vectorstore
from my_package.pdf2chunks import Pdf2Chunks
from dotenv import load_dotenv


class App(object):
    pinecone_index_name = "pdf-index"
    
    def __init__(self):
        load_dotenv()
        self.vectorstore = Vectorstore(
            os.getenv("OPENAI_API_KEY"),
            os.getenv("PINECONE_API_KEY"),
            os.getenv("PINECONE_API_ENV"),
            self.pinecone_index_name,
        )

    def get_vectorstore(self):
        return self.vectorstore.get_index(self.pinecone_index_name)

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

