import os
from typing import List, Optional, Tuple, Union, Dict
import langchain
import numpy as np
import pinecone
from langchain.schema import Document
from langchain.vectorstores import Pinecone
from langchain.embeddings import OpenAIEmbeddings


class Vectorstore(object):
    def __init__(
        self,
        openai_api_key: str,
        pinecone_api_key: str,
        pinecone_api_env: str,
        pinecone_index_name: str,
    ) -> None:
        """
        Inizializza un'istanza di Vectorstore.

        Parameters:
        - openai_api_key (str): Chiave API di OpenAI.
        - pinecone_api_key (str): Chiave API di Pinecone.
        - pinecone_api_env (str): Ambiente di Pinecone (es. "production" o "development").
        - pinecone_index_name (str): Nome dell'indice di Pinecone.
        """
        self.openai_api_key = openai_api_key
        self.pinecone_api_key = pinecone_api_key
        self.pinecone_api_env = pinecone_api_env
        self.pinecone_index_name = pinecone_index_name
        self._pineconeConfig(pinecone_api_key, pinecone_api_env)

    def _pineconeConfig(self, apiKey: str, env: str) -> Union[pinecone.Index, None]:
        """
        Configura Pinecone con la chiave API e l'ambiente specificati.

        Parameters:
        - apiKey (str): Chiave API di Pinecone.
        - env (str): Ambiente di Pinecone.

        Returns:
        - pinecone.Index or None: Oggetto Pinecone configurato o None in caso di errore.
        """
        try:
            # Inizializza Pinecone
            return pinecone.init(
                api_key=apiKey,  # Trovato su app.pinecone.io
                environment=env,  # Trovato accanto all'API key nella console
            )
        except pinecone.exceptions.PineconeAPIException as e:
            print(f"Errore Pinecone API durante l'inizializzazione: {e}")
            return None
        except pinecone.exceptions.PineconeConnectionException as e:
            print(f"Errore di connessione a Pinecone durante l'inizializzazione: {e}")
            return None

    def create_index(
        self, index_name, metric: str = "cosine", dimension: int = 1536
    ) -> bool:
        if index_name in pinecone.list_indexes():
            return False
        else:
            pinecone.create_index(
                self.pinecone_index_name, metric=metric, dimension=dimension
            )
            return True

    @staticmethod
    def getEmbeddings(apiKey):
        return OpenAIEmbeddings(openai_api_key=apiKey)

    @staticmethod
    def genDocs(data: List[str], source: str) -> List[Document]:
        """
        Genera oggetti Document dai dati di input.

        Parameters:
        - data (List[str]): Lista di dati da convertire in oggetti Document.
        - source (str): Sorgente associata ai documenti.

        Returns:
        - docs (List[Document]): Lista di oggetti Document generati.
        """
        docs = []
        for row in data:
            doc = Document(
                page_content=row,
                metadata={"source": source},
            )
            docs.append(doc)
        return docs

    def get_index(
        self, index_name: str
    ) -> Tuple[Optional[Pinecone], langchain.embeddings.openai.OpenAIEmbeddings]:
        """
        Recupera l'indice di Pinecone o restituisce None se non esiste.

        Parameters:
        - index_name (str): Nome dell'indice da recuperare.

        Returns:
        - vectorstore (Pinecone): Oggetto Vectorstore se l'indice esiste, altrimenti None.
        - embeddings: Embeddings associati all'indice.
        """
        try:
            self._pineconeConfig(self.pinecone_api_key, self.pinecone_api_env)
            embeddings = Vectorstore.getEmbeddings(self.openai_api_key)
            if index_name in pinecone.list_indexes():
                index = pinecone.Index(index_name=index_name)
                vectorstore = Pinecone(index, embeddings, "text")
            else:
                vectorstore = None

            return vectorstore, embeddings
        except Exception as e:
            print(f"Errore durante il recupero dell'indice: {e}")
            return None, None

    def delete_index(self, index_name: str) -> bool:
        """
        Elimina un indice Pinecone.

        Parameters:
        - index_name (str): Nome dell'indice da eliminare.

        Returns:
        - bool: True se l'eliminazione ha avuto successo, altrimenti False.
        """
        try:
            # Configura Pinecone
            self._pineconeConfig(self.pinecone_api_key, self.pinecone_api_env)

            if index_name not in pinecone.list_indexes():
                print(f"Errore, l'indice specificato non è presente su Pinecone")
                return False

            # Elimina l'indice specificato
            pinecone.delete_index(index_name)

            return True  # Indica che l'eliminazione è riuscita
        except pinecone.exceptions.PineconeException as e:
            print(f"Errore Pinecone API durante l'eliminazione dell'indice: {e}")
            return False  # Indica che si è verificato un errore
        except pinecone.exceptions.PineconeConnectionException as e:
            print(
                f"Errore di connessione a Pinecone durante l'eliminazione dell'indice: {e}"
            )
            return False  # Indica che si è verificato un errore di connessione

    def upload_data(
        self,
        texts: List[str],
        source: str,
        metric: str = "cosine",
        dimension: int = 1536,
    ) -> bool:
        """
        Carica dati nell'indice di Pinecone. Nel caso non esistesse viene creato

        Parameters:
        - texts (List[str]): Lista di testi da caricare nell'indice.
        - source (str): Nome del file sorgente associato ai dati.
        - metric (str): Metrica dell'indice pinecone
        - dimension (int): Dimensione dell'indice

        Returns:
        - bool: True se il caricamento ha avuto successo, altrimenti False.
        """
        vectorstore, embeddings = self.get_index(self.pinecone_index_name)
        docs = Vectorstore.genDocs(texts, source)

        if not vectorstore:
            try:
                # Crea l'indice solo se non esiste
                pinecone.create_index(
                    self.pinecone_index_name, metric=metric, dimension=dimension
                )
                vectorstore, _ = self.get_index(self.pinecone_index_name)
            except pinecone.exceptions.PineconeException as e:
                print(
                    f"Si è verificato un errore di Pinecone durante la creazione dell'indice: {e}"
                )
                return False

        try:
            # Provare a vedere soluzioni alternative
            # TODO provare a cambiare in pinecone.Pinecone
            vectorstore = Pinecone.from_documents(
                docs, embeddings, index_name=self.pinecone_index_name
            )
            return True
        except pinecone.exceptions.PineconeException as e:
            print(
                f"Si è verificato un errore di Pinecone durante il caricamento dei dati: {e}"
            )
            return False

    def delete_data(
        self, index_name: str, sources: List[str], dimension: int = 1536
    ) -> ([str], [str], [str]):
        """
        Elimina dati dall'indice di Pinecone.

        Parameters:
        - index_name (str): Nome dell'indice da cui eliminare i dati.
        - sources (List[str]): Lista di nomi di file da cui eliminare i dati.
        - dimension (int): Dimensione dell'indice di Pinecone.

        Returns:
        - Una tupla contenente tre liste:
            - La prima lista contiene i nomi di file per i quali l'eliminazione ha avuto successo.
            - La seconda lista contiene i nomi di file che non sono stati trovati nell'indice.
            - La terza lista contiene i nomi di file per i quali si è verificato un errore durante l'eliminazione.
        """
        success, not_found, error = [], [], []
        self._pineconeConfig(self.pinecone_api_key, self.pinecone_api_env)
        index = pinecone.Index(index_name)
        ids_and_source = self._get_ids_and_source_from_index(index, dimension=dimension)
        for source in sources:
            ids = [
                item["id"] for item in ids_and_source if str(item["source"]) == source
            ]

            if len(ids) <= 0:
                not_found.append(source)
            try:
                index.delete(ids=ids, namespace="")
                success.append(source)
            except pinecone.exceptions.PineconeException:
                error.append(source)

        return success, not_found, error

    def get_all_source(self, index_name: str, dimension: int = 1536) -> List[str]:
        """
        Restituisce tutte le sorgenti dei dati che sono presenti sul vectorstore

        Parameters:
        - index_name (str): Nome dell'indice sul quale si vuole eseguire la ricerca.
        - dimension (int): Dimensione dell'indice pinecone

        Returns:
        - []: restituisce una stringa in cui vengono elencati tutte le sorgenti presenti
        """
        self._pineconeConfig(self.pinecone_api_key, self.pinecone_api_env)
        if index_name in pinecone.list_indexes():
            index = pinecone.Index(index_name)
        else:
            return
        ids_and_source = self._get_ids_and_source_from_index(index, dimension=dimension)
        # Crea un insieme (set) per raccogliere le diverse sorgenti
        unique_sources = []

        # Scorrere l'array e aggiungere le sorgenti uniche all'insieme
        for item in ids_and_source:
            if str(item["source"]) not in unique_sources:
                unique_sources.append(str(item["source"]))

        return unique_sources

    def _get_ids_and_source_from_query(
        self, index: pinecone.Index, input_vector: List[float], top_k: int = 10000
    ) -> List[Dict[str, str]]:
        """
        Ottiene oggetti con campi 'id' e 'source' dai risultati di una query in un indice Pinecone.

        Parameters:
        - index (pinecone.Index): Il riferimento all'indice Pinecone.
        - input_vector (list): Il vettore di input per la query.
        - top_k (int, optional): Il numero massimo di risultati da restituire (predefinito a 10000).

        Returns:
        - results (list): Una lista di dizionari, ciascuno rappresenta un oggetto con campi 'id' e 'source'.
        """
        try:
            results = []
            query_results = index.query(
                vector=input_vector,
                top_k=top_k,
                include_values=False,
                include_metadata=True,
            )
            for item in query_results["matches"]:
                results.append({"id": item["id"], "source": item["metadata"]["source"]})
            return results
        except pinecone.exceptions.PineconeException as e:
            print(f"Si è verificato un errore di Pinecone durante la query: {e}")
            return []

    def _get_ids_and_source_from_index(
        self, index: pinecone.Index, dimension: int
    ) -> List[Dict[str, str]]:
        """
        Ottiene oggetti con campi 'id' e 'source' da un indice Pinecone utilizzando vettori casuali.

        Parameters:
        - index: Il riferimento all'indice Pinecone.
        - dimension (int): Il numero di dimensioni dei vettori casuali.

        Returns:
        - ids_and_source (list): Una lista di dizionari, ciascuno rappresenta un oggetto con campi 'id' e 'source'.
        """
        try:
            num_vectors = index.describe_index_stats()["total_vector_count"]
            ids_and_source = []
            while len(ids_and_source) < num_vectors:
                # Creazione di un vettore casuale
                input_vector = np.random.rand(dimension).tolist()
                ids_and_source.extend(
                    self._get_ids_and_source_from_query(index, input_vector)
                )
            return ids_and_source
        except pinecone.exceptions.PineconeException as e:
            print(
                f"Si è verificato un errore di Pinecone durante l'ottenimento degli ID e delle fonti: {e}"
            )
            return []
