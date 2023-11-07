import PyPDF2
from langchain.text_splitter import CharacterTextSplitter


class Pdf2Chunks(object):

    @staticmethod
    def getText(url: str) -> str:
        p = PDFTextExtractor(url)
        return p.extract_text()

    @staticmethod
    def getChunks(url: str, separator: str = "\n", chunk_size: int = 1000, chunk_overlap: int = 200, length_function: callable = len) -> str:
        """
        Estrae e restituisce il testo suddiviso in chunk da un documento PDF.

        Parameters:
        - url (str): L'URL del documento PDF.
        - separator (str, optional): Il separatore utilizzato per dividere il testo in chunk (predefinito: "\n").
        - chunk_size (int, optional): La dimensione dei chunk (predefinito: 1000).
        - chunk_overlap (int, optional): La sovrapposizione tra i chunk (predefinito: 200).
        - length_function (callable, optional): Una funzione che calcola la lunghezza del testo (predefinito: len).

        Returns:
        - chunks (str): Il testo suddiviso in chunk.
        """
        text_splitter = CharacterTextSplitter(
            separator=separator,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=length_function
        )
        chunks = text_splitter.split_text(Pdf2Chunks.getText(url))
        return chunks



class PDFTextExtractor:
    def __init__(self, pdf_path):
        self.pdf_path = pdf_path

    def extract_text(self):
        text = ""
        try:
            # Apre il file PDF in modalità di lettura binaria
            with open(self.pdf_path, "rb") as pdf_file:
                # Crea un oggetto PdfFileReader
                pdf_reader = PyPDF2.PdfReader(pdf_file)

                # Itera tra le pagine del PDF
                for page in pdf_reader.pages:
                    text += page.extract_text()

        except Exception as e:
            # Gestisce eventuali errori durante l'estrazione del testo
            print(f"Errore durante l'estrazione del testo: {str(e)}")
            print("Assicurarsi di aver caricato un PDF")

        return text

