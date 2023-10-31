import PyPDF2
from langchain.text_splitter import CharacterTextSplitter


class Pdf2Chunks(object):

    @staticmethod
    def getText(url: str) -> str:
        p = PDFTextExtractor(url)
        return p.extract_text()

    @staticmethod
    def getChunks(url: str) -> str:
        #ToDO impostarli come parametri esterni
        text_splitter = CharacterTextSplitter(
            separator="\n",
            chunk_size=1000,
            chunk_overlap=200,
            length_function=len
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

