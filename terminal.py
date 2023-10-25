#!/usr/bin/env python3

import os, argparse, json
import shutil
from typing import List
from my_package.app import App
from my_package.LLM import LLM

def clean_pycache(path):
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if dir == "__pycache__":
                shutil.rmtree(os.path.join(root, dir))

def get_file_paths() -> List[str]:
    """
    Ottiene i percorsi dei file da parte dell'utente e restituisce una lista di percorsi validi.

    L'utente inserisce i percorsi dei file da caricare uno alla volta. Per terminare l'inserimento,
    può premere Invio senza inserire alcun percorso.

    Returns:
        - Una lista contiene i percorsi dei file inseriti dall'utente.
    """

    file_paths = []  # Inizializza una lista vuota per i percorsi dei file

    while True:
        file_path = input(
            "Inserisci il percorso di un file da caricare (o premi Invio per terminare): "
        )

        if not file_path:
            break  # L'utente ha premuto Invio per terminare l'inserimento

        if os.path.exists(file_path):
            file_paths.append(file_path)
        else:
            print(f"Il file {file_path} non è stato trovato")

    return file_paths

def chat(vecs):
    """
    Avvia una chatbot interattiva che utilizza un modello di linguaggio per rispondere a domande.

    Questa funzione inizializza un chatbot utilizzando un modello di linguaggio (LLM) preaddestrato
    con i vettori di parole forniti (vecs). Il chatbot rimane attivo finché l'utente non decide di
    uscire (digitando 'q'). L'utente può porre domande al chatbot e ricevere risposte basate sul
    modello di linguaggio.

    parameter:
    - vecs: Vettori di parole per inizializzare il modello di linguaggio.

    Usage:
    - Per uscire dalla chat, digita 'q' quando richiesto.
    - Altrimenti, inserisci la tua domanda e il chatbot fornirà una risposta basata sul modello.
    """
    llm = LLM(vecs)

    while True:
        query = input(
            "Hai una domanda? Scrivi 'q' per uscire o inserisci la tua domanda: "
        )

        if query.lower() == "q":
            break  # Esci dal ciclo se l'utente scrive 'q'
        else:
            print(llm.do_query(query))

def get_choice(prompt: str) -> int:
    while True:
        print(prompt)
        try:
            choice = int(input("Opzione numero: "))
            return choice
        except ValueError:
            print("Scelta non valida. Inserisci un numero valido.")

def save_to_json_file(data: [], file_path: str):
    """
    Salva un array di stringhe in un file JSON.

    Parameters:
    - data (List[str]): Un array di stringhe da salvare nel file JSON.
    - file_path (str): Il percorso del file JSON in cui salvare i dati.

    Returns:
    - True se il salvataggio ha avuto successo, altrimenti False.
    """
    try:
        with open(file_path, 'w') as json_file:
            json.dump(data, json_file, indent=4)
        return True
    except Exception as e:
        print(f"Si è verificato un errore durante il salvataggio del file JSON: {str(e)}")
        return False

menu = """
Benvenuto nell'App PDF Chatbot!

Scegli un'opzione:
1. Carica documenti PDF
2. Interagisci con il Chatbot
3. Esci

Inserisci il numero corrispondente all'opzione desiderata:
"""


def main(debug=False) -> None:
    # Inizializza variabili
    app = App()
    vectorstore = None
    while True:
        choice = get_choice(menu)
        if choice:
            if choice == 1:
                # Carica documenti PDF
                if debug:
                    pdf_files = [os.path.abspath("Costituzione.pdf")]
                else:
                    pdf_files = get_file_paths()
                print("\n")
                    
                if pdf_files:
                    uploaded, not_uploaded = app.load_pdfs(pdf_files)
                    save_to_json_file(uploaded, "cronologia.json")
                    for pdf in not_uploaded:
                        print("Il file " + pdf + " non è stato caricato correttamente")

            elif choice == 2:
                if not vectorstore:
                    vectorstore, _ = app.get_vectorstore()
                if vectorstore:
                    chat(vectorstore)
                else:
                    print(
                        "Non sono stati trovati dati presenti su pinecone. Per poter chattare con il bot è necessario caricare prima i dati"
                    )

            elif choice == 3:
                print("Ci vediamo in giro")
                break
            else:
                print("L'opzione inserita non era presente nel menu")


if __name__ == "__main__":
    # Aggiungi il supporto per il parametro --debug
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Esegui in modalità debug")

    args = parser.parse_args()
    main(args.debug)
    clean_pycache(os.getcwd())
