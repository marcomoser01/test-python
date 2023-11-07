#!/usr/bin/env python3

import os, argparse, json
import platform
import shutil
from typing import List
from my_package.Chatbot import Chatbot
from my_package.PDF_ChatBot import PDF_ChatBot

menu = """
Benvenuto nell'App PDF Chatbot!

Scegli un'opzione:
\t-1. Per pulire la console
\t0. Esci
\t1. Carica documenti PDF
\t2. Interagisci con il Chatbot
\t3. Rimuovi documenti dalla memoria
\t4. Eliminare tutti i dati, compreso l'indice
\t5. Visualizzare quali file sono salvati

Inserisci il numero corrispondente all'opzione desiderata:
"""

pdf_ChatBot = None

def clean_pycache(path):
    for root, dirs, files in os.walk(path):
        for dir in dirs:
            if dir == "__pycache__":
                shutil.rmtree(os.path.join(root, dir))

def clear_terminal():
    if platform.system() == "Windows":
        os.system("cls")  # Pulisce il terminale su Windows
    else:
        os.system("clear")  # Pulisce il terminale su macOS e Linux

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
            print(f"Il file {file_path} è stato trovato e verrà caricato")
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
    if pdf_ChatBot.init_ChatBot("gpt-4", vecs):
        while True:
            query = input(
                "\nHai una domanda? Scrivi 'q' per uscire o inserisci la tua domanda: "
            )

            if query.lower() == "q":
                break  # Esci dal ciclo se l'utente scrive 'q'
            else:
                print()
                print(pdf_ChatBot.interact_with_chatbot(query))
    else:
        print("Non siamo riusciti ad inizializzare una conversazione con il chatbot")

def print_all_sources(sources: []) -> None:
    print("Sul vector store sono presenti le seguenti sorgenti:\n")
    for item in sources:
        print(f"- {item}")
    print("")

def get_files_name(sources: []) -> List[str]:
    files_name = []  # Inizializza una lista vuota per i percorsi dei file

    while True:
        file_name = input(
            "Inserisci il nome del file che vuoi che venga dimenticato (o premi Invio per terminare): "
        )

        if not file_name:
            break  # L'utente ha premuto Invio per terminare l'inserimento
        if file_name in sources:
            files_name.append(file_name)
        else:
            print(
                "La sorgente che ha specificato non è presente sul vector store, in seguito rielenco le sorgenti:\n"
            )
            for item in sources:
                print(f"- {item}")
        print("")
    return files_name

def get_choice(prompt: str) -> int:
    while True:
        print(prompt)
        try:
            choice = int(input("Opzione numero: "))
            print("\n")
            return choice
        except ValueError:
            print("Scelta non valida. Inserisci un numero valido.")

def choice_1(debug: bool):
    if debug:
        pdf_files = [os.path.abspath("data/Costituzione.pdf")]
    else:
        pdf_files = get_file_paths()
    print("\n")

    if pdf_files:
        uploaded, not_uploaded, history_state = pdf_ChatBot.upload_pdfs(pdf_files, "cronologia.json")
        for pdf in not_uploaded:
            print("Il file " + pdf + " non è stato caricato correttamente")
        
        if not history_state:
            print("I seguenti file sono stati caricati, ma non è stato possibile aggiungerli alla cronologia.")
            for file in uploaded:
                print(f"- {file}")

def choice_2():
    vectorstore, _ = pdf_ChatBot.get_vectorstore()
    if vectorstore:
        chat(vectorstore)
    else:
        print(
            "Non sono stati trovati dati presenti su pinecone. Per poter chattare con il bot è necessario caricare prima i dati"
        )

def choice_3():
    all_sources = pdf_ChatBot.get_sources()
    if all_sources:
        print_all_sources(all_sources)
        files_name = get_files_name(all_sources)
        if files_name:
            #TODO gestire il risultato del metodo
            pdf_ChatBot.delete_data_vectorstore(files_name)
    else:
        print(
            "Non è stato trovato l'indice desiderato, assicurarsi prima che l'indice esisti"
        )

def choice_5():
    all_sources = pdf_ChatBot.get_sources()
    if all_sources:
        print_all_sources(all_sources)




def main(debug=False) -> None:
    while True:
        choice = get_choice(menu)
        if choice is not None:
            if choice == -1:
                clear_terminal()

            elif choice == 0:
                print("Ci vediamo in giro")
                break

            elif choice == 1:
                # Carica documenti PDF
                choice_1(debug)

            elif choice == 2:
                choice_2()

            elif choice == 3:
                choice_3()

            elif choice == 4:
                pdf_ChatBot.delete_index()

            elif choice == 5:
                choice_5()

            else:
                print("L'opzione inserita non era presente nel menu")


if __name__ == "__main__":
    # Aggiungi il supporto per il parametro --debug
    parser = argparse.ArgumentParser()
    parser.add_argument("--debug", action="store_true", help="Esegui in modalità debug")

    args = parser.parse_args()
    
    pdf_ChatBot = PDF_ChatBot("pdf-index")
    main(args.debug)
    clean_pycache(os.getcwd())
