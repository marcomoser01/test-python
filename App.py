#!/usr/bin/env python3


import json

from flask import Flask, request, jsonify
from my_package.ApiRoutes import ApiRoutes

app = Flask(__name__)


@app.route("/")
def info():
    result = {
        "description": "L'applicativo da la possibilità all'utente di caricare uno o più PDF ed interrogarli sucessivamente tramite un'intelligenza artificiale"
    }
    return jsonify(result)

@app.route("/createIndex", methods=['GET'])
def createIndex():
    res = ""
    routes = ApiRoutes()
    res = routes.create_index()
    if res:
        res = {"state": 200, "message": "L'indice è stato creato"}
    else:
        res = {"state": 410, "message": "Non sono riuscito a creare l'indice"}

    return jsonify(res)

@app.route("/getSources", methods=['GET'])
def getSources():
    res = ""
    routes = ApiRoutes()
    res = routes.get_sources()

    return jsonify(res)

@app.route("/delete/<source>", methods=['GET'])
def deleteSource(source):
    res = ""
    routes = ApiRoutes()
    
    success, not_found, error  = routes.delete_source([source])
    res = {
        "success": success,
        "not_found": not_found,
        "error": error
    }

    return jsonify(res)

@app.route("/delete/sources", methods=['POST'])
def deleteSources():
    res = ""
    data = request.get_json()
    if data is None:
        res = {
			"error": "missing parameters"
		}
    else:
        routes = ApiRoutes()
        success, not_found, error = routes.delete_source(data['sources'])
        res = {
            "success": success,
            "not_found": not_found,
            "error": error
        }

    return jsonify(res)

@app.route("/deleteAll", methods=['GET'])
def deleteAll():
    res = ""
    routes = ApiRoutes()
    success, not_found, error = routes.delete_all()
    res = {
        "success": success,
        "not_found": not_found,
        "error": error
    }

    return jsonify(res)

@app.route("/deleteIndex", methods=['GET'])
def deleteIndex():
    res = ""
    routes = ApiRoutes()
    res = routes.delete_index()
    if res:
        res = {"state": 200, "message": "L'indice è stato rimosso con successo"}
    else:
        res = {"state": 410, "message": "Non sono riuscito a rimuovere l'indice"}

    return jsonify(res)

@app.route("/chat", methods=['POST'])
def chat():
    '''
    {
        "query": "Path dei PDF che si vuole caricare. Devono essere path assolute"
    }
    '''
    res = ""
    data = request.get_json()
    if data is None:
        res = {
			"error": "missing parameters"
		}
    else:
        routes = ApiRoutes()
        if routes.init_ChatBot('gpt-4'):
            res = {
                "state": 200,
                "message": routes.chat(data['query'])
            }
            
        else:
            res = {
                "state": 410, 
                "message": "Ci sono stati dei problemi nel collegamento con il chatbot"
            }

    return jsonify(res)


@app.route("/upload/pdf", methods=['POST'])
def upload():
    '''
    {
        "path": [
            "Path dei PDF che si vuole caricare. Devono essere path assolute"
        ]
    }
    '''
    res = ""
    data = request.get_json()
    if data is None:
        res = {
			"error": "missing parameters"
		}
    else:
        routes = ApiRoutes()
        uploaded, not_uploaded, wrong_path = routes.upload_pdfs(data['path'])
        res = {
            "uploaded": uploaded,
            "not_uploaded": not_uploaded,
            "wrong_path": wrong_path
        }

    return jsonify(res)


if __name__ == "__main__":
    app.run(debug=True)