# Importa bibliotecas necessárias
from flask import Flask, request, jsonify
from flask_cors import CORS
import json
import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.cluster import KMeans
# Realiza o download das stopwords da Lingua Portuguesa
nltk.download("stopwords")
stop_words_pt = stopwords.words("portuguese")

# Configura o falsk para iniciar o servidor
app = Flask(__name__)
CORS(app)

# Função para carregar os itens
def carregar_itens():
    with open("itens.json", "r", encoding="utf-8") as f:
        dados = json.load(f)
    return dados["itens"]

itens = carregar_itens()
textos = [item["descricao"] for item in itens]

# Construção do modelo TF-IDF
vetorizer = TfidfVectorizer(stop_words=stop_words_pt)
matriz_tfidf = vetorizer.fit_transform(textos)

# Agrupar itens usando K-Means, com 3 clusters e 10 tentativas
kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
clusters = kmeans.fit_predict(matriz_tfidf)
# Cria a rota para realizar a busca
@app.route("/buscar", methods=["GET"])
def buscar_itens():
    # Obtem os dados da consulta
    query = request.args.get("q", "")
    if not query:
        return jsonify({"error": "Nenhuma consulta fornecida"}), 400
    # Realiza a vetorização, obtem o itens com maior similaridade e os oredenam
    query_vector = vetorizer.transform([query])
    similaridades = cosine_similarity(query_vector, matriz_tfidf).flatten()
    indices_ordenados = np.argsort(similaridades)[::-1]
    # Faz o retorno das informações
    resultados = []
    for i in indices_ordenados:
        if similaridades[i] > 0:
            item = {
                "nome": itens[i]["nome"],
                "descricao": itens[i]["descricao"],
                "relevancia": float(similaridades[i]),
                "preco": itens[i]["preco"],
                "img": itens[i]["img"],
                "sugestoes": []
            }
            # Retorna sugestões dentro do mesmo cluster
            cluster_do_item = clusters[i]
            sugestoes = [
                itens[j]["nome"] for j in range(len(itens)) 
                if clusters[j] == cluster_do_item and j != i
            ][:3]
            item["sugestoes"] = sugestoes
            resultados.append(item)
    
    return jsonify({"resultados": resultados})
# Define a porta que opera o servidor Flask
if __name__ == "__main__":
    app.run(debug=True, port=5000)
