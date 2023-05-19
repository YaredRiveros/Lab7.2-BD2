import nltk
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
nltk.download('punkt')
import numpy as np
import os

class InvertIndex: 

    def __init__(self, index_file):
        self.index_file = index_file
        self.idf = {}
        self.tf = {}
        self.tfidf = {}
    
    def building(self):

        if os.path.exists(self.index_file): #Si el índice ya fue creado, no hacer nada
            pass
        else:
            num_roots = 500
            derivador = SnowballStemmer('spanish')
            libros = ['libro1.txt', 'libro2.txt', 'libro3.txt','libro4.txt','libro5.txt','libro6.txt']

            # Initialize a set for stopwords
            stoplist = set()

            # Read stop words from file into the set
            with open('stoplist.txt', 'r',encoding='ISO-8859-1') as file:
                stoplist = set(file.read().split('\n'))

            # Initialize defaultdict for tokens
            tokens = defaultdict(set)


            # Process each book
            for file_id, archivo in enumerate(libros, start=1):
                with open(archivo, 'r',encoding='ISO-8859-1') as file:
                    tokens_in_file = nltk.word_tokenize(file.read().lower())
                    tokens_in_file = [token for token in tokens_in_file if token.isalpha()]  # filter out non-alphanumeric characters
                    for token in tokens_in_file:
                        root = derivador.stem(token)
                        if root not in stoplist:
                            tokens[root].add(file_id)

            # Limit tokens to the 500 most frequent ones
            tokens = dict(sorted(tokens.items(), key=lambda x: len(x[1]), reverse=True)[:num_roots])

            # Sort the dictionary alphabetically
            tokens = dict(sorted(tokens.items()))


            # tf

            tf = {
                "libro1.txt": {},
                "libro2.txt": {},
                "libro3.txt": {},
                "libro4.txt": {},
                "libro5.txt": {},
                "libro6.txt": {}
            }

            ## Cuanto cuántas veces aparece cada palabra en cada libro (TF)
            for file_id, archivo in enumerate(libros, start=1):
                with open(archivo, 'r',encoding='ISO-8859-1') as file:
                    tokens_in_file = nltk.word_tokenize(file.read().lower())
                    tokens_in_file = [token for token in tokens_in_file if token.isalpha()]
                    for token in tokens_in_file:
                        if(token in tokens):
                            root = derivador.stem(token)
                            if root in tf[archivo]:
                                tf[archivo][root] += 1
                            else:
                                tf[archivo][root] = 1

            # idf
            idf = {}
            for token in tokens:
                idf[token] = np.log10(len(libros) / len(tokens[token])) #idf = log(numero total de documentos / numero de documentos en los que aparece la palabra)
            

            for doc in tf:
                for word in tf[doc]:
                    tf[doc][word] = 1 + np.log10(tf[doc][word])
            self.tf = tf


            self.idf = idf

            # Guardo el idf en un archivo
            with open('idf.txt', 'w',encoding='ISO-8859-1') as file:
                for word in self.idf:
                    file.write(f"{word} {self.idf[word]}\n")


            # Calcular tf-idf
            textos_tfidf = {}
            for doc in tf:
                textos_tfidf[doc] = {}
                for word in tf[doc]:
                    #Producto punto entre escalares es el producto normal.
                    # No realizo producto entre arrays porque estoy usando diccionarios, pero es la misma idea
                    textos_tfidf[doc][word] = tf[doc][word] * idf[word] 

            self.tfidf = textos_tfidf

            # Escribo en el archivo index.txt el tf-idf de cada palabra en cada documento
            with open(self.index_file, 'w',encoding='ISO-8859-1') as file:
                for doc in self.tfidf:
                    for word in self.tfidf[doc]:
                        file.write(f"{doc} {word} {self.tfidf[doc][word]}\n")



    def retrieval(self, query, k): #k es el numero de documentos mas relevantes que se quieren obtener
        # diccionario para el score
        score = {}
        # preprocesar la query: extraer los terminos unicos
        query = nltk.word_tokenize(query.lower())
        query = [token for token in query if token.isalpha()]
        # calcular el tf del query
        tf_query = {}
        for token in query:
            if token in tf_query:
                tf_query[token] += 1
            else:
                tf_query[token] = 1
        for word in tf_query:
            tf_query[word] = 1 + np.log10(tf_query[word])


        # Leo el idf desde el archivo idf.txt
        with open('idf.txt', 'r',encoding='ISO-8859-1') as file:
            for line in file:
                word, idf = line.split()
                self.idf[word] = float(idf)

        # calcular el idf del query
        idf_query = {}
        for word in tf_query:
            if word in self.idf:
                idf_query[word] = self.idf[word]
            else:
                idf_query[word] = 0

        # calcular el tf-idf del query
        tfidf_query = {}
        for word in tf_query:
            tfidf_query[word] = tf_query[word] * idf_query[word]
        
        # cargo el tf-idf desde el archivo index.txt
        with open(self.index_file, 'r',encoding='ISO-8859-1') as file:
            for line in file:
                doc, word, tfidf = line.split()
                tfidf = float(tfidf)
                if doc not in self.tfidf:
                    self.tfidf[doc] = {}
                self.tfidf[doc][word] = tfidf


        # aplicar similitud de coseno y guardarlo en el diccionario score
        for doc in self.tfidf:
            score[doc] = InvertIndex.cosine_sim(tfidf_query, self.tfidf[doc])

        # ordenar el score de forma descendente
        result = sorted(score.items(), key= lambda tup: tup[1], reverse=True)

        # retornamos los k documentos mas relevantes (de mayor similitud al query)
        return result[:k] #retorna una lista de tuplas (nombre del documento, similitud) 
    
    def cosine_sim(Q, Doc):  
        # Calcula la suma de los productos de los elementos de 2 diccionarios
        # Si un elemento no está en el otro diccionario, se asume que su valor es 0
        # Esto es porque el producto de 0 por cualquier número es 0

        # 1. Obtengo las palabras de ambos documentos
        words = set(Q.keys()) | set(Doc.keys())
        # 2. Calculo el producto punto
        dot_product = sum(Q.get(word, 0) * Doc.get(word, 0) for word in words)
        # 3. Calculo las normas
        norm_Q = np.sqrt(sum(Q.get(word, 0) ** 2 for word in words))
        norm_Doc = np.sqrt(sum(Doc.get(word, 0) ** 2 for word in words))

        # 4. Calculo la similitud
        return dot_product / (norm_Q * norm_Doc)