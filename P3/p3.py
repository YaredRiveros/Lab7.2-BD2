import nltk
from nltk.stem.snowball import SnowballStemmer
from collections import defaultdict
nltk.download('punkt')
import numpy as np


def preprocesamiento():
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

    return (tf,idf)



# 1. Preprocesamiento

preProcesado = preprocesamiento()      #preProcesado[0] = tf, preProcesado[1] = idf
#print("Numero de palabras en cada libro: ", preProcesado)

# 2. Similitud de coseno

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


def compute_tfidf(preProcesado): #falta agregar de segundo parámetro lo necesario para el IDF
    tf = preProcesado[0]
    #1. amortiguo la matriz de TF
    for doc in tf:
        for word in tf[doc]:
            tf[doc][word] = 1 + np.log10(tf[doc][word])

    # Guardo el IDF
    idf = preProcesado[1]


    # Calcular el TF-IDF de cada palabra en cada documento
    textos_tfidf = {}
    for doc in tf:
        textos_tfidf[doc] = {}
        for word in tf[doc]:
            #Producto punto entre escalares es el producto normal.
            # No realizo producto entre arrays porque estoy usando diccionarios, pero es la misma idea
            textos_tfidf[doc][word] = tf[doc][word] * idf[word] 

    return textos_tfidf


  

textos_tfidf = compute_tfidf(preProcesado)  #diccionario de doc: palabra: tfidf

matriz = []
#Creo la matriz de similitud de coseno
# En cada fila, se calcula la similitud de coseno entre el documento de esa fila y todos los demás documentos.
# La diagonal principal de la matriz es 1 porque la similitud de coseno de un documento consigo mismo es 1.
for doc1 in textos_tfidf:
    row = []
    for doc2 in textos_tfidf:  
        row.append(cosine_sim(textos_tfidf[doc1], textos_tfidf[doc2]))
    matriz.append(row)


def imprimir_matriz(matriz):
    max_length = max(len(str(elemento)) for fila in matriz for elemento in fila)
    for fila in matriz:
        for elemento in fila:
            print(str(elemento).ljust(max_length), end=' ')
        print()  # Imprimir una nueva línea después de cada fila

print("Matriz de similitud de coseno: ")



imprimir_matriz(matriz)







