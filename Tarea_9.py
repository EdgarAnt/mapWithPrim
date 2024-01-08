
import cv2
import numpy as np
from collections import defaultdict


# con esta funcion reviso si un punto ya esta en alguna de mis listas
def isInTheList(elemento, arreglo):
    for i in arreglo:
        if np.array_equal(i, elemento):
            return True
    return False


# para cargar el mapa
mapa = cv2.imread('C:\\Users\\Laptop\\Downloads\\mapa3.png')
# pasamos la imagen a escala de grises
gray = cv2.cvtColor(mapa, cv2.COLOR_BGR2GRAY)
# muestro la imagen en escala de grises
    #cv2.imshow('mapa', gray)
    #cv2.waitKey()
# obtengo una binarizacion en blaco todos lo pixeles cuyo valor en sea entre 254 y 255
ret, th1 = cv2.threshold(gray, 254, 255, cv2.THRESH_BINARY)
# hago un kernel de 11x11 de unos. Los Kernels se acostumbra hacerse de tamaño no par y cuadrados
# para que se den una idea algo asi:
"""
1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1
1 1 1 1 1 1 1 1 1 1 1
"""
kernel = np.ones((11, 11), np.uint8)
# aplico un filtro de dilatacion. Este filtro hace que los puntos blancos se expandan
# probocando que algunos puntitos negros desaparecan #le pueden hacer un cv.imshow para que vean el resultado
th1 = cv2.dilate(th1, kernel, 1)
kernel = np.ones((11, 11), np.uint8)
# Despues aplico uno de erosion que hace lo opuesto al de dilatacion
th1 = cv2.erode(th1, kernel, 1)
# aplico un flitro gausiando de 5x5  para suavisar los bordes
th1 = cv2.GaussianBlur(th1, (5, 5), cv2.BORDER_DEFAULT)
# muestro como queda mi mapa
    #cv2.imshow('thres', th1)
    #cv2.waitKey()
# Aplico la deteccion de Esquinas de Harris. para mas informacion consulten https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
dst = cv2.cornerHarris(th1, 2, 3, 0.05)
ret, dst = cv2.threshold(dst, 0.04 * dst.max(), 255, 0)
dst = np.uint8(dst)
ret, th2 = cv2.threshold(th1, 235, 255, cv2.THRESH_BINARY)
th2 = cv2.dilate(th2, kernel, 1)
# aqui devuelvo la imagen binarizada a tres canales
th2 = cv2.cvtColor(th2, cv2.COLOR_GRAY2BGR)
# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst, 30, cv2.CV_32S)
vertices = np.int0(centroids)

aux1 = vertices
aux2 = vertices
verticesConectados = []
aristas = []
# aqui voy a buscar cuales son las esquinas que estan conectadas
for h in range(len(aux1)):
    i = aux1[h]
    for k in range(h, len(aux2)):
        j = aux2[k]
        if not (i == j).all():
            # aqui deberian sacar los puntos de intermedios y verificar si i y j estan conectados
            p1 = (i+j)/2
            p2 = (p1+j)/2
            p3 = (p1+i)/2
            p4 = (p2+j)/2
            p5 = (p2+i)/2
            p6 = (p2+p1)/2
            p7 = (p1+p3)/2

            # si estan conectados calcular el costo (la distancia en pixeles entre ellos usan teorema de pitagoras)
            # y agregarlos al grafo
            if(th2[int(p1[1])][int(p1[0])]==[255,255,255]).all() and \
                (th2[int(p2[1])][int(p2[0])] ==[255, 255, 255]).all() and \
                (th2[int(p3[1])][int(p3[0])] ==[255, 255, 255]).all() and \
                (th2[int(p4[1])][int(p4[0])] ==[255, 255, 255]).all() and \
                (th2[int(p5[1])][int(p5[0])] ==[255, 255, 255]).all() and \
                (th2[int(p6[1])][int(p6[0])] ==[255, 255, 255]).all() and \
                (th2[int(p7[1])][int(p7[0])] ==[255, 255, 255]).all():
                costA = np.linalg.norm(i-j)
                aristas.append([i,j,costA])
                if not isInTheList(i,verticesConectados):
                    verticesConectados.append(i)
                if not isInTheList(j,verticesConectados):
                    verticesConectados.append(j)

graph = {'nodos': verticesConectados, 'aristas': aristas}

k = defaultdict(list)

for n1,n2,c in graph['aristas']:
    k[n1[0],n1[1]].append((n2,c))
    k[n2[0], n2[1]].append((n1, c))




# Algoritmo de Prim

listaVisitados = []
grafoResultante = {}
listaOrdenada = []
resultado = []

# Elegimos el nodo de origen
origen = verticesConectados[0]
# Lo agregamos a la lista de visitados
listaVisitados.append(origen)
# Se agregan sus aydacentes a la lista ordenada
for destino, peso in k[origen[0],origen[1]]:
  listaOrdenada.append((origen, destino, peso))
'''ORDENAMIENTO INSERT PARA LA LISTA'''
while listaOrdenada:
    pos=0
    act=0
    listAux=[]
    for i in range(len(listaOrdenada)):
        listAux=listaOrdenada[i]
        act=listaOrdenada[i][2]
        pos=i
        while pos> 0 and listaOrdenada[pos-1][2] > act:
            listaOrdenada[pos] = listaOrdenada[pos-1]
            pos=pos-1
        listaOrdenada[pos]=listAux

# Se toma el vertice de la lista ordenada y lo eliminamos
    vertice = listaOrdenada.pop(0)
    d = vertice[1]

# En caso de que el destino no esta en la lista de visitados
    if not isInTheList(d,listaVisitados):
#6. Agregar a la lista el nodo destino
        listaVisitados.append(d)
# Agregamos a la lista los aydacentes del nodo destino
        for key, lista in k[d[0],d[1]]:
            if not isInTheList(key,listaVisitados):
                listaOrdenada.append((d, key, lista))

    origen = vertice[0]
    destino = vertice[1]
    peso = vertice[2]

# Agregamos el vertice al nodo resultante
    if isInTheList(origen, grafoResultante):
        if isInTheList(destino, grafoResultante):
            lista = grafoResultante[origen[0], origen[1]]
            grafoResultante[origen[0], destino[1]] = lista + [origen, destino, peso]
            lista = grafoResultante[destino[0], destino[1]]
            lista.append([origen, destino, peso])
            grafoResultante[destino[0], destino[1]] = lista
        else:
            grafoResultante[destino[0],destino[1]] = [origen, destino, peso]
            lista = grafoResultante[origen[0],origen[1]]
            lista.append([origen, destino, peso])
            grafoResultante[origen[0],origen[1]] = lista
    elif isInTheList(destino, grafoResultante):
        grafoResultante[origen] = [origen, destino, peso]
        lista = grafoResultante [destino]
        lista.append([origen, destino, peso])
        grafoResultante[destino] = lista
    else:
        grafoResultante[destino[0],destino[1]] = [origen, destino, peso]
        grafoResultante[origen[0],origen[1]] = [origen, destino, peso]

resultado = []

for key, lista in grafoResultante.items():
    resultado.append(lista)

#for arista in aristas:
 #   cv2.line(th2, tuple(arista[0]), tuple(arista[1]), (0,255,0), 1)

for arista in resultado:
    cv2.line(th2, tuple(arista[0]), tuple(arista[1]), (0, 255, 0), 3)

#aqui pinto los puntos de las esquinas que son circulos de de radio de 5 pixeles, el -1 indica que van rellenados los circulos
#point tiene la forma [fila, columna]

for point in vertices:
    cv2.circle(th2, (point[0], point[1]), 5, (255, 0, 0), -1)
    cv2.waitKey(1)

# aqui muestro como quedo de chingon el grafo
cv2.imshow('Resultado', th2)
cv2.waitKey()
