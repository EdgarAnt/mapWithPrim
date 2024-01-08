

import cv2
import numpy as np




def Prim(grafos):
    
    listaVisitados = []
    grafoResultante = {}
    listaOrdenada = []

    origen = list(grafos.keys())[0]


    for destino, peso in grafos[origen]:
        listaOrdenada.append((origen, destino, peso))
        
        
        
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
        

    while listaOrdenada:
      #5.-TOMAR VERTICE DE LA LISTA ORDENADA Y ELIMINARLO
      vertice = listaOrdenada.pop(0)
      d = vertice[1]
    
      #6.-SI EL DESTINO NO ESTA EN LA LISTA DE VISITADOS
      if d not in listaVisitados:
        
          
        #print(vertice)
        #print("----------------")
        #print(grafos)
        #7.- AGREGAR A LA LISTA DE VISITADOS EN NODO DESTINO
        listaVisitados.append(d)
        #8.- AGREGAR A LA LISTA ORDENADA LOS ADYACENTES DEL NODO DESTINO 
        #"d" QUE NO HAN SIDO VISITADOS
        
        #print(type(d))
        #print(d)
        #print(grafos[d])
        
        if d not in grafos:
            continue
        
        
        
        for key, lista in grafos[d]:
          if key not in listaVisitados:
            listaOrdenada.append((d, key, lista))
            
            
        #####ORDENAMIENTO APLICADO A LA LISTA :
        listaOrdenada = [(c,a,b) for a,b,c in listaOrdenada]
        listaOrdenada.sort()
        listaOrdenada = [(a,b,c) for c,a,b in listaOrdenada]
        #9.-AGREGAR VERTICE AL GRAFO RESULTANTE
        # PARA COMPRENDER MEJOR, EN LAS SIGUIENTES LINEAS SE TOMA EL "VERTICE", QUE EN ESTE CASO
        # ES UNA TUPLA QUE CONTIENE TRES VALORES; EL VERTICE EN SU POSICIÓN 0 ES EL VALOR DEL NODO ORIGEN
        # EL VÉRTICE EN SU POSICIÓN 1 ES EL NODO DESTINO, Y EL VÉRTICE EN SU POSICIÓN 2 ES EL PESO DE LA ARISTA ENTRE AMBOS NODOS,
        # Y A CONTINUACIÓN SE AGREGAN ESOS VALORES AL GRAFO
        origen  = vertice[0]
        destino = vertice[1]
        peso    = vertice[2]
    
        if origen in grafoResultante:
          if destino in grafoResultante:
            lista = grafoResultante[origen]
            grafoResultante[origen] = lista + [(destino, peso)]
            lista = grafoResultante[destino]
            lista.append((origen, peso))
            grafoResultante[destino] = lista
          else:
            grafoResultante[destino] = [(origen, peso)]
            lista = grafoResultante[origen]
            lista.append((destino, peso))
            grafoResultante[origen] = lista
        elif destino in grafoResultante:
          grafoResultante[origen] = [(destino, peso)]
          lista = grafoResultante [destino]
          lista.append((origen, peso))
          grafoResultante[destino] = lista
        else:
          grafoResultante[destino] = [(origen, peso)]
          grafoResultante[origen] = [(destino, peso)]    
    
    
    
    
    return grafoResultante;
    """
    for key, lista in grafoResultante.items():
        print(key,end='jjjjj')
        print(lista)
    """
    
"""
#cone sta funcion reviso si un punto ya esta en alguna de mis listas
def isInTheList(elemento,arreglo):
    for i in arreglo:
        if np.array_equal(i,elemento):
            return True            
    return False  
"""    


#para cargar el mapa
mapa = cv2.imread('C:\\Users\\Laptop\\Downloads\\map3chiquito.PNG')
#pasamos la imagen a escala de grises
gray = cv2.cvtColor(mapa,cv2.COLOR_BGR2GRAY)


#muestro la imagen en escala de grises
"""
cv2.imshow('mapa',gray)
cv2.waitKey()
"""

#obtengo un binarizacion en blaco todos lo pixeles cuyo valor en sea entre 254 y 255
ret,th1 = cv2.threshold(gray,254,255,cv2.THRESH_BINARY)
#hago un kernel de 11x11 de unos. Los Kernels se acostumbra hacerse de tamaño no par y cuadrados
#para que se den una idea algo asi:
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
kernel = np.ones((11,11), np.uint8) 
#aplico un filtro de dilatacion. Este filtro hace que los puntos los puntos blancos se expandan 
#probocando que algunos puntitos negros desaparecan #le pueden hacer un cv.imshow para que vean el resultado
th1 = cv2.dilate(th1,kernel,1)
kernel = np.ones((11,11), np.uint8) 
#Despues aplico uno de erosion que hace lo opuesto al de dilatacion
th1 = cv2.erode(th1,kernel,1)
#aplico un flitro gausiando de 5x5  para suavisar los bordes 
th1 = cv2.GaussianBlur(th1,(5,5),cv2.BORDER_DEFAULT) 


#muestro como queda mi mapa
"""
cv2.imshow('thres',th1)
cv2.waitKey()
"""

#Aplico la deteccion de Esquinas de Harris. para mas informacion consulten https://docs.opencv.org/3.4/dc/d0d/tutorial_py_features_harris.html
dst = cv2.cornerHarris(th1,2,3,0.05)
ret, dst = cv2.threshold(dst,0.04*dst.max(),255,0)
dst = np.uint8(dst)
ret,th2 = cv2.threshold(th1,235,255,cv2.THRESH_BINARY)
th2 = cv2.dilate(th2,kernel,1)
#aqui devuelvo la imagen binarizada a tres canales
th2 = cv2.cvtColor(th2,cv2.COLOR_GRAY2BGR)
# find centroids
ret, labels, stats, centroids = cv2.connectedComponentsWithStats(dst,30, cv2.CV_32S)
vertices=np.int0(centroids)

aux1=vertices
aux2=vertices
verticesConectados=[]
aristas=[]

#print(aux1)

grafos = {}

#aqui voy a buscar cuales son las esquinas que estan conectadas
for h in range(len(aux1)):
    i=aux1[h]
    for k in range(h,len(aux2)):
        j=aux2[k]
        if not (i==j).all():
            
            
            #sacar puntos intermedios
            
            #Calcular puntos de i y j
            puntoM1 = [ ((i[0]+j[0])//2) , ((i[1]+j[1])//2)  ]
        
            #Calcular puntos de i y puntoM1
            puntoM2 = [ ((i[0]+puntoM1[0])//2) , ((i[1]+puntoM1[1])//2)  ]
            
            #Calcular puntos de i y puntoM2
            puntoM4 = [ ((i[0]+puntoM2[0])//2) , ((i[1]+puntoM2[1])//2)  ]
            
            #Calcular puntos de puntoM2 y puntoM1
            puntoM5 = [ ((puntoM2[0]+puntoM1[0])//2) , ((puntoM2[1]+puntoM1[1])//2)  ]
            
            
            #Calcular puntos de puntoM1 y j
            puntoM3 = [ ((puntoM1[0]+j[0])//2) , ((puntoM1[1]+j[1])//2)  ]
            
            #Calcular puntos de puntoM1 y puntoM3
            puntoM6 = [ ((puntoM1[0]+puntoM3[0])//2) , ((puntoM1[1]+puntoM3[1])//2)  ]
            
            #Calcular puntos de puntoM3 y j
            puntoM7 = [ ((puntoM3[0]+j[0])//2) , ((puntoM3[1]+j[1])//2)  ]
            
            
            blanco = [255,255,255]
            
            
            
            """
            
            print('es negro' if (th2[puntoM1[1],puntoM1[0]]==[0,0,0]).all() else 'es blanco',end='')
            print(th2[puntoM1[1],puntoM1[0]])
            """
            
            # Verificar que los 7 puntos sean igual a color blanco significa que hay conexion
            
            if( 
                    (th2[puntoM1[1],puntoM1[0]]==blanco).all() and
                    
                    (th2[puntoM2[1],puntoM2[0]]==blanco).all() and
                    
                    (th2[puntoM3[1],puntoM3[0]]==blanco).all() and
                    
                    (th2[puntoM4[1],puntoM4[0]]==blanco).all() and
                    
                    (th2[puntoM5[1],puntoM5[0]]==blanco).all() and
                    
                    (th2[puntoM6[1],puntoM6[0]]==blanco).all() and
                    
                    (th2[puntoM7[1],puntoM7[0]]==blanco).all() 
            ):
                
                
                #calcular peso con teorema de pitagora
                
                peso =  (  (  ( ( i[0]-j[0] )**2 ) + ( (i[1]-j[1])**2  )   )**0.5 )
                
                
                
                
                #crear grado
                
                if tuple(i) not in grafos.keys():
                    grafos[tuple(i)] = [ [tuple(j) ,peso] ]
                else:
                    grafos[tuple(i)] += [ [tuple(j) ,peso] ]
                
                
                if tuple(j) not in grafos.keys():
                    grafos[tuple(j)] = [ [tuple(i) ,peso] ]
                else:
                    grafos[tuple(j)] += [ [tuple(i) ,peso] ]
                
                
                aristas.append([i,j])
                
            
            
            
            
            #aqui deberian sacar los puntos de intermedios y verificar si i y j estan conectados
                #si estan conectados calcular el costo (la distancia en pixeles entre ellos usan teorema de pitagoras papá) y agregarlos al grafo

#aqui yo dibujo mis lineas de las aristas de color verde, el uno es el grueso de la linea
#arista[0]   y arista[1]  tienen la forma de [fila, columna]

#print(grafos)



for arista in aristas:
    cv2.line(th2, tuple(arista[0]), tuple(arista[1]), (0,255,0), 1)


  
#formato es  columna , fila


#cv2.circle(th2,(50, 250), 5, (0,0,255), -1)    


#aqui pinto los puntos de las esquinas que son circulos de de radio de 5 pixeles, el -1 indica que van rellenados los circulos
#point tiene la forma [fila, columna]


for point in vertices:
    cv2.circle(th2,(point[0], point[1]), 5, (255,0,0), -1)    
    #print(th2[point[0],20])

    
    #print(point[0],point[1])
    #print(th2[point[0],point[1]])
 
    
 
print( len(vertices) )     
print( len(grafos) )  
camino = Prim(grafos)
print( len(camino) )  


for origen,adyacente in camino.items():
    cv2.circle(mapa,(origen), 5, (255,0,0), -1) 
    for destino in adyacente:
        cv2.line(mapa, origen, destino[0], (0,0,255), 2)
    




cv2.waitKey()
    

#aqui muestro como quedo de chingon el grafo
cv2.imshow('points',mapa)

cv2.waitKey()
cv2.destroyAllWindows()



    



