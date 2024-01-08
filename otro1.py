#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import cv2
import numpy as np

#algoritmo prim

def Prim(grafos):
    
    
    resultante = {}
    listaOrdenada = []
    listaVisitados = []

    origen = list(grafos.keys())[0]

    for destino, peso in grafos[origen]:
        listaOrdenada.append((origen, destino, peso))
        
    listaOrdenada = [(c,a,b) for a,b,c in listaOrdenada]
    listaOrdenada.sort()
    listaOrdenada = [(a,b,c) for c,a,b in listaOrdenada]
        
    while listaOrdenada:
      
      vertice = listaOrdenada.pop(0)
      d = vertice[1]
    
      if d not in listaVisitados:

        listaVisitados.append(d)
        
    
        for key, lista in grafos[d]:
          if key not in listaVisitados:
            listaOrdenada.append((d, key, lista))
            
            
        
        listaOrdenada = [(c,a,b) for a,b,c in listaOrdenada]
        listaOrdenada.sort()
        listaOrdenada = [(a,b,c) for c,a,b in listaOrdenada]
       
        
        origen  = vertice[0]
        destino = vertice[1]
        peso    = vertice[2]
    
        if origen in resultante:
          if destino in resultante:
            lista = resultante[origen]
            resultante[origen] = lista + [(destino, peso)]
            lista = resultante[destino]
            lista.append((origen, peso))
            resultante[destino] = lista
          else:
            resultante[destino] = [(origen, peso)]
            lista = resultante[origen]
            lista.append((destino, peso))
            resultante[origen] = lista
        elif destino in resultante:
          resultante[origen] = [(destino, peso)]
          lista = resultante [destino]
          lista.append((origen, peso))
          resultante[destino] = lista
        else:
          resultante[destino] = [(origen, peso)]
          resultante[origen] = [(destino, peso)]    
    
    
    
    
    return resultante;
    
    
#para cargar el mapa
mapa=cv2.imread('mapa3.png')
#pasamos la imagen a escala de grises
gray = cv2.cvtColor(mapa,cv2.COLOR_BGR2GRAY)

#obtengo un binarizacion en blaco todos lo pixeles cuyo valor en sea entre 254 y 255
ret,th1 = cv2.threshold(gray,254,255,cv2.THRESH_BINARY)
#hago un kernel de 11x11 de unos. Los Kernels se acostumbra hacerse
#de tamaño no par y cuadrados

kernel = np.ones((11,11), np.uint8) 
#aplico un filtro de dilatacion. Este filtro hace que los puntos los puntos 
#blancos se expandan 
#probocando que algunos puntitos negros desaparecan 
#le pueden hacer un cv.imshow para que vean el resultado
th1 = cv2.dilate(th1,kernel,1)
kernel = np.ones((11,11), np.uint8) 
#Despues aplico uno de erosion que hace lo opuesto al de dilatacion
th1 = cv2.erode(th1,kernel,1)
#aplico un flitro gausiando de 5x5  para suavisar los bordes 
th1 = cv2.GaussianBlur(th1,(5,5),cv2.BORDER_DEFAULT) 



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


grafos = {}

#buscar cuales son las esquinas que estan conectadas
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
            
            
            # Verificar que los 9 puntos sean igual a color blanco
            # significa que hay conexion
            
            if( 
                    (th2[i[1],i[0]]==blanco).all() and
                    
                    (th2[puntoM1[1],puntoM1[0]]==blanco).all() and
                    
                    (th2[puntoM2[1],puntoM2[0]]==blanco).all() and
                    
                    (th2[puntoM3[1],puntoM3[0]]==blanco).all() and
                    
                    (th2[puntoM4[1],puntoM4[0]]==blanco).all() and
                    
                    (th2[puntoM5[1],puntoM5[0]]==blanco).all() and
                    
                    (th2[puntoM6[1],puntoM6[0]]==blanco).all() and
                    
                    (th2[puntoM7[1],puntoM7[0]]==blanco).all() and
                    
                    (th2[j[1],j[0]]==blanco).all()
            ):
                
                
                #calcular peso con teorema de pitagora
                
                peso =  (  (  ( ( i[0]-j[0] )**2 ) + ( (i[1]-j[1])**2  )   )**0.5 )
                
                #crear grafo
                
                if tuple(i) not in grafos.keys():
                    grafos[tuple(i)] = [ [tuple(j) ,peso] ]
                else:
                    grafos[tuple(i)] += [ [tuple(j) ,peso] ]
                
                
                if tuple(j) not in grafos.keys():
                    grafos[tuple(j)] = [ [tuple(i) ,peso] ]
                else:
                    grafos[tuple(j)] += [ [tuple(i) ,peso] ]
                
                
                
                

costeMinimo = Prim(grafos)

for origen,adyacente in costeMinimo.items():
    cv2.circle(mapa,(origen), 5, (255,0,0), -1) 
    
    for destino in adyacente:
        cv2.line(mapa, origen, destino[0], (0,0,255), 2)
    


#resultado final
cv2.imshow('Mapa',mapa)

cv2.waitKey()

cv2.destroyAllWindows()
