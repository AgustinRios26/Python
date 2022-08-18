import cv2 as cv
import os
import numpy as np
#tiempo que tarda
from time import time
#pasar nuestra carpeta con cual vamos a entrenar, las fotos tienen que estar aisladas
dataruta="C:\Agus\Python\Reconocimiento Facial 1\data" 
#sacamos nuestra lita
listadata=os.listdir(dataruta)
#print("data",listadata)
#agregar etiquetas
ids=[]
#otra matriz
rostrosdata=[]
id=0
tiempoinicial=time()
#bucle para recorrer todos los archivos
for fila in listadata:
    #para entrar a cada carpeta
    rutacompleta=dataruta+"/"+fila
    print("iniciando lectura....")
    #para que entre a cada foto, para que entre a la lista de la carpeta
    for archivo in os.listdir(rutacompleta):
        #saber en que proceso me estoy encontrando
        print("imagenes: ",fila+"/"+archivo)
        #puedo trabajar con mis id
        ids.append(id)
        #forma de transformar a grises con 0 o de la forma tradicional
        rostrosdata.append(cv.imread(rutacompleta+"/"+archivo, 0))
        
        
    id=id+1
    #para saber el tiempo que se tarda
    tiempofinallectura=time()
    tiempototallectura=tiempofinallectura-tiempoinicial
    print("tiempo total lectura: ", tiempototallectura)
#engen face recognozing opencv nos dice el reconocimiento facial y ver las sintaxis
#poner el metodo de entrenamiento
entrenamientomodelo1=cv.face.EigenFaceRecognizer_create()
print("iniciando el entrenamiento....... espere")
#le asignamos datos, agregar toda la informacion que recore y ponemos los rostros y los nparray 
entrenamientomodelo1.train(rostrosdata, np.array(ids))
tiempofinalentrenamiento=time()
tiempototalentrenamiento=tiempofinalentrenamiento-tiempototallectura
print("Tiempo entrenamiento total",tiempototalentrenamiento)
#Guardamos en un archivo entrenado , guardaremos nuestro entrenamiento
entrenamientomodelo1.write("entrenamientoeigenfacerecognizer.xml")
print("Entrenamiento finalizado")