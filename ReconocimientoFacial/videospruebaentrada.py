import cv2 as cv
#os crea carpetas, archivos en codigo python
import os
#bajarle resolucion
import imutils
#para hacerlo con diferentes videos para entrenarlp
modelo="FotosElon"
#ruta principal
ruta1="C:/Agus/Python/Reconocimiento Facial 1"
rutacompleta=ruta1+"/"+ modelo
#si la ruta existe
if not os.path.exists(rutacompleta):
    #si no existe crea la carpeta
    os.makedirs(rutacompleta)


ruidos=cv.CascadeClassifier("C:\Agus\Python\Entrenamientoopencvruidos\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml")
#CAMBIA EL VIDEO CAPTURE PORQUE SE PONE EL LINK
camara=cv.VideoCapture("C:/Agus/Python/Reconocimiento Facial 1/ElonMusk.mp4")
#identificador que se aumenta a medida que se capture las imagenes
id=0
while True:
    respuesta,captura=camara.read()
    if respuesta==False:
        break
    #cambiar resolucion
    captura=imutils.resize(captura,width=640)
    grises=cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    #crea una copia de las capturas
    idcaptura=captura.copy()
    #detecta las caras y la imagen que va a tener los ruidos y factor de escala y podemos poner cuantos rostros va a encontrar
    #el 1.3 para que reduce las imagenes y el otro dice que la cantidad de contornos es 5 que puede encontrar
    caras=ruidos.detectMultiScale(grises,1.3,5)
    #recorrer todos los rostros, x y esquina 1 y esquina 2
    for (x,y,e1,e2) in caras:
        #hacemos un rectangulo
        cv.rectangle(captura, (x,y), (x+e1,y+e2), (255,0,0),2)
        #saca fragmento de nuestro rostro y almacena en carpeta, son las coordenadas de la captura
        rostrocapturado=idcaptura[y:y+e2,x:x+e1]
        #tamaño rostro, en un cuadrado y como se va a intercalar
        rostrocapturado=cv.resize(rostrocapturado, (160,160), interpolation=cv.INTER_CUBIC)
        #ponemos nombre a cada foto que saca, entra en el bucle, imagen 0 despues 1
        cv.imwrite(rutacompleta+"/imagen_{}.jpg".format(id), rostrocapturado)
        #aumenta el id
        id=id+1


    cv.imshow("resultado rostro", captura)
#cuando llega a 350 imagenes se cierra
    if id==351:
        break 

camara.release()
cv.destroyAllWindows()
