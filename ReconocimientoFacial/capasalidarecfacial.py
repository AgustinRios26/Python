import cv2 as cv
import os

dataruta="C:\Agus\Python\Reconocimiento Facial 1\data" 
#sacamos nuestra lita
listadata=os.listdir(dataruta)
entrenamientomodelo1=cv.face.EigenFaceRecognizer_create()
#a nuestro entrenamiento lo ponemos para que lo lea
entrenamientomodelo1.read("C:/Agus/Python/entrenamientoeigenfacerecognizer.xml")
#los ruidos
ruidos=cv.CascadeClassifier("C:\Agus\Python\Entrenamientoopencvruidos\opencv-master\data\haarcascades\haarcascade_frontalface_default.xml")
#camara en vivo
#si queremos agregar algun video tenemos que poner el link del archivo
#ejemplo el de auron C:\Agus\Python\Reconocimiento Facial 1/videoauron.mp4
camara = cv.VideoCapture(0)
#si tenemos la ca,ara prendida
while True:
    _,captura=camara.read()
    grises=cv.cvtColor(captura, cv.COLOR_BGR2GRAY)
    idcaptura=grises.copy()
    #detectar las caras
    caras=ruidos.detectMultiScale(grises,1.3,5)
    #enmarcar nuestra imagen
    for (x,y,e1,e2) in caras:
        #saca fragmento de nuestro rostro y almacena en carpeta, son las coordenadas de la captura
        rostrocapturado=idcaptura[y:y+e2,x:x+e1]
        #tamaño rostro, en un cuadrado y como se va a intercalar
        rostrocapturado=cv.resize(rostrocapturado, (160,160), interpolation=cv.INTER_CUBIC)
        #resultado predice esa info con lo que capture en el rostro con una prediccion si se parece un rostro
        resultado=entrenamientomodelo1.predict(rostrocapturado)
        #agregar texto de la imagen. que le forma formato resultado, el -5 significa que sube la letra
        #despues la escala y los colores, grosor y trabajamos con un rectangulo (cvline)
        cv.putText(captura, "{}".format(resultado), (x,y-5), 1,1.3, (0,255,0), 1, cv.LINE_AA)
        #la prediccion
        if resultado[1]<2000:
            cv.putText(captura, "No encontrado", (x,y-20), 2,1.1, (0,255,0), 1, cv.LINE_AA)
            cv.rectangle(captura, (x,y), (x+e1,y+e2), (255,0,0),2)   
        else:
            cv.putText(captura, "{}".format(listadata[resultado[0]]), (x,y-20), 2,0.7, (0,255,0), 1, cv.LINE_AA)
            cv.rectangle(captura, (x,y), (x+e1,y+e2), (255,0,0),2)  

    #muestra la camara con el rectangulo
    cv.imshow("Resultado", captura)
    #cerrar la ventana con la s
    if cv.waitKey(1)==ord("s"):
        break

camara.release()
cv.destroyAllWindows()
