from cv2 import cv2
import numpy as np
#hacer modelado matematico
original = cv2.imread("C:\Agus\Python\Monedas contorno\monedas.jpg")
gris = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
#si es erroneo cambiar valores gauss y kernel siempre impares
valorgauss=3
valorkernel=3
#Contorno mas exacto, hace un suavizado y las imagenes borrosas gaussianblur
#Tiene 2 variables, la imagen y unas variables usamos desenfoque gaussiano
gauss= cv2.GaussianBlur(gris, (valorgauss,valorgauss), 0)
#cannyopencv saca el ruido que ha quedado valor maximo 255 y minimo 0
#Tendremos los contornos esos valores aleatorios
canny=cv2.Canny(gauss, 60, 100)
#numpy trabaja con funciones, me interesa los contornos de afuera para contarlos 
# a eso decirle tamaño de la funcion y el npuint8 es de 8 bytes siempre 8bytes en matrices
kernel= np.ones((valorkernel,valorkernel),np.uint8)
#Cierre de los contornos, morphology corrige la imagen, corrige ruido
#Ruido se encuentra adentro, entonces se usa clausura y no apertura
cierre=cv2.morphologyEx(canny, cv2.MORPH_CLOSE, kernel)
#poner los resultados, al cambiar morfologia hay que hacer un copy, sacamos el dato como external y usamos contornos simples
contornos, jerarquia=cv2.findContours(cierre.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#muestra las monedas encontradas, para eso ponerle formato y usar len para contar los caracteres en la cadena de contornos
print("monedas encontradas:{}".format(len(contornos)))
#Dibujar los contornos(sobre la imagen original), -1 todos los contornos, 
#despues poner el color, despues el tamaño
cv2.drawContours(original,contornos, -1, (0,0,255), 2)

#Mostrar resultado
cv2.imshow("Grises", gris)
cv2.imshow("Gauss", gauss)
cv2.imshow("Canny", canny)
cv2.imshow("Resultado", original)
cv2.imshow("Cierre", cierre)

cv2.waitKey(0)

