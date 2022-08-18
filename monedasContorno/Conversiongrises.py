#ddd
from cv2 import cv2
#imread lee la imagen
imagen=cv2.imread("C:\Agus\Python\Monedas contorno\contorno.jpg")
#poner la imagen gris, para ver el code abrir en google Color Space Conversions opencv
grises=cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
#TRESHOLD umbral simple, separar el objeto del entorno que se encuentra en blanco y negro, en google threshold opencv
#el maximo 255 siempre, el minimo podemos cambiarlo
#Como el umbral tiene 2 salidas agregamos un _,
_, umbral=cv2.threshold(grises, 100, 255, cv2.THRESH_BINARY)
#una vez umbralizado tenemos que hacer un reverse para poder buscar los contornos
#buscar en google findcontours opencv, primero la imagen, despues una lista para este modo y chain approx simple, la none es mas compleja 
contorno, jerarquia = cv2.findContours(umbral, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
#dibujar nuestro contorno y personalizar colores
#drawcontours opencv en google
#primero imagen original, despues el contorno, 1 para el primer contorno, 2 para el siguiente y -1 para todos los contornos
#despues el color rgb y al final el grosor
cv2.drawContours(imagen, contorno, -1, (91, 40, 27),3)



#mostrar la imagen
cv2.imshow("Original",imagen)
cv2.imshow("Grises", grises)
cv2.imshow("Umbral", umbral)
print(_)

#valor 1 camara o video (fluido) en imagenes 0
cv2.waitKey(0)
#destruya ventana emergentes
cv2.destroyAllWindows()