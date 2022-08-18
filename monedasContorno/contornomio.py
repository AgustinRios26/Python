#ddd
from cv2 import cv2
#imread lee la imagen
imagen=cv2.imread("C:\Agus\Python\Monedas contorno\contorno.jpg")
#mostrar la imagen
cv2.imshow("C:\Agus\Python\Monedas contorno\contorno.jpg",imagen)
#valor 1 camara o video (fluido) en imagenes 0
cv2.waitKey(0)
#destruya ventana emergentes
cv2.destroyAllWindows()