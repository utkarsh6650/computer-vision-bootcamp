import cv2
img1 = cv2.imread('image.jpg')

cv2.imshow('Image', img1)
cv2.waitKey(0)
cv2.destroyAllWindows()

cv2.imwrite('saved_image.jpg', img1)