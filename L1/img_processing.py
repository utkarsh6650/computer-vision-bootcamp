import cv2
img = cv2.imread('image.jpg')

#resizing
resize_img = cv2.resize(img, (300,300))

#grey
grey_img = cv2.cvtColor(resize_img, cv2.COLOR_BGR2GRAY)

#bluring
blur_img = cv2.GaussianBlur(resize_img, (15,15), 0)

#edges
edge_img = cv2.Canny(resize_img, 100, 100)

cv2.imshow('Original Image', img)
cv2.imshow('Resized Image', resize_img)
cv2.imshow('Grey Image', grey_img) 
cv2.imshow('Blurred Image', blur_img)
cv2.imshow('Edge Image', edge_img)
cv2.waitKey(0)
cv2.destroyAllWindows()