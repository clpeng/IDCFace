import cv2

img = cv2.imread('super_2_img/015216_enrypt_super_2_fingerprint.png')
img = cv2.resize(img, dsize=(128,128))
cv2.imwrite('super_2_img/015216_enrypt_super_2_fingerprint_respize_128.png', img)