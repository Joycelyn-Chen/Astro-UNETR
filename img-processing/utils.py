import cv2 

def apply_otsus_thresholding(image):
    thres_T, binary_image = cv2.threshold(image, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return cv2.bitwise_not(binary_image), thres_T

def find_connected_components(binary_image):
    return cv2.connectedComponentsWithStats(binary_image)