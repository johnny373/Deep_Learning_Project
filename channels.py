import cv2
import matplotlib.pyplot as plt
import sys
import numpy as np
import os

def get_14_channels(img):
#     print(img)
    channels = np.zeros((img.shape[0], img.shape[1], 3))
#    print(img.shape)
#     channels[:, :, 0] = img[:, :, 2]  # R
#     channels[:, :, 1] = img[:, :, 1]  # G
#     channels[:, :, 2] = img[:, :, 0]  # B
    
    channels[:, :, 0] = (2 * img[:, :, 1]) - img[:, :, 2] - img[:, :, 0]  # ExG
    ExG = channels[:, :, 0]
    channels[:, :, 1] = (1.4 *img[:, :, 2]) - img[:, :, 1]  # ExR
    channels[:, :, 2] = (0.881 * img[:, :, 1]) - (0.441 * img[:, :, 2]) - (0.385 * img[:, :, 0]) - 18.78745  # CIVE
#     channels[:, :, 6] = (img[:, :, 1] - img[:, :, 2]) / (img[:, :, 1] + img[:, :, 2])  # NDI
    
#     hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
#     channels[:, :, 7] = hsv[:, :, 0]  # HUE
#     channels[:, :, 8] = hsv[:, :, 1]  # SAT
#     channels[:, :, 9] = hsv[:, :, 2]  # VAL
    
#     channels[:, :, 10] = cv2.Sobel(ExG, cv2.CV_16S, 1, 0)  # dxExG
#     channels[:, :, 11] = cv2.Sobel(ExG, cv2.CV_16S, 0, 1)  # dyExG
#     channels[:, :, 4] = cv2.Laplacian(ExG, cv2.CV_64F)  # LaplacianExG
    
#    blurred = cv2.GaussianBlur(ExG, (5, 5), 0)
#    channels[:, :, 3] = cv2.Canny(img, 30, 150)  # CannyExG
    channels_3 = np.zeros((img.shape[0], img.shape[1], 3))

    channels_3[:,:,0] = (2 * img[:, :, 1]) - img[:, :, 2] - img[:, :, 0]
    channels_3[:,:,1] = (0.881 * img[:, :, 1]) - (0.441 * img[:, :, 2]) - (0.385 * img[:, :, 0]) - 18.78745
    channels_3[:,:,2] = (img[:, :, 1] - img[:, :, 2]) / (img[:, :, 1] + img[:, :, 2])  # NDI
    
    return channels_3
def get_me(image):
    gaussian_3 = cv2.GaussianBlur(image, (0, 0), 10.0)
    unsharp_image = cv2.addWeighted(image, 4.0, gaussian_3, -3.0, 0)
    channels = image+unsharp_image
    return channels
    
Path = "/home/ma/Pytorch-Medical-Segmentation/data/Test/final_maize/"
Path_out = 'data/Test/final_maize_img+unsharp/'
allFileList = os.listdir(Path)
for file in allFileList:
    file1 = file.split(".jpg",1)
    img_pth = Path+file
    print(img_pth)
    image = cv2.imread(img_pth)
    channels = get_me(image)
    print(file1[0]+":"+str(channels.shape))
    # np.save(Path_out+file1[0]+".npy", channels)
    cv2.imwrite(Path_out+file1[0]+'.jpg', channels)