import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
import glob
import imutils




def padding(img, H, W):
    # change the validity
    if int(H<img.shape[0]) or int(W<img.shape[1]): # if the input image has higher dimension than the required padding
        return [img,False] # 
    
    # total pixels to be added
    h = H - int(img.shape[0]) # Pixels to be added along y axis =256
    w = W - int(img.shape[1]) # Pixels to be added along x axis =256
    
    # distribution of the pixel
    h_T = h//2 # Top height to add =128
    h_B = h - h_T # Bottom height to add =128
    w_L = w//2 # Left width to add =128
    w_R = w - w_L # Right width to add =128
    
    # Apply Padding
    padded = np.pad(img, pad_width=[(h_T, h_B),(w_L, w_R)], constant_values=255) # To make the resulting image in the size of H*T
    return padded,True




# Main block 

image_path = "FULL_PATH_OF_THE_DIRECOTRY_CONTAINING_IMAGE(S)"
parent = glob.glob(image_path+"/*")
for im_path in parent:
 img_org = cv.imread(im_path)
 img_org = cv.cvtColor(img_org,cv.COLOR_BGR2RGB)
 cv.imwrite('testing.jpg', img_org)
 img = imutils.resize(img_org,width=512) # choose one parameter H or W (H and W are the height or width from the ideal expected size)
 
 gray = cv.cvtColor(img,cv.COLOR_RGB2GRAY)
 padded_org,flag = padding(gray,H=256,W=512)
 plt.subplot(1,2,1)
 plt.title(f"Original({img_org.shape[0]}, {img_org.shape[1]}")
 plt.imshow(img_org)
 plt.subplot(1,2,2)
 plt.title(f"Padded{padded_org.shape}")
 plt.imshow(padded_org,cmap='gray')
 plt.show()

