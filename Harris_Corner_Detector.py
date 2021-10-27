# Import packages
import numpy as np 
import matplotlib.pyplot as plt 
import cv2

# Read input image
image = cv2.imread('chessboard00.png')
# Covert the image to Grayscale
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
# Convert image to numpy array
img = np.asarray(gray).astype('float32') 

# Plot Color image and Grayscale image
fig, axs = plt.subplots(1,2, figsize=(12, 5))
axs[0].set_title('Original Image')
axs[0].axis('off')
axs[0].imshow(image)
axs[1].set_title('Grayscale Image')
axs[1].axis('off')
axs[1].imshow(img)
plt.show()

# Spatial derivative calculation
## Define kernels for convolution
a = np.array([[-1,0,1]])  
aT = np.transpose(a)     
b = np.array([[1,1,1]])   
bT = np.transpose(b)       
abT = a*bT  # Kernel x
baT = b*aT  # Kernel y
## Convolve image with Kernel x and y
Ix = cv2.filter2D(img, cv2.CV_32F, abT) # Ix = dI/dx
Iy = cv2.filter2D(img, cv2.CV_32F, baT) # Iy = dI/dy

# Ix², Iy² & IxIy
A = np.power(Ix,2)
B = np.power(Iy,2)
C = Ix*Iy

# Apply Gaussian blur (kernel_size = 9x9 & σ = 2)
Ix_2 = cv2.GaussianBlur(A, (9,9), 2) # <Ix²> = g x Ix²
Iy_2 = cv2.GaussianBlur(B, (9,9), 2) # <Iy²> = g x Iy²
IxIy = cv2.GaussianBlur(C, (9,9), 2) # <IxIy> = g x IxIy

# Harris response calculation with E & R
k = 0.04
height= img.shape[0]
width = img.shape[1] 
window_size = 3 
step = 1
Sxx = np.zeros((height,width))
Sxy = np.zeros((height,width))
Syy = np.zeros((height,width))
detM = np.zeros((height,width))
trM = np.zeros((height,width))
E = np.zeros((height,width)) 
R = np.zeros((height,width)) 
M = np.zeros((height,width))
Max = np.zeros((height,width))
result = np.zeros((height,width))
max = 0

## We will use Summation to find Harris response
## credit : https://muthu.co/harris-corner-detector-implementation-in-python/
## Calculate the sum of squares of our gradient at each pixel by shifting a window over all the pixels in our image
for y in range(step, height-step):
    for x in range(step, width-step):
        # Shift window
        windowIxx = Ix_2[y-step:y+step+1, x-step:x+step+1]
        windowIyy = Iy_2[y-step:y+step+1, x-step:x+step+1]
        windowIxy = IxIy[y-step:y+step+1, x-step:x+step+1]
        # Summation
        Sxx[y,x] = windowIxx.sum()
        Sxy[y,x] = windowIxy.sum()
        Syy[y,x] = windowIyy.sum()
        # M' = (Ix_square * Iy_square) - (Ixy_square²)
        detM[y,x] = (Sxx[y,x] * Syy[y,x]) - (Sxy[y,x]**2)
        # Trace
        trM[y,x] = Sxx[y,x] + Syy[y,x]
        # R = det(M) -k*trace(M)² (Harris Response)
        R[y,x] = detM[y,x] - k*(trM[y,x]**2)
        #E = det(M)/trace(M)
        E[y,x] = detM[y,x]/trM[y,x]
        # Select max value in E
        if E[y,x] > max:
            max = E[y,x]
        # Select max value in R
        if R[y,x] > max:
            max = R[y,x]
    
# Find Local max
def localMax(p,i,j) :  
        return ( p[i][j] > np.array([p[i-1][j],p[i][j-1],p[i-1][j-1],p[i+1][j],p[i][j+1],p[i+1][j+1],p[i-1][j+1],p[i+1][j-1]]).max() )

# Find corners using R (Harris response)
for y in range(height-step) : 
    for x in range(width-step) : 
        if R[y,x] > 0.04*max  and localMax(R,y,x) : 
            result[y,x] = 1 
## Plot corners
a, b = np.where(result == 1)
plt.plot(b, a, 'r+')
plt.imshow(img, 'gray')
plt.axis('off')
plt.savefig('R_Output')
plt.show()

# Find corners using E
for y in range(height-step) : 
    for x in range(width-step) :
        if E[y,x] > 0.04*max  and localMax(E,y,x) : 
            result[y,x] = 1 

## Plot corners
c, d = np.where(result == 1)
plt.plot(d, c, 'r+')
plt.imshow(img, 'gray')
plt.axis('off')
plt.savefig('E_Output')
plt.show()