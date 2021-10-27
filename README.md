# Harris Corner Detector :
## Introduction : 
In this work, we present an implementation of the Harris corner detector to detect corners in a chessboard image. This feature detector relies on the analysis of the eigenvalues of the autocorrelation matrix. The algorithm comprises several steps, including several measures for the classification of corners, a generic non-maximum suppression method for selecting interest points, and the possibility to obtain the corners position with subpixel accuaracy.

[image1]: ./Input/chessboard00.png "Input Image"
![alt text][image1]

A corner is a point whose local neighborhood stands in two dominant and different edge directions. In other words, a corner can be interpreted as the junction of two edges, where and edge is sudden change in image brightness. Corners are the important deatures in the image, and they are generally termed as interest points which are invariant to translation, rotation and illumination.

The idea behind the Harris corner detection algorithm is to locate interest points where the surrounding neighbourhood shows edges in more the one direction. The basic idea is to find the difference in intensity for a displacement of (u,v) in all directions.

## Process of Harris Corner Detection Algorithm :

The Harris Corner Detector algorithm in simple words is as follows :

**Step 1:** Convert original image to grayscale.

**Step 2:** Apply Kernel to find the x and y gradient values for every pixel in the grayscale image.

**Step 3:** Apply Gaussian filter to smooth out any noise.

**Step 4:** For each pixel p in the grayscale image, consider m x m window around it and compute the corner strength function (With each such window found, a score R is computed).
 
**Step 5:** Find all pixels that exceed a certain threshold are the local maxima within a certain window (to prevent redundant dupes of features)

**Step 6:** Compute a feature descriptor of all such points.

## Output : 

[image6]: ./Output/Output.png ""
![alt text][image6]

## Conclusion :
In this work, we presented an implementation of the Harris corner detector to select the prominent features on the image.

The final output depends on the threshold and the distance value. Also, it depend on the size of the matrix. For 3x3 matrix, it was not so good and contained some non-corner points too. But for 9x9 matrix, the output was accurate enough to in-built harris detector.

The detector algorithm implemented relies on the analysis of the eigenvalues of the auto-correlation matrix. The algorithm comprises several steps, including several measures for the classification of corners, a generic non-maximum suppression method for selecting interest points, and the possibility to obtain the corners position with subpixel accuracy.
