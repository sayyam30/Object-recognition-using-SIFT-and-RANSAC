# SIFT ALGORITHM
import cv2
import matplotlib.pyplot as plt

img1 = cv2.imread('img_1.jpg', cv2.IMREAD_GRAYSCALE)
img2 = cv2.imread('img_2.jpg', cv2.IMREAD_GRAYSCALE)

sift = cv2.SIFT_create()
kp1, des1 = sift.detectAndCompute(img1, None)
kp2, des2 = sift.detectAndCompute(img2, None)

bf = cv2.BFMatcher()
matches = bf.knnMatch(des1, des2, k=2)

# Apply Lowe's ratio test
good = []
for m, n in matches:
    if m.distance < 0.75 * n.distance:
        good.append(m)

img3 = cv2.drawMatches(img1, kp1, img2, kp2, good, None, flags=2)
plt.imshow(img3), plt.title("SIFT Matches"), plt.show()

# RANSAC ALGORITHM
import numpy as np
if len(good) > 10:
    src_pts = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1,1,2)
    dst_pts = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1,1,2)

    M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
    matchesMask = mask.ravel().tolist()

    ransac_result = cv2.drawMatches(img1, kp1, img2, kp2, good, None,
                                    matchColor=(0,255,0),
                                    matchesMask=matchesMask,
                                    flags=2)
    plt.imshow(ransac_result), plt.title("RANSAC Filtered Matches"), plt.show()

# HARRIS CORNER DETECTION
img = cv2.imread('grid_harris.png')
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray = np.float32(gray)

dst = cv2.cornerHarris(gray, 2, 3, 0.04)
dst = cv2.dilate(dst, None)

img[dst > 0.01 * dst.max()] = [0, 0, 255]  # red corner dots
plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)), plt.title("Harris Corners"), plt.show()



