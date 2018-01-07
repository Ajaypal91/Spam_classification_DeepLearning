from skimage import feature
from skimage import exposure
import numpy as np
import os
import cv2
from scipy.stats import itemfreq
from scipy.stats import signaltonoise
from matplotlib import pyplot as plt

# img_path = "Data/Image_Spam_Hunter/Ham/zzz_01736_2f9f491a0b_m.jpg"
# img_path = "Data/Image_Spam_Hunter/Ham/zzz_104_c1a35af827_m.jpg"

# img_path = "Data/Image_Spam_Hunter/SpamImages/0nf2Sd2Za7.jpg"
img_path = "Data/Dredze/personal_image_ham/1011.png"
# img_path = "Data/Dredze/personal_image_spam/1.png"
# img_path = "Data/Image_Spam_Hunter/SpamImages/4Eg4s6AtLj.jpg"

img = cv2.imread(img_path)
print 'ads'

# hist1 = cv2.calcHist([img],[0],None,[256],[0,256])
# hist2 = cv2.calcHist([img],[1],None,[256],[0,256])
# hist3 = cv2.calcHist([img],[2],None,[256],[0,256])
#
# sum1 = 0
# for i in range(256):
#     sum1+=hist1[i]*i
#
# sum2 = 0
# for i in range(256):
#     sum2+=hist2[i]*i
#
# sum3 = 0
# for i in range(256):
#     sum3+=hist3[i]*i



# print sum1/float(np.sum(hist1))
# print sum2/float(np.sum(hist1))
# print sum3/float(np.sum(hist1))

# print min(hue)
# print max(hue)


#for hsv
# hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
# # hsv = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]
#
# h_hist = cv2.calcHist([hsv], [0], None, [180], [0, 180])
# s_hist = cv2.calcHist([hsv], [1], None, [256], [0, 256])
# v_hist = cv2.calcHist([hsv], [2], None, [256], [0, 256])
#
#
# color = ('b', 'g', 'r')
# plt.plot(h_hist, color=color[0])
# plt.plot(s_hist, color=color[1])
# plt.plot(v_hist, color=color[2])
# plt.xlim([0, 256])
# plt.show()

#for lbp
# numberOfPoints = 24
# radius = 8
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# lbp = feature.local_binary_pattern(gray, numberOfPoints,radius, method="uniform")
# hist = itemfreq(lbp.ravel())
#
# print 'dsa'

#for hog decriptor
# hog = cv2.HOGDescriptor()
# h = hog.compute(img)
# h2 = hog.computeGradient(img)
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# (H, hogImage) = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
# hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
# hogImage = hogImage.astype("uint8")
# hist = cv2.calcHist([hogImage],[0],None,[256],[0,256])
# entropy = 0
# denom = float(np.sum(hist))
# prob = np.array([float(x)/denom for x in hist if x != 0])
# log_prob = np.log10(prob)
# entropy+= np.dot(prob,log_prob)
# print "Entropy = " + str(-entropy)
# cv2.imshow("Original Grayscale Image", gray)
# cv2.imshow("HOG Image", hogImage)
# cv2.waitKey(0)


# fd, hog_image = feature.hog(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), orientations=8, pixels_per_cell=(8, 8), cells_per_block=(1, 1), transform_sqrt=True, visualise=True)
# fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)

# ax1.axis('off')
# ax1.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cmap=plt.cm.gray)
# ax1.set_title('Input image')
# ax1.set_adjustable('box-forced')

# # Rescale histogram for better display
# hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))

# ax2.axis('off')
# ax2.imshow(hog_image, cmap=plt.cm.gray)
# ax2.set_title('Histogram of Oriented Gradients')
# ax1.set_adjustable('box-forced')
# plt.show()

# hog_hist = cv2.calcHist(hog_image,[0],None,[256],[0,256])
# plt.plot(hog_hist, color='g')
# plt.show()

# fd, hog_image = feature.hog(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), orientations=9, pixels_per_cell=(10, 10), cells_per_block=(2, 2), visualise=True)
#
# # Rescale histogram for better display
# # visualize the HOG image
# hogImage = exposure.rescale_intensity(hog_image, out_range=(0, 255))
# hogImage = hogImage.astype("uint8")
# cv2.imshow("HOG Image", hogImage)
# cv2.waitKey(0)

# #canny edges
# sigma=0.33
# gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# blurred = cv2.GaussianBlur(gray, (3, 3), 0)

# # apply Canny edge detection using a wide threshold, tight
# # threshold, and automatically determined threshold
# wide = cv2.Canny(blurred, 10, 200)
# tight = cv2.Canny(blurred, 225, 250)
# # compute the median of the single channel pixel intensities
# v = np.median(blurred)

# # apply automatic Canny edge detection using the computed median
# lower = int(max(0, (1.0 - sigma) * v))
# upper = int(min(255, (1.0 + sigma) * v))
# auto = cv2.Canny(blurred, lower, upper)
# wide = cv2.Canny(blurred, 10, 200)
# tight = cv2.Canny(blurred, 225, 250)
# # show the images
# cv2.imshow("Original", img)
# cv2.imshow("Edges", np.hstack([wide, tight, auto]))

# minLineLength = 100
# maxLineGap = 10
# lines = cv2.HoughLinesP(auto, 1, np.pi / 180, 100, minLineLength, maxLineGap)
# lines_wide = cv2.HoughLinesP(wide, 1, np.pi / 180, 100, minLineLength, maxLineGap)
# lines_tight = cv2.HoughLinesP(tight, 1, np.pi / 180, 100, minLineLength, maxLineGap)

# for i in range(lines.shape[0]):
#     x1, y1, x2, y2 = lines[i][0]
#     cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

# cv2.imwrite('houghlines5.jpg',img)
# print "number of line =" +str(lines.shape[0])
# cv2.waitKey(0)

# #snr (Signalt to noise ration)
# snr = signaltonoise(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), axis=None)
# print snr
# def get_histogram_for_channel(img, channel, bins=[256], range=[0, 256]):
#     return cv2.calcHist([img], [channel], None, bins, range)

# # Denoising
# dst = cv2.fastNlMeansDenoisingColored(img,None,10,10,7,21)
# # cv2.imshow("Original", img)
# # cv2.imshow("After noise removal", dst)
# # cv2.imshow("Noise", img-dst)

# noise_img = img - dst

# entropy = 0
# #for each channel
# for x in range(img.shape[2]):
#     hist = get_histogram_for_channel(noise_img, x)
#     denom = float(np.sum(hist))
#     prob = np.array([float(x)/denom for x in hist if x != 0])
#     log_prob = np.log10(prob)
#     entropy+= np.dot(prob,log_prob)

# print "entropy = " + str(-entropy)
# cv2.waitKey(0)