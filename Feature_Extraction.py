from skimage import exposure
from scipy.stats import itemfreq
from skimage import feature
import cv2
import numpy as np
import os
from matplotlib import pyplot as plt
from scipy.stats import signaltonoise
# opencv2 has BGR channels B = 0 index , G = 1, R = 2

class Extract_Features :

    def __init__(self, path):
        self.img = cv2.imread(path) #3 channels are read automatically
        self.path = path
        self.incorrect_shape = False
        self.corrupt_image = False
        self.shape = None
        self.preprocess_anomolies()


    #function to check various metadata of image
    def preprocess_anomolies(self):

        print self.path

        if (not isinstance(self.img, np.ndarray)):
            print "Image could not be read " + self.path
            self.corrupt_image = True
            return

        self.shape = self.img.shape

        # if image is less than 25*25*3 pixels don't consider it
        if (self.img.size <= 1875 or self.shape[0] < 25 or self.shape[1] < 25):
            print "Image is too small" + self.path
            self.corrupt_image = True
            return

        #if imaage does not have 3 channels:
        if (self.img.shape[2] != 3):
            print "This image does not contains 3 channels"
            self.incorrect_shape = True

    def plt_hist(self):

        color = ('b', 'g', 'r')
        for i, col in enumerate(color):
            histr = cv2.calcHist([self.img], [i], None, [256], [0, 256])
            print sum(histr)
            plt.plot(histr, color=col)
            plt.xlim([0, 256])
        plt.show()

    def get_histogram_for_channel(self,img,channel,bins=[256],range=[0,256]):
        return cv2.calcHist([img],[channel],None,bins,range)

    #the mean is calculated with weighted sum of all the pixel values divided by the sum of the histogram
    #np.sum(hist) here is the size of the image, in this case it is the total count of all pixels
    def get_mean_hist_for_channel(self, channel):
        hist = self.get_histogram_for_channel(self.img, channel)
        sum1 = 0
        for i in range(256):
            sum1 += hist[i] * i
        return (sum1/float(np.sum(hist)))[0]


    #function to calculate uncertainity or entropy of image -summatation(P*log10(P))
    def calculate_entropy(self):

        entropy = 0
        #for each channel
        for x in range(self.shape[2]):
            hist = self.get_histogram_for_channel(self.img, x)
            denom = float(np.sum(hist))
            prob = np.array([float(x)/denom for x in hist if x != 0])
            log_prob = np.log10(prob)
            entropy+= np.dot(prob,log_prob)

        return -entropy

    def get_height(self):
        return self.shape[0]

    def get_width(self):
        return self.shape[1]

    def get_aspect_ratio(self):
        return (self.shape[1]/float(self.shape[0]))

    #size of image on disk
    def get_filesize(self):
        return os.path.getsize(self.path)

    #compression ratio is size of uncompressed image to compressed image
    def get_compression_ratio(self):
        return (float(self.img.size)/float(self.get_filesize()))

    def get_image_area(self):
        return self.shape[0]*self.shape[1]

    def get_variance_of_rgb(self):
        b,g,r = cv2.split(self.img)
        r_mean = self.get_mean_hist_for_channel(2)
        g_mean = self.get_mean_hist_for_channel(1)
        b_mean = self.get_mean_hist_for_channel(0)

        denom = b.shape[0]*b.shape[1] - 1
        r_var = np.sum(np.square(r - r_mean))/float(denom)
        g_var = np.sum(np.square(g - g_mean)) / float(denom)
        b_var = np.sum(np.square(b - b_mean)) / float(denom)

        return r_var,g_var,b_var

    def get_skewness_of_rgb(self):
        r_var,g_var,b_var = self.get_variance_of_rgb()
        b, g, r = cv2.split(self.img)
        r_mean = self.get_mean_hist_for_channel(2)
        g_mean = self.get_mean_hist_for_channel(1)
        b_mean = self.get_mean_hist_for_channel(0)

        denom = b.shape[0]*b.shape[1] - 1
        r_numer = np.sum(np.power(r - r_mean,3)) / float(denom)
        g_numer = np.sum(np.power(g - g_mean,3)) / float(denom)
        b_numer = np.sum(np.power(b - b_mean,3)) / float(denom)

        r_skew = r_numer / np.power(np.sqrt(r_var),3)
        g_skew = g_numer / np.power(np.sqrt(g_var), 3)
        b_skew = b_numer / np.power(np.sqrt(b_var), 3)

        return r_skew,g_skew,b_skew

    def get_kurtosis_of_rgb(self):
        r_var, g_var, b_var = self.get_variance_of_rgb()
        b, g, r = cv2.split(self.img)
        r_mean = self.get_mean_hist_for_channel(2)
        g_mean = self.get_mean_hist_for_channel(1)
        b_mean = self.get_mean_hist_for_channel(0)

        denom = b.shape[0] * b.shape[1] - 1
        r_numer = np.sum(np.power(r - r_mean, 4)) / float(denom)
        g_numer = np.sum(np.power(g - g_mean, 4)) / float(denom)
        b_numer = np.sum(np.power(b - b_mean, 4)) / float(denom)

        r_kurt = r_numer / np.power(np.sqrt(r_var), 4)
        g_kurt = g_numer / np.power(np.sqrt(g_var), 4)
        b_kurt = b_numer / np.power(np.sqrt(b_var), 4)

        return r_kurt, g_kurt, b_kurt

    def get_hsv_mean(self):
        hsv_image = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        hue, sat, val = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

        denom = hue.shape[0] * hue.shape[1]
        hue_mean = np.sum(hue) / float(denom)
        sat_mean = np.sum(sat) / float(denom)
        val_mean = np.sum(val) / float(denom)

        return hue_mean,sat_mean,val_mean

    def plt_hist_hsv(self):

        hsv_image = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        hue, sat, val = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

        plt.figure(figsize=(10, 8))
        plt.subplot(311)  # plot in the first cell
        plt.subplots_adjust(hspace=.5)
        plt.title("Hue")
        plt.hist(np.ndarray.flatten(hue), bins=180)
        plt.subplot(312)  # plot in the second cell
        plt.title("Saturation")
        plt.hist(np.ndarray.flatten(sat), bins=128)
        plt.subplot(313)  # plot in the third cell
        plt.title("Luminosity Value")
        plt.hist(np.ndarray.flatten(val), bins=128)
        plt.show()

    #function to calculate uncertainity or entropy of hsv image -summatation(P*log10(P))
    def calculate_entropy_hsv(self):
        hsv_image = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)

        entropy = 0

        h_hist = self.get_histogram_for_channel( hsv_image, 0, [180], [0,180])
        s_hist = self.get_histogram_for_channel( hsv_image, 1, [256], [0, 256])
        v_hist = self.get_histogram_for_channel( hsv_image, 2, [256], [0, 256])

        denom = float(np.sum(h_hist))
        prob = np.array([float(x)/denom for x in h_hist if x != 0])
        log_prob = np.log10(prob)
        entropy += np.dot(prob,log_prob)

        denom = float(np.sum(s_hist))
        prob = np.array([float(x) / denom for x in s_hist if x != 0])
        log_prob = np.log10(prob)
        entropy += np.dot(prob, log_prob)

        denom = float(np.sum(v_hist))
        prob = np.array([float(x) / denom for x in v_hist if x != 0])
        log_prob = np.log10(prob)
        entropy += np.dot(prob, log_prob)

        return -entropy

    def get_variance_of_hsv(self):

        hsv_image = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        hue, sat, val = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

        hue_mean, sat_mean, val_mean = self.get_hsv_mean()

        denom = hue.shape[0]*hue.shape[1] - 1
        h_var = np.sum(np.square(hue - hue_mean)) / float(denom)
        s_var = np.sum(np.square(sat - sat_mean)) / float(denom)
        v_var = np.sum(np.square(val - val_mean)) / float(denom)

        return h_var,s_var,v_var

    def get_skewness_of_hsv(self):
        hue_var, sat_var, val_var = self.get_variance_of_hsv()
        hsv_image = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        hue, sat, val = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

        hue_mean, sat_mean, val_mean = self.get_hsv_mean()

        denom = hue.shape[0] * hue.shape[1] - 1
        h_numer = np.sum(np.power(hue - hue_mean,3)) / float(denom)
        s_numer = np.sum(np.power(sat - sat_mean,3)) / float(denom)
        v_numer = np.sum(np.power(val - val_mean,3)) / float(denom)

        h_skew = 0.0
        s_skew = 0.0
        v_skew = 0.0
        if h_numer != 0:
            h_skew = h_numer / np.power(np.sqrt(hue_var), 3)
        if s_numer != 0:
            s_skew = s_numer / np.power(np.sqrt(sat_var), 3)
        if v_numer != 0:
            v_skew = v_numer / np.power(np.sqrt(val_var), 3)

        return h_skew,s_skew,v_skew

    def get_kurtosis_of_hsv(self):
        hue_var, sat_var, val_var = self.get_variance_of_hsv()
        hsv_image = cv2.cvtColor(self.img, cv2.COLOR_RGB2HSV)
        hue, sat, val = hsv_image[:, :, 0], hsv_image[:, :, 1], hsv_image[:, :, 2]

        hue_mean, sat_mean, val_mean = self.get_hsv_mean()

        denom = hue.shape[0] * hue.shape[1] - 1
        h_numer = np.sum(np.power(hue - hue_mean, 4)) / float(denom)
        s_numer = np.sum(np.power(sat - sat_mean, 4)) / float(denom)
        v_numer = np.sum(np.power(val - val_mean, 4)) / float(denom)

        h_kurt = 0.0
        s_kurt = 0.0
        v_kurt = 0.0
        if h_numer != 0:
            h_kurt = h_numer / np.power(np.sqrt(hue_var), 4)
        if s_numer != 0:
            s_kurt = s_numer / np.power(np.sqrt(sat_var), 4)
        if v_numer != 0:
            v_kurt = v_numer / np.power(np.sqrt(val_var), 4)

        return h_kurt,s_kurt,v_kurt

    #lbp is that of grayscale image, radius used is 3 and number of points = 24
    def get_lbp_entropy(self):
        numberOfPoints = 24
        radius = 3
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        lbp = feature.local_binary_pattern(gray, numberOfPoints, radius, method="uniform")
        hist = itemfreq(lbp.ravel())

        entropy = 0
        #total points
        denom = np.sum(hist[:,1])
        prob = np.array([float(x) / denom for x in hist[:,1] if x != 0])
        log_prob = np.log10(prob)
        entropy += np.dot(prob, log_prob)

        return -entropy

    def get_HOG_entropy(self):
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        (H, hogImage) = feature.hog(gray, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(2, 2), transform_sqrt=True, visualise=True)
        hogImage = exposure.rescale_intensity(hogImage, out_range=(0, 255))
        hogImage = hogImage.astype("uint8")
        hist = cv2.calcHist([hogImage], [0], None, [256], [0, 256])
        entropy = 0
        denom = float(np.sum(hist))
        prob = np.array([float(x) / denom for x in hist if x != 0])
        log_prob = np.log10(prob)
        entropy += np.dot(prob, log_prob)
        # print "Entropy = " + str(-entropy)
        # cv2.imshow("Original Grayscale Image", gray)
        # cv2.imshow("HOG Image", hogImage)
        # cv2.waitKey(0)
        return -entropy

    def get_snr(self):
        return signaltonoise(cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY),None)

    def get_num_edges_and_avg_lenght(self):

        # canny edges
        sigma = 0.33
        gray = cv2.cvtColor(self.img, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)

        # compute the median of the single channel pixel intensities
        v = np.median(blurred)

        # apply automatic Canny edge detection using the computed median
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        auto = cv2.Canny(blurred, lower, upper)
        # show the images
        # cv2.imshow("Original", img)
        # cv2.imshow("Edges", np.hstack([wide, tight, auto]))

        max_lines = 0
        #will selec the maximum number of edges from all these canny images
        #parameters for hough lines
        minLineLength = 100
        maxLineGap = 10
        lines = cv2.HoughLinesP(auto, 1, np.pi / 180, 100, minLineLength, maxLineGap)
        if (isinstance(lines, np.ndarray)):
            max_lines = lines.shape[0]
        #if there are any lines
        #calculate avg length using euclidean distance
        avg_dist = 0
        dist = []
        if max_lines > 0 :
            for i in range(lines.shape[0]):
                x1, y1, x2, y2 = lines[i][0]
                dist.append(np.sqrt(np.square(x2-x1)+np.square(y2-y1)))

        if max_lines > 0:
            avg_dist = np.sum(dist)/float(max_lines)

        return max_lines,avg_dist

    #entropy obtained from histogram of noise after noise was removed from it
    def entropy_of_noise(self):
        # Denoising
        dst = cv2.fastNlMeansDenoisingColored(self.img, None, 10, 10, 7, 21)

        # cv2.imshow("Original", img)
        # cv2.imshow("After noise removal", dst)
        # cv2.imshow("Noise", img-dst)

        noise_img = self.img - dst

        entropy = 0
        # for each channel
        for x in range(self.img.shape[2]):
            hist = self.get_histogram_for_channel(noise_img, x)
            denom = float(np.sum(hist))
            prob = np.array([float(x) / denom for x in hist if x != 0])
            log_prob = np.log10(prob)
            entropy += np.dot(prob, log_prob)

        # print "entropy = " + str(-entropy)

        return -entropy