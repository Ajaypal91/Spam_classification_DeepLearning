import numpy as np
import pandas as pd
import os
import cv2
import csv
import Feature_Extraction as FE
reload(FE)

# input_path = "Data/Image_Spam_Hunter/SpamImages"
# output_path = "Data/Image_Spam_Hunter/ImageHunter_Spam.csv"
input_path = "Data/Dredze/spam_archive_jmlr_1"
output_path = "Data/Dredze/spam_archive_jmlr_1.csv"



#keep count of images in folder
count_of_images = 0
#keep track of different file formats
file_formats = set()

features_header = [['name','height','width','aspect_ratio','compression_ratio','file_size','image_area',
                    'entr_color','b_mean','g_mean','r_mean',
                   'r_skew','g_skew','b_skew',
                   'r_var','g_var','b_var',
                   'r_kurt','g_kurt','b_kurt',
                   'entr_hsv','h_mean','s_mean','v_mean',
                   'h_skew','s_skew','v_skew',
                   'h_var','s_var','v_var',
                   'h_kurt','s_kurt','v_kurt',
                   'lbp','entr_HOG','edges','avg_edge_len','snr','entr_noise']]


#function to write header for csv file
def write_header_to_file(writer):
    writer.writerows(features_header)


def add_color_features(feature_extractor):
    image_features = []
    #add entropy feature
    image_features.append(feature_extractor.calculate_entropy())
    # add mean of blue channel
    image_features.append(feature_extractor.get_mean_hist_for_channel(0))
    # add mean of green channel
    image_features.append(feature_extractor.get_mean_hist_for_channel(1))
    # add mean of red channel
    image_features.append(feature_extractor.get_mean_hist_for_channel(2))
    #add skew features for rgb hist
    r_skew,g_skew,b_skew = feature_extractor.get_skewness_of_rgb()
    image_features.extend([r_skew,g_skew,b_skew])
    # add variance features for rgb hist
    r_var, g_var, b_var = feature_extractor.get_variance_of_rgb()
    image_features.extend([r_var,g_var,b_var])
    # add kurtosis features for rgb hist
    r_kurt, g_kurt, b_kurt = feature_extractor.get_kurtosis_of_rgb()
    image_features.extend([r_kurt,g_kurt,b_kurt])
    #add entropy of hsv
    image_features.append(feature_extractor.calculate_entropy_hsv())
    #add hsv mean features
    h_mean, s_mean, v_mean = feature_extractor.get_hsv_mean()
    image_features.extend([h_mean,s_mean,v_mean])
    # add hsv mean features
    h_skew, s_skew, v_skew = feature_extractor.get_skewness_of_hsv()
    image_features.extend([h_skew, s_skew, v_skew])
    # add hsv mean features
    h_var, s_var, v_var = feature_extractor.get_variance_of_hsv()
    image_features.extend([h_var, s_var, v_var])
    # add hsv mean features
    h_kurt, s_kurt, v_kurt = feature_extractor.get_kurtosis_of_hsv()
    image_features.extend([h_kurt, s_kurt, v_kurt])
    # add entropy of lbp
    image_features.append(feature_extractor.get_lbp_entropy())
    # add entropy of HOG image
    image_features.append(feature_extractor.get_HOG_entropy())
    #edges and avg_edge len
    edges, edge_len = feature_extractor.get_num_edges_and_avg_lenght()
    image_features.extend([edges,edge_len])
    #signal to noise ratio
    image_features.append(feature_extractor.get_snr())
    #entropy of noise
    image_features.append(feature_extractor.entropy_of_noise())

    # feature_extractor.plt_hist()
    # feature_extractor.plt_hist_hsv()
    return image_features


def add_metadata_features(feature_extractor):
    image_feature = []
    image_feature.append(feature_extractor.get_height())
    image_feature.append(feature_extractor.get_width())
    image_feature.append(feature_extractor.get_aspect_ratio())
    image_feature.append(feature_extractor.get_compression_ratio())
    image_feature.append(feature_extractor.get_filesize())
    image_feature.append(feature_extractor.get_image_area())
    return image_feature

def create_feature_csv():
    feature_file = open(output_path, "w")
    writer = csv.writer(feature_file)
    write_header_to_file(writer)

    features = []
    for image_name in os.listdir(input_path):
        image_features = []
        global count_of_images
        count_of_images+=1
        file_formats.add(image_name[-3:])

        feature_extractor = FE.Extract_Features(input_path+"/"+image_name)

        #check if image is not corrupted and is a rgb image
        if (not feature_extractor.corrupt_image):

            #add name of image
            image_features.append(image_name)

            #add metadata features
            image_features.extend(add_metadata_features(feature_extractor))

            #add color features
            image_features.extend(add_color_features(feature_extractor))

            # add image_features to features
            features.append(image_features)


    #write to csv
    writer.writerows(features)

create_feature_csv()
print "Total number of images in folder = " + str(count_of_images)
print "Different file formats = " + ", ".join(list(file_formats))
