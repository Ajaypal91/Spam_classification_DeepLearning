import numpy as np
import pandas as pd
from sklearn.preprocessing import  StandardScaler
from sklearn.model_selection import train_test_split

#1 for spam 0 for ham
#parameters
features_header = ['height','width','aspect_ratio','compression_ratio','file_size','image_area',
                    'entr_color','b_mean','g_mean','r_mean',
                   'r_skew','g_skew','b_skew',
                   'r_var','g_var','b_var',
                   'r_kurt','g_kurt','b_kurt',
                   'entr_hsv','h_mean','s_mean','v_mean',
                   'h_skew','s_skew','v_skew',
                   'h_var','s_var','v_var',
                   'h_kurt','s_kurt','v_kurt',
                   'lbp','entr_HOG','edges','avg_edge_len','snr','entr_noise']


spam_dataset_path = "Data/Image_Spam_Hunter/ImageHunter_Spam.csv"
ham_dataset_path =  "Data/Image_Spam_Hunter/ImageSpamHunter_Ham.csv"

test_size_for_split = 0.2



def read_dataset(path,colmns):
    dataset = pd.read_csv(path,usecols=colmns)
    return dataset

def get_processed_dataset():
    # read dataset
    spam_dataset = read_dataset(spam_dataset_path, features_header)
    ham_dataset = read_dataset(ham_dataset_path, features_header)

    # remove duplicates to remove the same image features in dataset
    spam_dataset.drop_duplicates()
    ham_dataset.drop_duplicates()

    # add labels
    spam_dataset["label"] = 1
    ham_dataset["label"] = 0

    # join all dataset
    dataset = pd.concat([spam_dataset, ham_dataset])

    # X, Y features
    X = dataset.iloc[:, :38].values
    Y = dataset.iloc[:, 38].values

    # split the dataset
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=test_size_for_split, random_state=0)

    # scale the features
    sc = StandardScaler()
    X_train = sc.fit_transform(X_train)
    X_test = sc.transform(X_test)

    return X_train,X_test,Y_train,Y_test

