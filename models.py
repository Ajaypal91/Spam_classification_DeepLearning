import numpy as np
import Model_preprocessing as MP
import Simple_perceptron_model as SP
reload(SP)
reload(MP)


#get the dataset
X_train, X_test, Y_train, Y_test = MP.get_processed_dataset()
X = np.concatenate((X_train,X_test))
Y = np.concatenate((Y_train,Y_test))


#get model from k means classifier
# acc,classifier = SP.get_classifier_from_k_fold(X, Y)
# print acc
# print acc.mean() #mean score
# print acc.std()  #variance

# #get model classifier
# classifier = SP.get_simple_perceptron_classifier(X_train,Y_train,X_test,Y_test, nb_epochs=1000, batch_size=200)

#stratifier cv
cvscores = SP.get_acc_from_stratified_cross_val(X,Y)
print cvscores

#
#
# classifier_GS = SP.simple_perceptron_based_GridSearch(X_train,Y_train)
# print classifier_GS.best_params_
# print classifier_GS.best_score_
# classifier = classifier_GS.best_estimator_
# #
# #predicting the training damta
# Y_pred = classifier.predict(X_test)
# Y_pred = (Y_pred > 0.7)
# # # Y_test = (Y_test == 1)
# # # print Y_pred[0:50]
# # # print Y_test[0:50]
# # #
# # # #making confustion matrix
# from sklearn.metrics import confusion_matrix
# cm = confusion_matrix(Y_test, Y_pred)
#
# print cm