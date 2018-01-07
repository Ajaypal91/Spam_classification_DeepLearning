from keras.models import Sequential
from keras.layers import Dense, Dropout
from sklearn.model_selection import GridSearchCV,cross_val_score, StratifiedKFold
from keras.wrappers.scikit_learn import KerasClassifier


def simple_perceptron_model():
    # initialization of ANN
    classifier = Sequential()

    # adding input layer and first hidden layer
    classifier.add(Dense(output_dim=15, init="uniform", activation="relu", input_dim=38))
    classifier.add(Dropout(p=0.1))

    # adding output layer
    classifier.add(Dense(output_dim=1, init="uniform", activation="sigmoid"))

    # compiling ANN
    classifier.compile(optimizer="adam", loss="binary_crossentropy", metrics=['accuracy'])

    return classifier

def simple_perceptron_model_for_GS(optimizer):
    # initialization of ANN
    classifier = Sequential()

    # adding input layer and first hidden layer
    classifier.add(Dense(output_dim=20, init="uniform", activation="relu", input_dim=38))

    # adding output layer
    classifier.add(Dense(output_dim=1, init="uniform", activation="sigmoid"))

    # compiling ANN
    classifier.compile(optimizer=optimizer, loss="binary_crossentropy", metrics=['accuracy'])

    return classifier


def get_simple_perceptron_classifier(X_train, Y_train, X_test,Y_test,nb_epochs=500, batch_size=100):
    
    classifier = simple_perceptron_model()
    # fitting ANN to training set
    classifier.fit(X_train, Y_train,validation_split=(X_test,Y_test), batch_size=batch_size, nb_epoch=nb_epochs)
    
    return classifier

def simple_perceptron_based_GridSearch(X_train, Y_train):
     classifier = KerasClassifier(build_fn=simple_perceptron_model_for_GS)    
     
     paramerters = { 'batch_size':[100,200],
                'nb_epoch':[250, 500],
                 'optimizer':['adam','rmsprop']
                 }    
     #5 fold cross validation
     grid_search = GridSearchCV(estimator=classifier,
                                param_grid=paramerters,
                                scoring='accuracy',
                                n_jobs = 1,
                                cv=10)
     grid_search = grid_search.fit(X_train,Y_train)
     return grid_search


def get_acc_from_stratified_cross_val(X,Y,batch_size=100,nb_epochs=500):
    seed = 7
    # define 10-fold cross validation test harness
    kfold = StratifiedKFold(n_splits=10, shuffle=True, random_state=seed)
    cvscores = []
    for train, test in kfold.split(X, Y):
        # create model
        classifier = simple_perceptron_model()
        # Fit the model
        classifier.fit(X[train], Y[train], batch_size=batch_size, nb_epoch=nb_epochs)
        # evaluate the model
        scores = classifier.evaluate(X[test], Y[test], verbose=0)
        print("%s: %.2f%%" % (classifier.metrics_names[1], scores[1] * 100))
        cvscores.append(scores[1] * 100)
    return cvscores