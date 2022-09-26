import keras
import numpy as np
from keras.datasets import mnist
import tensorflow as tf
from keras.models import Sequential
from keras.layers import Dense
from keras.utils import to_categorical
import csv


#*********MUST SET ONE TO TRUE AND REST TO FALSE TO RUN*************
standard_test = False #Runs code for 1 lr and 1 bs
st_test_lr = 0.1
bs = 4096

lr_optimization = False #Runs code for Multiple LRs
learning_rates = [0.0001, 0.001, 0.01, 0.1, 1.0, 10, 100]

avg_accu_lr_optimization = False #Runs Code For Collecting Avg Accu at LRs
lr = 0.001
step = 2
learning_rates_avg = []

#Generating set of learning rates
for i in range(16):
    if(lr) > 10.00:
        learning_rates_avg.append(10.000)
        break
    elif(lr) > 0.010  and (lr/step) < 0.010:
        learning_rates_avg.append(0.010)
    elif(lr) > 0.100  and (lr/step) < 0.100:
        learning_rates_avg.append(0.100)
    elif(lr) > 1.000 and (lr/step) < 1.00:
        learning_rates_avg.append(1.000)
    learning_rates_avg.append(round(lr,3))
    lr = lr * step

batch_size_optimization = True #Runs code for using Avg accu and multuple BSs
batch_sizes = [4096, 512] #Define size of the bathces of samples
#*********************************************************************


(train_images, train_labels), (test_images, test_labels) = mnist.load_data('data')

# Normalize the images.
train_images = (train_images / 255)
test_images = (test_images / 255)

# Flatten the images.
train_images = train_images.reshape((-1, 784))
test_images = test_images.reshape((-1, 784))

#print(train_images.shape) # (60000, 784)
#print(test_images.shape)  # (10000, 784)

model = Sequential([
  Dense(100, activation='relu', input_shape=(784,)),
  Dense(100, activation='relu'),
  Dense(10, activation='softmax'),
])
if(standard_test):
    filename="/Users/emadhaskett/Desktop/Research/MLP_Tensor_Flow/TensorFlow-Keras-Multilevel-Perceptron-MNIST/log.csv"
    #Ensures csv file is empty each iteration
    f = open(filename, "w")
    f.truncate()
    f.close()
    
    model.compile(
    optimizer = keras.optimizers.SGD(learning_rate= st_test_lr),
    loss='categorical_crossentropy',
    metrics=['acc'],
    )

    #Prints layers
    model.summary()


    #streams epoch results to a CSV file, If append is “True” then it appends if the file exists. 
    history_logger=tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

    print("\n\t***********TRAINING: St************\n")
    hist = model.fit(
    train_images, # training data
    to_categorical(train_labels), # training targets
    epochs=30,
    batch_size=4096,
    callbacks=[history_logger] #Stores loss and accu in csv file
    )

    #test = np.mean(hist.history['acc'])
    #print(test)

    print("\n\t************TESTING************\n")
    model.evaluate(
    test_images,
    to_categorical(test_labels)
    )
elif(lr_optimization):
    for lr in learning_rates:
        filename="/Users/emadhaskett/Desktop/Research/MLP_Tensor_Flow/TensorFlow-Keras-Multilevel-Perceptron-MNIST/lr_"+str(lr)+".csv"
        #Ensures csv file is empty each iteration
        f = open(filename, "w")
        f.truncate()
        f.close()
        
        #Builds the models layers and sets optimizer to Stochastic Gradient
        #Descent, loss to cross entropy loss
        
        model.compile(
        optimizer = keras.optimizers.SGD(learning_rate= lr),
        loss='categorical_crossentropy',
        metrics=['acc'],
        )

        #Prints layers
        model.summary()


        #streams epoch results to a CSV file, If append is “True” then it appends if the file exists. 
        history_logger=tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

        print("\n\t***********TRAINING: "+str(lr)+"************\n")
        hist = model.fit(
        train_images, # training data
        to_categorical(train_labels), # training targets
        epochs=30,
        batch_size=4096,
        callbacks=[history_logger] #Stores loss and accu in csv file
        )
        
        #test = np.mean(hist.history['acc'])
        #print(test)

        print("\n\t************TESTING: "+str(lr)+"************\n")
        model.evaluate(
        test_images,
        to_categorical(test_labels)
        )
elif(avg_accu_lr_optimization):
    filename="/Users/emadhaskett/Desktop/Research/MLP_Tensor_Flow/TensorFlow-Keras-Multilevel-Perceptron-MNIST/avg_accu_lr_.csv"
    #Ensures csv file is empty each iteration
    f = open(filename, "w")
    f.truncate()
    f.close()
    
    learningRates = []
    avg_accu = []
    for lr in learning_rates_avg:
        
        #Builds the models layers and sets optimizer to Stochastic Gradient
        #Descent, loss to cross entropy loss
        model.compile(
        optimizer = keras.optimizers.SGD(learning_rate= lr),
        loss='categorical_crossentropy',
        metrics=['acc'],
        )

        #Prints layers
        model.summary()


        #streams epoch results to a CSV file, If append is “True” then it appends if the file exists. 
        history_logger=tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

        print("\n\t***********TRAINING: "+str(lr)+"************\n")
        hist = model.fit(
        train_images, # training data
        to_categorical(train_labels), # training targets
        epochs=30,
        batch_size=4096
        )
        
        #print(lr, hist.history['acc'])
        learningRates.append(lr)
        avg_accu.append(np.mean(hist.history['acc']))
        #print(learningRates, avg_accu)
        
        #test = np.mean(hist.history['acc'])
        
        #print(test)

        print("\n\t************TESTING: "+str(lr)+"************\n")
        model.evaluate(
        test_images,
        to_categorical(test_labels)
        )
        
    #print("Learning Rates: " +str(learningRates) +"; Avg Accus: " + str(avg_accu))
    with open(filename, 'w') as f:
        writer = csv.writer(f)
        writer.writerows([learningRates])
        writer.writerows([avg_accu])
elif(batch_size_optimization):
    for bs in batch_sizes:
        filename="/Users/emadhaskett/Desktop/Research/MLP_Tensor_Flow/TensorFlow-Keras-Multilevel-Perceptron-MNIST/avg_accu_"+str(bs)+"_.csv"
        #Ensures csv file is empty each iteration
        f = open(filename, "w")
        f.truncate()
        f.close()
        
        learningRates = []
        avg_accu = []
        print("BATCH SIZE " + str(bs))
        for lr in learning_rates_avg:
            
            #Builds the models layers and sets optimizer to Stochastic Gradient
            #Descent, loss to cross entropy loss
            model.compile(
            optimizer = keras.optimizers.SGD(learning_rate= lr),
            loss='categorical_crossentropy',
            metrics=['acc'],
            )

            #Prints layers
            model.summary()


            #streams epoch results to a CSV file, If append is “True” then it appends if the file exists. 
            history_logger=tf.keras.callbacks.CSVLogger(filename, separator=",", append=True)

            print("\n\t***********TRAINING: "+str(lr)+"************\n")
            hist = model.fit(
            train_images, # training data
            to_categorical(train_labels), # training targets
            epochs=30,
            batch_size= bs
            )
            
            #print(lr, hist.history['acc'])
            learningRates.append(lr)
            avg_accu.append(np.mean(hist.history['acc']))
            #print(learningRates, avg_accu)
            
            #test = np.mean(hist.history['acc'])
            
            #print(test)

            print("\n\t************TESTING: "+str(lr)+"************\n")
            model.evaluate(
            test_images,
            to_categorical(test_labels)
            )
            
        #print("Learning Rates: " +str(learningRates) +"; Avg Accus: " + str(avg_accu))
        with open(filename, 'w') as f:
            writer = csv.writer(f)
            writer.writerows([learningRates])
            writer.writerows([avg_accu])
