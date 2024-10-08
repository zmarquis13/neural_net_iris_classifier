Zeno Marquis

5/10/2024


Project:
This project is a neural network classifier implemented from scratch in c++. It uses
no neural network-related packages and all aspects of the neural network are
hand-coded. The neural network classifies iris flowers based on measurement data.

Overview:
    This program uses a neural network with 4 input neurons, 3 hidden layer
    neurons, and 3 output layer neurons to classify iris flowers using
    data on flower length, flower width, sepal length, and sepal width. The data
    comes from Fisher's iris database.

Assumptions:
    This program uses random initial assignment of weights for the neural
    network. As a result, there is a chance that it does not converge on an
    effective classification solution. Additional sources of randomness include
    random shuffling of the flowers into training and testing data. As such,
    there is variation in the accuracy of the program from run to run.

Usage:
    Compile the program using the command "make" or "make ANN" and run it using
    the command "./ANN"


Files:
   
    main.cpp - main file for the program

    Network.hpp - contains network struct

    Iris.hpp - contains iris class

    Makefile - builds the program


Testing:
    The program was tested by running it numerous times using training data,
    and by entering made up flowers using mashups and variations of flower
    data in the data set.




