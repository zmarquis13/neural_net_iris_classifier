/**
* Neuron.hpp - header file for Neuron struct
* Author: Zeno Marquis
* HW6: Artificial Neural Networks
* 5/10/2024
*/


#include <vector>
#include <cstdlib>
#include <cmath>
#include <iostream>
#include <stdlib.h>
#include <ctime>
#include <stdio.h>

#include "Iris.hpp"

using namespace std;

//Neuron struct definition
struct Network{

    Network(){
        
        //4-3-3 network structure
        for (int i = 0; i < 12; i++){

            //generates a random number from -1 to 1
            float random_weight = 2*(static_cast <float> (rand())/ static_cast <float> (RAND_MAX)) - 1;

            //set weights from input to hidden
            if (i % 4 == 0){
                input_1_weights.push_back(random_weight);
            } else if (i % 4 == 1){
                input_2_weights.push_back(random_weight);
            } else if (i % 4 == 2){
                input_3_weights.push_back(random_weight);
            } else {
                input_4_weights.push_back(random_weight);
            }
        }

        //set weights from hidden to output
        for (int i = 0; i < 9; i++){
            float random_weight = 2*(static_cast <float> (rand())/ static_cast <float> (RAND_MAX)) - 1;

            if (i % 3 == 0){
                intermediate_1_weights.push_back(random_weight);
            } else if (i % 3 == 1){
                intermediate_2_weights.push_back(random_weight);
            } else {
                intermediate_3_weights.push_back(random_weight);
            }

        }

        //set bias weights (biases 0-2 impact hidden neurons, 3-5 impact output)
        for (int i = 0; i < 6; i++){
            //generates a random number from -1 to 1
            float random_weight = 2*(static_cast <float> (rand())/ static_cast <float> (RAND_MAX)) - 1;

            bias.push_back(random_weight);
            
        }

    }

    //weights from input to hidden layer
    vector<float> input_1_weights;
    vector<float> input_2_weights;
    vector<float> input_3_weights;
    vector<float> input_4_weights;

    //weights from hidden layer to output layer
    vector<float> intermediate_1_weights;
    vector<float> intermediate_2_weights;
    vector<float> intermediate_3_weights;

    //bias weights
    vector<float> bias;

    /**
     * Forward propagates through the network based on the given flower
     * 
     * @param input_flower an Iris flower class
     * 
     * @return a vector representing the three output node values
     */
    vector<float> forward_propagate(Iris input_flower){

        vector<float> flower_attributes;

        //store flower attributes as inputs
        flower_attributes.push_back(input_flower.sepal_length);
        flower_attributes.push_back(input_flower.sepal_width);
        flower_attributes.push_back(input_flower.petal_length);
        flower_attributes.push_back(input_flower.petal_width);

        //raw = bias + weight*value + weight*value...
        float raw_intermediate_1 = bias[0] + input_1_weights[0]*flower_attributes[0] + input_2_weights[0]*flower_attributes[1] + input_3_weights[0]*flower_attributes[2] + input_4_weights[0]*flower_attributes[3];

        //output = sigmoid(raw)
        float intermediate_1 = sigmoid(raw_intermediate_1);

        float raw_intermediate_2 = bias[1] + input_1_weights[1]*flower_attributes[0] + input_2_weights[1]*flower_attributes[1] + input_3_weights[1]*flower_attributes[2] + input_4_weights[1]*flower_attributes[3];

        float intermediate_2 = sigmoid(raw_intermediate_2);

        float raw_intermediate_3 = bias[2] + input_1_weights[2]*flower_attributes[0] + input_2_weights[2]*flower_attributes[1] + input_3_weights[2]*flower_attributes[2] + input_4_weights[2]*flower_attributes[3];

        float intermediate_3 = sigmoid(raw_intermediate_3);

        //output layer calculations
        float raw_setosa = bias[3] + intermediate_1_weights[0]*intermediate_1 + intermediate_2_weights[0]*intermediate_2 + intermediate_3_weights[0]*intermediate_3;
        float out_setosa = sigmoid(raw_setosa);

        float raw_versicolor = bias[4] + intermediate_1_weights[1]*intermediate_1 + intermediate_2_weights[1]*intermediate_2 + intermediate_3_weights[1]*intermediate_3;

        float out_versicolor = sigmoid(raw_versicolor);

        float raw_virginica = bias[5] + intermediate_1_weights[2]*intermediate_1 + intermediate_2_weights[2]*intermediate_2 + intermediate_3_weights[2]*intermediate_3;

        float out_virginica = sigmoid(raw_virginica);

        vector<float> results;
        results.push_back(out_setosa);
        results.push_back(out_versicolor);
        results.push_back(out_virginica);

        return results;

    }

    /**
     * Trains the network using a given flower
     * 
     * @param input_flower an Iris flower class
     */
    void train(Iris input_flower){

        vector<float> flower_attributes;

        //scaled + normalized flower attributes as input
        flower_attributes.push_back(input_flower.sepal_length);
        flower_attributes.push_back(input_flower.sepal_width);
        flower_attributes.push_back(input_flower.petal_length);
        flower_attributes.push_back(input_flower.petal_width);

        //same process as forward function, use a different function so that
        //hidden layer values are available without passing them
        float raw_intermediate_1 = bias[0] + input_1_weights[0]*flower_attributes[0] + input_2_weights[0]*flower_attributes[1] + input_3_weights[0]*flower_attributes[2] + input_4_weights[0]*flower_attributes[3];

        float intermediate_1 = sigmoid(raw_intermediate_1);

        float raw_intermediate_2 = bias[1] + input_1_weights[1]*flower_attributes[0] + input_2_weights[1]*flower_attributes[1] + input_3_weights[1]*flower_attributes[2] + input_4_weights[1]*flower_attributes[3];

        float intermediate_2 = sigmoid(raw_intermediate_2);

        float raw_intermediate_3 = bias[2] + input_1_weights[2]*flower_attributes[0] + input_2_weights[2]*flower_attributes[1] + input_3_weights[2]*flower_attributes[2] + input_4_weights[2]*flower_attributes[3];

        float intermediate_3 = sigmoid(raw_intermediate_3);

        float raw_setosa = bias[3] + intermediate_1_weights[0]*intermediate_1 + intermediate_2_weights[0]*intermediate_2 + intermediate_3_weights[0]*intermediate_3;

        float out_setosa = sigmoid(raw_setosa);

        float raw_versicolor = bias[4] + intermediate_1_weights[1]*intermediate_1 + intermediate_2_weights[1]*intermediate_2 + intermediate_3_weights[1]*intermediate_3;

        float out_versicolor = sigmoid(raw_versicolor);

        float raw_virginica = bias[5] + intermediate_1_weights[2]*intermediate_1 + intermediate_2_weights[2]*intermediate_2 + intermediate_3_weights[2]*intermediate_3;

        float out_virginica = sigmoid(raw_virginica);

        float true_values[3] = {0.0, 0.0, 0.0};
        true_values[input_flower.flower_type] = 1.0;

        float loss1, loss2, loss3;

        loss1 = true_values[0] - out_setosa;
        loss2 = true_values[1] - out_versicolor;
        loss3 = true_values[2] - out_virginica;

        //learning rate (set experimentally)
        float lr = 0.9;

        //error = derivative(output value)*(expected - actual)
        float output_error1 = loss1*sigmoid_derivative(out_setosa);
        float output_error2 = loss2*sigmoid_derivative(out_versicolor);
        float output_error3 = loss3*sigmoid_derivative(out_virginica);

        //hidden layer error = o'(t)*(sum(weight(i,j)*error))
        float intermediate_error1 = sigmoid_derivative(intermediate_1)*(intermediate_1_weights[0]*output_error1 + intermediate_1_weights[1]*output_error2 + intermediate_1_weights[2]*output_error3);

        float intermediate_error2 = sigmoid_derivative(intermediate_2)*(intermediate_2_weights[0]*output_error2 + intermediate_2_weights[1]*output_error2 + intermediate_2_weights[2]*output_error3);

        float intermediate_error3 = sigmoid_derivative(intermediate_3)*(intermediate_3_weights[0]*output_error1 + intermediate_3_weights[1]*output_error2 + intermediate_3_weights[2]*output_error3);


        //update input layer biases
        bias[4] += lr*output_error1;
        bias[5] += lr*output_error2;
        bias[6] += lr*output_error3;

        //update hidden layer biases
        bias[0] += lr*intermediate_error1;
        bias[1] += lr*intermediate_error2;
        bias[2] += lr*intermediate_error3;

        //update hidden to output weights with:
        //weight = weight + learning rate*output neuron error*input neuron value
        intermediate_1_weights[0] += lr*output_error1*intermediate_1;
        intermediate_1_weights[1] += lr*output_error2*intermediate_1;
        intermediate_1_weights[2] += lr*output_error3*intermediate_1;

        intermediate_2_weights[0] += lr*output_error1*intermediate_2;
        intermediate_2_weights[1] += lr*output_error2*intermediate_2;
        intermediate_2_weights[2] += lr*output_error3*intermediate_2;

        intermediate_3_weights[0] += lr*output_error1*intermediate_3;
        intermediate_3_weights[1] += lr*output_error2*intermediate_3;
        intermediate_3_weights[2] += lr*output_error3*intermediate_3;

        //update input to hidden weights in the same manner
        input_1_weights[0] += lr*intermediate_error1*flower_attributes[0];
        input_1_weights[1] += lr*intermediate_error2*flower_attributes[0];
        input_1_weights[2] += lr*intermediate_error3*flower_attributes[0];

        input_2_weights[0] += lr*intermediate_error1*flower_attributes[1];
        input_2_weights[1] += lr*intermediate_error2*flower_attributes[1];
        input_2_weights[2] += lr*intermediate_error3*flower_attributes[1];

        input_3_weights[0] += lr*intermediate_error1*flower_attributes[2];
        input_3_weights[1] += lr*intermediate_error2*flower_attributes[2];
        input_3_weights[2] += lr*intermediate_error3*flower_attributes[2];

        input_4_weights[0] += lr*intermediate_error1*flower_attributes[3];
        input_4_weights[1] += lr*intermediate_error2*flower_attributes[3];
        input_4_weights[2] += lr*intermediate_error3*flower_attributes[3];

    }

    /**
     * Tests the performance of the network in classifying a given flower
     * 
     * @param input_flower an Iris flower class
     * 
     * @return a bool representing whether or not the classification was correct
     */
    bool test(Iris input_flower){

        vector<float> results = forward_propagate(input_flower);

        float out_setosa = results[0];
        float out_versicolor = results[1];
        float out_virginica = results[2];

        int prediction;

        //one-hot encoding of output
        if (out_setosa >= out_versicolor && out_setosa >= out_virginica){
            prediction = 0;

        } else if (out_versicolor >= out_virginica){
            prediction = 1;

        } else {
            prediction = 2;

        }

        //if output matches flower type, return true
        if (prediction == input_flower.flower_type){
            return true;
        } else {
            return false;
        }
    }

    /**
     * Prints the network's prediction as to the type of the given flower
     * 
     * @param input_flower an Iris flower class
     */
    void predict(Iris input_flower){
        
        vector<float> results = forward_propagate(input_flower);

        float out_setosa = results[0];
        float out_versicolor = results[1];
        float out_virginica = results[2];

        //make prediction based on highest value
        if (out_setosa >= out_versicolor && out_setosa >= out_virginica){
            cout << "predicted: setosa\n";
        } else if (out_versicolor >= out_virginica){
            cout << "predicted: versicolor\n";
        } else {
            cout << "predicted: virginica\n";
        }
    }

    //sigmoid function
    float sigmoid(float input){
        return (1/(1 + exp(-1*input)));
    }

    //sigmoid derivative function
    float sigmoid_derivative(float input){
        return (input*(1 - input));
    }

};