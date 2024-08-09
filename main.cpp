/**
* main.cpp - main file for neural network program
* Author: Zeno Marquis
* HW6: Artificial Neural Networks
* 5/10/2024
*/


#include <iostream>
#include <vector>
#include <fstream>
#include <algorithm>
#include <string>
#include <cstdlib>
#include <random>

#include "Iris.hpp"
#include "Network.hpp"

using namespace std;

ifstream open_file(string filename);

/**
     * Main function for program
     */
int main(){

    //open data file
    ifstream iris_data = open_file("ANN - Iris data.txt");

    vector<Iris> flowers;
    string line;
    float sl, sw, pl, pw;
    int type; 

    getline(iris_data, line);

    //calculate total values to get mean
    float total_sl, total_sw, total_pl, total_pw;
    total_sl = 0.0;
    total_sw = 0.0;
    total_pl = 0.0;
    total_pw = 0.0;


    while (!iris_data.eof() && line.size() != 0){

        //get variables and type from file
        sl = stof(line.substr(0, 3));
        total_sl += sl;

        sw = stof(line.substr(4, 3));
        total_sw += sw;

        pl = stof(line.substr(8, 3));
        total_pl += pl;

        pw = stof(line.substr(12, 3));
        total_pw += pw;


        if (line.size() == 27){
            type = 0;
        } else if (line.size() == 31){
            type = 1;
        } else {
            type = 2;
        }

        flowers.push_back(Iris(sl, sw, pl, pw, type));

        getline(iris_data, line);

    }

    //track standard deviation of variables
    float dev_pl = 0.0;
    float dev_pw = 0.0;
    float dev_sl = 0.0;
    float dev_sw = 0.0;

    for (int i = 0; i < 150; i++){
        //normalize to a mean of 0 and add difference from mean to stdev
        flowers.at(i).petal_length -= total_pl/150;
        dev_pl += fabs(flowers.at(i).petal_length);

        flowers.at(i).petal_width -= total_pw/150;
        dev_pw += fabs(flowers.at(i).petal_width);

        flowers.at(i).sepal_length -= total_sl/150;
        dev_sl += fabs(flowers.at(i).sepal_length);

        flowers.at(i).sepal_width -= total_sw/150;
        dev_sw += fabs(flowers.at(i).sepal_width);

    }

    //divide each value by the standard deviation to get stdev of 1
    for (int i = 0; i < 150; i++){
        flowers.at(i).petal_length = flowers.at(i).petal_length/(dev_pl/150);
        flowers.at(i).petal_width = flowers.at(i).petal_width/(dev_pw/150);
        flowers.at(i).sepal_length = flowers.at(i).sepal_length/(dev_sl/150);
        flowers.at(i).sepal_width = flowers.at(i).sepal_width/(dev_sw/150);
    }

    srand(time(NULL));
    
    vector<Iris> training;
    vector<Iris> testing;

    //training data = flowers 0 to 119
    //testing data = flowers 120-149
    while (flowers.size() != 0){

        //shuffle the data into training and testing data
        int index = rand()%flowers.size();

        if (flowers.size() > 30){

            training.push_back(flowers.at(index));
            flowers.erase(flowers.begin() + index);

        } else {
            testing.push_back(flowers.at(index));
            flowers.erase(flowers.begin() + index);
        }
    }

    Network net;

    //train network
    for (int i = 0; i < 120; i++){
        
        if (i == 0){
            cout << "Training...\n";
        }

        Iris next_flower = training.at(i);
        net.train(next_flower);

    }

    cout << "Training complete\n";

    float accuracy;
    float correct = 0.0;
    float total = 0.0;

    cout << "Enter 1 to see the results of testing 30 flowers, 2 to enter";
    cout << " your own flower, or 3 to do both: ";
    int input;
    cin >> input;

    if (input == 1 || input == 3){
        for (int i = 0; i < 30; i++){
        
            total += 1.0;

            Iris next_flower = testing.at(i);

            //count correct results
            if (net.test(next_flower)){
                correct += 1.0;
            }
        }

        //calculate and print accuracy
        accuracy = correct/total;
        cout << "accuracy: " << accuracy*100 << '%' << endl;

    }
    
    if (input == 2 || input == 3){

        float sl, sw, pl, pw;

        //get input and normalize it
        cout << "Enter flower sepal length (in mm): ";
        cin >> sl;
        sl -= total_sl/150;
        sl = sl/(dev_sl/150);

        cout << "Enter flower sepal width (in mm): ";
        cin >> sw;
        sw -= total_sw/150;
        sw = sw/(dev_sw/150);

        cout << "Enter flower petal length (in mm): ";
        cin >> pl;
        pl -= total_pl/150;
        pl = pl/(dev_pl/150);

        cout << "Enter flower petal width (in mm): ";
        cin >> pw;
        pw -= total_pw/150;
        pw = pw/(dev_pw/150);

        Iris *input_flower = new Iris(sl, sw, pl, pw, 3);

        //make prediction based on input
        net.predict(*input_flower);
        
    }

    return 0;
}


/**
     * Opens the given input file
     * 
     * @param filename a string representing the input file name
     * 
     * @return the input file stream generated from opening the file
     */
ifstream open_file(string filename){

    //make sure file opens
    ifstream infile;
    infile.open(filename); 
    if (infile.fail()) {
        cerr << "ERROR: Error opening file";
        exit(EXIT_FAILURE);
    } else {
        return infile;
    }
}