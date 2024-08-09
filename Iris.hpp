/**
* Neuron.hpp - header file for Iris class
* Author: Zeno Marquis
* HW6: Artificial Neural Networks
* 5/10/2024
*/


#pragma once

#include <vector>
#include <set>
#include <iostream>

using namespace std;

//iris class definition
struct Iris{

    Iris(float sl, float sw, float pl, float pw, int type){
        sepal_length = sl;
        sepal_width = sw;
        petal_length = pl;
        petal_width = pw;
        flower_type = type;
    }

    /**
     * Prints type of the flower
     */
    void print_type(){
        if (flower_type == 0){
            cout << "setosa";
        } else if (flower_type == 1){
            cout << "versicolor";
        } else {
            cout << "virginica";
        }
    }

    //measurement attributes
    float sepal_length, sepal_width, petal_length, petal_width;

    //0 = setosa, 1 = versicolor, 2 = virginica
    int flower_type;

};
