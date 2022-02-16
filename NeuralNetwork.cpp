#include "NeuralNetwork.h"
#include <random>
#include <time.h>
#include <math.h>
#include <fstream>
#include "Timer.h"

void NeuralNetwork::randomize_weights()
{
    long int seed = (unsigned)(time(0));
    srand(seed);
    for (int i = 0; i < neurons_per_hidden_layer; i++)
    {
        hidden_weights.push_back(std::vector<double>());
        for (int j = 0; j < input_layer_count; j++)
        {
            double w = (0 + ((double)(rand()) / ((double)(RAND_MAX / (2.0 - 0.0))))) - 1;
            hidden_weights[i].push_back(w);
        }
    }
    for (int i = 0; i < neurons_per_hidden_layer; i++)
    {
        double w = (0 + ((double)(rand()) / ((double)(RAND_MAX / (2.0 - 0.0))))) - 1;
        output_weights.push_back(w);
    }
}

double NeuralNetwork::sigmoid(double num)
{
    return (1 / (1 + std::exp(-num)));
}

double NeuralNetwork::sigmoid_derivative(double num)
{
    return (num * (1 - num));
}


void NeuralNetwork::train(const std::vector<std::vector<double>>& trainingSet, const std::vector<int>& trainingLabels)
{
    Timer timer("Time to train the network");
    int sets = trainingSet.size();
    hidden_layer = std::vector<double>(neurons_per_hidden_layer);
    hidden_deltas = std::vector<double>(neurons_per_hidden_layer);
    for (int epochCount = 0; epochCount < epochs; epochCount++)
    {
        for (int setCount = 0; setCount < sets; setCount++)
        {
            input_layer = trainingSet[setCount];
            
            // Feedforward
            
            for (int i = 0; i < neurons_per_hidden_layer; i++)
            {
                hidden_layer[i] = 0;
                for (int j = 0; j < input_layer_count; j++)
                {
                    hidden_layer[i] += hidden_weights[i][j] * input_layer[j];
                }
                hidden_layer[i] = sigmoid(hidden_layer[i]);
            }

            output_layer = 0;

            for (int i = 0; i < neurons_per_hidden_layer; i++)
            {
                output_layer += hidden_layer[i] * output_weights[i];
            }

            output_layer = sigmoid(output_layer);

            // Backpropogation

            output_delta = (output_layer - trainingLabels[setCount]) * sigmoid_derivative(output_layer);

            for (int i = 0; i < neurons_per_hidden_layer; i++)
            {
                hidden_deltas[i] = (output_delta * output_weights[i]) * sigmoid_derivative(hidden_layer[i]);
            }

            for (int i = 0; i < neurons_per_hidden_layer; i++)
            {
                output_weights[i] = output_weights[i] - (learning_rate * output_delta * hidden_layer[i]);
            }

            for (int i = 0; i < neurons_per_hidden_layer; i++)
            {
                for (int j = 0; j < input_layer_count; j++)
                {
                    hidden_weights[i][j] = hidden_weights[i][j] - (learning_rate * hidden_deltas[i] * input_layer[j]);
                }
            }
        }
    }
}

void NeuralNetwork::predict(const std::vector<std::vector<double>>& testingSet, std::vector<int>& predictions)
{
    int sets = testingSet.size();
    hidden_layer = std::vector<double>(neurons_per_hidden_layer);
    for (int setCount = 0; setCount < sets; setCount++)
    {
        input_layer = testingSet[setCount];

        // Feedforward

        for (int i = 0; i < neurons_per_hidden_layer; i++)
        {
            hidden_layer[i] = 0;
            for (int j = 0; j < input_layer_count; j++)
            {
                hidden_layer[i] += hidden_weights[i][j] * input_layer[j];
            }
            hidden_layer[i] = sigmoid(hidden_layer[i]);
        }

        output_layer = 0;

        for (int i = 0; i < neurons_per_hidden_layer; i++)
        {
            output_layer += hidden_layer[i] * output_weights[i];
        }

        output_layer = sigmoid(output_layer);

        if (output_layer > 0.5)
        {
            predictions.push_back(1);
        }
        else
        {
            predictions.push_back(0);
        }
    }
}

int NeuralNetwork::predict(const std::vector<double>& testingSet)
{
    input_layer = testingSet;
    hidden_layer = std::vector<double>(neurons_per_hidden_layer);

        // Feedforward

        for (int i = 0; i < neurons_per_hidden_layer; i++)
        {
            hidden_layer[i] = 0;
            for (int j = 0; j < input_layer_count; j++)
            {
                hidden_layer[i] += hidden_weights[i][j] * input_layer[j];
            }
            hidden_layer[i] = sigmoid(hidden_layer[i]);
        }

        output_layer = 0;

        for (int i = 0; i < neurons_per_hidden_layer; i++)
        {
            output_layer += hidden_layer[i] * output_weights[i];
        }

        output_layer = sigmoid(output_layer);

        if (output_layer > 0.5)
        {
            return 1;
        }
        else
        {
            return 0;
        }
}

double NeuralNetwork::calculateAccuracy(const std::vector<int>& predictedLabels, const std::vector<int>& testingSetLabels)
{
    int labelCount = testingSetLabels.size();
    double result = 0;
    for (int i = 0; i < labelCount; i++)
    {
        if (predictedLabels[i] == testingSetLabels[i]) result++;
    }
    result = (result / labelCount) * 100;
    return result;
}

void NeuralNetwork::save(const char* name)
{
    std::ofstream output(name);
    for (int i = 0; i < neurons_per_hidden_layer; i++)
    {
        for (int j = 0; j < input_layer_count; j++)
        {
            output << hidden_weights[i][j] << '\n';
        }
    }
    for (int i = 0; i < neurons_per_hidden_layer; i++)
    {
        
        output << output_weights[i] << '\n';
    }
}

void NeuralNetwork::load(const char* path)
{
    std::ifstream input(path);
    for (int i = 0; i < neurons_per_hidden_layer; i++)
    {
        hidden_weights.push_back(std::vector<double>());
        for (int j = 0; j < input_layer_count; j++)
        {
            hidden_weights[i].push_back(double());
            input >> hidden_weights[i][j];
        }
    }
    for (int i = 0; i < neurons_per_hidden_layer; i++)
    {
        output_weights.push_back(double());
        input >> output_weights[i];
    }
}