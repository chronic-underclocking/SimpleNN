#pragma once
#include <vector>

struct NeuralNetwork
{
	int epochs;
	int input_layer_count;
	int neurons_per_hidden_layer;
	double learning_rate;
	std::vector<double> input_layer;
	std::vector<double> hidden_layer;
	double output_layer;
	std::vector<std::vector<double>> hidden_weights;
	std::vector<double> output_weights;
	double output_delta;
	std::vector<double> hidden_deltas;

	void randomize_weights();
	double sigmoid(double num);
	double sigmoid_derivative(double num);
	void train(const std::vector<std::vector<double>>& trainingSet, const std::vector<int>& trainingLabels);
	void predict(const std::vector<std::vector<double>>& testingSet, std::vector<int>& predictions);
	int predict(const std::vector<double>& testingSet);
	double calculateAccuracy(const std::vector<int>& predictedLabels, const std::vector<int>& testingSetLabels);
	void save(const char* name);
	void load(const char* path);
};