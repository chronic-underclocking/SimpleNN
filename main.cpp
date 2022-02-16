#include <iostream>
#include <string>
#include <filesystem>
#include <algorithm>
#include <random>
#include <vector>
#include <cmath>
#include "NeuralNetwork.h"
#define STB_IMAGE_IMPLEMENTATION
#include "stb_image.h"

enum class Animal {CAT, DOG};

struct Image
{
    std::vector<double> pixels;
    int label;
};

void imageToDoubleArray(std::string file, std::vector<double>& image)
{
    int width, height, bpp;
    uint8_t* greyscale_image = stbi_load(file.c_str(), &width, &height, &bpp, 1);
    if (greyscale_image != nullptr && width > 0 && height > 0)
    {
        if (bpp == 1)
        {
            double mean = 0;
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    int num = static_cast<int>(greyscale_image[i * width + j]);
                    mean += (double)num;
                }
            }
            mean /= width * height; 
            double diff_mean = 0;
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    int num = static_cast<int>(greyscale_image[i * width + j]);
                    double diff = (double)num - mean;
                    diff *= diff;
                    diff_mean += diff;
                }
            }
            diff_mean /= width * height;
            double std_deviation = std::sqrt(diff_mean);
            for (int i = 0; i < height; i++)
            {
                for (int j = 0; j < width; j++)
                {
                    int num = static_cast<int>(greyscale_image[i * width + j]);
                    double pixel = ((double)num - mean) / std_deviation;
                    image.push_back(pixel);
                }
            }
        }
    }
    stbi_image_free(greyscale_image);
}

int main()
{  
    std::vector<Image> data;
    std::string path = "Cats & Dogs Sample Dataset";
    for (const auto& entry : std::filesystem::directory_iterator(path))
    {
        Image img;
        std::string file = entry.path().string();
        imageToDoubleArray(file, img.pixels);
        img.label = ((file.find("cat") != -1) ? (int)Animal::CAT : (int)Animal::DOG);
        data.push_back(img);
    }
    
    // Shuffle the data
    auto rd = std::random_device{};
    auto rng = std::default_random_engine{rd()};
    std::shuffle(std::begin(data), std::end(data), rng);
    
    //Split the data into training (75%) and testing (25%) sets
    int data_size = data.size();
    int testing_size = data_size / 4;
    std::vector<std::vector<double>> trainingSetFeatures;
    std::vector<std::vector<double>> testingSetFeatures;
    std::vector<int> trainingSetLabels;
    std::vector<int> testingSetLabels;
    for (int i = 0; i < testing_size; i++)
    {
        testingSetFeatures.push_back(data[i].pixels);
        testingSetLabels.push_back(data[i].label);
    }
    for (int i = testing_size; i < data_size; i++)
    {
        trainingSetFeatures.push_back(data[i].pixels);
        trainingSetLabels.push_back(data[i].label);
    }

    // Create the NN
    NeuralNetwork nn;

    // Set the NN architecture
    nn.epochs = 150;
    nn.input_layer_count = 40 * 40;
    nn.neurons_per_hidden_layer = 100;
    nn.learning_rate = 0.2;
    nn.randomize_weights();

    // Train the NN
    nn.train(trainingSetFeatures, trainingSetLabels);

    // Test the NN
    std::vector<int> predictedLabels;
    nn.predict(testingSetFeatures, predictedLabels);
    double accuracy = nn.calculateAccuracy(predictedLabels, testingSetLabels);
    std::cout << "Accuracy: " << accuracy << '\n';

    // Save the model (final weights)
    nn.save("model.txt");
 
    // Load the model and use it on an image
    NeuralNetwork nn2;
    nn2.epochs = 150;
    nn2.input_layer_count = 40 * 40;
    nn2.neurons_per_hidden_layer = 100;
    nn2.learning_rate = 0.2;
    nn2.load("model.txt");
    std::vector<double> sample;
    std::string img = "sample.jpg";
    imageToDoubleArray(img, sample);
    int samplePrediction = nn2.predict(sample);
    system(img.c_str());
    if (samplePrediction == 1) std::cout << "Dog";
    else std::cout << "Cat";
   
    return 0;
}