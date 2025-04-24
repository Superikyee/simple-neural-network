#include <iostream>
#include <vector>
#include <stdexcept>
#include <random>
#include <cmath>
#include <algorithm>
#include <fstream>
#include <iomanip>
#include <cassert>

#include "include/mnist/mnist_reader.hpp"

//simple helper class for matrix operations
class Matrix {
  private:
    size_t rows, cols;

  public:
    std::vector<double> data;

    unsigned int index1d(unsigned int row, unsigned int col) const {
      return col*rows+row;
    }

    Matrix():
      rows(1), cols(1), data(1, 0.0) {}

    Matrix(size_t rows, size_t cols, double init_val)
      : rows(rows), cols(cols), data(rows*cols, init_val) {}

    Matrix(size_t rows, size_t cols)
      : rows(rows), cols(cols), data(rows*cols, 0.0) {}

    Matrix(size_t rows, size_t cols, std::vector<double> init):
      rows(rows), cols(cols), data(init) {
      assert(data.size() == rows*cols);
    }

    static Matrix random(size_t rows, size_t cols, std::normal_distribution<> d, std::mt19937 gen) {
      Matrix ret(rows, cols);
      for (unsigned int i = 0; i < ret.data.size(); i++) {
        ret.data[i] = d(gen);
      }
      return ret;
    }


    //get the index of the matrix element with highest value
    unsigned int argmax() {
      unsigned int ret = 0;

      for (unsigned int i = 0; i < data.size(); i++) {
        if (data[i] > data[ret]) {
          ret = i;
        }
      }

      return ret;
    }

    //add a column matrix to every column of this matrix
    Matrix addColumnwise(Matrix other) const {
      Matrix ret(*this);

      assert(other.getCols() == 1 && other.getRows() == rows);
      for (size_t i = 0; i < data.size(); i++) {
        ret.data[i] += other.data[i % rows];
      }
      return ret;
    }

    size_t getRows() const { return rows; }
    size_t getCols() const { return cols; }

    //get element
    double& operator()(size_t row, size_t col) {
      if (row >= rows || col >= cols) throw std::out_of_range("Row index out of bounds");
      return data[index1d(row, col)];
    }

    Matrix operator+(const Matrix& other) const {
      if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Matrix dimensions must match. r1 " + std::to_string(rows) + " r2 " + std::to_string(other.rows) + " c1 " + std::to_string(cols) + " c2 " + std::to_string(other.cols));
      Matrix result(rows, cols);
      for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = data[i] + other.data[i];
      }
      return result;
    }

    Matrix operator-() const {
      Matrix result(rows, cols);
      for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = -data[i];
      }
      return result; 
    }

    Matrix operator-(const Matrix& other) const {
      if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Matrix dimensions must match. r1 " + std::to_string(rows) + " r2 " + std::to_string(other.rows) + " c1 " + std::to_string(cols) + " c2 " + std::to_string(other.cols));
      Matrix result(rows, cols);
      for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = data[i] - other.data[i];
      }
      return result;
    }

    Matrix operator*(const Matrix& other) const {
      if (cols != other.rows) throw std::invalid_argument("Matrix dimensions are incompatible for multiplication, cols: " + std::to_string(cols) + ", other.rows: " + std::to_string(other.rows));
      Matrix result(rows, other.cols);
      for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < other.cols; ++j)
          for (size_t k = 0; k < cols; ++k)
            result(i, j) += data[index1d(i, k)] * other.data[other.index1d(k, j)];
      return result;
    }

    Matrix multiplyElementWise(const Matrix& other) {
      if (rows != other.rows || cols != other.cols) throw std::invalid_argument("Matrix dimensions must match. r1 " + std::to_string(rows) + " r2 " + std::to_string(other.rows) + " c1 " + std::to_string(cols) + " c2 " + std::to_string(other.cols));
      Matrix result(rows, cols);
      for (size_t i = 0; i < data.size(); i++) {
        result.data[i] = data[i] * other.data[i];
      }
      return result;
    }

    Matrix transpose() const {
      Matrix result(cols, rows);
      for (size_t i = 0; i < rows; ++i)
        for (size_t j = 0; j < cols; ++j)
          result(j, i) = data[index1d(i, j)];
      return result;
    }

    void print() const {
      for (size_t i = 0; i < rows; i++) {
        for (size_t j = 0; j < cols; j++) {
          std::cout << data[index1d(i, j)] << " ";
        }
        std::cout << '\n';
      }
    }
};

Matrix operator*(Matrix l, double r) {
  for (unsigned int i = 0; i < l.getRows(); i++) {
    for (unsigned int j = 0; j < l.getCols(); j++) {
      l(i, j) *= r;
    }
  }
  return l;
}

Matrix operator*(double l, Matrix r) {
  return r * l;
}

//sigmoid function
double sigmoid(double z) {
  return 1.0/(1.0+std::exp(-z));
}

//derivative of sigmoid function
double sigmoidPrime(double z) {
  return sigmoid(z)*(1-sigmoid(z));
}

//apply sigmoid to matrix
Matrix sigmoid (Matrix z) {
  for (unsigned int i = 0; i < z.data.size(); i++) {
    z.data[i] = sigmoid(z.data[i]);
  }
  return z;
}

//apply sigmoid derivative to matrix
Matrix sigmoidPrime (Matrix z) {
  for (unsigned int i = 0; i < z.data.size(); i++) {
    z.data[i] = sigmoidPrime(z.data[i]);
  }
  return z;
}

//struct to store training and test data
struct NeuralData {
  std::vector<double> input;
  std::vector<double> output;
};


class Network {
  public:
    //size of each layer
    std::vector<size_t> layerSizes;
    size_t numLayers;
    std::vector<Matrix> biases;
    std::vector<Matrix> weights;

    Network(std::vector<size_t> layerSizes) : layerSizes(layerSizes), numLayers(layerSizes.size()) {

      std::random_device rd;
      std::mt19937 gen(rd());
      //itinitalize weights and biases randomly
      //standard deviation is equal to inverse square root of number of inputs
      //since the bigger the net is, the more weights there are and the smaller they should be
      std::normal_distribution<> d(0, 1.0/std::sqrt(double(layerSizes[0])));

      biases = {Matrix()};
      weights = {Matrix()};
      
      for (size_t i = 1; i < numLayers; i++) {
        biases.push_back(Matrix::random(layerSizes[i], 1, d, gen));
        weights.push_back(Matrix::random(layerSizes[i], layerSizes[i - 1], d, gen));
      }
      //first element of weights and biases will be a dummy value :) (first layers don't have any)
    }


    Matrix feedforward(Matrix a) {
      for (size_t i = 1; i < numLayers; i++) {
        //multiply previous layer activations with respective weights and add biases
        a = sigmoid(weights[i] * a + biases[i]);
      }
      return a;
    }
    

    //stochastic gradient descent
    void SGD(std::vector<NeuralData> trainingData, int epochs, int miniBatchSize, double learnRate, double regularizationParameter, double dropoutRate, std::vector<NeuralData> testData = {}) {
      size_t numTrainingData = trainingData.size();
      std::random_device rd;
      std::mt19937 gen(rd());

      for (int epoch = 0; epoch < epochs; epoch++) {
        std::shuffle(trainingData.begin(), trainingData.end(), gen);
        //divide training data randomly into batches which we iterate over
        for (size_t i = 0; i < numTrainingData; i += miniBatchSize) {
          assert(trainingData[i].input.size() == layerSizes[0]);
          size_t batchSize = std::min(i+miniBatchSize, numTrainingData)-i;

          //input data: each matrix column is one training data sample
          //each row holds the input data for one neuron
          Matrix input(layerSizes[0], batchSize);
          for (unsigned int j = 0; j < batchSize; j++) {
            std::copy(trainingData[i+j].input.begin(), trainingData[i+j].input.end(), input.data.begin() + layerSizes[0]*j);
          }
          Matrix output(layerSizes[numLayers-1], batchSize);
          for (unsigned int j = 0; j < batchSize; j++) {
            std::copy(trainingData[i+j].output.begin(), trainingData[i+j].output.end(), output.data.begin() + layerSizes[numLayers-1]*j);
          }

          updateMiniBatch(input, output, learnRate, regularizationParameter/double(trainingData.size()), 1.0-dropoutRate);
        }
        if (!testData.empty()) {
          std::cout << "Epoch " << epoch << ": " << evaluate(testData) << " / " << testData.size() << "\n";
        } else {
          std::cout << "Epoch " << epoch << " complete\n";
        }
      }
    }

  private:

    //if dropout is used, create a mask where some rows will randomly be set to 0
    //if multiplied elementwise to neural activations, some neurons will be dropped out
    Matrix createDropoutMask(size_t numNeurons, size_t numColumns, size_t numToKeep) {
      Matrix ret(numNeurons, numColumns, double(numNeurons) / double(numToKeep));
      //non-dropped activations get divided by 1-dropoutRate to compensate

      std::random_device rd;
      std::mt19937 gen(rd());
      std::uniform_int_distribution<> distrib(0, numNeurons - 1);

      int remaining = numNeurons - numToKeep;
      while (remaining > 0) {
        int idx = distrib(gen);

        if (ret(idx, 0) == 1.0) {
          for (size_t i = 0; i < numColumns; i++) {
            ret(idx, i) = 0.0; //set whole row to 0
          }
          remaining--;
        }
      }

      return ret;
    }
    
    //main training function
    void updateMiniBatch(const Matrix& input, const Matrix& output, double eta, double lambdan, double keepRate) {
      double learnFactor = eta / double(input.getCols());
      size_t numOutputs = layerSizes[numLayers-1];
     
      //matrices for storing error in weights and biases
      std::vector<Matrix> nabla_b(numLayers);
      std::vector<Matrix> nabla_w(numLayers);
      
      std::vector<Matrix> activations = {input}; //starts as vector with one element, it being the input activation matrix
      std::vector<Matrix> preActivations = {Matrix()}; //same dummy element as weights/biases

      std::vector<Matrix> dropoutMasks = {Matrix()};

      //feedforward test data inputs and store the activations, using the dropout mask if dropout is used
      for (size_t i = 1; i < numLayers; i++) {
        size_t kept = layerSizes[i] * keepRate;
                                                          
        preActivations.push_back((weights[i]*activations[i-1]).addColumnwise(biases[i]));
        if (i == numLayers-1 || keepRate == 1.0) {
          activations.push_back(sigmoid(preActivations.back()));
        } else {
          dropoutMasks.push_back(createDropoutMask(layerSizes[i], input.getCols(), kept));
          activations.push_back(sigmoid(preActivations.back()).multiplyElementWise(dropoutMasks[i]));
        }
      }

      //neural error, weight and bias in last layer
      Matrix currentLayerError = cost_derivative(activations.back(), output);
      nabla_b.back() = currentLayerError;
      nabla_w.back() = currentLayerError * activations[activations.size() - 2].transpose();

      //calculate neural error, weight and bias delta in hidden layers
      for (int l = numLayers-2; l > 0; l--) {
        if (keepRate == 1.0) {
          currentLayerError = (weights[l+1].transpose() * currentLayerError).multiplyElementWise(sigmoidPrime(preActivations[l]));
        } else {
          currentLayerError = (weights[l+1].transpose() * currentLayerError).multiplyElementWise(sigmoidPrime(preActivations[l])).multiplyElementWise(dropoutMasks[l]);
        }
        nabla_b[l] = currentLayerError;
        nabla_w[l] = currentLayerError * activations[l-1].transpose();
      }

      //update weights and biases
      for (size_t i = 1; i < numLayers; ++i) {
        weights[i] = (weights[i] * (1-eta*lambdan)) - (learnFactor * nabla_w[i]); //getCols is number of samples in current batch
        biases[i] = biases[i] - (learnFactor * nabla_b[i] * Matrix(numOutputs, 1, std::vector<double>(numOutputs, 1.0)));
        //multiplying nabla bias by a column vector to sum up the rows. clever no?
      }
    }


    //see how much of the test data the net gets correct
    int evaluate(const std::vector<NeuralData>& testData) {
      int correct = 0;
      for (const auto& data : testData) {

        Matrix in(data.input.size(), 1, data.input);
        if (feedforward(in).argmax() == Matrix(data.output.size(), 1, data.output).argmax()) {
          correct++;
        }
      }
        return correct;
    }

    //partial derivative of cost function with respect to error of a neuron
    Matrix cost_derivative(const Matrix& output_activations, const Matrix& y) {
      return output_activations - y;
    }
};



int main() {

  std::cout << std::fixed << std::setprecision(5);

  mnist::MNIST_dataset<std::vector, std::vector<uint8_t>, uint8_t> dataset = mnist::read_dataset<std::vector, std::vector, uint8_t, uint8_t>("./");
  
  std::vector<NeuralData> trainingData(dataset.training_images.size());
  std::vector<NeuralData> testData(dataset.test_images.size());

  //setup training and test data matrices
  for (size_t i = 0; i < dataset.training_images.size(); i++) {
    trainingData[i].output = std::vector<double>(10, 0.0);
    trainingData[i].output[dataset.training_labels[i]] = 1.0;
   
    trainingData[i].input = std::vector<double>(28*28);
    for (size_t j = 0; j < 28*28; j++) {
      trainingData[i].input[j] = double(dataset.training_images[i][j]) / 255.0;
    }
  }

  for (size_t i = 0; i < dataset.test_images.size(); i++) {
    testData[i].output = std::vector<double>(10, 0.0);
    testData[i].output[dataset.test_labels[i]] = 1.0;
    
    testData[i].input = std::vector<double>(28*28);
    for (size_t j = 0; j < 28*28; j++) {
      testData[i].input[j] = double(dataset.test_images[i][j]) / 255.0;
    }
  }

  std::cout << "Nbr of training images = " << dataset.training_images.size() << std::endl;
  std::cout << "Nbr of training labels = " << dataset.training_labels.size() << std::endl;
  std::cout << "Nbr of test images = " << dataset.test_images.size() << std::endl;
  std::cout << "Nbr of test labels = " << dataset.test_labels.size() << std::endl;

  //28*28 neurons in input (image size)
  //one hidden layer of 30 neurons, 10 output neurons for 10 digits
  Network net({28*28, 30, 10});
  //train for 30 epochs, mini batch size 10, learn rate 0.5
  //l2 regularization rate 5, no dropout
  net.SGD(trainingData, 30, 10, 0.5, 5.0, 0.0, testData);

  return 0;
}
