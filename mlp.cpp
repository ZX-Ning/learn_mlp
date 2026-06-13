#define EIGEN_STACK_ALLOCATION_LIMIT 10000000
#define EIGEN_INITIALIZE_MATRICES_BY_ZERO

#include <algorithm>
#include <cstdio>
#include <print>
#include <span>
//
#include <Eigen/Dense>
//
#include <omp.h>

#include "data.h"

Data train_data[60'000];
Data test_data[10'000];

void read_data() {
    FILE* f_train_in = fopen("data/mnist/mnist_train.bin", "rb");
    FILE* f_test_in = fopen("data/mnist/mnist_test.bin", "rb");
    fread(train_data, sizeof(Data), 60'000, f_train_in);
    fread(test_data, sizeof(Data), 10'000, f_test_in);
    fclose(f_train_in);
    fclose(f_test_in);
}

constexpr int HIDDEN_LAYER_1_SIZE = 64;
using Input = Eigen::Vector<float, 28 * 28>;
using Hide1 = Eigen::Vector<float, HIDDEN_LAYER_1_SIZE>;
using Output = Eigen::Vector<float, 10>;

struct Weights {
    Eigen::Matrix<float, HIDDEN_LAYER_1_SIZE, 28 * 28> w1;
    Hide1 b1;
    Eigen::Matrix<float, 10, HIDDEN_LAYER_1_SIZE> w2;
    Output b2;

    Weights& operator+=(const Weights& rhs) {
        w1 += rhs.w1;
        b1 += rhs.b1;
        w2 += rhs.w2;
        b2 += rhs.b2;
        return *this;
    }
    Weights& operator-=(const Weights& rhs) {
        w1 -= rhs.w1;
        b1 -= rhs.b1;
        w2 -= rhs.w2;
        b2 -= rhs.b2;
        return *this;
    }
    Weights& operator*=(float rhs) {
        w1 *= rhs;
        b1 *= rhs;
        w2 *= rhs;
        b2 *= rhs;
        return *this;
    }
    Weights& operator/=(float rhs) {
        w1 /= rhs;
        b1 /= rhs;
        w2 /= rhs;
        b2 /= rhs;
        return *this;
    }
    void serialize(FILE* file) {
        fwrite(this, sizeof(Weights), 1, file);
    }
    void deserialize(FILE* file) {
        fread(this, sizeof(Weights), 1, file);
    }
};

#pragma omp declare reduction(weights_add:Weights : omp_out += omp_in) \
    initializer(omp_priv = Weights{})

float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
}

// float sigmoid_dx(float x) {
//     return sigmoid(x) * (1 - sigmoid(x));
// }

Output delta(const Output& output, int label) {
    Output expected{};
    expected[label] = 1.f;
    auto delta = output - expected;
    return delta;
}

float loss(const Output& delta) {
    return delta.squaredNorm() / 2.f;
}

struct Result {
    Input input;
    Hide1 z1;
    Hide1 a1;
    Output z2;
    Output y_hat;
};

// y_hat = sigmoid(z2)
// z2 = W2 a1 + b2
// a1 = sigmoid(z1)
// z1 = W1 input + b1
Result forward(const Input& in, const Weights& weights) {
    auto z1 = (weights.w1 * in) + weights.b1;
    auto a1 = (z1).unaryExpr(&sigmoid);
    auto z2 = (weights.w2 * a1) + weights.b2;
    auto y_hat = (z2).unaryExpr(&sigmoid);
    return {in, std::move(z1), std::move(a1), std::move(z2), std::move(y_hat)};
}

// delta: y_hat - y
// return gradient of weights
Weights backward(const Output& delta, const Weights& weights, const Result& forward) {
    Weights gradient{};

    // auto d_z2 = delta.cwiseProduct(forward.z2.unaryExpr(&sigmoid_dx));
    auto d_z2 = delta.cwiseProduct(forward.y_hat.unaryExpr([](float x) { return x * (1 - x); }));
    gradient.b2 = d_z2;
    gradient.w2 = d_z2 * forward.a1.transpose();
    auto d_a1 = weights.w2.transpose() * d_z2;
    // auto d_z1 = d_a1.cwiseProduct(forward.z1.unaryExpr(&sigmoid_dx));
    auto d_z1 = d_a1.cwiseProduct(forward.a1.unaryExpr([](float x) { return x * (1 - x); }));
    gradient.b1 = d_z1;
    gradient.w1 = d_z1 * forward.input.transpose();

    return gradient;
}

Weights randomWeights() {
    using w1_T = decltype(Weights::w1);
    using w2_T = decltype(Weights::w2);
    return {
        w1_T::Random() * std::sqrt(6.f / (784.f + HIDDEN_LAYER_1_SIZE)),
        Hide1::Zero(),
        w2_T::Random() * std::sqrt(6.f / (HIDDEN_LAYER_1_SIZE + 10.f)),
        Output::Zero()
    };
}

Input inputFromRaw(uint8_t* data) {
    return Eigen::Map<Eigen::Vector<uint8_t, 28 * 28>>(data, 28 * 28).cast<float>() / 255.f;
}

constexpr int EPOCHS = 100;
constexpr float RATE = 0.5f;
constexpr int BATCH_SIZE = 100;

Weights train() {
    constexpr size_t size = std::size(train_data);
    Weights weights = randomWeights();

    // training loop
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        float total_loss = 0.f;
        for (int i = 0; i < size; i += BATCH_SIZE) {
            Weights gradient{};
            float batch_total_loss = 0.f;
            int end = std::min<int>(i + BATCH_SIZE, size);
            int batch_size = end - i;
// one batch
#pragma omp parallel for reduction(+ : batch_total_loss) \
    reduction(weights_add : gradient)
            for (int j = i; j < end; j++) {
                auto& row = train_data[j];
                auto input = inputFromRaw(row.data);
                auto forward_result = forward(input, weights);
                auto d = delta(forward_result.y_hat, row.label);
                auto l = loss(d);
                auto gradient_step = backward(d, weights, forward_result);
                batch_total_loss += l;
                gradient += gradient_step;
            }

            gradient /= batch_size;
            total_loss += batch_total_loss;
            gradient *= RATE;
            weights -= gradient;
        }
        std::println("Epoch: {}, Avg loss: {}", epoch + 1, total_loss / size);
    }
    return weights;
}

int num_pick(const std::span<float, 10> guess) {
    float max = -INFINITY;
    int pick = 0;
    for (int i = 0; i < 10; i++) {
        if (guess[i] > max) {
            max = guess[i];
            pick = i;
        }
    }
    return pick;
}

void test(const Weights& weights) {
    int errors = 0;
    for (size_t i = 0; i < std::size(test_data); i++) {
        auto& row = test_data[i];
        auto forward_result = forward(inputFromRaw(row.data), weights);
        auto guess = forward_result.y_hat.array();
        int pick = num_pick(std::span<float, 10>{guess.begin(), guess.end()});
        if (pick != row.label) {
            // std::println("Not match on test {}, expect {}, get {}.", i, row.label, pick);
            errors++;
        }
    }
    std::println(
        "Test done. {} tested. Errors: {}. Error rate: {}",
        std::size(test_data),
        errors,
        static_cast<float>(errors) / std::size(test_data)
    );
}

int main() {
    read_data();
    auto weights = train();
    FILE* fOut = fopen("weights.bin", "wb");
    weights.serialize(fOut);
    test(weights);
    fclose(fOut);
    return 0;
}
