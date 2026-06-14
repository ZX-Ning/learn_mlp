#define EIGEN_STACK_ALLOCATION_LIMIT 10000000
#define EIGEN_INITIALIZE_MATRICES_BY_ZERO

#include <algorithm>
#include <cstdio>
#include <memory>
#include <print>
#include <random>
#include <ranges>
//
#include <omp.h>
//
#include <Eigen/Dense>

#include "data.h"

using std::unique_ptr;
constexpr int HIDDEN_LAYER_1_SIZE = 64;
constexpr int EPOCHS = 30;
constexpr float RATE = 5.f;
constexpr int BATCH_SIZE = 100;
constexpr int TRAIN_DATA_SIZE = 60'000;
constexpr int TEST_DATA_SIZE = 10'000;

struct Mnist {
    Data train_data[TRAIN_DATA_SIZE];
    Data test_data[TEST_DATA_SIZE];
};

unique_ptr<Mnist> read_data() {
    auto result = std::make_unique<Mnist>();
    FILE* f_train_in = fopen("data/mnist/mnist_train.bin", "rb");
    FILE* f_test_in = fopen("data/mnist/mnist_test.bin", "rb");
    fread(result->train_data, sizeof(Data), 60'000, f_train_in);
    fread(result->test_data, sizeof(Data), 10'000, f_test_in);
    fclose(f_train_in);
    fclose(f_test_in);
    return result;
}

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

float sigmoid(float x) {
    return 1.f / (1.f + std::exp(-x));
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

using InputBatch = Eigen::Matrix<float, 28 * 28, Eigen::Dynamic>;
struct ResultBatch {
    InputBatch input;
    // Eigen::Matrix<float, HIDDEN_LAYER_1_SIZE, Eigen::Dynamic> z1;
    Eigen::Matrix<float, HIDDEN_LAYER_1_SIZE, Eigen::Dynamic> a1;
    // Eigen::Matrix<float, 10, Eigen::Dynamic> z2;
    Eigen::Matrix<float, 10, Eigen::Dynamic> y_hat;
};
ResultBatch forward_batch(const InputBatch& in, const Weights& weights) {
    auto z1 = (weights.w1 * in).colwise() + weights.b1;
    auto a1 = (z1).unaryExpr(&sigmoid);
    auto z2 = (weights.w2 * a1).colwise() + weights.b2;
    auto y_hat = (z2).unaryExpr(&sigmoid);
    return {in, std::move(a1), std::move(y_hat)};
}

Weights backward_batch(
    const Eigen::Matrix<float, 10, Eigen::Dynamic>& delta,
    const Weights& weights,
    const ResultBatch& forward
) {
    int batch_size = delta.cols();
    Weights gradient{};

    auto d_z2 = delta.cwiseProduct(forward.y_hat.unaryExpr([](float x) { return x * (1 - x); }));
    gradient.b2 = d_z2.rowwise().sum() / batch_size;
    gradient.w2 = d_z2 * forward.a1.transpose() / batch_size;
    // Keep batch dimension; only average when forming parameter gradients.
    auto d_a1 = weights.w2.transpose() * d_z2;
    auto d_z1 = d_a1.cwiseProduct(forward.a1.unaryExpr([](float x) { return x * (1 - x); }));
    gradient.b1 = d_z1.rowwise().sum() / batch_size;
    gradient.w1 = d_z1 * forward.input.transpose() / batch_size;

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

Input inputFromRaw(const uint8_t* data) {
    return (Eigen::Map<Eigen::Vector<uint8_t, 28 * 28>>(
                const_cast<uint8_t*>(data), 28 * 28
           )
                .cast<float>()) /
           255.f;
}

void test(const Weights& weights, const Mnist& data) {
    int errors = 0;
    for (size_t i = 0; i < TEST_DATA_SIZE; i++) {
        auto& row = data.test_data[i];
        auto forward_result = forward(inputFromRaw(row.data), weights);
        Eigen::Index pick = 0;
        forward_result.y_hat.maxCoeff(&pick);
        if (pick != row.label) {
            // std::println("Not match on test {}, expect {}, get {}.", i, row.label, pick);
            errors++;
        }
    }
    std::println(
        "Test done. {} tested. Errors: {}. Error rate: {}",
        TEST_DATA_SIZE,
        errors,
        static_cast<float>(errors) / TEST_DATA_SIZE
    );
}

Weights train(const Mnist& data) {
    Weights weights = randomWeights();
    InputBatch batch_buf;
    Eigen::Matrix<float, 10, Eigen::Dynamic> expected_buf;

    std::vector<int> indices =
        std::ranges::views::iota(0, TRAIN_DATA_SIZE) |
        std::ranges::to<std::vector<int>>();
    std::random_device rd;
    std::mt19937 g(rd());

    // Shuffle the indices randomly

    // training loop
    for (int epoch = 0; epoch < EPOCHS; epoch++) {
        std::ranges::shuffle(indices, g);
        float total_loss = 0.f;
        for (int i = 0; i < TRAIN_DATA_SIZE; i += BATCH_SIZE) {
            int end = std::min<int>(i + BATCH_SIZE, TRAIN_DATA_SIZE);
            int batch_size = end - i;
            // one batch
            batch_buf.resize(Eigen::NoChange, batch_size);
            expected_buf.resize(Eigen::NoChange, batch_size);
            expected_buf.setZero();
            for (int j = i, k = 0; j < end; j++, k++) {
                auto& row = data.train_data[indices[j]];
                Input input = inputFromRaw(row.data);
                batch_buf.col(k) = input;
                expected_buf(row.label, k) = 1.f;
            }
            auto forward_result = forward_batch(batch_buf, weights);
            Eigen::Matrix<float, 10, Eigen::Dynamic> d = forward_result.y_hat - expected_buf;
            float l = d.squaredNorm() / 2.f;
            Weights avg_gradient = backward_batch(d, weights, forward_result);
            avg_gradient *= RATE;
            weights -= avg_gradient;
            total_loss += l;
        }
        std::println("Epoch: {}, Avg loss: {}", epoch + 1, total_loss / TRAIN_DATA_SIZE);
        // test(weights, data);
    }
    return weights;
}

int main() {
    read_data();
    auto data = read_data();
    auto weights = train(*data);
    FILE* fOut = fopen("weights.bin", "wb");
    weights.serialize(fOut);
    test(weights, *data);
    fclose(fOut);
    return 0;
}
