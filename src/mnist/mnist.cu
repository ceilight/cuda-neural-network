#include <mnist.cuh>

#include <ctime>

#include <thrust/host_vector.h>

Minist::Minist(std::string minst_data_path, float learning_rate, float l2,
               float beta) {
  // init
  dataset.reset(new DataSet(minst_data_path, true));

  conv1.reset(new Conv(28, 28, 1, 32, 5, 5, 0, 0, 1, 1, true));
  conv1_relu.reset(new ReLU(true));
  max_pool1.reset(new MaxPool(2, 2, 0, 0, 2, 2));

  conv2.reset(new Conv(12, 12, 32, 64, 5, 5, 0, 0, 1, 1, true));
  conv2_relu.reset(new ReLU(true));
  max_pool2.reset(new MaxPool(2, 2, 0, 0, 2, 2));

  conv3.reset(new Conv(4, 4, 64, 128, 3, 3, 0, 0, 1, 1, true));
  conv3_relu.reset(new ReLU(true));

  flatten.reset(new Flatten(true));
  fc1.reset(new Linear(128 * 2 * 2, 128, true));
  fc1_relu.reset(new ReLU(true));

  fc2.reset(new Linear(128, 10, true));
  fc2_relu.reset(new ReLU(true));

  log_softmax.reset(new LogSoftmax(1));
  nll_loss.reset(new NLLLoss());

  // connect
  dataset->connect(*conv1)
      .connect(*conv1_relu)
      .connect(*max_pool1)
      .connect(*conv2)
      .connect(*conv2_relu)
      .connect(*max_pool2)
      .connect(*conv3)
      .connect(*conv3_relu)
      .connect(*flatten)
      .connect(*fc1)
      .connect(*fc1_relu)
      .connect(*fc2)
      .connect(*fc2_relu)
      .connect(*log_softmax)
      .connect(*nll_loss);

  // regist parameters
  rmsprop.reset(new RMSProp(learning_rate, l2, beta));
  rmsprop->regist(conv1->parameters());
  rmsprop->regist(conv2->parameters());
  rmsprop->regist(conv3->parameters());
  rmsprop->regist(fc1->parameters());
  rmsprop->regist(fc2->parameters());
}

void Minist::train(int epochs, int batch_size) {
#ifdef STATS
  load_time = 0;
  verify_time = 0;
  optim_time = 0;
  forward_time.resize(14, 0);
  backward_time.resize(14, 0);
#endif

  for (int epoch = 0; epoch < epochs; epoch++) {
    int idx = 1;

    while (dataset->has_next(true)) {
      forward(batch_size, true);
      backward();
      rmsprop->step();

      if (idx % 10 == 0) {
        float loss = this->nll_loss->get_output()->get_data()[0];
        auto acc = top1_accuracy(this->log_softmax->get_output()->get_data(),
                                 10, this->dataset->get_label()->get_data());

        std::cout << "Epoch: " << epoch << ", Batch: " << idx
                  << ", NLLLoss: " << loss
                  << ", Train Accuracy: " << (float(acc.first) / acc.second)
                  << std::endl;
      }
      ++idx;
    }

    test(batch_size);
    dataset->reset();
  }

#ifdef STATS
  std::cout << std::endl;
  std::cout << "-------------------------------------------" << std::endl;
  std::cout << "Load: " << load_time / idx << std::endl;
  for (int i = 0; i < (int)forward_time.size(); i++)
      std::cout << "Forward #" << i << ": " << forward_time[i] / idx
                << std::endl;
  std::cout << "Verify: " << verify_time / idx << std::endl;
  for (int i = 0; i < (int)backward_time.size(); i++)
      std::cout << "Backward #" << i << ": " << backward_time[i] / idx
                << std::endl;
  std::cout << "Optimize: " << optim_time << std::endl;
  std::cout << "-------------------------------------------" << std::endl;
#endif
}

void Minist::test(int batch_size) {
  int idx = 1;
  int count = 0;
  int total = 0;

  while (dataset->has_next(false)) {
    forward(batch_size, false);
    auto acc = top1_accuracy(this->log_softmax->get_output()->get_data(), 10,
                             this->dataset->get_label()->get_data());
    // if (idx % 10 == 0) {
    //   std::cout << "Batch: " << idx
    //             << ", Test Accuracy: " << (float(acc.first) / acc.second)
    //             << std::endl;
    // }

    count += acc.first;
    total += acc.second;
    ++idx;
  }

  std::cout << "Total Accuracy: " << (float(count) / total) << std::endl;
}

void Minist::forward(int batch_size, bool is_train) {
#ifdef STATS
  std::clock_t start;

  start = std::clock();
#endif
  dataset->forward(batch_size, is_train);
#ifdef STATS
  load_time += (std::clock() - start) / (float)CLOCKS_PER_SEC;
#endif
  const Storage* labels = dataset->get_label();

#ifdef STATS
  int p = 0;
  start = std::clock();
  conv1->forward();
  forward_time[p++] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  conv1_relu->forward();
  forward_time[p++] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  max_pool1->forward();
  forward_time[p++] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  conv2->forward();
  forward_time[p++] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  conv2_relu->forward();
  forward_time[p++] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  max_pool2->forward();
  forward_time[p++] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  conv3->forward();
  forward_time[p++] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  conv3_relu->forward();
  forward_time[p++] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  flatten->forward();
  forward_time[p++] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  fc1->forward();
  forward_time[p++] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  fc1_relu->forward();
  forward_time[p++] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  fc2->forward();
  forward_time[p++] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  fc2_relu->forward();
  forward_time[p++] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  log_softmax->forward();
  forward_time[p++] += (std::clock() - start) / (float)CLOCKS_PER_SEC;

  if (is_train) {
    start = std::clock();
    nll_loss->forward(labels);
    verify_time += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  }
#else
  conv1->forward();
  conv1_relu->forward();
  max_pool1->forward();

  conv2->forward();
  conv2_relu->forward();
  max_pool2->forward();

  conv3->forward();
  conv3_relu->forward();

  flatten->forward();
  fc1->forward();
  fc1_relu->forward();

  fc2->forward();
  fc2_relu->forward();

  log_softmax->forward();

  if (is_train) nll_loss->forward(labels);
#endif
}

void Minist::backward() {

#ifdef STATS
  std::clock_t start;
  start = std::clock();
  int p = 13;
#endif

#ifdef STATS
  start = std::clock();
  nll_loss->backward();
  verify_time += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  log_softmax->backward();
  backward_time[p--] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  fc2_relu->backward();
  backward_time[p--] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  fc2->backward();
  backward_time[p--] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  fc1_relu->backward();
  backward_time[p--] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  fc1->backward();
  backward_time[p--] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  flatten->backward();
  backward_time[p--] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  conv3_relu->backward();
  backward_time[p--] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  conv3->backward();
  backward_time[p--] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  max_pool2->backward();
  backward_time[p--] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  conv2_relu->backward();
  backward_time[p--] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  conv2->backward();
  backward_time[p--] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  max_pool1->backward();
  backward_time[p--] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  conv1_relu->backward();
  backward_time[p--] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
  start = std::clock();
  conv1->backward();
  backward_time[p--] += (std::clock() - start) / (float)CLOCKS_PER_SEC;
#else
  nll_loss->backward();
  log_softmax->backward();

  fc2_relu->backward();
  fc2->backward();

  fc1_relu->backward();
  fc1->backward();
  flatten->backward();

  conv3_relu->backward();
  conv3->backward();

  max_pool2->backward();
  conv2_relu->backward();
  conv2->backward();

  max_pool1->backward();
  conv1_relu->backward();
  conv1->backward();
#endif
}

std::pair<int, int> Minist::top1_accuracy(
    const thrust::host_vector<float> &probs,
    int cls_size,
    const thrust::host_vector<float> &labels) {
  int count = 0;
  int size = labels.size() / cls_size;
  for (int i = 0; i < size; i++) {
    int max_pos = -1;
    float max_value = -FLT_MAX;
    int max_pos2 = -1;
    float max_value2 = -FLT_MAX;

    for (int j = 0; j < cls_size; j++) {
      int index = i * cls_size + j;
      if (probs[index] > max_value) {
        max_value = probs[index];
        max_pos = j;
      }
      if (labels[index] > max_value2) {
        max_value2 = labels[index];
        max_pos2 = j;
      }
    }
    if (max_pos == max_pos2) ++count;
  }
  return {count, size};
}
