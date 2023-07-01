//
// Created by fss on 22-12-20.
//
#include <glog/logging.h>
#include <gtest/gtest.h>
#include "ops/relu_op.hpp"
#include "layer/relu_layer.hpp"
#include "factory/layer_factory.hpp"

TEST(test_layer, forward_relu1) {
  using namespace kuiper_infer;
  float thresh = 0.f;
  // 初始化一个relu operator 并设置属性
  std::shared_ptr<Operator> relu_op = std::make_shared<ReluOperator>(thresh);

  // 有三个值的一个tensor<float>
  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
  input->index(0) = -1.f; //output对应的应该是0
  input->index(1) = -2.f; //output对应的应该是0
  input->index(2) = 3.f; //output对应的应该是3
  // 主要第一个算子，经典又简单，我们这里开始！

  std::vector<std::shared_ptr<Tensor<float>>> inputs; //作为一个批次去处理

  std::vector<std::shared_ptr<Tensor<float>>> outputs; //放结果
  inputs.push_back(input);
  ReluLayer layer(relu_op);
  // 因为是4.1 所以没有作业 4.2才有
// 一个批次是1
  layer.Forwards(inputs, outputs);
  ASSERT_EQ(outputs.size(), 1);

  for (int i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->index(0), 0.f);
    ASSERT_EQ(outputs.at(i)->index(1), 0.f);
    ASSERT_EQ(outputs.at(i)->index(2), 3.f);
  }
}

// 注册模式的意义：
// 如果有一个算子列表或一个网络
// op list
// conv1
// conv2
// relu 
// sigmoid
// linear
// ...

// 有注册机制的理论调用
/**
 * ops = {conv1, conv2, relu, sigmoid, linear}
 * layers = {}
 * for op in ops:
 *   layers.push_back(LayerRegisterer::CreateLayer(op))
 * 初始化完毕
*/

// 如果没有注册机制
/**
 * 模型多少层，就要执行多少次
 * ConvLayer conv1(conv1_op);
 * ConvLayer conv2(conv2_op);
 * ReluLayer relu(relu_op);
 * SigmoidLayer sigmoid(sigmoid_op);
 * LinearLayer linear(linear_op);
 * 
 * layers.push_back({conv1, conv2, relu, sigmoid, linear})
*/


// 还有个问题是，算子不是要提前定义出来？
// std::shared_ptr<Operator> relu_op = std::make_shared<ReluOperator>(thresh);
// 所以上面的算子列表定义之前是否要先像上面把每个定义后再将op类型（而不是op名字）放入ops中

TEST(test_layer, forward_relu2) {
  using namespace kuiper_infer;
  float thresh = 0.f;
  std::shared_ptr<Operator> relu_op = std::make_shared<ReluOperator>(thresh);
  std::shared_ptr<Layer> relu_layer = LayerRegisterer::CreateLayer(relu_op);

  std::shared_ptr<Tensor<float>> input = std::make_shared<Tensor<float>>(1, 1, 3);
  input->index(0) = -1.f;
  input->index(1) = -2.f;
  input->index(2) = 3.f;
  std::vector<std::shared_ptr<Tensor<float>>> inputs;
  std::vector<std::shared_ptr<Tensor<float>>> outputs;
  inputs.push_back(input);
  relu_layer->Forwards(inputs, outputs);
  ASSERT_EQ(outputs.size(), 1);
  for (int i = 0; i < outputs.size(); ++i) {
    ASSERT_EQ(outputs.at(i)->index(0), 0.f);
    ASSERT_EQ(outputs.at(i)->index(1), 0.f);
    ASSERT_EQ(outputs.at(i)->index(2), 3.f);
  }
}