# WebNN Delegate for TensorFlow Lite

| Status        | (Proposed)       |
:-------------- |:---------------------------------------------------- |
| **RFC #**     | [NNN](https://github.com/tensorflow/community/pull/NNN) (update when you have community PR #)|
| **Author(s)** | Ningxin Hu (ningxin.hu@intel.com)                    |
| **Sponsor**   | Ping Yu (piyu@google.com)                            |
| **Updated**   | 2021-09-10                                           |

## Objective

Implement a new [TensorFlow Lite Delegate](https://www.tensorflow.org/lite/performance/delegates) based on [Web Neural Network API](https://www.w3.org/TR/webnn/) (WebNN), a W3C specification for constructing and executing computational graphs of neural networks. This change would enable hardware acceleration of TensorFlow Lite WebAssembly runtime by leveraging on-device accelerators such as the GPU and Digital Signal Processor (DSP) when running in Web browsers and JavaScript runtime (e.g. Node.js).

## Motivation

To answer [the key user needs](https://docs.google.com/presentation/d/14hbhzAduLCr_deYi6u6Z0otCSngf0lbvaHL_icfTXmY/edit#slide=id.gd4f136c3ce_0_3) of supporting more models and improving the performance, TensorFlow.js has integrated the TensorFlow Lite WebAssembly runtime and exposed it via [tfjs-tflite API](https://js.tensorflow.org/api_tflite/0.0.1-alpha.4/) and [Task API](https://js.tensorflow.org/api_tasks/0.0.1-alpha.8/).

However, the users of TensorFlow Lite WebAssembly runtime can only access [128-bit SIMD instructions](https://github.com/WebAssembly/simd) and [multi-threading](https://github.com/WebAssembly/threads) of CPU device via [XNNPACK delegate](https://github.com/huningxin/tensorflow/tree/webnn_delegate/tensorflow/lite/delegates/xnnpack). As a comparison, the users of TensorFlow Lite native runtime could access GPU acceleration via [GPU delegate](https://www.tensorflow.org/lite/performance/gpu) and other hardware accelerators via [NNAPI delegate](https://www.tensorflow.org/lite/performance/nnapi). The lack of access to platform capabilities beneficial for ML such as dedicated ML hardware accelerators constraints the scope of experiences and leads to inefficient implementations on modern hardware. This disadvantages the users of TensorFlow Lite WebAssembly runtime in comparison to the users of its native runtime.

According to the [data](https://www.w3.org/2020/06/machine-learning-workshop/talks/access_purpose_built_ml_hardware_with_web_neural_network_api.html#slide-4), when testing MobileNetV2 on a mainstream laptop, the native inference could be 9.7x faster than WebAssembly SIMD. If enabling WebAssembly multi-threading, the native inference is still about 4x faster. That's because the native inference could leverage the longer SIMD (e.g. [AVX 256](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)) and optimized memory layout (e.g. [blocked memory layout](https://oneapi-src.github.io/oneDNN/dev_guide_understanding_memory_formats.html)) for that specific SIMD instruction set. The data also demostrates when native inference accesses the ML specialized hardware, such as [VNNI](https://en.wikichip.org/wiki/x86/avx512_vnni) instruction on laptop or DSP on smartphone, for the 8-bit quantized model inference, the performance gap would be even over 10x. This capability would not only speed up the inference performance but also help save the power consumption. For example, as the [slide](https://www.w3.org/2020/06/machine-learning-workshop/talks/accelerate_ml_inference_on_mobile_devices_with_android_nnapi.html#slide-8) illustrates, by leveraging DSP through NNAPI, the power reduction could be 3.7x.

The WebNN API is being standardized by W3C Web Machine Learning [Working Group](https://www.w3.org/groups/wg/webmachinelearning) (WebML WG) after two years incubation within W3C Web Machine Learning [Community Group](https://www.w3.org/groups/cg/webmachinelearning). In June 2021, W3C WebML WG published WebNN [First Public Working Draft](https://www.w3.org/2020/Process-20200915/#fpwd) and plans to release the [Candidate Recommendation](https://www.w3.org/2020/Process-20200915/#RecsCR) in Q2 2022. WebNN may be implemented in Web browsers by using the available native operating system machine learning APIs, such as Android/ChromeOS [Neural Networks API](https://developer.android.com/ndk/guides/neuralnetworks), Windows [DirectML API](https://docs.microsoft.com/en-us/windows/ai/directml/dml-intro) and macOS/iOS [ML Compute API](https://developer.apple.com/documentation/mlcompute/). This architecture allows JavaScript ML frameworks to tap into cutting-edge machine learning innovations in the operating system and the hardware platform underneath it without being tied to platform-specific capabilities, bridging the gap between software and hardware through a hardware-agnostic abstraction layer.

As a related work, [OpenCV.js](https://docs.opencv.org/3.4/d5/d10/tutorial_js_root.html) (the OpenCV WebAssembly runtime) has started implementing WebNN backend of dnn (deep neural networks) module. The [early result](https://github.com/opencv/opencv/pull/20406) showed 7x speedup for SqueezeNet compared to its optimized WebAssembly backend when running in Node.js/Electron.js runtime.

## User Benefit

A headline might be: "Accelerating the TensorFlow Lite WebAssembly runtime with WebNN API". 

1. Users will be able to enable hardware acceleration of TensorFlow Lite WebAssembly runtime by leveraging on-device accelerators, such as the GPU and DSP, across operating systems and devices. That will bring close-to-native performance and power reduction.
2. Users will be able to run 8-bit quantized model by TensorFlow Lite WebAssembly runtime with specialized hardware features, such as CPU VNNI instruction and Edge TPU.
3. Other contributors will be able to maintain one WebNN delegate implementation for various type of accelerators for TensorLow Lite WebAssembly runtime.

## Design Proposal

The major part of the implementation is in `webnn_delegate.cc` and the interfaces are declared in `webnn_delegate.h`. The two files are placed in the folder with path `tensorflow/lite/delegates/webnn`. 

### Interfaces

The `webnn_delegate.h` exposes `TfLiteWebNNDelegateOptions`, `TfLiteWebNNDelegateCreate` and `TfLiteWebNNDelegateDelete` interfaces.

`TfLiteWebNNDelegateOptions` is a structure that is used to supply options when creating a WebNN delegate. The options map to [`MLContextOptions`](https://www.w3.org/TR/webnn/#dictdef-mlcontextoptions) for device and power preferences.

It may be implemented as:
```c++
typedef struct {
  // enum class DevicePreference : uint32_t {
  //     Default = 0x00000000,
  //     Gpu = 0x00000001,
  //     Cpu = 0x00000002,
  // };
  uint32_t devicePreference;
  // enum class PowerPreference : uint32_t {
  //     Default = 0x00000000,
  //     High_performance = 0x00000001,
  //     Low_power = 0x00000002,
  // };
  uint32_t powerPreference;
} TfLiteWebNNDelegateOptions;
```

`TfLiteWebNNDelegateOptionsDefault()` is used to create a structure with the default WebNN delegate options.
```c++
TfLiteWebNNDelegateOptions TfLiteWebNNDelegateOptionsDefault();
```

`TfLiteWebNNDelegateCreate()` is the main entry point to create a new instance of WebNN delegate. It takes `TfLiteWebNNDelegateOptions` and returns a pointer to  `TfLiteDelegate` that is defined in [`tensorflow/lite/c/common.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h). `TfLiteDelegate` is a structure that is the main interface that TensorFlow Lite runtime interacts with WebNN delegate implementation. When `options` is set to `nullptr`, the above default values are used.
```c++
TfLiteDelegate* TfLiteWebNNDelegateCreate(const TfLiteWebNNDelegateOptions* options);
```

`TfLiteWebNNDelegateDelete()` destroys a delegate created with `TfLiteWebNNDelegateCreate()` call.
```c++
void TfLiteWebNNDelegateDelete(TfLiteDelegate* delegate);
```

### Implementation

`webnn_delegate.cc` implements `TfLiteDelegate` and `TfLiteRegistration` defined in [`tensorflow/lite/c/common.h`](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/lite/c/common.h) with `Delegate` and `Subgraph` classes. The implementations are in namespace `delegate::webnn::`.

#### `Delegate` class

`Delegate` class implements the `TfLiteDelegate`. The following sample code list major methods and members that may be implemented for this class.

```c++
class Delegate {
 public:
  explicit Delegate(const TfLiteWebNNDelegateOptions* options);
  TfLiteIntArray* PrepareOpsToDelegate(TfLiteContext* context);

  // Other methods

 private:
  TfLiteDelegate delegate_ = {
      reinterpret_cast<void*>(this),  // .data_
      DelegatePrepare,                // .Prepare
      nullptr,                        // .CopyFromBufferHandle
      nullptr,                        // .CopyToBufferHandle
      nullptr,                        // .FreeBufferHandle
      kTfLiteDelegateFlagsNone,       // .flags
  };

  // WebNN MLContextOptions
  MLContextOptions context_options_;

  // Other members
};
```

The constructor of `Delegate` translates the `TfLiteWebNNDelegateOptions` to `MLContextOptions`.

The `delegate_` is an instance of `TfLiteDelegate`. It bridges `TfLiteDelegate` structure to `Delegate` class.

The `TfLiteDelegate::data_` is used to identify `Delegate` itself. 

The `TfLiteDelegate::Prepare` function pointer is set to `DelegatePrepare()`. This function is invoked by `ModifyGraphWithDelegate()`. This prepare is called, giving the delegate a view of the current graph through `TfLiteContext`. It looks at the nodes by `Delegate::PrepareOpsToDelegate()` and call `ReplaceNodeSubsetsWithDelegateKernels()` to ask the TensorFlow Lite runtime to create macro-nodes to represent delegated subgraphs (implemented by `Subgraph`) of the original graph.

```c++
TfLiteStatus DelegatePrepare(TfLiteContext* context, TfLiteDelegate* delegate) {
  TfLiteIntArray* ops_to_replace =
      static_cast<::tflite::webnn::Delegate*>(delegate->data_)
          ->PrepareOpsToDelegate(context);
  if (ops_to_replace == nullptr) {
    return kTfLiteError;
  }

  const TfLiteStatus status = context->ReplaceNodeSubsetsWithDelegateKernels(
      context, kSubgraphRegistration, ops_to_replace, delegate);
  TfLiteIntArrayFree(ops_to_replace);
  return status;
}
```

`Delegate::PrepareOpsToDelegate()` takes a `TfLiteContext` and returns a list of nodes in `TfLiteIntArray` that can be delegated by WebNN delegate. It basically implements the following steps:
 1. Create `nodes_to_delegate` as an instance of `TfLiteIntArray` with length of 0.
 1. Get the execution plan from the context by `TfLiteContext::GetExecutionPlan`.
 1. Iterate each node of the execution plan and get node info of `TfLiteNode` and registration of `TfLiteRegistration`. For each node, execute following steps:
    1. Call `Subgraph::VisitNode` with the node info and its registration. For each type of the node (`builtin_code`), it checks the data type, shape and attributes are supported by WebNN or not. Please see details of `Subgraph::VisitNode` in following section.
    1. If the node is supported by WebNN, append it into the `nodes_to_delegate` and continue the iteration.
 1. When all nodes are iterated, return the `nodes_to_delegates`.

Since WebNN delegate doesn't allocate its own buffers, `TfLiteDelegate::CopyFromBufferHandle`, `TfLiteDelegate::CopyToBufferHandle` and `TfLiteDelegate::FreeBufferHandle` are all set to `nullptr`.

`TfLiteDelegate::flags` is set to `kTfLiteDelegateFlagsNone` for basic functionality. WebNN delegate may support other flags, such as for dynamic sized tensors. 

#### `Subgraph` class

`kSubgraphRegistration` is an instance of `TfLiteRegistration`. It bridges `TfLiteRegistration` structure to `Subgraph` class.

```c++
const TfLiteRegistration kSubgraphRegistration = {
    /*.init=*/SubgraphInit,
    /*.free=*/SubgraphFree,
    /*.prepare=*/SubgraphPrepare,
    /*.invoke=*/SubgraphInvoke,
    /*.profiling_string=*/nullptr,
    /*.builtin_code=*/0,
    /*.custom_name=*/"TfLiteWebNNDelegate",
    /*.version=*/2,
};
```

Respectively, `SubgraphInit` calls `Subgraph::Create()`, `SubgraphFree` deletes `Subgraph` instance, `SubgraphPrepare` calls `Subgraph::Prepare()` and `SubgraphInvoke` calls `Subgraph::Invoke()`.

`Subgraph` implements `TfLiteRegistration` by WebNN interfaces, such as [`MLGraph`](https://www.w3.org/TR/webnn/#api-mlgraph). The following sample code list major methods and members that may be implemented for this class.

```c++
class Subgraph {
 public:
  static Subgraph* Create(TfLiteContext* context,
                          const TfLiteDelegateParams* params,
                          const Delegate* delegate);
  TfLiteStatus Prepare(TfLiteContext* context);
  TfLiteStatus Invoke(TfLiteContext* context);
  static TfLiteStatus VisitNode(
      const ml::GraphBuilder& builder, TfLiteContext* context,
      TfLiteRegistration* registration, TfLiteNode* node, int node_index,
      const std::unordered_set<int>& quasi_static_tensors,
      std::vector<ml::Operand>& webnn_operands,
      std::vector<std::unique_ptr<char>>& constant_buffers);
 private:
  Subgraph(ml::Graph graph, std::unordered_set<int>&& inputs, std::unordered_set<int>&& outputs);
  ml::Graph ml_graph_;
  std::unordered_set<int> inputs_;
  std::unordered_set<int> outputs_;
  std::unordered_map<int, ml::Input> ml_inputs_;
  std::unordered_map<int, ml::ArrayBufferView> ml_outputs_;
```

`Create()` takes `TfLiteContext`, `TfLiteDelegateParams`, `Delegate` and returns a `Subgraph` pointer points to the new instance of `Subgraph`. The returned pointer will be stored with the node in the `user_data` field, accessible within prepare and invoke functions. It basically executes the following steps:
 1. Create WebNN context of [`MLContext`](https://www.w3.org/TR/webnn/#api-mlcontext) and graph builder of [`MLGraphBuilder`](https://www.w3.org/TR/webnn/#api-mlgraphbuilder).
 1. Create a vector of WebNN operands `webnn_operands` of type [`MLOperand`](https://www.w3.org/TR/webnn/#api-mloperand) indexed by TFLite tensor ID.
 1. Get the input tensors by accessing `TfLiteDelegateParams::input_tensors`. For each input tensor, execute the following steps:
    1. Check whether the data is allocated for this input tensor.
    1. If the data is allocated, create a WebNN constant operand by [`builder.constant`](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-constant) according to its data type and shape. Insert this operand into `webnn_operands` indexed by TFLite tensor ID.
    1. Otherwise, create a WebNN input operand by [`builder.input`](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-input) with the name of the tensor ID and [`MLOperandDescriptor`](https://www.w3.org/TR/webnn/#api-mloperanddescriptor) according to its data type and shape. Insert this operand into `webnn_operands` indexed by TFLite tensor ID.
 1. Get the nodes to be replaced by accessing `TfLiteDelegateParams::nodes_to_replace`. For each node, execute the following steps:
    1. Retrieve its node info of `TfLiteNode` and registration of `TfLiteRegistration`.
    1. Call `Subgraph::VisitNode` with the TFLite context, node info and registration together with WebNN builder and operands. It builds the actual WebNN operation according to the TFLite node. It puts the WebNN output operand of that operation into `webnn_operands`. Please see details of `Subgraph::VisitNode` in following section.
 1. Get the TFLite output tensors by accessing `TfLiteDelegateParams::output_tensors`. Build [`MLNamedOperands`](https://www.w3.org/TR/webnn/#typedefdef-mlnamedoperands) by iterating the TFLite output tensors and executing following steps for each output tensor:
    1. Find the WebNN operand in `webnn_operands` by the output tensor ID.
    1. Insert the WebNN operand into `MLNamedOperands` and associate it with the output tensor ID as the name.
 1. Build WebNN [`MLGraph`](https://www.w3.org/TR/webnn/#api-mlgraph) by calling [`builder.build`](https://www.w3.org/TR/webnn/#dom-mlgraphbuilder-build) with the `MLNamedOperands`.

 As the WebNN `MLGraph` is built by `Create()`, `Prepare()` is implemented as a null operation.

`Invoke()` computes the delegated subgraph of `MLGraph` with inputs and outputs of the `TfLiteContext`. It basically executes the following steps:
 1. Create [`MLNamedInputs`](https://www.w3.org/TR/webnn/#typedefdef-mlnamedinputs) and bind TFLite input tensors' buffer. The inputs are indexed by tensor ID (`int`).
 2. Create [`MLNamedOutputs`](https://www.w3.org/TR/webnn/#typedefdef-mlnamedoutputs) and bind TFLite output tensors' buffer. The outputs are indexed by tensor ID (`int`).
 3. Call [`MLGraph.compute()`](https://www.w3.org/TR/webnn/#dom-mlgraph-compute) with inputs and outputs. After this call completes, the results are placed into the TFLite output tensors' buffer.

`VisitNode()` builds a WebNN operation with `ml::GraphBuilder` based on TFLite node represented by `TfLiteRegistration` and `TfLiteNode` indexed by `node_index`. It basically executes the following steps:
 1. Check the TFLite node's `TfLiteRegistration::builtin_code`.
 2. Based on the `builtin_code`, such as `kTfLiteBuiltinAdd`, call corresponding WebNN operation building method, such as `VisitAddNode`.

The following code illustrates the kernel of TensorFlow Lite `kTfLiteBuiltinAdd` may be implemented with WebNN API:

```c++
static TfLiteStatus VisitAddNode(
    const ml::GraphBuilder& builder, TfLiteContext* logging_context, int node_index,
    TfLiteNode* node, const TfLiteTensor* tensors,
    const TfLiteAddParams* add_params,
    std::vector<ml::Operand>& webnn_operands,
    std::vector<std::unique_ptr<char>>& constant_buffers) {
  TF_LITE_ENSURE_STATUS(
      CheckNumInputsAndOutputs(logging_context, node, 2, 1, node_index));

  const int input1_tensor_id = node->inputs->data[0];
  const TfLiteTensor& input1_tensor = tensors[input1_tensor_id];
  TF_LITE_ENSURE_STATUS(CheckTensorFloat32Type(
      logging_context, input1_tensor, input1_tensor_id, node_index));

  const int input2_tensor_id = node->inputs->data[1];
  const TfLiteTensor& input2_tensor = tensors[input2_tensor_id];
  TF_LITE_ENSURE_STATUS(CheckTensorFloat32(
      logging_context, input2_tensor, input2_tensor_id, node_index));

  const int output_tensor_id = node->outputs->data[0];
  const TfLiteTensor& output_tensor = tensors[output_tensor_id];
  TF_LITE_ENSURE_STATUS(CheckTensorFloat32(
      logging_context, output_tensor, output_tensor_id, node_index));

  if (builder) {
    webnn_operands[output_tensor_id] =
        builder.Add(webnn_operands[input1_tensor_id], webnn_operands[input2_tensor_id]);
  }

  if (add_params != nullptr) {
    TF_LITE_ENSURE_STATUS(VisitActivation(
        builder, logging_context, node_index, output_tensor_id, output_tensor_id,
        add_params->activation, webnn_operands, constant_buffers));
  }

  return kTfLiteOk;
}
```

`VisitNode` is called by both `Delegate::PrepareOpsToDelegate` and `Subgraph::Create` for two different usages. When called by `Delegate::PrepareOpsToDelegate`, the `builder` will be an invalid `ml::GraphBuilder`. It leads `if (builder)` to be false. At that stage, it only checks whether this TFLite node is supported by WebNN or not. When called by `Subgraph::Create`, the `builder` will be a valid `ml::GraphBuilder`. It makes `if (builder)` to be true. At this stage, it builds the actual WebNN operation based on the TFLite node.

### Alternatives Considered

To access GPU, one alternative solution would be implementing a WebGL/WebGPU delegate. TensorFlow.js already uses WebGL and are working on a WebGPU backend. We believe this alternative is insufficient for two reasons. First, although graphics abstraction layers provide the flexibility of general programmability of the GPU graphics pipelines, they are unable to tap into hardware-specific optimizations and special instructions that are available to the operating system internals. The hardware ecosystem has been investing significantly in innovating in the ML space, and much of that is about improving the performance of intensive compute workloads in machine learning scenarios. Some key technologies that are important to model performance may not be uniformly accessible to applications through generic graphics pipeline states. Secondly, there are other accelerators, such as DPS and Edge TPU, that are not exposed through WebGL and WebGPU API. A WebGL/WebGPU delegate would not be able to enable the hardware accelerations on those accelerators.

Actually, the WebNN delegate is also able to run on top of [WebNN-polyfill](https://github.com/webmachinelearning/webnn-polyfill), a JavaScript implementation of WebNN API based on TensorFlow.js kernels. Through this path, TensorFlow Lite WebAssembly runtime could leverage the WebGL/WebGPU implementation of TensorFlow.js ops instead of implementing a WebGL/WebGPU delegate.

### Performance Implications

There will be no performance implications for users of TensorFlow Lite WebAssembly runtime if they won't enable WebNN delegate, because the model will still be executed by XNNPACK delegate and built-in operation kernels. If users enable WebNN delegate, TensorFlow Lite runtime would partition the graph based on the operations supported by WebNN delegate. Each partition that is handled by WebNN delegate will be replaced by a WebNN delegate node in the original graph that evaluates the partition on its invoke call. Depending on the model, the final graph can end up with one or more nodes, the latter meaning that some ops are not supported by WebNN delegate. If the whole model could be handled by WebNN delegate, the performance characteristics would depend on the native operating system API and the type of hardware device that are used by the WebNN implementation of the Web browsers or JavaScript runtime. If there are multiple nodes in the final graph, there is an overhead for passing the results from the delegated subgraph to the main graph that results due to memory copies (for example, GPU to CPU) and layout conversions (e.g. plain layout to blocked layout). Such overhead might offset performance gains especially when there are a large amount of memory copies. WebNN API allows to specify the [device preference](https://www.w3.org/TR/webnn/#enumdef-mldevicepreference) when creating the `MLContext` for neural network graph compilation and compute. Users could keep using the same device (e.g. CPU) of WebNN delegate with other executors (e.g. XNNPACK delegate) that avoids unnecessary memory copies across devices.

The implementation of WebNN delegate will follow the [TensorFlow Lite delegate development guide](https://www.tensorflow.org/lite/performance/implementing_delegate). So the TensorFlow Lite [Model Benchmark Tool](https://github.com/tensorflow/tensorflow/tree/f9ef3a8a0b64ad6393785f3259e9a24af09c84ad/tensorflow/lite/tools/benchmark), [Inference Diff tool](https://github.com/tensorflow/tensorflow/tree/f9ef3a8a0b64ad6393785f3259e9a24af09c84ad/tensorflow/lite/tools/evaluation/tasks/inference_diff) and [task specific evaluation tools](https://www.tensorflow.org/lite/performance/delegates#task-based_evaluation) could be used to test and benchmark. The [TensorFlow.js Model Benchmark](https://github.com/tensorflow/tfjs/tree/master/e2e/benchmarks/local-benchmark) could be used to benchmark WebNN delegate once it supports TensorFlow Lite WebAssembly runtime.

### Dependencies

The TensorFlow Lite WebNN delegate takes a dependency on the WebNN API and implementations. The WebNN API is being standardized by W3C Web Machine Learning Working Group. The WebNN API implementations include:
* [WebNN-polyfill](https://github.com/webmachinelearning/webnn-polyfill): a JavaScript implementation based on TensorFlow.js kernels. It could run where TensorFlow.js could run and depends on JavaScript, WebAssembly, WebGL and WebGPU that TensorFlow.js uses. This project is maintained by W3C Web Machine Learning Community Group.
* [WebNN-native](https://github.com/webmachinelearning/webnn-native): a C++ implementation based on native ML API. It provides WebNN C/C++ headers (`webnn.h` and `webnn_cpp.h`) for C++ code to use. TensorFlow Lite WebNN delegate implementation will use these interfaces. The backend implementation of WebNN-native uses native ML API, including [DirectML](https://docs.microsoft.com/en-us/windows/ai/directml/dml-intro) on Windows and [OpenVINO](https://docs.openvinotoolkit.org/latest/index.html) on Linux/Windows. More backend implementations, such as [NNAPI](https://developer.android.com/ndk/guides/neuralnetworks), are to be added. This project is maintained by W3C Web Machine Learning Community Group.
* [WebNN-native binding for Node.js](https://github.com/webmachinelearning/webnn-native/tree/main/node): a Node.js [C++ addon](https://nodejs.org/api/addons.html) that is based on WebNN-native and exposes JavaScript API. It allows JavaScript/WebAssembly apps and frameworks, such as TensorFlow.js and TensorFlow Lite WebAssembly runtime, to access WebNN native implementation when running in Node.js and Electron.js. This project is maintained by W3C Web Machine Learning Community Group.
* [Emscripten](https://emscripten.org/): the compiler that compiles C++ code to WebAssembly. The implementation of WebNN support will be submitted and maintained by authors of this RFC.
* WebNN implementation in Web browsers, such as Chrome browser of ChromeOS. The implementation would be maintained by individual browser vendor.

### Engineering Impact

The WebNN delegate build will be produced with a newly added Bazel BUILD rule (e.g. cc_library with name `webnn_delegate`). The build will compile WebNN delegate sources (e.g. `webnn_delegate.h` and `webnn_delegate.cc`). According to the current implementation of 10 ops, the compilation time is about a few seconds. And the size of WebAssembly binary is increased by about 20KB. With more ops are implemented, the binary size is expected to be increased by another dozen of KB.

As a proposal, the code would be maintained by TensorFlow.js Special Interest Group (SIG).

### Platforms and Environments

The WebNN delegate works on all platforms supported by TensorFlow Lite WebAssembly runtime that include:
* Web browsers without WebNN API implemented: it is supported by WebNN-polyfill
* Web browsers with WebNN API implemented: it is supported by the native WebNN implementation of the browser
* Node.js/Electron.js: it is supported by the WebNN-native binding for Node.js

### Best Practices

This change will follow the [TensorFlow Lite delegate development guide](https://www.tensorflow.org/lite/performance/implementing_delegate) and won't change any best practices. 

### Tutorials and Examples

The C++ example that enables WebNN delegate.
```c++
// Available options.
struct TFLiteWebModelRunnerOptions {
  // Set the number of threads available to the interpreter.
  // -1 means to let interpreter set the threads count available to itself.
  int num_threads = kDefaultNumThreads;

  // Enable WebNN delegate or not.
  bool enable_webnn_delegate = false;

  // Device preference of WebNN delegate: 0 - default, 1 - gpu, 2 - cpu.
  int webnn_device_preference = 0;
};

TfLiteStatus TFLiteWebModelRunner::InitFromBuffer(
    const char* model_buffer_data, size_t model_buffer_size,
    std::unique_ptr<tflite::OpResolver> resolver) {
  // Initilaize the model from flatbuffer.
  const char* model_buffer = reinterpret_cast<const char*>(model_buffer_data);
  flatbuffers::Verifier verifier(reinterpret_cast<const uint8_t*>(model_buffer),
                                 model_buffer_size);
  if (!tflite::VerifyModelBuffer(verifier)) {
    return kTfLiteError;
  }
  model_ =
      tflite::FlatBufferModel::BuildFromBuffer(model_buffer, model_buffer_size);

  // Initialize the interpreter from the model.
  const auto interpreter_builder_result =
      tflite::InterpreterBuilder(model_->GetModel(), *resolver, nullptr)(
          &interpreter_, options_.num_threads);
  if (interpreter_builder_result != kTfLiteOk) {
    return interpreter_builder_result;
  }
  if (!model_->initialized()) {
    return kTfLiteError;
  }

  // Enable WebNN delegate if requested.
  if (options_.enable_webnn_delegate) {
    TfLiteWebNNDelegateOptions options =
        TfLiteWebNNDelegateOptionsDefault();
    options.devicePreference = options_.webnn_device_preference;
    auto webnn_delegate = TfLiteWebNNDelegateCreate(&options);
    auto delegate_ptr = tflite::Interpreter::TfLiteDelegatePtr(webnn_delegate, [](TfLiteDelegate* delegate) {
      TfLiteWebNNDelegateDelete(delegate);
    });
    if (interpreter_->ModifyGraphWithDelegate(std::move(delegate_ptr)) != kTfLiteOk) {
        printf("Failed to apply webnn delegate.\n");
    }
  }

  // Allocate memory for the tensors in the model.
  return interpreter_->AllocateTensors();
}
```

The JavaScript example that use WebNN delegate.
```javascript
// Create the model runner with the model.

// Load WASM module and model.
const [module, modelArrayBuffer] = await Promise.all([
    tflite_model_runner_ModuleFactory(),
    (await fetch(MODEL_PATH)).arrayBuffer(),
]);
const modelBytes = new Uint8Array(modelArrayBuffer);
const offset = module._malloc(modelBytes.length);
module.HEAPU8.set(modelBytes, offset);

// Create model runner with WebNN delegate enabled.
const modelRunnerResult =
    module.TFLiteWebModelRunner.CreateFromBufferAndOptions(
        offset, modelBytes.length, {
        numThreads: Math.min(
            4, Math.max(1, (navigator.hardwareConcurrency || 1) / 2)),
        enableWebNNDelegate: true,
        webNNDevicePreference: 0)
    });
if (!modelRunnerResult.ok()) {
    throw new Error(
        'Failed to create TFLiteWebModelRunner: ' + modelRunner.errorMessage());
}


const modelRunner = modelRunnerResult.value();

// Set input and invoke modelRunner.Infer() with WebNN delegate.

```

### Compatibility

The change in this proposal concerns the low-level constructs inside the TensorFlow Lite WebAssembly runtime with minimal to no impact to the high-level exposures and API. The existing models supported by TensorFlow Lite WebAssembly runtime will be supported with WebNN delegate enabled.

### User Impact

This feature will be rolled out with the tfjs-tflite. There are two ways:

Via NPM
```js
// Import @tensorflow/tfjs-tflite.
import * as tflite from '@tensorflow/tfjs-tflite';
```

Via a script tag
```js
<script src="https://cdn.jsdelivr.net/npm/@tensorflow/tfjs-tflite/dist/tf-tflite.min.js"></script>
```

## Questions and Discussion Topics

* What's the path to support GPU buffers by TensorFlow Lite WebAssembly runtime? The use case is to interact with WebGL/WebGPU based pre and post-processing code. WebNN API supports taking WebGL and WebGPU textures/buffers as inputs and outputs.
* What's the path to integrate this change into MediaPipe Web solution.
