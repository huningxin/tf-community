# Title of RFC

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

However, the users of TensorFlow Lite WebAssembly runtime can only access [128-bit SIMD instructions](https://github.com/WebAssembly/simd) and [multi-threading](https://github.com/WebAssembly/threads) of CPU device via [XNNPACK delegate](https://github.com/huningxin/tensorflow/tree/webnn_delegate/tensorflow/lite/delegates/xnnpack). As a comparison, the users of TensorFlow Lite native runtime could access GPU acceleration via [GPU delegate](https://www.tensorflow.org/lite/performance/gpu) and DSP acceleration via [NNAPI delegate](https://www.tensorflow.org/lite/performance/nnapi). The lack of access to platform capabilities beneficial for ML such as dedicated ML hardware accelerators constraints the scope of experiences and leads to inefficient implementations on modern hardware. This disadvantages the users of TensorFlow Lite WebAssembly runtime in comparison to the users of its native runtime.

According to [data](https://www.w3.org/2020/06/machine-learning-workshop/talks/access_purpose_built_ml_hardware_with_web_neural_network_api.html#slide-4), when testing MobileNetV2 on a mainstream laptop, the native inference could be 9.7 times faster than WebAssembly SIMD. If enabling WebAssembly multi-threading, the native inference is still about 4 times faster. That's because the native inference could leverage the longer SIMD (e.g. [AVX 256](https://en.wikipedia.org/wiki/Advanced_Vector_Extensions)) and optimized memory layout (e.g. [blocked memory layout](https://oneapi-src.github.io/oneDNN/dev_guide_understanding_memory_formats.html)) for specific SIMD instruction set. The data also demostrates when native inference accesses the ML specialized hardware, such as [VNNI](https://en.wikichip.org/wiki/x86/avx512_vnni) instruction on laptop or DSP on smartphone, for the 8-bit quantized model inference, the performance gap would be even over 10 times. This capability would not only speed up the inference performance but also help save the power consumption. For example, as the [slide](https://www.w3.org/2020/06/machine-learning-workshop/talks/accelerate_ml_inference_on_mobile_devices_with_android_nnapi.html#slide-8) illustrates, by leveraging DSP through NNAPI, the power reduction could be 3.7x.

The WebNN API is being standardized by W3C Web Machine Learning [Working Group](https://www.w3.org/groups/wg/webmachinelearning) (WebML WG) after two years incubation by W3C Web Machine Learning [Community Group](https://www.w3.org/groups/cg/webmachinelearning). W3C WebML WG published WebNN [First Public Working Draft](https://www.w3.org/2020/Process-20200915/#fpwd) in Q2 2021 and plans to release the [Candidate Recommendation](https://www.w3.org/2020/Process-20200915/#RecsCR) in Q2 2022. WebNN may be implemented in Web browsers by using the available native operating system machine learning APIs, such as Android [Neural Networks API](https://developer.android.com/ndk/guides/neuralnetworks), Windows [DirectML API](https://docs.microsoft.com/en-us/windows/ai/directml/dml-intro) and macOS/iOS [ML Compute API](https://developer.apple.com/documentation/mlcompute/). This architecture allows JavaScript ML frameworks to tap into cutting-edge machine learning innovations in the operating system and the hardware platform underneath it without being tied to platform-specific capabilities, bridging the gap between software and hardware through a hardware-agnostic abstraction layer.

As a related work, [OpenCV.js]() (the OpenCV WebAssembly runtime) has started implementing WebNN backend of dnn (deep neural network) module. The [early result](https://github.com/opencv/opencv/pull/20406) showed 2x speedup for GoogleNet compared to its optimized WebAssembly backend when running in Node.js/Electron.js runtime.

## User Benefit

How will users (or other contributors) benefit from this work? What would be the
headline in the release notes or blog post?



## Design Proposal

This is the meat of the document, where you explain your proposal. If you have
multiple alternatives, be sure to use sub-sections for better separation of the
idea, and list pros/cons to each approach. If there are alternatives that you
have eliminated, you should also list those here, and explain why you believe
your chosen approach is superior.

Make sure you’ve thought through and addressed the following sections. If a section is not relevant to your specific proposal, please explain why, e.g. your RFC addresses a convention or process, not an API.


### Alternatives Considered
* Make sure to discuss the relative merits of alternatives to your proposal.

### Performance Implications
* Do you expect any (speed / memory)? How will you confirm?
* There should be microbenchmarks. Are there?
* There should be end-to-end tests and benchmarks. If there are not (since this is still a design), how will you track that these will be created?

### Dependencies
* Dependencies: does this proposal add any new dependencies to TensorFlow?
* Dependent projects: are there other areas of TensorFlow or things that use TensorFlow (TFX/pipelines, TensorBoard, etc.) that this affects? How have you identified these dependencies and are you sure they are complete? If there are dependencies, how are you managing those changes?

### Engineering Impact
* Do you expect changes to binary size / startup time / build time / test times?
* Who will maintain this code? Is this code in its own buildable unit? Can this code be tested in its own? Is visibility suitably restricted to only a small API surface for others to use?

### Platforms and Environments
* Platforms: does this work on all platforms supported by TensorFlow? If not, why is that ok? Will it work on embedded/mobile? Does it impact automatic code generation or mobile stripping tooling? Will it work with transformation tools?
* Execution environments (Cloud services, accelerator hardware): what impact do you expect and how will you confirm?

### Best Practices
* Does this proposal change best practices for some aspect of using/developing TensorFlow? How will these changes be communicated/enforced?

### Tutorials and Examples
* If design changes existing API or creates new ones, the design owner should create end-to-end examples (ideally, a tutorial) which reflects how new feature will be used. Some things to consider related to the tutorial:
    - The minimum requirements for this are to consider how this would be used in a Keras-based workflow, as well as a non-Keras (low-level) workflow. If either isn’t applicable, explain why.
    - It should show the usage of the new feature in an end to end example (from data reading to serving, if applicable). Many new features have unexpected effects in parts far away from the place of change that can be found by running through an end-to-end example. TFX [Examples](https://github.com/tensorflow/tfx/tree/master/tfx/examples) have historically been good in identifying such unexpected side-effects and are as such one recommended path for testing things end-to-end.
    - This should be written as if it is documentation of the new feature, i.e., consumable by a user, not a TensorFlow developer. 
    - The code does not need to work (since the feature is not implemented yet) but the expectation is that the code does work before the feature can be merged. 

### Compatibility
* Does the design conform to the backwards & forwards compatibility [requirements](https://www.tensorflow.org/programmers_guide/version_compat)?
* How will this proposal interact with other parts of the TensorFlow Ecosystem?
    - How will it work with TFLite?
    - How will it work with distribution strategies?
    - How will it interact with tf.function?
    - Will this work on GPU/TPU?
    - How will it serialize to a SavedModel?

### User Impact
* What are the user-facing changes? How will this feature be rolled out?

## Detailed Design

This section is optional. Elaborate on details if they’re important to
understanding the design, but would make it hard to read the proposal section
above.

## Questions and Discussion Topics

Seed this with open questions you require feedback on from the RFC process.
