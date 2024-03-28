#include <iostream>
#include <vector>
#include <random>
#include <chrono>

#include "onnxruntime_cxx_api.h"
#include "./model/model.ort.h"


int main(int argc, char* argv[])
{
    std::cout << "hello ONNX" << std::endl;

    // Seed RNG and distribution
    unsigned seed = std::chrono::system_clock::now().time_since_epoch().count();
    std::default_random_engine generator (seed);
    std::normal_distribution<double> distribution(0.0,1.0);

    // Init ORT Session
    Ort::SessionOptions sessionOptions;
    std::unique_ptr<Ort::Session> mSession;
    Ort::RunOptions mRunOptions {nullptr};
    Ort::Env mEnv {};

    // set number of threads
    sessionOptions.SetIntraOpNumThreads(1);
    sessionOptions.SetInterOpNumThreads(1);

    // load model
    try {
      mSession = std::make_unique<Ort::Session>(mEnv, (void*) model_ort_start, model_ort_size, sessionOptions);
    } catch (std::exception& e) {
      printf("Exception: %s\n", e.what());
    }

    auto info = Ort::MemoryInfo::CreateCpu(OrtDeviceAllocator, OrtMemTypeCPU);

    // used to get input/output names
    Ort::AllocatorWithDefaultOptions allocator;

    // collect information about model inputs
    size_t node_count = mSession->GetInputCount();
    std::cout << "input count: " << node_count << std::endl;

    std::vector<std::vector<int64_t>> mInputShapes(node_count);
    std::vector<std::string> mInputNames(node_count);

    for (size_t i = 0; i < node_count; i++){
        auto tmp = mSession->GetInputNameAllocated(i, allocator);
        mInputNames[i] = tmp.get();
        mInputShapes[i] = mSession->GetInputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "  input " << i << " shape: ";
        for(size_t j=0; j<mInputShapes[i].size(); j++){
            std::cout << mInputShapes[i][j] << ", ";
        }
        std::cout << std::endl;
    }

    // collect information about model outputs
    node_count = mSession->GetOutputCount();
    std::cout << "output count: " << node_count << std::endl;

    std::vector<std::vector<int64_t>> mOutputShapes(node_count);
    std::vector<std::string> mOutputNames(node_count);

    for (size_t i = 0; i < node_count; i++){
        auto tmp = mSession->GetOutputNameAllocated(i, allocator);
        mOutputNames[i] = tmp.get();
        mOutputShapes[i] = mSession->GetOutputTypeInfo(i).GetTensorTypeAndShapeInfo().GetShape();
        std::cout << "  output " << i << " shape: ";
        for(size_t j=0; j<mOutputShapes[i].size(); j++){
            std::cout << mOutputShapes[i][j] << ", ";
        }
        std::cout << std::endl;
    }

    
    // allocate input and output data
    std::vector<float> mZScratch;
    std::vector<float> mYScratch;
    mZScratch.resize(64);
    mYScratch.resize(144);

    // fill input with N(0, 1) values
    std::cout << "zInput: [";
    for(size_t i=0; i<mZScratch.size(); i++){
        mZScratch[i] = distribution(generator);
        if(i%8 == 0) std::cout << std::endl << "  ";
        printf("% .2f, ", mZScratch[i]);
    }
    std::cout << "]" << std::endl;

    // fill output with 0.0f
    std::fill(mYScratch.begin(), mYScratch.end(), 0.0f);

    // create input and output tensors
    std::vector<Ort::Value> mInputTensors;
    std::vector<Ort::Value> mOutputTensors;
    mInputTensors.push_back(Ort::Value::CreateTensor<float>(info, mZScratch.data(), mZScratch.size(), mInputShapes[0].data(), mInputShapes[0].size()));
    mOutputTensors.push_back(Ort::Value::CreateTensor<float>(info, mYScratch.data(), mYScratch.size(), mOutputShapes[0].data(), mOutputShapes[0].size()));

    // convert input and output names into C strings
    const char* inputNamesCstrs[] = {mInputNames[0].c_str()};
    const char* outputNamesCstrs[] = {mOutputNames[0].c_str()};

    // run inference on model
    mSession->Run(mRunOptions, inputNamesCstrs, mInputTensors.data(), mInputTensors.size(), outputNamesCstrs, mOutputTensors.data(), mOutputTensors.size());

    // print output
    std::cout << "Output: [";
    for(size_t i=0; i<mYScratch.size(); i++){
        if(i%16 == 0) std::cout << std::endl << "  ";
        printf("% .1f, ", mYScratch[i]);
    }
    std::cout << "]" << std::endl;

    return 0;
}
