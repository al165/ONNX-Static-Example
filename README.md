# Example static ONNX Project

A basic example of building a cross-platform, standalone app that runs ONNX models with no dependancies.

Builds with Cmake.

**TODO:** test on Windows, MacOS Intel

The ONNX code is mostly derived from the example in [olilarkin/iPlug2OnnxRuntime](https://github.com/olilarkin/iPlug2OnnxRuntime/blob/master/iPlug2OnnxRuntime/LSTMModelInference.h).

## Instructions
1. Download **static** libraries for your target platform from [csukuangfj/onnxruntime-libs](https://huggingface.co/csukuangfj/onnxruntime-libs/tree/main) 
    - alternatively, build onnxruntime such as with [ort-builder](https://github.com/olilarkin/ort-builder/tree/bfbd362c9660fce9600a43732e3f8b53d5fb243a)
2. Extract and copy ```include/``` and ```lib/``` folders to this root directory
3. Run:
    ```shell
    $ mkdir build
    $ cd build
    $ cmake ..
    $ cmake --build .
    ```  
4. Run the example with ```./ONNX_Test```


## Exporting and converting your own models
1. Export trained model as ```.onnx``` file, for example with pytorch:
    ```python
    torch.onnx.export(
        model.cpu(), 
        torch.randn(model_input.shape), 
        "model.onnx", 
        input_names=["Input"], 
        output_names=["Output"]
    )
    ```
2. If you are using [minimal build](https://onnxruntime.ai/docs/build/custom.html#minimal-build) of onnxruntime, convert model to ```.ort``` file:
    ```shell
    $ pip install onnxruntime
    $ python -m onnxruntime.tools.convert_onnx_models_to_ort model.onnx --enable_type_reduction
    ```
3. Convert model file to C header files:
    ```shell
    $ pip install bin2c
    $ python -m bin2c -o ./model/modelname model.ort
    ```
    (This produces 2 files: ```./model/modelname.c``` and ```./model/modelname.h```)