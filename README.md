# Taming Lookup Tables for Image Retouching

## Requirements and Dependencies
+ python==3.7
+ cuda==11.1
+ numpy==1.21.5
+ torch==1.8.0
+ torchvision==0.9.0
+ opencv-python==4.5.5.62
+ matplotlib==3.5.1
+ scipy==1.7.3
+ GPU: NVIDIA GeForce RTX 3090
+ CPU: Intel(R) Xeon(R) Platinum 

## How to run the codes

### 1. Install the tilinear interpolation package

#### 1.1 For GPU User
+ Check your cuda version
```bash
    $ls /usr/local/
```
+ Change the cuda path in your setting
```bash
    $cd trilinear_cpp
    $vim setup.sh
```
```bash
    export CUDA_HOME=/usr/local/your_cuda_version && python setup.py install
```
+ Install the package
```python
    $sh setup.sh
```
#### 1.2 For CPU User

+ Install the package
```bash
    $cd trilinear_cpp
    $python setup.py
```
**ATTENTION:** If you follow the CPU install instruction with the GPU in your device, the default programme will still install the GPU version. If you want to only install the CPU version, please follow this step:
+ Check the setup.py codes
```bash
    $vim setup.py
```

+ Substitute the codes
```python
    # line 5
    if torch.cuda.is_available():

    # substitute line 5 with :
    # if False
```

### 2. Inference the demo images
```bash
    $python inference_demo.py
```

### 3. Optional: Check the retouched results
The retouched images will be saved in the default dir: ./test_image_output

### 3. Optional: Check the total LUT size
```bash
du -sh *.npy
```

```bash
% output
60K     basis_lut.npy
13M     classifier_INT8.npy
100K    Model_lsb_int8.npy
100K    Model_msb_int8.npy
```
