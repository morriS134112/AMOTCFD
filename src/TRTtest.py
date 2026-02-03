import tensorrt as trt
import numpy as np
import pycuda.driver as cuda
import pycuda.autoinit

# 加載 TensorRT 引擎
trt_engine_path = "IDK.trt"
with open(trt_engine_path, "rb") as f:
    engine_data = f.read()

runtime = trt.Runtime(trt.Logger(trt.Logger.WARNING))
engine = runtime.deserialize_cuda_engine(engine_data)

# 創建執行上下文
context = engine.create_execution_context()

# 準備輸入數據
input_data = np.random.randn(1, 3, 608, 1088).astype(np.float32)
output_shape = (1, 1, 152, 272)
output_data = np.zeros(output_shape, dtype=np.float32)

# 分配 GPU 記憶體
d_input = cuda.mem_alloc(input_data.nbytes)
d_output = cuda.mem_alloc(output_data.nbytes)

# 創建一個 CUDA 流
stream = cuda.Stream()

# 將數據傳送到 GPU
cuda.memcpy_htod_async(d_input, input_data, stream)

# 綁定輸入和輸出
bindings = {}
for i in range(engine.num_io_tensors):
    name = engine.get_tensor_name(i)
    if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT:
        bindings[name] = int(d_input)
    else:
        bindings[name] = int(d_output)

# 執行推理
context.execute_async_v3(stream_handle=stream.handle)

# 從 GPU 取回結果
cuda.memcpy_dtoh_async(output_data, d_output, stream)

# 同步流
stream.synchronize()

# 假設已有 pytorch_output
pytorch_output = np.zeros(output_shape, dtype=np.float32)  # 請替換為實際的 PyTorch 輸出

print("TensorRT 輸出:", output_data.shape)
print("PyTorch 輸出:", pytorch_output.shape)
print("最大差異:", np.max(np.abs(pytorch_output - output_data)))