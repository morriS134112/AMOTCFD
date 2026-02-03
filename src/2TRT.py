import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit

TRT_LOGGER = trt.Logger(trt.Logger.WARNING)

# 加載 ONNX 模型
onnx_file_path = "IDK_simplified.onnx"  # 或 "dla_seg.onnx"
with open(onnx_file_path, "rb") as f:
    onnx_model = f.read()

# 創建 TensorRT Builder 和 Network
builder = trt.Builder(TRT_LOGGER)
network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
network = builder.create_network(network_flags)
parser = trt.OnnxParser(network, TRT_LOGGER)

if not parser.parse(onnx_model):
    print("錯誤：無法解析 ONNX 模型")
    for error in range(parser.num_errors):
        print(parser.get_error(error))
    exit()

# 配置 Builder Config
config = builder.create_builder_config()
config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)  # 1GB 工作空間
config.set_flag(trt.BuilderFlag.FP16)  # 使用 FP16 提升性能

# 構建引擎
profile = builder.create_optimization_profile()
# 如果輸入大小是動態的，需要設置最小/最佳/最大輸入尺寸
# 這裡需要根據實際的輸入名稱和維度設置
# profile.set_shape("input_name", min_shape, opt_shape, max_shape)
# config.add_optimization_profile(profile)

serialized_engine = builder.build_serialized_network(network, config)

# 保存引擎
trt_engine_path = "IDK.trt"
with open(trt_engine_path, "wb") as f:
    f.write(serialized_engine)

print(f"TensorRT 引擎已保存至 {trt_engine_path}")