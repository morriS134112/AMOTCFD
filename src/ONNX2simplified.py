import onnx
from onnxsim import simplify

# 加載 ONNX 模型
onnx_path = "IDK.onnx"
simplified_onnx_path = "IDK_simplified.onnx"

# 加載原始模型
model = onnx.load(onnx_path)

# 簡化模型
model_simp, check = simplify(model)
assert check, "簡化失敗"

# 保存簡化後的模型
onnx.save(model_simp, simplified_onnx_path)

print(f"簡化後的 ONNX 模型已保存至 {simplified_onnx_path}")