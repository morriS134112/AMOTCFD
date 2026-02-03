import torch
import onnx
import logging
from lib.models.networks.pose_dla_dcn import get_pose_net

# 設置 Python 日誌級別（可選）
logging.basicConfig(level=logging.DEBUG)

# 創建模型
heads = {'hm': 1, 'wh': 4, 'id': 128, 'reg': 2}
model = get_pose_net(34, heads)
model.eval()

# 載入權重
checkpoint = torch.load(r"E:\FairMOT\models\IDK.pth", map_location="cpu")
state_dict = checkpoint.get("state_dict", checkpoint)
model.load_state_dict(state_dict, strict=True)
print("權重載入成功 (strict=True)")

# 定義包裝器模型
class ONNXWrapper(torch.nn.Module):
    def __init__(self, model):
        super(ONNXWrapper, self).__init__()
        self.model = model

    def forward(self, x):
        outputs = self.model(x)[0]
        return (outputs['hm'], outputs['wh'], outputs['id'], outputs['reg'])

onnx_model = ONNXWrapper(model)

# 創建虛擬輸入
dummy_input = torch.randn(1, 3, 608, 1088)
print("Dummy input shape:", dummy_input.shape)

# 測試模型輸出
with torch.no_grad():
    outputs = onnx_model(dummy_input)
    for name, output in zip(["hm", "wh", "id", "reg"], outputs):
        print(f"{name} shape: {output.shape}")
    print("模型輸出測試成功")

# 導出 ONNX 模型
onnx_path = "IDK.onnx"
torch.onnx.export(
    onnx_model,
    dummy_input,
    onnx_path,
    verbose=True,
    opset_version=12,
    input_names=["input"],
    output_names=["hm", "wh", "id", "reg"],
    dynamic_axes={
        "input": {0: "batch_size"},
        "hm": {0: "batch_size"},
        "wh": {0: "batch_size"},
        "id": {0: "batch_size"},
        "reg": {0: "batch_size"}
    }
)

# 驗證 ONNX 模型
onnx_model_loaded = onnx.load(onnx_path)
onnx.checker.check_model(onnx_model_loaded)
print("ONNX 模型驗證成功")
print("Inputs:", [input.name for input in onnx_model_loaded.graph.input])