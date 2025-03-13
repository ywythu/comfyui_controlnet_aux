from ..utils import common_annotator_call, define_preprocessor_inputs, INPUT
import comfy.model_management as model_management

class Zoe_Depth_Map_Preprocessor:
    # 类级别缓存变量
    cached_model = None
    
    @classmethod
    def INPUT_TYPES(s):
        return define_preprocessor_inputs(resolution=INPUT.RESOLUTION())

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "execute"

    CATEGORY = "ControlNet Preprocessors/Normal and Depth Estimators"

    def execute(self, image, resolution=512, **kwargs):
        from custom_controlnet_aux.zoe import ZoeDetector

        # 检查是否已有缓存模型
        if Zoe_Depth_Map_Preprocessor.cached_model is None:
            # 加载模型并更新缓存
            Zoe_Depth_Map_Preprocessor.cached_model = ZoeDetector.from_pretrained().to(model_management.get_torch_device())
            print("ZOE: 已加载并缓存模型")
        
        # 使用缓存的模型
        model = Zoe_Depth_Map_Preprocessor.cached_model
        
        out = common_annotator_call(model, image, resolution=resolution)
        # 不再删除模型，保留缓存
        return (out, )

NODE_CLASS_MAPPINGS = {
    "Zoe-DepthMapPreprocessor": Zoe_Depth_Map_Preprocessor
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "Zoe-DepthMapPreprocessor": "Zoe Depth Map"
}