import torch
import random
import numpy as np
# from transformers import AutoTokenizer, AutoModelForSequenceClassification # 确保这些被导入，因为你正在使用它们
# 上面两行保持注释，因为它们可能在 feature_ref_generater.py 中定义，或者通过其他方式传入

config = {}


def get_device():
    # 确保config["use_gpu"]存在并且为True时才返回cuda
    return (
        torch.device("cuda:0") if config.get("use_gpu", False) and torch.cuda.is_available() else torch.device("cpu")
    )


HWT = "HWT"
MGT = "MGT"


def init_random_seeds():
    print("Init random seeds")
    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    # 只有当CUDA可用时才设置CUDA相关的随机种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(0)
        torch.cuda.manual_seed_all(0)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


class FeatureExtractor:
    def __init__(self, model_obj, net_loader_obj=None): # 将参数名改为net_loader_obj以避免混淆
        self.llm_model = model_obj
        self.net_loader = net_loader_obj # 存储 NetLoader 实例

        self.device = get_device() # 在初始化时确定设备

        # 将LLM模型移动到指定的设备
        if hasattr(self.llm_model, 'model') and self.llm_model.model is not None:
            self.llm_model.model.to(self.device)
            self.llm_model.model.eval() # 将模型设置为评估模式

        # 如果 net_loader_obj 存在，并且它内部的 .net 是 PyTorch 模型，则将其移动到相同的设备
        if self.net_loader is not None and hasattr(self.net_loader, 'net') and isinstance(self.net_loader.net, torch.nn.Module):
            # 注意：NetLoader的__init__中已经执行了 self.net = self.net.to(DEVICE)
            # 所以这里理论上可以不用再to(self.device)了，
            # 但为了代码健壮性，以防万一或未来NetLoader行为改变，可以保留。
            # 这里是NetLoader的.net属性，而不是NetLoader对象本身
            self.net_loader.net.to(self.device) 
            self.net_loader.net.eval() # 将 NetLoader 内部的 PyTorch 模型设置为评估模式
            
            # 同样将 NetLoader 内部的 sigma, sigma0_u, ep 也移动到设备
            # 由于它们是张量，可以直接移动
            if hasattr(self.net_loader, 'sigma') and self.net_loader.sigma is not None:
                self.net_loader.sigma = self.net_loader.sigma.to(self.device)
            if hasattr(self.net_loader, 'sigma0_u') and self.net_loader.sigma0_u is not None:
                self.net_loader.sigma0_u = self.net_loader.sigma0_u.to(self.device)
            if hasattr(self.net_loader, 'ep') and self.net_loader.ep is not None:
                self.net_loader.ep = self.net_loader.ep.to(self.device)


    def process(self, text, net_required=True):
        # Tokenize
        tokens = self.llm_model.tokenizer(
            [text], # 这里的[text]很重要，表示输入是一个list，即使只有一个文本
            padding="max_length",
            truncation=True,
            max_length=100,  # 将最大长度设置为RoBERTa的512
            return_tensors="pt",
        )
        # 将输入令牌移动到与模型相同的设备
        tokens = {k: v.to(self.device) for k, v in tokens.items()}

        # Predict
        with torch.no_grad(): # 在推理时禁用梯度计算，节省内存和加速
            outputs = self.llm_model.model(**tokens)
        
        # Get the feature for input text
        attention_mask = tokens["attention_mask"].unsqueeze(-1)
        hidden_states_masked = (
            outputs.last_hidden_state * attention_mask
        )  # Ignore the padding tokens
        
        if net_required and self.net_loader is not None:
            # 确保传递给self.net_loader.net的数据也在正确的设备上
            # 这里调用的是 NetLoader 内部的 net 模型
            feature = self.net_loader.net(hidden_states_masked) 
            return feature.detach() # .detach() 确保从计算图中分离，避免不必要的内存占用
        else:
            return hidden_states_masked.detach() # 返回前分离，如果不需要net

    def process_sents(self, sents, net_required=True):
        features = []
        for sent in sents:
            # 确保每个单独的特征也在正确的设备上
            features.append(self.process(sent, net_required))
        if not features:
            # 返回一个空张量，并确保它在正确的设备上
            return torch.tensor([], device=self.device)
        # 确保拼接后的张量也在正确的设备上
        return torch.cat(features, dim=0).detach() # .detach() 确保从计算图中分离