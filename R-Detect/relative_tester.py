import torch
import nltk
from roberta_model_loader import RobertaModelLoader # 确保这个类提供了tokenizer和model
from feature_ref_loader import feature_ref_loader
from meta_train import net as global_net_loader_instance # 导入全局实例，但我们将替换它
from regression_model_loader import regression_model
from MMD import MMD_3_Sample_Test
from utils import FeatureExtractor, HWT, MGT, config, get_device # 导入get_device

class RelativeTester:
    def __init__(self):
        print("Relative Tester init")
        self.device = get_device() # 在RelativeTester初始化时确定设备

        # 1. 实例化 RobertaModelLoader
        # 确保 RobertaModelLoader 内部正确加载了 tokenizer 和 model
        self.roberta_model_loader = RobertaModelLoader()

        # 2. 实例化 NetLoader，并在 __init__ 中移动其内部模型到正确的设备
        # 不再依赖全局的 net 实例，而是在这里创建并管理
        # 假设 NetLoader 已经被修改为在初始化时将自己的 .net 和相关张量移动到DEVICE
        from meta_train import NetLoader # 确保导入 NetLoader 类本身
        self.net_loader_instance = NetLoader() 
        # NetLoader的__init__应该已经处理了 .to(DEVICE) 的问题，这里不再显式调用

        # 3. 实例化 FeatureExtractor
        # 将 RobertaModelLoader 实例和 NetLoader 实例的 .net 模型传递给 FeatureExtractor
        self.feature_extractor = FeatureExtractor(self.roberta_model_loader, self.net_loader_instance)
        # FeatureExtractor 的 __init__ 中应该会负责将 self.roberta_model_loader.model
        # 和 self.net_loader_instance.net 移动到 self.device

        # 4. 加载特征参考，并确保它们被移动到正确的设备
        self.feature_hwt_ref = feature_ref_loader(config["feature_ref_HWT"])
        self.feature_mgt_ref = feature_ref_loader(config["feature_ref_MGT"])
        
        # 显式将加载的特征参考张量移动到当前设备
        self.feature_hwt_ref = self.feature_hwt_ref.to(self.device)
        self.feature_mgt_ref = self.feature_mgt_ref.to(self.device)

        # TODO: 回归模型也可能需要移动到GPU，如果它是一个PyTorch模型
        # 例如: if hasattr(regression_model, 'to'): regression_model.to(self.device)


    def sents_split(self, text):
        # nltk.download("punkt", quiet=True)
        # 只需要下载一次，如果之前已经成功下载，这一行可以移除或保持不变
        # 或者确保在程序启动时一次性下载
        sents = nltk.sent_tokenize(text)
        return [sent for sent in sents if 5 < len(sent.split())]

    def test(self, input_text, threshold=0.2, round=20):
        print("Relative Tester test")
        # Split the input text
        sents = self.sents_split(input_text)
        print("DEBUG: sents:", len(sents))
        # Extract features
        # feature_for_sents 应该已经在 FeatureExtractor.process_sents 中移动到GPU了
        feature_for_sents = self.feature_extractor.process_sents(sents, False)
        
        if len(feature_for_sents) <= 1:
            return "Too short to test! Please input more than 2 sentences."
        
        # Cutoff the features
        min_len = min(
            len(feature_for_sents),
            len(self.feature_hwt_ref),
            len(self.feature_mgt_ref),
        )
        # Calculate MMD
        h_u_list = []
        p_value_list = []
        t_list = []

        for i in range(round):
            # 确保采样后的张量也在GPU上。
            # 因为 feature_for_sents, self.feature_hwt_ref, self.feature_mgt_ref 
            # 已经在GPU上，所以采样结果也应该在GPU上。这里不再需要显式 .to(self.device)
            feature_for_sents_sample = feature_for_sents[
                torch.randperm(len(feature_for_sents), device=self.device)[:min_len] # 确保randperm也在同一设备
            ]
            feature_hwt_ref_sample = self.feature_hwt_ref[
                torch.randperm(len(self.feature_hwt_ref), device=self.device)[:min_len] # 确保randperm也在同一设备
            ]
            feature_mgt_ref_sample = self.feature_mgt_ref[
                torch.randperm(len(self.feature_mgt_ref), device=self.device)[:min_len] # 确保randperm也在同一设备
            ]
            
            # 调用 self.net_loader_instance.net，它已经在GPU上
            h_u, p_value, t, *rest = MMD_3_Sample_Test(
                self.net_loader_instance.net(feature_for_sents_sample), # 使用实例化的net
                self.net_loader_instance.net(feature_hwt_ref_sample),   # 使用实例化的net
                self.net_loader_instance.net(feature_mgt_ref_sample),   # 使用实例化的net
                feature_for_sents_sample.view(feature_for_sents_sample.shape[0], -1),
                feature_hwt_ref_sample.view(feature_hwt_ref_sample.shape[0], -1),
                feature_mgt_ref_sample.view(feature_mgt_ref_sample.shape[0], -1),
                self.net_loader_instance.sigma,    # 使用实例化的net_loader_instance的属性
                self.net_loader_instance.sigma0_u, # 使用实例化的net_loader_instance的属性
                self.net_loader_instance.ep,       # 使用实例化的net_loader_instance的属性
                0.05,
            )

            h_u_list.append(h_u)
            p_value_list.append(p_value)
            t_list.append(t)

        power = sum(h_u_list) / len(h_u_list)
        print("DEBUG: power:", power)
        print("DEBUG: power list:", h_u_list)
        # Return the result
        return (
            "Most likely Human Write"
            if power <= threshold
            else "Most likely Machine Generated"
        )
