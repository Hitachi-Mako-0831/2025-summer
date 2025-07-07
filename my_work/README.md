# REWRITE-Based Machine Generated Text Detection

## Quick Start

安装依赖：
```bash
conda create -n rewrite_env python=3.12
conda activate rewrite_env
pip3 install -U pip
pip3 install -r requirements.txt 
```
下载style representation需要的文本嵌入模型：(有6个G，只看代码逻辑的话可以跳过这一步)

```bash
mkdir ./pretrained_weights
git clone https://huggingface.co/sentence-transformers/paraphrase-distilroberta-base-v1 pretrained_weights/paraphrase-distilroberta-base-v1
```
运行实验：
- 可以修改`main.py`中`model_client`使用的OPENAI api和url为自己使用的版本；
- 修改line 66的数据范围来决定本次实验需要使用的样本范围；
- 修改line 67的文件路径来决定使用哪个数据集进行实验；
- 修改line 121 122的文件路径来存储实验过程产生的数据

上述部分修改完成后，运行`main.py`，可以得到实验结果

## Project View

- `main.py`：进行实验，读入一个数据集，使用GPT-3.5-turbo重写，然后提取特征训练分类器，得到分类器的F1 score
  
- `utils.py`：存储了供`main.py`使用的一些方法。比较关键的部分包括：
  - `get_stat(index)`:对数据集中第`index`个数据进行处理，得到它的离散特征和连续特征，保存到`feature_vectors`中，其中连续特征的获取依赖于`style.py`提供的文本嵌入模型调用功能
  - `classifier(data_range)`:根据输入的`data_range`,使用给定范围的特征向量，让分类器进行训练和预测，从而得到F1 score.
- `style.py`：本段程序根据style representation的源码部分，得到了调用文本嵌入模型的一个函数`get_all_embeddings`.从而实现输入一个文本，得到一个文本对应的由模型推理出的表示文本风格的向量。
- `arguments.py`,`utilities`,`datasets`,`models`,`pretrained_weights`为style representation源码的依赖文件与模型，后续需要整合修改
- `find_feature_xxx.py`是用实验得到的特征，来进行特征可视化分析的程序
- `dataset`存储了我们的数据集，每个数据集下有三个json文件，分别为HWT, MGT以及combine。combine文件将前二者的数据随机打乱后合并，实验运行时只需要从combine读取数据即可。
- 每个`XXX_result`文件夹存储了两个json文件，分别是重写文本数据和对每个文本数据处理得到的特征向量，存储机制方便实验再运行，减轻运算和OPENAI API消耗。