目前实现了五项baseline的复现，部分的结果有保留。
每个文件夹需要单独配置环境，各个文件夹有不同的依赖文件requirements.txt, 使用的python版本分别为：
- Binoculars: `python = 3.9`
- detect-gpt: `python = 3.12`
- fast-detect-gpt: `python = 3.8`
- ghostbuster: `python = 3.10`
- R-detect: `python = 3.12`


要运行得到各个baseline的结果，需要先从`my_work`中复制dataset到对应baseline文件夹下，然后分别运行各个文件夹下的如下文件：
- Binoculars: `test.py`.
- detect-gpt: `run.py`.
- fast-detect-gpt: `scripts/local_infer.py`
- ghostbuster: `test.py`
- R-detect: `run_evaluation.py`
