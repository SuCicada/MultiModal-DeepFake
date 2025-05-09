```bash
conda create -n DGM4 python=3.8


conda install --yes -c pytorch pytorch=1.10.0 torchvision==0.11.1 cudatoolkit=11.3
pip install -r requirements.txt
conda install -c conda-forge ruamel_yaml
```

## 如何跑
```bash
conda activate DGM4
python inference.py  # 这是推理文件 
```

## point
1. '--config', './configs/my.yaml',
2. my.yaml:     val_file: ["./configs/my.json"]
3. 图片配置都在 my.json
