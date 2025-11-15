在原 YOLOS 实现的基础上增加对 DINOv3 backbone 的支持
## 运行
1. 安装依赖项
2. 将 PASCAL VOC 2012 和 COCO 2017 数据集放到 data 目录下
3. 运行 sh voc2coco/voc2coco.sh 生成 VOC 的 COCO 格式标注
4. 登录 HuggingFace 以加载 DINOv3 模型；vanilla ViT 的预训练参数参考原仓库 hustvl/YOLOS
5. 在 configs/ 目录中修改对应的配置文件
6. 执行 torchrun main.py --config_path path/to/config/file