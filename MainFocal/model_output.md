- https://gemini.google.com/share/f3f96a6e857b 项目主进程（数据集载入和标注）
- https://gemini.google.com/share/f8f556c1e39e 3D点云标注可视化程序（对应visualize_sonar_infos.py）
- https://gemini.google.com/share/2180c1f509cf 旧的生成3D BOX以及标注错误的修复
- https://gemini.google.com/share/ca2d89748681 点云数据范围统计
- https://gemini.google.com/share/d07e6aae6150 github问题

### 一键安装所有依赖
pip install -r requirements.txt
### 训练你的 Focal SST (完整版)
python tools/train.py --cfg_file tools/cfgs/sonar_models/focal_sst.yaml
