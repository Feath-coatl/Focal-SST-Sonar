# Focal SST 说明文档(施工中)

## 文件架构

### Data_augment
用于原始数据集的数据增强
- *build_object_db.py*：指定数据集文件夹，提取所有的目标，生成为.pkl文件。
- *augment_utils.py*：提供用于数据增强的相关组件函数
- *run_augmentation.py*: 数据增强主流程，利用上面生成的.pkl文件和组件函数。

详情见该文件夹内 *README.md*

### Data_label
用于数据的标注测试与错误修复（该测试已结束，不再用于项目主流程）
- *bb_err_detect*：用于检测原始数据集中的标注错误，终端输出存在标注错误的文件。
- *bb_err_fix.py*：将点云目标类标注出来的离群噪点筛出，重新标注为正确的噪声类。
- *data_bb_calc.py*：计算目标点云的包围盒，测试标注效果。
  - 该包围盒参数为7D，分别是cx,cy,cz,lx,ly,lz,rot。

### Data_visual_analyze
用于数据的可视化与数据集相关分析
- *analyze_intensity.py*：分析数据集的强度信息，输出为.json文件。
- *calculate_dataset_range.py*：分析数据集的尺寸范围。
  - 运行结果已经在该程序首部给出，无需再次运行。
- *pointcloud_statistics.py*： 输出.json文件，给出数据集中每类目标所处数据文件位置，统计信息包括该目标在该帧点数以及中心位置。
- *<dual_>pointcloud_visualizer.py*：点云可视化程序，含dual_字样的为对比程序，可以对比同名的原始文件与增强后文件的效果。

## 数据集架构

## 其余说明