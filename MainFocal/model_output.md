- https://gemini.google.com/share/f3f96a6e857b 项目主进程（数据集载入和标注）
- https://gemini.google.com/share/f8f556c1e39e 3D点云标注可视化程序（对应visualize_sonar_infos.py）
- https://gemini.google.com/share/2180c1f509cf 旧的生成3D BOX以及标注错误的修复
- https://gemini.google.com/share/ca2d89748681 点云数据范围统计
- https://gemini.google.com/share/d07e6aae6150 github问题

## 我用的是python3.9.25 cuda12.1 pytorch2.4.1 

### 一键安装所有依赖
> pip install -r requirements.txt

### 安装spconv库（请选择合适的版本）
项目地址：https://github.com/traveller59/spconv?tab=readme-ov-file

### 安装合适的tensorrt和onnx库

### 完成项目设置(应位于focalSST-master文件夹)
> python setup.py develop

### 训练你的 Focal SST (完整版)
> cd tools
> python train.py --cfg_file cfgs/sonar_models/focal_sst.yaml



# 遇到的问题：
## 1.在执行**python setup.py develop**时遇到报错：

× Getting requirements to build editable did not run successfully.
  │ exit code: 1
  ╰─> [19 lines of output]
      Traceback (most recent call last):
        File "/root/miniconda3/envs/py3925/lib/python3.9/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 389, in <module>
          main()
        File "/root/miniconda3/envs/py3925/lib/python3.9/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 373, in main
          json_out["return_val"] = hook(**hook_input["kwargs"])
        File "/root/miniconda3/envs/py3925/lib/python3.9/site-packages/pip/_vendor/pyproject_hooks/_in_process/_in_process.py", line 157, in get_requires_for_build_editable
          return hook(config_settings)
        File "/tmp/pip-build-env-qx3rdu2_/overlay/lib/python3.9/site-packages/setuptools/build_meta.py", line 479, in get_requires_for_build_editable
          return self.get_requires_for_build_wheel(config_settings)
        File "/tmp/pip-build-env-qx3rdu2_/overlay/lib/python3.9/site-packages/setuptools/build_meta.py", line 333, in get_requires_for_build_wheel
          return self._get_build_requires(config_settings, requirements=[])
        File "/tmp/pip-build-env-qx3rdu2_/overlay/lib/python3.9/site-packages/setuptools/build_meta.py", line 301, in _get_build_requires
          self.run_setup()
        File "/tmp/pip-build-env-qx3rdu2_/overlay/lib/python3.9/site-packages/setuptools/build_meta.py", line 518, in run_setup
          super().run_setup(setup_script=setup_script)
        File "/tmp/pip-build-env-qx3rdu2_/overlay/lib/python3.9/site-packages/setuptools/build_meta.py", line 317, in run_setup
          exec(code, locals())
        File "<string>", line 5, in <module>
      ModuleNotFoundError: No module named 'torch'
      [end of output]

*note: This error originates from a subprocess, and is likely not a problem with pip.
ERROR: Failed to build 'file:///root/autodl-tmp/code/Modelproject/MainFocal/focalSST-master' when getting requirements to build editable*

**解决方案**：确保在该conda环境下安装了torch，然后执行**pip install -e . --no-build-isolation**

