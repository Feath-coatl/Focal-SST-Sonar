#!/usr/bin/env python3
"""
数据诊断工具：检查具体哪些数据文件有问题
"""
import numpy as np
from pathlib import Path
import sys

# 配置路径
DATA_PATH = Path('data/sonar/points')

# 从报错信息中得知的问题文件
PROBLEM_FILES = ['a_0180077', 'd_0170751', 'c_0140342']

def diagnose_file(filename):
    """详细诊断单个文件"""
    filepath = DATA_PATH / f'{filename}.txt'
    
    print("=" * 80)
    print(f"诊断文件: {filepath}")
    print("=" * 80)
    
    if not filepath.exists():
        print(f"❌ 文件不存在！")
        return False
    
    print(f"✓ 文件存在，大小: {filepath.stat().st_size} bytes")
    print()
    
    # 1. 尝试直接读取文件内容
    print("【步骤1】检查文件内容...")
    try:
        with open(filepath, 'r') as f:
            lines = f.readlines()
        print(f"✓ 文件有 {len(lines)} 行")
    except Exception as e:
        print(f"❌ 无法读取文件: {e}")
        return False
    
    # 2. 检查前几行和后几行
    print("\n【步骤2】显示前5行:")
    for i, line in enumerate(lines[:5]):
        print(f"  Line {i+1}: {line.strip()}")
    
    print("\n【步骤3】显示后5行:")
    for i, line in enumerate(lines[-5:]):
        print(f"  Line {len(lines)-5+i+1}: {line.strip()}")
    
    # 3. 逐行解析，找出问题行
    print("\n【步骤4】逐行解析并检查数值...")
    problem_lines = []
    valid_points = []
    
    for line_num, line in enumerate(lines, 1):
        values = line.strip().split()
        
        if len(values) < 4:
            problem_lines.append({
                'line': line_num,
                'reason': f'列数不足 (只有{len(values)}列)',
                'content': line.strip()
            })
            continue
        
        try:
            x = float(values[0])
            y = float(values[1])
            z = float(values[2])
            intensity = float(values[3])
            
            # 检查各种问题
            issues = []
            
            if not np.isfinite(x):
                issues.append(f'x={values[0]} 不是有限数')
            if not np.isfinite(y):
                issues.append(f'y={values[1]} 不是有限数')
            if not np.isfinite(z):
                issues.append(f'z={values[2]} 不是有限数')
            if not np.isfinite(intensity):
                issues.append(f'intensity={values[3]} 不是有限数')
            
            if intensity < 0:
                issues.append(f'intensity={intensity} 是负数')
            
            # 检查是否超大
            if abs(x) > 1e10:
                issues.append(f'x={x:.2e} 数值过大')
            if abs(y) > 1e10:
                issues.append(f'y={y:.2e} 数值过大')
            if abs(z) > 1e10:
                issues.append(f'z={z:.2e} 数值过大')
            if intensity > 1e15:
                issues.append(f'intensity={intensity:.2e} 数值过大')
            
            if issues:
                problem_lines.append({
                    'line': line_num,
                    'reason': '; '.join(issues),
                    'content': line.strip()
                })
            else:
                valid_points.append([x, y, z, intensity])
                
        except ValueError as e:
            problem_lines.append({
                'line': line_num,
                'reason': f'无法转换为浮点数: {e}',
                'content': line.strip()
            })
        except OverflowError as e:
            problem_lines.append({
                'line': line_num,
                'reason': f'数值溢出: {e}',
                'content': line.strip()
            })
    
    # 4. 报告问题
    print(f"\n✓ 成功解析: {len(valid_points)} 个点")
    print(f"❌ 发现问题: {len(problem_lines)} 行")
    
    if problem_lines:
        print("\n【步骤5】问题行详情（最多显示前20个）:")
        for i, prob in enumerate(problem_lines[:20]):
            print(f"\n  问题 #{i+1}:")
            print(f"    行号: {prob['line']}")
            print(f"    原因: {prob['reason']}")
            print(f"    内容: {prob['content'][:100]}...")
    
    # 5. 统计有效数据
    if valid_points:
        points_array = np.array(valid_points)
        print("\n【步骤6】有效数据统计:")
        print(f"  X 范围: [{points_array[:, 0].min():.2f}, {points_array[:, 0].max():.2f}]")
        print(f"  Y 范围: [{points_array[:, 1].min():.2f}, {points_array[:, 1].max():.2f}]")
        print(f"  Z 范围: [{points_array[:, 2].min():.2f}, {points_array[:, 2].max():.2f}]")
        print(f"  Intensity 范围: [{points_array[:, 3].min():.2e}, {points_array[:, 3].max():.2e}]")
        print(f"  Intensity 均值: {points_array[:, 3].mean():.2e}")
        print(f"  Intensity 中位数: {np.median(points_array[:, 3]):.2e}")
    
    # 6. 尝试用numpy.loadtxt加载，看看会不会触发异常
    print("\n【步骤7】测试numpy.loadtxt加载...")
    try:
        data = np.loadtxt(str(filepath), dtype=np.float32)
        print(f"✓ numpy.loadtxt 成功加载, shape: {data.shape}")
        
        # 检查加载后的数据
        if np.any(np.isnan(data)):
            print(f"❌ 加载后的数据包含 NaN: {np.sum(np.isnan(data))} 个")
        if np.any(np.isinf(data)):
            print(f"❌ 加载后的数据包含 Inf: {np.sum(np.isinf(data))} 个")
            
    except Exception as e:
        print(f"❌ numpy.loadtxt 失败: {type(e).__name__}: {e}")
    
    # 7. 测试对数运算
    if valid_points:
        print("\n【步骤8】测试对数变换...")
        try:
            intensities = points_array[:, 3]
            log_intensities = np.log10(intensities + 1)
            
            if np.any(np.isnan(log_intensities)):
                print(f"❌ 对数变换产生 NaN: {np.sum(np.isnan(log_intensities))} 个")
            elif np.any(np.isinf(log_intensities)):
                print(f"❌ 对数变换产生 Inf: {np.sum(np.isinf(log_intensities))} 个")
            else:
                print(f"✓ 对数变换成功")
                print(f"  Log10(intensity+1) 范围: [{log_intensities.min():.2f}, {log_intensities.max():.2f}]")
        except Exception as e:
            print(f"❌ 对数变换失败: {type(e).__name__}: {e}")
    
    print("\n" + "=" * 80)
    return len(problem_lines) == 0


def main():
    print("\n")
    print("=" * 80)
    print("数据诊断工具 - 检查声呐点云数据文件")
    print("=" * 80)
    print()
    
    if not DATA_PATH.exists():
        print(f"❌ 数据路径不存在: {DATA_PATH}")
        print("   请检查路径配置")
        sys.exit(1)
    
    print(f"数据路径: {DATA_PATH}")
    print(f"问题文件列表: {PROBLEM_FILES}")
    print()
    
    # 诊断每个问题文件
    results = {}
    for filename in PROBLEM_FILES:
        try:
            is_ok = diagnose_file(filename)
            results[filename] = 'OK' if is_ok else 'FAILED'
        except Exception as e:
            print(f"❌ 诊断 {filename} 时发生异常: {e}")
            import traceback
            traceback.print_exc()
            results[filename] = 'ERROR'
        print()
    
    # 总结
    print("=" * 80)
    print("诊断总结:")
    print("=" * 80)
    for filename, status in results.items():
        status_icon = "✓" if status == "OK" else "❌"
        print(f"  {status_icon} {filename}.txt: {status}")
    print("=" * 80)
    
    # 建议
    print("\n【建议】:")
    failed_files = [f for f, s in results.items() if s != 'OK']
    if failed_files:
        print(f"1. 发现 {len(failed_files)} 个有问题的文件")
        print(f"2. 请检查这些文件的原始数据源")
        print(f"3. 可以选择:")
        print(f"   a) 修复源数据")
        print(f"   b) 从训练集中排除这些文件")
        print(f"   c) 在数据加载时自动跳过问题行")
    else:
        print("所有检查的文件都没有明显问题")
        print("浮点异常可能发生在numpy内部操作时")


if __name__ == '__main__':
    main()
