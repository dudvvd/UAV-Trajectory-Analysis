# UAV 相邻图像轨迹估计

本项目从无人机红外相机相邻图片出发，使用传统计算机视觉方法估计相对运动，并从首帧已知经度、纬度、高度递推后续轨迹。程序会把估计结果与 CSV 中的真实定位数据对比，输出误差 CSV 和轨迹变化图。

## 重要说明

当前数据只有单目相邻图像和每帧真实经纬高，没有相机内参、姿态角、云台角、地面平面约束等信息。因此，单目图像本身不能唯一恢复真实尺度和绝对航向。本项目采用工程上常见的可配置假设：

- 使用相机水平视场角 `--horizontal-fov-deg` 和当前高度把像素位移换算为米。
- 使用 `--yaw-deg` 假设图像坐标系与 ENU 坐标系的夹角。
- 默认保持首帧高度；可用 `--estimate-altitude` 尝试从 `orb_affine` 仿射尺度估计高度变化。
- 真实定位数据只用于首帧初始化和后续误差验证，不在默认递推过程中更新估计位置。

如果后续能提供相机内参、镜头视场角、无人机航向/俯仰/横滚、云台角，估计精度会明显更可靠。

## 环境安装

按需求使用预先准备好的 conda 环境 `uav`：

```powershell
conda activate uav
python -m pip install -r requirements.txt
python -m pip install -e .
```

若不安装为包，也可以在项目根目录临时设置：

```powershell
$env:PYTHONPATH = "src"
```

## 数据结构

项目默认读取：

- 图片目录：`data/IR-image`
- 定位文件：`data/定位数据.csv`

CSV 需要包含字段：

```text
时间戳,经度,纬度,高度
```

图片文件名需要是时间戳，例如 `1734763664200.jpg`。程序会按时间戳把图片和定位记录做内连接。

## 快速测试几张图片

从第 0 帧开始，只处理 10 帧：

```powershell
conda run -n uav python run_trajectory.py --max-frames 10 --print-rows 10
```

从中间某段开始测试：

```powershell
conda run -n uav python run_trajectory.py --start-index 500 --max-frames 20 --print-rows 20
```

输出包含：

- `outputs/orb_affine_start*_n*_results.csv`：逐帧真实值、预测值、误差和图像匹配信息
- `outputs/orb_affine_start*_n*_trajectory.png`：真实轨迹与估计轨迹对比图

## 运行全部数据

```powershell
conda run -n uav python run_trajectory.py
```

全量数据约 2.1 万张图片，运行时间取决于 CPU 性能。可以通过 `--max-width` 降低处理分辨率来提速，例如：

```powershell
conda run -n uav python run_trajectory.py --max-width 640
```

## 方法对比

项目实现了两种非深度学习方案：

- `orb_affine`：ORB 特征点 + KNN 匹配 + RANSAC 部分仿射矩阵，可估计平移和尺度，默认方法。
- `phase`：相位相关法，只估计全局平移，速度快，适合作为对照。

单独指定方法：

```powershell
conda run -n uav python run_trajectory.py --method phase --max-frames 50
```

同一段数据对比两种方法：

```powershell
conda run -n uav python compare_methods.py --start-index 0 --max-frames 50
```

对比摘要会保存到 `outputs/compare/method_summary.csv`。

## 关键参数

- `--start-index`：选定初始图像在匹配序列中的索引。
- `--max-frames`：处理帧数；不填则处理全部。
- `--horizontal-fov-deg`：相机水平视场角，默认 `60` 度。
- `--yaw-deg`：图像坐标系相对 ENU 的旋转角，默认 `0` 度。
- `--max-width`：图像处理最大宽度，默认 `960`。
- `--estimate-altitude`：使用图像仿射尺度估计高度变化；默认固定首帧高度，这在缺少相机内参/姿态时通常更稳定。
- `--calibration-frames`：使用前 N 帧真值拟合像素位移到 ENU 位移的线性映射。该参数会使用首帧之后的真值，只适合标定和评估，不代表纯部署输入。

## 结果字段

结果 CSV 中主要字段：

- `truth_longitude/truth_latitude/truth_altitude`：CSV 真实位置。
- `pred_longitude/pred_latitude/pred_altitude`：图像递推位置。
- `horizontal_error_m`：水平误差。
- `altitude_error_m`：高度误差。
- `total_error_m`：三维误差。
- `dx_px/dy_px/scale/confidence/matches/inliers`：相邻图像运动估计诊断信息。

## 开发流程总结

1. 读取 CSV 并规范字段，把时间戳转为整数。
2. 扫描图片目录，按图片文件名解析时间戳。
3. 将图片和真实定位按时间戳匹配。
4. 用首帧真实经纬高建立局部 ENU 坐标参考。
5. 对相邻图片计算图像位移和尺度。
6. 根据高度、视场角和航向假设把像素运动转换为 ENU 位移。
7. 从首帧开始递推预测轨迹。
8. 将预测轨迹转回经纬高，与真实数据计算误差。
9. 保存逐帧结果 CSV，并绘制真实轨迹/预测轨迹图。

## 后续改进建议

- 接入相机内参、真实水平/垂直 FOV 和畸变参数。
- 接入无人机 IMU 姿态和云台姿态，替代固定 `yaw_deg` 假设。
- 对低置信度帧做跳过、插值或滑动窗口鲁棒优化。
- 在特征较弱的红外图像上加入 CLAHE 增强、网格化特征筛选和光流跟踪。
