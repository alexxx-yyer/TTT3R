# ROS Bag Export Guide

这个脚本可以将TTT3R的输出数据导出为ROS bag文件，方便在RViz中可视化。

## 功能特点

- **功能分离**: 独立的脚本，不修改demo.py的核心功能
- **完整数据**: 导出点云和相机位姿
- **RViz兼容**: 标准ROS消息格式，可直接在RViz中播放

## 依赖安装

首先需要安装ROS相关的Python包：

```bash
# 对于ROS Noetic (Python 3)
pip install rospkg catkin_pkg

# 或者如果你使用的是ROS 2
pip install rosbag2
```

注意：这个脚本需要在ROS环境中运行，或者至少安装了ROS的Python包。

## 使用方法

### 1. 运行demo.py生成数据

首先运行demo.py生成输出数据：

```bash
python demo.py \
    --model_path src/cut3r_512_dpt_4_64.pth \
    --seq_path examples/westlake.mp4 \
    --output_dir demo_tmp \
    --size 512
```

这会生成包含以下子目录的输出：
- `demo_tmp/depth/` - 深度图
- `demo_tmp/color/` - 彩色图像
- `demo_tmp/camera/` - 相机位姿和内参

### 2. 导出为ROS bag

运行导出脚本：

```bash
python export_to_rosbag.py \
    --output_dir demo_tmp \
    --bag_file pointcloud.bag \
    --frame_rate 30.0 \
    --topic_prefix /ttt3r
```

参数说明：
- `--output_dir`: demo.py的输出目录
- `--bag_file`: 输出的bag文件路径
- `--frame_rate`: 帧率（Hz），默认30.0
- `--topic_prefix`: ROS话题前缀，默认`/ttt3r`

### 3. 在RViz中播放

```bash
# 启动RViz
rviz

# 在另一个终端播放bag文件
rosbag play pointcloud.bag
```

在RViz中添加以下显示：
1. **PointCloud2**: 话题 `/ttt3r/pointcloud`
   - Fixed Frame: `map`
   - Color Transformer: `RGB8`
   
2. **Pose** (可选): 话题 `/ttt3r/camera_pose`
   - Fixed Frame: `map`

## RViz配置示例

1. 打开RViz: `rviz`
2. 设置Fixed Frame为 `map`
3. 添加PointCloud2显示：
   - 点击"Add" -> "PointCloud2"
   - Topic选择 `/ttt3r/pointcloud`
   - Color Transformer选择 `RGB8`
   - Size (Pixels)可以设置为 `2` 或 `3`
4. (可选) 添加Pose显示：
   - 点击"Add" -> "Pose"
   - Topic选择 `/ttt3r/camera_pose`
   - 可以调整箭头大小和颜色

## 话题说明

导出的bag文件包含以下话题：

- `/ttt3r/pointcloud` (sensor_msgs/PointCloud2)
  - 包含RGB彩色点云数据
  - Frame ID: `map`
  
- `/ttt3r/camera_pose` (geometry_msgs/PoseStamped)
  - 相机在世界坐标系中的位姿
  - Frame ID: `map`

## 注意事项

1. **ROS环境**: 脚本需要在ROS环境中运行，或者至少安装了ROS的Python包
2. **内存使用**: 大型点云数据可能占用较多内存
3. **帧率**: 根据原始视频的帧率设置合适的`--frame_rate`参数
4. **坐标系**: 所有数据都在`map`坐标系中，相机位姿是camera-to-world变换

## 故障排除

### 问题: 找不到ROS模块

```bash
# 确保ROS环境已source
source /opt/ros/noetic/setup.bash  # 对于ROS Noetic
# 或
source /opt/ros/melodic/setup.bash  # 对于ROS Melodic
```

### 问题: 点云在RViz中不显示

- 检查Fixed Frame是否设置为`map`
- 检查点云大小设置是否太小
- 尝试调整RViz的View设置

### 问题: 点云颜色不正确

- 确保Color Transformer设置为`RGB8`
- 检查点云数据是否包含有效的RGB信息

## 示例工作流

```bash
# 1. 运行推理
python demo.py \
    --model_path src/cut3r_512_dpt_4_64.pth \
    --seq_path examples/westlake.mp4 \
    --output_dir demo_tmp \
    --size 512

# 2. 导出为ROS bag
python export_to_rosbag.py \
    --output_dir demo_tmp \
    --bag_file westlake.bag \
    --frame_rate 30.0

# 3. 播放bag文件
rosbag play westlake.bag

# 4. 在RViz中可视化
rviz
```

