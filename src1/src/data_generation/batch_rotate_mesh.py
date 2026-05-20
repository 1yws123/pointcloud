import open3d as o3d
import numpy as np
import os
import glob
from tqdm import tqdm
import math

def batch_rotate_meshes(input_dir, output_dir, rotation_axis='x', angle_deg=-90):
    """
    批量旋转文件夹下的所有 .obj 文件，并保存到新目录。
    
    :param input_dir: 原始 mesh 所在的目录
    :param output_dir: 旋转后 mesh 保存的目录
    :param rotation_axis: 旋转轴 ('x', 'y', 'z')
    :param angle_deg: 旋转角度（度）
    """
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)

    # 获取所有 obj 文件
    mesh_files = glob.glob(os.path.join(input_dir, "*.obj"))
    
    if not mesh_files:
        print(f"❌ 错误: 在目录 {input_dir} 中没有找到任何 .obj 文件！")
        return

    print(f"找到 {len(mesh_files)} 个 .obj 文件，准备处理...")
    
    # 转换角度为弧度
    angle_rad = np.radians(angle_deg)
    
    # 根据指定的轴创建旋转参数 (x, y, z)
    if rotation_axis.lower() == 'x':
        rot_xyz = (angle_rad, 0, 0)
    elif rotation_axis.lower() == 'y':
        rot_xyz = (0, angle_rad, 0)
    elif rotation_axis.lower() == 'z':
        rot_xyz = (0, 0, angle_rad)
    else:
        raise ValueError("rotation_axis 必须是 'x', 'y' 或 'z'")

    # 获取一个基础网格用来生成旋转矩阵
    # Open3D 的 get_rotation_matrix_from_xyz 是一个类方法或基于对象的，我们可以随便建一个空网格调用它
    dummy_mesh = o3d.geometry.TriangleMesh()
    R = dummy_mesh.get_rotation_matrix_from_xyz(rot_xyz)

    success_count = 0
    fail_count = 0

    # 使用 tqdm 显示进度条
    for filepath in tqdm(mesh_files, desc="正在旋转网格"):
        filename = os.path.basename(filepath)
        out_path = os.path.join(output_dir, filename)
        
        try:
            # 1. 加载 Mesh
            mesh = o3d.io.read_triangle_mesh(filepath)
            
            if not mesh.has_vertices():
                print(f"\n⚠️ 警告: {filename} 是空的或读取失败，跳过。")
                fail_count += 1
                continue
                
            # 2. 应用旋转 (以原点 0,0,0 为中心旋转)
            mesh.rotate(R, center=(0, 0, 0))
            
            # (可选) 重新计算法线，旋转后法线方向也需要更新
            mesh.compute_vertex_normals()
            
            # 3. 保存旋转后的 Mesh
            # write_triangle_mesh 默认会保存为 obj 格式
            o3d.io.write_triangle_mesh(out_path, mesh)
            
            success_count += 1
            
        except Exception as e:
            print(f"\n❌ 错误: 处理 {filename} 时发生异常 - {e}")
            fail_count += 1

    print("\n" + "="*40)
    print("🎉 批量处理完成！")
    print(f"总计: {len(mesh_files)} 个文件")
    print(f"成功: {success_count} 个")
    print(f"失败: {fail_count} 个")
    print(f"输出目录: {output_dir}")
    print("="*40)

if __name__ == "__main__":
    # 配置路径
    INPUT_DIR = "/home/yuwenshi/B737/G58_4594_mesh"
    OUTPUT_DIR = "/home/yuwenshi/B737/B737_4594/mesh"
    
    # 执行批量旋转 (绕 X 轴旋转 90 度)
    batch_rotate_meshes(INPUT_DIR, OUTPUT_DIR, rotation_axis='x', angle_deg=90)