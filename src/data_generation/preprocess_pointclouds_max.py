import trimesh
import numpy as np
import os
import glob
from tqdm import tqdm
import pathlib
import open3d as o3d
from concurrent.futures import ProcessPoolExecutor, as_completed

POOL_SIZE_UNIFORM = 30000
POOL_SIZE_CURVATURE = 30000
POOL_SIZE_IMPORTANCE = 30000
IMPORTANCE_CANDIDATES = 100000


def fps_downsample(points, num_required):
    if points.shape[0] == 0:
        return np.zeros((num_required, 3), dtype=np.float32)
    if points.shape[0] <= num_required:
        indices = np.random.choice(points.shape[0], num_required, replace=True)
        return points[indices]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd_down = pcd.farthest_point_down_sample(num_required)
    return np.asarray(pcd_down.points)


def random_subsample(points, num_required):
    if points.shape[0] == 0:
        return np.zeros((num_required, 3), dtype=np.float32)
    if points.shape[0] <= num_required:
        indices = np.random.choice(points.shape[0], num_required, replace=True)
        return points[indices]
    indices = np.random.choice(points.shape[0], num_required, replace=False)
    return points[indices]


def process_model(model_path, output_root_dir,
                  pool_uniform=POOL_SIZE_UNIFORM,
                  pool_curvature=POOL_SIZE_CURVATURE,
                  pool_importance=POOL_SIZE_IMPORTANCE,
                  subdivision_iterations=2,
                  sharp_edge_angle_threshold_deg=15.0):
    try:
        p = pathlib.Path(model_path)
        model_id = p.stem

        sdf_npz_path = os.path.join(output_root_dir, 'sdf', f"{model_id}.npz")
        shifts, scale = None, None
        if os.path.exists(sdf_npz_path):
            sdf_data = np.load(sdf_npz_path)
            shifts = sdf_data['shifts']
            scale = sdf_data['scale']
        else:
            print(f"未找到 SDF，采用本地归一化: {sdf_npz_path}")

        mesh = trimesh.load(model_path, force='mesh')
        trimesh.repair.fix_normals(mesh)

        v = mesh.vertices
        if shifts is not None and scale is not None:
            v = v - shifts
            v = v * scale
        else:
            shifts = (v.max(axis=0) + v.min(axis=0)) / 2
            v = v - shifts
            scale = (1 / np.abs(v).max()) * 0.9
            v = v * scale
        mesh.vertices = v

        # --- A. Curvature pool：取 top-25% 高曲率顶点（带法向） ---
        mesh_for_curvature = mesh.copy()
        if len(mesh_for_curvature.faces) > 100000:
            o3d_mesh = o3d.geometry.TriangleMesh()
            o3d_mesh.vertices = o3d.utility.Vector3dVector(mesh_for_curvature.vertices)
            o3d_mesh.triangles = o3d.utility.Vector3iVector(mesh_for_curvature.faces)
            o3d_mesh = o3d_mesh.simplify_quadric_decimation(target_number_of_triangles=100000)
            mesh_for_curvature = trimesh.Trimesh(
                vertices=np.asarray(o3d_mesh.vertices),
                faces=np.asarray(o3d_mesh.triangles)
            )
            trimesh.repair.fix_normals(mesh_for_curvature)

        radius = 0.02
        curvatures = trimesh.curvature.discrete_mean_curvature_measure(
            mesh_for_curvature, mesh_for_curvature.vertices, radius
        )
        curvature_score = np.maximum(curvatures, 0)
        sorted_indices = np.argsort(curvature_score)[::-1]
        split_idx = max(int(len(sorted_indices) * 0.25), 1)
        top_idx = sorted_indices[:split_idx]
        curvature_pool_raw = mesh_for_curvature.vertices[top_idx]
        curvature_normals_raw = mesh_for_curvature.vertex_normals[top_idx]
        # cap
        if curvature_pool_raw.shape[0] > pool_curvature:
            idx = np.random.choice(curvature_pool_raw.shape[0], pool_curvature, replace=False)
            curvature_pool_pts = curvature_pool_raw[idx]
            curvature_pool_nrm = curvature_normals_raw[idx]
        else:
            curvature_pool_pts = curvature_pool_raw
            curvature_pool_nrm = curvature_normals_raw

        # --- B. Uniform pool：细分网格顶点（带法向） ---
        subdivided_mesh = mesh.copy()
        for _ in range(subdivision_iterations):
            subdivided_mesh = subdivided_mesh.subdivide()
        trimesh.repair.fix_normals(subdivided_mesh)
        uniform_verts = subdivided_mesh.vertices
        uniform_norms = subdivided_mesh.vertex_normals
        if uniform_verts.shape[0] > pool_uniform:
            idx = np.random.choice(uniform_verts.shape[0], pool_uniform, replace=False)
            uniform_pool_pts = uniform_verts[idx]
            uniform_pool_nrm = uniform_norms[idx]
        else:
            uniform_pool_pts = uniform_verts
            uniform_pool_nrm = uniform_norms

        # --- C. Importance pool：沿锐边线性插值（带法向） ---
        angle_rad = np.deg2rad(sharp_edge_angle_threshold_deg)
        edge_angles = mesh.face_adjacency_angles
        sharp_edge_indices = mesh.face_adjacency_edges[edge_angles > angle_rad]

        if len(sharp_edge_indices) > 0:
            sharp_lines = mesh.vertices[sharp_edge_indices]           # [E, 2, 3]
            sharp_line_normals = mesh.vertex_normals[sharp_edge_indices]  # [E, 2, 3]
            line_indices = np.random.randint(0, len(sharp_lines), size=IMPORTANCE_CANDIDATES)
            t = np.random.random(size=(IMPORTANCE_CANDIDATES, 1)).astype(np.float32)
            importance_pool_raw = (1 - t) * sharp_lines[line_indices, 0] + t * sharp_lines[line_indices, 1]
            importance_normals_raw = (1 - t) * sharp_line_normals[line_indices, 0] + t * sharp_line_normals[line_indices, 1]
            nrm_len = np.linalg.norm(importance_normals_raw, axis=-1, keepdims=True) + 1e-8
            importance_normals_raw = importance_normals_raw / nrm_len
        else:
            importance_pool_raw = uniform_pool_pts
            importance_normals_raw = uniform_pool_nrm

        if importance_pool_raw.shape[0] > pool_importance:
            idx = np.random.choice(importance_pool_raw.shape[0], pool_importance, replace=False)
            importance_pool_pts = importance_pool_raw[idx]
            importance_pool_nrm = importance_normals_raw[idx]
        else:
            importance_pool_pts = importance_pool_raw
            importance_pool_nrm = importance_normals_raw

        # --- 保存 ---
        final_output_dir = os.path.join(output_root_dir, 'pc1')
        os.makedirs(final_output_dir, exist_ok=True)
        output_path = os.path.join(final_output_dir, f"{model_id}_pc.npz")
        np.savez(
            output_path,
            # --- 新版字段（动态采样） ---
            uniform_pool=uniform_pool_pts.astype(np.float32),
            uniform_pool_nrm=uniform_pool_nrm.astype(np.float32),
            curvature_pool=curvature_pool_pts.astype(np.float32),
            curvature_pool_nrm=curvature_pool_nrm.astype(np.float32),
            importance_pool=importance_pool_pts.astype(np.float32),
            importance_pool_nrm=importance_pool_nrm.astype(np.float32),
            # --- 兼容字段（老代码依赖，下采样到 4000 兜底） ---
            uniform=fps_downsample(uniform_pool_pts, 4000).astype(np.float32),
            curvature=fps_downsample(curvature_pool_pts, 4000).astype(np.float32),
            importance=fps_downsample(importance_pool_pts, 4000).astype(np.float32),
            shift=np.asarray(shifts, dtype=np.float32),
            scale=np.array([scale], dtype=np.float32),
        )
        return True
    except Exception as e:
        return f"处理 {os.path.basename(model_path)} 失败: {e}"


if __name__ == '__main__':
    INPUT_MESH_DIR = r"/home/yuwenshi/B737/B737_4594/mesh"
    OUTPUT_PC_ROOT = r"/home/yuwenshi/B737/B737_4594"

    model_files = glob.glob(os.path.join(INPUT_MESH_DIR, "*.obj"))
    if not model_files:
        print(f"错误: 在 '{INPUT_MESH_DIR}' 未找到任何 .obj 文件")
    else:
        final_output_dir = os.path.join(OUTPUT_PC_ROOT, 'pc1')
        os.makedirs(final_output_dir, exist_ok=True)

        files_to_process = []
        for model_path in model_files:
            model_id = pathlib.Path(model_path).stem
            output_path = os.path.join(final_output_dir, f"{model_id}_pc.npz")
            if not os.path.exists(output_path):
                files_to_process.append(model_path)
            else:
                # 若已有文件但缺少新字段，也需要重新处理
                try:
                    with np.load(output_path) as d:
                        if 'uniform_pool' not in d.files:
                            files_to_process.append(model_path)
                except Exception:
                    files_to_process.append(model_path)

        print(f"总计找到 {len(model_files)} 个文件。")
        print(f"已处理 {len(model_files) - len(files_to_process)} 个，剩余 {len(files_to_process)} 个待处理...")

        if len(files_to_process) > 0:
            max_workers = 16
            print(f"启动多进程加速... {max_workers} 核")
            with ProcessPoolExecutor(max_workers=max_workers) as executor:
                futures = [executor.submit(process_model, path, OUTPUT_PC_ROOT) for path in files_to_process]
                for future in tqdm(as_completed(futures), total=len(futures), desc="Batch Processing"):
                    res = future.result()
                    if res is not True:
                        tqdm.write(str(res))
            print(f"\n完成！输出: {final_output_dir}")
        else:
            print("\n全部已处理。")
