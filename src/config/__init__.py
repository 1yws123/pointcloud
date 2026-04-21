import argparse
import os
import sys


DEFAULT_MODEL_CONFIG = {
    "latent_dim": 128,
    "plane_resolution": 128,
    "plane_features": 32,
    "num_fourier_freqs": 8,
    # 对齐 Hunyuan3D 体量（2048 + 2048 + 4096 = 8192），细节分支占 50%
    "num_points_uniform": 2048,
    "num_points_curvature": 2048,
    "num_points_importance": 4096,
    "dropout": 0.1,
}


def model_config_dict():
    """返回模型结构默认配置的浅拷贝。

    注意：如果调用方使用了 YAML config，应通过
    `_apply_model_overrides_from_yaml` 把 yaml.model.* 覆盖进去。
    """
    return dict(DEFAULT_MODEL_CONFIG)


# --------------------------------------------------------------------
# YAML 加载（可选依赖 PyYAML；若没装就抛错但不影响不用 --config 的情况）
# --------------------------------------------------------------------
def _load_yaml(path):
    try:
        import yaml  # type: ignore
    except ImportError:
        print(
            "[config] 检测到 --config 参数，但当前环境没有安装 PyYAML。"
            "请 `pip install pyyaml` 或去掉 --config。",
            file=sys.stderr,
        )
        raise
    with open(path, "r") as f:
        return yaml.safe_load(f) or {}


def _apply_yaml_to_args(parser, argv, yaml_path):
    """把 YAML 的顶层键作为 argparse 默认值覆盖进去；CLI 参数仍然是最高优先级。"""
    cfg = _load_yaml(yaml_path)
    # 只挑 argparse 已经定义过的顶层键，避免把 model.* 这类结构化键误当成 CLI 参数
    valid_keys = {a.dest for a in parser._actions if a.dest != "help"}
    overrides = {k: v for k, v in cfg.items() if k in valid_keys}
    parser.set_defaults(**overrides)
    return cfg


def _apply_model_overrides(model_cfg, yaml_cfg):
    """把 yaml 里的 `model:` 子项叠加到 model_cfg 上。"""
    if not yaml_cfg:
        return model_cfg
    m = yaml_cfg.get("model", {}) or {}
    for k, v in m.items():
        if k in model_cfg:
            model_cfg[k] = v
        else:
            model_cfg[k] = v  # 允许新增字段
    return model_cfg


# --------------------------------------------------------------------
# 公共：让 train/eval 两个入口都能用 --config 指定 YAML
# --------------------------------------------------------------------
def _add_config_arg(parser):
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML 配置文件路径（会覆盖 argparse 默认值；CLI 优先级最高）。",
    )


def _maybe_load_yaml(parser, argv=None):
    """解析 --config 并把 yaml 映射到 argparse 默认值。返回 yaml_cfg（dict 或 {}）。"""
    argv = sys.argv[1:] if argv is None else argv
    # 轻量 pre-parse，只抓 --config
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=str, default=None)
    pre_args, _ = pre.parse_known_args(argv)
    if pre_args.config and os.path.exists(pre_args.config):
        return _apply_yaml_to_args(parser, argv, pre_args.config)
    if pre_args.config and not os.path.exists(pre_args.config):
        print(f"[config] 警告：--config 指定的 {pre_args.config} 不存在，继续使用 argparse 默认。",
              file=sys.stderr)
    return {}


def build_train_arg_parser():
    parser = argparse.ArgumentParser(description="End-to-end aero training with structured trainer/evaluator modules")
    _add_config_arg(parser)
    parser.add_argument("--pc_root", type=str, default="/home/yuwenshi/B737/B737_4594/pc1")
    parser.add_argument("--aero_root", type=str, default="/home/yuwenshi/B737/G58_4594_aero")
    parser.add_argument("--sdf_dir", type=str, default="/home/yuwenshi/B737/B737_4594/sdf")
    parser.add_argument("--save_dir", type=str, default="/home/yuwenshi/B737/model_Aero/encoder_4594/checkpoints_6")
    parser.add_argument("--epochs", type=int, default=1500)
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--eval_interval", type=int, default=10)
    parser.add_argument("--plot_interval", type=int, default=50)
    parser.add_argument("--checkpoint_interval", type=int, default=200)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--train_gpu", type=str, default="4")
    # 正则与早停
    parser.add_argument("--kl_beta", type=float, default=1e-4,
                        help="KL 项系数，让 VAE 的 logvar 真正起正则作用")
    parser.add_argument("--jitter_std", type=float, default=1e-3,
                        help="训练时点云 jitter 强度；原来 1e-4 太小形同虚设")
    parser.add_argument("--early_stop_patience", type=int, default=30,
                        help="val_mae 连续这么多次评估无改善则停止；0 = 关闭")
    # 数据增强
    parser.add_argument("--mirror_prob", type=float, default=0.5,
                        help="左右镜像概率（XZ 对称飞机，β=0 直飞下保物理）")
    parser.add_argument("--point_dropout", type=float, default=0.0,
                        help="每个点被随机 drop 的概率，用随机已保留点补位")

    _maybe_load_yaml(parser)
    return parser


def build_eval_arg_parser():
    parser = argparse.ArgumentParser(description="Offline evaluator for aero checkpoints")
    _add_config_arg(parser)
    parser.add_argument("--pc_root", type=str, default="/home/yuwenshi/B737/B737_4594/pc1")
    parser.add_argument("--aero_root", type=str, default="/home/yuwenshi/B737/G58_4594_aero")
    parser.add_argument("--sdf_dir", type=str, default="/home/yuwenshi/B737/B737_4594/sdf")
    parser.add_argument("--ckpt_path", type=str, default="/home/yuwenshi/B737/model_Aero/encoder_4594/checkpoints_6/best_cl_model.pth")
    parser.add_argument("--save_dir", type=str, default="/home/yuwenshi/B737/model_Aero/encoder_4594/checkpoints_6")
    parser.add_argument("--batch_size", type=int, default=20)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--eval_gpu", type=str, default="5")

    _maybe_load_yaml(parser)
    return parser


def load_yaml_model_overrides(yaml_path=None):
    """辅助：让 train.py 能从 yaml 把 model.* 覆盖合并进 model_config_dict()。"""
    if yaml_path is None or not os.path.exists(yaml_path):
        return {}
    cfg = _load_yaml(yaml_path)
    return cfg.get("model", {}) or {}
