from pathlib import Path
import os

ROOT = Path(__file__).resolve().parent

def _p(*parts):
    return str(ROOT.joinpath(*parts))

DATA = {
    "gan_train": "/path/to/gan_and_segmenter_trian_dataset",
    "gan_val": "/path/to/validation_dataset",
    "gan_test": "/path/to/gan_and_segmentation_test_dataset" ,
    "train_new": "/path/to/train_segemtation_dataset",
}

PATHS = {
    "weight_dir": _p("weight"),
    "gan_run_dir": _p("runs", "wgan_gp_brats"),
    "gan_images_dir": _p("runs", "wgan_gp_brats", "images"),
    "gan_ckpt_dir": _p("runs", "wgan_gp_brats", "ckpt"),
    "generated_dir": _p("generated_multi_proc"),
    "pre_unet_latest": _p("weight", "unet3d_latest.pth"),
    "pre_unet_best": _p("weight", "unet3d_best.pth"),
    "unet_binary": _p("weight", "unet3d_brats_binary.pth"),
    "unet_binary_t2": _p("weight", "unet3d_brats_binary_t2_only.pth"),
    "gan_resume": _p("runs", "wgan_gp_brats", "ckpt", "latest.pt"),
    "gan_test_ckpt": os.environ.get("GAN_TEST_CKPT", _p("runs", "wgan_gp_brats", "ckpt", "epoch_210.pt")),
}

TRAIN_PRE_UNET = {
    "seed": 10,
    "batch_size": 8,
    "num_workers": 4,
    "epochs": 50,
    "lr": 1e-4,
    "print_every": 1,
    "crop_d": 96,
    "crop_h": 128,
    "crop_w": 128,
    "tumour_crop_prob": 0.75,
    "w_dice": 1.0,
    "w_bce": 0.5,
    "base_ch": 32,
    "cuda_alloc_conf": "max_split_size_mb:128",
    "cudnn_benchmark": False,
}

TRAIN_GAN = {
    "seed": 10,
    "batch_size": 2,
    "num_workers": 4,
    "val_num_workers": 4,
    "epochs": 210,
    "print_every": 1,
    "n_critic": 3,
    "target_key": "t1n",
    "src_key": "t2w",
    "lr_g": 1e-4,
    "lr_d": 1e-4,
    "betas": (0.0, 0.9),
    "lmb": {
        "REC": 29.80156504981695,
        "SSIM": 13.786615122945529,
        "PERC": 1.7759984790312648,
        "SEG": 1.0420501798159252,
        "CLS": 0.461275942713654,
        "GP": 10.0,
    },
    "resume": True,
    "generator_in_channels": 4,
    "generator_out_channels": 1,
    "generator_base_channels": 64,
    "critic_in_channels": 2,
    "critic_base_channels": 64,
    "critic_num_classes": 3,
    "seg_in_dim": 1,
    "seg_c_dim": 3,
    "seg_base_ch": 32,
    "seg_out_dim": 1,
    "pretrained_seg_ckpt": PATHS["pre_unet_best"],
    "perc_hub_repo": "zaidau/MedicalNet-models",
    "perc_hub_model": "medicalnet_resnet50",
    "perc_feat_l2norm": False,
    "perc_criterion": "l1",
    "perc_feature_mask": True,
    "perc_mask_thresh01": 0.02,
    "perc_mask_dilate_k": 5,
    "lr_milestones": (80, 140, 200),
    "use_amp": False,
    "cudnn_benchmark": True,
    "val_max_cases": 8,
    "val_label_index": 2,
}

TRAIN_UNET = {
    "seed": 10,
    "batch_size": 8,
    "num_workers": 4,
    "lr": 1e-3,
    "epochs": 100,
    "model_path": PATHS["unet_binary"],
}

TRAIN_UNET_T2 = {
    "seed": 10,
    "batch_size": 8,
    "num_workers": 4,
    "lr": 1e-3,
    "epochs": 100,
    "model_path": PATHS["unet_binary_t2"],
}

TEST_GAN = {
    "seed": 10,
    "batch_size": 1,
    "num_workers": 4,
    "checkpoint_path": PATHS["gan_test_ckpt"],
    "output_dir": PATHS["generated_dir"],
    "domains": {"flair": 0, "t1c": 1, "t1": 2},
    "targets": ("flair", "t1c", "t1"),
    "real_key": {"flair": "t2f", "t1c": "t1c", "t1": "t1n"},
    "brain_mask_threshold": 0.01,
    "generator_in_channels": 4,
    "generator_out_channels": 1,
    "generator_base_channels": 64,
}

TEST_UNET = {
    "seed": 10,
    "batch_size": 1,
    "num_workers": 4,
    "model_path": PATHS["unet_binary"],
    "tumour_min_voxels": 0,
    "count_positive_labels_only": True,
}

TEST_UNET_T2 = {
    "seed": 10,
    "batch_size": 1,
    "num_workers": 4,
    "model_path": PATHS["unet_binary_t2"],
}

ASSETS = {
    "gif_dir": _p("assets", "gifs"),
    "figure_dir": _p("assets", "figures"),
    "gif_file": _p("assets", "gifs", "sampling_example.gif"),
    "framework_figure": _p("assets", "figures", "framework.png"),
    "generator_figure": _p("assets", "figures", "generator_architecture.png"),
    "attention_figure": _p("assets", "figures", "memory_bounded_hybrid_attention_block.png"),
    "comparison_figure": _p("assets", "figures", "comparison_t2_to_t2f_t1n_t1c.png"),
}

def ensure_project_dirs():
    Path(PATHS["weight_dir"]).mkdir(parents=True, exist_ok=True)
    Path(PATHS["gan_run_dir"]).mkdir(parents=True, exist_ok=True)
    Path(PATHS["gan_images_dir"]).mkdir(parents=True, exist_ok=True)
    Path(PATHS["gan_ckpt_dir"]).mkdir(parents=True, exist_ok=True)
    Path(PATHS["generated_dir"]).mkdir(parents=True, exist_ok=True)
    Path(ASSETS["gif_dir"]).mkdir(parents=True, exist_ok=True)
    Path(ASSETS["figure_dir"]).mkdir(parents=True, exist_ok=True)
