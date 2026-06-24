import transformers
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List


@dataclass
class ModelArguments:
    model_name_or_path: Optional[str] = field(default="Qwen/Qwen2.5-VL-3B-Instruct")
    tune_mm_llm: bool = field(default=False)
    tune_mm_mlp: bool = field(default=False)
    tune_mm_vision: bool = field(default=False)

@dataclass
class DataArguments:
    dataset_use: str = field(default="")
    robotwin_data_root: Optional[str] = field(
        default=None,
        metadata={"help": "Root path of the RobotWin LeRobot dataset."},
    )
    robotwin_test_ratio: float = field(
        default=0.05,
        metadata={"help": "Fraction of RobotWin task directories held out for test/eval."},
    )
    robotwin_split_seed: int = field(
        default=0,
        metadata={"help": "Seed for deterministic RobotWin task-level train/test split."},
    )
    robotwin_q2_frame_stride: int = field(
        default=8,
        metadata={"help": "Frame stride for regular RobotWin Q2 samples."},
    )
    robotwin_boundary_extra_frames: int = field(
        default=2,
        metadata={"help": "Deprecated compatibility option; RobotWin Q2 done frames are fixed to three-frame windows."},
    )
    robotwin_done_sample_prob: float = field(
        default=0.4,
        metadata={"help": "Probability of sampling RobotWin Q2 done examples during training."},
    )
    data_flatten: bool = field(default=False)
    data_packing: bool = field(default=False)
    base_interval: int = field(default=2)
    max_pixels: int = field(default=28 * 28 * 576)
    min_pixels: int = field(default=28 * 28 * 16)
    video_max_frames: Optional[int] = field(default=8)
    video_min_frames: Optional[int] = field(default=4)
    video_max_pixels: int = field(default=1024 * 28 * 28)
    video_min_pixels: int = field(default=256 * 28 * 28)
    video_fps: float = 2


@dataclass
class TrainingArguments(transformers.TrainingArguments):
    cache_dir: Optional[str] = field(default=None)
    optim: str = field(default="adamw_torch")
    model_max_length: int = field(
        default=512,
        metadata={
            "help": "Maximum sequence length. Sequences will be right padded (and possibly truncated)."
        },
    )
    mm_projector_lr: Optional[float] = None
    vision_tower_lr: Optional[float] = None

    ## Lora config
    lora_enable: bool = field(default=False)
    lora_r: int = field(default=64)
    lora_alpha: int = field(default=128)
    lora_dropout: float = field(default=0.0)

    ## RobotWin regression loss weights
    robotwin_done_loss_weight: float = field(default=1.0)
    robotwin_progress_loss_weight: float = field(default=1.0)
    robotwin_replan_loss_weight: float = field(default=0.0)
    robotwin_incident_loss_weight: float = field(default=0.0)
    robotwin_train_query_embeddings: bool = field(default=True)
    robotwin_init_checkpoint: Optional[str] = field(
        default=None,
        metadata={"help": "Optional RobotWin wrapper checkpoint (.bin or directory) to initialize from before training."},
    )
