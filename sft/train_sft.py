from __future__ import annotations

import argparse
import csv
import hashlib
import json
import logging
import math
import os
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
import yaml
from datasets import Dataset, DatasetDict, load_dataset
from dotenv import load_dotenv
from huggingface_hub import CommitOperationDelete, HfApi, hf_hub_download, list_repo_files, snapshot_download
from huggingface_hub.utils import EntryNotFoundError, HfHubHTTPError
from peft import LoraConfig, PeftModel, TaskType, get_peft_model, prepare_model_for_kbit_training
from torch.utils.data import DataLoader, WeightedRandomSampler
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, Trainer, TrainerCallback, TrainingArguments, set_seed

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SFT_ROOT = Path(__file__).resolve().parent
GRPO_ROOT = PROJECT_ROOT / "grpo"
ENV_PATH = PROJECT_ROOT / ".env"
CONFIG_PATH = SFT_ROOT / "sft_config.yaml"

sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(GRPO_ROOT))

from train_sft_qlora import align_resume_trainer_state, load_dataset_resilient  # noqa: E402
from rewards.common import mask_env, row_to_meter_fields, row_to_prompt, row_to_requested_lines, score_count_adherence  # noqa: E402
from rewards.meter import score_meter_poem  # noqa: E402
from split_utils import MIN_SAFE_STRATIFY_GROUP_SIZE, WEIGHT_ALPHA, annotate_split_columns, build_distribution_report, build_train_weights, split_dataset  # noqa: E402


def str2bool(v: str | bool) -> bool:
    if isinstance(v, bool):
        return v
    value = str(v).strip().lower()
    if value in {"1", "true", "yes", "y", "on"}:
        return True
    if value in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"invalid boolean value: {v}")


def parse_str_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return [str(item).strip() for item in value if str(item).strip()]
    text = str(value).strip()
    if not text:
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def env_or(name: str, default: Any) -> Any:
    value = os.getenv(name)
    return default if value is None else value


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_text(text: str) -> str:
    return hashlib.sha256(str(text).encode("utf-8")).hexdigest()


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def dump_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows


def setup_logging(run_dir: Path, level: str) -> logging.Logger:
    run_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("sft")
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.handlers.clear()

    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")

    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    logger.addHandler(sh)

    fh = logging.FileHandler(run_dir / "train.log", encoding="utf-8")
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger


class JsonlWriter:
    def __init__(self, path: Path, enabled: bool = True):
        self.path = path
        self.enabled = enabled
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        with self.path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


class JsonlCsvMirror:
    def __init__(self, jsonl_path: Path, csv_path: Path, enabled: bool = True):
        self.jsonl = JsonlWriter(jsonl_path, enabled=enabled)
        self.csv_path = csv_path
        self.enabled = enabled
        self.rows: list[dict[str, Any]] = []

    def write(self, payload: dict[str, Any]) -> None:
        if not self.enabled:
            return
        self.rows.append(payload)
        self.jsonl.write(payload)
        fieldnames = sorted({key for row in self.rows for key in row.keys()})
        with self.csv_path.open("w", encoding="utf-8", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            for row in self.rows:
                writer.writerow(row)


@dataclass
class RuntimeConfig:
    mode: str
    run_name: str
    model_repo_id: str
    dataset_id: str
    base_model_id: str
    trust_remote_code: bool
    torch_dtype: str
    model_use_cache: bool
    init_adapter_repo_id: str
    init_adapter_subfolder: str
    init_from_adapter: bool
    init_adapter_path: str
    continuation_namespace: str
    resume_mode: str
    resume_path: str
    output_dir: Path
    seed: int
    push_to_hub: bool
    max_seq_length: int
    learning_rate: float
    weight_decay: float
    warmup_ratio: float
    lr_scheduler_type: str
    max_grad_norm: float
    per_device_train_batch_size: int
    per_device_eval_batch_size: int
    gradient_accumulation_steps: int
    num_train_epochs: float
    max_steps: int
    save_steps: int
    live_eval_steps: int
    full_eval_steps: int
    logging_steps: int
    save_total_limit: int
    remote_keep_last_checkpoints: int
    bf16: str
    fp16: bool
    gradient_checkpointing: bool
    optim: str
    load_best_model_at_end: bool
    metric_for_best_model: str
    greater_is_better: bool
    heartbeat_seconds: int
    jsonl_metrics: bool
    jsonl_events: bool
    debug_dump_first_batch: bool
    use_weighted_sampler: bool
    max_bayts: int
    min_bayts: int
    drop_meters: list[str]
    packing: bool
    completion_only_loss: bool
    truncation_mode: str
    lora_r: int
    lora_alpha: int
    lora_dropout: float
    lora_target_modules: list[str]
    lora_bias: str
    lora_use_rslora: bool
    lora_task_type: str
    load_in_4bit: bool
    bnb_4bit_quant_type: str
    bnb_4bit_use_double_quant: bool
    bnb_4bit_compute_dtype: str
    prepare_model_for_kbit_training: bool
    probe_per_meter: int
    live_eval_per_meter: int
    full_eval_per_meter: int
    oversample_max_multiplier: float
    enable_meter_aux: bool
    meter_aux_lambda: float
    meter_aux_normalize_by_log_num_classes: bool
    probe_max_new_tokens: int
    probe_batch_size: int
    probe_do_sample: bool
    probe_temperature: float
    limit_train_rows: int
    limit_eval_rows: int
    limit_test_rows: int
    plateau_min_full_eval_points: int
    plateau_patience: int
    plateau_full_eval_min_delta: float
    plateau_probe_meter_min_delta: float
    plateau_probe_count_min_delta: float
    plateau_stop_on_plateau: bool
    log_level: str
    hf_home: Path
    hub_cache: Path

    @property
    def eval_steps(self) -> int:
        return int(self.live_eval_steps)


def flatten_for_fingerprint(cfg: RuntimeConfig) -> str:
    payload = {
        "model_repo_id": cfg.model_repo_id,
        "dataset_id": cfg.dataset_id,
        "base_model_id": cfg.base_model_id,
        "trust_remote_code": cfg.trust_remote_code,
        "torch_dtype": cfg.torch_dtype,
        "model_use_cache": cfg.model_use_cache,
        "init_adapter_repo_id": cfg.init_adapter_repo_id,
        "init_adapter_subfolder": cfg.init_adapter_subfolder,
        "init_from_adapter": cfg.init_from_adapter,
        "init_adapter_path": cfg.init_adapter_path,
        "continuation_namespace": cfg.continuation_namespace,
        "max_seq_length": cfg.max_seq_length,
        "learning_rate": cfg.learning_rate,
        "weight_decay": cfg.weight_decay,
        "gradient_accumulation_steps": cfg.gradient_accumulation_steps,
        "optim": cfg.optim,
        "packing": cfg.packing,
        "completion_only_loss": cfg.completion_only_loss,
        "truncation_mode": cfg.truncation_mode,
        "use_weighted_sampler": cfg.use_weighted_sampler,
        "max_bayts": cfg.max_bayts,
        "min_bayts": cfg.min_bayts,
        "drop_meters": cfg.drop_meters,
        "lora_r": cfg.lora_r,
        "lora_alpha": cfg.lora_alpha,
        "lora_dropout": cfg.lora_dropout,
        "lora_target_modules": cfg.lora_target_modules,
        "lora_bias": cfg.lora_bias,
        "lora_use_rslora": cfg.lora_use_rslora,
        "lora_task_type": cfg.lora_task_type,
        "load_in_4bit": cfg.load_in_4bit,
        "bnb_4bit_quant_type": cfg.bnb_4bit_quant_type,
        "bnb_4bit_use_double_quant": cfg.bnb_4bit_use_double_quant,
        "bnb_4bit_compute_dtype": cfg.bnb_4bit_compute_dtype,
        "prepare_model_for_kbit_training": cfg.prepare_model_for_kbit_training,
        "load_best_model_at_end": cfg.load_best_model_at_end,
        "metric_for_best_model": cfg.metric_for_best_model,
        "greater_is_better": cfg.greater_is_better,
        "probe_per_meter": cfg.probe_per_meter,
        "live_eval_per_meter": cfg.live_eval_per_meter,
        "full_eval_per_meter": cfg.full_eval_per_meter,
        "oversample_max_multiplier": cfg.oversample_max_multiplier,
        "enable_meter_aux": cfg.enable_meter_aux,
        "meter_aux_lambda": cfg.meter_aux_lambda,
        "meter_aux_normalize_by_log_num_classes": cfg.meter_aux_normalize_by_log_num_classes,
        "plateau_patience": cfg.plateau_patience,
    }
    return sha256_text(json.dumps(payload, sort_keys=True))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Continuation SFT workspace")
    parser.add_argument("--mode", choices=["train", "sanity"], default="train")
    parser.add_argument("--resume", choices=["auto", "fresh", "from_path"], default=os.getenv("SFT_RESUME_MODE", "auto"))
    parser.add_argument("--resume_path", default=os.getenv("SFT_RESUME_PATH", ""))
    parser.add_argument("--run_name", default="")
    parser.add_argument("--config", default=str(CONFIG_PATH))
    return parser.parse_args()


def build_runtime_config(args: argparse.Namespace) -> RuntimeConfig:
    load_dotenv(ENV_PATH, override=False)
    raw = load_yaml(Path(args.config))
    defaults = raw.get("defaults", {})
    model = raw.get("model", {})
    quantization = raw.get("quantization", {})
    optimizer = raw.get("optimizer", {})
    training = raw.get("training", {})
    mode_overrides = raw.get(args.mode, {})
    splits = raw.get("splits", {})
    plateau = raw.get("plateau", {})
    probe = raw.get("probe_generation", {})
    data = raw.get("data", {})
    lora = raw.get("lora", {})
    meter_aux = raw.get("meter_aux", {})
    output_root = SFT_ROOT / "outputs" / ("sanity_check" if args.mode == "sanity" else "train")
    run_name = args.run_name.strip() or f"{args.mode}_{datetime.now(timezone.utc).strftime('%Y%m%d_%H%M%S')}"

    hf_home = Path(str(defaults.get("hf_home", os.getenv("HF_HOME", str(SFT_ROOT / ".hf_cache"))))).resolve()
    hub_cache = Path(str(defaults.get("hub_cache", os.getenv("HF_HUB_CACHE", str(hf_home / "hub"))))).resolve()

    return RuntimeConfig(
        mode=args.mode,
        run_name=run_name,
        model_repo_id=str(env_or("SFT_MODEL_REPO_ID", defaults.get("model_repo_id", os.getenv("SFT_ADAPTER_REPO", "Shaer-AI/yehia-sft-qlora")))),
        dataset_id=str(env_or("SFT_DATASET_ID", defaults.get("dataset_id", "Shaer-AI/ashaar-with-descriptions-baseform-final-trimmed"))),
        base_model_id=str(env_or("SFT_BASE_MODEL_ID", model.get("base_model", defaults.get("base_model_id", os.getenv("BASE_MODEL_ID", "Navid-AI/Yehia-7B-preview"))))),
        trust_remote_code=str2bool(env_or("SFT_TRUST_REMOTE_CODE", model.get("trust_remote_code", True))),
        torch_dtype=str(env_or("SFT_TORCH_DTYPE", model.get("torch_dtype", "bfloat16"))),
        model_use_cache=str2bool(env_or("SFT_MODEL_USE_CACHE", model.get("use_cache", False))),
        init_adapter_repo_id=str(env_or("SFT_INIT_ADAPTER_REPO_ID", defaults.get("init_adapter_repo_id", os.getenv("SFT_ADAPTER_REPO", "Shaer-AI/yehia-sft-qlora")))),
        init_adapter_subfolder=str(env_or("SFT_INIT_ADAPTER_SUBFOLDER", defaults.get("init_adapter_subfolder", "adapters/full/latest"))),
        init_from_adapter=str2bool(env_or("SFT_INIT_FROM_ADAPTER", mode_overrides.get("init_from_adapter", defaults.get("init_from_adapter", True)))),
        init_adapter_path=str(env_or("SFT_INIT_ADAPTER_PATH", mode_overrides.get("init_adapter_path", defaults.get("init_adapter_path", "")))),
        continuation_namespace=str(env_or("SFT_CONTINUATION_NAMESPACE", defaults.get("continuation_namespace", "continuation"))),
        resume_mode=args.resume,
        resume_path=args.resume_path,
        output_dir=(output_root / run_name).resolve(),
        seed=int(defaults.get("seed", 42)),
        push_to_hub=str2bool(env_or("SFT_PUSH_TO_HUB", defaults.get("push_to_hub", True))),
        max_seq_length=int(data.get("max_length", defaults.get("max_seq_length", 2048))),
        learning_rate=float(env_or("SFT_LEARNING_RATE", mode_overrides.get("learning_rate", training.get("learning_rate", optimizer.get("learning_rate", defaults.get("learning_rate", 2e-4)))))),
        weight_decay=float(optimizer.get("weight_decay", defaults.get("weight_decay", 0.01))),
        warmup_ratio=float(optimizer.get("warmup_ratio", defaults.get("warmup_ratio", 0.03))),
        lr_scheduler_type=str(optimizer.get("lr_scheduler_type", defaults.get("lr_scheduler_type", "cosine"))),
        max_grad_norm=float(optimizer.get("max_grad_norm", defaults.get("max_grad_norm", 1.0))),
        per_device_train_batch_size=int(mode_overrides.get("per_device_train_batch_size", training.get("per_device_train_batch_size", defaults.get("per_device_train_batch_size", 1)))),
        per_device_eval_batch_size=int(mode_overrides.get("per_device_eval_batch_size", training.get("per_device_eval_batch_size", defaults.get("per_device_eval_batch_size", 1)))),
        gradient_accumulation_steps=int(mode_overrides.get("gradient_accumulation_steps", training.get("gradient_accumulation_steps", defaults.get("gradient_accumulation_steps", 8)))),
        num_train_epochs=float(env_or("SFT_NUM_TRAIN_EPOCHS", mode_overrides.get("num_train_epochs", training.get("num_train_epochs", defaults.get("num_train_epochs", 1.0))))),
        max_steps=int(env_or("SFT_MAX_STEPS", mode_overrides.get("max_steps", defaults.get("max_steps", 1000)))),
        save_steps=int(env_or("SFT_SAVE_STEPS", mode_overrides.get("save_steps", training.get("save_steps", defaults.get("save_steps", 50))))),
        live_eval_steps=int(env_or("SFT_LIVE_EVAL_STEPS", mode_overrides.get("live_eval_steps", training.get("eval_steps", defaults.get("live_eval_steps", 50))))),
        full_eval_steps=int(env_or("SFT_FULL_EVAL_STEPS", mode_overrides.get("full_eval_steps", training.get("eval_steps", defaults.get("full_eval_steps", 100))))),
        logging_steps=int(mode_overrides.get("logging_steps", training.get("logging_steps", defaults.get("logging_steps", 1)))),
        save_total_limit=int(env_or("SFT_SAVE_TOTAL_LIMIT", mode_overrides.get("save_total_limit", training.get("save_total_limit", defaults.get("save_total_limit", 6))))),
        remote_keep_last_checkpoints=int(defaults.get("remote_keep_last_checkpoints", 12)),
        bf16=str(env_or("SFT_BF16", training.get("bf16", defaults.get("bf16", "auto")))).lower(),
        fp16=str2bool(env_or("SFT_FP16", training.get("fp16", False))),
        gradient_checkpointing=str2bool(env_or("SFT_GRADIENT_CHECKPOINTING", model.get("gradient_checkpointing", defaults.get("gradient_checkpointing", True)))),
        optim=str(env_or("SFT_OPTIM", optimizer.get("optim", "paged_adamw_8bit"))),
        load_best_model_at_end=str2bool(training.get("load_best_model_at_end", False)),
        metric_for_best_model=str(training.get("metric_for_best_model", "eval_loss")),
        greater_is_better=str2bool(training.get("greater_is_better", False)),
        heartbeat_seconds=int(defaults.get("heartbeat_seconds", 30)),
        jsonl_metrics=str2bool(defaults.get("jsonl_metrics", True)),
        jsonl_events=str2bool(defaults.get("jsonl_events", True)),
        debug_dump_first_batch=str2bool(defaults.get("debug_dump_first_batch", True)),
        use_weighted_sampler=str2bool(env_or("SFT_USE_WEIGHTED_SAMPLER", mode_overrides.get("use_weighted_sampler", defaults.get("use_weighted_sampler", False)))),
        max_bayts=int(env_or("SFT_MAX_BAYTS", mode_overrides.get("max_bayts", data.get("max_bayts", 0)))),
        min_bayts=int(env_or("SFT_MIN_BAYTS", mode_overrides.get("min_bayts", data.get("min_bayts", 0)))),
        drop_meters=parse_str_list(env_or("SFT_DROP_METERS", mode_overrides.get("drop_meters", data.get("drop_meters", [])))),
        packing=str2bool(data.get("packing", False)),
        completion_only_loss=str2bool(data.get("completion_only_loss", True)),
        truncation_mode=str(data.get("truncation_mode", "keep_start")),
        lora_r=int(env_or("SFT_LORA_R", lora.get("r", 64))),
        lora_alpha=int(env_or("SFT_LORA_ALPHA", lora.get("alpha", 128))),
        lora_dropout=float(env_or("SFT_LORA_DROPOUT", lora.get("dropout", 0.05))),
        lora_target_modules=parse_str_list(env_or("SFT_LORA_TARGET_MODULES", lora.get("target_modules", ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"]))),
        lora_bias=str(lora.get("bias", "none")),
        lora_use_rslora=str2bool(lora.get("use_rslora", False)),
        lora_task_type=str(lora.get("task_type", "CAUSAL_LM")),
        load_in_4bit=str2bool(env_or("SFT_LOAD_IN_4BIT", quantization.get("load_in_4bit", True))),
        bnb_4bit_quant_type=str(env_or("SFT_BNB_4BIT_QUANT_TYPE", quantization.get("bnb_4bit_quant_type", "nf4"))),
        bnb_4bit_use_double_quant=str2bool(env_or("SFT_BNB_4BIT_USE_DOUBLE_QUANT", quantization.get("bnb_4bit_use_double_quant", True))),
        bnb_4bit_compute_dtype=str(env_or("SFT_BNB_4BIT_COMPUTE_DTYPE", quantization.get("bnb_4bit_compute_dtype", "bfloat16"))),
        prepare_model_for_kbit_training=str2bool(env_or("SFT_PREPARE_MODEL_FOR_KBIT_TRAINING", quantization.get("prepare_model_for_kbit_training", True))),
        probe_per_meter=int(splits.get("probe_per_meter", 2)),
        live_eval_per_meter=int(splits.get("live_eval_per_meter", 4)),
        full_eval_per_meter=int(splits.get("full_eval_per_meter", 8)),
        oversample_max_multiplier=float(env_or("SFT_OVERSAMPLE_MAX_MULTIPLIER", splits.get("oversample_max_multiplier", 3.0))),
        enable_meter_aux=str2bool(env_or("SFT_ENABLE_METER_AUX", meter_aux.get("enabled", False))),
        meter_aux_lambda=float(env_or("SFT_METER_AUX_LAMBDA", meter_aux.get("lambda", 0.1))),
        meter_aux_normalize_by_log_num_classes=str2bool(
            env_or("SFT_METER_AUX_NORMALIZE", meter_aux.get("normalize_by_log_num_classes", True))
        ),
        probe_max_new_tokens=int(env_or("SFT_PROBE_MAX_NEW_TOKENS", probe.get("max_new_tokens", 384))),
        probe_batch_size=int(env_or("SFT_PROBE_BATCH_SIZE", probe.get("batch_size", 4))),
        probe_do_sample=str2bool(env_or("SFT_PROBE_DO_SAMPLE", probe.get("do_sample", False))),
        probe_temperature=float(env_or("SFT_PROBE_TEMPERATURE", probe.get("temperature", 0.8))),
        limit_train_rows=int(env_or("SFT_LIMIT_TRAIN_ROWS", 0)),
        limit_eval_rows=int(env_or("SFT_LIMIT_EVAL_ROWS", 0)),
        limit_test_rows=int(env_or("SFT_LIMIT_TEST_ROWS", 0)),
        plateau_min_full_eval_points=int(plateau.get("min_full_eval_points", 2)),
        plateau_patience=int(plateau.get("patience", 4)),
        plateau_full_eval_min_delta=float(plateau.get("full_eval_min_delta", 0.005)),
        plateau_probe_meter_min_delta=float(plateau.get("probe_meter_min_delta", 0.01)),
        plateau_probe_count_min_delta=float(plateau.get("probe_count_min_delta", 0.005)),
        plateau_stop_on_plateau=str2bool(plateau.get("stop_on_plateau", False)),
        log_level=str(defaults.get("log_level", "INFO")),
        hf_home=hf_home,
        hub_cache=hub_cache,
    )


def resolve_dtype(name: str) -> torch.dtype:
    value = str(name).strip().lower()
    mapping = {
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
        "float32": torch.float32,
        "fp32": torch.float32,
    }
    if value not in mapping:
        raise ValueError(f"unsupported torch dtype: {name}")
    return mapping[value]


def detect_precision(cfg: RuntimeConfig) -> tuple[bool, bool]:
    bf16_flag = str(cfg.bf16).strip().lower()
    if bf16_flag == "true":
        return True, False
    if bf16_flag == "false":
        return False, bool(cfg.fp16)
    if bool(torch.cuda.is_available() and torch.cuda.is_bf16_supported()):
        return True, False
    return False, bool(cfg.fp16)


def ensure_repo(api: HfApi, repo_id: str, token: str) -> None:
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True)


def is_retryable_hub_error(exc: Exception) -> bool:
    response = getattr(exc, "response", None)
    status_code = getattr(response, "status_code", None)
    if status_code in {408, 409, 423, 429, 500, 502, 503, 504}:
        return True
    if isinstance(exc, HfHubHTTPError) and status_code is None:
        return True
    return False


def run_hub_op(
    op_name: str,
    fn,
    *,
    logger: logging.Logger | None = None,
    max_attempts: int = 4,
    base_delay_seconds: float = 5.0,
):
    attempt = 1
    while True:
        try:
            return fn()
        except Exception as exc:
            retryable = is_retryable_hub_error(exc)
            if attempt >= max_attempts or not retryable:
                raise
            delay = base_delay_seconds * (2 ** (attempt - 1))
            if logger is not None:
                logger.warning(
                    "Hub op failed, retrying | op=%s attempt=%d/%d delay=%.1fs error=%r",
                    op_name,
                    attempt,
                    max_attempts,
                    delay,
                    exc,
                )
            time.sleep(delay)
            attempt += 1


def read_remote_json(api: HfApi, repo_id: str, path_in_repo: str, token: str, cache_dir: Path) -> dict[str, Any] | None:
    try:
        local_path = hf_hub_download(
            repo_id=repo_id,
            repo_type="model",
            filename=path_in_repo,
            token=token,
            cache_dir=str(cache_dir),
        )
    except (EntryNotFoundError, HfHubHTTPError, FileNotFoundError):
        return None
    with open(local_path, "r", encoding="utf-8") as f:
        return json.load(f)


def upload_json(api: HfApi, repo_id: str, path_in_repo: str, payload: dict[str, Any], token: str) -> None:
    run_hub_op(
        f"upload_json:{path_in_repo}",
        lambda: api.upload_file(
            path_or_fileobj=json.dumps(payload, ensure_ascii=False, indent=2).encode("utf-8"),
            path_in_repo=path_in_repo,
            repo_id=repo_id,
            repo_type="model",
            token=token,
        ),
    )


def find_latest_checkpoint_prefix(api: HfApi, repo_id: str, namespace: str, mode: str, token: str) -> str | None:
    info = api.model_info(repo_id=repo_id, token=token)
    pat = re.compile(rf"^checkpoints/{re.escape(namespace)}/{re.escape(mode)}/([^/]+)/checkpoint-(\d+)/adapter_config\.json$")
    best_step = -1
    best_prefix = None
    for sib in info.siblings or []:
        m = pat.match(sib.rfilename)
        if not m:
            continue
        step = int(m.group(2))
        prefix = f"checkpoints/{namespace}/{mode}/{m.group(1)}/checkpoint-{step}"
        if step > best_step:
            best_step = step
            best_prefix = prefix
    return best_prefix


def download_checkpoint_prefix(repo_id: str, prefix: str, token: str, cache_dir: Path) -> str:
    snapshot_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        token=token,
        allow_patterns=[f"{prefix}/*"],
        cache_dir=str(cache_dir),
    )
    local_path = Path(snapshot_dir) / prefix
    if not local_path.exists():
        raise RuntimeError(f"downloaded checkpoint prefix missing local path: {local_path}")
    return str(local_path)


def resolve_adapter_source(repo_id: str, subfolder: str, token: str, cache_dir: Path) -> str:
    snapshot_dir = snapshot_download(
        repo_id=repo_id,
        repo_type="model",
        token=token,
        allow_patterns=[f"{subfolder}/*"],
        cache_dir=str(cache_dir),
    )
    local = Path(snapshot_dir) / subfolder
    if not (local / "adapter_config.json").exists():
        raise FileNotFoundError(f"adapter_config.json missing in {local}")
    return str(local)


def resolve_resume_checkpoint(
    cfg: RuntimeConfig,
    api: HfApi,
    token: str,
    logger: logging.Logger,
    events_writer: JsonlWriter,
    fingerprint: str,
) -> tuple[str | None, dict[str, Any]]:
    manifest_path = f"manifests/{cfg.continuation_namespace}/{cfg.mode}/latest.json"
    decision = {
        "timestamp_utc": utc_now_iso(),
        "mode": cfg.mode,
        "resume_mode": cfg.resume_mode,
        "result": "fresh",
        "reason": "",
        "checkpoint_repo_path": None,
        "local_resume_path": None,
    }

    if cfg.resume_mode == "fresh":
        decision["reason"] = "explicit_fresh"
        events_writer.write({"event_type": "resume_decision", **decision})
        return None, decision

    if cfg.resume_mode == "from_path":
        raw = cfg.resume_path.strip()
        if not raw:
            raise RuntimeError("resume_path is required for resume_mode=from_path")
        if os.path.isdir(raw):
            decision["result"] = "resume"
            decision["reason"] = "explicit_local_path"
            decision["local_resume_path"] = raw
            events_writer.write({"event_type": "resume_decision", **decision})
            return raw, decision
        local = download_checkpoint_prefix(cfg.model_repo_id, raw, token, cfg.hub_cache)
        decision["result"] = "resume"
        decision["reason"] = "explicit_remote_prefix"
        decision["checkpoint_repo_path"] = raw
        decision["local_resume_path"] = local
        events_writer.write({"event_type": "resume_decision", **decision})
        return local, decision

    if not cfg.push_to_hub:
        # Local-only run: never consult the Hub (its checkpoint scan could pick up someone else's checkpoints).
        # Resume from the newest checkpoint inside this run's own output directory, if there is one.
        local_step = latest_complete_checkpoint_step(cfg.output_dir) if cfg.output_dir.is_dir() else 0
        if local_step > 0:
            local = str(cfg.output_dir / f"checkpoint-{local_step}")
            decision["result"] = "resume"
            decision["reason"] = "auto_local_checkpoint"
            decision["local_resume_path"] = local
            events_writer.write({"event_type": "resume_decision", **decision})
            return local, decision
        decision["reason"] = "auto_local_no_checkpoint"
        events_writer.write({"event_type": "resume_decision", **decision})
        return None, decision

    manifest = read_remote_json(api, cfg.model_repo_id, manifest_path, token, cfg.hub_cache)
    if manifest:
        compatible = (
            manifest.get("base_model_id") == cfg.base_model_id
            and manifest.get("dataset_id") == cfg.dataset_id
            and manifest.get("config_fingerprint") == fingerprint
            and manifest.get("checkpoint_path_in_repo")
        )
        if compatible:
            ckpt_prefix = str(manifest["checkpoint_path_in_repo"])
            local = download_checkpoint_prefix(cfg.model_repo_id, ckpt_prefix, token, cfg.hub_cache)
            decision["result"] = "resume"
            decision["reason"] = "manifest_latest_compatible"
            decision["checkpoint_repo_path"] = ckpt_prefix
            decision["local_resume_path"] = local
            events_writer.write({"event_type": "resume_decision", **decision})
            return local, decision
        decision["reason"] = "manifest_incompatible"

    fallback = find_latest_checkpoint_prefix(api, cfg.model_repo_id, cfg.continuation_namespace, cfg.mode, token)
    if fallback:
        local = download_checkpoint_prefix(cfg.model_repo_id, fallback, token, cfg.hub_cache)
        decision["result"] = "resume"
        decision["reason"] = "fallback_latest_checkpoint_scan"
        decision["checkpoint_repo_path"] = fallback
        decision["local_resume_path"] = local
        events_writer.write({"event_type": "resume_decision", **decision})
        return local, decision

    decision["reason"] = decision["reason"] or "no_checkpoint_found"
    events_writer.write({"event_type": "resume_decision", **decision})
    return None, decision


def annotate_dataset(ds: Dataset) -> Dataset:
    def _map_row(row: dict[str, Any], idx: int) -> dict[str, Any]:
        base_meter, form, meter_label = row_to_meter_fields(row)
        requested_lines = row_to_requested_lines(row)
        requested_bayts = max(1, int(requested_lines) // 2) if requested_lines else 1
        source_index = row.get("source_index")
        if source_index is None or str(source_index).strip() == "":
            source_index = idx
        return {
            "source_index": int(source_index),
            "base_meter": base_meter,
            "form": str(form).strip() or str(row.get("form") or "").strip() or "UNKNOWN_FORM",
            "meter_label": meter_label,
            "requested_lines": requested_lines,
            "requested_bayts": requested_bayts,
            "prompt_hash": sha256_text(row_to_prompt(row)),
        }

    return ds.map(_map_row, with_indices=True, desc="annotate_dataset")


def ensure_split_ready_dataset(ds: Dataset) -> Dataset:
    required = {"length_bucket", "sampler_group", "split_group", "split_group_level"}
    prepared = ds if required.issubset(set(ds.column_names)) else annotate_split_columns(ds, min_group_size=MIN_SAFE_STRATIFY_GROUP_SIZE)
    return annotate_dataset(prepared)


def load_or_build_dataset_splits(cfg: RuntimeConfig, hf_token: str, logger: logging.Logger) -> tuple[DatasetDict, dict[str, Any]]:
    try:
        loaded = load_dataset(cfg.dataset_id, cache_dir=str(cfg.hub_cache), token=hf_token)
        if isinstance(loaded, DatasetDict) and {"train", "eval", "test"}.issubset(set(loaded.keys())):
            split_ds = DatasetDict(
                {
                    "train": ensure_split_ready_dataset(loaded["train"]),
                    "eval": ensure_split_ready_dataset(loaded["eval"]),
                    "test": ensure_split_ready_dataset(loaded["test"]),
                }
            )
            summary = {
                "timestamp_utc": utc_now_iso(),
                "source": "published_dataset_splits",
                "dataset_id": cfg.dataset_id,
                "split_counts": {name: len(split) for name, split in split_ds.items()},
                "distribution_tables": build_distribution_report(split_ds["train"], split_ds["eval"], split_ds["test"]),
            }
            logger.info("Loaded published dataset splits directly from %s", cfg.dataset_id)
            return split_ds, summary
    except Exception as exc:
        logger.warning("Published split loading failed for %s: %r", cfg.dataset_id, exc)

    raw_train = load_dataset_resilient(cfg.dataset_id, hf_token, logger, cfg.hub_cache)
    split_ds, summary = split_dataset(raw_train, seed=cfg.seed, min_group_size=MIN_SAFE_STRATIFY_GROUP_SIZE)
    split_ds = DatasetDict({name: ensure_split_ready_dataset(split) for name, split in split_ds.items()})
    summary = {
        "timestamp_utc": utc_now_iso(),
        "source": "local_stratified_fallback",
        "dataset_id": cfg.dataset_id,
        **summary,
    }
    logger.info("Built local stratified train/eval/test splits for %s", cfg.dataset_id)
    return split_ds, summary


def filter_dataset_for_run(ds: Dataset, cfg: RuntimeConfig, logger: logging.Logger) -> tuple[Dataset, dict[str, Any]]:
    original_rows = len(ds)
    original_meter_counts = Counter(str(x) for x in ds["base_meter"])
    original_bayt_counts = Counter(int(x) for x in ds["requested_bayts"])
    drop_meters = set(cfg.drop_meters)

    def keep(row: dict[str, Any]) -> bool:
        meter = str(row["base_meter"])
        bayts = int(row["requested_bayts"])
        if drop_meters and meter in drop_meters:
            return False
        if cfg.min_bayts > 0 and bayts < cfg.min_bayts:
            return False
        if cfg.max_bayts > 0 and bayts > cfg.max_bayts:
            return False
        return True

    filtered = ds.filter(keep, desc="filter_dataset_for_run")
    filtered_meter_counts = Counter(str(x) for x in filtered["base_meter"])
    filtered_bayt_counts = Counter(int(x) for x in filtered["requested_bayts"])
    summary = {
        "timestamp_utc": utc_now_iso(),
        "original_rows": original_rows,
        "filtered_rows": len(filtered),
        "dropped_rows": original_rows - len(filtered),
        "min_bayts": cfg.min_bayts,
        "max_bayts": cfg.max_bayts,
        "drop_meters": sorted(drop_meters),
        "original_meter_counts": dict(sorted(original_meter_counts.items())),
        "filtered_meter_counts": dict(sorted(filtered_meter_counts.items())),
        "original_requested_bayts_counts": dict(sorted(original_bayt_counts.items())),
        "filtered_requested_bayts_counts": dict(sorted(filtered_bayt_counts.items())),
    }
    logger.info(
        "Dataset filter: rows %d -> %d | min_bayts=%s max_bayts=%s drop_meters=%s",
        original_rows,
        len(filtered),
        cfg.min_bayts or "none",
        cfg.max_bayts or "none",
        sorted(drop_meters),
    )
    if not len(filtered):
        raise RuntimeError("dataset is empty after SFT run filters")
    return filtered, summary


def select_probe_indices(indices: list[int], requested_bayts: list[int], count: int) -> list[int]:
    if not indices or count <= 0:
        return []
    scored = sorted(indices, key=lambda idx: (requested_bayts[idx], idx))
    if count == 1:
        return [scored[0]]
    target_long = 4
    remainder = [idx for idx in scored[1:]]
    long_pick = min(
        remainder,
        key=lambda idx: (abs(requested_bayts[idx] - target_long), requested_bayts[idx] < target_long, requested_bayts[idx], idx),
    ) if remainder else scored[0]
    chosen = [scored[0], long_pick]
    chosen = list(dict.fromkeys(chosen))
    cursor = 1
    while len(chosen) < count and cursor < len(scored) - 1:
        chosen.append(scored[cursor])
        cursor += 1
    return chosen[:count]


def cap_rows_per_meter(ds: Dataset, per_meter: int, seed: int) -> Dataset:
    by_meter: dict[str, list[int]] = defaultdict(list)
    for idx, meter in enumerate(ds["base_meter"]):
        by_meter[str(meter)].append(idx)
    rng = random.Random(seed)
    selected: list[int] = []
    for meter in sorted(by_meter):
        idxs = list(by_meter[meter])
        rng.shuffle(idxs)
        selected.extend(sorted(idxs[: min(per_meter, len(idxs))]))
    return ds.select(sorted(selected))


def build_probe_bank_from_test(ds: Dataset, cfg: RuntimeConfig, logger: logging.Logger) -> Dataset:
    if not len(ds):
        return ds
    by_meter: dict[str, list[int]] = defaultdict(list)
    requested_bayts = [int(x) for x in ds["requested_bayts"]]
    for idx, meter in enumerate(ds["base_meter"]):
        by_meter[str(meter)].append(idx)

    selected: list[int] = []
    for meter in sorted(by_meter):
        idxs = by_meter[meter]
        target = min(cfg.probe_per_meter, len(idxs))
        selected.extend(select_probe_indices(idxs, requested_bayts, target))
    selected = sorted(set(selected))
    probe_ds = ds.select(selected)
    logger.info("Built probe bank from test split: %d rows", len(probe_ds))
    return probe_ds


def maybe_limit_dataset_rows(ds: Dataset, limit: int, seed: int, label: str, logger: logging.Logger) -> Dataset:
    if limit <= 0 or len(ds) <= limit:
        return ds
    rng = random.Random(seed)
    idxs = list(range(len(ds)))
    rng.shuffle(idxs)
    selected = sorted(idxs[:limit])
    limited = ds.select(selected)
    logger.info("Applied row cap for %s: %d -> %d", label, len(ds), len(limited))
    return limited


def preprocess_dataset(
    ds: Dataset,
    tokenizer: AutoTokenizer,
    cfg: RuntimeConfig,
    desc: str,
    logger: logging.Logger,
    packing: bool = False,
) -> Dataset:
    def fn(batch: dict[str, list[Any]], indices: list[int]) -> dict[str, list[Any]]:
        out_input_ids: list[list[int]] = []
        out_attn: list[list[int]] = []
        out_labels: list[list[int]] = []
        out_prompt_lens: list[int] = []
        out_supervised: list[int] = []
        out_source_index: list[int] = []
        out_base_meter: list[str] = []
        out_form: list[str] = []
        out_length_bucket: list[str] = []
        out_meter_label: list[str] = []
        out_requested_bayts: list[int] = []
        out_prompt_hash: list[str] = []
        out_sampler_group: list[str] = []
        out_meter_class_id: list[int] = []

        prompts = batch["sft_prompt"]
        full_texts = batch["sft_full_text"]
        base_meters = batch["base_meter"]
        forms = batch["form"]
        length_buckets = batch["length_bucket"]
        meter_labels = batch["meter_label"]
        requested_bayts = batch["requested_bayts"]
        prompt_hashes = batch["prompt_hash"]
        sampler_groups = batch["sampler_group"]
        source_indices = batch["source_index"]
        meter_class_ids = batch["meter_class_id"]

        for i, (prompt, full_text) in enumerate(zip(prompts, full_texts)):
            full_ids = tokenizer(full_text, add_special_tokens=False).input_ids
            prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
            if cfg.max_seq_length > 0 and len(full_ids) > cfg.max_seq_length:
                if cfg.truncation_mode == "keep_end":
                    trim = len(full_ids) - cfg.max_seq_length
                    full_ids = full_ids[-cfg.max_seq_length :]
                    prompt_len = max(0, len(prompt_ids) - trim)
                else:
                    full_ids = full_ids[: cfg.max_seq_length]
                    prompt_len = min(len(prompt_ids), len(full_ids))
            else:
                prompt_len = min(len(prompt_ids), len(full_ids))
            labels = full_ids.copy()
            if cfg.completion_only_loss:
                for pos in range(prompt_len):
                    labels[pos] = -100
            supervised = sum(1 for x in labels if x != -100)

            out_input_ids.append(full_ids)
            out_attn.append([1] * len(full_ids))
            out_labels.append(labels)
            out_prompt_lens.append(prompt_len)
            out_supervised.append(supervised)
            out_source_index.append(int(source_indices[i]))
            out_base_meter.append(str(base_meters[i]))
            out_form.append(str(forms[i]))
            out_length_bucket.append(str(length_buckets[i]))
            out_meter_label.append(str(meter_labels[i]))
            out_requested_bayts.append(int(requested_bayts[i]))
            out_prompt_hash.append(str(prompt_hashes[i]))
            out_sampler_group.append(str(sampler_groups[i]))
            out_meter_class_id.append(int(meter_class_ids[i]))

        return {
            "input_ids": out_input_ids,
            "attention_mask": out_attn,
            "labels": out_labels,
            "prompt_len": out_prompt_lens,
            "supervised_tokens": out_supervised,
            "source_index": out_source_index,
            "base_meter": out_base_meter,
            "form": out_form,
            "length_bucket": out_length_bucket,
            "meter_label": out_meter_label,
            "requested_bayts": out_requested_bayts,
            "prompt_hash": out_prompt_hash,
            "sampler_group": out_sampler_group,
            "meter_class_id": out_meter_class_id,
        }

    mapped = ds.map(fn, batched=True, with_indices=True, desc=desc)
    filtered = mapped.filter(lambda ex: ex["supervised_tokens"] > 0)
    if packing:
        filtered = pack_preprocessed_dataset(filtered, cfg, logger, desc=f"{desc}_packed")
    logger.info("%s rows after preprocessing/filtering: %d", desc, len(filtered))
    return filtered


def pack_preprocessed_dataset(ds: Dataset, cfg: RuntimeConfig, logger: logging.Logger, desc: str) -> Dataset:
    packed_rows: list[dict[str, Any]] = []
    current_input_ids: list[int] = []
    current_labels: list[int] = []
    current_meta: dict[str, Any] | None = None
    packed_examples = 0

    def flush_current() -> None:
        nonlocal current_input_ids, current_labels, current_meta, packed_examples
        if not current_input_ids or current_meta is None:
            return
        supervised_tokens = sum(1 for token in current_labels if token != -100)
        if supervised_tokens > 0:
            packed_rows.append(
                {
                    "input_ids": list(current_input_ids),
                    "attention_mask": [1] * len(current_input_ids),
                    "labels": list(current_labels),
                    "prompt_len": 0,
                    "supervised_tokens": supervised_tokens,
                    "source_index": int(current_meta["source_index"]),
                    "base_meter": str(current_meta["base_meter"]),
                    "form": str(current_meta["form"]),
                    "length_bucket": str(current_meta["length_bucket"]),
                    "meter_label": str(current_meta["meter_label"]),
                    "requested_bayts": int(current_meta["requested_bayts"]),
                    "prompt_hash": str(current_meta["prompt_hash"]),
                    "sampler_group": str(current_meta["sampler_group"]),
                    "meter_class_id": int(current_meta["meter_class_id"]),
                    "packed_examples": packed_examples,
                }
            )
        current_input_ids = []
        current_labels = []
        current_meta = None
        packed_examples = 0

    by_group: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in ds:
        by_group[str(row["sampler_group"])].append(row)

    for group in sorted(by_group):
        current_input_ids = []
        current_labels = []
        current_meta = None
        packed_examples = 0
        for row in by_group[group]:
            row_input_ids = list(row["input_ids"])
            row_labels = list(row["labels"])
            cursor = 0
            while cursor < len(row_input_ids):
                if current_meta is None:
                    current_meta = {
                        "source_index": int(row["source_index"]),
                        "base_meter": str(row["base_meter"]),
                        "form": str(row["form"]),
                        "length_bucket": str(row["length_bucket"]),
                        "meter_label": str(row["meter_label"]),
                        "requested_bayts": int(row["requested_bayts"]),
                        "prompt_hash": str(row["prompt_hash"]),
                        "sampler_group": str(row["sampler_group"]),
                        "meter_class_id": int(row["meter_class_id"]),
                    }
                remaining = cfg.max_seq_length - len(current_input_ids)
                take = min(remaining, len(row_input_ids) - cursor)
                current_input_ids.extend(row_input_ids[cursor : cursor + take])
                current_labels.extend(row_labels[cursor : cursor + take])
                cursor += take
                if cursor == len(row_input_ids):
                    packed_examples += 1
                if len(current_input_ids) == cfg.max_seq_length:
                    flush_current()
        flush_current()

    packed_ds = Dataset.from_list(packed_rows)
    logger.info("%s sequences after packing: %d", desc, len(packed_ds))
    return packed_ds


class Seq2SeqLikeCollator:
    def __init__(self, pad_token_id: int):
        self.pad_token_id = pad_token_id

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, torch.Tensor]:
        max_len = max(len(f["input_ids"]) for f in features)
        input_ids = []
        attention_mask = []
        labels = []
        for f in features:
            pad = max_len - len(f["input_ids"])
            input_ids.append(f["input_ids"] + [self.pad_token_id] * pad)
            attention_mask.append(f["attention_mask"] + [0] * pad)
            labels.append(f["labels"] + [-100] * pad)
        return {
            "input_ids": torch.tensor(input_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "meter_class_id": torch.tensor([int(f["meter_class_id"]) for f in features], dtype=torch.long),
        }


def first_batch_debug(train_ds: Dataset, collator: Seq2SeqLikeCollator, out_path: Path, logger: logging.Logger) -> None:
    sample_n = min(4, len(train_ds))
    batch = collator([train_ds[i] for i in range(sample_n)])
    labels = batch["labels"]
    payload = {
        "sample_n": sample_n,
        "shape_input_ids": list(batch["input_ids"].shape),
        "shape_labels": list(batch["labels"].shape),
        "masked_count": int(labels.eq(-100).sum().item()),
        "supervised_count": int(labels.ne(-100).sum().item()),
        "has_masked": bool(labels.eq(-100).any().item()),
        "has_supervised": bool(labels.ne(-100).any().item()),
    }
    dump_json(out_path, payload)
    logger.info("First batch debug: %s", payload)
    if not payload["has_masked"] or not payload["has_supervised"]:
        raise RuntimeError("masking sanity failed")


def build_model_and_tokenizer(cfg: RuntimeConfig, logger: logging.Logger, bf16: bool, fp16: bool, adapter_source: str | None):
    tokenizer = AutoTokenizer.from_pretrained(
        cfg.base_model_id,
        trust_remote_code=cfg.trust_remote_code,
        use_fast=False,
        cache_dir=str(cfg.hub_cache),
    )
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"

    bnb_cfg = None
    if cfg.load_in_4bit:
        compute_dtype = resolve_dtype(cfg.bnb_4bit_compute_dtype)
        bnb_cfg = BitsAndBytesConfig(
            load_in_4bit=cfg.load_in_4bit,
            bnb_4bit_quant_type=cfg.bnb_4bit_quant_type,
            bnb_4bit_use_double_quant=cfg.bnb_4bit_use_double_quant,
            bnb_4bit_compute_dtype=compute_dtype,
        )

    logger.info("Loading base model for SFT...")
    load_kwargs = {
        "device_map": "auto",
        "torch_dtype": resolve_dtype(cfg.torch_dtype),
        "trust_remote_code": cfg.trust_remote_code,
        "cache_dir": str(cfg.hub_cache),
    }
    if bnb_cfg is not None:
        load_kwargs["quantization_config"] = bnb_cfg
    base_model = AutoModelForCausalLM.from_pretrained(cfg.base_model_id, **load_kwargs)
    base_model.config.use_cache = cfg.model_use_cache
    if cfg.prepare_model_for_kbit_training and cfg.load_in_4bit:
        base_model = prepare_model_for_kbit_training(base_model, use_gradient_checkpointing=cfg.gradient_checkpointing)

    if adapter_source:
        logger.info("Loading trainable LoRA adapter from %s", adapter_source)
        model = PeftModel.from_pretrained(base_model, adapter_source, is_trainable=True)
    else:
        if not cfg.load_in_4bit:
            try:
                import peft.tuners.lora.model as peft_lora_model

                peft_lora_model.is_bnb_available = lambda: False
                peft_lora_model.is_bnb_4bit_available = lambda: False
            except Exception as exc:
                logger.warning("Could not disable bitsandbytes LoRA dispatch for non-quantized run: %r", exc)
        target_modules: str | list[str]
        if len(cfg.lora_target_modules) == 1 and cfg.lora_target_modules[0].strip().lower() == "all-linear":
            target_modules = "all-linear"
        else:
            target_modules = cfg.lora_target_modules
        logger.info(
            "Creating fresh LoRA adapter | r=%d alpha=%d dropout=%.4f targets=%s use_rslora=%s",
            cfg.lora_r,
            cfg.lora_alpha,
            cfg.lora_dropout,
            target_modules,
            cfg.lora_use_rslora,
        )
        lora_cfg = LoraConfig(
            r=cfg.lora_r,
            lora_alpha=cfg.lora_alpha,
            lora_dropout=cfg.lora_dropout,
            bias=cfg.lora_bias,
            task_type=getattr(TaskType, str(cfg.lora_task_type).upper()),
            target_modules=target_modules,
            use_rslora=cfg.lora_use_rslora,
        )
        model = get_peft_model(base_model, lora_cfg)
    model.print_trainable_parameters()
    return model, tokenizer, bnb_cfg


def build_train_sampler_weights(train_ds: Dataset, cfg: RuntimeConfig) -> tuple[list[float], dict[str, Any]]:
    if not cfg.use_weighted_sampler:
        return [], {
            "enabled": False,
            "sampler": "default_unweighted",
            "group_key": "base_meter||form||length_bucket",
            "alpha": WEIGHT_ALPHA,
        }
    sample_weights, report = build_train_weights(train_ds, alpha=WEIGHT_ALPHA)
    report = {
        "enabled": True,
        "sampler": "WeightedRandomSampler",
        "replacement": True,
        "num_samples": len(train_ds),
        "group_key": "base_meter||form||length_bucket",
        **report,
    }
    return sample_weights, report


def build_meter_class_map(ds: Dataset) -> dict[str, int]:
    labels = sorted({str(x) for x in ds["base_meter"]})
    return {label: idx for idx, label in enumerate(labels)}


def add_meter_class_ids(ds: Dataset, meter_class_map: dict[str, int]) -> Dataset:
    def _map(row: dict[str, Any]) -> dict[str, Any]:
        meter = str(row["base_meter"])
        if meter not in meter_class_map:
            raise KeyError(f"missing meter in class map: {meter}")
        return {"meter_class_id": int(meter_class_map[meter])}

    return ds.map(_map, desc="add_meter_class_ids")


def meter_aux_state_path(base_dir: Path) -> Path:
    return base_dir / "meter_aux_head.pt"


def meter_aux_config_path(base_dir: Path) -> Path:
    return base_dir / "meter_aux_config.json"


def get_model_hidden_size(model) -> int:
    for attr in ("hidden_size", "n_embd", "d_model"):
        value = getattr(model.config, attr, None)
        if value is not None:
            return int(value)
    raise RuntimeError("could not infer model hidden size")


def save_meter_aux_artifacts(model, output_dir: Path) -> bool:
    if model is None:
        return False
    head = getattr(model, "meter_aux_head", None)
    label_to_id = getattr(model, "meter_aux_label_to_id", None)
    if head is None or label_to_id is None:
        return False
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(head.state_dict(), meter_aux_state_path(output_dir))
    dump_json(
        meter_aux_config_path(output_dir),
        {
            "num_classes": int(getattr(model, "meter_aux_num_classes", len(label_to_id))),
            "label_to_id": {str(k): int(v) for k, v in sorted(label_to_id.items())},
        },
    )
    return True


def attach_meter_aux_head(
    model,
    meter_class_map: dict[str, int],
    logger: logging.Logger,
    source_dir: str | None = None,
) -> None:
    label_to_id = {str(k): int(v) for k, v in sorted(meter_class_map.items())}
    hidden_size = get_model_hidden_size(model)
    head = torch.nn.Linear(hidden_size, len(label_to_id))
    first_param = next(model.parameters())
    head = head.to(device=first_param.device, dtype=first_param.dtype)
    model.add_module("meter_aux_head", head)
    model.meter_aux_num_classes = len(label_to_id)
    model.meter_aux_label_to_id = label_to_id
    model.meter_aux_id_to_label = {int(v): str(k) for k, v in label_to_id.items()}

    loaded = False
    if source_dir:
        state_path = meter_aux_state_path(Path(source_dir))
        config_path = meter_aux_config_path(Path(source_dir))
        if state_path.exists():
            payload = torch.load(state_path, map_location="cpu", weights_only=False)
            head.load_state_dict(payload, strict=True)
            loaded = True
            if config_path.exists():
                cfg_payload = read_json(config_path) or {}
                label_map = cfg_payload.get("label_to_id") or {}
                if label_map:
                    model.meter_aux_label_to_id = {str(k): int(v) for k, v in sorted(label_map.items())}
                    model.meter_aux_id_to_label = {int(v): str(k) for k, v in model.meter_aux_label_to_id.items()}
                    model.meter_aux_num_classes = len(model.meter_aux_label_to_id)
    logger.info(
        "Meter auxiliary head %s | classes=%d source=%s",
        "loaded" if loaded else "initialized",
        len(model.meter_aux_label_to_id),
        source_dir or "fresh",
    )


class MeterAuxEvalRunner:
    def __init__(
        self,
        cfg: RuntimeConfig,
        tokenizer: AutoTokenizer,
        collator: Seq2SeqLikeCollator,
        logger: logging.Logger,
        aux_eval_writer: JsonlWriter,
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.collator = collator
        self.logger = logger
        self.aux_eval_writer = aux_eval_writer

    def run(self, model, ds: Dataset, step: int, split_name: str) -> dict[str, Any]:
        if not self.cfg.enable_meter_aux:
            return {}
        if not len(ds):
            return {}

        was_training = bool(model.training)
        original_use_cache = bool(getattr(model.config, "use_cache", False))
        device = next(model.parameters()).device
        batch_size = max(1, int(self.cfg.per_device_eval_batch_size))
        rows: list[dict[str, Any]] = []

        model.eval()
        model.config.use_cache = False
        try:
            for start in range(0, len(ds), batch_size):
                batch_rows = [ds[i] for i in range(start, min(start + batch_size, len(ds)))]
                batch = self.collator(batch_rows)
                meter_targets = batch.pop("meter_class_id").to(device)
                inputs = {k: v.to(device) for k, v in batch.items()}
                with torch.inference_mode():
                    outputs = model(**inputs, output_hidden_states=True, return_dict=True)
                    ce_loss = outputs.loss
                    hidden = outputs.hidden_states[-1]
                    label_mask = inputs["labels"].ne(-100)
                    pooled = masked_mean_pool(hidden, label_mask)
                    meter_logits = model.meter_aux_head(pooled)
                    meter_loss_raw = F.cross_entropy(meter_logits, meter_targets)
                    meter_loss = normalize_meter_loss(
                        meter_loss_raw,
                        int(model.meter_aux_num_classes),
                        self.cfg.meter_aux_normalize_by_log_num_classes,
                    )
                    total_loss = ce_loss + (self.cfg.meter_aux_lambda * meter_loss)
                    preds = torch.argmax(meter_logits, dim=-1)

                for idx, row in enumerate(batch_rows):
                    rows.append(
                        {
                            "base_meter": str(row["base_meter"]),
                            "length_bucket": str(row["length_bucket"]),
                            "ce_loss": float(ce_loss.detach().item()),
                            "meter_loss": float(meter_loss.detach().item()),
                            "total_loss": float(total_loss.detach().item()),
                            "meter_correct": int(preds[idx].item() == int(meter_targets[idx].item())),
                        }
                    )
        finally:
            model.config.use_cache = original_use_cache
            if was_training:
                model.train()

        by_meter: dict[str, list[dict[str, Any]]] = defaultdict(list)
        by_length_bucket: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in rows:
            by_meter[row["base_meter"]].append(row)
            by_length_bucket[row["length_bucket"]].append(row)

        def aggregate(items: list[dict[str, Any]]) -> dict[str, float]:
            count = max(1, len(items))
            return {
                "count": int(len(items)),
                "ce_loss": float(sum(x["ce_loss"] for x in items) / count),
                "meter_loss": float(sum(x["meter_loss"] for x in items) / count),
                "total_loss": float(sum(x["total_loss"] for x in items) / count),
                "meter_accuracy": float(sum(x["meter_correct"] for x in items) / count),
            }

        summary = {
            "timestamp_utc": utc_now_iso(),
            "global_step": int(step),
            "mode": f"{split_name}_aux_eval",
            f"{split_name}_ce_loss": aggregate(rows)["ce_loss"],
            f"{split_name}_meter_loss": aggregate(rows)["meter_loss"],
            f"{split_name}_total_loss": aggregate(rows)["total_loss"],
            f"{split_name}_meter_accuracy": aggregate(rows)["meter_accuracy"],
            "by_meter": {k: aggregate(v) for k, v in sorted(by_meter.items())},
            "by_length_bucket": {k: aggregate(v) for k, v in sorted(by_length_bucket.items())},
        }
        self.aux_eval_writer.write(summary)
        self.logger.info(
            "%s aux eval | step=%d ce=%.4f meter=%.4f total=%.4f acc=%.4f",
            split_name,
            step,
            summary[f"{split_name}_ce_loss"],
            summary[f"{split_name}_meter_loss"],
            summary[f"{split_name}_total_loss"],
            summary[f"{split_name}_meter_accuracy"],
        )
        return summary


def masked_mean_pool(hidden: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.to(hidden.dtype).unsqueeze(-1)
    denom = weights.sum(dim=1).clamp_min(1.0)
    return (hidden * weights).sum(dim=1) / denom


def normalize_meter_loss(loss: torch.Tensor, num_classes: int, enabled: bool) -> torch.Tensor:
    if not enabled:
        return loss
    if num_classes <= 1:
        return loss
    return loss / math.log(float(num_classes))


class WeightedSFTTrainer(Trainer):
    def __init__(
        self,
        *args,
        train_sample_weights: list[float] | None = None,
        cfg: RuntimeConfig | None = None,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.train_sample_weights = train_sample_weights
        self.cfg = cfg
        self.enable_meter_aux = bool(cfg.enable_meter_aux) if cfg is not None else False
        self.pending_train_component_metrics: dict[str, float] = {}
        self.pending_eval_component_metrics: dict[str, float] = {}

    def get_train_dataloader(self) -> DataLoader:
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")
        if not self.train_sample_weights:
            return super().get_train_dataloader()

        sampler = WeightedRandomSampler(
            weights=torch.as_tensor(self.train_sample_weights, dtype=torch.double),
            num_samples=len(self.train_sample_weights),
            replacement=True,
        )
        return DataLoader(
            self.train_dataset,
            batch_size=self._train_batch_size,
            sampler=sampler,
            collate_fn=self.data_collator,
            num_workers=self.args.dataloader_num_workers,
            pin_memory=self.args.dataloader_pin_memory,
        )

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        meter_targets = inputs.pop("meter_class_id", None)
        labels = inputs.get("labels")
        should_compute_meter = self.enable_meter_aux and meter_targets is not None and hasattr(model, "meter_aux_head")
        outputs = model(**inputs, output_hidden_states=should_compute_meter, return_dict=True)
        ce_loss = outputs.loss
        total_loss = ce_loss
        meter_loss_raw = None
        meter_loss = None
        meter_accuracy = None

        if should_compute_meter and labels is not None:
            hidden = outputs.hidden_states[-1]
            label_mask = labels.ne(-100)
            pooled = masked_mean_pool(hidden, label_mask)
            meter_logits = model.meter_aux_head(pooled)
            meter_loss_raw = F.cross_entropy(meter_logits, meter_targets)
            meter_loss = normalize_meter_loss(
                meter_loss_raw,
                int(getattr(model, "meter_aux_num_classes", meter_logits.shape[-1])),
                bool(self.cfg.meter_aux_normalize_by_log_num_classes),
            )
            total_loss = ce_loss + (float(self.cfg.meter_aux_lambda) * meter_loss)
            preds = torch.argmax(meter_logits, dim=-1)
            meter_accuracy = float((preds == meter_targets).float().mean().item())

        if model.training:
            self.pending_train_component_metrics = {
                "train_ce_loss": float(ce_loss.detach().item()),
                "train_total_loss": float(total_loss.detach().item()),
            }
            if meter_loss is not None and meter_loss_raw is not None:
                self.pending_train_component_metrics.update(
                    {
                        "train_meter_loss": float(meter_loss.detach().item()),
                        "train_meter_loss_raw": float(meter_loss_raw.detach().item()),
                        "train_meter_accuracy": float(meter_accuracy or 0.0),
                    }
                )

        loss_to_return = total_loss if model.training else ce_loss
        return (loss_to_return, outputs) if return_outputs else loss_to_return

    def log(self, logs: dict[str, float], start_time: float | None = None) -> None:
        merged = dict(logs)
        if "loss" in logs and self.pending_train_component_metrics:
            merged.update(self.pending_train_component_metrics)
            self.pending_train_component_metrics = {}
        if any(str(k).startswith("eval_") for k in logs.keys()) and self.pending_eval_component_metrics:
            merged.update(self.pending_eval_component_metrics)
            self.pending_eval_component_metrics = {}
        # Trainer.log() has version-dependent signatures, so keep the call
        # compatible by forwarding only the merged payload.
        super().log(merged)


class MetricsCallback(TrainerCallback):
    def __init__(self, metrics_store: JsonlCsvMirror, events_writer: JsonlWriter, logger: logging.Logger, heartbeat_seconds: int):
        self.metrics_store = metrics_store
        self.events_writer = events_writer
        self.logger = logger
        self.heartbeat_seconds = heartbeat_seconds
        self.last_heartbeat = time.time()

    def on_log(self, args, state, control, logs=None, **kwargs):
        if not state.is_world_process_zero:
            return
        logs = logs or {}
        mode = "train"
        if any(str(k).startswith("eval_") for k in logs.keys()):
            mode = "eval"
        row = {
            "timestamp_utc": utc_now_iso(),
            "mode": mode,
            "global_step": int(state.global_step),
            "epoch": float(state.epoch) if state.epoch is not None else None,
            **{k: (float(v) if isinstance(v, (int, float)) else v) for k, v in logs.items()},
        }
        self.metrics_store.write(row)

    def on_step_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return
        now = time.time()
        if now - self.last_heartbeat >= self.heartbeat_seconds:
            self.last_heartbeat = now
            evt = {
                "timestamp_utc": utc_now_iso(),
                "event_type": "heartbeat",
                "global_step": int(state.global_step),
                "epoch": float(state.epoch) if state.epoch is not None else None,
            }
            self.events_writer.write(evt)
            self.logger.info("Heartbeat | step=%s epoch=%s", evt["global_step"], evt["epoch"])


class ContinuationHubCallback(TrainerCallback):
    def __init__(
        self,
        api: HfApi,
        token: str,
        cfg: RuntimeConfig,
        tokenizer: AutoTokenizer,
        logger: logging.Logger,
        events_writer: JsonlWriter,
        fingerprint: str,
    ):
        self.api = api
        self.token = token
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.logger = logger
        self.events_writer = events_writer
        self.fingerprint = fingerprint
        self.latest_checkpoint_repo_path: str | None = None
        self.best_adapter_step: int | None = None
        self.best_adapter_metric: float | None = None
        self.history: list[dict[str, Any]] = []

    def checkpoint_repo_path(self, step: int) -> str:
        return f"checkpoints/{self.cfg.continuation_namespace}/{self.cfg.mode}/{self.cfg.run_name}/checkpoint-{step}"

    def latest_manifest_path(self) -> str:
        return f"manifests/{self.cfg.continuation_namespace}/{self.cfg.mode}/latest.json"

    def best_manifest_path(self) -> str:
        return f"manifests/{self.cfg.continuation_namespace}/{self.cfg.mode}/best.json"

    def history_manifest_path(self) -> str:
        return f"manifests/{self.cfg.continuation_namespace}/{self.cfg.mode}/history/{self.cfg.run_name}.json"

    def latest_adapter_path(self) -> str:
        return f"adapters/{self.cfg.continuation_namespace}/{self.cfg.mode}/latest"

    def best_adapter_path(self) -> str:
        return f"adapters/{self.cfg.continuation_namespace}/{self.cfg.mode}/best"

    def write_manifests(self, step: int, checkpoint_path_in_repo: str, adapter_latest_path: str | None = None, best_payload: dict[str, Any] | None = None) -> None:
        latest_payload = {
            "repo_id": self.cfg.model_repo_id,
            "run_name": self.cfg.run_name,
            "base_model_id": self.cfg.base_model_id,
            "dataset_id": self.cfg.dataset_id,
            "checkpoint_path_in_repo": checkpoint_path_in_repo,
            "adapter_latest_path_in_repo": adapter_latest_path,
            "global_step": int(step),
            "timestamp_utc": utc_now_iso(),
            "config_fingerprint": self.fingerprint,
        }
        upload_json(self.api, self.cfg.model_repo_id, self.latest_manifest_path(), latest_payload, self.token)
        self.history.append(latest_payload)
        upload_json(self.api, self.cfg.model_repo_id, self.history_manifest_path(), {"items": self.history}, self.token)
        if best_payload is not None:
            upload_json(self.api, self.cfg.model_repo_id, self.best_manifest_path(), best_payload, self.token)

    def prune_old_remote_checkpoints(self) -> None:
        keep_n = max(0, int(self.cfg.remote_keep_last_checkpoints))
        if keep_n <= 0:
            return
        pattern = re.compile(
            rf"^checkpoints/{re.escape(self.cfg.continuation_namespace)}/{re.escape(self.cfg.mode)}/{re.escape(self.cfg.run_name)}/checkpoint-(\d+)/.+$"
        )
        grouped: dict[int, list[str]] = {}
        for path in list_repo_files(repo_id=self.cfg.model_repo_id, repo_type="model", token=self.token):
            m = pattern.match(path)
            if not m:
                continue
            step = int(m.group(1))
            grouped.setdefault(step, []).append(path)
        if len(grouped) <= keep_n:
            return
        delete_steps = sorted(grouped.keys())[: len(grouped) - keep_n]
        delete_files = [path for step in delete_steps for path in grouped[step]]
        if not delete_files:
            return
        ops = [CommitOperationDelete(path_in_repo=path) for path in delete_files]
        run_hub_op(
            f"prune_checkpoints:{self.cfg.run_name}",
            lambda: self.api.create_commit(
                repo_id=self.cfg.model_repo_id,
                repo_type="model",
                token=self.token,
                operations=ops,
                commit_message=f"Prune continuation checkpoints for {self.cfg.run_name}",
            ),
            logger=self.logger,
        )
        self.events_writer.write(
            {
                "timestamp_utc": utc_now_iso(),
                "event_type": "checkpoint_pruned",
                "deleted_steps": delete_steps,
                "deleted_files_count": len(delete_files),
            }
        )

    def upload_adapter_export(self, model, tokenizer, path_in_repo: str, tag: str) -> None:
        export_dir = self.cfg.output_dir / f"_{tag}_adapter_export"
        if export_dir.exists():
            shutil.rmtree(export_dir)
        export_dir.mkdir(parents=True, exist_ok=True)
        model.save_pretrained(export_dir)
        tokenizer.save_pretrained(export_dir)
        save_meter_aux_artifacts(model, export_dir)
        run_hub_op(
            f"upload_adapter_export:{path_in_repo}",
            lambda: self.api.upload_folder(
                folder_path=str(export_dir),
                repo_id=self.cfg.model_repo_id,
                repo_type="model",
                token=self.token,
                path_in_repo=path_in_repo,
            ),
            logger=self.logger,
        )

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or not self.cfg.push_to_hub:
            return
        step = int(state.global_step)
        local_ckpt = Path(args.output_dir) / f"checkpoint-{step}"
        if not local_ckpt.exists():
            self.logger.warning("Expected checkpoint missing on save: %s", local_ckpt)
            return
        save_meter_aux_artifacts(kwargs.get("model"), local_ckpt)
        repo_path = self.checkpoint_repo_path(step)
        t0 = time.time()
        try:
            run_hub_op(
                f"upload_checkpoint:{repo_path}",
                lambda: self.api.upload_folder(
                    folder_path=str(local_ckpt),
                    repo_id=self.cfg.model_repo_id,
                    repo_type="model",
                    token=self.token,
                    path_in_repo=repo_path,
                ),
                logger=self.logger,
            )
        except Exception as exc:
            elapsed = int((time.time() - t0) * 1000)
            self.events_writer.write(
                {
                    "timestamp_utc": utc_now_iso(),
                    "event_type": "checkpoint_upload_failed",
                    "global_step": step,
                    "hf_path": repo_path,
                    "duration_ms": elapsed,
                    "error": repr(exc),
                }
            )
            self.logger.warning("Checkpoint upload failed at step=%d to %s: %r", step, repo_path, exc)
            return
        elapsed = int((time.time() - t0) * 1000)
        self.latest_checkpoint_repo_path = repo_path
        self.write_manifests(step=step, checkpoint_path_in_repo=repo_path)
        self.prune_old_remote_checkpoints()
        self.events_writer.write(
            {
                "timestamp_utc": utc_now_iso(),
                "event_type": "checkpoint_uploaded",
                "global_step": step,
                "hf_path": repo_path,
                "duration_ms": elapsed,
            }
        )
        self.logger.info("Uploaded checkpoint step=%d to %s (%d ms)", step, repo_path, elapsed)

    def on_train_end(self, args, state, control, **kwargs):
        if not state.is_world_process_zero or not self.cfg.push_to_hub:
            return
        model = kwargs.get("model")
        tokenizer = kwargs.get("tokenizer") or self.tokenizer
        if model is None or tokenizer is None:
            return
        self.upload_adapter_export(model, tokenizer, self.latest_adapter_path(), "latest")
        step = int(state.global_step)
        self.write_manifests(
            step=step,
            checkpoint_path_in_repo=self.latest_checkpoint_repo_path or self.checkpoint_repo_path(step),
            adapter_latest_path=self.latest_adapter_path(),
        )
        self.events_writer.write(
            {
                "timestamp_utc": utc_now_iso(),
                "event_type": "adapter_latest_uploaded",
                "global_step": step,
                "hf_path": self.latest_adapter_path(),
            }
        )


class ProbeRunner:
    def __init__(
        self,
        cfg: RuntimeConfig,
        tokenizer: AutoTokenizer,
        probe_ds: Dataset,
        logger: logging.Logger,
        probe_generations_writer: JsonlWriter,
        probe_metrics_store: JsonlCsvMirror,
        events_writer: JsonlWriter,
    ):
        self.cfg = cfg
        self.tokenizer = tokenizer
        self.probe_ds = probe_ds
        self.logger = logger
        self.probe_generations_writer = probe_generations_writer
        self.probe_metrics_store = probe_metrics_store
        self.events_writer = events_writer

    def run(self, model, step: int) -> dict[str, Any]:
        device = next(model.parameters()).device
        rows: list[dict[str, Any]] = []
        generate_kwargs = {
            "max_new_tokens": self.cfg.probe_max_new_tokens,
            "do_sample": self.cfg.probe_do_sample,
            "pad_token_id": self.tokenizer.pad_token_id,
            "eos_token_id": self.tokenizer.eos_token_id,
        }
        if self.cfg.probe_do_sample:
            generate_kwargs["temperature"] = self.cfg.probe_temperature

        probe_rows = list(self.probe_ds)
        batch_size = max(1, int(self.cfg.probe_batch_size))
        was_training = bool(model.training)
        original_use_cache = bool(getattr(model.config, "use_cache", False))
        model.eval()
        model.config.use_cache = True
        try:
            for start in range(0, len(probe_rows), batch_size):
                batch_rows = probe_rows[start : start + batch_size]
                prompts = [str(row["sft_prompt"]) for row in batch_rows]
                encoded = self.tokenizer(
                    prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.cfg.max_seq_length,
                )
                encoded = {k: v.to(device) for k, v in encoded.items()}
                prompt_width = int(encoded["input_ids"].shape[1])
                with torch.inference_mode():
                    outputs = model.generate(**encoded, **generate_kwargs)

                for i, row in enumerate(batch_rows):
                    generated_ids = outputs[i][prompt_width:]
                    completion_text = self.tokenizer.decode(generated_ids, skip_special_tokens=True).strip()
                    meter_result = score_meter_poem(completion_text, row["meter_label"], row["base_meter"])
                    count_result = score_count_adherence(int(row["requested_bayts"]), completion_text)
                    payload = {
                        "timestamp_utc": utc_now_iso(),
                        "global_step": step,
                        "source_index": int(row["source_index"]),
                        "base_meter": str(row["base_meter"]),
                        "meter_label": str(row["meter_label"]),
                        "requested_bayts": int(row["requested_bayts"]),
                        "prompt_hash": str(row["prompt_hash"]),
                        "prompt_text": prompts[i],
                        "completion_text": completion_text,
                        "probe_meter_score": float(meter_result["score"]),
                        "probe_count_adherence": float(count_result["score"]),
                        "probe_generated_bayts": int(count_result["generated_bayts"]),
                    }
                    rows.append(payload)
                    self.probe_generations_writer.write(payload)
        finally:
            model.config.use_cache = original_use_cache
            if was_training:
                model.train()

        by_meter: dict[str, list[float]] = defaultdict(list)
        by_meter_count: dict[str, list[float]] = defaultdict(list)
        for row in rows:
            by_meter[row["base_meter"]].append(float(row["probe_meter_score"]))
            by_meter_count[row["base_meter"]].append(float(row["probe_count_adherence"]))

        summary = {
            "timestamp_utc": utc_now_iso(),
            "global_step": step,
            "mode": "probe",
            "probe_meter_mean": float(sum(r["probe_meter_score"] for r in rows) / len(rows)) if rows else 0.0,
            "probe_count_adherence_mean": float(sum(r["probe_count_adherence"] for r in rows) / len(rows)) if rows else 0.0,
            "probe_rows": len(rows),
            "per_meter_probe_meter_mean": {meter: float(sum(vals) / len(vals)) for meter, vals in sorted(by_meter.items())},
            "per_meter_probe_count_mean": {meter: float(sum(vals) / len(vals)) for meter, vals in sorted(by_meter_count.items())},
        }
        self.probe_metrics_store.write(summary)
        self.events_writer.write(
            {
                "timestamp_utc": utc_now_iso(),
                "event_type": "probe_completed",
                "global_step": step,
                "probe_rows": len(rows),
                "probe_meter_mean": summary["probe_meter_mean"],
                "probe_count_adherence_mean": summary["probe_count_adherence_mean"],
            }
        )
        self.logger.info(
            "Probe completed | step=%d rows=%d meter_mean=%.4f count_mean=%.4f",
            step,
            len(rows),
            summary["probe_meter_mean"],
            summary["probe_count_adherence_mean"],
        )
        return summary


class MeterAuxEvalCallback(TrainerCallback):
    def __init__(
        self,
        cfg: RuntimeConfig,
        trainer_ref: dict[str, Any],
        eval_runner: MeterAuxEvalRunner,
        metrics_store: JsonlCsvMirror,
        events_writer: JsonlWriter,
        logger: logging.Logger,
    ):
        self.cfg = cfg
        self.trainer_ref = trainer_ref
        self.eval_runner = eval_runner
        self.metrics_store = metrics_store
        self.events_writer = events_writer
        self.logger = logger
        self.best_total_loss: float | None = None
        self.best_total_step: int | None = None
        self.last_completed_step: int | None = None
        self.latest_summary: dict[str, Any] = {}

    def on_evaluate(self, args, state, control, **kwargs):
        trainer = self.trainer_ref.get("trainer")
        model = kwargs.get("model")
        if not self.cfg.enable_meter_aux or trainer is None or model is None:
            return
        step = int(state.global_step)
        if self.last_completed_step == step:
            return
        summary = self.eval_runner.run(model, trainer.eval_dataset, step, split_name="eval")
        self.last_completed_step = step
        self.latest_summary = summary
        scalar_row = {
            "timestamp_utc": utc_now_iso(),
            "mode": "eval_aux",
            "global_step": step,
            "eval_ce_loss": float(summary.get("eval_ce_loss", 0.0)),
            "eval_meter_loss": float(summary.get("eval_meter_loss", 0.0)),
            "eval_total_loss": float(summary.get("eval_total_loss", 0.0)),
            "eval_meter_accuracy": float(summary.get("eval_meter_accuracy", 0.0)),
        }
        self.metrics_store.write(scalar_row)
        if trainer is not None:
            trainer.pending_eval_component_metrics = {
                "eval_ce_loss": scalar_row["eval_ce_loss"],
                "eval_meter_loss": scalar_row["eval_meter_loss"],
                "eval_total_loss": scalar_row["eval_total_loss"],
                "eval_meter_accuracy": scalar_row["eval_meter_accuracy"],
            }
        total_loss = scalar_row["eval_total_loss"]
        if self.best_total_loss is None or total_loss < self.best_total_loss:
            self.best_total_loss = total_loss
            self.best_total_step = step
        self.events_writer.write(
            {
                "timestamp_utc": utc_now_iso(),
                "event_type": "eval_aux_completed",
                "global_step": step,
                "eval_total_loss": total_loss,
                "eval_meter_accuracy": scalar_row["eval_meter_accuracy"],
            }
        )


class RescueSaveCallback(TrainerCallback):
    """Extra resume checkpoints between the regular ones, for runs that can be interrupted (for example on Colab).

    Off by default (SFT_RESCUE_SAVE_STEPS and SFT_RESCUE_SAVE_MINUTES unset or 0), so the paper's schedule is untouched. When on, a
    checkpoint is also written every `every` optimiser steps and/or once `minutes` of wall-clock time have passed since the last
    checkpoint of any kind (the timer starts when training starts, so model loading does not count, and restarts after every save).
    Evaluation, the probe generation and the best-checkpoint selection still happen only at the regular `save_steps` (a save without an
    evaluation leaves the Trainer's best-checkpoint bookkeeping alone, and rotation never deletes the best checkpoint). Hugging Face
    does not allow `save_steps` below `eval_steps` together with load_best_model_at_end, which is why this is a separate mechanism.
    """

    def __init__(self, every: int = 0, minutes: float = 0.0):
        self.every = int(every)
        self.interval_s = max(0.0, float(minutes)) * 60.0
        self.last_save = time.monotonic()

    def on_train_begin(self, args, state, control, **kwargs):
        self.last_save = time.monotonic()
        return control

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step <= 0:
            return control
        if self.every > 0 and state.global_step % self.every == 0:
            control.should_save = True
        if self.interval_s > 0 and time.monotonic() - self.last_save >= self.interval_s:
            control.should_save = True
        return control

    def on_save(self, args, state, control, **kwargs):
        self.last_save = time.monotonic()
        return control


class ProbeOnSaveCallback(TrainerCallback):
    def __init__(
        self,
        cfg: RuntimeConfig,
        probe_runner: ProbeRunner,
        logger: logging.Logger,
    ):
        self.cfg = cfg
        self.probe_runner = probe_runner
        self.logger = logger
        self.last_probe_step = -1

    def on_save(self, args, state, control, **kwargs):
        if not state.is_world_process_zero:
            return

        step = int(state.global_step)
        model = kwargs.get("model")

        if step > 0 and step % self.cfg.save_steps == 0 and step != self.last_probe_step:
            self.probe_runner.run(model, step)
            self.last_probe_step = step
        return control


def summarize_dataset(ds: Dataset) -> dict[str, Any]:
    meter_counts = Counter(str(x) for x in ds["base_meter"])
    form_counts = Counter(str(x) for x in ds["form"]) if "form" in ds.column_names else Counter()
    bucket_counts = Counter(str(x) for x in ds["length_bucket"]) if "length_bucket" in ds.column_names else Counter()
    lengths = Counter(str(int(x)) for x in ds["requested_bayts"])
    return {
        "rows": len(ds),
        "unique_meters": len(meter_counts),
        "meter_counts": dict(sorted(meter_counts.items())),
        "form_counts": dict(sorted(form_counts.items())),
        "length_bucket_counts": dict(sorted(bucket_counts.items())),
        "requested_bayts_counts_top20": dict(list(sorted(lengths.items(), key=lambda item: int(item[0]))[:20])),
    }


def write_probe_bank(path: Path, probe_ds: Dataset) -> None:
    with path.open("w", encoding="utf-8") as f:
        for row in probe_ds:
            f.write(
                json.dumps(
                    {
                        "source_index": int(row["source_index"]),
                        "base_meter": str(row["base_meter"]),
                        "form": str(row.get("form", "")),
                        "length_bucket": str(row.get("length_bucket", "")),
                        "meter_label": str(row["meter_label"]),
                        "requested_bayts": int(row["requested_bayts"]),
                        "prompt_hash": str(row["prompt_hash"]),
                        "sft_prompt": str(row["sft_prompt"]),
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )


def latest_checkpoint_step(output_dir: Path) -> int:
    pattern = re.compile(r"^checkpoint-(\d+)$")
    steps = []
    for child in output_dir.iterdir():
        if not child.is_dir():
            continue
        m = pattern.match(child.name)
        if m:
            steps.append(int(m.group(1)))
    return max(steps) if steps else 0


def latest_complete_checkpoint_step(output_dir: Path) -> int:
    """Newest checkpoint-N whose files were all written (a Colab/Drive disconnect can leave a half-written one)."""
    pattern = re.compile(r"^checkpoint-(\d+)$")
    required = ("trainer_state.json", "optimizer.pt", "scheduler.pt")
    steps = []
    for child in output_dir.iterdir():
        m = pattern.match(child.name)
        if not (child.is_dir() and m and all((child / name).exists() for name in required) and (
            (child / "adapter_model.safetensors").exists() or (child / "adapter_model.bin").exists()
        )):
            continue
        try:  # HF writes trainer_state.json last; make sure a kill did not truncate it
            json.loads((child / "trainer_state.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        steps.append(int(m.group(1)))
    return max(steps) if steps else 0


def run_training() -> None:
    if ENV_PATH.exists():
        load_dotenv(ENV_PATH, override=False)

    args = parse_args()
    cfg = build_runtime_config(args)
    set_seed(cfg.seed)

    hf_token = os.getenv("HF_TOKEN", "").strip()
    if not hf_token:
        raise RuntimeError("HF_TOKEN is required")

    cfg.hf_home.mkdir(parents=True, exist_ok=True)
    cfg.hub_cache.mkdir(parents=True, exist_ok=True)
    os.environ["HF_HOME"] = str(cfg.hf_home)
    os.environ["HF_HUB_CACHE"] = str(cfg.hub_cache)
    os.environ["TRANSFORMERS_CACHE"] = str(cfg.hub_cache)

    cfg.output_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logging(cfg.output_dir, cfg.log_level)
    logger.info("Starting SFT | mode=%s run_name=%s", cfg.mode, cfg.run_name)
    if cfg.base_model_id == "Navid-AI/Yehia-7B-preview":
        logger.info("Base model is the paper's: %s", cfg.base_model_id)
    else:
        logger.warning("Base model is %s, NOT the paper's Navid-AI/Yehia-7B-preview: results are a documented deviation.", cfg.base_model_id)
    logger.info("Output dir: %s", cfg.output_dir)

    config_snapshot = asdict(cfg)
    config_snapshot["output_dir"] = str(cfg.output_dir)
    config_snapshot["hf_home"] = str(cfg.hf_home)
    config_snapshot["hub_cache"] = str(cfg.hub_cache)
    dump_json(cfg.output_dir / "config_snapshot.json", config_snapshot)
    dump_json(cfg.output_dir / "env_snapshot_masked.json", mask_env(dict(os.environ)))

    metrics_store = JsonlCsvMirror(cfg.output_dir / "metrics.jsonl", cfg.output_dir / "metrics.csv", enabled=cfg.jsonl_metrics)
    events_writer = JsonlWriter(cfg.output_dir / "events.jsonl", enabled=cfg.jsonl_events)
    probe_generations_writer = JsonlWriter(cfg.output_dir / "probe_generations.jsonl", enabled=True)
    probe_metrics_store = JsonlCsvMirror(cfg.output_dir / "probe_metrics.jsonl", cfg.output_dir / "probe_metrics.csv", enabled=True)
    aux_eval_writer = JsonlWriter(cfg.output_dir / "meter_aux_eval.jsonl", enabled=True)

    api = HfApi(token=hf_token)
    if cfg.push_to_hub:
        ensure_repo(api, cfg.model_repo_id, hf_token)
    else:
        logger.info("push_to_hub is off: nothing will be created or uploaded on the Hub (HF_TOKEN is only used to download gated files)")

    fingerprint = flatten_for_fingerprint(cfg)
    dump_json(cfg.output_dir / "effective_runtime_config.json", {**config_snapshot, "config_fingerprint": fingerprint})

    resume_path, resume_decision = resolve_resume_checkpoint(cfg, api, hf_token, logger, events_writer, fingerprint)
    dump_json(cfg.output_dir / "resume_decision.json", resume_decision)
    logger.info("Resume decision: %s", resume_decision)
    align_resume_trainer_state(resume_path, cfg, logger)

    split_ds, split_summary = load_or_build_dataset_splits(cfg, hf_token, logger)
    filter_summary_by_split: dict[str, Any] = {}
    filtered_splits: dict[str, Dataset] = {}
    for split_name in ["train", "eval", "test"]:
        filtered_splits[split_name], filter_summary_by_split[split_name] = filter_dataset_for_run(split_ds[split_name], cfg, logger)

    train_raw = filtered_splits["train"]
    live_eval_raw = filtered_splits["eval"]
    test_raw = filtered_splits["test"]
    meter_class_map = build_meter_class_map(train_raw)
    dump_json(cfg.output_dir / "meter_aux_label_map.json", {"label_to_id": meter_class_map})
    train_raw = add_meter_class_ids(train_raw, meter_class_map)
    live_eval_raw = add_meter_class_ids(live_eval_raw, meter_class_map)
    test_raw = add_meter_class_ids(test_raw, meter_class_map)
    if cfg.mode == "sanity":
        train_raw = cap_rows_per_meter(train_raw, per_meter=6, seed=cfg.seed)
        live_eval_raw = cap_rows_per_meter(live_eval_raw, per_meter=2, seed=cfg.seed)
        test_raw = cap_rows_per_meter(test_raw, per_meter=4, seed=cfg.seed)
        split_summary["sanity_caps"] = {
            "train_per_meter": 6,
            "live_eval_per_meter": 2,
            "test_per_meter": 4,
            "probe_per_meter": cfg.probe_per_meter,
        }
        split_summary["sanity_split_counts"] = {
            "train": len(train_raw),
            "live_eval": len(live_eval_raw),
            "test": len(test_raw),
        }
    train_raw = maybe_limit_dataset_rows(train_raw, cfg.limit_train_rows, cfg.seed + 11, "train", logger)
    live_eval_raw = maybe_limit_dataset_rows(live_eval_raw, cfg.limit_eval_rows, cfg.seed + 22, "eval", logger)
    test_raw = maybe_limit_dataset_rows(test_raw, cfg.limit_test_rows, cfg.seed + 33, "test", logger)
    probe_raw = build_probe_bank_from_test(test_raw, cfg, logger)
    dataset_summary = {
        "timestamp_utc": utc_now_iso(),
        "dataset_id": cfg.dataset_id,
        "split_source": split_summary.get("source", "unknown"),
        "split_counts": {
            "train": len(train_raw),
            "eval": len(live_eval_raw),
            "test": len(test_raw),
            "probe_bank": len(probe_raw),
        },
        "distribution_tables": build_distribution_report(train_raw, live_eval_raw, test_raw),
        "filter_summary_by_split": filter_summary_by_split,
        "row_caps": {
            "train": cfg.limit_train_rows,
            "eval": cfg.limit_eval_rows,
            "test": cfg.limit_test_rows,
        },
    }
    dump_json(cfg.output_dir / "dataset_summary.json", dataset_summary)
    split_summary["filter_summary_by_split"] = filter_summary_by_split
    split_summary["distribution_tables_after_filter"] = dataset_summary["distribution_tables"]
    dump_json(cfg.output_dir / "split_summary.json", split_summary)
    write_probe_bank(cfg.output_dir / "probe_bank.jsonl", probe_raw)

    bf16, fp16 = detect_precision(cfg)
    logger.info("Precision selected: bf16=%s fp16=%s", bf16, fp16)

    adapter_source: str | None
    if resume_path:
        adapter_source = resume_path
    elif cfg.init_adapter_path.strip():
        adapter_source = cfg.init_adapter_path.strip()
    elif cfg.init_from_adapter:
        adapter_source = resolve_adapter_source(cfg.init_adapter_repo_id, cfg.init_adapter_subfolder, hf_token, cfg.hub_cache)
    else:
        adapter_source = None
    logger.info("Adapter source: %s", adapter_source or "fresh_lora")
    model, tokenizer, _ = build_model_and_tokenizer(cfg, logger, bf16, fp16, adapter_source)
    if cfg.enable_meter_aux:
        attach_meter_aux_head(model, meter_class_map, logger, source_dir=adapter_source)

    train_ds = preprocess_dataset(train_raw, tokenizer, cfg, "train_preprocess", logger, packing=cfg.packing)
    live_eval_ds = preprocess_dataset(live_eval_raw, tokenizer, cfg, "live_eval_preprocess", logger)
    test_ds = preprocess_dataset(test_raw, tokenizer, cfg, "test_preprocess", logger)

    if not len(train_ds) or not len(live_eval_ds) or not len(test_ds):
        raise RuntimeError("one of the SFT datasets is empty after preprocessing")

    sample_weights, sampler_report = build_train_sampler_weights(train_ds, cfg)
    split_summary["train_sampler_report"] = sampler_report
    split_summary["dataloader_sampler"] = {
        "train": "WeightedRandomSampler" if sample_weights else "default_unweighted",
        "eval": "default_unweighted",
        "test": "default_unweighted",
    }
    dump_json(cfg.output_dir / "split_summary.json", split_summary)

    collator = Seq2SeqLikeCollator(tokenizer.pad_token_id)
    if cfg.debug_dump_first_batch:
        first_batch_debug(train_ds, collator, cfg.output_dir / "first_batch_debug.json", logger)

    training_args = TrainingArguments(
        output_dir=str(cfg.output_dir),
        per_device_train_batch_size=cfg.per_device_train_batch_size,
        per_device_eval_batch_size=cfg.per_device_eval_batch_size,
        gradient_accumulation_steps=cfg.gradient_accumulation_steps,
        learning_rate=cfg.learning_rate,
        weight_decay=cfg.weight_decay,
        warmup_ratio=cfg.warmup_ratio,
        lr_scheduler_type=cfg.lr_scheduler_type,
        max_grad_norm=cfg.max_grad_norm,
        num_train_epochs=cfg.num_train_epochs,
        max_steps=cfg.max_steps,
        logging_strategy="steps",
        logging_steps=cfg.logging_steps,
        logging_first_step=True,
        evaluation_strategy="steps",
        eval_steps=cfg.live_eval_steps,
        save_strategy="steps",
        save_steps=cfg.save_steps,
        save_total_limit=cfg.save_total_limit,
        bf16=bf16,
        fp16=fp16,
        gradient_checkpointing=cfg.gradient_checkpointing,
        optim=cfg.optim,
        remove_unused_columns=False,
        report_to=[],
        save_safetensors=True,
        run_name=cfg.run_name,
        seed=cfg.seed,
        load_best_model_at_end=cfg.load_best_model_at_end,
        metric_for_best_model=cfg.metric_for_best_model if cfg.load_best_model_at_end else None,
        greater_is_better=cfg.greater_is_better if cfg.load_best_model_at_end else None,
    )

    metrics_cb = MetricsCallback(metrics_store, events_writer, logger, cfg.heartbeat_seconds)
    hub_cb = ContinuationHubCallback(api, hf_token, cfg, tokenizer, logger, events_writer, fingerprint)
    probe_runner = ProbeRunner(cfg, tokenizer, probe_raw, logger, probe_generations_writer, probe_metrics_store, events_writer)
    aux_eval_runner = MeterAuxEvalRunner(cfg, tokenizer, collator, logger, aux_eval_writer)
    trainer_holder: dict[str, Any] = {}
    meter_aux_eval_cb = MeterAuxEvalCallback(
        cfg=cfg,
        trainer_ref=trainer_holder,
        eval_runner=aux_eval_runner,
        metrics_store=metrics_store,
        events_writer=events_writer,
        logger=logger,
    )
    probe_cb = ProbeOnSaveCallback(
        cfg=cfg,
        probe_runner=probe_runner,
        logger=logger,
    )

    callbacks = [metrics_cb, hub_cb, meter_aux_eval_cb, probe_cb]
    rescue_every = int(os.getenv("SFT_RESCUE_SAVE_STEPS", "0") or 0)
    rescue_minutes = float(os.getenv("SFT_RESCUE_SAVE_MINUTES", "0") or 0)
    if rescue_every > 0 or rescue_minutes > 0:
        callbacks.append(RescueSaveCallback(rescue_every, rescue_minutes))
        if rescue_every > 0:
            logger.info("Extra resume checkpoints every %d steps (regular: every %d; evaluation and best-checkpoint selection stay on "
                        "the regular schedule)", rescue_every, cfg.save_steps)
        if rescue_minutes > 0:
            logger.info("Extra resume checkpoint whenever %.1f minutes have passed since the last checkpoint (regular: every %d steps; "
                        "evaluation and best-checkpoint selection stay on the regular schedule)", rescue_minutes, cfg.save_steps)
    trainer = WeightedSFTTrainer(
        model=model,
        args=training_args,
        train_dataset=train_ds,
        eval_dataset=live_eval_ds,
        tokenizer=tokenizer,
        data_collator=collator,
        callbacks=callbacks,
        train_sample_weights=sample_weights,
        cfg=cfg,
    )
    trainer_holder["trainer"] = trainer

    logger.info("Running initial eval before training...")
    initial_eval = trainer.evaluate(metric_key_prefix="eval")
    metrics_store.write(
        {
            "timestamp_utc": utc_now_iso(),
            "mode": "eval",
            "global_step": 0,
            **{k: float(v) if isinstance(v, (int, float)) else v for k, v in initial_eval.items()},
        }
    )
    initial_eval_aux = dict(meter_aux_eval_cb.latest_summary) if cfg.enable_meter_aux else {}
    initial_probe = probe_runner.run(model, 0)

    logger.info("Starting SFT training...")
    original_torch_load = torch.load

    def trusted_checkpoint_torch_load(*args, **kwargs):
        kwargs.setdefault("weights_only", False)
        return original_torch_load(*args, **kwargs)

    if resume_path:
        torch.load = trusted_checkpoint_torch_load
    try:
        train_result = trainer.train(resume_from_checkpoint=resume_path)
    finally:
        torch.load = original_torch_load
    logger.info("Training finished. global_step=%s", trainer.state.global_step)

    final_eval = trainer.evaluate(metric_key_prefix="eval")
    final_test = trainer.evaluate(eval_dataset=test_ds, metric_key_prefix="test")
    final_eval_aux = (
        aux_eval_runner.run(model, live_eval_ds, int(trainer.state.global_step), split_name="eval")
        if cfg.enable_meter_aux
        else {}
    )
    final_test_aux = (
        aux_eval_runner.run(model, test_ds, int(trainer.state.global_step), split_name="test")
        if cfg.enable_meter_aux
        else {}
    )
    metrics_store.write(
        {
            "timestamp_utc": utc_now_iso(),
            "mode": "test_eval",
            "global_step": int(trainer.state.global_step),
            **{k: float(v) if isinstance(v, (int, float)) else v for k, v in final_test.items()},
        }
    )
    if final_test_aux:
        metrics_store.write(
            {
                "timestamp_utc": utc_now_iso(),
                "mode": "test_aux_eval",
                "global_step": int(trainer.state.global_step),
                "test_ce_loss": float(final_test_aux.get("test_ce_loss", 0.0)),
                "test_meter_loss": float(final_test_aux.get("test_meter_loss", 0.0)),
                "test_total_loss": float(final_test_aux.get("test_total_loss", 0.0)),
                "test_meter_accuracy": float(final_test_aux.get("test_meter_accuracy", 0.0)),
            }
        )
    final_probe = probe_runner.run(model, int(trainer.state.global_step))
    best_checkpoint_step = None
    best_model_checkpoint = getattr(trainer.state, "best_model_checkpoint", None)
    if best_model_checkpoint:
        m = re.search(r"checkpoint-(\d+)", str(best_model_checkpoint))
        if m:
            best_checkpoint_step = int(m.group(1))
    if cfg.push_to_hub and best_checkpoint_step is not None:
        hub_cb.upload_adapter_export(model, tokenizer, hub_cb.best_adapter_path(), "best")
        hub_cb.best_adapter_step = best_checkpoint_step
        hub_cb.best_adapter_metric = float(getattr(trainer.state, "best_metric", float(final_eval.get("eval_loss", 0.0))) or 0.0)
        best_payload = {
            "repo_id": cfg.model_repo_id,
            "run_name": cfg.run_name,
            "global_step": best_checkpoint_step,
            "best_eval_loss": hub_cb.best_adapter_metric,
            "probe_meter_mean": final_probe.get("probe_meter_mean"),
            "probe_count_adherence_mean": final_probe.get("probe_count_adherence_mean"),
            "adapter_best_path_in_repo": hub_cb.best_adapter_path(),
            "checkpoint_path_in_repo": hub_cb.checkpoint_repo_path(best_checkpoint_step),
            "timestamp_utc": utc_now_iso(),
            "config_fingerprint": hub_cb.fingerprint,
        }
        hub_cb.write_manifests(
            step=int(trainer.state.global_step),
            checkpoint_path_in_repo=hub_cb.latest_checkpoint_repo_path or hub_cb.checkpoint_repo_path(int(trainer.state.global_step)),
            adapter_latest_path=hub_cb.latest_adapter_path(),
            best_payload=best_payload,
        )

    summary = {
        "timestamp_utc": utc_now_iso(),
        "mode": cfg.mode,
        "run_name": cfg.run_name,
        "resume_decision": resume_decision,
        "global_step": int(trainer.state.global_step),
        "train_global_step": int(trainer.state.global_step),
        "best_eval_loss": float(getattr(trainer.state, "best_metric", float("nan"))) if getattr(trainer.state, "best_metric", None) is not None else None,
        "best_eval_checkpoint": str(best_model_checkpoint) if best_model_checkpoint else None,
        "latest_checkpoint_repo_path": hub_cb.latest_checkpoint_repo_path,
        "best_adapter_path": hub_cb.best_adapter_path() if hub_cb.best_adapter_step is not None else None,
        "latest_adapter_path": hub_cb.latest_adapter_path(),
        "initial_eval": initial_eval,
        "initial_eval_aux": initial_eval_aux,
        "initial_probe": initial_probe,
        "final_eval": final_eval,
        "final_eval_aux": final_eval_aux,
        "final_test": final_test,
        "final_test_eval": final_test,
        "final_test_aux": final_test_aux,
        "final_probe": final_probe,
        "best_eval_total_loss": meter_aux_eval_cb.best_total_loss,
        "best_eval_total_loss_step": meter_aux_eval_cb.best_total_step,
        "plateau_status": None,
        "train_runtime": float(train_result.metrics.get("train_runtime", 0.0)),
        "train_loss": float(train_result.metrics.get("train_loss", float("nan"))),
        "last_checkpoint_step_local": latest_checkpoint_step(cfg.output_dir),
    }
    dump_json(cfg.output_dir / "run_summary.json", summary)
    events_writer.write({"timestamp_utc": utc_now_iso(), "event_type": "run_complete", "mode": cfg.mode, "status": "ok"})
    logger.info("Run complete. Summary written to %s", cfg.output_dir / "run_summary.json")


if __name__ == "__main__":
    run_training()
