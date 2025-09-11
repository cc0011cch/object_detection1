from pathlib import Path
from typing import Any, Dict, Tuple
import logging
import torch


def save_checkpoint(state: Dict[str, Any], out_dir: Path, is_best: bool):
    logger = logging.getLogger("train")
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save(state, out_dir / "last.pth")
    if is_best:
        torch.save(state, out_dir / "best.pth")
    logger.info(f"[checkpoint] Saved last.pth (best={is_best}) in {out_dir}")


def load_checkpoint_if_any(model, optimizer, scheduler, ckpt_path: Path, device) -> Tuple[int, float, int, Any]:
    logger = logging.getLogger("train")
    if ckpt_path is None or not ckpt_path.exists():
        return 0, float("inf"), None, None
    ckpt = torch.load(ckpt_path, map_location=device)
    model.load_state_dict(ckpt["model"])
    if optimizer is not None and "optimizer" in ckpt:
        optimizer.load_state_dict(ckpt["optimizer"])
    if scheduler is not None and "scheduler" in ckpt and ckpt["scheduler"] is not None:
        scheduler.load_state_dict(ckpt["scheduler"])
    start_epoch = ckpt.get("epoch", 0)
    best_val = ckpt.get("best_val", float("inf"))
    best_epoch = ckpt.get("best_epoch", None)
    logger.info(
        f"[resume] Loaded checkpoint from {ckpt_path} at epoch {start_epoch} (best_val={best_val:.4f}, best_epoch={best_epoch})"
    )
    return start_epoch, best_val, best_epoch, ckpt.get("extra", None)

