# ... keep the imports and earlier definitions unchanged ...

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["retinanet", "detr"], required=True)
    ap.add_argument("--train-ann", required=True)
    ap.add_argument("--val-ann", required=True)
    ap.add_argument("--train-images", default="./data/coco/train2017")
    ap.add_argument("--val-images", default="./data/coco/val2017")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--weight-decay", type=float, default=0.0001)
    ap.add_argument("--warmup-steps", type=int, default=100)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--out", default="./runs/exp1")
    ap.add_argument("--early-stop-patience", type=int, default=5)
    ap.add_argument("--resume", default="")
    ap.add_argument("--albu", action="store_true")
    ap.add_argument("--albu-strength", choices=["light", "medium"], default="light")
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--max-train-batches", type=int, default=None)
    ap.add_argument("--max-val-batches", type=int, default=None)
    args = ap.parse_args()

    # ... dataset & dataloader setup unchanged ...

    # Optimizer & scheduler setup unchanged ...

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = SummaryWriter(log_dir=str(out_dir))
    print(f"[info] Checkpoints will be saved to: {out_dir}")

    # Resume logic unchanged ...

    patience = args.early_stop_patience
    epochs_no_improve = 0
    global_step = 0

    for epoch in range(start_epoch, args.epochs):
        model.train()
        running = 0.0
        for bidx, (images, targets) in enumerate(dl_train):
            if args.max_train_batches is not None and bidx >= args.max_train_batches:
                break
            images = [img.to(device) for img in images]
            targets = [{k: v.to(device) if torch.is_tensor(v) else v for k, v in t.items()} for t in targets]

            optimizer.zero_grad(set_to_none=True)
            if use_amp:
                with torch.autocast(device_type=device.type, dtype=torch.float16):
                    loss_dict, loss = train_forward(images, targets)
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
            else:
                loss_dict, loss = train_forward(images, targets)
                loss.backward()
                optimizer.step()

            scheduler.step()

            loss_val = float(loss.detach().cpu().item())
            running += loss_val
            writer.add_scalar("train/loss_step", loss_val, global_step)

            # ---- NEW: print detailed loss info ----
            if bidx % 10 == 0:  # print every 10 steps (you can change)
                losses_str = " ".join([f"{k}:{v.item():.4f}" for k,v in loss_dict.items()])
                print(f"Epoch {epoch}, Iter {bidx}, {losses_str} total:{loss_val:.4f}")

            global_step += 1

        train_epoch_loss = running / max(1, (bidx + 1))
        writer.add_scalar("train/loss_epoch", train_epoch_loss, epoch)

        # Validation
        val_loss = evaluate_loss(
            dl_val, val_forward, device, scaler=None, amp=False, max_batches=args.max_val_batches
        )
        writer.add_scalar("val/loss_epoch", val_loss, epoch)

        # ---- NEW: print validation result ----
        print(f"[Eval] Epoch {epoch} finished. train_loss={train_epoch_loss:.4f}, val_loss={val_loss:.4f}")

        # Save checkpoint
        is_best = val_loss < best_val
        if is_best:
            best_val = val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        save_checkpoint(
            {
                "epoch": epoch + 1,
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict() if scheduler is not None else None,
                "best_val": best_val,
                "args": vars(args),
            },
            out_dir=out_dir,
            is_best=is_best,
        )
        print(f"[checkpoint] Saved to {out_dir}/last.pth (best={is_best})")

        if patience is not None and epochs_no_improve >= patience:
            print(f"[early stop] No improvement for {patience} epochs. Best val loss={best_val:.4f}")
            break

    writer.close()
    print(f"[done] Best val loss={best_val:.4f}. Checkpoints at {out_dir}")
