"""
Defensive checkpoint loader. Our own train.py saves {'model':..., 'ema_model':...},
but pytorch-ddpm (and by extension SecMI/PIA, built on it) save
{'net_model':..., 'ema_model':..., 'sched':..., ...} -- confirmed from their
actual main.py source. This loader tries known key conventions in order, and
fails LOUDLY with a clear diagnostic (which keys were tried, what mismatched)
rather than silently loading something wrong -- important since we can't test
this against the real SecMI checkpoint file directly (no internet access to
fetch it in this environment).

Usage:
    from checkpoint_loader import load_checkpoint_flexible
    key_used = load_checkpoint_flexible("secmi_cifar10.pt", model, device)
"""
import torch


# Order matters: prefer EMA weights (standard for eval/generation quality),
# trying our own convention first, then pytorch-ddpm/SecMI/PIA's convention.
EMA_KEY_CANDIDATES = ["ema_model", "ema"]
RAW_KEY_CANDIDATES = ["net_model", "model", "model_state_dict", "state_dict"]


def load_checkpoint_flexible(path, model, device, prefer_ema=True, strict=True):
    """
    Loads a checkpoint into `model` in place. Returns the key name that worked.
    Raises RuntimeError with a detailed diagnostic if nothing works.
    """
    raw = torch.load(path, map_location=device)

    if not isinstance(raw, dict):
        raise RuntimeError(
            f"Checkpoint at {path} is not a dict (got {type(raw)}). Cannot "
            f"determine structure automatically.")

    # Case 1: raw IS the state_dict itself (no wrapper) -- heuristic: its values
    # are tensors, not further dicts/other structures.
    looks_like_bare_state_dict = all(
        hasattr(v, "shape") for v in raw.values()
    ) and len(raw) > 0

    candidates = (EMA_KEY_CANDIDATES + RAW_KEY_CANDIDATES) if prefer_ema \
        else (RAW_KEY_CANDIDATES + EMA_KEY_CANDIDATES)

    tried = []
    for key in candidates:
        if key not in raw:
            continue
        tried.append(key)
        state_dict = raw[key]
        # Strip 'module.' prefix if present (DataParallel-trained checkpoints) --
        # confirmed this exact handling from SecMI's own get_model() function.
        if any(k.startswith("module.") for k in state_dict.keys()):
            state_dict = {(k[7:] if k.startswith("module.") else k): v
                          for k, v in state_dict.items()}
        try:
            missing, unexpected = model.load_state_dict(state_dict, strict=strict)
            return key
        except RuntimeError as e:
            print(f"[checkpoint_loader] key '{key}' present but load_state_dict "
                  f"failed (strict={strict}): {e}\n  -- trying next candidate key.")
            continue

    if looks_like_bare_state_dict:
        try:
            model.load_state_dict(raw, strict=strict)
            return "<root, bare state_dict>"
        except RuntimeError as e:
            tried.append("<root, bare state_dict>")
            print(f"[checkpoint_loader] treating checkpoint root as a bare "
                  f"state_dict also failed: {e}")

    raise RuntimeError(
        f"Could not load checkpoint at {path} under any known key convention.\n"
        f"  Keys tried: {tried}\n"
        f"  Keys actually present in the checkpoint file: {list(raw.keys())}\n"
        f"  This likely means either (a) the architecture in ddpm/model.py doesn't "
        f"exactly match the checkpoint's architecture (check missing_keys/"
        f"unexpected_keys in the errors printed above), or (b) this checkpoint "
        f"uses a key convention not yet in EMA_KEY_CANDIDATES/RAW_KEY_CANDIDATES "
        f"above -- add the actual key name (printed in 'present in the checkpoint "
        f"file' above) to one of those lists.")


def diagnose_checkpoint(path, model, device):
    """
    Non-destructive diagnostic: for each key present in the checkpoint, report
    whether model.load_state_dict would succeed (strict), and if not, print the
    missing/unexpected keys -- without actually mutating `model`'s weights.
    Useful for debugging an architecture mismatch before committing to a fix.
    """
    raw = torch.load(path, map_location=device)
    if not isinstance(raw, dict):
        print(f"Checkpoint is not a dict: {type(raw)}")
        return

    print(f"Checkpoint top-level keys: {list(raw.keys())}")
    for key, val in raw.items():
        if not (isinstance(val, dict) and len(val) > 0 and
                all(hasattr(v, "shape") for v in val.values())):
            print(f"  '{key}': not a state_dict-like object (type={type(val)}), skipping")
            continue
        model_keys = set(model.state_dict().keys())
        ckpt_keys = set(val.keys())
        missing = model_keys - ckpt_keys
        unexpected = ckpt_keys - model_keys
        shape_mismatches = []
        for k in model_keys & ckpt_keys:
            if model.state_dict()[k].shape != val[k].shape:
                shape_mismatches.append(
                    (k, tuple(model.state_dict()[k].shape), tuple(val[k].shape)))
        print(f"  '{key}': {len(ckpt_keys)} tensors. "
              f"missing_in_ckpt={len(missing)}, unexpected_in_ckpt={len(unexpected)}, "
              f"shape_mismatches={len(shape_mismatches)}")
        if missing:
            print(f"    missing (in our model, not in ckpt): {sorted(missing)[:5]}"
                  f"{' ...' if len(missing) > 5 else ''}")
        if unexpected:
            print(f"    unexpected (in ckpt, not in our model): {sorted(unexpected)[:5]}"
                  f"{' ...' if len(unexpected) > 5 else ''}")
        if shape_mismatches:
            print(f"    shape mismatches: {shape_mismatches[:5]}")
        if not missing and not unexpected and not shape_mismatches:
            print(f"    -> PERFECT MATCH, this key will load cleanly with strict=True")