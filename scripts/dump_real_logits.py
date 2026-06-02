"""Tier 2: dump REAL [T,V] logits from a handful of train.jsonl samples.

This is the correctness / realism anchor, NOT the main benchmark. We run the
actual VLM on a small subset (16-32 samples), once on the original image and
once on a perturbed image, and save the per-token logits over the response span.
Purpose: (a) prove the kernel gives sane entropy/KL on real data, and
(b) check the synthetic generator's distribution isn't unrepresentative.

Run on the GPU server (needs the model weights + a GPU). Each sample's logits
are large ([T, 151936] fp16 ~= 0.3 MB/token), so keep N small and save per-sample.

NOTE: confirm the repo's exact perturbation op (blur vs noise) and model id
before the final run; defaults below match the common VL-Calibration setup.
"""
from __future__ import annotations
import argparse, ast, io, json, os
import torch


def load_row_image(row):
    """Decode the image for a row. Handles both (a) embedded bytes in the
    train_*_pipeline.jsonl `images` string field, and (b) a plain file path."""
    from PIL import Image
    field = row["images"]
    if isinstance(field, str) and field.lstrip().startswith("["):
        imgs = ast.literal_eval(field)          # literals only: bytes/str/list/dict
        entry = imgs[0]
        if isinstance(entry, dict) and entry.get("bytes"):
            return Image.open(io.BytesIO(entry["bytes"])).convert("RGB")
        field = entry["path"] if isinstance(entry, dict) else entry
    if isinstance(field, list):
        field = field[0]
    return Image.open(field).convert("RGB")


def clean_question(q):
    return q.replace("<image>", "").strip()


def perturb_image(img, kind="gaussian_blur"):
    """Image perturbation used to induce the original-vs-perturbed KL.
    Replace/confirm against the repo's actual perturbation if it differs."""
    from torchvision.transforms import functional as TF
    if kind == "gaussian_blur":
        return TF.gaussian_blur(img, kernel_size=[15, 15], sigma=[5.0, 5.0])
    if kind == "noise":
        import torch as _t
        t = TF.to_tensor(img)
        return TF.to_pil_image((t + 0.3 * _t.randn_like(t)).clamp(0, 1))
    raise ValueError(kind)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", required=True, help="path to train.jsonl")
    ap.add_argument("--model", default="Qwen/Qwen3-VL-4B-Instruct")
    ap.add_argument("--n", type=int, default=16, help="number of samples to dump")
    ap.add_argument("--perturb", default="gaussian_blur", choices=["gaussian_blur", "noise"])
    ap.add_argument("--max-new-tokens", type=int, default=128)
    ap.add_argument("--out", default="data/real")
    args = ap.parse_args()

    from transformers import AutoModelForImageTextToText, AutoProcessor
    from PIL import Image
    os.makedirs(args.out, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    proc = AutoProcessor.from_pretrained(args.model)
    model = AutoModelForImageTextToText.from_pretrained(
        args.model, torch_dtype=torch.bfloat16, device_map=device).eval()

    def run_one(image, question):
        # Teacher-forced pass over a short generated response, returning [T, V] logits.
        msgs = [{"role": "user", "content": [
            {"type": "image"}, {"type": "text", "text": question}]}]
        text = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=False)
        inputs = proc(text=[text], images=[image], return_tensors="pt").to(device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=args.max_new_tokens,
                                 do_sample=False, return_dict_in_generate=True,
                                 output_logits=True)
        # stack per-step logits -> [T, V]
        return torch.stack(gen.logits, dim=0).squeeze(1).float().cpu()

    with open(args.jsonl) as f:
        rows = [json.loads(l) for l in f][: args.n]

    for i, row in enumerate(rows):
        q = clean_question(row.get("problem") or row.get("question"))
        img = load_row_image(row)
        lo = run_one(img, q)
        lp = run_one(perturb_image(img, args.perturb), q)
        T = min(lo.shape[0], lp.shape[0])
        out = os.path.join(args.out, f"real_{i:03d}.pt")
        torch.save({"logits_orig": lo[:T].half(), "logits_pert": lp[:T].half(),
                    "T": T, "V": lo.shape[1], "question": q}, out)
        print(f"[{i}] wrote {out}  T={T}")


if __name__ == "__main__":
    main()
