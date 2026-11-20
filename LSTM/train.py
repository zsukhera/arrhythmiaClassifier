"""Train/evaluate one LSTM baseline run.

python LSTM/train.py --scheme {nlrva,aami3} --split {random,ds1ds2_heldout,ds1ds2_refit} --seed S
       [--arch conv|pool] [--epochs E (refit: fixed epoch count)]

Early stopping (random / ds1ds2_heldout) on validation macro-F1; the DS2 / random-test set is only
scored once, with the selected checkpoint, after training finishes. Runs resume from ckpt_last.pt.
"""
import argparse, json, time
from pathlib import Path
import numpy as np, torch, torch.nn as nn
from lstm_common import (HERE, SCHEMES, load_windows, get_split, ConvBiLSTM, n_params,
                         normalise, metrics, macro_f1_present)

ap = argparse.ArgumentParser()
ap.add_argument("--scheme", required=True, choices=list(SCHEMES))
ap.add_argument("--split", required=True, choices=["random", "ds1ds2_heldout", "ds1ds2_refit"])
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--arch", default="conv", choices=["conv", "pool"])
ap.add_argument("--epochs", type=int, default=40, help="max epochs (or fixed epochs for refit)")
ap.add_argument("--sched_epochs", type=int, default=40, help="cosine LR horizon (same for refit runs)")
ap.add_argument("--min_epochs", type=int, default=10, help="no early stop / checkpoint selection before this")
ap.add_argument("--patience", type=int, default=15)
ap.add_argument("--rr", action="store_true", help="add 4 causal RR-ratio side features (data/rr.npz)")
ap.add_argument("--bs", type=int, default=256)
ap.add_argument("--lr", type=float, default=1e-3)
ap.add_argument("--wd", type=float, default=1e-2)
args = ap.parse_args()

torch.set_num_threads(4)
torch.manual_seed(args.seed); np.random.seed(args.seed)
dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tag = f"{args.scheme}_{args.split}_{args.arch}" + ("rr" if args.rr else "") + (f"_e{args.epochs}" if args.split == "ds1ds2_refit" else "")
out = HERE / "runs" / tag / f"seed{args.seed}"
out.mkdir(parents=True, exist_ok=True)
if (out / "metrics.json").exists():
    print("done already:", out); raise SystemExit

classes = SCHEMES[args.scheme]; K = len(classes)
X, sym, rec, rs = load_windows()
sp = get_split(args.scheme, args.split, sym, rec, rs, seed_holdout=args.seed)
y = sp["y"]
Xg = torch.from_numpy(X).to(dev)              # float16 on GPU (~290 MB)
yg = torch.from_numpy(y).to(dev)
RRg = torch.from_numpy(np.load(HERE / "data" / "rr.npz")["rr"]).to(dev) if args.rr else None
fwd = lambda b: model(normalise(Xg[b]), RRg[b] if args.rr else None)
tr, va, te = sp["train"], sp["val"], sp["test"]
cnt = np.bincount(y[tr], minlength=K)
w = cnt.sum() / (K * np.maximum(cnt, 1))      # 'balanced' class weights
print(f"[{tag} s{args.seed}] train={len(tr)} val={len(va)} test={len(te)} counts={cnt.tolist()} w={np.round(w,2).tolist()}")

model = ConvBiLSTM(K, arch=args.arch, n_rr=4 if args.rr else 0).to(dev)
P = n_params(model); print("params", P)
opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
sched = torch.optim.lr_scheduler.LambdaLR(
    opt, lambda e: 0.5 * (1 + np.cos(np.pi * min(e, args.sched_epochs) / args.sched_epochs)) * 0.97 + 0.03)
crit = nn.CrossEntropyLoss(weight=torch.tensor(w, dtype=torch.float32, device=dev))
use_amp = dev.type == "cuda"
scaler = torch.amp.GradScaler("cuda", enabled=use_amp)


@torch.no_grad()
def predict(idx):
    model.eval(); outp = []
    for i in range(0, len(idx), 2048):
        b = torch.from_numpy(idx[i:i + 2048]).to(dev)
        with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
            outp.append(torch.softmax(fwd(b).float(), 1).cpu())
    return torch.cat(outp).numpy()


start, hist, best, best_ep, bad, t_train = 0, [], -1.0, 0, 0, 0.0
last = out / "ckpt_last.pt"
if last.exists():
    ck = torch.load(last, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model"]); opt.load_state_dict(ck["opt"]); sched.load_state_dict(ck["sched"]); scaler.load_state_dict(ck["scaler"])
    start, hist, best, best_ep, bad, t_train = ck["epoch"], ck["hist"], ck["best"], ck["best_ep"], ck["bad"], ck["t_train"]
    torch.set_rng_state(ck["rng"]); np.random.set_state(ck["nprng"])
    print("resumed at epoch", start)

trg = torch.from_numpy(tr).to(dev)
for ep in range(start, args.epochs):
    t0 = time.time(); model.train(); tot = 0.0
    perm = trg[torch.randperm(len(trg), device=dev)]
    for i in range(0, len(perm), args.bs):
        b = perm[i:i + args.bs]
        with torch.autocast(dev.type, dtype=torch.float16, enabled=use_amp):
            loss = crit(fwd(b).float(), yg[b])
        opt.zero_grad(set_to_none=True)
        scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); tot += loss.item() * len(b)
    sched.step()
    if dev.type == "cuda": torch.cuda.synchronize()
    t_train += time.time() - t0
    rec_ = dict(epoch=ep + 1, loss=tot / len(tr), lr=opt.param_groups[0]["lr"], time=time.time() - t0)
    if len(va):
        f = macro_f1_present(y[va], predict(va).argmax(1), K); rec_["val_macro_f1"] = f
        if ep + 1 >= args.min_epochs and f > best:
            best, best_ep, bad = f, ep + 1, 0
            torch.save(model.state_dict(), out / "best.pt")
        else:
            bad += 1
    hist.append(rec_); print(json.dumps(rec_), flush=True)
    torch.save(dict(model=model.state_dict(), opt=opt.state_dict(), sched=sched.state_dict(), scaler=scaler.state_dict(), epoch=ep + 1,
                    hist=hist, best=best, best_ep=best_ep, bad=bad, t_train=t_train,
                    rng=torch.get_rng_state(), nprng=np.random.get_state()), last)
    if len(va) and ep + 1 >= args.min_epochs and bad >= args.patience:
        print("early stop; best epoch", best_ep); break

if len(va):
    model.load_state_dict(torch.load(out / "best.pt", map_location=dev))
else:
    torch.save(model.state_dict(), out / "best.pt"); best_ep = len(hist)
t0 = time.time(); prob = predict(te); t_inf = time.time() - t0
np.save(out / "test_probs.npy", prob.astype(np.float32)); np.save(out / "test_y.npy", y[te])
if len(va):
    np.save(out / "val_probs.npy", predict(va).astype(np.float32)); np.save(out / "val_y.npy", y[va])
m = metrics(y[te], prob.argmax(1), classes)
m.update(tag=tag, seed=args.seed, params=P, best_epoch=best_ep, best_val_macro_f1=best,
         epochs_run=len(hist), train_time_s=t_train, infer_time_s=t_inf, n_train=len(tr), n_val=len(va),
         hist=hist, args=vars(args))
json.dump(m, open(out / "metrics.json", "w"), indent=1)
last.unlink(missing_ok=True)
print(f"TEST acc={m['acc']:.4f} ovr={m['ovr_acc']:.4f} macroF1={m['macro_f1']:.4f} "
      + " ".join(f"{c}:Se={v['se']:.3f}/PPV={v['ppv']:.3f}" for c, v in m['per_class'].items()))
