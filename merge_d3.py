import numpy as np, os
files = ["data/eff_wuke_dagger2.npz", "data/dagger_wuke600c.npz"]
xs=[]; tfs=[]; ms=[]; ys=[]; auxs=[]
for f in files:
    p = np.load(f)
    for k, lst in (("x", xs), ("tf", tfs), ("mask", ms), ("y", ys), ("aux", auxs)):
        arr = p[k] if k in p.files else np.zeros(len(p["y"]), dtype=np.float32)
        lst.append(arr)
    print(f"  {f}: {len(p['y'])} 样本", flush=True)
kw = {k: np.concatenate(v) for k, v in (("x", xs), ("tf", tfs), ("mask", ms), ("y", ys), ("aux", auxs))}
out = "data/eff_wuke_dagger3.npz"
np.savez_compressed(out, **kw)
print("merged", {k: kw[k].shape for k in kw}, os.path.getsize(out)//1024//1024, "MB")
