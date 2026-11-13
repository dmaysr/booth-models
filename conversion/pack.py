"""Pack the four ONNX graphs for download: graphs keep fp32 weights as external data (filled in on the user's
computer), and all float weights travel once in fp16 in a single blob. usage: pack.py SRC_DIR OUT_DIR"""
import sys, json, hashlib, numpy as np, onnx
from onnx import numpy_helper, TensorProto
src, out = sys.argv[1], sys.argv[2]
import os; os.makedirs(out, exist_ok=True)
blob = open(f"{out}/studiomax-weights.f16", "wb"); boff = 0; man = {"version": 1, "models": {}}
for name in ("prep", "cfm", "irdec", "voc"):
    m = onnx.load(f"{src}/{name}.onnx", load_external_data=True); ents = []; off32 = 0
    for t in m.graph.initializer:
        if t.data_type != TensorProto.FLOAT: continue
        a = numpy_helper.to_array(t).astype(np.float32)
        if a.size < 256: continue
        h = a.astype(np.float16)
        if not np.isfinite(h).all() or np.abs(a).max() > 60000: raise SystemExit(f"{name}/{t.name} does not fit fp16")
        blob.write(h.tobytes()); n = a.size
        ents.append([boff, n, off32]); boff += n * 2
        t.ClearField("raw_data"); del t.float_data[:]
        t.data_location = TensorProto.EXTERNAL; del t.external_data[:]
        for k, v in (("location", f"{name}.bin"), ("offset", str(off32)), ("length", str(n * 4))):
            e = t.external_data.add(); e.key, e.value = k, v
        off32 += n * 4
    onnx.save(m, f"{out}/{name}.onnx")
    man["models"][name] = {"bin": f"{name}.bin", "binBytes": off32, "chunks": ents}
    print(name, len(ents), "tensors", off32 // 1e6, "MB fp32")
blob.close(); man["blobBytes"] = boff
h = hashlib.sha256(); f = open(f"{out}/studiomax-weights.f16", "rb")
for c in iter(lambda: f.read(1 << 24), b""): h.update(c)
man["blobSha256"] = h.hexdigest(); json.dump(man, open(f"{out}/manifest.json", "w"))
print("blob", boff // 1e6, "MB", man["blobSha256"][:16])
