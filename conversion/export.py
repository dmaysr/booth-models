"""Studio Max: export Resemble Enhance (MIT) to ONNX in three parts, using only plain ops
(STFT/ISTFT/mel as fixed convolutions) so ONNX Runtime can run them on CPU or GPU.
  prep.onnx : wav (1,N) [peak-normalized, padded], lambd (1,) -> cond (1,128,T), z0 (1,64,T) [scaled latent of the original]
  cfm.onnx  : psi (1,64,T), cond (1,128,T), t (1,) -> v (1,64,T)  [one velocity evaluation; JS runs the midpoint solver]
  dec.onnx  : z (1,64,T) [unscaled], noise (1,128,T+10) -> wav (1,T*420)
usage: export.py RUN_DIR|random OUT_DIR [check]"""
import sys, math, functools, numpy as np, torch, torch.nn as nn, torch.nn.functional as F
torch.load = functools.partial(torch.load, weights_only=False)
from pathlib import Path
from resemble_enhance.enhancer.enhancer import Enhancer
from resemble_enhance.enhancer.hparams import HParams
from resemble_enhance.inference import remove_weight_norm_recursively

def build(run):
    if run == "random":
        torch.manual_seed(1)
        hp = HParams(lcfm_training_mode="cfm")
        m = Enhancer(hp)
    else:
        run = Path(run); hp = HParams.load(run); m = Enhancer(hp)
        sd = torch.load(run / "ds/G/default/mp_rank_00_model_states.pt", map_location="cpu")["module"]
        m.load_state_dict(sd)
    m.eval(); remove_weight_norm_recursively(m); fix_filters(m.vocoder, m.hp.univnet_nc)
    # padding="same" with dilation exports as auto_pad, which ONNX Runtime refuses; spell the padding out
    for mod in m.modules():
        if isinstance(mod, (nn.Conv1d, nn.Conv2d)) and mod.padding == "same":
            mod.padding = tuple(d * (k - 1) // 2 for k, d in zip(mod.kernel_size, mod.dilation))
    return m

from resemble_enhance.enhancer.univnet.alias_free_torch.resample import UpSample1d
from resemble_enhance.enhancer.univnet.alias_free_torch.filter import LowPassFilter1d
def _up(self, x):
    x = F.pad(x, (self.pad, self.pad), mode="replicate")
    x = self.ratio * F.conv_transpose1d(x, self.fC, stride=self.stride, groups=self.fC.shape[0])
    return x[..., self.pad_left:-self.pad_right]
def _lp(self, x):
    if self.padding: x = F.pad(x, (self.pad_left, self.pad_right), mode=self.padding_mode)
    return F.conv1d(x, self.fC, stride=self.stride, groups=self.fC.shape[0])
from resemble_enhance.enhancer.univnet.lvcnet import LVCBlock
def _lvc(self, x, w, bias, dilation=1, hop_size=256):
    """same maths as the original (dilation 1). w: (b K co ks*ci) already in multiply layout, bias (b co K)"""
    b, ci, L = x.shape; K, co = w.shape[1], w.shape[2]; ks = w.shape[3] // ci; p = (ks - 1) // 2
    xp = F.pad(x, (p, p))
    xs = torch.cat([xp[:, :, k:k + L] for k in range(ks)], 1).reshape(b, ks * ci, K, hop_size).permute(0, 2, 1, 3)
    o = torch.matmul(w, xs) + bias.permute(0, 2, 1)[..., None]
    return o.permute(0, 2, 1, 3).reshape(b, co, L)

from resemble_enhance.enhancer.univnet.lvcnet import KernelPredictor
def prep_kp(kp):
    """kernel_conv as one matrix multiply whose output columns come out ordered (layer, out, tap, in), frames first"""
    W = kp.kernel_conv.weight.detach(); Bc = kp.kernel_conv.bias.detach()  # (Lr*ci*co*ks, H, 3)
    Lr, ci, co, ks = kp.conv_layers, kp.conv_in_channels, kp.conv_out_channels, kp.conv_kernel_size
    H, kc = W.shape[1], W.shape[2]
    idx = torch.arange(W.shape[0]).view(Lr, ci, co, ks).permute(0, 2, 3, 1).reshape(-1)
    kp.register_buffer("Wm", W[idx].permute(2, 1, 0).reshape(kc * H, -1).contiguous())   # (kc*H, C) rows ordered (tap, hidden)
    kp.register_buffer("Bm", Bc[idx].contiguous())
    kp.kc = kc
def _kp_forward(self, c):
    b, _, K = c.shape
    c = self.input_conv(c)
    for r in self.residual_convs: c = c + r(c)
    p = (self.kc - 1) // 2; cp = F.pad(c, (p, p))
    cu = torch.cat([cp[:, :, k:k + K] for k in range(self.kc)], 1).permute(0, 2, 1)  # (b K kc*H)
    kern = torch.matmul(cu, self.Wm) + self.Bm
    kern = kern.reshape(b, K, self.conv_layers, self.conv_out_channels, self.conv_kernel_size * self.conv_in_channels)
    bias = self.bias_conv(c).reshape(b, self.conv_layers, self.conv_out_channels, K)
    return kern, bias
def _lvcblock_forward(self, x, c):
    _, ci, _ = x.shape
    x = self.convt_pre(x); x = self.amp_block(x)
    kernels, bias = self.kernel_predictor(c)
    for i, conv in enumerate(self.conv_blocks):
        o = self.location_variable_convolution(conv(x), kernels[:, :, i], bias[:, i], hop_size=self.cond_hop_length)
        x = x + torch.sigmoid(o[:, :ci, :]) * torch.tanh(o[:, ci:, :])
    return x
LVCBlock.location_variable_convolution = _lvc

def fix_filters(m, C):
    # the anti-alias filters expand to the channel count at run time; the exporter needs them fixed
    for mod in m.modules():
        if isinstance(mod, (UpSample1d, LowPassFilter1d)):
            mod.register_buffer("fC", mod.filter.expand(C, -1, -1).contiguous())
    UpSample1d.forward = _up; LowPassFilter1d.forward = _lp
    for mod in m.modules():
        if isinstance(mod, KernelPredictor): prep_kp(mod)
    KernelPredictor.forward = _kp_forward; LVCBlock.forward = _lvcblock_forward

def hann(n): return torch.hann_window(n, periodic=True, dtype=torch.float64)

class STFT(nn.Module):
    """center=True STFT as a strided conv; returns re, im (b f t)"""
    def __init__(self, n_fft, hop, pad_mode):
        super().__init__()
        self.n_fft, self.hop, self.pad_mode = n_fft, hop, pad_mode
        k = torch.arange(n_fft // 2 + 1, dtype=torch.float64)[:, None]; n = torch.arange(n_fft, dtype=torch.float64)[None]
        w = hann(n_fft)[None]; ang = 2 * math.pi * k * n / n_fft
        self.register_buffer("kr", (torch.cos(ang) * w).float()[:, None, :])
        self.register_buffer("ki", (-torch.sin(ang) * w).float()[:, None, :])
    def forward(self, x):
        p = self.n_fft // 2
        x = F.pad(x[:, None], (p, p), mode=self.pad_mode)
        return F.conv1d(x, self.kr, stride=self.hop), F.conv1d(x, self.ki, stride=self.hop)

class ISTFT(nn.Module):
    def __init__(self, n_fft, hop):
        super().__init__()
        self.n_fft, self.hop = n_fft, hop
        nb = n_fft // 2 + 1; k = torch.arange(nb, dtype=torch.float64)[None]; n = torch.arange(n_fft, dtype=torch.float64)[:, None]
        c = torch.full((nb,), 2.0, dtype=torch.float64); c[0] = 1; c[-1] = 1
        ang = 2 * math.pi * n * k / n_fft; w = hann(n_fft)[:, None]
        self.register_buffer("br", (torch.cos(ang) * c / n_fft * w).float())   # (n_fft, nb)
        self.register_buffer("bi", (-torch.sin(ang) * c / n_fft * w).float())
        self.register_buffer("ola", torch.eye(n_fft).float()[:, None, :])        # (n_fft,1,n_fft)
        self.register_buffer("w2", (hann(n_fft) ** 2).float()[None, :, None])
    def forward(self, re, im):
        fr = torch.matmul(self.br, re) + torch.matmul(self.bi, im)  # (b n_fft t)
        y = F.conv_transpose1d(fr, self.ola, stride=self.hop)
        env = F.conv_transpose1d(self.w2.expand(1, self.n_fft, fr.shape[-1]), self.ola, stride=self.hop)
        p = self.n_fft // 2
        y = y[..., p:-p]; env = env[..., p:-p]
        return (y / torch.where(env > 1e-11, env, torch.ones_like(env)))[:, 0]

class Mel(nn.Module):
    def __init__(self, mel_fn):
        super().__init__()
        hp = mel_fn.hp
        self.stft = STFT(hp.n_fft, hp.hop_size, "constant")
        self.register_buffer("fb", mel_fn.melspec.mel_scale.fb.T.contiguous().float())  # (n_mels, nb)
        self.pre, self.minmag, self.mindb = hp.preemphasis, hp.stft_magnitude_min, mel_fn.min_level_db
    def forward(self, wav):
        wav = torch.cat([wav[:, :1], wav[:, 1:] - self.pre * wav[:, :-1]], 1)
        re, im = self.stft(wav)
        mag = torch.sqrt(re * re + im * im)
        mel = torch.matmul(self.fb, mag)
        db = torch.log10(torch.clamp(mel, min=self.minmag)) * 20
        return ((db - self.mindb) / (-self.mindb + 15))[..., :-1]

class Prep(nn.Module):
    def __init__(self, m):
        super().__init__()
        self.mel = Mel(m.mel_fn); self.norm = m.normalizer; self.den = m.denoiser.net; self.enc = m.lcfm.ae.encoder
        hop = m.denoiser.hp.hop_size; self.dstft = STFT(hop * 4, hop, "reflect"); self.distft = ISTFT(hop * 4, hop)
        self.zs = m.lcfm.z_scale
    def nrm(self, x): return (x - self.norm.running_mean) / self.norm.running_std
    def denoise(self, x):
        x = x / (x.abs().amax(-1, keepdim=True) + 1e-7)
        re, im = self.dstft(x); re, im = re[..., :-1], im[..., :-1]
        mag = torch.sqrt(re * re + im * im); nz = mag > 0
        cos = torch.where(nz, re / torch.where(nz, mag, torch.ones_like(mag)), torch.ones_like(mag)); sin = torch.where(nz, im / torch.where(nz, mag, torch.ones_like(mag)), torch.zeros_like(mag))
        o = self.den(torch.stack([mag, cos, sin], 1)); mm, r, i = o[:, 0], o[:, 1], o[:, 2]
        mm = torch.sigmoid(mm); r = torch.tanh(r); i = torch.tanh(i)
        mg = torch.sqrt(r * r + i * i + 1e-7); cr, sr_ = r / mg, i / mg
        smag = F.relu(mag * mm); scos = cos * cr - sin * sr_; ssin = sin * cr + cos * sr_
        sre, sim = smag * scos, smag * ssin
        sre = torch.cat([sre, sre[..., -1:]], -1); sim = torch.cat([sim, sim[..., -1:]], -1)
        y = self.distft(sre, sim)
        return F.pad(y, (0, x.shape[-1] - y.shape[-1]))
    def forward(self, wav, lambd):
        mo = self.nrm(self.mel(wav))
        md = self.nrm(self.mel(self.denoise(wav)))
        cond = lambd * md + (1 - lambd) * mo
        return cond, self.enc(mo) * self.zs

class CFMStep(nn.Module):
    def __init__(self, m): super().__init__(); self.c = m.lcfm.cfm
    def forward(self, psi, cond, t):
        return self.c.net(psi, l=cond, g=self.c.emb(t.clamp(0, 1)))

class IrDec(nn.Module):
    def __init__(self, m): super().__init__(); self.dec = m.lcfm.ae.decoder
    def forward(self, z): return self.dec(z)

class Voc(nn.Module):
    """vocoder on decoded features (1,160,T) and noise (1,128,T); no padding inside, so it can run in overlapping pieces"""
    def __init__(self, m): super().__init__(); self.v = m.vocoder
    def forward(self, feat, noise):
        v = self.v; h = v.conv_pre(noise)
        for b in v.blocks: h = b(h, feat)
        return v.conv_post(h)[:, 0]

class Dec(nn.Module):
    def __init__(self, m): super().__init__(); self.dec = m.lcfm.ae.decoder; self.v = m.vocoder
    def forward(self, z, noise):
        x = self.dec(z); v = self.v
        x = F.pad(x, (0, 10))
        h = v.conv_pre(noise)
        for b in v.blocks: h = b(h, x)
        h = v.conv_post(h)
        return h[..., : -v.scale_factor * 10][:, 0]

if __name__ == "__main__":
    run, out = sys.argv[1], Path(sys.argv[2]); out.mkdir(parents=True, exist_ok=True)
    m = build(run)
    print("params M", sum(p.numel() for p in m.parameters()) / 1e6)
    N = 44100 * 3 + 441; T = N // 420
    wav = torch.randn(1, N) * 0.1; wav = wav / wav.abs().max()
    prep, step, dec = Prep(m).eval(), CFMStep(m).eval(), Dec(m).eval()
    kw = dict(opset_version=17, do_constant_folding=True, dynamo=False)
    with torch.no_grad():
        torch.onnx.export(prep, (wav, torch.tensor([0.5])), out / "prep.onnx", input_names=["wav", "lambd"], output_names=["cond", "z0"], dynamic_axes={"wav": {1: "n"}, "cond": {2: "t"}, "z0": {2: "t"}}, **kw)
        cond, z0 = prep(wav, torch.tensor([0.5]))
        torch.onnx.export(step, (z0, cond, torch.tensor([0.3])), out / "cfm.onnx", input_names=["psi", "cond", "t"], output_names=["v"], dynamic_axes={"psi": {2: "t"}, "cond": {2: "t"}, "v": {2: "t"}}, **kw)
        torch.onnx.export(IrDec(m).eval(), (z0 / 5,), out / "irdec.onnx", input_names=["z"], output_names=["feat"], dynamic_axes={"z": {2: "t"}, "feat": {2: "t"}}, **kw)
        feat = IrDec(m)(z0 / 5)
        torch.onnx.export(Voc(m).eval(), (feat[..., :40], torch.randn(1, 128, 40)), out / "voc.onnx", input_names=["feat", "noise"], output_names=["wav"], dynamic_axes={"feat": {2: "t"}, "noise": {2: "t"}, "wav": {1: "n"}}, **kw)
    print("exported", [(p.name, p.stat().st_size // 1e6) for p in out.glob("*.onnx*")])
