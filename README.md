# Booth AI models

Downloadable AI models for [Booth](https://mayworks.com), the MayWorks podcast studio. Booth fetches these once, checks them against a fingerprint and stores them on your computer. You don't need to download anything from here yourself.

## Studio Max (`studiomax-v1` release)

Booth's Studio Max enhance style. It's **Resemble Enhance** by Resemble AI, converted to ONNX so it runs in the Booth desktop app's built-in AI engine (ONNX Runtime) on the processor or graphics card.

| File | What it is |
|---|---|
| `manifest.json` | File list, sizes and SHA-256 fingerprint of the weights |
| `prep.onnx`, `cfm.onnx`, `irdec.onnx`, `voc.onnx` | The four model graphs (denoise and features, flow-matching enhancer, latent decoder, vocoder) |
| `studiomax-weights.f16` | All weights in half precision. Booth expands them to full precision when it installs |

The model and weights are unchanged from Resemble AI's release, apart from the format conversion. Booth runs the speech-enhancement path with 32 solver steps, denoise strength 0.4 and prior temperature 0.5.

## Licence and credits

- **Resemble Enhance** © Resemble AI, MIT License. https://github.com/resemble-ai/resemble-enhance (weights: https://huggingface.co/ResembleAI/resemble-enhance)
- The vocoder includes code adapted from **BigVGAN** (NVIDIA, MIT License) and **alias-free-torch** (junjun3518, Apache License 2.0), as in the original repository.
- The pipeline recipe (cleaned input, re-synthesis, some of the cleaned voice blended back in) is adapted from **CrispVoice** by Francium Tech (MIT). https://github.com/Francium-Tech/CrispVoice

See `LICENSE` for the full MIT text.
