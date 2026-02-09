#!/usr/bin/env python3
"""Export DPDFNet PyTorch models to ONNX (spectrum-in, spectrum-out).

Exports the core model without STFT/ISTFT, since those are handled in Rust.
All complex tensor ops are replaced with real-valued equivalents so ONNX
can represent the graph.

Input:  spec [B, 1, T, F, 2]   (real-valued representation of complex spectrum)
Output: spec_enhanced [B, 1, T, F, 2], lsnr [B, T]

Usage:
    python DPDFNet/export_pytorch_to_onnx.py                          # all models
    python DPDFNet/export_pytorch_to_onnx.py --model_name dpdfnet2    # single model
"""

import argparse
import sys
from pathlib import Path
from typing import Tuple

SCRIPT_DIR = Path(__file__).parent
CHECKPOINT_DIR = SCRIPT_DIR / "model_zoo" / "checkpoints"
MODELS_DIR = SCRIPT_DIR / "model_zoo" / "stateless"

# Add DPDFNet source code to path
sys.path.insert(0, str(SCRIPT_DIR / "DPDFNet"))

import torch
from torch import Tensor, nn

MODELS = {
    "baseline":           {"sr": 16000, "win_len": 320, "dprnn_num_blocks": 0},
    "dpdfnet2":           {"sr": 16000, "win_len": 320, "dprnn_num_blocks": 2},
    "dpdfnet4":           {"sr": 16000, "win_len": 320, "dprnn_num_blocks": 4},
    "dpdfnet8":           {"sr": 16000, "win_len": 320, "dprnn_num_blocks": 8},
    "dpdfnet2_48khz_hr":  {"sr": 48000, "win_len": 960, "dprnn_num_blocks": 2},
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _magnitude_real(x: Tensor) -> Tensor:
    """Compute magnitude from real-valued [.., 2] tensor without complex types."""
    return torch.sqrt(x[..., 0] ** 2 + x[..., 1] ** 2)


# ---------------------------------------------------------------------------
# ONNX wrapper: spectrum-in / spectrum-out (no STFT/ISTFT, no complex types)
# ---------------------------------------------------------------------------

class DPDFNetONNX_16k(nn.Module):
    """ONNX-exportable wrapper for 16kHz DPDFNet models.

    Input:  spec [B, 1, T, F, 2]  (F = win_len//2 + 1, e.g. 161)
    Output: spec_enhanced [B, 1, T, F, 2], lsnr [B, T]
    """

    def __init__(self, model):
        super().__init__()
        # Copy needed sub-modules and buffers
        self.erb_fb = model.erb_fb            # [F, E]
        self.erb_to_db = model.erb_to_db
        self.nb_df = model.nb_df
        self.mask_method = model.mask_method

        self.erb_norm = model.erb_norm
        self.spec_norm = model.spec_norm
        self.pad_feat = model.pad_feat
        self.enc = model.enc
        self.erb_dec = model.erb_dec
        self.mask = model.mask
        self.df_dec = model.df_dec
        self.df_out_transform = model.df_out_transform
        self.df_op = model.df_op              # already replaced with DFreal

    def forward(self, spec: Tensor) -> Tuple[Tensor, Tensor]:
        # spec: [B, 1, T, F, 2]

        # --- Feature extraction (real-valued, no complex ops) ---
        # magnitude: sqrt(re^2 + im^2)
        magnitude = _magnitude_real(spec)            # [B, 1, T, F]
        magnitude = magnitude.squeeze(1)             # [B, T, F]

        feat_erb = (magnitude ** 2) @ self.erb_fb    # [B, T, E]
        if self.erb_to_db:
            feat_erb = 10 * torch.log10(feat_erb + 1e-10)

        feat_spec = spec[:, 0, :, :self.nb_df, :]   # [B, T, F', 2]

        # Normalization (SpecNorm patched to avoid complex ops)
        feat_erb = self.erb_norm(feat_erb)
        feat_spec = self.spec_norm(feat_spec)

        # Reshape
        feat_erb = feat_erb.unsqueeze(1)             # [B, 1, T, E]
        feat_spec = feat_spec.permute(0, 3, 1, 2)   # [B, 2, T, F']

        feat_erb = self.pad_feat(feat_erb)
        feat_spec = self.pad_feat(feat_spec)

        # --- Encoder / Decoder ---
        e0, e1, e2, e3, emb, c0, lsnr = self.enc(feat_erb, feat_spec)
        m = self.erb_dec(emb, e3, e2, e1, e0)

        df_coefs = self.df_dec(emb, c0)
        df_coefs = self.df_out_transform(df_coefs)

        # --- Apply mask + deep filter (all real-valued) ---
        if self.mask_method == 'before_df':
            spec = self.mask(spec, m)
            spec_e = self.df_op(spec.clone(), df_coefs)
        elif self.mask_method == 'separate':
            spec_m = self.mask(spec, m)
            spec_e = self.df_op(spec.clone(), df_coefs)
            spec_e[..., self.nb_df:, :] = spec_m[..., self.nb_df:, :]
        elif self.mask_method == 'after_df':
            spec_e = self.df_op(spec.clone(), df_coefs)
            spec_e = self.mask(spec_e, m)
        else:
            raise ValueError(f'Unknown mask_method: {self.mask_method}')

        return spec_e, lsnr


class DPDFNetONNX_48k(nn.Module):
    """ONNX-exportable wrapper for 48kHz DPDFNet model.

    Input:  spec [B, 1, T, F, 2]  (F = 481)
    Output: spec_enhanced [B, 1, T, F, 2], lsnr [B, T]
    """

    def __init__(self, model):
        super().__init__()
        self.erb_to_db = model.erb_to_db
        self.nb_df = model.nb_df
        self.mask_method = model.mask_method

        self.erb_norm = model.erb_norm        # MagNorm48
        self.spec_norm = model.spec_norm      # SpecNorm48
        self.pad_feat = model.pad_feat
        self.enc = model.enc
        self.erb_dec = model.erb_dec
        self.df_dec = model.df_dec
        self.df_out_transform = model.df_out_transform
        self.df_op = model.df_op              # already replaced with DFreal

    def forward(self, spec: Tensor) -> Tuple[Tensor, Tensor]:
        # spec: [B, 1, T, F, 2]

        # --- Feature extraction (real-valued) ---
        magnitude = _magnitude_real(spec)            # [B, 1, T, F]
        magnitude = magnitude.squeeze(1)             # [B, T, F]

        # 48kHz uses raw magnitude (not squared, no ERB filterbank)
        feat_erb = magnitude
        if self.erb_to_db:
            feat_erb = 10 * torch.log10(feat_erb + 1e-10)

        feat_spec = spec[:, 0, :, :self.nb_df, :]   # [B, T, F', 2]

        # Normalization
        feat_erb = self.erb_norm(feat_erb)
        feat_spec = self.spec_norm(feat_spec)

        # Reshape
        feat_erb = feat_erb.unsqueeze(1)             # [B, 1, T, F]
        feat_spec = feat_spec.permute(0, 3, 1, 2)   # [B, 2, T, F']

        feat_erb = self.pad_feat(feat_erb)
        feat_spec = self.pad_feat(feat_spec)

        # --- Encoder / Decoder ---
        e0, e1, e2, e3, emb, c0, lsnr = self.enc(feat_erb, feat_spec)
        m = self.erb_dec(emb, e3, e2, e1, e0)

        df_coefs = self.df_dec(emb, c0)
        df_coefs = self.df_out_transform(df_coefs)

        # --- Apply mask + deep filter (all real-valued) ---
        # 48kHz applies mask directly (no ERB inverse filterbank)
        if self.mask_method == 'before_df':
            spec = spec * m.unsqueeze(-1)
            spec_e = self.df_op(spec.clone(), df_coefs)
        elif self.mask_method == 'separate':
            spec_m = spec * m.unsqueeze(-1)
            spec_e = self.df_op(spec.clone(), df_coefs)
            spec_e[..., self.nb_df:, :] = spec_m[..., self.nb_df:, :]
        elif self.mask_method == 'after_df':
            spec_e = self.df_op(spec.clone(), df_coefs)
            spec_e = spec_e * m.unsqueeze(-1)
        else:
            raise ValueError(f'Unknown mask_method: {self.mask_method}')

        return spec_e, lsnr


# ---------------------------------------------------------------------------
# Export logic
# ---------------------------------------------------------------------------

def export_model(checkpoint_path: str, model_name: str, output_dir: str):
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}. Choose from: {list(MODELS.keys())}")
        sys.exit(1)

    cfg = MODELS[model_name]
    sr = cfg["sr"]
    win_len = cfg["win_len"]
    dprnn_num_blocks = cfg["dprnn_num_blocks"]
    freq_bins = win_len // 2 + 1
    hop_size = win_len // 2

    print(f"Exporting {model_name} (sr={sr}, win={win_len}, F={freq_bins}, dprnn={dprnn_num_blocks})")

    # Import and instantiate the original model
    is_48k = model_name == "dpdfnet2_48khz_hr"
    if is_48k:
        from dpdfnet_48khz_hr import DPDFNet48HR
        model = DPDFNet48HR()
    else:
        from dpdfnet import DPDFNet
        model = DPDFNet(
            n_fft=win_len,
            samplerate=sr,
            dprnn_num_blocks=dprnn_num_blocks,
        )

    # Load checkpoint
    sd = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()

    print(f"  Model loaded from {checkpoint_path}")
    num_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {num_params / 1e6:.2f}M")

    # Wrap in ONNX-compatible module (no STFT/ISTFT)
    # Note: Source code already uses real-valued ops (no complex types, no einops)
    if is_48k:
        onnx_model = DPDFNetONNX_48k(model)
    else:
        onnx_model = DPDFNetONNX_16k(model)
    onnx_model.eval()

    # Create output directory
    out_dir = Path(output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    # Dummy input: 100 frames of spectrum [B=1, C=1, T=100, F, 2]
    T = 100
    dummy_spec = torch.randn(1, 1, T, freq_bins, 2)
    onnx_path = out_dir / "model.onnx"

    # Verify forward pass works
    print(f"  Testing forward pass...")
    with torch.no_grad():
        spec_e, lsnr = onnx_model(dummy_spec)
    print(f"  Forward pass OK: spec_e={list(spec_e.shape)}, lsnr={list(lsnr.shape)}")

    print(f"  Exporting to {onnx_path}...")
    torch.onnx.export(
        onnx_model,
        (dummy_spec,),
        str(onnx_path),
        opset_version=17,
        input_names=["spec"],
        output_names=["spec_enhanced", "lsnr"],
        dynamic_axes={
            "spec": {2: "num_frames"},
            "spec_enhanced": {2: "num_frames"},
            "lsnr": {1: "num_frames"},
        },
    )

    print(f"  [OK] Exported to {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # Write config
    config_path = out_dir / "config.ini"
    config_path.write_text(
        f"[model]\n"
        f"sample_rate = {sr}\n"
        f"win_len = {win_len}\n"
        f"hop_size = {hop_size}\n"
        f"freq_bins = {freq_bins}\n"
        f"export_mode = pytorch_core\n"
        f"interface = spec_real\n"
    )
    print(f"  Written {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Export DPDFNet to ONNX (spectrum-in/out)")
    parser.add_argument("--checkpoint", help="Path to .pth file (default: model_zoo/checkpoints/<name>.pth)")
    parser.add_argument("--model_name", choices=sorted(MODELS.keys()), help="Single model to export (default: all)")
    parser.add_argument("--output_dir", default=str(MODELS_DIR), help="Output directory")
    args = parser.parse_args()

    if args.model_name:
        # Single model
        ckpt = args.checkpoint or str(CHECKPOINT_DIR / f"{args.model_name}.pth")
        export_model(ckpt, args.model_name, args.output_dir)
    else:
        # All models
        print("DPDFNet PyTorch -> ONNX Export (all models)")
        print("=" * 50)
        success = 0
        for name in MODELS:
            ckpt = args.checkpoint or str(CHECKPOINT_DIR / f"{name}.pth")
            if not Path(ckpt).exists():
                print(f"\n[SKIP] {name}: checkpoint not found at {ckpt}")
                continue
            try:
                export_model(ckpt, name, args.output_dir)
                success += 1
            except Exception as e:
                print(f"\n[FAIL] {name}: {e}")
                import traceback
                traceback.print_exc()
        print(f"\n{'=' * 50}")
        print(f"Exported {success}/{len(MODELS)} models")


if __name__ == "__main__":
    main()
