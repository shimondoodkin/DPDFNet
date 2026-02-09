#!/usr/bin/env python3
"""Export DPDFNet PyTorch models to stateful single-frame ONNX.

Each call processes T=1 spectrum frame. All recurrent states (GRU, DPRNN,
normalization running averages, temporal conv buffers) are externalized as
explicit ONNX inputs/outputs — matching the DeepVQE pattern.

Input:  spec [1,1,1,F,2] + state tensors
Output: spec_enhanced [1,1,1,F,2] + lsnr [1,1] + updated state tensors

Usage:
    python DPDFNet/export_stateful_onnx.py                          # all models
    python DPDFNet/export_stateful_onnx.py --model_name dpdfnet2    # single model
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple

SCRIPT_DIR = Path(__file__).parent
CHECKPOINT_DIR = SCRIPT_DIR / "model_zoo" / "checkpoints"
MODELS_DIR = SCRIPT_DIR / "model_zoo" / "stateful"

# Add DPDFNet source code to path and script dir for importing export_pytorch_to_onnx
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "DPDFNet"))

import torch
from torch import Tensor, nn
import torch.nn.functional as torchF
import numpy as np

# Reuse helpers from existing export script
from export_pytorch_to_onnx import (  # noqa: E402 - resolved via sys.path
    MODELS,
    _magnitude_real,
)
from multiframe import df_real as _df_real_onnx  # noqa: E402


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _run_conv_skip_pad(conv_seq: nn.Sequential, x: Tensor) -> Tensor:
    """Run Conv2dNormAct/ConvTranspose2dNormAct on pre-padded input (skip ConstantPad2d)."""
    for layer in conv_seq:
        if isinstance(layer, nn.ConstantPad2d):
            continue
        x = layer(x)
    return x


def _squeezed_gru_step(sgru, x: Tensor, h: Tensor) -> Tuple[Tensor, Tensor]:
    """Run SqueezedGRU_S for T=1 with explicit hidden state.

    Bypasses _execute_rnn which ignores state in eval mode.
    """
    x_in = sgru.linear_in(x)
    out, h_new = sgru.gru(x_in, h)
    out = sgru.linear_out(out)
    if sgru.gru_skip is not None:
        out = out + sgru.gru_skip(x)
    return out, h_new


def _dprnn_forward(dprnn, x: Tensor, inter_states: List[Tensor]) -> Tuple[Tensor, List[Tensor]]:
    """Run DPRNN for T=1 with explicit inter_gru states.

    intra_gru (bidirectional on freq) needs no state.
    inter_gru (unidirectional on time) is stateful.
    """
    if isinstance(dprnn, nn.Identity):
        return x, inter_states

    x = dprnn.input_proj(x)
    new_states = []
    for i, block in enumerate(dprnn.blocks):
        B, C, T, F = x.shape  # T=1

        # Intra: bidirectional on freq axis, no state needed
        x_intra = x.permute(0, 2, 3, 1).reshape(B * T, F, C)  # (B*T, F, C)
        x_intra, _ = block.intra_gru(x_intra)
        x_intra = block.ln_intra(block.fc_intra(x_intra))
        x_intra = x_intra.reshape(B, T, F, C).permute(0, 3, 1, 2)  # (B, C, T, F)
        x = x + x_intra

        # Inter: unidirectional on time axis, STATEFUL
        x_inter = x.permute(0, 3, 2, 1).reshape(B * F, T, C)  # (B*F, T, C)
        # inter_states[i] shape: [1, B*F, hidden_dim]
        x_inter, h_new = block.inter_gru(x_inter, inter_states[i])
        x_inter = block.ln_inter(block.fc_inter(x_inter))
        x_inter = x_inter.reshape(B, F, T, C).permute(0, 3, 2, 1)  # (B, C, T, F)
        x = x + x_inter
        new_states.append(h_new)

    x = dprnn.output_proj(x)
    return x, new_states


def _df_with_window(window: Tensor, coefs: Tensor, num_freqs: int) -> Tensor:
    """Apply deep filter using pre-built 5-frame window.

    Args:
        window: [B, C, 5, F, 2] — past 4 frames + current frame
        coefs: [B, O, 1, F', 2] — filter coefficients for current frame
        num_freqs: number of DF frequency bins
    """
    # Build stacked windows: [B, C, N=5, T=1, F, 2]
    windows_list = [window[:, :, i:i+1, :, :] for i in range(5)]
    spec_u = torch.stack(windows_list, dim=2)  # [B, C, N=5, T=1, F, 2]

    spec_f = spec_u[..., :num_freqs, :]  # [B, C, N=5, T=1, F', 2]

    # coefs: [B, O, T=1, F', 2] -> [B, 1, N=5, T=1, F', 2]
    new_shape = [coefs.shape[0], -1, 5] + list(coefs.shape[2:])
    coefs = coefs.view(new_shape)  # [B, 1, N=5, T=1, F', 2]

    filtered = _df_real_onnx(spec_f, coefs)  # [B, C, T=1, F', 2]

    # High-freq passthrough from current frame (index 4)
    spec_hi = window[:, :, 4:5, num_freqs:, :]  # [B, C, 1, F_hi, 2]
    return torch.cat([filtered, spec_hi], dim=-2)  # [B, C, 1, F, 2]


def _get_conv_tpad(conv_seq: nn.Sequential) -> int:
    """Get the temporal padding size from a Conv2dNormAct's ConstantPad2d."""
    for layer in conv_seq:
        if isinstance(layer, nn.ConstantPad2d):
            # pad = (left, right, top, bottom) where top is temporal pad-before
            return layer.padding[2]  # top padding = kernel_t - 1
    return 0


# ---------------------------------------------------------------------------
# 16kHz Stateful Wrapper
# ---------------------------------------------------------------------------

class StatefulDPDFNet_16k(nn.Module):
    """Stateful single-frame wrapper for 16kHz DPDFNet models.

    All internal state is externalized as explicit inputs/outputs.
    """

    def __init__(self, model):
        super().__init__()
        # Copy sub-modules
        self.erb_fb = model.erb_fb          # [F, E]
        self.erb_to_db = model.erb_to_db
        self.nb_df = model.nb_df
        self.nb_erb = model.nb_erb
        self.freq_bins = model.freq_bins
        self.mask_method = model.mask_method
        self.df_order = model.df_order

        # Encoder
        self.erb_conv0 = model.enc.erb_conv0
        self.erb_conv1 = model.enc.erb_conv1
        self.erb_conv2 = model.enc.erb_conv2
        self.erb_conv3 = model.enc.erb_conv3
        self.df_conv0 = model.enc.df_conv0
        self.df_conv1 = model.enc.df_conv1
        self.dprnn_erb = model.enc.dprnn_erb
        self.dprnn_df = model.enc.dprnn_df
        self.df_fc_emb = model.enc.df_fc_emb
        self.combine = model.enc.combine
        self.emb_gru = model.enc.emb_gru
        self.lsnr_fc = model.enc.lsnr_fc
        self.lsnr_scale = model.enc.lsnr_scale
        self.lsnr_offset = model.enc.lsnr_offset

        # ERB Decoder
        self.erb_dec_emb_gru = model.erb_dec.emb_gru
        self.conv3p = model.erb_dec.conv3p
        self.convt3 = model.erb_dec.convt3
        self.conv2p = model.erb_dec.conv2p
        self.convt2 = model.erb_dec.convt2
        self.conv1p = model.erb_dec.conv1p
        self.convt1 = model.erb_dec.convt1
        self.conv0p = model.erb_dec.conv0p
        self.conv0_out = model.erb_dec.conv0_out

        # Mask
        self.mask = model.mask

        # DF Decoder
        self.df_gru = model.df_dec.df_gru
        self.df_skip = model.df_dec.df_skip
        self.df_convp = model.df_dec.df_convp
        self.df_out = model.df_dec.df_out
        self.df_bins = model.df_dec.df_bins
        self.df_out_ch = model.df_dec.df_out_ch

        # DF transform + op
        self.df_out_transform = model.df_out_transform
        self.df_op_num_freqs = model.df_op.num_freqs

        # Normalization params
        self.erb_norm_alpha = model.erb_norm.alpha
        self.erb_norm_eps = model.erb_norm.eps
        self.spec_norm_alpha = model.spec_norm.alpha
        self.spec_norm_eps = model.spec_norm.eps

        # Compute buffer sizes from conv padding
        self.erb_conv0_tpad = _get_conv_tpad(self.erb_conv0)
        self.df_conv0_tpad = _get_conv_tpad(self.df_conv0)
        self.df_convp_tpad = _get_conv_tpad(self.df_convp)

        # Count DPRNN blocks
        self.has_dprnn = not isinstance(self.dprnn_erb, nn.Identity)
        self.n_dprnn_blocks = len(self.dprnn_erb.blocks) if self.has_dprnn else 0

    def forward(self, spec: Tensor,
                erb_norm_mu: Tensor, spec_norm_s: Tensor,
                enc_gru_h: Tensor, erb_dec_gru_h: Tensor, df_dec_gru_h: Tensor,
                *args) -> Tuple:
        """Process single frame with explicit state.

        Remaining *args are: dprnn states..., erb_conv0_buf, df_conv0_buf,
        df_convp_buf, df_spec_buf
        """
        # Unpack variable-length DPRNN states and fixed conv buffers
        n_dprnn = self.n_dprnn_blocks * 2  # erb + df
        dprnn_states = list(args[:n_dprnn])
        erb_conv0_buf = args[n_dprnn]
        df_conv0_buf = args[n_dprnn + 1]
        df_convp_buf = args[n_dprnn + 2]
        df_spec_buf = args[n_dprnn + 3]

        # spec: [1, 1, 1, F, 2]
        B = 1

        # 1. Feature extraction
        magnitude = _magnitude_real(spec).squeeze(1)  # [1, 1, F]

        feat_erb = (magnitude ** 2) @ self.erb_fb  # [1, 1, E]
        if self.erb_to_db:
            feat_erb = 10 * torch.log10(feat_erb + 1e-10)

        feat_spec = spec[:, 0, :, :self.nb_df, :]  # [1, 1, F', 2]

        # 2. Stateful ErbNorm (T=1)
        erb_norm_mu = self.erb_norm_alpha * erb_norm_mu + (1 - self.erb_norm_alpha) * feat_erb[:, 0]
        # Use fixed variance = 40^2 = 1600 (dynamic_var=False for all models)
        feat_erb_normed = (feat_erb[:, 0] - erb_norm_mu) / (torch.sqrt(torch.tensor(1600.0)) + self.erb_norm_eps)
        feat_erb_normed = feat_erb_normed.unsqueeze(1)  # [1, 1, E]

        # 3. Stateful SpecNorm (T=1)
        x_abs = _magnitude_real(feat_spec)  # [1, 1, F']
        spec_norm_s = self.spec_norm_alpha * spec_norm_s + (1 - self.spec_norm_alpha) * x_abs[:, 0]
        feat_spec_r = feat_spec[:, 0, :, 0] / torch.sqrt(spec_norm_s + self.spec_norm_eps)
        feat_spec_i = feat_spec[:, 0, :, 1] / torch.sqrt(spec_norm_s + self.spec_norm_eps)
        feat_spec_normed = torch.stack([feat_spec_r, feat_spec_i], dim=-1).unsqueeze(1)  # [1, 1, F', 2]

        # 4. Reshape for encoder
        feat_erb_enc = feat_erb_normed.unsqueeze(1)  # [1, 1, 1, E]
        feat_spec_enc = feat_spec_normed.permute(0, 3, 1, 2)  # [1, 2, 1, F']

        # No pad_feat for stateful (purely causal, no lookahead)

        # 5. ERB encoder path with temporal conv buffer
        e0_in = torch.cat([erb_conv0_buf, feat_erb_enc], dim=2)  # [1, 1, 1+tpad, E]
        erb_conv0_buf_out = e0_in[:, :, -self.erb_conv0_tpad:, :]
        e0 = _run_conv_skip_pad(self.erb_conv0, e0_in)  # [1, 64, 1, E]

        e1 = self.erb_conv1(e0)  # kernel_t=1, no temporal pad needed
        e2 = self.erb_conv2(e1)
        e3 = self.erb_conv3(e2)

        # 6. DPRNN on ERB path
        dprnn_erb_states = dprnn_states[:self.n_dprnn_blocks]
        dprnn_df_states = dprnn_states[self.n_dprnn_blocks:]
        e3_dprnn, dprnn_erb_states_out = _dprnn_forward(self.dprnn_erb, e3, dprnn_erb_states)

        # 7. DF encoder path with temporal conv buffer
        c0_in = torch.cat([df_conv0_buf, feat_spec_enc], dim=2)  # [1, 2, 1+tpad, F']
        df_conv0_buf_out = c0_in[:, :, -self.df_conv0_tpad:, :]
        c0 = _run_conv_skip_pad(self.df_conv0, c0_in)  # [1, 64, 1, F']

        c1 = self.df_conv1(c0)  # kernel_t=1

        # 8. DPRNN on DF path
        c1_dprnn, dprnn_df_states_out = _dprnn_forward(self.dprnn_df, c1, dprnn_df_states)

        # 9. Embedding combination
        cemb = c1_dprnn.permute(0, 2, 3, 1).flatten(2)  # [1, 1, C*F/2]
        cemb = self.df_fc_emb(cemb)
        emb = e3_dprnn.permute(0, 2, 3, 1).flatten(2)  # [1, 1, C*F/4]
        emb = self.combine(emb, cemb)

        # 10. Stateful encoder GRU
        emb, enc_gru_h_out = _squeezed_gru_step(self.emb_gru, emb, enc_gru_h)

        # LSNR
        lsnr = self.lsnr_fc(emb).squeeze(-1) * self.lsnr_scale + self.lsnr_offset  # [1, 1]

        # 11. Stateful ERB decoder GRU
        _, _, _, f8 = e3.shape
        emb_dec, erb_dec_gru_h_out = _squeezed_gru_step(self.erb_dec_emb_gru, emb, erb_dec_gru_h)
        emb_dec = emb_dec.view(B, 1, f8, -1).permute(0, 3, 1, 2)  # [1, C, 1, F/4]

        # Decoder conv pathway
        e3_dec = self.convt3(self.conv3p(e3) + emb_dec)
        e2_dec = self.convt2(self.conv2p(e2) + e3_dec)
        e1_dec = self.convt1(self.conv1p(e1) + e2_dec)
        m = self.conv0_out(self.conv0p(e0) + e1_dec)  # [1, 1, 1, E]

        # 12. Stateful DF decoder GRU
        c_df, df_dec_gru_h_out = _squeezed_gru_step(self.df_gru, emb, df_dec_gru_h)
        if self.df_skip is not None:
            c_df = c_df + self.df_skip(emb)

        # df_convp with temporal buffer
        c0_convp_in = torch.cat([df_convp_buf, c0], dim=2)  # [1, 64, 1+tpad, F']
        df_convp_buf_out = c0_convp_in[:, :, -self.df_convp_tpad:, :]
        c0_pathway = _run_conv_skip_pad(self.df_convp, c0_convp_in)  # [1, df_out_ch, 1, F']
        c0_pathway = c0_pathway.permute(0, 2, 3, 1)  # [1, 1, F', df_out_ch]

        c_out = self.df_out(c_df)  # [1, 1, F'*df_out_ch]
        c_out = c_out.view(B, 1, self.df_bins, self.df_out_ch) + c0_pathway
        df_coefs = self.df_out_transform(c_out)  # [1, O, 1, F', 2]

        # 13. Apply mask
        if self.mask_method == 'before_df':
            spec_masked = self.mask(spec, m)
        elif self.mask_method == 'after_df':
            spec_masked = spec.clone()
        elif self.mask_method == 'separate':
            spec_masked = spec.clone()
        else:
            spec_masked = self.mask(spec, m)

        # 14. Deep filter with buffer
        df_window = torch.cat([df_spec_buf, spec_masked], dim=2)  # [1, 1, 5, F, 2]
        df_spec_buf_out = df_window[:, :, -self.df_order + 1:, :, :]  # keep last 4

        spec_e = _df_with_window(df_window, df_coefs, self.df_op_num_freqs)

        # Handle mask_method variants
        if self.mask_method == 'after_df':
            spec_e = self.mask(spec_e, m)
        elif self.mask_method == 'separate':
            spec_m = self.mask(spec, m)
            spec_e[..., self.nb_df:, :] = spec_m[..., self.nb_df:, :]

        # 15. Collect all DPRNN states
        dprnn_states_out = dprnn_erb_states_out + dprnn_df_states_out

        return (spec_e, lsnr,
                erb_norm_mu, spec_norm_s,
                enc_gru_h_out, erb_dec_gru_h_out, df_dec_gru_h_out,
                *dprnn_states_out,
                erb_conv0_buf_out, df_conv0_buf_out,
                df_convp_buf_out, df_spec_buf_out)


# ---------------------------------------------------------------------------
# 48kHz Stateful Wrapper
# ---------------------------------------------------------------------------

class StatefulDPDFNet_48k(nn.Module):
    """Stateful single-frame wrapper for 48kHz DPDFNet model."""

    def __init__(self, model):
        super().__init__()
        self.erb_to_db = model.erb_to_db
        self.nb_df = model.nb_df
        self.nb_erb = model.nb_erb
        self.freq_bins = model.freq_bins
        self.mask_method = model.mask_method
        self.df_order = model.df_order

        # Encoder
        self.erb_conv0 = model.enc.erb_conv0
        self.erb_conv1 = model.enc.erb_conv1
        self.erb_conv2 = model.enc.erb_conv2
        self.erb_conv3 = model.enc.erb_conv3
        self.df_conv0 = model.enc.df_conv0
        self.df_conv1 = model.enc.df_conv1
        self.dprnn_erb = model.enc.dprnn_erb
        self.dprnn_df = model.enc.dprnn_df
        self.erb_fc_emb = model.enc.erb_fc_emb
        self.df_fc_emb = model.enc.df_fc_emb
        self.combine = model.enc.combine
        self.emb_gru = model.enc.emb_gru
        self.lsnr_fc = model.enc.lsnr_fc
        self.lsnr_scale = model.enc.lsnr_scale
        self.lsnr_offset = model.enc.lsnr_offset

        # ERB Decoder (48kHz has erb_fc_emb in decoder)
        self.erb_dec_emb_gru = model.erb_dec.emb_gru
        self.erb_dec_fc_emb = model.erb_dec.erb_fc_emb
        self.conv3p = model.erb_dec.conv3p
        self.convt3 = model.erb_dec.convt3
        self.conv2p = model.erb_dec.conv2p
        self.convt2 = model.erb_dec.convt2
        self.conv1p = model.erb_dec.conv1p
        self.convt1 = model.erb_dec.convt1
        self.conv0p = model.erb_dec.conv0p
        self.conv0_out = model.erb_dec.conv0_out

        # Mask
        self.mask = model.mask

        # DF Decoder
        self.df_gru = model.df_dec.df_gru
        self.df_skip = model.df_dec.df_skip
        self.df_convp = model.df_dec.df_convp
        self.df_out = model.df_dec.df_out
        self.df_bins = model.df_dec.df_bins
        self.df_out_ch = model.df_dec.df_out_ch

        # DF transform + op
        self.df_out_transform = model.df_out_transform
        self.df_op_num_freqs = model.df_op.num_freqs

        # Normalization params (48kHz uses MagNorm48 and SpecNorm48)
        self.erb_norm_alpha = model.erb_norm.alpha
        self.erb_norm_eps = model.erb_norm.eps
        self.spec_norm_alpha = model.spec_norm.alpha
        self.spec_norm_eps = model.spec_norm.eps

        # Compute buffer sizes from conv padding
        self.erb_conv0_tpad = _get_conv_tpad(self.erb_conv0)
        self.df_conv0_tpad = _get_conv_tpad(self.df_conv0)
        self.df_convp_tpad = _get_conv_tpad(self.df_convp)

        # Count DPRNN blocks
        self.has_dprnn = not isinstance(self.dprnn_erb, nn.Identity)
        self.n_dprnn_blocks = len(self.dprnn_erb.blocks) if self.has_dprnn else 0

    def forward(self, spec: Tensor,
                erb_norm_mu: Tensor, spec_norm_s: Tensor,
                enc_gru_h: Tensor, erb_dec_gru_h: Tensor, df_dec_gru_h: Tensor,
                *args) -> Tuple:
        # Unpack variable-length DPRNN states and fixed conv buffers
        n_dprnn = self.n_dprnn_blocks * 2
        dprnn_states = list(args[:n_dprnn])
        erb_conv0_buf = args[n_dprnn]
        df_conv0_buf = args[n_dprnn + 1]
        df_convp_buf = args[n_dprnn + 2]
        df_spec_buf = args[n_dprnn + 3]

        B = 1

        # 1. Feature extraction (48kHz: raw magnitude, no ERB filterbank)
        magnitude = _magnitude_real(spec).squeeze(1)  # [1, 1, F]

        feat_erb = magnitude  # No squaring, no ERB filterbank
        if self.erb_to_db:
            feat_erb = 10 * torch.log10(feat_erb + 1e-10)

        feat_spec = spec[:, 0, :, :self.nb_df, :]  # [1, 1, F', 2]

        # 2. Stateful MagNorm48 (T=1, dynamic_var=False)
        erb_norm_mu = self.erb_norm_alpha * erb_norm_mu + (1 - self.erb_norm_alpha) * feat_erb[:, 0]
        feat_erb_normed = (feat_erb[:, 0] - erb_norm_mu) / (torch.sqrt(torch.tensor(1600.0)) + self.erb_norm_eps)
        feat_erb_normed = feat_erb_normed.unsqueeze(1)  # [1, 1, F]

        # 3. Stateful SpecNorm48 (T=1)
        x_abs = _magnitude_real(feat_spec)  # [1, 1, F']
        spec_norm_s = self.spec_norm_alpha * spec_norm_s + (1 - self.spec_norm_alpha) * x_abs[:, 0]
        feat_spec_r = feat_spec[:, 0, :, 0] / torch.sqrt(spec_norm_s + self.spec_norm_eps)
        feat_spec_i = feat_spec[:, 0, :, 1] / torch.sqrt(spec_norm_s + self.spec_norm_eps)
        feat_spec_normed = torch.stack([feat_spec_r, feat_spec_i], dim=-1).unsqueeze(1)

        # 4. Reshape for encoder
        feat_erb_enc = feat_erb_normed.unsqueeze(1)  # [1, 1, 1, F]
        feat_spec_enc = feat_spec_normed.permute(0, 3, 1, 2)  # [1, 2, 1, F']

        # 5. ERB encoder path (48kHz: feat_erb[..., :-1] -> 481->480)
        erb_input = feat_erb_enc[..., :-1]  # [1, 1, 1, 480]

        e0_in = torch.cat([erb_conv0_buf, erb_input], dim=2)
        erb_conv0_buf_out = e0_in[:, :, -self.erb_conv0_tpad:, :]
        e0 = _run_conv_skip_pad(self.erb_conv0, e0_in)

        e1 = self.erb_conv1(e0)  # fstride=3
        e2 = self.erb_conv2(e1)  # fstride=2
        e3 = self.erb_conv3(e2)  # fstride=2

        # 6. DPRNN on ERB path
        dprnn_erb_states = dprnn_states[:self.n_dprnn_blocks]
        dprnn_df_states = dprnn_states[self.n_dprnn_blocks:]
        e3_dprnn, dprnn_erb_states_out = _dprnn_forward(self.dprnn_erb, e3, dprnn_erb_states)

        # 7. DF encoder path
        c0_in = torch.cat([df_conv0_buf, feat_spec_enc], dim=2)
        df_conv0_buf_out = c0_in[:, :, -self.df_conv0_tpad:, :]
        c0 = _run_conv_skip_pad(self.df_conv0, c0_in)

        c1 = self.df_conv1(c0)

        # 8. DPRNN on DF path
        c1_dprnn, dprnn_df_states_out = _dprnn_forward(self.dprnn_df, c1, dprnn_df_states)

        # 9. Embedding combination (48kHz has separate erb_fc_emb and df_fc_emb)
        cemb = c1_dprnn.permute(0, 2, 3, 1).flatten(2)
        cemb = self.df_fc_emb(cemb)
        emb = e3_dprnn.permute(0, 2, 3, 1).flatten(2)
        emb = self.erb_fc_emb(emb)
        emb = self.combine(emb, cemb)

        # 10. Stateful encoder GRU
        emb, enc_gru_h_out = _squeezed_gru_step(self.emb_gru, emb, enc_gru_h)

        # LSNR
        lsnr = self.lsnr_fc(emb).squeeze(-1) * self.lsnr_scale + self.lsnr_offset

        # 11. Stateful ERB decoder GRU
        _, _, _, f8 = e3.shape
        emb_dec, erb_dec_gru_h_out = _squeezed_gru_step(self.erb_dec_emb_gru, emb, erb_dec_gru_h)
        # 48kHz: extra erb_fc_emb in decoder
        emb_dec = self.erb_dec_fc_emb(emb_dec)
        emb_dec = emb_dec.view(B, 1, f8, -1).permute(0, 3, 1, 2)

        # Decoder conv pathway (48kHz has different strides: 2, 2, 3)
        e3_dec = self.convt3(self.conv3p(e3) + emb_dec)
        e2_dec = self.convt2(self.conv2p(e2) + e3_dec)
        e1_dec = self.convt1(self.conv1p(e1) + e2_dec)
        m = self.conv0_out(self.conv0p(e0) + e1_dec)
        # 48kHz: mirror duplication for highest bin
        m = torchF.pad(m, pad=(0, 1, 0, 0), mode='reflect')  # [1, 1, 1, 481]

        # 12. Stateful DF decoder GRU
        c_df, df_dec_gru_h_out = _squeezed_gru_step(self.df_gru, emb, df_dec_gru_h)
        if self.df_skip is not None:
            c_df = c_df + self.df_skip(emb)

        # df_convp with temporal buffer
        c0_convp_in = torch.cat([df_convp_buf, c0], dim=2)
        df_convp_buf_out = c0_convp_in[:, :, -self.df_convp_tpad:, :]
        c0_pathway = _run_conv_skip_pad(self.df_convp, c0_convp_in)
        c0_pathway = c0_pathway.permute(0, 2, 3, 1)

        c_out = self.df_out(c_df)
        c_out = c_out.view(B, 1, self.df_bins, self.df_out_ch) + c0_pathway
        df_coefs = self.df_out_transform(c_out)

        # 13. Apply mask (48kHz: direct multiply, no ERB inverse)
        if self.mask_method == 'before_df':
            spec_masked = spec * m.unsqueeze(-1)
        elif self.mask_method == 'after_df':
            spec_masked = spec.clone()
        elif self.mask_method == 'separate':
            spec_masked = spec.clone()
        else:
            spec_masked = spec * m.unsqueeze(-1)

        # 14. Deep filter with buffer
        df_window = torch.cat([df_spec_buf, spec_masked], dim=2)
        df_spec_buf_out = df_window[:, :, -self.df_order + 1:, :, :]

        spec_e = _df_with_window(df_window, df_coefs, self.df_op_num_freqs)

        if self.mask_method == 'after_df':
            spec_e = spec_e * m.unsqueeze(-1)
        elif self.mask_method == 'separate':
            spec_m = spec * m.unsqueeze(-1)
            spec_e[..., self.nb_df:, :] = spec_m[..., self.nb_df:, :]

        # 15. Collect DPRNN states
        dprnn_states_out = dprnn_erb_states_out + dprnn_df_states_out

        return (spec_e, lsnr,
                erb_norm_mu, spec_norm_s,
                enc_gru_h_out, erb_dec_gru_h_out, df_dec_gru_h_out,
                *dprnn_states_out,
                erb_conv0_buf_out, df_conv0_buf_out,
                df_convp_buf_out, df_spec_buf_out)


# ---------------------------------------------------------------------------
# Initial state computation
# ---------------------------------------------------------------------------

def compute_initial_states(model_name: str, model, is_48k: bool):
    """Compute initial state tensors and their metadata.

    Introspects the model to get correct GRU num_layers and dimensions.
    """
    cfg = MODELS[model_name]
    freq_bins = cfg["win_len"] // 2 + 1
    dprnn_num_blocks = cfg["dprnn_num_blocks"]

    nb_df = model.nb_df
    conv_ch = 64
    if is_48k:
        nb_erb = 481
        nb_erb_input = 480  # 481 - 1 (trimmed in forward)
    else:
        nb_erb = model.nb_erb
        nb_erb_input = nb_erb

    # Read actual GRU configs from model
    enc_gru = model.enc.emb_gru.gru
    erb_dec_gru = model.erb_dec.emb_gru.gru
    df_dec_gru = model.df_dec.df_gru.gru

    enc_gru_layers = enc_gru.num_layers
    erb_dec_gru_layers = erb_dec_gru.num_layers
    df_dec_gru_layers = df_dec_gru.num_layers
    gru_hidden = enc_gru.hidden_size

    print(f"  GRU layers: enc={enc_gru_layers}, erb_dec={erb_dec_gru_layers}, df_dec={df_dec_gru_layers}, hidden={gru_hidden}")

    states = []

    # 1. erb_norm_mu
    if is_48k:
        from init_norms import InitMagNorm
        init_mag = InitMagNorm()
        mu_init = init_mag.get_ampirical_mu_0(freq_bins).unsqueeze(0)  # [1, 481]
    else:
        # ErbNorm init: linear from -60 to -90 over 32 bands
        mu_init = torch.linspace(-60.0, -90.0, nb_erb).unsqueeze(0)  # [1, 32]
    states.append(("erb_norm_mu", list(mu_init.shape), mu_init))

    # 2. spec_norm_s
    if is_48k:
        from init_norms import InitSpecNorm
        init_spec = InitSpecNorm()
        s_init = init_spec.get_ampirical_s_0(nb_df).unsqueeze(0)  # [1, 96]
    else:
        s_init = torch.linspace(0.001, 0.0001, nb_df).unsqueeze(0)  # [1, 96]
    states.append(("spec_norm_s", list(s_init.shape), s_init))

    # 3. enc_gru_h — shape [num_layers, batch, hidden]
    states.append(("enc_gru_h", [enc_gru_layers, 1, gru_hidden],
                   torch.zeros(enc_gru_layers, 1, gru_hidden)))

    # 4. erb_dec_gru_h
    states.append(("erb_dec_gru_h", [erb_dec_gru_layers, 1, gru_hidden],
                   torch.zeros(erb_dec_gru_layers, 1, gru_hidden)))

    # 5. df_dec_gru_h
    states.append(("df_dec_gru_h", [df_dec_gru_layers, 1, gru_hidden],
                   torch.zeros(df_dec_gru_layers, 1, gru_hidden)))

    # 6. DPRNN states (if any)
    if dprnn_num_blocks > 0:
        # Get actual DPRNN inter_gru hidden_size and num_layers
        erb_dprnn = model.enc.dprnn_erb
        df_dprnn = model.enc.dprnn_df

        # Compute F dimension after conv path by running a dummy forward
        # ERB path: after erb_conv0..erb_conv3
        dummy_erb = torch.zeros(1, 1, 1, nb_erb_input)
        with torch.no_grad():
            e0 = model.enc.erb_conv0(dummy_erb)
            e1 = model.enc.erb_conv1(e0)
            e2 = model.enc.erb_conv2(e1)
            e3 = model.enc.erb_conv3(e2)
        dprnn_erb_f = e3.shape[3]  # F dimension

        # DF path: after df_conv0, df_conv1
        dummy_df = torch.zeros(1, 2, 1, nb_df)
        with torch.no_grad():
            c0 = model.enc.df_conv0(dummy_df)
            c1 = model.enc.df_conv1(c0)
        dprnn_df_f = c1.shape[3]  # F dimension

        erb_inter_hidden = erb_dprnn.blocks[0].inter_gru.hidden_size
        erb_inter_layers = erb_dprnn.blocks[0].inter_gru.num_layers
        df_inter_hidden = df_dprnn.blocks[0].inter_gru.hidden_size
        df_inter_layers = df_dprnn.blocks[0].inter_gru.num_layers

        print(f"  DPRNN erb: F={dprnn_erb_f}, inter_hidden={erb_inter_hidden}, inter_layers={erb_inter_layers}")
        print(f"  DPRNN df:  F={dprnn_df_f}, inter_hidden={df_inter_hidden}, inter_layers={df_inter_layers}")

        # inter_gru state: [num_layers, B*F, hidden] for T=1
        for i in range(dprnn_num_blocks):
            states.append((f"dprnn_erb_h{i}",
                          [erb_inter_layers, dprnn_erb_f, erb_inter_hidden],
                          torch.zeros(erb_inter_layers, dprnn_erb_f, erb_inter_hidden)))
        for i in range(dprnn_num_blocks):
            states.append((f"dprnn_df_h{i}",
                          [df_inter_layers, dprnn_df_f, df_inter_hidden],
                          torch.zeros(df_inter_layers, dprnn_df_f, df_inter_hidden)))

    # 7. Conv temporal buffers
    erb_conv0_tpad = _get_conv_tpad(model.enc.erb_conv0)
    df_conv0_tpad = _get_conv_tpad(model.enc.df_conv0)
    df_convp_tpad = _get_conv_tpad(model.df_dec.df_convp)

    states.append(("erb_conv0_buf", [1, 1, erb_conv0_tpad, nb_erb_input],
                   torch.zeros(1, 1, erb_conv0_tpad, nb_erb_input)))

    states.append(("df_conv0_buf", [1, 2, df_conv0_tpad, nb_df],
                   torch.zeros(1, 2, df_conv0_tpad, nb_df)))

    states.append(("df_convp_buf", [1, conv_ch, df_convp_tpad, nb_df],
                   torch.zeros(1, conv_ch, df_convp_tpad, nb_df)))

    # df_spec_buf: past df_order-1 masked frames
    df_order = model.df_order
    states.append(("df_spec_buf", [1, 1, df_order - 1, freq_bins, 2],
                   torch.zeros(1, 1, df_order - 1, freq_bins, 2)))

    return states


def save_initial_states(states, output_dir: Path):
    """Save initial states as binary + JSON manifest."""
    manifest = []
    all_data = []

    offset = 0
    for name, shape, tensor in states:
        data = tensor.detach().cpu().numpy().astype(np.float32)
        flat = data.flatten()
        all_data.append(flat)
        manifest.append({
            "name": name,
            "shape": shape,
            "offset": offset,
            "num_elements": len(flat),
        })
        offset += len(flat)

    # Save binary
    bin_path = output_dir / "initial_states.bin"
    all_flat = np.concatenate(all_data)
    all_flat.tofile(str(bin_path))

    # Save manifest
    manifest_path = output_dir / "initial_states.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))

    print(f"  Saved {len(states)} initial states to {bin_path} ({bin_path.stat().st_size} bytes)")
    print(f"  Manifest: {manifest_path}")


# ---------------------------------------------------------------------------
# Export
# ---------------------------------------------------------------------------

def export_stateful_model(checkpoint_path: str, model_name: str, output_dir: str):
    if model_name not in MODELS:
        print(f"Unknown model: {model_name}. Choose from: {list(MODELS.keys())}")
        sys.exit(1)

    cfg = MODELS[model_name]
    sr = cfg["sr"]
    win_len = cfg["win_len"]
    dprnn_num_blocks = cfg["dprnn_num_blocks"]
    freq_bins = win_len // 2 + 1
    hop_size = win_len // 2

    print(f"\nExporting STATEFUL {model_name} (sr={sr}, win={win_len}, F={freq_bins}, dprnn={dprnn_num_blocks})")

    # Load model
    is_48k = model_name == "dpdfnet2_48khz_hr"
    if is_48k:
        from dpdfnet_48khz_hr import DPDFNet48HR
        model = DPDFNet48HR()
    else:
        from dpdfnet import DPDFNet
        model = DPDFNet(n_fft=win_len, samplerate=sr, dprnn_num_blocks=dprnn_num_blocks)

    sd = torch.load(checkpoint_path, weights_only=True, map_location="cpu")
    model.load_state_dict(sd)
    model.eval()
    print(f"  Model loaded from {checkpoint_path}")

    # Source code already uses real-valued ops (no complex types, no einops)

    # Create stateful wrapper
    if is_48k:
        stateful_model = StatefulDPDFNet_48k(model)
    else:
        stateful_model = StatefulDPDFNet_16k(model)
    stateful_model.eval()

    # Compute initial states (introspects model for GRU dimensions)
    states = compute_initial_states(model_name, model, is_48k)

    # Build dummy inputs
    dummy_spec = torch.randn(1, 1, 1, freq_bins, 2)
    dummy_inputs = [dummy_spec] + [s[2].clone() for s in states]

    # Test forward pass
    print(f"  Testing stateful forward pass...")
    with torch.no_grad():
        outputs = stateful_model(*dummy_inputs)
    spec_e = outputs[0]
    lsnr = outputs[1]
    print(f"  Forward pass OK: spec_e={list(spec_e.shape)}, lsnr={list(lsnr.shape)}")
    print(f"  Total outputs: {len(outputs)} (spec_e + lsnr + {len(states)} states)")

    # Build input/output names
    input_names = ["spec"] + [s[0] for s in states]
    output_names = ["spec_enhanced", "lsnr"] + [s[0] + "_out" for s in states]

    # Create output directory
    out_dir = Path(output_dir) / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = out_dir / "model_stateful.onnx"
    print(f"  Exporting to {onnx_path}...")

    torch.onnx.export(
        stateful_model,
        tuple(dummy_inputs),
        str(onnx_path),
        opset_version=17,
        input_names=input_names,
        output_names=output_names,
    )

    print(f"  [OK] Exported to {onnx_path} ({onnx_path.stat().st_size / 1e6:.1f} MB)")

    # Save initial states
    save_initial_states(states, out_dir)

    # Update config.ini
    config_path = out_dir / "config.ini"
    state_names = ",".join(s[0] for s in states)
    config_text = (
        f"[model]\n"
        f"sample_rate = {sr}\n"
        f"win_len = {win_len}\n"
        f"hop_size = {hop_size}\n"
        f"freq_bins = {freq_bins}\n"
        f"export_mode = pytorch_core\n"
        f"interface = spec_real\n"
        f"\n"
        f"[stateful]\n"
        f"model_file = model_stateful.onnx\n"
        f"num_states = {len(states)}\n"
        f"state_names = {state_names}\n"
    )
    config_path.write_text(config_text)
    print(f"  Written {config_path}")


def main():
    parser = argparse.ArgumentParser(description="Export DPDFNet to stateful single-frame ONNX")
    parser.add_argument("--checkpoint", help="Path to .pth file")
    parser.add_argument("--model_name", choices=sorted(MODELS.keys()), help="Single model to export")
    parser.add_argument("--output_dir", default=str(MODELS_DIR), help="Output directory")
    args = parser.parse_args()

    if args.model_name:
        ckpt = args.checkpoint or str(CHECKPOINT_DIR / f"{args.model_name}.pth")
        export_stateful_model(ckpt, args.model_name, args.output_dir)
    else:
        print("DPDFNet Stateful ONNX Export (all models)")
        print("=" * 50)
        success = 0
        for name in MODELS:
            ckpt = args.checkpoint or str(CHECKPOINT_DIR / f"{name}.pth")
            if not Path(ckpt).exists():
                print(f"\n[SKIP] {name}: checkpoint not found at {ckpt}")
                continue
            try:
                export_stateful_model(ckpt, name, args.output_dir)
                success += 1
            except Exception as e:
                print(f"\n[FAIL] {name}: {e}")
                import traceback
                traceback.print_exc()
        print(f"\n{'=' * 50}")
        print(f"Exported {success}/{len(MODELS)} stateful models")


if __name__ == "__main__":
    main()
