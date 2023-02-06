import json
import random
import sys
from pathlib import Path

import torch
import numpy as np
from torch.nn import functional as F

# import cupy as cp
from numba import jit, prange
CYTHON = False

def maximum_path(value, mask, max_neg_val=None):
    """
    Monotonic alignment search algorithm
    Numpy-friendly version. It's about 4 times faster than torch version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    if max_neg_val is None:
        max_neg_val = -np.inf  # Patch for Sphinx complaint
    value = value * mask

    device = value.device
    dtype = value.dtype
    value = value.cpu().detach().numpy()
    mask = mask.cpu().detach().numpy().astype(np.bool)

    b, t_x, t_y = value.shape
    direction = np.zeros(value.shape, dtype=np.int64)
    v = np.zeros((b, t_x), dtype=np.float32)
    x_range = np.arange(t_x, dtype=np.float32).reshape(1, -1)
    for j in range(t_y):
        v0 = np.pad(v, [[0, 0], [1, 0]], mode="constant", constant_values=max_neg_val)[:, :-1]
        v1 = v
        max_mask = v1 >= v0
        v_max = np.where(max_mask, v1, v0)
        direction[:, :, j] = max_mask

        index_mask = x_range <= j
        v = np.where(index_mask, v_max + value[:, :, j], max_neg_val)
    direction = np.where(mask, direction, 1)

    path = np.zeros(value.shape, dtype=np.float32)
    index = mask[:, :, 0].sum(1).astype(np.int64) - 1
    index_range = np.arange(b)
    for j in reversed(range(t_y)):
        path[index_range, index, j] = 1
        index = index + direction[index_range, index, j] - 1
    path = path * mask.astype(np.float32)
    path = torch.from_numpy(path).to(device=device, dtype=dtype)
    return path

@jit(parallel=True)
def maximum_path_numba(value, mask, max_neg_val=None):
    """
    Monotonic alignment search algorithm
    Numpy-friendly version. It's about 4 times faster than torch version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    if max_neg_val is None:
        max_neg_val = -np.inf  # Patch for Sphinx complaint
    value = value * mask

    # device = value.device
    # dtype = value.dtype
    # value = value.cpu().detach().numpy()
    # mask = mask.cpu().detach().numpy().astype(np.bool)

    b, t_x, t_y = value.shape
    direction = np.zeros(value.shape, dtype=np.int64)
    v = np.zeros((b, t_x), dtype=np.float32)
    x_range = np.arange(t_x, dtype=np.float32).reshape(1, -1)
    for j in prange(t_y):
        v0 = np.pad(v, [[0, 0], [1, 0]], mode="constant", constant_values=max_neg_val)[:, :-1]
        v1 = v
        max_mask = v1 >= v0
        v_max = np.where(max_mask, v1, v0)
        direction[:, :, j] = max_mask

        index_mask = x_range <= j
        v = np.where(index_mask, v_max + value[:, :, j], max_neg_val)
    direction = np.where(mask, direction, 1)

    path = np.zeros(value.shape, dtype=np.float32)
    index = mask[:, :, 0].sum(1).astype(np.int64) - 1
    index_range = np.arange(b)
    # for j in reversed(prange(t_y)):
    for j in prange(t_y):
        path[index_range, index, (t_y-1)-j] = 1
        index = index + direction[index_range, index, (t_y-1)-j] - 1
    path = path * mask.astype(np.float32)
    # path = torch.from_numpy(path).to(device=device, dtype=dtype)
    return path


# import pytorch_pfn_extras as ppe
# ppe.cuda.use_torch_mempool_in_cupy()
# print("torch.cuda.memory_allocated()", torch.cuda.memory_allocated())
def maximum_path_cupy(value, mask, max_neg_val=None):
    """
    Monotonic alignment search algorithm
    Numpy-friendly version. It's about 4 times faster than torch version.
    value: [b, t_x, t_y]
    mask: [b, t_x, t_y]
    """
    if max_neg_val is None:
        max_neg_val = -cp.inf  # Patch for Sphinx complaint
    value = value * mask

    device = value.device
    dtype = value.dtype
    # value = value.cpu().detach().numpy()
    # mask = mask.cpu().detach().numpy().astype(cp.bool)
    value = cp.array(value)
    mask = cp.array(mask).astype(cp.bool)

    b, t_x, t_y = value.shape
    direction = cp.zeros(value.shape, dtype=cp.int64)
    v = cp.zeros((b, t_x), dtype=cp.float32)
    x_range = cp.arange(t_x, dtype=cp.float32).reshape(1, -1)
    for j in range(t_y):
        v0 = cp.pad(v, [[0, 0], [1, 0]], mode="constant", constant_values=max_neg_val)[:, :-1]
        v1 = v
        max_mask = v1 >= v0
        v_max = cp.where(max_mask, v1, v0)
        direction[:, :, j] = max_mask

        index_mask = x_range <= j
        v = cp.where(index_mask, v_max + value[:, :, j], max_neg_val)
    direction = cp.where(mask, direction, 1)

    path = cp.zeros(value.shape, dtype=cp.float32)
    index = mask[:, :, 0].sum(1).astype(cp.int64) - 1
    index_range = cp.arange(b)
    for j in reversed(range(t_y)):
        path[index_range, index, j] = 1
        index = index + direction[index_range, index, j] - 1
    path = path * mask.astype(cp.float32)
    path = torch.as_tensor(path, device=device)
    return path

def rand_segments(x: torch.tensor, x_lengths: torch.tensor = None, segment_size=4):
    """Create random segments based on the input lengths.

    Args:
        x (torch.tensor): Input tensor.
        x_lengths (torch.tensor): Input lengths.
        segment_size (int): Expected output segment size.

    Shapes:
        - x: :math:`[B, C, T]`
        - x_lengths: :math:`[B]`
    """
    B, _, T = x.size()
    if x_lengths is None:
        x_lengths = T
    max_idxs = x_lengths - segment_size + 1
    assert all(max_idxs > 0), " [!] At least one sample is shorter than the segment size."
    segment_indices = (torch.rand([B]).type_as(x) * max_idxs).long()
    ret = segment(x, segment_indices, segment_size)
    return ret, segment_indices
def segment(x: torch.tensor, segment_indices: torch.tensor, segment_size=4):
    """Segment each sample in a batch based on the provided segment indices

    Args:
        x (torch.tensor): Input tensor.
        segment_indices (torch.tensor): Segment indices.
        segment_size (int): Expected output segment size.
    """
    segments = torch.zeros_like(x[:, :, :segment_size])
    for i in range(x.size(0)):
        index_start = segment_indices[i]
        index_end = index_start + segment_size
        segments[i] = x[i, :, index_start:index_end]
    return segments
# from https://gist.github.com/jihunchoi/f1434a77df9db1bb337417854b398df1
def sequence_mask(sequence_length, max_len=None):
    """Create a sequence mask for filtering padding in a sequence tensor.

    Args:
        sequence_length (torch.tensor): Sequence lengths.
        max_len (int, Optional): Maximum sequence length. Defaults to None.

    Shapes:
        - mask: :math:`[B, T_max]`
    """
    if max_len is None:
        max_len = sequence_length.data.max()
    seq_range = torch.arange(max_len, dtype=sequence_length.dtype, device=sequence_length.device)
    # B x T_max
    mask = seq_range.unsqueeze(0) < sequence_length.unsqueeze(1)
    return mask







DEFAULT_MIN_BIN_WIDTH = 1e-3
DEFAULT_MIN_BIN_HEIGHT = 1e-3
DEFAULT_MIN_DERIVATIVE = 1e-3
def piecewise_rational_quadratic_transform(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails=None,
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):

    if tails is None:
        spline_fn = rational_quadratic_spline
        spline_kwargs = {}
    else:
        spline_fn = unconstrained_rational_quadratic_spline
        spline_kwargs = {"tails": tails, "tail_bound": tail_bound}

    outputs, logabsdet = spline_fn(
        inputs=inputs,
        unnormalized_widths=unnormalized_widths,
        unnormalized_heights=unnormalized_heights,
        unnormalized_derivatives=unnormalized_derivatives,
        inverse=inverse,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
        **spline_kwargs,
    )
    return outputs, logabsdet

def searchsorted(bin_locations, inputs, eps=1e-6):
    bin_locations[..., -1] += eps
    return torch.sum(inputs[..., None] >= bin_locations, dim=-1) - 1


def unconstrained_rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    tails="linear",
    tail_bound=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    inside_interval_mask = (inputs >= -tail_bound) & (inputs <= tail_bound)
    outside_interval_mask = ~inside_interval_mask

    outputs = torch.zeros_like(inputs)
    logabsdet = torch.zeros_like(inputs)

    if tails == "linear":
        unnormalized_derivatives = F.pad(unnormalized_derivatives, pad=(1, 1))
        constant = np.log(np.exp(1 - min_derivative) - 1)
        unnormalized_derivatives[..., 0] = constant
        unnormalized_derivatives[..., -1] = constant

        outputs[outside_interval_mask] = inputs[outside_interval_mask]
        logabsdet[outside_interval_mask] = 0
    else:
        raise RuntimeError("{} tails are not implemented.".format(tails))

    outputs[inside_interval_mask], logabsdet[inside_interval_mask] = rational_quadratic_spline(
        inputs=inputs[inside_interval_mask],
        unnormalized_widths=unnormalized_widths[inside_interval_mask, :],
        unnormalized_heights=unnormalized_heights[inside_interval_mask, :],
        unnormalized_derivatives=unnormalized_derivatives[inside_interval_mask, :],
        inverse=inverse,
        left=-tail_bound,
        right=tail_bound,
        bottom=-tail_bound,
        top=tail_bound,
        min_bin_width=min_bin_width,
        min_bin_height=min_bin_height,
        min_derivative=min_derivative,
    )

    return outputs, logabsdet


def rational_quadratic_spline(
    inputs,
    unnormalized_widths,
    unnormalized_heights,
    unnormalized_derivatives,
    inverse=False,
    left=0.0,
    right=1.0,
    bottom=0.0,
    top=1.0,
    min_bin_width=DEFAULT_MIN_BIN_WIDTH,
    min_bin_height=DEFAULT_MIN_BIN_HEIGHT,
    min_derivative=DEFAULT_MIN_DERIVATIVE,
):
    if torch.min(inputs) < left or torch.max(inputs) > right:
        raise ValueError("Input to a transform is not within its domain")

    num_bins = unnormalized_widths.shape[-1]

    if min_bin_width * num_bins > 1.0:
        raise ValueError("Minimal bin width too large for the number of bins")
    if min_bin_height * num_bins > 1.0:
        raise ValueError("Minimal bin height too large for the number of bins")

    widths = F.softmax(unnormalized_widths, dim=-1)
    widths = min_bin_width + (1 - min_bin_width * num_bins) * widths
    cumwidths = torch.cumsum(widths, dim=-1)
    cumwidths = F.pad(cumwidths, pad=(1, 0), mode="constant", value=0.0)
    cumwidths = (right - left) * cumwidths + left
    cumwidths[..., 0] = left
    cumwidths[..., -1] = right
    widths = cumwidths[..., 1:] - cumwidths[..., :-1]

    derivatives = min_derivative + F.softplus(unnormalized_derivatives)

    heights = F.softmax(unnormalized_heights, dim=-1)
    heights = min_bin_height + (1 - min_bin_height * num_bins) * heights
    cumheights = torch.cumsum(heights, dim=-1)
    cumheights = F.pad(cumheights, pad=(1, 0), mode="constant", value=0.0)
    cumheights = (top - bottom) * cumheights + bottom
    cumheights[..., 0] = bottom
    cumheights[..., -1] = top
    heights = cumheights[..., 1:] - cumheights[..., :-1]

    if inverse:
        bin_idx = searchsorted(cumheights, inputs)[..., None]
    else:
        bin_idx = searchsorted(cumwidths, inputs)[..., None]

    input_cumwidths = cumwidths.gather(-1, bin_idx)[..., 0]
    input_bin_widths = widths.gather(-1, bin_idx)[..., 0]

    input_cumheights = cumheights.gather(-1, bin_idx)[..., 0]
    delta = heights / widths
    input_delta = delta.gather(-1, bin_idx)[..., 0]

    input_derivatives = derivatives.gather(-1, bin_idx)[..., 0]
    input_derivatives_plus_one = derivatives[..., 1:].gather(-1, bin_idx)[..., 0]

    input_heights = heights.gather(-1, bin_idx)[..., 0]

    if inverse:
        a = (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        ) + input_heights * (input_delta - input_derivatives)
        b = input_heights * input_derivatives - (inputs - input_cumheights) * (
            input_derivatives + input_derivatives_plus_one - 2 * input_delta
        )
        c = -input_delta * (inputs - input_cumheights)

        discriminant = b.pow(2) - 4 * a * c
        assert (discriminant >= 0).all()

        root = (2 * c) / (-b - torch.sqrt(discriminant))
        outputs = root * input_bin_widths + input_cumwidths

        theta_one_minus_theta = root * (1 - root)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        )
        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * root.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - root).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, -logabsdet
    else:
        theta = (inputs - input_cumwidths) / input_bin_widths
        theta_one_minus_theta = theta * (1 - theta)

        numerator = input_heights * (input_delta * theta.pow(2) + input_derivatives * theta_one_minus_theta)
        denominator = input_delta + (
            (input_derivatives + input_derivatives_plus_one - 2 * input_delta) * theta_one_minus_theta
        )
        outputs = input_cumheights + numerator / denominator

        derivative_numerator = input_delta.pow(2) * (
            input_derivatives_plus_one * theta.pow(2)
            + 2 * input_delta * theta_one_minus_theta
            + input_derivatives * (1 - theta).pow(2)
        )
        logabsdet = torch.log(derivative_numerator) - 2 * torch.log(denominator)

        return outputs, logabsdet



from typing import Dict, List, Tuple

from torch.utils.data.sampler import WeightedRandomSampler
def get_language_weighted_sampler(items: list):
    language_names = np.array([item[3] for item in items])
    unique_language_names = np.unique(language_names).tolist()
    language_ids = [unique_language_names.index(l) for l in language_names]
    language_count = np.array([len(np.where(language_names == l)[0]) for l in unique_language_names])
    weight_language = 1.0 / language_count
    dataset_samples_weight = torch.from_numpy(np.array([weight_language[l] for l in language_ids])).double()
    return WeightedRandomSampler(dataset_samples_weight, len(dataset_samples_weight))



import os
import re
from glob import glob
# def vctk(root_path, meta_files=None, wavs_path="wav48", ignored_speakers=None):

#     items = []

#     with open(f'{root_path}/metadata.csv') as f:
#         lines = f.read().split("\n")
#         for line in lines:
#             fname = line.split("|")[0]
#             text = line.split("|")[1]

#             speaker_id = fname.split("_")[0]
#             # if isinstance(ignored_speakers, list):
#             #     if speaker_id in ignored_speakers:
#             #         continue
#             # wav_file = os.path.join(root_path, "wavs", speaker_id, fname)
#             wav_file = os.path.join(root_path, "wavs", fname)
#             items.append([text, wav_file, "VCTK_" + speaker_id])
#             # items.append([text, wav_file, "VCTK_" + speaker_id, "en"])
#             # items.append([text, wav_file, "VCTK_" + speaker_id])
#     return items
# def xvaspeech(root_path, meta_files=None):

#     num_speakers = 0

#     lang = root_path.split("/")[-1]
#     root_path = "/".join(root_path.split("/")[:-1])

#     csv_files = glob(root_path + f'/{lang}_**/metadata.csv', recursive=True)

#     # print(f'csv_files, {csv_files}')


#     items = []
#     for csv_file in csv_files:

#         # ======== DEBUG
#         # if "it_f4_danse" not in csv_file and "it_f4_nate" not in csv_file and "it_sk_malenordcommander" not in csv_file and "it_sk_femalenord" not in csv_file and "it_sk_femalecommander" not in csv_file:
#         # if "it_f4_danse" not in csv_file and "it_f4_nate" not in csv_file and "it_sk_malenordcommander":
#         # if "it_f4_nate" not in csv_file and "it_sk_malenordcommander":
#         # if "de_f4_nate" not in csv_file:
#         #     pass
#         # else:
#             # continue
#         # if "it_" in csv_file and "it_f4_nate" not in csv_file or "en_" in csv_file:
#         #     continue
#         # ========

#         csv_file = csv_file.replace("\\", "/")
#         if os.path.isfile(csv_file):
#             txt_file = csv_file
#         else:
#             txt_file = os.path.join(root_path, csv_file)

#         folder = os.path.dirname(txt_file)

#         # speaker_name_match = (txt_file.split("/female/")[1] if "/female/" in txt_file else txt_file.split("/male/")[1]).split("/")[0]
#         # if speaker_name_match is None:
#         #     continue
#         # speaker_name = speaker_name_match.group("speaker_name")
#         speaker_name = root_path.split("/")[-1]
#         # ignore speakers
#         # if isinstance(ignored_speakers, list):
#         #     if speaker_name in ignored_speakers:
#         #         continue
#         print(" | > {}".format(csv_file))
#         has_registered_at_least_one = False
#         with open(txt_file, "r", encoding="utf-8") as ttf:
#             for line in ttf:
#                 cols = line.split("|")
#                 wav_file = os.path.join(folder, "wavs", (cols[0] + ".wav") if ".wav" not in cols[0] else cols[0])
#                 # if not meta_files:
#                 #     # wav_file = os.path.join(folder, "wavs", cols[0] + ".wav")
#                 #     wav_file = os.path.join(folder, "wavs", (cols[0] + ".wav") if ".wav" not in cols[0] else cols[0])
#                 # else:
#                 #     # wav_file = os.path.join(root_path, folder.replace("metadata.csv", ""), "wavs", cols[0] + ".wav")
#                 #     wav_file = os.path.join(root_path, folder.replace("metadata.csv", ""), "wavs", (cols[0] + ".wav") if ".wav" not in cols[0] else cols[0])
#                 # if os.path.isfile(wav_file):
#                 if os.path.exists(wav_file):
#                     text = cols[1].strip()
#                     items.append([text, wav_file, speaker_name])
#                     has_registered_at_least_one = True
#                 else:
#                     # M-AI-Labs have some missing samples, so just print the warning
#                     # print("> File %s does not exist!" % (wav_file))
#                     pass

#         if has_registered_at_least_one:
#             num_speakers += 1

#     # print(f'mailabs formatter items, {len(items)}')
#     return items, num_speakers
# def mailabs(root_path, meta_files=None, ignored_speakers=None):
#     # print("=====================", "mailabs")

#     """Normalizes M-AI-Labs meta data files to TTS format

#     Args:
#         root_path (str): root folder of the MAILAB language folder.
#         meta_files (str):  list of meta files to be used in the training. If None, finds all the csv files
#             recursively. Defaults to None
#     """
#     speaker_regex = re.compile("by_book/(male|female)/(?P<speaker_name>[^/]+)/")
#     if not meta_files:
#         csv_files = glob(root_path + "/**/metadata.csv", recursive=True)
#     else:
#         csv_files = meta_files



#     # meta_files = [f.strip() for f in meta_files.split(",")]
#     items = []
#     for csv_file in csv_files:
#         csv_file = csv_file.replace("\\", "/")
#         if "/mix/" in csv_file:
#             continue
#         if os.path.isfile(csv_file):
#             txt_file = csv_file
#         else:
#             txt_file = os.path.join(root_path, csv_file)

#         folder = os.path.dirname(txt_file)
#         # print(f'txt_file, {txt_file}')
#         # print(f'folder, {folder}')
#         # print(f'speaker_regex, {speaker_regex}')
#         # determine speaker based on folder structure...
#         # speaker_name_match = speaker_regex.search(txt_file)
#         # print(f'speaker_name_match, {speaker_name_match}')
#         speaker_name_match = (txt_file.split("/female/")[1] if "/female/" in txt_file else txt_file.split("/male/")[1]).split("/")[0]
#         if speaker_name_match is None:
#             continue
#         # speaker_name = speaker_name_match.group("speaker_name")
#         speaker_name = speaker_name_match
#         # ignore speakers
#         if isinstance(ignored_speakers, list):
#             if speaker_name in ignored_speakers:
#                 continue
#         print(" | > {}".format(csv_file))
#         with open(txt_file, "r", encoding="utf-8") as ttf:
#             for line in ttf:
#                 cols = line.split("|")
#                 if not meta_files:
#                     # wav_file = os.path.join(folder, "wavs", cols[0] + ".wav")
#                     wav_file = os.path.join(folder, "wavs", (cols[0] + ".wav") if ".wav" not in cols[0] else cols[0])
#                 else:
#                     # wav_file = os.path.join(root_path, folder.replace("metadata.csv", ""), "wavs", cols[0] + ".wav")
#                     wav_file = os.path.join(root_path, folder.replace("metadata.csv", ""), "wavs", (cols[0] + ".wav") if ".wav" not in cols[0] else cols[0])
#                 if os.path.isfile(wav_file):
#                     text = cols[1].strip()
#                     items.append([text, wav_file, speaker_name])
#                 else:
#                     # M-AI-Labs have some missing samples, so just print the warning
#                     # print("> File %s does not exist!" % (wav_file))
#                     pass
#     # print(f'mailabs formatter items, {len(items)}')
#     return items

from collections import Counter
def split_dataset(items):
    """Split a dataset into train and eval. Consider speaker distribution in multi-speaker training.

    Args:
        items (List[List]): A list of samples. Each sample is a list of `[audio_path, text, speaker_id]`.
    """
    speakers = [item[-1] for item in items]
    is_multi_speaker = len(set(speakers)) > 1
    eval_split_size = min(500, int(len(items) * 0.01))
    # eval_split_size = min(10, int(len(items) * 0.01))
    # assert eval_split_size > 0, " [!] You do not have enough samples to train. You need at least 100 samples."
    np.random.seed(0)
    np.random.shuffle(items)
    if is_multi_speaker:
        items_eval = []
        speakers = [item[-1] for item in items]
        speaker_counter = Counter(speakers)
        while len(items_eval) < eval_split_size:
            item_idx = np.random.randint(0, len(items))
            speaker_to_be_removed = items[item_idx][-1]
            if speaker_counter[speaker_to_be_removed] > 1:
                items_eval.append(items[item_idx])
                speaker_counter[speaker_to_be_removed] -= 1
                del items[item_idx]
        return items_eval, items
    return items[:eval_split_size], items[eval_split_size:]

from math import exp
from torch.autograd import Variable

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-((x - window_size // 2) ** 2) / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()
def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window
def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    # TODO: check if you need AMP disabled
    # with torch.cuda.amp.autocast(enabled=False):
    mu1_sq = mu1.float().pow(2)
    mu2_sq = mu2.float().pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    return ssim_map.mean(1).mean(1).mean(1)
def ssim(img1, img2, window_size=11, size_average=True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel).type_as(img1)
    window = window.type_as(img1)
    return _ssim(img1, img2, window, window_size, channel, size_average)


def make_symbols(
    characters,
    phonemes=None,
    punctuations="!'(),-.:;? ",
    pad="_",
    eos="~",
    bos="^",
    unique=True,
):  # pylint: disable=redefined-outer-name
    """Function to create symbols and phonemes
    TODO: create phonemes_to_id and symbols_to_id dicts here."""
    _symbols = list(characters)
    _symbols = [bos] + _symbols if len(bos) > 0 and bos is not None else _symbols
    _symbols = [eos] + _symbols if len(bos) > 0 and eos is not None else _symbols
    _symbols = [pad] + _symbols if len(bos) > 0 and pad is not None else _symbols
    _phonemes = None
    if phonemes is not None:
        _phonemes_sorted = (
            sorted(list(set(phonemes))) if unique else sorted(list(phonemes))
        )  # this is to keep previous models compatible.
        # Prepend "@" to ARPAbet symbols to ensure uniqueness (some are the same as uppercase letters):
        # _arpabet = ["@" + s for s in _phonemes_sorted]
        # Export all symbols:
        _phonemes = [pad, eos, bos] + list(_phonemes_sorted) + list(punctuations)
        # _symbols += _arpabet
    return _symbols, _phonemes
# Regular expression matching text enclosed in curly braces:
_CURLY_RE = re.compile(r"(.*?)\{(.+?)\}(.*)")
_whitespace_re = re.compile(r"\s+")
def _should_keep_symbol(s):
    return s in _symbol_to_id and s not in ["~", "^", "_"]
def lowercase(text):
    return text.lower()
def replace_symbols(text, lang="en"):
    text = text.replace(";", ",")
    text = text.replace("-", " ")
    text = text.replace(":", ",")
    if lang == "en":
        text = text.replace("&", " and ")
    elif lang == "fr":
        text = text.replace("&", " et ")
    elif lang == "pt":
        text = text.replace("&", " e ")
    return text
def remove_aux_symbols(text):
    text = re.sub(r"[\<\>\(\)\[\]\"]+", "", text)
    return text
def collapse_whitespace(text):
    return re.sub(_whitespace_re, " ", text).strip()
def multilingual_cleaners(text):
    """Pipeline for multilingual text"""
    text = lowercase(text)
    text = replace_symbols(text, lang=None)
    text = remove_aux_symbols(text)
    text = collapse_whitespace(text)
    return text
def _clean_text(text, cleaner_names):
    for name in cleaner_names:
        # cleaner = getattr(cleaners, name)
        cleaner = multilingual_cleaners
        if not cleaner:
            raise Exception("Unknown cleaner: %s" % name)
        text = cleaner(text)
    return text
def _symbols_to_sequence(syms):
    return [_symbol_to_id[s] for s in syms if _should_keep_symbol(s)]
def _arpabet_to_sequence(text):
    return _symbols_to_sequence(["@" + s for s in text.split()])
def intersperse(sequence, token):
    result = [token] * (len(sequence) * 2 + 1)
    result[1::2] = sequence
    return result
def text_to_sequence(
    text: str, cleaner_names: List[str], custom_symbols: List[str] = None, tp: Dict = None, add_blank: bool = False
) -> List[int]:
    """Converts a string of text to a sequence of IDs corresponding to the symbols in the text.
    If `custom_symbols` is provided, it will override the default symbols.

    Args:
      text (str): string to convert to a sequence
      cleaner_names (List[str]): names of the cleaner functions to run the text through
      tp (Dict): dictionary of character parameters to use a custom character set.
      add_blank (bool): option to add a blank token between each token.

    Returns:
      List[int]: List of integers corresponding to the symbols in the text
    """
    # pylint: disable=global-statement
    global _symbol_to_id, _symbols

    if custom_symbols is not None:
        _symbols = custom_symbols
    elif tp:
        _symbols, _ = make_symbols(**tp)
    _symbol_to_id = {s: i for i, s in enumerate(_symbols)}

    sequence = []

    # Check for curly braces and treat their contents as ARPAbet:
    while text:
        m = _CURLY_RE.match(text)
        if not m:
            sequence += _symbols_to_sequence(_clean_text(text, cleaner_names))
            break
        sequence += _symbols_to_sequence(_clean_text(m.group(1), cleaner_names))
        sequence += _arpabet_to_sequence(m.group(2))
        text = m.group(3)

    if add_blank:
        sequence = intersperse(sequence, len(_symbols))  # add a blank token (new), whose id number is len(_symbols)
    return sequence

import librosa.util as librosa_util
from scipy.signal import get_window
def window_sumsquare(window, n_frames, hop_length=200, win_length=800,
                     n_fft=800, dtype=np.float32, norm=None):
    """
    # from librosa 0.6
    Compute the sum-square envelope of a window function at a given hop length.

    This is used to estimate modulation effects induced by windowing
    observations in short-time fourier transforms.

    Parameters
    ----------
    window : string, tuple, number, callable, or list-like
        Window specification, as in `get_window`

    n_frames : int > 0
        The number of analysis frames

    hop_length : int > 0
        The number of samples to advance between frames

    win_length : [optional]
        The length of the window function.  By default, this matches `n_fft`.

    n_fft : int > 0
        The length of each analysis frame.

    dtype : np.dtype
        The data type of the output

    Returns
    -------
    wss : np.ndarray, shape=`(n_fft + hop_length * (n_frames - 1))`
        The sum-squared envelope of the window function
    """
    if win_length is None:
        win_length = n_fft

    n = n_fft + hop_length * (n_frames - 1)
    x = np.zeros(n, dtype=dtype)

    # Compute the squared window at the desired length
    win_sq = get_window(window, win_length, fftbins=True)
    win_sq = librosa_util.normalize(win_sq, norm=norm)**2
    win_sq = librosa_util.pad_center(win_sq, n_fft)

    # Fill the envelope
    for i in range(n_frames):
        sample = i * hop_length
        x[sample:min(n, sample + n_fft)] += win_sq[:max(0, min(n_fft, n - sample))]
    return x

def _pad_data(x, length):
    _pad = 0
    assert x.ndim == 1
    return np.pad(x, (0, length - x.shape[0]), mode="constant", constant_values=_pad)
def _pad_stop_target(x: np.ndarray, length: int, pad_val=1) -> np.ndarray:
    """Pad stop target array.

    Args:
        x (np.ndarray): Stop target array.
        length (int): Length after padding.
        pad_val (int, optional): Padding value. Defaults to 1.

    Returns:
        np.ndarray: Padded stop target array.
    """
    assert x.ndim == 1
    return np.pad(x, (0, length - x.shape[0]), mode="constant", constant_values=pad_val)
def _pad_tensor(x, length):
    _pad = 0.0
    assert x.ndim == 2
    x = np.pad(x, [[0, 0], [0, length - x.shape[1]]], mode="constant", constant_values=_pad)
    return x
def prepare_tensor(inputs, out_steps):
    max_len = max((x.shape[1] for x in inputs))
    remainder = max_len % out_steps
    pad_len = max_len + (out_steps - remainder) if remainder > 0 else max_len
    return np.stack([_pad_tensor(x, pad_len) for x in inputs])
def prepare_data(inputs):
    max_len = max((len(x) for x in inputs))
    return np.stack([_pad_data(x, max_len) for x in inputs])
def prepare_stop_target(inputs, out_steps):
    """Pad row vectors with 1."""
    max_len = max((x.shape[0] for x in inputs))
    remainder = max_len % out_steps
    pad_len = max_len + (out_steps - remainder) if remainder > 0 else max_len
    return np.stack([_pad_stop_target(x, pad_len) for x in inputs])



def convert_pad_shape(pad_shape):
    l = pad_shape[::-1]
    pad_shape = [item for sublist in l for item in sublist]
    return pad_shape
def generate_path(duration, mask):
    """
    duration: [b, t_x]
    mask: [b, t_x, t_y]
    """
    device = duration.device
    b, t_x, t_y = mask.shape
    cum_duration = torch.cumsum(duration, 1)
    path = torch.zeros(b, t_x, t_y, dtype=mask.dtype).to(device=device)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path * mask
    return path


def format_time (seconds):
    time_str = ""
    if seconds>60*60*24:
        days = int(seconds/(60*60*24))
        time_str += f'{days}d '
        seconds -= days*(60*60*24)
    if seconds>60*60:
        hours = int(seconds/(60*60))
        time_str += f'{hours}h '
        seconds -= hours*(60*60)
    if seconds>60:
        minutes = int(seconds/(60))
        time_str += f'{minutes}m '
        seconds -= minutes*(60)
    if seconds>0:
        time_str += f'{int(seconds)}s '

    return time_str