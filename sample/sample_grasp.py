"""Generate FlowHOI Stage 1 from text and BPS, or replay a Stage 1 reference."""

import argparse
from dataclasses import fields
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import torch
import trimesh

from sample.condition_hands import ClassifierFreeGuidance
from sample.sampling_utils import sample_flow_matching
from utils.fixseed import fixseed
from utils.mesh_to_bps import encode_object_bps, load_object_basis
from utils.model_util import create_model_and_diffusion, load_saved_model
from utils.parser_util import GenerateArgs


HAND_CHANNELS = 108
GRASP_FRAMES = 50
GRAB_DATA = Path('dataset/GRAB_HANDS')
DEFAULT_CHECKPOINT = Path('save/egodex_grasp/model000500000.pt')


def save_npz(path, **arrays):
    """Write a complete archive before replacing the output file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(dir=path.parent) as temp:
        staging = Path(temp) / 'result.npz'
        np.savez_compressed(staging, **arrays)
        os.replace(staging, path)


def checkpoint_args(checkpoint, config):
    saved = json.loads(checkpoint.with_name('args.json').read_text())
    saved.update(config)
    args = GenerateArgs(**{field.name: saved[field.name] for field in fields(GenerateArgs)
                           if field.name in saved})
    args.model_path = str(checkpoint)
    args.dataset = 'grab'
    args.mean_name = 'Mean_grab_grasp.npy'
    args.std_name = 'Std_grab_grasp.npy'
    if (args.dit_version != 'base' or not args.traj_only or not args.hands_only
            or not args.use_bps or args.bps_dim != 3072 or args.unconstrained
            or args.use_initial_pos or args.wrist_only or args.drop_redundant):
        raise ValueError('Expected a checkpoint for text-and-BPS Stage 1 with 108 hand channels')
    return args


def read_reference(path):
    with np.load(path, allow_pickle=False) as source:
        config = json.loads(source['args_json'].item())
        if any(config.get(key) for key in ('guidance', 'inpainting', 'use_initial_pos')):
            raise ValueError('Replay requires a Stage 1 reference captured without pose guidance or inpainting')
        keys = ('text', 'object_bps', 'noise', 'mean', 'std', 'normalized_motion', 'motion', 'scale')
        data = {key: source[key] for key in keys}
        for key in ('t5_emb', 't5_mask', 'reference_motion', 'data_id', 'lengths', 'mask'):
            if key in source:
                data[key] = source[key]
    return config, data


def custom_bps(cli):
    stored = np.load(cli.bps, allow_pickle=False) if cli.bps else None
    if stored is not None and (stored.shape != (1, 1024, 3) or not np.isfinite(stored).all()):
        raise ValueError('--bps must contain one finite array shaped (1, 1024, 3)')
    if cli.mesh is None:
        return stored.astype(np.float32)
    mesh = trimesh.load(cli.mesh, force='mesh', process=False)
    if not isinstance(mesh, trimesh.Trimesh) or not len(mesh.faces):
        raise ValueError('--mesh must be a prepared triangle mesh in metres')
    basis = load_object_basis(cli.basis)
    encoded = encode_object_bps(mesh.vertices, basis)
    if stored is not None and not np.allclose(stored, encoded, atol=2e-8, rtol=0):
        raise ValueError('--bps does not match the supplied mesh and basis')
    if cli.mirror_x:
        return encode_object_bps(mesh.vertices * [-1, 1, 1], basis)
    return encoded


def generate(cli):
    config, reference = read_reference(cli.reference) if cli.reference else ({}, None)
    checkpoint = cli.model_path or Path(config.get('model_path') or DEFAULT_CHECKPOINT)
    protected = (checkpoint, checkpoint.with_name('args.json'), cli.basis, cli.reference, cli.bps, cli.mesh)
    if any(path is not None and path.resolve() == cli.output.resolve() for path in protected):
        raise ValueError('--output must be separate from the input files')
    args = checkpoint_args(checkpoint, config)
    args.seed = config.get('seed', 42) if cli.seed is None else cli.seed
    if cli.sampling_steps is not None:
        args.num_sampling_steps = cli.sampling_steps
    if cli.device is None:
        if reference is not None and config.get('device') == -1:
            device = torch.device('cpu')
        else:
            device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device(cli.device)
    fixseed(args.seed)

    shape = (HAND_CHANNELS, 1, GRASP_FRAMES)
    if reference is not None:
        text = reference['text'].tolist()
        count = len(text)
        object_bps = reference['object_bps']
        noise = reference['noise']
        mean, std = reference['mean'], reference['std']
        scale = float(reference['scale'])
        seeds = np.full(count, args.seed)
    else:
        count = cli.num_samples or 10
        object_bps = np.repeat(custom_bps(cli)[None], count, axis=0)
        text = [cli.text] * count
        seeds = args.seed + np.arange(count)
        noise = torch.stack([torch.randn(shape, generator=torch.Generator().manual_seed(int(seed)))
                             for seed in seeds]).numpy()
        mean = np.load(GRAB_DATA / args.mean_name, allow_pickle=False)[:HAND_CHANNELS]
        std = np.load(GRAB_DATA / args.std_name, allow_pickle=False)[:HAND_CHANNELS]
        std = std * args.std_scale_shift[0] + args.std_scale_shift[1]
        scale = 1.0 if cli.guidance_scale is None else cli.guidance_scale

    if object_bps.shape != (count, 1, 1024, 3) or noise.shape != (count, *shape):
        raise ValueError('BPS or initial noise shape does not match the Stage 1 batch')
    if (mean.shape != (HAND_CHANNELS,) or std.shape != mean.shape or np.any(std <= 0)
            or not all(np.isfinite(value).all() for value in (object_bps, noise, mean, std))):
        raise ValueError('Nonfinite inputs or invalid GRAB normalization statistics')

    model, diffusion = create_model_and_diffusion(args, None)
    load_saved_model(model, checkpoint)
    model.to(device).eval()
    model = ClassifierFreeGuidance(model)
    if reference is not None and 't5_emb' in reference:
        text_emb = torch.from_numpy(reference['t5_emb'])
        text_mask = torch.from_numpy(reference['t5_mask']) if 't5_mask' in reference else None
    else:
        unique_text = list(dict.fromkeys(text))
        index = [unique_text.index(prompt) for prompt in text]
        text_emb, text_mask = model.model.encode_text_with_mask(unique_text)
        text_emb, text_mask = text_emb[index].cpu(), text_mask[index].cpu()

    lengths = reference.get('lengths', np.full(count, GRASP_FRAMES)) if reference is not None else np.full(count, GRASP_FRAMES)
    mask = reference.get('mask') if reference is not None else None
    if mask is None:
        mask = np.arange(GRASP_FRAMES)[None, None, None] < lengths[:, None, None, None]
    chunks = []
    batch_size = cli.batch_size or (count if reference is not None else 16)
    for start in range(0, count, batch_size):
        batch = slice(start, start + batch_size)
        y = {
            'text': text[batch],
            'object_bps': torch.from_numpy(object_bps[batch]),
            't5_emb': text_emb[batch],
            'lengths': torch.from_numpy(lengths[batch]),
            'mask': torch.from_numpy(mask[batch]),
            'scale': scale,
            'grasp_model': True,
        }
        if text_mask is not None:
            y['t5_mask'] = text_mask[batch]
        initial = torch.from_numpy(noise[batch])
        sample = sample_flow_matching(model, diffusion, tuple(initial.shape), {'y': y}, noise=initial)[-1]
        chunks.append(sample[:, :, 0].transpose(1, 2).cpu())
    normalized = torch.cat(chunks)
    motion = normalized * torch.as_tensor(std, dtype=normalized.dtype) + torch.as_tensor(mean, dtype=normalized.dtype)
    if not torch.isfinite(motion).all():
        raise RuntimeError('Stage 1 generated nonfinite motion')

    args.guidance = args.inpainting = False
    args.guidance_param = scale
    args.device = -1 if device.type == 'cpu' else device.index or 0
    output = dict(motion=motion.numpy(), normalized_motion=normalized.numpy(), text=text, seeds=seeds,
                  object_bps=object_bps, noise=noise, mean=mean, std=std, lengths=lengths, mask=mask,
                  t5_emb=text_emb.numpy(), scale=scale, args_json=json.dumps(vars(args), sort_keys=True),
                  input_mirrored=cli.mirror_x, mesh_path=str(cli.mesh.resolve()) if cli.mesh else '',
                  basis_path=str(cli.basis.resolve()))
    if text_mask is not None:
        output['t5_mask'] = text_mask.numpy()
    if reference is not None:
        for key in ('motion', 'normalized_motion'):
            if not np.allclose(output[key], reference[key], rtol=1e-6, atol=2e-6):
                error = np.max(np.abs(output[key] - reference[key]))
                raise ValueError(f'Reference replay differs for {key}. Maximum absolute error: {error:.9g}')
        error = np.max(np.abs(output['motion'] - reference['motion']))
        output['reference_max_error'] = error
        for key in ('reference_motion', 'data_id'):
            if key in reference:
                output[key] = reference[key]
        print(f'Reference replay maximum absolute error: {error:.9g}')
    save_npz(cli.output, **output)
    print(f'Saved {motion.shape[0]} Stage 1 trajectories to {cli.output}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument('--text', help='Grasp prompt')
    mode.add_argument('--reference', type=Path, help='Replay an unguided archive from --save_stage1')
    parser.add_argument('--bps', type=Path, help='object_bps.npy from utils.mesh_to_bps')
    parser.add_argument('--mesh', type=Path, help='Prepared metric object_mesh.ply')
    parser.add_argument('--basis', type=Path, default=GRAB_DATA / 'bps.pt')
    parser.add_argument('--mirror-x', action='store_true', help='Condition on the X-reflected mesh for left-to-right fallback')
    parser.add_argument('--model-path', type=Path)
    parser.add_argument('--num-samples', type=int)
    parser.add_argument('--batch-size', type=int)
    parser.add_argument('--sampling-steps', type=int)
    parser.add_argument('--guidance-scale', type=float)
    parser.add_argument('--seed', type=int)
    parser.add_argument('--device', help='cpu, cuda:0, etc. Default uses the reference CPU setting or available CUDA')
    parser.add_argument('--output', type=Path, required=True)
    cli = parser.parse_args()
    if cli.output.suffix != '.npz':
        parser.error('--output must end in .npz')
    if cli.reference and (cli.bps or cli.mesh or cli.mirror_x or cli.seed is not None
                          or cli.num_samples is not None or cli.sampling_steps is not None
                          or cli.guidance_scale is not None):
        parser.error('--reference reuses its conditions, seed, sample count and sampling settings')
    if cli.text is not None and (not cli.text.strip() or (cli.bps is None and cli.mesh is None)):
        parser.error('custom sampling requires a prompt and --bps or --mesh')
    if cli.mirror_x and cli.mesh is None:
        parser.error('--mirror-x requires the original prepared --mesh')
    if any(value is not None and value < 1 for value in (cli.num_samples, cli.batch_size, cli.sampling_steps)):
        parser.error('sample count, batch size and sampling steps must be positive')
    if cli.seed is not None and not 0 <= cli.seed < 2**32:
        parser.error('--seed must be between 0 and 2**32 - 1')
    if cli.guidance_scale is not None and (not np.isfinite(cli.guidance_scale) or cli.guidance_scale < 1):
        parser.error('--guidance-scale must be finite and at least 1')
    try:
        generate(cli)
    except (KeyError, ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == '__main__':
    main()
