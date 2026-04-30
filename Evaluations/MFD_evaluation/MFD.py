"""
python MFD.py   --dataset brats   --data_root_real /scratch/deepai/tmp/DATASET/brats2023/test_new   --data_root_fake /data/deepai/GitHub_final/3D-MC-SAGAN/generated_multi_proc_80   --pretrain_path /data/deepai/GitHub_final/3D-MC-SAGAN/Evaluations/MFD_evaluation/pretrained/resnet_50_23dataset.pth   --path_to_activations /data/deepai/GitHub_final/3D-MC-SAGAN/Evaluations/MFD_evaluation/activations_all_modalities   --modalities t1n,t1c,t2f   --gpu_id 0

"""
import argparse
import importlib.util
import os
import sys

import numpy as np
import torch
from scipy import linalg
from torch.utils.data import DataLoader
from data_loader import BRATSVolumes
from model import generate_model

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PARENT_DIR = os.path.dirname(CURRENT_DIR)

if CURRENT_DIR not in sys.path:
    sys.path.insert(0, CURRENT_DIR)
if PARENT_DIR not in sys.path:
    sys.path.append(PARENT_DIR)


SUPPORTED_DATASETS = ('brats',)
DEFAULT_MODALITIES = ('t1n', 't1c', 't2f')





def resolve_modalities(modalities_argument):
    if isinstance(modalities_argument, (list, tuple)):
        requested = list(modalities_argument)
    else:
        requested = [item.strip() for item in modalities_argument.split(',') if item.strip()]

    if not requested:
        raise ValueError('No modalities were provided.')

    for modality in requested:
        if modality not in DEFAULT_MODALITIES:
            raise ValueError(
                "Unsupported modality '{}'. Choose from {}.".format(
                    modality, ', '.join(DEFAULT_MODALITIES)
                )
            )
    return requested


def get_feature_extractor(sets):
    model, _ = generate_model(sets)
    checkpoint = torch.load(sets.pretrain_path, map_location='cpu' if sets.no_cuda else None)
    state_dict = checkpoint['state_dict'] if isinstance(checkpoint, dict) and 'state_dict' in checkpoint else checkpoint
    state_dict = adapt_checkpoint_keys(model, state_dict)
    model.load_state_dict(state_dict)
    model.eval()
    print('Feature extractor is ready, and the pretrained weights have been loaded successfully')
    return model


def adapt_checkpoint_keys(model, checkpoint_state_dict):
    model_keys = list(model.state_dict().keys())
    checkpoint_keys = list(checkpoint_state_dict.keys())

    if not model_keys or not checkpoint_keys:
        return checkpoint_state_dict

    model_has_module_prefix = model_keys[0].startswith('module.')
    checkpoint_has_module_prefix = checkpoint_keys[0].startswith('module.')

    if model_has_module_prefix == checkpoint_has_module_prefix:
        return checkpoint_state_dict

    if model_has_module_prefix:
        return {'module.' + key: value for key, value in checkpoint_state_dict.items()}

    return {key.replace('module.', '', 1): value for key, value in checkpoint_state_dict.items()}


def unpack_batch(batch):
    if isinstance(batch, (list, tuple)):
        return batch[0]
    return batch


def move_to_device(batch, device):
    if device.type == 'cuda':
        return batch.cuda(non_blocking=False)
    return batch.to(device)


def get_activations(model, data_loader, sets, num_samples):
    pred_arr = np.empty((num_samples, sets.dims))
    write_index = 0
    total_batches = len(data_loader)
    bar_width = 32

    for batch_index, batch in enumerate(data_loader):
        batch = unpack_batch(batch)
        batch = move_to_device(batch, sets.device)

        progress = batch_index + 1
        ratio = progress / total_batches
        filled = int(bar_width * ratio)
        bar = '█' * filled + '░' * (bar_width - filled)
        percent = ratio * 100.0
        print(
            f'\rFeature extraction in progress |{bar}| {percent:6.2f}%  ({progress}/{total_batches})',
            end='',
            flush=True
        )

        with torch.no_grad():
            pred = model(batch)

        pred_np = pred.detach().cpu().numpy()
        next_index = min(write_index + pred_np.shape[0], pred_arr.shape[0])
        pred_arr[write_index:next_index] = pred_np[:next_index - write_index]
        write_index = next_index

        if write_index >= pred_arr.shape[0]:
            break

    print()
    return pred_arr


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, 'Mismatch detected in mean vector lengths between the two sets'
    assert sigma1.shape == sigma2.shape, 'Real and generated covariance matrices have incompatible dimensions'

    diff = mu1 - mu2

    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('MFD computation encountered a near-singular covariance product') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Complex-valued component detected: {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return diff.dot(diff) + np.trace(sigma1) + np.trace(sigma2) - 2 * tr_covmean


def process_feature_vecs(activations):
    mu = np.mean(activations, axis=0)
    sigma = np.cov(activations, rowvar=False)
    return mu, sigma


def save_feature_statistics(output_dir, mu_real, sigma_real, mu_fake, sigma_fake):
    os.makedirs(output_dir, exist_ok=True)

    path_to_mu_real = os.path.join(output_dir, 'mu_real.npy')
    path_to_sigma_real = os.path.join(output_dir, 'sigma_real.npy')
    path_to_mu_fake = os.path.join(output_dir, 'mu_fake.npy')
    path_to_sigma_fake = os.path.join(output_dir, 'sigma_fake.npy')

    np.save(path_to_mu_real, mu_real)



def parse_opts():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', required=True, type=str, help='Dataset (brats)')
    parser.add_argument('--img_size', default=256, type=int, help='Image size')
    parser.add_argument('--data_root_real', required=True, type=str, help='Path to real data')
    parser.add_argument('--data_root_fake', required=True, type=str,
                        help='Path to fake data root. This can be the common root for all modalities or a single modality folder.')
    parser.add_argument('--pretrain_path', required=True, type=str, help='Path to pretrained model')
    parser.add_argument('--path_to_activations', required=True, type=str, help='Path to activations')
    parser.add_argument('--modalities', default=','.join(DEFAULT_MODALITIES), type=str,
                        help='Comma-separated list of modalities to evaluate in one run. Supported: t1n,t1c,t2f')
    parser.add_argument('--num_samples', default=80, type=int,
                        help='Number of samples to use per modality. Default is kept at 80 to preserve the original script behavior.')
    parser.add_argument('--n_seg_classes', default=2, type=int, help='Number of segmentation classes')
    parser.add_argument('--learning_rate', default=0.001, type=float,
                        help='Initial learning rate (divided by 10 while training by lr scheduler)')
    parser.add_argument('--num_workers', default=16, type=int, help='Number of jobs')
    parser.add_argument('--batch_size', default=1, type=int, help='Batch Size')
    parser.add_argument('--phase', default='test', type=str, help='Phase of train or test')
    parser.add_argument('--save_intervals', default=10, type=int, help='Interation for saving model')
    parser.add_argument('--n_epochs', default=200, type=int, help='Number of total epochs to run')
    parser.add_argument('--input_D', default=256, type=int, help='Input size of depth')
    parser.add_argument('--input_H', default=256, type=int, help='Input size of height')
    parser.add_argument('--input_W', default=256, type=int, help='Input size of width')
    parser.add_argument('--resume_path', default='', type=str, help='Path for resume model.')
    parser.add_argument('--new_layer_names', default=['conv_seg'], type=list, help='New layer except for backbone')
    parser.add_argument('--no_cuda', action='store_true', help='If true, cuda is not used.')
    parser.set_defaults(no_cuda=False)
    parser.add_argument('--gpu_id', default=0, type=int, help='Gpu id')
    parser.add_argument('--model', default='resnet', type=str,
                        help='(resnet | preresnet | wideresnet | resnext | densenet | ') 
    parser.add_argument('--model_depth', default=50, type=int, help='Depth of resnet (10 | 18 | 34 | 50 | 101)')
    parser.add_argument('--resnet_shortcut', default='B', type=str, help='Shortcut type of resnet (A | B)')
    parser.add_argument('--manual_seed', default=1, type=int, help='Manually set random seed')
    parser.add_argument('--ci_test', action='store_true', help='If true, ci testing is used.')
    args = parser.parse_args()
    args.save_folder = './trails/models/{}_{}'.format(args.model, args.model_depth)
    return args


def build_dataloader(dataset, sets):
    return DataLoader(
        dataset,
        batch_size=sets.batch_size,
        shuffle=False,
        num_workers=sets.num_workers,
        pin_memory=False,
    )


def build_datasets(sets, modality):
    real_data = BRATSVolumes(
        sets.data_root_real,
        normalize=None,
        mode='real',
        img_size=sets.img_size,
        modality=modality,
    )
    fake_data = BRATSVolumes(
        sets.data_root_fake,
        normalize=None,
        mode='fake',
        img_size=sets.img_size,
        modality=modality,
    )
    return real_data, fake_data


def get_num_samples(sets, dataset):
    if len(dataset) < sets.num_samples:
        raise ValueError(
            'Requested num_samples={} but only {} samples were found.'.format(sets.num_samples, len(dataset))
        )
    return sets.num_samples


def run_single_modality(model, sets, modality):
    print('\n' + '=' * 80)
    print('Running MFD for modality: {}'.format(modality))
    print('=' * 80)

    real_data, fake_data = build_datasets(sets, modality)
    print('fully downloaded real data: ', len(real_data), '  fake data: ', len(fake_data))
    real_data_loader = build_dataloader(real_data, sets)
    fake_data_loader = build_dataloader(fake_data, sets)

    num_real_samples = get_num_samples(sets, real_data)
    num_fake_samples = get_num_samples(sets, fake_data)

    print('Get activations from real data ...')
    activations_real = get_activations(model, real_data_loader, sets, num_real_samples)
    mu_real, sigma_real = process_feature_vecs(activations_real)

    print('\nGet activations from fake/generated data ...')
    activations_fake = get_activations(model, fake_data_loader, sets, num_fake_samples)
    mu_fake, sigma_fake = process_feature_vecs(activations_fake)

    output_dir = os.path.join(sets.path_to_activations, modality)
    save_feature_statistics(output_dir, mu_real, sigma_real, mu_fake, sigma_fake)

    fid = calculate_frechet_distance(mu_real, sigma_real, mu_fake, sigma_fake)
    print('The MFD score for {} is:'.format(modality))
    print(fid)
    return fid


if __name__ == '__main__':
    sets = parse_opts()
    sets.target_type = 'normal'
    sets.phase = 'test'
    sets.batch_size = 1
    sets.dims = 2048
    sets.modalities = resolve_modalities(sets.modalities)

    if sets.dataset not in SUPPORTED_DATASETS:
        raise ValueError(
            "Dataloader for '{}' is not implemented here. This updated script only uses the BraTS loader.".format(
                sets.dataset
            )
        )

    if not sets.no_cuda:
        sets.device = torch.device('cuda')
    else:
        sets.device = torch.device('cpu')




    print('Load model ...')
    model = get_feature_extractor(sets)
    model = model.to(sets.device)

    all_fid_scores = {}
    for modality in sets.modalities:
        all_fid_scores[modality] = run_single_modality(model, sets, modality)

    print('\n' + '=' * 80)
    print('MFD summary')
    print('=' * 80)
    for modality in sets.modalities:
        print('{}: {}'.format(modality, all_fid_scores[modality]))
