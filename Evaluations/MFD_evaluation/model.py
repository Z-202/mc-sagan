import importlib.util
import os

import torch
from torch import nn


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
LOCAL_RESNET_PATH = os.path.join(CURRENT_DIR, 'resnet.py')


def _load_local_resnet_module():
    if not os.path.exists(LOCAL_RESNET_PATH):
        return None

    spec = importlib.util.spec_from_file_location('local_resnet_module', LOCAL_RESNET_PATH)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


resnet = _load_local_resnet_module()
if resnet is None:
    try:
        from models import resnet  # type: ignore
    except ImportError:
        import resnet  # type: ignore


_MODEL_FACTORY = {
    10: resnet.resnet10,
    18: resnet.resnet18,
    34: resnet.resnet34,
    50: resnet.resnet50,
    101: resnet.resnet101,
    152: resnet.resnet152,
    200: resnet.resnet200,
}


def generate_model(opt):
    assert opt.model in ['resnet']
    assert opt.model_depth in _MODEL_FACTORY

    build_model = _MODEL_FACTORY[opt.model_depth]
    model = build_model(
        sample_input_W=opt.input_W,
        sample_input_H=opt.input_H,
        sample_input_D=opt.input_D,
        shortcut_type=opt.resnet_shortcut,
        no_cuda=opt.no_cuda,
        num_seg_classes=opt.n_seg_classes,
    )

    if not opt.no_cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = str(opt.gpu_id)
        model = model.cuda()
        model = nn.DataParallel(model)
        net_dict = model.state_dict()
    else:
        net_dict = model.state_dict()

    if opt.phase != 'test' and opt.pretrain_path:
        print('loading pretrained model {}'.format(opt.pretrain_path))
        pretrain = torch.load(opt.pretrain_path)
        pretrain_dict = {k: v for k, v in pretrain['state_dict'].items() if k in net_dict.keys()}

        net_dict.update(pretrain_dict)
        model.load_state_dict(net_dict)

        new_parameters = []
        for pname, p in model.named_parameters():
            for layer_name in opt.new_layer_names:
                if pname.find(layer_name) >= 0:
                    new_parameters.append(p)
                    break

        new_parameters_id = list(map(id, new_parameters))
        base_parameters = list(filter(lambda p: id(p) not in new_parameters_id, model.parameters()))
        parameters = {
            'base_parameters': base_parameters,
            'new_parameters': new_parameters,
        }

        return model, parameters

    return model, model.parameters()
