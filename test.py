import argparse
import sys, os

import torch
import torch.backends.cudnn
import torch.cuda
import torch.nn
import torch.utils.data
from torchpack import distributed as dist
from torchpack.callbacks import Callbacks, SaverRestore
from torchpack.environ import auto_set_run_dir, set_run_dir
from torchpack.utils.config import configs
from torchpack.utils.logging import logger
from tqdm import tqdm

from core import builder
from core.callbacks import MeanIoU
from core.trainers import SemanticKITTITrainer
from model_zoo import minkunet_test, spvcnn_test

import pdb
import numpy as np
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import cv2

def main() -> None:
    dist.init()

    torch.backends.cudnn.benchmark = True
    torch.cuda.set_device(dist.local_rank())

    parser = argparse.ArgumentParser()
    # parser.add_argument('config', metavar='FILE', help='config file')
    parser.add_argument('--run-dir', metavar='DIR', help='run directory')
    parser.add_argument('--name', type=str, help='model name')
    parser.add_argument('--gpu', default='0', help='gpu index')
    args, opts = parser.parse_known_args()

    args.config = args.name + '/metainfo/configs.yaml'
    configs.load(args.config, recursive=True)
    configs.update(opts)

    if args.run_dir is None:
        args.run_dir = auto_set_run_dir()
    else:
        set_run_dir(args.run_dir)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    logger.info(' '.join([sys.executable] + sys.argv))
    logger.info(f'Experiment started: "{args.run_dir}".' + '\n' + f'{configs}')

    dataset = builder.make_dataset()
    dataflow = {}
    #for split in dataset:
    #    print(split, " U Y " * 10)

    for split in dataset:
        if split == "train":
            continue
        sampler = torch.utils.data.distributed.DistributedSampler(
            dataset[split],
            num_replicas=dist.size(),
            rank=dist.rank(),
            shuffle=(split == 'train'))
        dataflow[split] = torch.utils.data.DataLoader(
            dataset[split],
            batch_size=1,
            sampler=sampler,
            num_workers=configs.workers_per_gpu,
            pin_memory=True,
            collate_fn=dataset[split].collate_fn)

    assert os.path.exists(args.name + '/checkpoints/max-iou-test.pt')
    if 'spvcnn' in args.name.lower():
        model = spvcnn_test(weight_path=args.name+'/checkpoints/max-iou-test.pt', configs=configs)
    elif 'mink' in args.name.lower():
        # model = minkunet(args.name)
        model = minkunet_test(weight_path=args.name + '/checkpoints/max-iou-test.pt', configs=configs)
        #print(args.name, " * " * 10)
    else:
        raise NotImplementedError

    model = torch.nn.parallel.DistributedDataParallel(
        model.cuda(),
        device_ids=[dist.local_rank()],
        find_unused_parameters=True)
    model.eval()

    criterion = builder.make_criterion()
    optimizer = builder.make_optimizer(model)
    scheduler = builder.make_scheduler(optimizer)

    trainer = SemanticKITTITrainer(model=model,
                                   criterion=criterion,
                                   optimizer=optimizer,
                                   scheduler=scheduler,
                                   num_workers=configs.workers_per_gpu,
                                   seed=configs.train.seed)

    callbacks = Callbacks([
        SaverRestore(),
        MeanIoU(configs.data.num_classes, configs.data.ignore_label)
    ])
    callbacks._set_trainer(trainer)
    trainer.callbacks = callbacks
    trainer.dataflow = dataflow['test']

    trainer.before_train()
    trainer.before_epoch()

    model.eval()

    tt = []
    pre = []
    for feed_dict in tqdm(dataflow['test'], desc='eval'):
        _inputs = {}
        for key, value in feed_dict.items():
            #print(key, " H " * 10)
            if 'name' not in key:
                _inputs[key] = value.cuda()

        #print(feed_dict['file_name'], " J K " * 10)
        inputs = _inputs['lidar']
        # targets = feed_dict['targets'].F.long().cuda(non_blocking=True)
        outputs = model(inputs)
        #print(outputs.shape, " GF " * 10)
        invs = feed_dict['inverse_map']
        all_labels = feed_dict['targets_mapped']
        _outputs = []
        _targets = []
        #print(invs.C[:, -1].max() + 1, " > M " * 10)
        for idx in range(invs.C[:, -1].max() + 1):
            cur_scene_pts = (inputs.C[:, -1] == idx).cpu().numpy()
            cur_inv = invs.F[invs.C[:, -1] == idx].cpu().numpy()
            cur_label = (all_labels.C[:, -1] == idx).cpu().numpy()
            outputs_mapped = outputs[cur_scene_pts][cur_inv].argmax(1)
            targets_mapped = all_labels.F[cur_label]
            _outputs.append(outputs_mapped)
            _targets.append(targets_mapped)
        outputs = torch.cat(_outputs, 0)
        targets = torch.cat(_targets, 0)
        output_dict = {'outputs': outputs, 'targets': targets}
        trainer.after_step(output_dict)
        #print(outputs, targets.shape, " ^% " * 10)
        _,dir2 = feed_dict['file_name'][0].split('/sequences/',1)
        new_save_dir = "/dev/pre_our2" + '/sequences/' +dir2.replace('velodyne','predictions')[:-3]+'label'
        if not os.path.exists(os.path.dirname(new_save_dir)):
            try:
                os.makedirs(os.path.dirname(new_save_dir))
            except OSError as exc:
                if exc.errno != errno.EEXIST:
                    raise
        #print(outputs.cpu().numpy().dtype, torch.unique(outputs), " * ^ " * 10)
        outputs.cpu().numpy().astype(np.uint32).tofile(new_save_dir)
        #print(targets.cpu().numpy().shape, " * " * 10)
        tt.append(targets.cpu().numpy())
        pre.append(outputs.cpu().numpy())
        
    
    trainer.after_epoch()
    tt = np.concatenate(tt, axis=0)
    pre = np.concatenate(pre, axis=0)
    tt = tt.tolist()
    pre = pre.tolist()
    tt=  [str(k) for k in tt]
    pre = [str(k) for k in pre]
    C = confusion_matrix(tt, pre, labels=['0','1','2','3','4','5','6','7','8','9','10','11','12','13','14','15','16','17','18'])
    #f = plt.gcf()  #获取当前图像
    #f.savefig('cccc.png')
    print(C)
    #print(tt.tolist(), np.unique(tt), np.unique(pre), C, C.shape, "> " *10)
    cv2.imwrite("ccccc.png", C)

if __name__ == '__main__':
    main()
