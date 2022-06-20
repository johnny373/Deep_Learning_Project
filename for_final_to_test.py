import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
devicess = [0]


import time
import argparse
import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch import nn
from torchvision import transforms
import torch.distributed as dist
import math
import torchio
from torchio.transforms import (
    ZNormalization,
    CropOrPad,
    Compose,
)
from tqdm import tqdm
from torchvision import utils
from for_final_hparams import hparams as hp
from utils.metric import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingLR
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")



source_train_dir = hp.source_train_dir
print("source_train_dir:",source_train_dir)
label_train_dir = hp.label_train_dir
print("label_train_dir:",label_train_dir)

source_test_dir = hp.source_test_dir
label_test_dir = hp.label_test_dir

output_dir_test_float = hp.output_dir_test_float
output_dir_test_int = hp.output_dir_test_int


def parse_training_args(parser):
    """
    Parse commandline arguments.
    """

    parser.add_argument('-o', '--output_dir', type=str, default=hp.output_dir, required=False, help='Directory to save checkpoints')
    parser.add_argument('--latest-checkpoint-file', type=str, default=hp.latest_checkpoint_file, help='Store the latest checkpoint in each epoch')

    # training
    training = parser.add_argument_group('training setup')
    training.add_argument('--epochs', type=int, default=hp.total_epochs, help='Number of total epochs to run')
    training.add_argument('--epochs-per-checkpoint', type=int, default=hp.epochs_per_checkpoint, help='Number of epochs per checkpoint')
    training.add_argument('--batch', type=int, default=hp.batch_size, help='batch-size')  
    parser.add_argument(
        '-k',
        "--ckpt",
        type=str,
        default=hp.ckpt,
        help="path to the checkpoints to resume training",
    )
    parser.add_argument("--init-lr", type=float, default=hp.init_lr, help="learning rate")
    # TODO
    parser.add_argument(
        "--local_rank", type=int, default=0, help="local rank for distributed training"
    )

    training.add_argument('--amp-run', action='store_true', help='Enable AMP')
    training.add_argument('--cudnn-enabled', default=True, help='Enable cudnn')
    training.add_argument('--cudnn-benchmark', default=True, help='Run cudnn benchmark')
    training.add_argument('--disable-uniform-initialize-bn-weight', action='store_true', help='disable uniform initialization of batchnorm layer weight')

    return parser


def test():

    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Testing')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark

    from for_final_use_data import MedData_test

    os.makedirs(output_dir_test_int, exist_ok=True)
    os.makedirs(output_dir_test_float, exist_ok=True)
    if hp.network == 'Unet':
        from models.two_d.unet import Unet
        model = Unet(in_channels=hp.in_class, classes=hp.out_class)
    elif hp.network == 'MiniSeg':
        from models.two_d.miniseg import MiniSeg
        model = MiniSeg(in_input=hp.in_class, classes=hp.out_class)
    elif hp.network == 'fcn':
        from models.two_d.fcn import FCN32s as fcn
        model = fcn(in_class =hp.in_class,n_class=hp.out_class)
    elif hp.network == 'SegNet':
        from models.two_d.segnet import SegNet
        model = SegNet(input_nbr=hp.in_class,label_nbr=hp.out_class)
    elif hp.network == 'DeepLabV3':
        from models.two_d.deeplab import DeepLabV3
        model = DeepLabV3(in_class=hp.in_class,class_num=hp.out_class)
    elif hp.network == 'unetpp':
        from models.two_d.unetpp import ResNet34UnetPlus
        model = ResNet34UnetPlus(num_channels=hp.in_class,num_class=hp.out_class)
    elif hp.network == 'PSPNet':
        from models.two_d.pspnet import PSPNet
        model = PSPNet(in_class=hp.in_class,n_classes=hp.out_class)

    model = torch.nn.DataParallel(model, device_ids=devicess)


    print("load model:", args.ckpt)
    print("test load model => ",os.path.join(args.output_dir, args.latest_checkpoint_file))

    ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file))


    model.load_state_dict(ckpt["model"])


    model.cuda()



    test_dataset = MedData_test(source_test_dir,label_test_dir)
    znorm = ZNormalization()


    if hp.mode == '3d':
        patch_overlap = hp.patch_overlap
        patch_size = hp.patch_size
    elif hp.mode == '2d':
        patch_overlap = hp.patch_overlap
        patch_size = hp.patch_size
    
    for i,subj in enumerate(test_dataset.subjects):
        subj = znorm(subj)
        grid_sampler = torchio.inference.GridSampler(
                subj,
                patch_size,
                patch_overlap,
            )

        patch_loader = torch.utils.data.DataLoader(grid_sampler, batch_size=args.batch)
        aggregator = torchio.inference.GridAggregator(grid_sampler)
        aggregator_1 = torchio.inference.GridAggregator(grid_sampler)
        model.eval()
        with torch.no_grad():
            for patches_batch in tqdm(patch_loader):


                input_tensor = patches_batch['source'][torchio.DATA].to(device)
                locations = patches_batch[torchio.LOCATION]

                if hp.mode == '2d':
                    input_tensor = input_tensor.squeeze(4)
                outputs = model(input_tensor)

                if hp.mode == '2d':
                    outputs = outputs.unsqueeze(4)
                logits = torch.sigmoid(outputs)

                labels = logits.clone()
                labels[labels>0.5] = 1
                labels[labels<=0.5] = 0

                aggregator.add_batch(logits, locations)
                aggregator_1.add_batch(labels, locations)
        output_tensor = aggregator.get_output_tensor()
        output_tensor_1 = aggregator_1.get_output_tensor()




        affine = subj['source']['affine']
        if (hp.in_class == 1) and (hp.out_class == 1) :
            label_image = torchio.ScalarImage(tensor=output_tensor.numpy(), affine=affine)
            label_image.save(os.path.join(output_dir_test_float,f"{i:04d}-result_float"+hp.save_arch))

            # f"{str(i):04d}-result_float.mhd"

            output_image = torchio.ScalarImage(tensor=output_tensor_1.numpy(), affine=affine)
            output_image.save(os.path.join(output_dir_test_int,subj['img_name']+"_bg"+hp.save_arch))
        else:
            output_tensor = output_tensor.unsqueeze(1)
            output_tensor_1= output_tensor_1.unsqueeze(1)
            

            output_image_lung_float = torchio.ScalarImage(tensor=output_tensor[0].numpy(), affine=affine)
            output_image_lung_float.save(os.path.join(output_dir_test_float,subj['img_name']+"_o_crop"+hp.save_arch))
            output_image_lung_int = torchio.ScalarImage(tensor=output_tensor_1[0].numpy(), affine=affine)
            output_image_lung_int.save(os.path.join(output_dir_test_int,subj['img_name']+"_o_crop"+hp.save_arch))

            output_image_trachea_float = torchio.ScalarImage(tensor=output_tensor[1].numpy(), affine=affine)
            output_image_trachea_float.save(os.path.join(output_dir_test_float,subj['img_name']+"_bean"+hp.save_arch))
            output_image_trachea_int = torchio.ScalarImage(tensor=output_tensor_1[1].numpy(), affine=affine)
            output_image_trachea_int.save(os.path.join(output_dir_test_int,subj['img_name']+"_bean"+hp.save_arch))

            output_image_vein_float = torchio.ScalarImage(tensor=output_tensor[2].numpy(), affine=affine)
            output_image_vein_float.save(os.path.join(output_dir_test_float,subj['img_name']+"_maize"+hp.save_arch))
            output_image_vein_int = torchio.ScalarImage(tensor=output_tensor_1[2].numpy(), affine=affine)
            output_image_vein_int.save(os.path.join(output_dir_test_int,subj['img_name']+"_maize"+hp.save_arch))
            
            output_image_vein_float = torchio.ScalarImage(tensor=output_tensor[3].numpy(), affine=affine)
            output_image_vein_float.save(os.path.join(output_dir_test_float,subj['img_name']+"_weed"+hp.save_arch))
            output_image_vein_int = torchio.ScalarImage(tensor=output_tensor_1[3].numpy(), affine=affine)
            output_image_vein_int.save(os.path.join(output_dir_test_int,subj['img_name']+"_weed"+hp.save_arch))

if __name__ == '__main__':
  test()
