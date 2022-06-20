import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0, 1'
devicess = [0,1]

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
)
from tqdm import tqdm
from torchvision import utils
from for_final_hparams import hparams as hp
from utils.metric import metric
from torch.optim.lr_scheduler import ReduceLROnPlateau,StepLR,CosineAnnealingLR
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")



source_train_dir = hp.source_train_dir
label_train_dir = hp.label_train_dir


source_test_dir = hp.source_test_dir
label_test_dir = hp.label_test_dir

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



def train():

    parser = argparse.ArgumentParser(description='PyTorch Medical Segmentation Training')
    parser = parse_training_args(parser)
    args, _ = parser.parse_known_args()

    args = parser.parse_args()


    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.enabled = args.cudnn_enabled
    torch.backends.cudnn.benchmark = args.cudnn_benchmark


    from for_final_use_data import MedData_train
    os.makedirs(args.output_dir, exist_ok=True)

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
    optimizer = torch.optim.Adam(model.parameters(), lr=args.init_lr)
    #optimizer = nn.DataParallel(optimizer, device_ids=devicess) #新增的

    # scheduler = ReduceLROnPlateau(optimizer, 'min',factor=0.5, patience=20, verbose=True)
    scheduler = StepLR(optimizer, step_size=hp.scheduer_step_size, gamma=hp.scheduer_gamma)
    # scheduler = CosineAnnealingLR(optimizer, T_max=50, eta_min=5e-6)

    if args.ckpt is not None:
        print("load model:", args.ckpt)
        print(os.path.join(args.output_dir, args.latest_checkpoint_file))
        ckpt = torch.load(os.path.join(args.output_dir, args.latest_checkpoint_file), map_location=lambda storage, loc: storage)

        model.load_state_dict(ckpt["model"])
        optimizer.load_state_dict(ckpt["optim"])

        for state in optimizer.state.values():
            for k, v in state.items():
                if torch.is_tensor(v):
                    state[k] = v.cuda()

        # scheduler.load_state_dict(ckpt["scheduler"])
        elapsed_epochs = ckpt["epoch"]
    else:
        elapsed_epochs = 0

    model.cuda()

# 我的loss的部分，他這邊的dice感覺很怪效果很差
    from loss_function import Binary_Loss,DiceLoss
    from dice_loss import GDiceLoss,GDiceLossV2
    from Focal_loss import FocalLoss_B,FocalLoss_M
    # ~ criterion = Binary_Loss().cuda()
    criterion = GDiceLossV2().cuda()
    #criterion = FocalLoss_B().cuda()
    # ~ criterion = FocalLoss_M().cuda()


    writer = SummaryWriter(args.output_dir)



    train_dataset = MedData_train(source_train_dir,label_train_dir)
    train_loader = DataLoader(train_dataset.queue_dataset, 
                            batch_size=args.batch, 
                            shuffle=True,
                            pin_memory=True,
                            drop_last=True)

    model.train()

    epochs = args.epochs - elapsed_epochs
    iteration = elapsed_epochs * len(train_loader)



    for epoch in range(1, epochs + 1):
        print("epoch:"+str(epoch))
        epoch += elapsed_epochs

        num_iters = 0


        for i, batch in enumerate(train_loader):
            

            if hp.debug:
                if i >=1:
                    break

            print(f"Batch: {i}/{len(train_loader)} epoch {epoch}")

            optimizer.zero_grad()


            if (hp.in_class == 1) and (hp.out_class == 1) :
                x = batch['source']['data']
                y = batch['label']['data']

                #y[y!=0] = 1 
            
                x = x.type(torch.FloatTensor).cuda()
                y = y.type(torch.FloatTensor).cuda()
                
            else:
                x = batch['source']['data']
                print(batch['source']['data'].shape)
                y_atery = batch['atery']['data']
                y_lung = batch['lung']['data']
                y_trachea = batch['trachea']['data']
                y_vein = batch['vein']['data']

                x = x.type(torch.FloatTensor).cuda()
                y = torch.cat((y_atery,y_lung,y_trachea,y_vein),1) 
                #y = torch.cat((y_lung,y_trachea,y_vein),1) 
                y = y.type(torch.FloatTensor).cuda()


            if hp.mode == '2d':
                x = x.squeeze(4)
                y = y.squeeze(4)

                y[y!=0] = 1
                
                #print(y.max())
            print(x.shape)
            outputs = model(x) #卡在這邊 -> 調整模型的conv size


            # for metrics
            logits = torch.sigmoid(outputs)
            labels = logits.clone()
            labels[labels>0.5] = 1
            labels[labels<=0.5] = 0


            loss = criterion(outputs, y)

            num_iters += 1
            loss.backward()

            optimizer.step()
            iteration += 1


            false_positive_rate,false_negtive_rate,dice = metric(y.cpu(),labels.cpu())
            ## log
            writer.add_scalar('Training/Loss', loss.item(),iteration)
            writer.add_scalar('Training/false_positive_rate', false_positive_rate,iteration)
            writer.add_scalar('Training/false_negtive_rate', false_negtive_rate,iteration)
            writer.add_scalar('Training/dice', dice,iteration)
            


            print("loss:"+str(loss.item()))
            print('lr:'+str(scheduler._last_lr[0]))

            

        scheduler.step()


        # Store latest checkpoint in each epoch
        torch.save(
            {
                "model": model.state_dict(),
                "optim": optimizer.state_dict(),
                "scheduler":scheduler.state_dict(),
                "epoch": epoch,

            },
            os.path.join(args.output_dir, args.latest_checkpoint_file),
        )




        # Save checkpoint
        if epoch % args.epochs_per_checkpoint == 0:

            torch.save(
                {
                    
                    "model": model.state_dict(),
                    "optim": optimizer.state_dict(),
                    "epoch": epoch,
                },
                os.path.join(args.output_dir, f"checkpoint_{epoch:04d}.pt"),
            )
        


            
            with torch.no_grad():
                if hp.mode == '2d':
                    x = x.unsqueeze(4)
                    y = y.unsqueeze(4)
                    outputs = outputs.unsqueeze(4)
                    
                x = x[0].cpu().detach().numpy()
                y = y[0].cpu().detach().numpy()
                outputs = outputs[0].cpu().detach().numpy()
                affine = batch['source']['affine'][0].numpy()




                if (hp.in_class == 1) and (hp.out_class == 1) :
                    source_image = torchio.ScalarImage(tensor=x, affine=affine)
                    source_image.save(os.path.join(args.output_dir,f"step-{epoch:04d}-source"+hp.save_arch))
                    # source_image.save(os.path.join(args.output_dir,("step-{}-source.mhd").format(epoch)))

                    label_image = torchio.ScalarImage(tensor=y, affine=affine)
                    label_image.save(os.path.join(args.output_dir,f"step-{epoch:04d}-gt"+hp.save_arch))

                    output_image = torchio.ScalarImage(tensor=outputs, affine=affine)
                    output_image.save(os.path.join(args.output_dir,f"step-{epoch:04d}-predict"+hp.save_arch))
                else:
                    y = np.expand_dims(y, axis=1)
                    outputs = np.expand_dims(outputs, axis=1)


    writer.close()
if __name__ == '__main__':
        train()
