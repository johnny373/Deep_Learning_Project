class hparams:



    output_dir = "/home/ma/Pytorch-Medical-Segmentation/final_1110422_Unet"

    aug = None

    latest_checkpoint_file = 'checkpoint_latest.pt'

    total_epochs = 100

    epochs_per_checkpoint = 10

    batch_size = 20

    ckpt = None

    init_lr = 0.001

    scheduer_step_size = 20

    scheduer_gamma = 0.8

    debug = False

    mode = '2d'

    # 可以選Unet、MiniSeg、fcn、SegNet、DeepLabV3、unetpp、PSPNet

    network = 'Unet'

    in_class = 3 #這邊是表示通道數量 灰階:1, RGB:3

    out_class = 4



    crop_or_pad_size = 1024,768,1 # if 2D: 256,256,1

    patch_size =400,400,1 # if 2D: 128,128,1 



    # for test

    patch_overlap = 4,4,0 # if 2D: 4,4,0



    fold_arch = '*.jpg'

    label_arch = '*.png'



    save_arch = '.png'



    #source_train_dir = "data/Train/new_img"

    #"/home/ma/Pytorch-Medical-Segmentation/data/Train/final_img+unsharp/"

    source_train_dir = "data/Train/final_img+unsharp"

    label_train_dir = 'data/Train/new_label'

    #test_bean or test_maize

    #final_bean_3 or final_maize_3

    source_test_dir = '/home/ma/Pytorch-Medical-Segmentation/data/Test/final_bean'

    label_test_dir = '/home/ma/Pytorch-Medical-Segmentation/data/Test/label'



    output_dir_test_float = output_dir+'/final/float/b/'

    output_dir_test_int = output_dir+'/final/int/b/'