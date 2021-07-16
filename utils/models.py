from model import UNet, HRNet, SETR, HRNet, UNet3Plus, SegNet, HarDMSEG, DeepLab, PSPNet, DANet
from model import EANet, OCRNet, ResUNet, OCNet, AttentionUNet, DenseUNet, DinkNet50, UNet16
from model import SCSEUnet, R2UNet, R2AttentionUNet, CSNet, NestedUNet, NestedUNet_Big
from model import MultiResUnet, U2NET, U2NETP, OneNet, LightNet, CENet, SETR, HarDMSEG_alter
from model import LRFEANet, SimpleUNet
import jittor as jt


modelSet = [
    'unet', 'hrnet', 'setr', 'unet3p', 'segnet', 'hardnet', 'deeplab',
    'pspnet', 'danet', 'eanet', 'ocrnet', 'resunet', 'ocnet', 'attunet',
    'dense', 'dlink', 'ternaus', 'scseunet', 'r2', 'r2att', 'csnet', 'unetpp',
    'unetppbig', 'multires', 'u2net', 'u2netp', 'onenet', 'lightnet', 'cenet',
    'setr', 'hardalter', 'lrfea', 'simple'
]


def get_model(args):
    if args.model == 'unet':
        model = UNet(n_channels = 3, n_classes = args.class_num, bilinear = True)
    elif args.model == 'hrnet':
        model = HRNet(in_ch=3, out_ch = args.class_num)
    elif args.model == 'setr':
        model = SETR(
            patch_size=(32, 32), 
            in_channels=3, 
            out_channels=args.class_num, 
            hidden_size=1024, 
            num_hidden_layers=8, 
            num_attention_heads=16, 
            decode_features=[512, 256, 128, 64]
        )
    elif args.model == 'unet3p':
        model = UNet3Plus(in_ch=3, n_classes=2)
    elif args.model == 'segnet':
        model = SegNet(3, 2)
    elif args.model == 'hardnet':
        model = HarDMSEG()
    elif args.model == 'deeplab':
        model = DeepLab()
    elif args.model == 'pspnet':
        model = PSPNet()
    elif args.model == 'danet':
        model = DANet()
    elif args.model == 'eanet':
        model = EANet(num_classes=2)
    elif args.model == 'ocrnet':
        model = OCRNet()
    elif args.model == 'resunet':
        model = ResUNet()
    elif args.model == 'ocnet':
        model = OCNet()
    elif args.model == 'attunet':
        model = AttentionUNet()
    elif args.model == 'dlink':
        model = DinkNet50(num_classes=2)
    elif args.model == 'dense':
        model = DenseUNet()
    elif args.model == 'ternaus':
        model = UNet16(num_classes=2)
    elif args.model == 'scseunet':
        model = SCSEUnet()
    elif args.model == 'r2':
        model = R2UNet()
    elif args.model == 'r2att':
        model = R2AttentionUNet()
    elif args.model == 'csnet':
        model = CSNet(2, 3)
    elif args.model == 'unetpp':
        model = NestedUNet()
    elif args.model == 'unetppbig':
        model = NestedUNet_Big()
    elif args.model == 'multires':
        model = MultiResUnet()
    elif args.model == 'u2net':
        model = U2NET()
    elif args.model == 'u2netp':
        model = U2NETP()
    elif args.model == 'onenet':
        model = OneNet()
    elif args.model == 'lightnet':
        model = LightNet()
    elif args.model == 'cenet':
        model = CENet()
    elif args.model == 'setr':
        model = SETR()
    elif args.model == 'hardalter':
        model = HarDMSEG_alter()
    elif args.model == 'lrfea':
        model = LRFEANet()
    elif args.model == 'simple':
        model = SimpleUNet()
    else:
        print("Error: model undefined")
        exit(0)

    # self-supervised pretrained_weights
    if args.pretrain:
        model.load_parameters(jt.load(args.checkpoint))

    return model
