import argparse
import os
import time
from glob import glob
import matplotlib.pyplot as plt
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn
import yaml
from albumentations.augmentations import transforms
from albumentations.core.composition import Compose
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import random
import albumentations as albu

import ccnet
import u2net
from dataset import Dataset
from metrics import iou_score
from utils import AverageMeter

""" 
需要指定参数：--name dsb2018_96_NestedUNet_woDS
"""


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('--name', default="Puddle-1000 Dataset_CCU2NET_0515ccnet_foggy3.2",
                        help='model name')

    args = parser.parse_args()

    return args


def set_rand_seed(seed=42):
    print("Random Seed: ", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def main():
    set_rand_seed(42)
    args = parse_args()

    with open('models/%s/config.yml' % args.name, 'r') as f:
        config = yaml.load(f, Loader=yaml.FullLoader)

    print('-' * 20)
    for key in config.keys():
        print('%s: %s' % (key, str(config[key])))
    print('-' * 20)

    cudnn.benchmark = True

    # create model
    print("=> creating model %s" % config['arch'])
    model =u2net.__dict__[config['arch']](config['num_classes'], config['in_ch'])

    model = model.cuda()

    # Data loading code
    img_ids_val = glob(os.path.join('inputs', config['dataset_val'], 'images', '*' + config['img_ext']))
    img_ids_val = [os.path.splitext(os.path.basename(p))[0] for p in img_ids_val]
    val_img_ids = img_ids_val

    #_, val_img_ids = train_test_split(img_ids, test_size=1, random_state=41)

    model.load_state_dict(torch.load('models/%s/model.pth' %
                                     config['name']))
    model.eval()

    val_transform = Compose([
        # transforms.Resize(config['input_h'], config['input_w']),
        albu.Resize(config['input_h'], config['input_w']),
        transforms.Normalize(),
    ])

    val_dataset = Dataset(
        img_ids=val_img_ids,
        img_dir=os.path.join('inputs', config['dataset_val'], 'images'),
        mask_dir=os.path.join('inputs', config['dataset_val'], 'masks'),
        img_ext=config['img_ext'],
        mask_ext=config['mask_ext'],
        num_classes=config['num_classes'],
        transform=val_transform)
    val_loader = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=1,
        shuffle=False,
        num_workers=config['num_workers'],
        drop_last=False)

    avg_meter = AverageMeter()

    for c in range(config['num_classes']):
        os.makedirs(os.path.join('outputs', config['name'], str(c)), exist_ok=True)
    result = []
    total_time = 0
    with torch.no_grad():
        for input, target, meta in tqdm(val_loader, total=len(val_loader)):
            input = input.cuda()
            target = target.cuda()

            # compute output
            #if config['deep_supervision']:
            t1 = time.time()
            output = model(input)
            total_time += time.time() - t1
            output = output[0]
            #else:
                #output = model(input)

            iou = iou_score(output, target)
            result.append(iou)
            avg_meter.update(iou, input.size(0))

            output = torch.sigmoid(output).cpu().numpy()

            for i in range(len(output)):
                for c in range(config['num_classes']):
                    cv2.imwrite(os.path.join('outputs', config['name'], str(c), meta['img_id'][i] + '.png'),
                                (output*255).astype('uint8'))
    # r_1_2 = (result[0] + result[1]) / 2
    # r_3_4 = (result[2] + result[3]) / 2
    # r = (result[2] + result[3] + result[0] + result[1]) / 4
    print('IoU: %.4f' % avg_meter.avg)
    print(total_time / 198)

    plot_examples(input, target, model, num_examples=3)

    torch.cuda.empty_cache()


def plot_examples(datax, datay, model, num_examples=3):
    fig, ax = plt.subplots(nrows=3, ncols=3, figsize=(18, 4 * num_examples))
    m = datax.shape[0]
    for row_num in range(num_examples):
        image_indx = np.random.randint(m)
        image_arr = model(datax[image_indx:image_indx + 1]).squeeze(0).detach().cpu().numpy()
        ax[row_num][0].imshow(np.transpose(datax[image_indx].cpu().numpy(), (1, 2, 0))[:, :, 0])
        ax[row_num][0].set_title("Orignal Image")
        ax[row_num][1].imshow(np.squeeze((image_arr > 0.40)[0, :, :].astype(int)))
        ax[row_num][1].set_title("Segmented Image localization")
        ax[row_num][2].imshow(np.transpose(datay[image_indx].cpu().numpy(), (1, 2, 0))[:, :, 0])
        ax[row_num][2].set_title("Target image")
    plt.show()


if __name__ == '__main__':
    main()
