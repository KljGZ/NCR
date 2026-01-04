# -*- coding = utf-8 -*-
import cv2
import torch
import numpy as np

def add_trigger(args, image, test=False,trigger=None):
    if trigger is None:
        pixel_max = max(1,torch.max(image))
        if args.trigger == 'square':
            pixel_max = torch.max(image) if torch.max(image) > 1 else 1

            if args.dataset in ['cifar', 'gtsrb']:
                pixel_max = 1
            image[:, args.triggerY:args.triggerY + 5, args.triggerX:args.triggerX + 5] = pixel_max
        elif args.trigger == 'pattern':
            pixel_max = torch.max(image) if torch.max(image) > 1 else 1
            image[:, args.triggerY + 0, args.triggerX + 0] = pixel_max
            image[:, args.triggerY + 1, args.triggerX + 1] = pixel_max
            image[:, args.triggerY - 1, args.triggerX + 1] = pixel_max
            image[:, args.triggerY + 1, args.triggerX - 1] = pixel_max
        elif args.trigger == 'watermark':
            if args.watermark is None:
                args.watermark = cv2.imread('./utils/watermark.png', cv2.IMREAD_GRAYSCALE)
                args.watermark = cv2.bitwise_not(args.watermark)
                args.watermark = cv2.resize(args.watermark, dsize=image[0].shape, interpolation=cv2.INTER_CUBIC)
                pixel_max = np.max(args.watermark)
                args.watermark = args.watermark.astype(np.float64) / pixel_max
                # cifar [0,1] else max>1
                pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
                args.watermark *= pixel_max_dataset
            max_pixel = max(np.max(args.watermark), torch.max(image))
            image += args.watermark
            image[image > max_pixel] = max_pixel
        elif args.trigger == 'apple':
            if args.apple is None:
                args.apple = cv2.imread('./utils/apple.png', cv2.IMREAD_GRAYSCALE)
                args.apple = cv2.bitwise_not(args.apple)
                args.apple = cv2.resize(args.apple, dsize=image[0].shape, interpolation=cv2.INTER_CUBIC)
                pixel_max = np.max(args.apple)
                args.apple = args.apple.astype(np.float64) / pixel_max
                # cifar [0,1] else max>1
                pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
                args.apple *= pixel_max_dataset
            max_pixel = max(np.max(args.apple), torch.max(image))
            image += args.apple
            image[image > max_pixel] = max_pixel
        elif args.trigger == 'hallokitty':
            if args.hallokitty is None:
                args.hallokitty = cv2.imread('./utils/halloKitty.png')
                pixel_max = np.max(args.hallokitty)
                args.hallokitty = args.hallokitty.astype(np.float64) / pixel_max
                args.hallokitty = torch.from_numpy(args.hallokitty)
                # cifar [0,1] else max>1
                pixel_max_dataset = torch.max(image).item() if torch.max(image).item() > 1 else 1
                args.hallokitty *= pixel_max_dataset
            image = args.hallokitty * 0.5 + image * 0.5
            max_pixel = max(torch.max(args.hallokitty), torch.max(image))
            image[image > max_pixel] = max_pixel
        # save the most recent backdoor image in test dataset
        # args.save_img(image)
    else:
        if image.dim()==4:
            image[0, :, args.triggerY:args.triggerY + 5, args.triggerX:args.triggerX + 5]  = trigger
        if image.dim()==3:
            image[ :, args.triggerY:args.triggerY + 5, args.triggerX:args.triggerX + 5] = trigger
    return image
