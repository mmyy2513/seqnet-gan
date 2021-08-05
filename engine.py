import math
import sys
from copy import deepcopy
import cv2
import torch
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm
import torchvision.transforms as transforms
import random
import numpy as np
from torch.autograd import Variable

from eval_func import eval_detection, eval_search_cuhk, eval_search_prw
from utils.utils import MetricLogger, SmoothedValue, mkdir, reduce_dict, warmup_lr_scheduler
from trainer import DGNet_Trainer, to_gray

tf = transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])

def recover(inp):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0)) # C,H,W(tensor) to H,W,C(numpy)
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean # un-normalize
    inp = inp * 255.0 # 0-1 to 0-255
    inp = np.clip(inp, 0, 255)
    return inp

def to_device(images, targets, device):
    
    images = [image.to(device) for image in images]
    #images = [transforms.Grayscale(num_output_channels=3)(image) for image in images]
    
    for t in targets:
        t["boxes"] = t["boxes"].to(device)
        t["labels"] = t["labels"].to(device)
    return images, targets


def train_one_epoch(cfg, model,optimizer, data_loader, device, epoch, tfboard=None, model_g = None):
    model.train()
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)

    # warmup learning rate in the first epoch
    if epoch == 0:
        warmup_factor = 1.0 / 1000
        # FIXME: min(1000, len(data_loader) - 1)
        warmup_iters = len(data_loader) - 1
        warmup_scheduler = warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
        

    for i, (images, targets) in enumerate(
        metric_logger.log_every(data_loader, cfg.DISP_PERIOD, header)
    ):
        
        save_name = targets[0]['img_name']
        images, targets = to_device(images, targets, device)

#!!!! GENERATE ############################################################################################################################           
        try:
            if random.random() > 0.5 and model_g is not None:
                imgs = []
                targets_ = []
                
                with torch.no_grad():
                    for image, target in zip(images, targets):
                        
                        #* save original things
                        imgs.append(image)
                        targets_.append(target)
                        
                        #* tensor2numpy
                        im = image.cpu().numpy().transpose(1,2,0)*255
                        
                        #* original image
                        #origin = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                        #cv2.imwrite("res/original/"+save_name,origin)
                        
                        #* get ground-truth coordinates
                        coords = target['boxes']
                        
                        #* vars to save gt info
                        gallery = []
                        size_origin = []
                        coord_origin = []

                        #* get size, coords of each bbox
                        #* bbox resize to generate
                        for n, i in enumerate(coords):
                            x1, y1, x2, y2 = int(i[0].item()), int(i[1].item()), int(i[2].item()), int(i[3].item())
                            size_origin.append((abs(x1-x2), abs(y1-y2)))
                            coord_origin.append((x1,y1,x2,y2))
                            warp = cv2.resize(im[y1:y2, x1:x2], (128,256), interpolation= 3)
                            warp = transforms.ToTensor()(warp.astype('uint8')).unsqueeze(0)
                            warp = tf(warp)
                            gallery.append(warp)
                        
                        #* if only 1 bbox --> pass
                        if len(gallery) > 1:
                            encode = model_g.gen_a.encode # encode function
                            style_encode = model_g.gen_a.encode # encode function
                            id_encode = model_g.id_a # encode function
                            decode = model_g.gen_a.decode # decode function

                            #* images to reconstruct
                            structure = torch.stack(gallery).squeeze(1)
                            
                            #* to gray
                            bg_img = structure
                            gray = to_gray(False)
                            bg_img = gray(bg_img)
                            bg_img = Variable(bg_img.to(device))
                            
                            #* construct vector
                            s = encode(bg_img)
                            
                            #* appearance vector
                            a = []
                            for ref in gallery:
                                id_img = ref
                                id_img = Variable(id_img.to(device))
                                n, c, h, w = id_img.size()
                                feat, _ = id_encode(id_img)
                                a.append(feat)
                                                        
                            #* generate
                            out = []
                            for i in range(s.size(0)):
                                #* pick 1 reference appearance image
                                ind = random.randint(0, len(gallery)-1)
                                f = a[ind]
                                
                                s_tmp = s[i,:,:,:]
                                outputs = decode(s_tmp.unsqueeze(0), f)
                                tmp = recover(outputs[0].data.cpu())
                                out.append(tmp)

                            #* copy to original image
                            for i in range(len(out)):
                                out[i] = cv2.resize(out[i], (size_origin[i][0], size_origin[i][1]))
                                im[coord_origin[i][1]:coord_origin[i][3], coord_origin[i][0]:coord_origin[i][2]] = out[i]
                                
                            #* reconstructed image   
                            tmp3 = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
                            cv2.imwrite("res/after_"+save_name, tmp3)
                        
                        #* save reconstructed image(tensor) - augment
                        imgs.append(transforms.ToTensor()(im.astype('uint8')))
                        targets_.append(target)
                        
                images = imgs
                targets = targets_
                
                images, targets = to_device(images, targets, device)
        except: pass
        
#!!!! ######################################################################################################################################   
        loss_dict = model(images, targets)
        losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())
        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print(f"Loss is {loss_value}, stopping training")
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        if cfg.SOLVER.CLIP_GRADIENTS > 0:
            clip_grad_norm_(model.parameters(), cfg.SOLVER.CLIP_GRADIENTS)
        optimizer.step()

        if epoch == 0:
            warmup_scheduler.step()

        metric_logger.update(loss=loss_value, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])
        """
        if tfboard:
            iter = epoch * len(data_loader) + i
            for k, v in loss_dict_reduced.items():
                tfboard.add_scalars("train", {k: v}, iter)
        """

@torch.no_grad()
def evaluate_performance(
    model, gallery_loader, query_loader, device, use_gt=False, use_cache=False, use_cbgm=False
):
    """
    Args:
        use_gt (bool, optional): Whether to use GT as detection results to verify the upper
                                bound of person search performance. Defaults to False.
        use_cache (bool, optional): Whether to use the cached features. Defaults to False.
        use_cbgm (bool, optional): Whether to use Context Bipartite Graph Matching algorithm.
                                Defaults to False.
    """
    model.eval()
    if use_cache:
        eval_cache = torch.load("data/eval_cache/eval_cache.pth")
        gallery_dets = eval_cache["gallery_dets"]
        gallery_feats = eval_cache["gallery_feats"]
        query_dets = eval_cache["query_dets"]
        query_feats = eval_cache["query_feats"]
        query_box_feats = eval_cache["query_box_feats"]
    else:
        print("\n")
        gallery_dets, gallery_feats = [], []
        for images, targets in tqdm(gallery_loader, ncols=0):
            images, targets = to_device(images, targets, device)
            if not use_gt:
                outputs = model(images)
            else:
                boxes = targets[0]["boxes"]
                n_boxes = boxes.size(0)
                embeddings = model(images, targets)
                outputs = [
                    {
                        "boxes": boxes,
                        "embeddings": torch.cat(embeddings),
                        "labels": torch.ones(n_boxes).to(device),
                        "scores": torch.ones(n_boxes).to(device),
                    }
                ]

            for output in outputs:
                box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
                gallery_dets.append(box_w_scores.cpu().numpy())
                gallery_feats.append(output["embeddings"].cpu().numpy())

        # regarding query image as gallery to detect all people
        # i.e. query person + surrounding people (context information)
        query_dets, query_feats = [], []
        for images, targets in tqdm(query_loader, ncols=0):
            images, targets = to_device(images, targets, device)
            # targets will be modified in the model, so deepcopy it
            outputs = model(images, deepcopy(targets), query_img_as_gallery=True)

            # consistency check
            gt_box = targets[0]["boxes"].squeeze()
            assert (
                gt_box - outputs[0]["boxes"][0]
            ).sum() <= 0.001, "GT box must be the first one in the detected boxes of query image"

            for output in outputs:
                box_w_scores = torch.cat([output["boxes"], output["scores"].unsqueeze(1)], dim=1)
                query_dets.append(box_w_scores.cpu().numpy())
                query_feats.append(output["embeddings"].cpu().numpy())

        # extract the features of query boxes
        query_box_feats = []
        for images, targets in tqdm(query_loader, ncols=0):
            images, targets = to_device(images, targets, device)
            embeddings = model(images, targets)
            assert len(embeddings) == 1, "batch size in test phase should be 1"
            query_box_feats.append(embeddings[0].cpu().numpy())

        mkdir("data/eval_cache")
        save_dict = {
            "gallery_dets": gallery_dets,
            "gallery_feats": gallery_feats,
            "query_dets": query_dets,
            "query_feats": query_feats,
            "query_box_feats": query_box_feats,
        }
        torch.save(save_dict, "data/eval_cache/eval_cache.pth")

    eval_detection(gallery_loader.dataset, gallery_dets, det_thresh=0.01)
    eval_search_func = (
        eval_search_cuhk if gallery_loader.dataset.name == "CUHK-SYSU" else eval_search_prw
    )
    eval_search_func(
        gallery_loader.dataset,
        query_loader.dataset,
        gallery_dets,
        gallery_feats,
        query_box_feats,
        query_dets,
        query_feats,
        cbgm=use_cbgm,
    )
