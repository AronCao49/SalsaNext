#!/usr/bin/env python3
# This file is covered by the LICENSE file in the root of this project.

import torch
import torch.nn as nn
import torch.optim as optim
import torch.backends.cudnn as cudnn
import imp
import yaml
import time
from PIL import Image
import __init__ as booger
import collections
import copy
import cv2
import os
import numpy as np

from tasks.semantic.modules.SalsaNext import *
from tasks.semantic.modules.SalsaNextAdf import *
from tasks.semantic.postproc.KNN import KNN


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

def load_state(net, state_dict, strict=False):
	if strict:
		net.load_state_dict(state_dict=state_dict)
	else:
		# customized partially load function
		net_state_keys = list(net.state_dict().keys())
		for name, param in state_dict.items():
			name_m = name if "module." in name else "module." + name
			if name_m in net.state_dict().keys():
				dst_param_shape = net.state_dict()[name_m].shape
				if param.shape == dst_param_shape:
					net.state_dict()[name_m].copy_(param.view(dst_param_shape))
					net_state_keys.remove(name_m)
		# indicating missed keys
		if net_state_keys:
			print(">> Failed to load: {}".format(net_state_keys))
			return net
	return net

class User():
  def __init__(self, ARCH, DATA, datadir, logdir, modeldir,split,uncertainty,mc=30):
    # parameters
    self.ARCH = ARCH
    self.DATA = DATA
    self.datadir = datadir
    self.logdir = logdir
    self.modeldir = modeldir
    self.uncertainty = uncertainty
    self.split = split
    self.mc = mc

    # get the data
    parserModule = imp.load_source("parserModule",
                                   booger.TRAIN_PATH + '/tasks/semantic/dataset/' +
                                   self.DATA["name"] + '/parser.py')
    self.parser = parserModule.Parser(root=self.datadir,
                                      train_sequences=self.DATA["split"]["train"],
                                      valid_sequences=self.DATA["split"]["valid"],
                                      test_sequences=self.DATA["split"]["test"],
                                      labels=self.DATA["labels"],
                                      color_map=self.DATA["color_map"],
                                      learning_map=self.DATA["learning_map"],
                                      learning_map_inv=self.DATA["learning_map_inv"],
                                      sensor=self.ARCH["dataset"]["sensor"],
                                      max_points=self.ARCH["dataset"]["max_points"],
                                      batch_size=1,
                                      workers=self.ARCH["train"]["workers"],
                                      gt=True,
                                      shuffle_train=False)

    # concatenate the encoder and the head
    with torch.no_grad():
        torch.nn.Module.dump_patches = True
        if self.uncertainty:
            self.model = SalsaNextUncertainty(self.parser.get_n_classes())
            self.model = nn.DataParallel(self.model)
            w_dict = torch.load(modeldir + "/SalsaNext",
                                map_location=lambda storage, loc: storage)
            self.model = load_state(self.model, w_dict['state_dict'])
        else:
            self.model = SalsaNext(self.parser.get_n_classes())
            self.model = nn.DataParallel(self.model)
            w_dict = torch.load(modeldir + "/SalsaNext",
                                map_location=lambda storage, loc: storage)
            self.model = load_state(self.model, w_dict['state_dict'])

    # use knn post processing?
    self.post = None
    if self.ARCH["post"]["KNN"]["use"]:
      self.post = KNN(self.ARCH["post"]["KNN"]["params"],
                      self.parser.get_n_classes())

    # GPU?
    self.gpu = False
    self.model_single = self.model
    self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Infering in device: ", self.device)
    if torch.cuda.is_available() and torch.cuda.device_count() > 0:
      cudnn.benchmark = True
      cudnn.fastest = True
      self.gpu = True
      self.model.cuda()

  def infer(self):
    cnn = []
    knn = []
    if self.split == None:

        self.infer_subset(loader=self.parser.get_train_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)

        # do valid set
        self.infer_subset(loader=self.parser.get_valid_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
        # do test set
        self.infer_subset(loader=self.parser.get_test_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)


    elif self.split == 'valid':
        self.infer_subset(loader=self.parser.get_valid_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    elif self.split == 'train':
        self.infer_subset(loader=self.parser.get_train_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    else:
        self.infer_subset(loader=self.parser.get_test_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    print("Mean CNN inference time:{}\t std:{}".format(np.mean(cnn), np.std(cnn)))
    print("Mean KNN inference time:{}\t std:{}".format(np.mean(knn), np.std(knn)))
    print("Total Frames:{}".format(len(cnn)))
    print("Finished Infering")

    return

  def infer_subset(self, loader, to_orig_fn,cnn,knn):
    # switch to evaluate mode
    if not self.uncertainty:
      self.model.eval()
    total_time=0
    total_frames=0
    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      end = time.time()

      for i, (proj_in, proj_mask, _, _, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(loader):
        # first cut to rela size (batch size one allows it)
        p_x = p_x[0, :npoints]
        p_y = p_y[0, :npoints]
        proj_range = proj_range[0, :npoints]
        unproj_range = unproj_range[0, :npoints]
        path_seq = path_seq[0]
        path_name = path_name[0]

        if self.gpu:
          proj_in = proj_in.cuda()
          p_x = p_x.cuda()
          p_y = p_y.cuda()
          if self.post:
            proj_range = proj_range.cuda()
            unproj_range = unproj_range.cuda()

        #compute output
        if self.uncertainty:
            proj_output_r,log_var_r = self.model(proj_in)
            for i in range(self.mc):
                log_var, proj_output = self.model(proj_in)
                log_var_r = torch.cat((log_var, log_var_r))
                proj_output_r = torch.cat((proj_output, proj_output_r))

            proj_output2,log_var2 = self.model(proj_in)
            proj_output = proj_output_r.var(dim=0, keepdim=True).mean(dim=1)
            log_var2 = log_var_r.mean(dim=0, keepdim=True).mean(dim=1)
            if self.post:
                # knn postproc
                unproj_argmax = self.post(proj_range,
                                          unproj_range,
                                          proj_argmax,
                                          p_x,
                                          p_y)
            else:
                # put in original pointcloud using indexes
                unproj_argmax = proj_argmax[p_y, p_x]

            # measure elapsed time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            frame_time = time.time() - end
            print("Infered seq", path_seq, "scan", path_name,
                  "in", frame_time, "sec")
            total_time += frame_time
            total_frames += 1
            end = time.time()

            # save scan
            # get the first scan in batch and project scan
            pred_np = unproj_argmax.cpu().numpy()
            pred_np = pred_np.reshape((-1)).astype(np.int32)

            # log_var2 = log_var2[0][p_y, p_x]
            # log_var2 = log_var2.cpu().numpy()
            # log_var2 = log_var2.reshape((-1)).astype(np.float32)

            log_var2 = log_var2[0][p_y, p_x]
            log_var2 = log_var2.cpu().numpy()
            log_var2 = log_var2.reshape((-1)).astype(np.float32)
            # assert proj_output.reshape((-1)).shape == log_var2.reshape((-1)).shape == pred_np.reshape((-1)).shape

            # map to original label
            pred_np = to_orig_fn(pred_np)

            # save scan
            path = os.path.join(self.logdir, "sequences",
                                path_seq, "predictions", path_name)
            pred_np.tofile(path)

            path = os.path.join(self.logdir, "sequences",
                                path_seq, "log_var", path_name)
            if not os.path.exists(os.path.join(self.logdir, "sequences",
                                               path_seq, "log_var")):
                os.makedirs(os.path.join(self.logdir, "sequences",
                                         path_seq, "log_var"))
            log_var2.tofile(path)

            proj_output = proj_output[0][p_y, p_x]
            proj_output = proj_output.cpu().numpy()
            proj_output = proj_output.reshape((-1)).astype(np.float32)

            path = os.path.join(self.logdir, "sequences",
                                path_seq, "uncert", path_name)
            if not os.path.exists(os.path.join(self.logdir, "sequences",
                                               path_seq, "uncert")):
                os.makedirs(os.path.join(self.logdir, "sequences",
                                         path_seq, "uncert"))
            proj_output.tofile(path)
            print(path)

            print(total_time / total_frames)
        else:
            proj_output = self.model(proj_in)
            proj_argmax = proj_output[0].argmax(dim=0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            res = time.time() - end
            print("Network seq", path_seq, "scan", path_name,
                  "in", res, "sec")
            end = time.time()
            cnn.append(res)

            if torch.cuda.is_available():
                torch.cuda.synchronize()
            res = time.time() - end
            print("Network seq", path_seq, "scan", path_name,
                  "in", res, "sec")
            end = time.time()
            cnn.append(res)

            if self.post:
                # knn postproc
                unproj_argmax = self.post(proj_range,
                                          unproj_range,
                                          proj_argmax,
                                          p_x,
                                          p_y)
            else:
                # put in original pointcloud using indexes
                unproj_argmax = proj_argmax[p_y, p_x]

            # measure elapsed time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            res = time.time() - end
            print("KNN Infered seq", path_seq, "scan", path_name,
                  "in", res, "sec")
            knn.append(res)
            end = time.time()

            # save scan
            # get the first scan in batch and project scan
            pred_np = unproj_argmax.cpu().numpy()
            pred_np = pred_np.reshape((-1)).astype(np.int32)

            # map to original label
            pred_np = to_orig_fn(pred_np)

            # save scan
            path = os.path.join(self.logdir, "sequences",
                                path_seq, "predictions", path_name)
            pred_np.tofile(path)
            print(path)
            
  def infer_test(self):
    cnn = []
    knn = []
    if self.split == None:

        self.infer_test_subset(loader=self.parser.get_train_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)

        # do valid set
        self.infer_test_subset(loader=self.parser.get_valid_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
        # do test set
        self.infer_test_subset(loader=self.parser.get_test_set(),
                          to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)


    elif self.split == 'valid':
        self.infer_test_subset(loader=self.parser.get_valid_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    elif self.split == 'train':
        self.infer_test_subset(loader=self.parser.get_train_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    else:
        self.infer_test_subset(loader=self.parser.get_test_set(),
                        to_orig_fn=self.parser.to_original, cnn=cnn, knn=knn)
    print("Mean CNN inference time:{}\t std:{}".format(np.mean(cnn), np.std(cnn)))
    print("Mean KNN inference time:{}\t std:{}".format(np.mean(knn), np.std(knn)))
    print("Total Frames:{}".format(len(cnn)))
    print("Finished Infering")

    return


  @staticmethod
  def intersectionAndUnionGPU(output, target, K, ignore_index=255):
    # 'K' classes, output and target sizes are N or N * L or N * H * W, each value in range 0 to K - 1.
    assert (output.dim() in [1, 2, 3])
    assert output.shape == target.shape
    output = output.view(-1)
    target = target.view(-1)
    output[target == ignore_index] = ignore_index
    intersection = output[output == target]
    area_intersection = torch.histc(intersection, bins=K, min=0, max=K-1)
    area_output = torch.histc(output, bins=K, min=0, max=K-1)
    area_target = torch.histc(target, bins=K, min=0, max=K-1)
    area_union = area_output + area_target - area_intersection
    return area_intersection, area_union, area_target
  
  
  def infer_test_subset(self, loader, to_orig_fn,cnn,knn):
    # switch to evaluate mode
    if not self.uncertainty:
      self.model.eval()
    total_time=0
    total_frames=0
    intersection_meter = AverageMeter()
    union_meter = AverageMeter()
    pred_class = torch.range(0,23).tolist()
    
    # For distance evaluation
    intersection_finer_list = [AverageMeter() for i in range(10, 100, 5)]
    union_finer_list = [AverageMeter() for i in range(10, 100, 5)]
    target_finer_list = [AverageMeter() for i in range(10, 100, 5)]
    
    # empty the cache to infer in high res
    if self.gpu:
      torch.cuda.empty_cache()

    with torch.no_grad():
      end = time.time()

      for i, (proj_in, proj_mask, _, unproj_labels, path_seq, path_name, p_x, p_y, proj_range, unproj_range, _, _, _, _, npoints) in enumerate(loader):
        # first cut to rela size (batch size one allows it)
        p_x = p_x[0, :npoints]
        p_y = p_y[0, :npoints]
        proj_range = proj_range[0, :npoints]
        unproj_range = unproj_range[0, :npoints]
        unproj_labels = unproj_labels[0, :npoints]
        path_seq = path_seq[0]
        path_name = path_name[0]

        if self.gpu:
          proj_in = proj_in.cuda()
          p_x = p_x.cuda()
          p_y = p_y.cuda()
          if self.post:
            proj_range = proj_range.cuda()
            unproj_range = unproj_range.cuda()

        #compute output
        if self.uncertainty:
            proj_output_r,log_var_r = self.model(proj_in)
            for i in range(self.mc):
                log_var, proj_output = self.model(proj_in)
                log_var_r = torch.cat((log_var, log_var_r))
                proj_output_r = torch.cat((proj_output, proj_output_r))

            proj_output2,log_var2 = self.model(proj_in)
            proj_output = proj_output_r.var(dim=0, keepdim=True).mean(dim=1)
            log_var2 = log_var_r.mean(dim=0, keepdim=True).mean(dim=1)
            if self.post:
                # knn postproc
                unproj_argmax = self.post(proj_range,
                                          unproj_range,
                                          proj_argmax,
                                          p_x,
                                          p_y)
            else:
                # put in original pointcloud using indexes
                unproj_argmax = proj_argmax[p_y, p_x]

            # measure elapsed time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            frame_time = time.time() - end
            print("Infered seq", path_seq, "scan", path_name,
                  "in", frame_time, "sec")
            total_time += frame_time
            total_frames += 1
            end = time.time()

            # save scan
            # get the first scan in batch and project scan
            pred_np = unproj_argmax.cpu().numpy()
            pred_np = pred_np.reshape((-1)).astype(np.int32)

        else:
            proj_output = self.model(proj_in)
            proj_argmax = proj_output[0].argmax(dim=0)
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            res = time.time() - end
            print("Network seq", path_seq, "scan", path_name,
                  "in", res, "sec")
            end = time.time()
            cnn.append(res)

            if self.post:
                # knn postproc
                unproj_argmax = self.post(proj_range,
                                          unproj_range,
                                          proj_argmax,
                                          p_x,
                                          p_y)
            else:
                # put in original pointcloud using indexes
                unproj_argmax = proj_argmax[p_y, p_x]

            # measure elapsed time
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            res = time.time() - end
            print("KNN Infered seq", path_seq, "scan", path_name,
                  "in", res, "sec")
            knn.append(res)
            end = time.time()

        # Update and record performance
        unproj_labels = unproj_labels.cuda()
        area_intersection, area_union, area_target = self.intersectionAndUnionGPU(
          unproj_argmax, unproj_labels, 24, 255
        )
        intersection, union = area_intersection.cpu().numpy(), area_union.cpu().numpy()
        intersection_meter.update(intersection), union_meter.update(union)
        
        # Distance evaluation
        r = unproj_range
        masks_finer = []
        for r_start in range(10, 100, 5):
            masks_finer.append((r > r_start) & (r < r_start + 5))
        for iii, mask_finer in enumerate(masks_finer):
            intersection, union, tgt = self.intersectionAndUnionGPU(unproj_argmax[mask_finer], unproj_labels[mask_finer], 24, 255)
            intersection, union, tgt = intersection.cpu().numpy(), union.cpu().numpy(), tgt.cpu().numpy()
            intersection_finer_list[iii].update(intersection), union_finer_list[iii].update(union), target_finer_list[iii].update(tgt)
      
    # Compute overall performance
    iou_class = intersection_meter.sum / (union_meter.sum + 1e-10)
    mIoU = np.mean(iou_class)
    print('Val result: mIoU {:.4f}.'.format(mIoU))
    for i in range(24):
        print('Class_{} Result: iou {:.4f}.'.format(i, iou_class[i]))
        
    # dist eval
    iou_class_finer = [intersection_finer_list[i].sum / (union_finer_list[i].sum + 1e-10) for i in range(len(list(range(10, 100, 5))))]
    accuracy_class_finer = [intersection_finer_list[i].sum / (target_finer_list[i].sum + 1e-10) for i in range(len(list(range(10, 100, 5))))]
    mIoU_finer = [np.mean(iou_class_finer[i]) for i in range(len(list(range(10, 100, 5))))]
    mAcc_finer = [np.mean(accuracy_class_finer[i]) for i in range(len(list(range(10, 100, 5))))]
    allAcc_finer = [sum(intersection_finer_list[i].sum) / (sum(target_finer_list[i].sum) + 1e-10) for i in range(len(list(range(10, 100, 5))))]
    metrics_dist = ['{}'.format(distance) for distance in range(10, 100, 5)]
    print("-"*80)
    for ii in range(len(list(range(10, 100, 5)))):
        print('Val result_{}: mIoU/mAcc/allAcc {:.4f}/{:.4f}/{:.4f}.'.format(metrics_dist[ii], mIoU_finer[ii], mAcc_finer[ii], allAcc_finer[ii]))
    print("-"*80)
    print('<<<<<<<<<<<<<<<<< End Evaluation <<<<<<<<<<<<<<<<<')  