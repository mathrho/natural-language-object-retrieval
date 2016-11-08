# coding: utf-8

from __future__ import division, print_function

import sys
import matplotlib
import numpy as np
import skimage.io
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
#get_ipython().magic(u'matplotlib inline')
sys.path.append('../')
sys.path.append('../external/caffe-natural-language-object-retrieval/python/')
sys.path.append('../external/caffe-natural-language-object-retrieval/examples/coco_caption/')
import caffe

import util
from captioner import Captioner
import retriever
from glob import glob, iglob
from computeIOU import computeIOU
from tqdm import tqdm
import xml.etree.ElementTree as ET


####################################################
pretrained_weights_path = '../models/scrc_full_vgg.caffemodel'
gpu_id = 0  # the GPU to test the SCRC model

# Initialize the retrieval model
image_net_proto = '../prototxt/VGG_ILSVRC_16_layers_deploy.prototxt'
lstm_net_proto = '../prototxt/scrc_word_to_preds_full.prototxt'
vocab_file = '../data/vocabulary.txt'
# utilize the captioner module from LRCN
captioner = Captioner(pretrained_weights_path, image_net_proto, lstm_net_proto,
                      vocab_file, gpu_id)
captioner.set_image_batch_size(40)  # decrease the number if your GPU memory is small
vocab_dict = retriever.build_vocab_dict_from_captioner(captioner)

####################################################
videofiles = sorted(glob('/home/zhenyang/Workspace/data/Tracker_Benchmark_v1.0/*'))
for videonfile in videofiles:
    video = videonfile.split('/')[-1]
    print(video)

    start_frame_id = 1
    if video == 'David':
        start_frame_id = 300
    elif video == 'Tiger1':
        start_frame_id = 6

    # First, select query
    query_file = '../data/OTB50Entities/' + video + '.xml'
    root = ET.parse( query_file ).getroot()
    # querier = prettify( querier )
    print(root[3][1].text)
    query = root[3][1].text

    # Second, get gt box
    gt_file = '/home/zhenyang/Workspace/data/Tracker_Benchmark_v1.0/' + video + '/groundtruth_rect.txt'
    try:
        gt_boxes = np.loadtxt(gt_file, delimiter=',').astype(int)
    except ValueError:
        gt_boxes = np.loadtxt(gt_file, delimiter='\t').astype(int)
    num_frames = gt_boxes.shape[0]

    counter = 0
    results = np.zeros((num_frames, 4), np.int)
    #frames = sorted(glob('/home/zhenyang/Workspace/data/Tracker_Benchmark_v1.0/'+video+'/img/*.jpg'))
    for fi in range(start_frame_id, num_frames+start_frame_id):
        im_file = '/home/zhenyang/Workspace/data/Tracker_Benchmark_v1.0/' + video + '/img/%04d.jpg' % (fi,)
        edgebox_file = '../data/OTB50_edgeboxes_top100/' +  video + '/%04d.txt' % (fi,) # pre-extracted EdgeBox proposals

        ###############################
        im = skimage.io.imread(im_file)
        imsize = np.array([im.shape[1], im.shape[0]])  # [width, height]
        candidate_boxes = np.loadtxt(edgebox_file).astype(int).reshape((-1, 4))

        #print(candidate_boxes.shape)
        #print(gt_box.shape)

        # Compute features
        region_feature = retriever.compute_descriptors_edgebox(captioner, im,
                                                        candidate_boxes)
        spatial_feature = retriever.compute_spatial_feat(candidate_boxes, imsize)
        descriptors = np.concatenate((region_feature, spatial_feature), axis=1)
        context_feature = captioner.compute_descriptors([im], output_name='fc7')

        # Compute scores of each candidate region
        scores = retriever.score_descriptors_context(descriptors, query,
                                                 context_feature, captioner,
                                                 vocab_dict)

        # Retrieve the top-scoring candidate region given the query
        retrieved_bbox = candidate_boxes[np.argmax(scores)]
        # Save the retrieval result
        results[counter, :] = retrieved_bbox
        counter = counter + 1

        # Visualize the retrieval result
        #plt.figure(figsize=(12, 8))
        #plt.imshow(im)
        #ax = plt.gca()
        #x_min, y_min, x_max, y_max = retrieved_bbox
        #ax.add_patch(mpatches.Rectangle((x_min, y_min), x_max-x_min+1, y_max-y_min+1,
        #                                fill=False, edgecolor='r', linewidth=5))
        #ax.add_patch(mpatches.Rectangle((gt_box[0], gt_box[1]), gt_box[2]-gt_box[0]+1, gt_box[3]-gt_box[1]+1,
        #                                fill=False, edgecolor='g', linewidth=5))
        #ax.add_patch(mpatches.Rectangle((bo_box[0], bo_box[1]), bo_box[2]-bo_box[0]+1, bo_box[3]-bo_box[1]+1,
        #                                fill=False, edgecolor='b', linestyle='dashed',linewidth=5))
        #_ = plt.title("query = '%s'" % query)
        #plt.savefig('../results/OTB50_edgeboxes_top100/' +  video + '.png')
        #plt.close( )

    # save results to file
    filename = '../results/OTB50_edgeboxes_top100/'+video+'_'+'scrc'+'.txt'
    if video == 'Tiger1':
        filename = '../results/OTB50_edgeboxes_top100/'+video+'_refined_'+'scrc'+'.txt'
    fp = open(filename, 'w')
    for jj in range(num_frames):
        fp.write('%d %d %d %d\n'%(int(results[jj,0]),int(results[jj,1]),int(results[jj,2]),int(results[jj,3])))
    fp.close()

print('Finish evlaluation on the whole data set.')

