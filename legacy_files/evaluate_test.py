from data_configs import DATASETS
import argparse
import numpy as np
import matplotlib.pyplot as plt
import json
import torch
from tqdm import tqdm
from legacy_files.vlm_localizer_test import localize
import os
from llm_prompting import select_proposal, filter_and_integrate

def get_args():
    parser = argparse.ArgumentParser(description='Evaluation for training-free video temporal grounding.')
    parser.add_argument('--dataset', default='charades', type=str, help='Specify the dataset. See supported datasets in data_configs.py.')
    parser.add_argument('--llm_output', default=None, type=str, help='LLM prompt output. If not specified, use only VLM for evaluation.')

    return parser.parse_args()


def calc_iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0], gt[1]
    inter = np.minimum(end, e) - np.maximum(start, s)
    union = np.maximum(end, e) - np.minimum(start, s)
    return inter.clip(min=0) / union


def eval(data, feature_path, stride, hyperparams):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])
    pbar = tqdm(data.items())

    for vid, ann in pbar:
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))
        
        for i in range(len(ann['sentences'])):
            gt = ann['timestamps'][i]
            query_json = [{'descriptions': ann['sentences'][i]}]
            proposals, ori_scores = localize(video_feature, duration, gt, query_json, stride, hyperparams)
            proposals = select_proposal(np.array(proposals))

            iou_ = calc_iou(proposals[:1], gt)[0]
            ious.append(max(iou_, 0))
            recall += thresh <= iou_

        pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})

    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))


def eval_test(data, feature_path, stride, hyperparams):
    pbar = tqdm(data.items())

    for vid, ann in pbar:
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))
        
        for i in range(len(ann['sentences'])):
            gt = ann['timestamps'][i]
            start = gt[0] / duration * video_feature.shape[0]
            end = gt[1] / duration * video_feature.shape[0]
            query_json = [{'descriptions': ann['sentences'][i]}]
            _, ori_scores = localize(video_feature, duration, gt, query_json, stride, hyperparams)

            ori_scores = ori_scores.squeeze()
            ori_scores_np = ori_scores.cpu().numpy()
            masked_sentences_list = ann['masked'][i]

            diff_scores_sum = None
            for key, masked_sentences in masked_sentences_list.items():
                for masked_sentence in masked_sentences:
                    query_json = [{'descriptions': masked_sentence}]
                    import pdb;pdb.set_trace()
                    _, masked_scores = localize(video_feature, duration, gt, query_json, stride, hyperparams)
                    masked_scores = masked_scores.squeeze()
                    masked_scores_np = masked_scores.cpu().numpy()

                    delta_ori = np.diff(ori_scores_np)
                    delta_masked = np.diff(masked_scores_np)

                    c = np.dot(delta_ori, delta_masked) / np.dot(delta_masked, delta_masked)
                    masked_scores_scaled_np = masked_scores_np * c

                    diff = ori_scores_np - masked_scores_scaled_np
                    t = np.median(diff)
                    masked_scores_adjusted_np = masked_scores_scaled_np + t

                    diff_scores = ori_scores_np - masked_scores_adjusted_np
                    if diff_scores_sum is None:
                        diff_scores_sum = diff_scores
                    else:
                        diff_scores_sum += diff_scores



            # verb_masked_sentences = ann['masked'][i]['verb_masked']
            # for verb_masked in verb_masked_sentences:
            #     query_json = [{'descriptions': verb_masked}]
            #     _, masked_scores = localize(video_feature, duration, gt, query_json, stride, hyperparams)
            #     masked_scores = masked_scores.squeeze()

            #     masked_scores_np = masked_scores.cpu().numpy()
            #     delta_ori = np.diff(ori_scores_np)
            #     delta_masked = np.diff(masked_scores_np)

            #     c = np.dot(delta_ori, delta_masked) / np.dot(delta_masked, delta_masked)
            #     masked_scores_scaled_np = masked_scores_np * c

            #     diff = ori_scores_np - masked_scores_scaled_np
            #     t = np.median(diff)
            #     masked_scores_adjusted_np = masked_scores_scaled_np + t

        
        fig, ax1 = plt.subplots(figsize=(10, 6))

        ax1.set_xlabel("Frame")
        ax1.set_ylabel("Original sentnence similarity", color='black')
        ax1.plot(ori_scores_np, label="S_original", linestyle='-')
        
        ax2 = ax1.twinx()
        ax2.set_ylabel('Diff simialrity b/w masked & original', color='blue')
        ax2.plot(diff_scores_sum, label='S_difference', linestyle='--')
        ax1.axvspan(start, end, color='grey', alpha=0.5, label='Timestamp')
        plt.title("Comparison of S_original and S_diffenence")
        plt.legend()
        plt.grid(True)
        save_path = os.path.join(f"./masked_plot_test/{vid}_sentence_{ann['sentences'][i]}.png")
        plt.savefig(save_path)
        plt.close()

            # plt.figure(figsize=(10, 6))
            # plt.plot(ori_scores_np, label="S_original", linestyle='-')
            # plt.plot(masked_scores_np, label="S_masked", linestyle='--')
            # plt.plot(masked_scores_adjusted_np, label="S_masked_adjusted", linestyle='-.')
            # plt.axvspan(start, end, color='grey', alpha=0.5, label='Timestamp')
            # plt.xlabel("Index")
            # plt.ylabel("Similarity")
            # plt.title("Comparison of S_original, S_masked, and S_masked_adjusted")
            # plt.legend()
            # plt.grid(True)
            # save_path = os.path.join(f"./masked_plot_test/{vid}_sentence_{ann['sentences'][i]}.png")
            # plt.savefig(save_path)
            # plt.close()

import random
def eval_test2(data, feature_path, stride, hyperparams):
    ious = []
    thresh = np.array([0.3, 0.5, 0.7])
    recall = np.array([0, 0, 0])

    # # Shuffle the data keys
    # shuffled_data = list(data.items())
    # random.shuffle(shuffled_data)
    # pbar = tqdm(shuffled_data)

    pbar = tqdm(data.items())

    for vid, ann in pbar:
        duration = ann['duration']
        video_feature = np.load(os.path.join(feature_path, vid+'.npy'))
        
        for i in range(len(ann['sentences'])):
            gt = ann['timestamps'][i]
            query_json = [{'descriptions': ann['sentences'][i]}]
            proposals = localize(video_feature, gt, duration, query_json, stride, hyperparams)
            proposals = select_proposal(np.array(proposals))

            iou_ = calc_iou(proposals[:1], gt)[0]
            ious.append(max(iou_, 0))
            recall += thresh <= iou_

        pbar.set_postfix({"mIoU": sum(ious) / len(ious), 'recall': str(recall / len(ious))})

    print('mIoU:', sum(ious) / len(ious))
    for th, r in zip(thresh, recall):
        print(f'R@{th}:', r / len(ious))

if __name__=='__main__':
    args = get_args()
    assert args.dataset in DATASETS, 'Unsupported dataset. To evaluate other datasets, please add the configuration in data_configs.py.'
    dataset = DATASETS[args.dataset]
        
    with open(args.llm_output) as f:
        data = json.load(f)
    eval_test2(data, dataset['feature_path'], dataset['stride'], dataset['hyper_parameters'])