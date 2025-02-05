import json
import numpy as np
import os

def calc_iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0], gt[1]
    inter = np.minimum(end, e) - np.maximum(start, s)
    union = np.maximum(end, e) - np.minimum(start, s)
    return inter.clip(min=0) / union


def select_proposal_with_score(inputs, gamma=0.6):
    weights = inputs[:, -1].clip(min=0)
    proposals = inputs[:, :-1]
    scores = np.zeros_like(weights)

    for j in range(scores.shape[0]):
        iou = calc_iou(proposals, proposals[j])
        scores[j] += (iou ** gamma * weights).sum()

    idx = np.argsort(-scores)
    sorted_inputs = inputs[idx]
    sorted_scores = scores[idx]

    return sorted_inputs, sorted_scores


# Original
# region
def select_proposal(inputs, gamma=0.6):
    weights = inputs[:, -1].clip(min=0)
    proposals = inputs[:, :-1]
    scores = np.zeros_like(weights)

    for j in range(scores.shape[0]):
        iou = calc_iou(proposals, proposals[j])
        scores[j] += (iou ** gamma * weights).sum()

    idx = np.argsort(-scores)
    return inputs[idx]

searched_proposals = []
def search_combination(cands, idx, cur=[], relation='sequentially'):
    if idx >= len(cands):
        cur = np.array(cur)
        # 현재 선택된 쿼리의 시작시간의 최대값이, 종료시간의 최소값보다 작으면 (겹치는 구간이 없으면) 모순이므로 return
        if relation == 'simultaneously' and cur.max(axis=0, keepdims=True)[:, 0] > cur.min(axis=0, keepdims=True)[:, 1]:
            return
        st = cur.min(axis=0, keepdims=True)[:, 0]
        end = cur.max(axis=0, keepdims=True)[:, 1]
        score = cur[:, -1].clip(min=0).prod()
        global searched_proposals
        searched_proposals.append([float(st), float(end), float(score)])
        return
    
    for cur_idx in range(len(cands[idx])):
        # 다음 쿼리의 시작 시간(cands[idx][cur_idx][0])이 현재 선택된 쿼리의 종료시간(cur[-1][1])보다 커야 연속적이라고 판단. 작으면 continue
        if len(cur) > 0 and relation == 'sequentially' and cands[idx][cur_idx][0] < cur[-1][1]:
            continue
        
        # 현재 쿼리의(cur) 시작시간이 다음 쿼리(cands[idx][cur_idx])의 시작시간보다 뒤면 모순, continue
        # if len(cur) > 0 and relation == 'sequentially' and cands[idx][cur_idx][0] < cur[-1][0]:
        #     continue

        # 조건을 만족하면 다음 쿼리 proposal을 선택하고. 다다음 쿼리에 대한 탐색
        search_combination(cands, idx+1, cur + [cands[idx][cur_idx]], relation)


def filter_and_integrate(sub_query_proposals, relation):
    if len(sub_query_proposals) == 0:
        return []
    global searched_proposals
    searched_proposals = []
    search_combination(sub_query_proposals, 0, cur=[], relation=relation)
    if len(searched_proposals) == 0:
        return []
    proposals = select_proposal(np.array(searched_proposals))

    return proposals.tolist()[:2]
# endregion