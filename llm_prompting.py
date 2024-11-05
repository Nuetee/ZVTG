import json
import numpy as np
import os

def calc_iou(candidates, gt):
    start, end = candidates[:,0], candidates[:,1]
    s, e = gt[0], gt[1]
    inter = np.minimum(end, e) - np.maximum(start, s)
    union = np.maximum(end, e) - np.minimum(start, s)
    return inter.clip(min=0) / union

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


def select_proposal2(inputs, gamma=0.6):
    weights = inputs[:, -1].clip(min=0)
    proposals = inputs[:, :-1]
    scores = np.zeros_like(weights)

    for j in range(scores.shape[0]):
        iou = calc_iou(proposals, proposals[j])
        scores[j] += (iou ** gamma * weights).sum()

    # 인덱스도 함께 저장한 상태로 정렬
    indexed_inputs = [(i, inputs[i]) for i in range(len(inputs))]  # (인덱스, proposal)의 튜플로 저장
    indexed_inputs.sort(key=lambda x: -scores[x[0]])  # scores를 기준으로 내림차순 정렬

    # 정렬된 proposal과 함께 인덱스를 반환
    sorted_indices = [x[0] for x in indexed_inputs]  # 원래 proposal의 인덱스들
    sorted_proposals = [x[1] for x in indexed_inputs]  # 정렬된 proposal들

    return np.array(sorted_proposals), sorted_indices

searched_proposals2 = []
searched_proposal_indices2 = []  # 추가된 리스트: 원본 인덱스를 추적
def search_combination2(cands, idx, cur=[], cur_indices=[], relation='sequentially'):
    if idx >= len(cands):
        cur = np.array(cur)
        # 현재 선택된 쿼리의 시작시간의 최대값이, 종료시간의 최소값보다 작으면 (겹치는 구간이 없으면) 모순이므로 return
        if relation == 'simultaneously' and cur.max(axis=0, keepdims=True)[:, 0] > cur.min(axis=0, keepdims=True)[:, 1]:
            return
        st = cur.min(axis=0, keepdims=True)[:, 0]
        end = cur.max(axis=0, keepdims=True)[:, 1]
        score = cur[:, -1].clip(min=0).prod()
        
        global searched_proposals2, searched_proposal_indices2
        searched_proposals2.append([float(st), float(end), float(score)])
        searched_proposal_indices2.append(cur_indices)  # 현재까지 선택된 proposal 인덱스 기록
        return
    
    for cur_idx in range(len(cands[idx])):
        # 다음 쿼리의 시작 시간(cands[idx][cur_idx][0])이 현재 선택된 쿼리의 종료시간(cur[-1][1])보다 커야 연속적이라고 판단. 작으면 continue
        if len(cur) > 0 and relation == 'sequentially' and cands[idx][cur_idx][0] < cur[-1][1]:
            continue
        
        # 현재 쿼리의(cur) 시작시간이 다음 쿼리(cands[idx][cur_idx])의 시작시간보다 뒤면 모순, continue
        # if len(cur) > 0 and relation == 'sequentially' and cands[idx][cur_idx][0] < cur[-1][0]:
        #     continue

        # 조건을 만족하면 다음 쿼리 proposal을 선택하고. 다다음 쿼리에 대한 탐색
        search_combination2(cands, idx+1, cur + [cands[idx][cur_idx]], cur_indices + [(idx, cur_idx)], relation)


def filter_and_integrate2(sub_query_proposals, relation):
    if len(sub_query_proposals) == 0:
        return [], []
    global searched_proposals2, searched_proposal_indices2
    searched_proposals2 = []
    searched_proposal_indices2 = []  # 인덱스 기록 초기화
    search_combination2(sub_query_proposals, 0, cur=[], cur_indices=[], relation=relation)
    
    if len(searched_proposals2) == 0:
        return [], []

    proposals = select_proposal2(np.array(searched_proposals2))
    
    # select_proposal에서 정렬된 proposal과 함께 원래 인덱스도 반환
    proposals, original_indices = select_proposal2(np.array(searched_proposals2))

    # 최종적으로 선택된 proposal과 그에 해당하는 원본 인덱스 추적
    selected_proposals = proposals.tolist()[:2]
    selected_indices = [searched_proposal_indices2[i] for i in original_indices[:2]]  # 상위 2개의 인덱스 추적

    return selected_proposals, selected_indices
