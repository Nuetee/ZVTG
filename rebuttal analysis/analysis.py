import json

with open("TAG_charades_analysis.json") as f:
    data = json.load(f)

iou_list = data["IoU list"]
score_var_list = data["Score variance"]
proposal_length_var_list = data["Proposal length variance"]

print(sum(iou_list) / len(iou_list))
print(sum(score_var_list) / len(score_var_list))
print(sum(proposal_length_var_list) / len(proposal_length_var_list))
print('\n')

with open("TAG_activitynet_analysis.json") as f:
    data = json.load(f)

iou_list = data["IoU list"]
score_var_list = data["Score variance"]
proposal_length_var_list = data["Proposal length variance"]

print(sum(iou_list) / len(iou_list))
print(sum(score_var_list) / len(score_var_list))
print(sum(proposal_length_var_list) / len(proposal_length_var_list))
print('\n')

with open("TFVTG_charades_analysis.json") as f:
    data = json.load(f)

iou_list = data["IoU list"]
score_var_list = data["Score variance"]
proposal_length_var_list = data["Proposal length variance"]

print(sum(iou_list) / len(iou_list))
print(sum(score_var_list) / len(score_var_list))
print(sum(proposal_length_var_list) / len(proposal_length_var_list))
print('\n')

with open("TFVTG_activitynet_analysis.json") as f:
    data = json.load(f)

iou_list = data["IoU list"]
score_var_list = data["Score variance"]
proposal_length_var_list = data["Proposal length variance"]

print(sum(iou_list) / len(iou_list))
print(sum(score_var_list) / len(score_var_list))
print(sum(proposal_length_var_list) / len(proposal_length_var_list))
print('\n')