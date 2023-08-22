import torch
import VSWL
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
import torch.nn as nn
from numpy import *
device = "cuda" if torch.cuda.is_available() else "cpu"

# load classifier
num_class = 2
freeze = True   # freeze encoder and only train linear classifier (less likely to overfit when training data is limited)
df = pd.read_csv("")
# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
VSWL_model1 = VSWL.load_img_classification_model1(num_cls=num_class, freeze_encoder=freeze, device=device)

cls_prompts = VSWL.generate_INB_class_prompts()

# process input images and class prompts
processed_txt = VSWL_model1.process_class_prompts(cls_prompts, device)
processed_imgs = VSWL_model1.process_img(df['File_path'].tolist(), device)

# zero-shot classification
similarities = VSWL.zero_shot_classification(
    VSWL_model1, processed_imgs, processed_txt)

# print(similarities)

labels = df[VSWL.src.constants.INB_COMPETITION_TASKS].to_numpy().argmax(axis=1)
pred2 = similarities[VSWL.src.constants.INB_COMPETITION_TASKS]
pred_similiarities = []
m = nn.Softmax()
my_array = np.array(pred2)
my_tensor = torch.tensor(my_array)
pred3 = m(my_tensor)
print(len(pred2))
pred4_0 = []
pred4_1 = []
cc0=0
cc1=0
count = 0
for i in range(len(pred2)):
     cc0 += pred3[i][0]
     cc1 += pred3[i][1]
     count +=1
     if count == 4:
        pred4_0.append(cc0/4)
        pred4_1.append(cc1/4)
        cc0 = 0
        cc1 = 0
        count = 0
print(len(pred4_0))
for i in range(len(pred4_0)):
    if(labels[i] == 1):
        pred_similiarities.append(pred4_1[i])
    if (labels[i] == 0):
        pred_similiarities.append(pred4_0[i])

pre4 = torch.cat((torch.Tensor(pred4_0).view(88,1),torch.Tensor(pred4_1).view(88,1)),1)
print(pre4.shape)
pred = pre4.numpy().argmax(axis=1)
acc = len(labels[labels == pred]) / len(labels)
# print(labels)
# print(pred2)
print("(pred_similiarities):",pred_similiarities)

print("mean(pred_similiarities):",mean(pred_similiarities))
# print(acc)
print(classification_report(labels, pred, digits=3))
print("auc",roc_auc_score(labels, pred))

