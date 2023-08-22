import torch
import torch.nn as nn
import numpy as np
import random
from torch.nn import functional
from torch.autograd import Variable

def cosine_similarity(x1, x2, dim=1, eps=1e-8):
    """Returns cosine similarity between x1 and x2, computed along dim."""
    w12 = torch.sum(x1 * x2, dim)
    w1 = torch.norm(x1, 2, dim)
    w2 = torch.norm(x2, 2, dim)
    return (w12 / (w1 * w2).clamp(min=eps)).squeeze()

def attention_fn1(query, context, temp1):
    batch_size, queryL = query.size(0), query.size(2)
    ih, iw = context.size(2), context.size(3)
    sourceL = ih * iw

    # --> batch x sourceL x ndf
    context = context.view(batch_size, -1, sourceL)
    contextT = torch.transpose(context, 1, 2).contiguous()

    # Get attention
    # (batch x sourceL x ndf)(batch x ndf x queryL)
    # -->batch x sourceL x queryL
    attn = torch.bmm(contextT, query)
    # --> batch*sourceL x queryL
    attn = attn.view(batch_size * sourceL, queryL)
    attn = nn.Softmax(dim=-1)(attn)
    # --> batch x sourceL x queryL
    attn = attn.view(batch_size, sourceL, queryL)
    # --> batch*queryL x sourceL
    attn = torch.transpose(attn, 1, 2).contiguous()
    attn = attn.view(batch_size * queryL, sourceL)
    attn = attn * temp1
    attn = nn.Softmax(dim=-1)(attn)
    attn = attn.view(batch_size, queryL, sourceL)
    # --> batch x sourceL x queryL
    attnT = torch.transpose(attn, 1, 2).contiguous()
    # (batch x ndf x sourceL)(batch x sourceL x queryL)
    # --> batch x ndf x queryL
    weightedContext = torch.bmm(context, attnT)
    return weightedContext, attn.view(batch_size, -1, ih, iw)

def dh_loss(
   m, y, images, ato):
    mse_loss = torch.nn.MSELoss().type(torch.cuda.FloatTensor)
    # zeros = torch.zeros(1, dtype=torch.float16).cuda()
    # ones = torch.ones(1, dtype=torch.float16).cuda()
    # ones_ = ones * 0.8
    # m = torch.where(m > ones_, zeros, m)
    # dehaze_loss = mse_loss( (1-m)  * y + m * ato, images) # transformer
    dehaze_loss = mse_loss( m * y + (1 - m) * ato, images) #reconly

    return dehaze_loss

class blur_loss(nn.Module):
    def __init__(self):
        """
        Loss on the variance of the image.
        Works in the grayscale.
        If the image is smooth, gets zero
        """
        super(blur_loss, self).__init__()
        blur = (1 / 25) * np.ones((5, 5))
        blur = blur.reshape(1, 1, blur.shape[0], blur.shape[1])
        self.mse = nn.MSELoss()
        self.blur = nn.Parameter(data=torch.cuda.FloatTensor(blur), requires_grad=False)
        image = np.zeros((5, 5))
        image[2, 2] = 1
        image = image.reshape(1, 1, image.shape[0], image.shape[1])
        self.image = nn.Parameter(data=torch.cuda.FloatTensor(image), requires_grad=False)
        self.gray_scale = GrayscaleLayer()

    def forward(self, x):
        x = self.gray_scale(x)
        return self.mse(functional.conv2d(x, self.image), functional.conv2d(x, self.blur))

class GrayscaleLayer(nn.Module):
    def __init__(self):
        super(GrayscaleLayer, self).__init__()

    def forward(self, x):
        return torch.mean(x, 1, keepdim=True)

def Blur_loss(m, y, images):
    blur_m_loss = blur_loss().type(torch.cuda.FloatTensor)
    # zeros = torch.zeros(1, dtype=torch.float16).cuda()
    # ones = torch.ones(1, dtype=torch.float16).cuda()
    # ones_ = ones * 0.8
    # m = torch.where(m > ones_, zeros, m)
    # Blur_loss2 = blur_m_loss(1-m)
    Blur_loss2 = blur_m_loss(m)
    return Blur_loss2

def inner_loss_d(cnn_code, rnn_mlp1, rnn_mlp2, b, fc, eps=1e-8, temp3=10.0):#rnn[2,768,2]
    batch_size = cnn_code.shape[0]
    cc = torch.cat((torch.split(cnn_code,1,dim=0)[0], torch.split(cnn_code,1,dim=0)[2], torch.split(cnn_code,1,dim=0)[4],torch.split(cnn_code,1,dim=0)[6],torch.split(cnn_code,1,dim=0)[8],torch.split(cnn_code,1,dim=0)[10]),dim=0)
    mlo = torch.cat((torch.split(cnn_code,1,dim=0)[1], torch.split(cnn_code,1,dim=0)[3], torch.split(cnn_code,1,dim=0)[5],torch.split(cnn_code,1,dim=0)[7],torch.split(cnn_code,1,dim=0)[9],torch.split(cnn_code,1,dim=0)[11]),dim=0)

    cc = cc.view(cc.size(0), cc.size(1) * cc.size(2) * cc.size(3))
    mlo = mlo.view(mlo.size(0), mlo.size(1) * mlo.size(2) * mlo.size(3))
    rnn_mlp1 = torch.cat(
        (torch.split(rnn_mlp1, 1, dim=0)[0], torch.split(rnn_mlp1, 1, dim=0)[4], torch.split(rnn_mlp1, 1, dim=0)[8]), 0)
    rnn_mlp2 = torch.cat(
        (torch.split(rnn_mlp2, 1, dim=0)[0], torch.split(rnn_mlp2, 1, dim=0)[4], torch.split(rnn_mlp2, 1, dim=0)[8]), 0)

    rnn_mlp1 = rnn_mlp1.view(rnn_mlp1.size(0), rnn_mlp1.size(1) * rnn_mlp1.size(2))
    rnn_mlp2 = rnn_mlp2.view(rnn_mlp2.size(0), rnn_mlp2.size(1) * rnn_mlp2.size(2))

    if cc.dim() == 2:
        cc = cc.unsqueeze(0)
        mlo = mlo.unsqueeze(0)
        rnn_mlp1 = rnn_mlp1.unsqueeze(0)
        rnn_mlp2 = rnn_mlp2.unsqueeze(0)


    cc_norm = torch.norm(cc, 2, dim=2, keepdim=True)
    mlo_norm = torch.norm(mlo, 2, dim=2, keepdim=True)

    rnn_mlp1_norm = torch.norm(rnn_mlp1, 2, dim=2, keepdim=True)
    rnn_mlp2_norm = torch.norm(rnn_mlp2, 2, dim=2, keepdim=True)

    labels = Variable(torch.LongTensor(range(int(batch_size / 2)))).to(cnn_code.device)
    scores0 = torch.bmm(cc, mlo.transpose(1, 2))
    norm0 = torch.bmm(cc_norm, mlo_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * temp3

    scores0_rnn = torch.bmm(rnn_mlp1, rnn_mlp2.transpose(1, 2))
    norm0_rnn = torch.bmm(rnn_mlp1_norm, rnn_mlp2_norm.transpose(1, 2))
    scores0_rnn = scores0_rnn / norm0_rnn.clamp(min=eps) * temp3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()
    scores1 = scores0.transpose(0,1)
    loss0 = nn.CrossEntropyLoss()(scores0, labels)
    loss1 = nn.CrossEntropyLoss()(scores1, labels)

    return loss0+loss1

def inner_loss_g(cnn_code, rnn_mlp1, rnn_mlp2, b, fc, eps=1e-8, temp3=10.0):
    batch_size = cnn_code.shape[0]
    cc = torch.cat((torch.split(cnn_code, 1, dim=0)[0], torch.split(cnn_code, 1, dim=0)[2],
                    torch.split(cnn_code, 1, dim=0)[4], torch.split(cnn_code, 1, dim=0)[6],
                    torch.split(cnn_code, 1, dim=0)[8], torch.split(cnn_code, 1, dim=0)[10]), dim=0)
    mlo = torch.cat((torch.split(cnn_code, 1, dim=0)[1], torch.split(cnn_code, 1, dim=0)[3],
                     torch.split(cnn_code, 1, dim=0)[5], torch.split(cnn_code, 1, dim=0)[7],
                     torch.split(cnn_code, 1, dim=0)[9], torch.split(cnn_code, 1, dim=0)[11]), dim=0)

    rnn_mlp1 = torch.cat((torch.split(rnn_mlp1, 1, dim=0)[0], torch.split(rnn_mlp1, 1, dim=0)[4], torch.split(rnn_mlp1, 1, dim=0)[8]), 0)
    rnn_mlp2 = torch.cat(
        (torch.split(rnn_mlp2, 1, dim=0)[0], torch.split(rnn_mlp2, 1, dim=0)[4], torch.split(rnn_mlp2, 1, dim=0)[8]), 0)

    rnn_mlp1 = rnn_mlp1.view(rnn_mlp1.size(0), rnn_mlp1.size(1) * rnn_mlp1.size(2))
    rnn_mlp2 = rnn_mlp2.view(rnn_mlp2.size(0), rnn_mlp2.size(1) * rnn_mlp2.size(2))

    if cc.dim() == 2:
        cc = cc.unsqueeze(0)
        mlo = mlo.unsqueeze(0)
        rnn_mlp1 = rnn_mlp1.unsqueeze(0)
        rnn_mlp2 = rnn_mlp2.unsqueeze(0)

    cc_norm = torch.norm(cc, 2, dim=2, keepdim=True)
    mlo_norm = torch.norm(mlo, 2, dim=2, keepdim=True)

    rnn_mlp1_norm = torch.norm(rnn_mlp1, 2, dim=2, keepdim=True)
    rnn_mlp2_norm = torch.norm(rnn_mlp2, 2, dim=2, keepdim=True)

    labels = Variable(torch.LongTensor(range(int(batch_size / 2)))).to(cnn_code.device)
    scores0 = torch.bmm(cc, mlo.transpose(1, 2))
    norm0 = torch.bmm(cc_norm, mlo_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * temp3

    # --> batch_size x batch_size   inner
    scores0 = scores0.squeeze()
    scores1 = scores0.transpose(0, 1)
    loss0 = nn.CrossEntropyLoss()(scores0, labels)
    loss1 = nn.CrossEntropyLoss()(scores1, labels)


    return loss0 + loss1

def global_loss(cnn_code, rnn_code, keyword, fc, eps=1e-8, temp3=10.0):

    batch_size = cnn_code.shape[0]
    # labels = Variable(torch.LongTensor(range(batch_size))).to(cnn_code.device)
    labels_inner = Variable(torch.LongTensor(range(3))).cuda()  # [0,1]
    # labels = torch.zeros(batch_size,dtype=torch.long).cuda()

    if cnn_code.dim() == 2:
        cnn_code = cnn_code.unsqueeze(0)
        rnn_code = rnn_code.unsqueeze(0)

    cnn_code_norm = torch.norm(cnn_code, 2, dim=2, keepdim=True)
    rnn_code_norm = torch.norm(rnn_code, 2, dim=2, keepdim=True)

    scores0 = torch.bmm(cnn_code, rnn_code.transpose(1, 2))
    norm0 = torch.bmm(cnn_code_norm, rnn_code_norm.transpose(1, 2))
    scores0 = scores0 / norm0.clamp(min=eps) * temp3

    # --> batch_size x batch_size
    scores0 = scores0.squeeze()

    # scores1 = scores0.transpose(0, 1)
    similarities_ = []
    for n in range(1):
        similarities_.append(torch.mean(scores0[(n * 4):((n + 1) * 4), (n * 4):((n + 1) * 4)]))  # [0:4,0:4]
        similarities_.append(torch.mean(scores0[(n * 4):((n + 1) * 4), ((n + 1) * 4):(n + 2) * 4]))  # [0:4,4:8]
        similarities_.append(torch.mean(scores0[(n * 4):((n + 1) * 4), ((n + 2) * 4):(n + 3) * 4]))  # [0:4,8:12]
        similarities_.append(torch.mean(scores0[((n + 1) * 4):(n + 2) * 4, (n * 4):((n + 1) * 4)]))  # [4:8,0:4]
        similarities_.append(
            torch.mean(scores0[((n + 1) * 4):(n + 2) * 4, ((n + 1) * 4):((n + 2) * 4)]))  # [4:8,4:8]
        similarities_.append(
            torch.mean(scores0[((n + 1) * 4):(n + 2) * 4, ((n + 2) * 4):((n + 3) * 4)]))  # [4:8,8:12]
        similarities_.append(torch.mean(scores0[((n + 2) * 4):(n + 3) * 4, (n * 4):((n + 1) * 4)]))  # [4:8,0:4]
        similarities_.append(
            torch.mean(scores0[((n + 2) * 4):(n + 3) * 4, ((n + 1) * 4):((n + 2) * 4)]))  # [4:8,4:8]
        similarities_.append(
            torch.mean(scores0[((n + 2) * 4):(n + 3) * 4, ((n + 2) * 4):((n + 3) * 4)]))  # [4:8,8:12]
    similarities_ = torch.stack(similarities_).view(3, 3)
    similarities1_ = similarities_.transpose(0, 1)
    loss0 = nn.CrossEntropyLoss()(similarities_, labels_inner)  # labels: arange(batch_size)
    loss1 = nn.CrossEntropyLoss()(similarities1_, labels_inner)

    idx_rank, rank = rank_kekyword(keyword, similarities_)
    label_ranks = []

    for i in range(len(rank) - 1):
        t = i + 1
        for j in range(t, len(rank)):
            for k in range(len(rank[i][0])):

                for w in range(len(rank[j][0])):
                    a = rank[i][0]
                    label_ranks.append(a[k])
                    b = rank[j][0]
                    label_ranks.append(b[w])
            if (t < len(rank)):
                t += 1
    if (len(label_ranks) == 0):
        coarse_to_fine_loss = 0
    else:
        a = int(len(label_ranks) / 2)
        label_ranks = torch.stack(label_ranks).view(a, 2)
        label_rank = torch.ones(a, dtype=torch.long).cuda()
        coarse_to_fine_loss = nn.CrossEntropyLoss()(label_ranks, label_rank)

    return loss0, loss1, coarse_to_fine_loss

def local_loss(
    img_features, words_emb, cap_lens, label, keyword, fc, temp1=4.0, temp2=5.0, temp3=10.0, agg="sum"
):
    batch_size = img_features.shape[0]
    att_maps = []
    similarities = []
    # cap_lens = cap_lens.data.tolist()
    for i in range(words_emb.shape[0]):

        # Get the i-th text description
        words_num = cap_lens[i]  # 25
        # TODO: remove [SEP]
        # word = words_emb[i, :, 1:words_num+1].unsqueeze(0).contiguous()    # [1, 768, 25]
        word = words_emb[i, :, :words_num].unsqueeze(0).contiguous()  # [1, 768, 25]
        word = word.repeat(batch_size, 1, 1)  # [48, 768, 25]
        context = img_features  # [48, 768, 19, 19]
        weiContext, attn = attention_fn1(
            word, context, temp1
        )  # [48, 768, 25], [48, 25, 19, 19]
        att_maps.append(
            attn[i].unsqueeze(0).contiguous()
        )  # add attention for curr index  [25, 19, 19]
        word = word.transpose(1, 2).contiguous()  # [48, 25, 768]
        weiContext = weiContext.transpose(1, 2).contiguous()  # [48, 25, 768]

        word = word.view(batch_size * words_num, -1)  # [1200, 768]
        weiContext = weiContext.view(batch_size * words_num, -1)  # [1200, 768]

        row_sim = cosine_similarity(word, weiContext)
        row_sim = row_sim.view(batch_size, words_num)  # [48, 25]

        row_sim.mul_(temp2).exp_()
        if agg == "sum":
            row_sim = row_sim.sum(dim=1, keepdim=True)  # [48, 1]
        else:
            row_sim = row_sim.mean(dim=1, keepdim=True)  # [48, 1]
        row_sim = torch.log(row_sim)

        similarities.append(row_sim)

    similarities = torch.cat(similarities, 1)  #
    similarities = similarities * temp3
    # similarities1 = similarities.transpose(0, 1)  # [48, 48]
    # ****************inner
    # similarities = torch.mean(similarities[])
    labels_inner = Variable(torch.LongTensor(range(3))).cuda() # [0,1]
    similarities_ = []
    for n in range(1):
        similarities_.append(torch.mean(similarities[(n * 4):((n + 1) * 4), (n * 4):((n + 1) * 4)]))  # [0:4,0:4]
        similarities_.append(torch.mean(similarities[(n * 4):((n + 1) * 4), ((n + 1) * 4):(n + 2) * 4]))  # [0:4,4:8]
        similarities_.append(torch.mean(similarities[(n * 4):((n + 1) * 4), ((n + 2) * 4):(n + 3) * 4]))  # [0:4,8:12]

        similarities_.append(torch.mean(similarities[((n + 1) * 4):(n + 2) * 4, (n * 4):((n + 1) * 4)]))  # [4:8,0:4]
        similarities_.append(
            torch.mean(similarities[((n + 1) * 4):(n + 2) * 4, ((n + 1) * 4):((n + 2) * 4)]))  # [4:8,4:8]
        similarities_.append(
            torch.mean(similarities[((n + 1) * 4):(n + 2) * 4, ((n + 2) * 4):((n + 3) * 4)]))  # [4:8,8:12]

        similarities_.append(torch.mean(similarities[((n + 2) * 4):(n + 3) * 4, (n * 4):((n + 1) * 4)]))  # [4:8,0:4]
        similarities_.append(
            torch.mean(similarities[((n + 2) * 4):(n + 3) * 4, ((n + 1) * 4):((n + 2) * 4)]))  # [4:8,4:8]
        similarities_.append(
            torch.mean(similarities[((n + 2) * 4):(n + 3) * 4, ((n + 2) * 4):((n + 3) * 4)]))  # [4:8,8:12]
    # labels_inner = torch.stack([itm for itm in labels_inner for i in range(2)])  # [0,0,1,1,2,2,3,3,4,4,5,5]
    similarities_ = torch.stack(similarities_).view(3, 3)
    similarities1_ = similarities_.transpose(0, 1)
    loss_inner = nn.CrossEntropyLoss()(similarities_, labels_inner)  # labels: arange(batch_size)
    loss_inner1 = nn.CrossEntropyLoss()(similarities1_, labels_inner)

    idx_rank, rank =  rank_kekyword(keyword, similarities_)
    label_ranks = []

    for i in range(len(rank) - 1):
        t = i + 1
        for j in range(t, len(rank)):
            for k in range(len(rank[i][0])):
                for w in range(len(rank[j][0])):
                    a = rank[i][0]
                    label_ranks.append(a[k])
                    b = rank[j][0]
                    label_ranks.append(b[w])
            if (t < len(rank)):
                t += 1

    if(len(label_ranks)==0):
        coarse_to_fine_loss = 0
    else:
        a = int(len(label_ranks) / 2)
        label_ranks = torch.stack(label_ranks).view(a, 2)
        label_rank = torch.ones(a,dtype=torch.long).cuda()
        coarse_to_fine_loss = nn.CrossEntropyLoss()(label_ranks, label_rank)

    return loss_inner, loss_inner1, att_maps, coarse_to_fine_loss

def rank_kekyword(keyword, similarities):
    nodumplicate_Keyword =keyword
    # for i in range(int(len(keyword))):
    #     for key in keyword[i]:
    #         if key not in nodumplicate_Keyword[i]:
    #             nodumplicate_Keyword[i].append(key)
    rank = []
    for i in range(int(len(nodumplicate_Keyword))):
        for j in range(int(len(nodumplicate_Keyword))):
            if (i != j) and (i < j):
               ranktmp=0
               # aa = nodumplicate_Keyword[i].split(' ')
               for key in nodumplicate_Keyword[i].split(' '):
                   if key in nodumplicate_Keyword[j].split(' ') and key!='[CLS]' and key!='[SEP]' and key!='[PAD]' and key!='':
                       ranktmp +=1
               rank.append(ranktmp)
    intra = []
    for i in range(similarities.size(0)):
        for j in range(similarities.size(1)):
            if (i != j) and (i < j):
                intra.append((similarities[i, j] + similarities[j, i]) / 2)
    inta_ranked =[]
    rankindex = np.argsort(rank)
    for i in range(len(intra)):
        inta_ranked.append(intra[rankindex[i]])

    k = 0
    s1 = list(inta_ranked)
    for j in range(len(rankindex) - 1):
        if rank[rankindex[j]] != rank[rankindex[j + 1]]:
            k += 1
            s1.insert(j + k, '#')
    j=0
    s2=[]
    for i in range(len(s1)):
        if (i == len(s1) - 1):
            s2.append([s1[j:(i+1)]])
        if str(s1[i]) =='#':
            s2.append([s1[j:(i)]])
            j=i+1
    return rankindex, s2