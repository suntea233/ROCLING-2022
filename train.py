from torch import optim
from torchcrf import CRF
from torch.utils.data import DataLoader
import tqdm
from transformers import get_linear_schedule_with_warmup,AutoTokenizer, AutoModelForTokenClassification, BertForTokenClassification,RobertaForTokenClassification
import torch
from sklearn.model_selection import train_test_split
import torch.nn as nn
from torch.autograd import Variable
from dataset import NERDataset
from model import BertNER
from utils import compute_kl_loss

DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

data = NERDataset(r"../input/rocling-dataset/train_dataset.json")
print(len(data))
dataloader = DataLoader(data, batch_size=2, num_workers=0,collate_fn=data.collate_fn)

# optimizer = optim.AdamW(model.parameters(), lr=1e-6,weight_decay = 1e-7)
criterion = nn.CrossEntropyLoss(ignore_index=21)
class FocalLoss(nn.Module):

    def __init__(self,ignore_index, weight=None, reduction='mean', gamma=0, eps=1e-7,):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = eps
        self.ce = torch.nn.CrossEntropyLoss(ignore_index=ignore_index,weight=weight, reduction=reduction)

    def forward(self, input, target):
        logp = self.ce(input, target)
        p = torch.exp(-logp)
        loss = (1 - p) ** self.gamma * logp
        return loss.mean()


focal = FocalLoss(ignore_index=21)



def train(lr,weight_decay,epochs):
    len_dataset = len(data)
    batch_size = 2
    total_steps = (len_dataset // batch_size) * epochs if len_dataset % batch_size == 0 else (len_dataset // batch_size + 1) * epochs

    model = BertNER().to(DEVICE)
    model.train()
    optimizer = optim.AdamW(model.parameters(), lr=lr,weight_decay = weight_decay)
    warm_up_ratio = 0.1
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps = warm_up_ratio * total_steps, num_training_steps = total_steps)
    crf = CRF(model.num_labels,batch_first=True).to(DEVICE)
    print(model.num_labels)
    for epoch in tqdm.tqdm(range(epochs)):
        total = []
        for character,label,attention_mask in dataloader:
            character,label,attention_mask = character.to(DEVICE),label.to(DEVICE),attention_mask.to(DEVICE)

            outputs = model(input_ids=character.contiguous(),attention_mask=attention_mask,labels=label.contiguous())
            logits = outputs.logits.to(DEVICE)
            outputs_2 = model(input_ids=character.contiguous(),attention_mask=attention_mask,labels=label.contiguous())
            logits2 = outputs_2.logits.to(DEVICE)

            ce_loss1 = focal(logits.contiguous().view(-1, model.num_labels), label.contiguous().view(-1))
            ce_loss2 = focal(logits2.contiguous().view(-1, model.num_labels), label.contiguous().view(-1))
            loss = 0.5 * ce_loss1 + 0.5 * ce_loss2
            kl_loss = compute_kl_loss(logits,logits2)

            loss = loss + kl_loss

            loss_mask = label.lt(21).to(DEVICE)
            label = label.to(DEVICE)

            for i in range(len(loss_mask)):
                loss_mask[i][0] = True
            crf_loss = crf(logits.to(DEVICE), label.to(DEVICE), loss_mask.to(DEVICE)) * (-1)


            loss = loss + crf_loss
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            scheduler.step()
            total.append(loss)
        print(sum(total)/len(total))
    return model


