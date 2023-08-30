from torch.utils.data import Dataset, DataLoader
import json
from transformers import BertTokenizerFast,RobertaTokenizerFast,ElectraTokenizerFast
from torch.nn.utils.rnn import pad_sequence
import torch
import torch.nn as nn
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class NERDataset(Dataset):
    def __init__(self,path):
        super(NERDataset, self).__init__()
        self.path = path
        self.tokenizer = BertTokenizerFast.from_pretrained("hfl/chinese-bert-wwm-ext")

        self.label2id = {
            "O": 0,
            "B-BODY": 1,
            "B-SYMP": 2,
            "B-INST": 3,
            'B-EXAM': 4,
            'B-CHEM': 5,
            'B-DISE': 6,
            'B-DRUG': 7,
            'B-SUPP': 8,
            'B-TREAT': 9,
            'B-TIME': 10,
            "I-BODY": 11,
            "I-SYMP": 12,
            "I-INST": 13,
            'I-EXAM': 14,
            'I-CHEM': 15,
            'I-DISE': 16,
            'I-DRUG': 17,
            'I-SUPP': 18,
            'I-TREAT': 19,
            'I-TIME': 20
        }
        # print(self.tokenizer.pad_token_id)
        self.characters,self.characters_labels = self.preprocessing()
        self.label_pad = 21
        self.id2label = {value:key for key,value in self.label2id.items()}

    def preprocessing(self):
        words = []
        words_labels = []
        characters = []
        characters_labels = []
        with open(self.path, "r", encoding='utf-8') as f:
            content = f.read()
            # print(type(eval(content)))
            for line in eval(content):
                character = line['characters']
                character_label = line['labels']

                characters.append(self.tokenizer.convert_tokens_to_ids(character))
                temp = []
                for label in character_label:
                    temp.append(self.label2id[label])
                characters_labels.append(temp)
            return characters,characters_labels


    def __len__(self):
        assert len(self.characters) == len(self.characters_labels)
        return len(self.characters)


    def __getitem__(self, item):
        return torch.tensor(self.characters[item]),torch.tensor(self.characters_labels[item])


    def collate_fn(self,batch):
        characters = []
        labels = []
        for character,label in batch:
            characters.append(character)
            labels.append(label)
        characters = pad_sequence(characters,padding_value=self.tokenizer.pad_token_id).transpose(0,1)
        labels = pad_sequence(labels,padding_value=self.label_pad).transpose(0,1)
        attention_mask = []
        for character in characters:
            temp = [1 for _ in range(len(character))]
            for index,value in enumerate(character):
                if value == 0:
                    temp[index] = 0
            attention_mask.append(temp)
        return characters,labels,torch.tensor(attention_mask)


