import torch.nn as nn
from transformers import BertPreTrainedModel, BertModel, ElectraModel
from torch.nn.utils.rnn import pad_sequence
from torchcrf import CRF
from transformers.modeling_outputs import TokenClassifierOutput


class BertNER(nn.Module):
    def __init__(self):
        super(BertNER, self).__init__()
        self.num_labels = 22
        self.hidden_size = 768
        self.robert = BertModel.from_pretrained("hfl/chinese-bert-wwm-ext")
        self.dropout = nn.Dropout(0.1)
        self.bilstm = nn.LSTM(
            input_size=self.hidden_size,  # 1024
            hidden_size=self.hidden_size // 2,  # 1024
            batch_first=True,
            num_layers=2,
            dropout=0.1,  # 0.5
            bidirectional=True
        )
        self.classifier = nn.Linear(self.hidden_size, self.num_labels)

    def forward(self, input_ids, token_type_ids=None, attention_mask=None, labels=None,
                position_ids=None, inputs_embeds=None, head_mask=None):
        outputs = self.robert(input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              position_ids=position_ids,
                              head_mask=head_mask,
                              inputs_embeds=inputs_embeds)
        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)
        #         print(sequence_output.shape)

        lstm_output, _ = self.bilstm(sequence_output)

        # 得到判别值
        logits = self.classifier(lstm_output)
        return TokenClassifierOutput(
            logits=logits,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
