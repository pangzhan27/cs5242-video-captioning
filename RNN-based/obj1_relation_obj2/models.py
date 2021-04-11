import torch.nn as nn
import torchvision.models as models
import copy
import torch.nn.functional as F
import torch
from torch.nn import Parameter
import math
from copy import deepcopy
import pickle
from queue import Queue
import numpy as np
from torch.autograd import Variable

class FocalLoss(nn.Module):
    def __init__(self, class_num, alpha=None, gamma=2, size_average=True):
        super(FocalLoss, self).__init__()
        if alpha is None:
            self.alpha = Variable(torch.ones(class_num, 1))
        else:
            if isinstance(alpha, Variable):
                self.alpha = alpha
            else:
                self.alpha = Variable(alpha)
        self.gamma = gamma
        self.class_num = class_num
        self.size_average = size_average

    def forward(self, inputs, targets):
        N = inputs.size(0)
        C = inputs.size(1)
        P = torch.exp(inputs)

        class_mask = inputs.data.new(N, C).fill_(0)
        class_mask = Variable(class_mask)
        ids = targets.view(-1, 1)
        class_mask.scatter_(1, ids.data, 1.)

        if inputs.is_cuda and not self.alpha.is_cuda:
            self.alpha = self.alpha.cuda()
        alpha = self.alpha[ids.data.view(-1)]

        probs = (P*class_mask).sum(1).view(-1,1)

        log_p = probs.log()
        batch_loss = -alpha*(torch.pow((1-probs), self.gamma))*log_p

        if self.size_average:
            loss = batch_loss.mean()
        else:
            loss = batch_loss.sum()
        return loss



class BeamSearchNode(object):
    def __init__(self, hidden, previousNode, id, logprob, addedprob, t):
        self.hid = hidden
        self.pre = previousNode
        self.id = id
        self.logp = logprob
        self.addp = addedprob
        self.t = t


class Attention(nn.Module):
    """
    Applies an attention mechanism on the output features from the decoder.
    """

    def __init__(self, dim):
        super(Attention, self).__init__()
        self.dim = dim
        self.linear1 = nn.Linear(dim * 2, dim)
        self.linear2 = nn.Linear(dim, 1, bias=False)
        #self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.linear1.weight)
        nn.init.xavier_normal_(self.linear2.weight)

    def forward(self, hidden_state, encoder_outputs):
        """
        Arguments:
            hidden_state {Variable} -- batch_size x dim
            encoder_outputs {Variable} -- batch_size x seq_len x dim
        Returns:
            Variable -- context vector of size batch_size x dim
        """
        batch_size, seq_len, _ = encoder_outputs.size()
        hidden_state = hidden_state.unsqueeze(1).repeat(1, seq_len, 1)
        inputs = torch.cat((encoder_outputs, hidden_state),
                           2).view(-1, self.dim * 2)
        o = self.linear2(F.tanh(self.linear1(inputs)))
        e = o.view(batch_size, seq_len)
        alpha = F.softmax(e, dim=1)
        context = torch.bmm(alpha.unsqueeze(1), encoder_outputs).squeeze(1)
        return context

class DecoderRNN(nn.Module):
    """
    Provides functionality for decoding in a seq2seq framework, with an option for attention.
    Args:
        vocab_size (int): size of the vocabulary
        max_len (int): a maximum allowed length for the sequence to be processed
        dim_hidden (int): the number of features in the hidden state `h`
        n_layers (int, optional): number of recurrent layers (default: 1)
        rnn_cell (str, optional): type of RNN cell (default: gru)
        bidirectional (bool, optional): if the encoder is bidirectional (default False)
        input_dropout_p (float, optional): dropout probability for the input sequence (default: 0)
        rnn_dropout_p (float, optional): dropout probability for the output sequence (default: 0)
    """

    def __init__(self,
                 vocab_size,
                 max_len,
                 dim_hidden,
                 dim_word,
                 n_layers=1,
                 rnn_cell='gru',
                 bidirectional=False,
                 input_dropout_p=0.1,
                 rnn_dropout_p=0.1):
        super(DecoderRNN, self).__init__()

        self.bidirectional_encoder = bidirectional

        self.dim_output0 = vocab_size[0]
        self.dim_output1 = vocab_size[1]
        self.dim_hidden = dim_hidden * 2 if bidirectional else dim_hidden
        self.dim_word = dim_word
        self.max_length = max_len
        self.beam_width = 10
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.input_dropout = nn.Dropout(input_dropout_p)
        self.embedding = nn.Embedding(self.dim_output0+self.dim_output1 + 1 , dim_word)

        with open("embedding.pickle", "rb") as fp:
            pretrained_weight = pickle.load(fp)
        self.embedding.weight.data.copy_(torch.from_numpy(pretrained_weight))
        self.embedding.weight.requires_grad = False

        self.attention = Attention(self.dim_hidden)
        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU
        self.rnn = self.rnn_cell(
            self.dim_hidden + dim_word,
            self.dim_hidden,
            n_layers,
            batch_first=True,
            dropout=rnn_dropout_p)

        self.out0 = nn.Linear(self.dim_hidden, self.dim_output0)
        self.out1 = nn.Linear(self.dim_hidden, self.dim_output1)

        self._init_weights()

    def forward(self,
                encoder_outputs,
                encoder_hidden,
                targets=None,
                mode='train',
                isbeam = False,
                opt={}):
        """
        Inputs: inputs, encoder_hidden, encoder_outputs, function, teacher_forcing_ratio
        - **encoder_hidden** (num_layers * num_directions, batch_size, dim_hidden): tensor containing the features in the
          hidden state `h` of encoder. Used as the initial hidden state of the decoder. (default `None`)
        - **encoder_outputs** (batch, seq_len, dim_hidden * num_directions): (default is `None`).
        - **targets** (batch, max_length): targets labels of the ground truth sentences
        Outputs: seq_probs,
        - **seq_logprobs** (batch_size, max_length, vocab_size): tensors containing the outputs of the decoding function.
        - **seq_preds** (batch_size, max_length): predicted symbols
        """
        batch_size, _, _ = encoder_outputs.size()
        decoder_hidden = self._init_rnn_state(encoder_hidden)

        seq_logprobs = []
        seq_preds = []
        self.rnn.flatten_parameters()

        if not mode and isbeam:
            beam_out = self.beam_decode(decoder_hidden, encoder_outputs)
            return beam_out, _

        for t in range(self.max_length):
            context = self.attention(decoder_hidden.squeeze(0), encoder_outputs)
            if mode:
                if t == 0:  # input <bos>
                    it = torch.LongTensor([0] * batch_size).to(self.device)
                else:
                    it = targets[t-1] + 1 if t ==1 else targets[t-1] + 36
                current_words = self.embedding(it)
                decoder_input = torch.cat([current_words, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                decoder_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)
                if t == 0 or t ==2:
                    logprobs = F.log_softmax(self.out0(decoder_output.squeeze(1)), dim=1)
                else:
                    logprobs = F.log_softmax(self.out1(decoder_output.squeeze(1)), dim=1)
                seq_logprobs.append(logprobs)

            else:
                if t == 0:  # input <bos>
                    it = torch.LongTensor([0] * batch_size).to(self.device)
                if t>0 :
                    it = it+1 if t ==1 else it + 36
                xt = self.embedding(it)
                decoder_input = torch.cat([xt, context], dim=1)
                decoder_input = self.input_dropout(decoder_input).unsqueeze(1)
                decoder_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)
                if t == 0 or t == 2:
                    logprobs = F.log_softmax(self.out0(decoder_output.squeeze(1)), dim=1)
                    seq_preds.append(self.out0(decoder_output.squeeze(1)))
                else:
                    logprobs = F.log_softmax(self.out1(decoder_output.squeeze(1)), dim=1)
                    seq_preds.append(self.out1(decoder_output.squeeze(1)))
                sampleLogprobs, it = torch.max(logprobs, 1)
                seq_logprobs.append(logprobs)
                it = it.view(-1).long()
                #seq_preds.append(it.view(-1, 1))

        return seq_logprobs, seq_preds

    def beam_decode(self, decoder_hiddens, encoder_outputs):
        batch_size = encoder_outputs.shape[0]
        collect_batch = []
        for idx in range(batch_size):
            it = torch.LongTensor([0]).to(self.device)
            node = BeamSearchNode(decoder_hiddens[:, idx, :].unsqueeze(1), None, it, 1, 1, 0)
            encoder_output = encoder_outputs[idx, :, :].unsqueeze(0)
            nodes = Queue()
            all_nodes = []
            nodes.put(node)
            all_nodes.append(node)

            while True:
                n = nodes.get()
                it = deepcopy(n.id)
                decoder_hidden = n.hid

                if n.t == 1:
                    it += 1
                if n.t == 2:
                    it += 36
                if n.t == 3:
                    if nodes.empty():
                       break
                    else:
                       continue

                xt = self.embedding(it).view(1,-1)
                context = self.attention(decoder_hidden.squeeze(0), encoder_output)
                decoder_input = torch.cat([xt, context], dim=1).unsqueeze(1)
                decoder_output, decoder_hidden = self.rnn(decoder_input, decoder_hidden)
                if n.t == 0 or n.t == 2:
                    logprobs = F.log_softmax(self.out0(decoder_output.squeeze(1)), dim=1)
                else:
                    logprobs = F.log_softmax(self.out1(decoder_output.squeeze(1)), dim=1)

                lprob, indexes = torch.topk(logprobs, self.beam_width)

                for newk in range(self.beam_width):
                    decode_t = indexes[0][newk]
                    log_p = lprob[0][newk]
                    new_node = BeamSearchNode(decoder_hidden, n, decode_t, torch.exp(log_p), n.addp + log_p, n.t + 1)
                    nodes.put(new_node)
                    all_nodes.append(new_node)

            endnodes = []
            for node in all_nodes:
                if node.t == 3:
                    endnodes.append(node)
            single_output = {}
            for node in endnodes:
                path_score = node.addp
                path, path_p = [], []
                while True:
                    path.append(node.id.cpu().detach().item())
                    path_p.append(node.logp.cpu().detach().item())
                    node = node.pre
                    if node.pre is None:
                        break
                path, path_p = path[::-1], path_p[::-1]
                name = '_'.join([str(kk) for kk in path])
                single_output[name] = (path_score, path, path_p)

            collect_batch.append(single_output)
        final_output = []
        all_names = set(collect_batch[0].keys()) | set(collect_batch[1].keys()) | set(collect_batch[2].keys())
        for name in all_names:
            final_scores, final_paths, final_path_ps = [], [], []
            for i in range(3):
                if name in collect_batch[i].keys():
                    final_scores.append(collect_batch[i][name][0].item())
                    final_paths.append(collect_batch[i][name][1])
                    final_path_ps.append(collect_batch[i][name][2])
            final_score = np.mean(np.array(final_scores))
            final_path = final_paths[0]
            final_path_p = np.mean(np.array(final_path_ps),axis=0)
            final_output.append([final_score, final_path, list(final_path_p)])

        final_output = sorted(final_output, key= lambda x: x[0], reverse=True)
        return final_output

    def _init_weights(self):
        """ init the weight of some layers
        """
        nn.init.xavier_normal_(self.out0.weight)
        nn.init.xavier_normal_(self.out1.weight)

    def _init_rnn_state(self, encoder_hidden):
        """ Initialize the encoder hidden state. """
        if encoder_hidden is None:
            return None
        if isinstance(encoder_hidden, tuple):
            encoder_hidden = tuple(
                [self._cat_directions(h) for h in encoder_hidden])
        else:
            encoder_hidden = self._cat_directions(encoder_hidden)
        return encoder_hidden

    def _cat_directions(self, h):
        """ If the encoder is bidirectional, do the following transformation.
            (#directions * #layers, #batch, dim_hidden) -> (#layers, #batch, #directions * dim_hidden)
        """
        if self.bidirectional_encoder:
            h = torch.cat([h[0:h.size(0):2], h[1:h.size(0):2]], 2)
        return h

class EncoderRNN(nn.Module):
    def __init__(self, dim_vid, dim_hidden, input_dropout_p=0.2, rnn_dropout_p=0.5,
                 n_layers=1, bidirectional=False, rnn_cell='gru'):
        """
        Args:
            hidden_dim (int): dim of hidden state of rnn
            input_dropout_p (int): dropout probability for the input sequence
            dropout_p (float): dropout probability for the output sequence
            n_layers (int): number of rnn layers
            rnn_cell (str): type of RNN cell ('LSTM'/'GRU')
        """
        super(EncoderRNN, self).__init__()
        self.dim_vid = dim_vid
        self.dim_hidden = dim_hidden
        self.input_dropout_p = input_dropout_p
        self.rnn_dropout_p = rnn_dropout_p
        self.n_layers = n_layers
        self.bidirectional = bidirectional
        self.rnn_cell = rnn_cell

        self.vid2hid = nn.Linear(dim_vid, dim_hidden)
        self.input_dropout = nn.Dropout(input_dropout_p)

        if rnn_cell.lower() == 'lstm':
            self.rnn_cell = nn.LSTM
        elif rnn_cell.lower() == 'gru':
            self.rnn_cell = nn.GRU

        self.rnn = self.rnn_cell(dim_hidden, dim_hidden, n_layers, batch_first=True,
                                bidirectional=bidirectional, dropout=self.rnn_dropout_p)

        self._init_hidden()

    def _init_hidden(self):
        nn.init.xavier_normal_(self.vid2hid.weight)

    def forward(self, vid_feats):
        """
        Applies a multi-layer RNN to an input sequence.
        Args:
            input_var (batch, seq_len): tensor containing the features of the input sequence.
            input_lengths (list of int, optional): A list that contains the lengths of sequences
              in the mini-batch
        Returns: output, hidden
            - **output** (batch, seq_len, hidden_size): variable containing the encoded features of the input sequence
            - **hidden** (num_layers * num_directions, batch, hidden_size): variable containing the features in the hidden state h
        """
        batch_size, seq_len, dim_vid = vid_feats.size()
        vid_feats = self.vid2hid(vid_feats.view(-1, dim_vid))
        vid_feats = self.input_dropout(vid_feats)
        vid_feats = vid_feats.view(batch_size, seq_len, self.dim_hidden)
        self.rnn.flatten_parameters()
        output, hidden = self.rnn(vid_feats)
        return output, hidden

class Resnet_LSTM(nn.Module):
    def __init__(self, num_classes,  dim, embedding=300):
        super(Resnet_LSTM, self).__init__()
        self.num_obj = num_classes[0]
        self.num_rel = num_classes[1]
        ###### 1.resnet101 ######
        model = models.resnet101(pretrained=True)
        for name, param in model.named_parameters():
            param.requires_grad = False
            if ('layer3' in name) or ('layer4' in name):
                param.requires_grad = True

        self.features_3 = nn.Sequential(
            model.conv1,
            model.bn1,
            model.relu,
            model.maxpool,
            model.layer1,
            model.layer2,
            model.layer3,
        )
        self.features_4 = model.layer4
        self.conv34 = nn.Conv2d(1024, 2048, 3, padding=1)
        self.pooling = nn.MaxPool2d(14, 14)

        self.encoder = EncoderRNN( dim, 512, bidirectional=True, input_dropout_p=0.2, rnn_cell='gru', rnn_dropout_p=0.5)
        self.decoder = DecoderRNN((self.num_obj,self.num_rel), 3,512, 300, input_dropout_p=0.2,
                                  rnn_cell='gru', rnn_dropout_p=0.5,  bidirectional=True)

    def forward(self, feature, target, istrain=True, isbeam = False):
        feature3 = self.features_3(feature)
        feature4 = self.features_4(feature3)
        feature4 = F.interpolate(feature4, scale_factor=2, mode='bilinear')
        feature34 = self.conv34(feature3)
        feature = feature4 + feature34
        feature = self.pooling(feature)
        feature = feature.view(feature.size(0), -1)
        split_feature = feature.split(10, dim=0)
        new_feature = torch.cat([i.unsqueeze(0) for i in split_feature])

        encoder_outputs, encoder_hidden = self.encoder(new_feature)
        seq_prob, seq_preds = self.decoder(encoder_outputs, encoder_hidden, target, istrain, isbeam)
        return seq_prob, seq_preds


    def get_config_optim(self, lr, lrp):
        return [
                {'params': self.features_3.parameters(), 'lr': lr * lrp},
                {'params': self.features_4.parameters(), 'lr': lr * lrp},
                {'params': self.conv34.parameters(), 'lr': lr},
                {'params': self.decoder.parameters(), 'lr': lr},
                {'params': self.encoder.parameters(), 'lr': lr},
                ]
