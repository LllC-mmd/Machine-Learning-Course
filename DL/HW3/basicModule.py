import torch
import torch.nn as nn
import torch.nn.utils.rnn as rnn_utils


class LSTMOps(torch.autograd.Function):

    @staticmethod
    def forward(ctx, inputs, W_ix, W_ih, b_i, W_fx, W_fh, b_f, W_ox, W_oh, b_o, W_hx, W_hh, b_h, h_prev, c_prev):
        # inputs: [seq_len=1, batch_size, embedding_size]
        it = torch.sigmoid(torch.matmul(inputs, W_ix) + torch.matmul(h_prev, W_ih) + b_i)
        ft = torch.sigmoid(torch.matmul(inputs, W_fx) + torch.matmul(h_prev, W_fh) + b_f)
        ot = torch.sigmoid(torch.matmul(inputs, W_ox) + torch.matmul(h_prev, W_oh) + b_o)
        c_tilde = torch.tanh(torch.matmul(inputs, W_hx) + torch.matmul(h_prev, W_hh) + b_h)
        ct = ft * c_prev + it * c_tilde
        ht = ot * torch.tanh(ct)

        ctx.save_for_backward(inputs, W_ih, W_ix, W_fh, W_fx, W_oh, W_ox, W_hh, W_hx, it, ft, ot, c_tilde, h_prev, c_prev)
        return ht, ct

    @staticmethod
    def backward(ctx, grad_h, grad_c):
        inputs, W_ih, W_ix, W_fh, W_fx, W_oh, W_ox, W_hh, W_hx, it, ft, ot, c_tilde, h_prev, c_prev = ctx.saved_tensors

        ct_active = torch.tanh(ft * c_prev + it * c_tilde)
        grad_c = grad_c + ot * (1.0 - ct_active * ct_active) * grad_h
        grad_ot = grad_h * ct_active
        grad_c_tilde = it * grad_c
        grad_it = grad_c * c_tilde
        grad_ft = grad_c * c_prev

        # inputs: [seq_len=1, batch_size, input_size]
        # it, grad_it, ht, h_prev, ct, c_prev: [1, batch_size, hidden_size]
        di = (1.0 - it) * it
        grad_W_ix = torch.sum(torch.matmul(inputs.unsqueeze(3), torch.unsqueeze(di*grad_it, 2)), dim=1).squeeze()
        grad_W_ih = torch.sum(torch.matmul(h_prev.unsqueeze(3), torch.unsqueeze(di*grad_it, 2)), dim=1).squeeze()
        grad_b_i = torch.sum(di * grad_it, dim=1)
        df = (1.0 - ft) * ft
        grad_W_fx = torch.sum(torch.matmul(inputs.unsqueeze(3), torch.unsqueeze(df*grad_ft, 2)), dim=1).squeeze()
        grad_W_fh = torch.sum(torch.matmul(h_prev.unsqueeze(3), torch.unsqueeze(df*grad_ft, 2)), dim=1).squeeze()
        grad_b_f = torch.sum(df * grad_ft, dim=1)
        do = (1.0 - ot) * ot
        grad_W_ox = torch.sum(torch.matmul(inputs.unsqueeze(3), torch.unsqueeze(do*grad_ot, 2)), dim=1).squeeze()
        grad_W_oh = torch.sum(torch.matmul(h_prev.unsqueeze(3), torch.unsqueeze(do*grad_ot, 2)), dim=1).squeeze()
        grad_b_o = torch.sum(do * grad_ot, dim=1)
        dc_tilde = (1.0 - c_tilde * c_tilde)
        grad_W_hx = torch.sum(torch.matmul(inputs.unsqueeze(3), torch.unsqueeze(dc_tilde*grad_c_tilde, 2)), dim=1).squeeze()
        grad_W_hh = torch.sum(torch.matmul(h_prev.unsqueeze(3), torch.unsqueeze(dc_tilde*grad_c_tilde, 2)), dim=1).squeeze()
        grad_b_h = torch.sum(dc_tilde * grad_c_tilde, dim=1).squeeze()

        grad_h_prev = torch.matmul(di*grad_it, W_ih.t()) + torch.matmul(df*grad_ft, W_fh.t()) \
                      + torch.matmul(do*grad_ot, W_oh.t()) + torch.matmul(dc_tilde*grad_c_tilde, W_hh.t())

        grad_c_prev = grad_c * ft

        grad_inputs = torch.matmul(di*grad_it, W_ix.t()) + torch.matmul(df*grad_ft, W_fx.t()) \
                      + torch.matmul(do*grad_ot, W_ox.t()) + torch.matmul(dc_tilde*grad_c_tilde, W_hx.t())

        return grad_inputs, grad_W_ix, grad_W_ih, grad_b_i, grad_W_fx, grad_W_fh, grad_b_f, grad_W_ox, grad_W_oh, grad_b_o, \
               grad_W_hx, grad_W_hh, grad_b_h, grad_h_prev, grad_c_prev


class LSTMBlock(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(LSTMBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # input gate
        self.W_ix = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_ih = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        # forget gate
        self.W_fx = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_fh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        # output gate
        self.W_ox = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_oh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        # memory cell
        self.W_hx = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hh = nn.Parameter(torch.Tensor(hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(hidden_size))

        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, inputs, h_prev=None, c_prev=None):
        # input: [1, batch_size, input_size]
        batch_size = inputs.size(1)
        if h_prev is None:
            h_prev = torch.zeros(1, batch_size, self.hidden_size, dtype=inputs.dtype, device=inputs.device)
        if c_prev is None:
            c_prev = torch.zeros(1, batch_size, self.hidden_size, dtype=inputs.dtype, device=inputs.device)
        h_new, c_new = LSTMOps.apply(inputs, self.W_ix, self.W_ih, self.b_i, self.W_fx, self.W_fh, self.b_f,
                                self.W_ox, self.W_oh, self.b_o, self.W_hx, self.W_hh, self.b_h, h_prev, c_prev)
        return h_new, c_new


class AttLSTMBlock(nn.Module):

    def __init__(self, input_size, hidden_size):
        super(AttLSTMBlock, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        # input gate
        self.W_ix = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_ih = nn.Parameter(torch.Tensor(2*hidden_size, hidden_size))
        self.b_i = nn.Parameter(torch.Tensor(hidden_size))
        # forget gate
        self.W_fx = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_fh = nn.Parameter(torch.Tensor(2*hidden_size, hidden_size))
        self.b_f = nn.Parameter(torch.Tensor(hidden_size))
        # output gate
        self.W_ox = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_oh = nn.Parameter(torch.Tensor(2*hidden_size, hidden_size))
        self.b_o = nn.Parameter(torch.Tensor(hidden_size))
        # memory cell
        self.W_hx = nn.Parameter(torch.Tensor(input_size, hidden_size))
        self.W_hh = nn.Parameter(torch.Tensor(2*hidden_size, hidden_size))
        self.b_h = nn.Parameter(torch.Tensor(hidden_size))

        for p in self.parameters():
            if p.data.ndimension() >= 2:
                nn.init.xavier_uniform_(p.data)
            else:
                nn.init.zeros_(p.data)

    def forward(self, inputs, h_prev=None, c_prev=None):
        # input: [1, batch_size, input_size]
        batch_size = inputs.size(1)
        if h_prev is None:
            h_prev = torch.zeros(1, batch_size, self.hidden_size, dtype=inputs.dtype, device=inputs.device)
        if c_prev is None:
            c_prev = torch.zeros(1, batch_size, self.hidden_size, dtype=inputs.dtype, device=inputs.device)
        h_new, c_new = LSTMOps.apply(inputs, self.W_ix, self.W_ih, self.b_i, self.W_fx, self.W_fh, self.b_f,
                                self.W_ox, self.W_oh, self.b_o, self.W_hx, self.W_hh, self.b_h, h_prev, c_prev)
        return h_new, c_new
'''
# Gradient Check for custom LSTMOps
nhid = 3
bcsz = 10
nid = 2

input_test = torch.randn(1, bcsz, nid, dtype=torch.double, requires_grad=True)
h_test = torch.randn(1, bcsz, nhid, dtype=torch.double, requires_grad=True)
c_test = torch.randn(1, bcsz, nhid, dtype=torch.double, requires_grad=True)

W_ix = torch.randn(nid, nhid, dtype=torch.double, requires_grad=True)
W_ih = torch.randn(nhid, nhid, dtype=torch.double, requires_grad=True)
b_i = torch.randn(nhid, dtype=torch.double, requires_grad=True)

W_fx = torch.randn(nid, nhid, dtype=torch.double, requires_grad=True)
W_fh = torch.randn(nhid, nhid, dtype=torch.double, requires_grad=True)
b_f = torch.randn(nhid, dtype=torch.double, requires_grad=True)

W_ox = torch.randn(nid, nhid, dtype=torch.double, requires_grad=True)
W_oh = torch.randn(nhid, nhid, dtype=torch.double, requires_grad=True)
b_o = torch.randn(nhid, dtype=torch.double, requires_grad=True)

W_hx = torch.randn(nid, nhid, dtype=torch.double, requires_grad=True)
W_hh = torch.randn(nhid, nhid, dtype=torch.double, requires_grad=True)
b_h = torch.randn(nhid, dtype=torch.double, requires_grad=True)

inputs = (input_test, W_ix, W_ih, b_i, W_fx, W_fh, b_f, W_ox, W_oh, b_o, W_hx, W_hh, b_h, h_test, c_test)
lstm_ops = LSTMOps.apply
test = torch.autograd.gradcheck(lstm_ops, inputs, eps=1e-4, atol=1e-5)
print(test)
'''

