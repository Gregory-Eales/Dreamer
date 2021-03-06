import torch
from torch.utils.data import IterableDataset, Dataset


class ReplayBuffer(Dataset):

    def __init__(self, hparams):

        super(ReplayBuffer, self).__init__()

        self.hparams = hparams

        self.obs_dim = obs_dim
        self.act_dim = act_dim

        self.s = []
        self.a = []
        self.r = []
        self.n_s = []
        self.t = []

    def store(self, s, a, r, n_s, t):
        self.s.append(s)
        self.a.append(a)
        self.r.append(r)
        self.n_s.append(n_s)
        if t:self.t.append(1)
        else: self.t.append(0)

    def __len__(self):
        return len(self.s)

    def __getitem__(self, idx):

        s = torch.Tensor(self.s).reshape(-1, self.obs_dim)[idx]
        a = torch.Tensor(self.a).reshape(-1, self.act_dim)[idx]
        r = torch.Tensor(self.r).reshape(-1, 1)[idx]
        n_s = torch.Tensor(self.n_s).reshape(-1, self.obs_dim)[idx]
        t = torch.Tensor(self.t).reshape(-1, 1)[idx]

        return s, a, r, n_s, t

    def __iter__(self):

        s = torch.Tensor(self.s).reshape(-1, self.obs_dim)
        a = torch.Tensor(self.a).reshape(-1, self.act_dim)
        r = torch.Tensor(self.r).reshape(-1, 1)
        n_s = torch.Tensor(self.n_s).reshape(-1, self.obs_dim)
        t = torch.Tensor(self.t).reshape(-1, 1)

        """
        for i in range(s.shape[0]):
            yield s[i], a[i], r[i], n_s[i], t[i]
        """
        
        num_batch = s.shape[0]//self.hparams.batch_size
        rem_batch = s.shape[0]%self.hparams.batch_size
        
        for i in range(num_batch):
            i1, i2 = i*16, (i+1)*self.hparams.batch_size
        
            yield s[i1:i2], a[i1:i2], r[i1:i2], n_s[i1:i2], t[i1:i2]
        
        
        i1, i2 = -rem_batch, 0
        yield s[i1:i2], a[i1:i2], r[i1:i2], n_s[i1:i2], t[i1:i2]