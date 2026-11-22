from data.utils import add_noise_norm, plot_u, plot_states_BCS
import numpy as np
import pandas as pd
import torch
from data.parameters import Parameters


def normalizar_x(x,xc,x0):
    xn=[(x[:,i]-x0[i])/xc[i] for i in range(3)]
    return np.array(xn).T

def split_data(n_in,n_out,data):
    a,b,c=split_sequences(data, n_in, n_out)
    x=a[:,:,:]
    y=b[:,:,-3:]
    u_train=b[:,:,0:4] #catch 0 1 and 2 values
    return x,y,u_train

# split a multivariate sequence into samples
def split_sequences(sequences, n_steps_in, n_steps_out):
    #https://machinelearningmastery.com/how-to-develop-lstm-models-for-multi-step-time-series-forecasting-of-household-power-consumption/
    X, y, u = list(), list(), list()
    for i in range(len(sequences)):
        # find the end of this pattern
        end_ix = i + n_steps_in
        out_end_ix = end_ix + n_steps_out-1
        # check if we are beyond the dataset
        if out_end_ix > len(sequences)-1:
            break
        # gather input and output parts of the pattern
        seq_x, seq_y, seq_u = sequences[i:end_ix, :], sequences[end_ix:out_end_ix+1, :], sequences[end_ix-1:out_end_ix, :]
        X.append(seq_x)
        y.append(seq_y)
        u.append(seq_u)
    return np.array(X), np.array(y), np.array(u)


def _as_f32(value):
    if isinstance(value, torch.Tensor):
        return value.detach().to(dtype=torch.float32).contiguous()
    array = np.ascontiguousarray(value, dtype=np.float32)
    return torch.from_numpy(array)


class OrderedBatchLoader(object):
    """Unshuffled batches. Order is part of the physics loss."""

    def __init__(self, X, y, u, batch_size):
        self.X = X
        self.y = y
        self.u = u
        self.batch_size = int(batch_size)

    def __iter__(self):
        n = int(self.X.shape[0])
        step = self.batch_size
        for start in range(0, n, step):
            stop = min(start + step, n)
            yield (
                _as_f32(self.X[start:stop]),
                _as_f32(self.y[start:stop]),
                _as_f32(self.u[start:stop]),
            )

    def __len__(self):
        n = int(self.X.shape[0])
        return int(np.ceil(n / float(self.batch_size)))


class BuildingDataset(object):
    def __init__(self, n_steps_in, n_steps_out, dados, split_point, batch_size, seed=0):
        self.seed = int(seed)
        np.random.seed(self.seed)
        torch.manual_seed(self.seed)
        self.n_steps_in, self.n_steps_out, self.batch_size, self.parameters = n_steps_in, n_steps_out, batch_size, Parameters()
        self.split_point = split_point
        self.tempo = dados['t']
        self.train_X, self.train_y, self.train_y_full, self.u_train = None, None, None, None
        self.pack_plot = None
        self.dataset_full = self.dataset(dados)
        self.dataset_full_noisy = self.dataset_noisy()
        self.x, self.u, self.un, self.xn = None, None, None, None
        self.train_dataset = None
        self.train = None
        self.pack = None
        self.test_y = None
        self.test_X = None
        self.prepare()

    def dataset(self, dados):
        def reshape_data(dataset, length):
            dataset_new = []
            for i in dataset:
                dataset_new.append(np.asarray(i).reshape([length, 1]))
            return dataset_new
        fk = dados['U'][:, 0:1]
        zc = dados['U'][:, 1:2]
        x1 = dados['x1']
        x2 = dados['x2']
        x3 = dados['x3']
        pmc = dados['U'][:, 2:3]
        pr = dados['U'][:, 3:4]
        tempo = dados['t']
        maxtime = fk.shape[0]
        nsim = maxtime
        dataset_full = [x1, x2, x3, fk, zc, pmc, pr, tempo]
        dataset_full = reshape_data(dataset_full, nsim)
        return dataset_full

    def dataset_noisy(self):
        # Multiplicative noise, in this order, on x1, x2, x3, f, zc, pm, pr.
        sigma = [0.01, 0.01, 0.01, 0.005, 0.001, 0.01, 0.01]
        dataset_full_noisy = []
        for i, d in enumerate(self.dataset_full[0:-1]):
            dataset_full_noisy.append(add_noise_norm(d, sigma[i]))
        dataset_full_noisy.append(self.tempo)
        return dataset_full_noisy

    def gen_fig(self):
        Figs = {}
        uplot = self.dataset_full[3:-1]
        Fig_u = plot_u(uplot)
        Fig_x = plot_states_BCS(self.x, self.tempo)
        uplot = [self.un[:, i] for i in range(4)]
        Fig_un = plot_u(uplot)
        Fig_xn = plot_states_BCS(self.xn, self.tempo, norm=True)
        Figs["un"] = Fig_un
        Figs["xn"] = Fig_xn
        Figs["u"] = Fig_u
        Figs["x"] = Fig_x
        return Figs

    def prepare(self):
        self.x = np.hstack(self.dataset_full_noisy[0:3])
        self.u = np.hstack(self.dataset_full_noisy[3:7])
        self.xn = self.parameters.normalizar_x(self.x)
        self.un = self.parameters.normalizar_u(self.u)

        df = pd.DataFrame(np.hstack([self.un, self.xn]), columns=['fn', 'zn', 'pmn', 'prn', 'pbh', 'pwh', 'q'])
        df_u = pd.DataFrame(np.hstack([self.u]), columns=['f', 'z', 'pm', 'pr'])
        dset = df.values.astype(float)
        du_set = df_u.values.astype(float)
        X, y, u_train = split_data(self.n_steps_in, self.n_steps_out, dset)

        train_X_full, train_y_full, u_train = X[:self.split_point, :], y[:self.split_point, :], u_train[:self.split_point, :]
        test_X_full, test_y_full = X[self.split_point:, :], y[self.split_point:, :]
        # Remove unmeasured variable q from the network input.
        train_y = train_y_full[:, :, 0:2]
        train_X = train_X_full[:, :, :-1]
        self.test_y = np.ascontiguousarray(test_y_full[:, :, 0:3], dtype=np.float32)
        self.test_X = np.ascontiguousarray(test_X_full[:, :, :-1], dtype=np.float32)
        self.train_X = np.ascontiguousarray(train_X, dtype=np.float32)
        self.train_y = np.ascontiguousarray(train_y, dtype=np.float32)
        self.train_y_full = np.ascontiguousarray(train_y_full, dtype=np.float32)
        self.u_train = np.ascontiguousarray(u_train, dtype=np.float32)
        self.pack_plot = [self.train_X, self.train_y, self.test_X, self.test_y]
        self.pack = (
            np.ascontiguousarray(y, dtype=np.float32),
            self.train_y_full,
            self.u_train,
        )
        # du_set is kept so the column build stays aligned with the original script.
        self.du_set = du_set

    def adam_dataset(self):
        loader = OrderedBatchLoader(self.train_X, self.train_y, self.u_train, self.batch_size)
        test_X = torch.from_numpy(np.ascontiguousarray(self.test_X, dtype=np.float32))
        test_y = torch.from_numpy(np.ascontiguousarray(self.test_y, dtype=np.float32))
        return [loader, test_X, test_y]

    def lbfgs_dataset(self):
        return [
            torch.from_numpy(np.ascontiguousarray(self.train_X, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(self.train_y, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(self.u_train, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(self.test_X, dtype=np.float32)),
            torch.from_numpy(np.ascontiguousarray(self.test_y, dtype=np.float32)),
        ]


def load_dataset(path):
    import pickle
    try:
        with open(path, "rb") as open_file:
            ds = pickle.load(open_file)
    except (ModuleNotFoundError, ImportError, pickle.UnpicklingError) as exc:
        raise RuntimeError(
            f"Could not load {path}. Rebuild it with main_build_dataset.py ({type(exc).__name__})."
        ) from exc
    train_X = getattr(ds, "train_X", None)
    if not isinstance(train_X, np.ndarray):
        raise RuntimeError(
            f"Could not load {path}. Rebuild it with main_build_dataset.py. Expected NumPy arrays."
        )
    return ds
