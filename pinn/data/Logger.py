import time
import numpy as np
import torch


def _device_name():
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


class Logger(object):
    def __init__(self, frequency=10):
        print("PyTorch version: {}".format(torch.__version__))
        print("Device: {}".format(_device_name()))

        self.start_time = time.time()
        self.frequency = frequency

    def __get_elapsed(self):
        elap = int(max(0, time.time() - self.start_time))
        return f"{elap // 60:2d}:{elap % 60:2d}"

    def __get_error_u(self):
        err = self.error_fn()
        if isinstance(err, torch.Tensor):
            return float(err.detach().cpu())
        return float(err)

    def set_error_fn(self, error_fn):
        self.error_fn = error_fn

    def log_train_start(self, model):
        print("\nTraining started")
        print("================")
        self.model = model
        print(f"PINN Mode:{self.model.pinn_mode}")
        lr = model.optimizer.param_groups[0]["lr"]
        print(f"Learning rate=:{np.array(lr):.4f}")
        print(self.model.summary_model())

    def log_train_epoch(self, epoch, loss, loss_f=0.0, loss_bc=0.0, custom="", is_iter=False):
        if epoch % self.frequency == 0:
            loss = float(loss.detach().cpu()) if isinstance(loss, torch.Tensor) else float(loss)
            loss_f = float(loss_f.detach().cpu()) if isinstance(loss_f, torch.Tensor) else float(loss_f)
            loss_bc = float(loss_bc.detach().cpu()) if isinstance(loss_bc, torch.Tensor) else float(loss_bc)
            print(f"{epoch:6d} {self.__get_elapsed()} {loss:.3e} {loss_bc:.3e} {loss_f:.3e} {self.__get_error_u():.3e}"+custom)

    def log_train_opt(self, name, mode, w):
        print(f"—— Starting {name} optimization —— Pinn_mode:{mode}")
        print(f'==============Weights===============')
        print(f'[ wbc ,  w1 ,  w2 ,  w3 ]')
        print(f'{w}')
        print('                                                    |==== Weigthed Residues ===|ODE parameters|')
        print('==============================================================================================================================')
        print('epoch | elap|  Total  | Loss BC |Loss ODE |  Test   | w1r1   | w2r2   |  w3r3  | rho |   PI   |')
        print('==============================================================================================================================')

    def log_train_end(self, epoch, train_state, custom=""):
        print("==================")
        print(f"Training finished (epoch {epoch}): duration = {self.__get_elapsed()}  Test = {self.__get_error_u():.4e}  " + custom)
