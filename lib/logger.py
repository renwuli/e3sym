from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

# Code referenced from https://gist.github.com/gyglim/1f8dfb1b5c82627ae3efcfbbadb9f514
import os
import time
import sys
import torch

USE_TENSORBOARD = True
try:
    import torch.utils.tensorboard as tensorboard

    print("Using tensorboard")
except:
    USE_TENSORBOARD = False


class Logger(object):
    def __init__(self, opt, cur_file=None):
        """Create a summary writer logging to log_dir."""
        time_str = time.strftime("%Y-%m-%d-%H-%M")
        save_dir = os.path.join(opt.save_dir, opt.name, time_str)
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        self.save_dir = save_dir
        print(f"==> Logs will be saved at {save_dir}")

        args = opt
        file_name = os.path.join(save_dir, "opt.txt")
        with open(file_name, "wt") as opt_file:
            opt_file.write("==> torch version: {}\n".format(torch.__version__))
            opt_file.write(
                "==> cudnn version: {}\n".format(torch.backends.cudnn.version())
            )
            opt_file.write("==> Cmd:\n")
            opt_file.write(str(sys.argv))
            opt_file.write("\n==> Opt:\n")
            for k, v in sorted(args.items()):
                opt_file.write("  %s: %s\n" % (str(k), str(v)))

        log_dir = os.path.join(save_dir, "logs")
        if USE_TENSORBOARD:
            self.writer = tensorboard.SummaryWriter(log_dir=log_dir)
        else:
            if not os.path.exists(os.path.dirname(log_dir)):
                os.mkdir(os.path.dirname(log_dir))
            if not os.path.exists(log_dir):
                os.mkdir(log_dir)
        self.log = open(log_dir + "/log.txt", "w")
        try:
            os.system("cp {}/opt.txt {}/".format(save_dir, log_dir))
        except:
            pass

        try:
            os.system(f"cp *.py {log_dir}/")

            if cur_file is not None:
                os.system(f"cp {cur_file} {log_dir}/")
        except:
            pass

    def write(self, txt):
        time_str = time.strftime("%Y/%m/%d-%H:%M:%S")
        log = "{}: {}\n".format(time_str, txt)
        self.log.write(log)

        self.log.flush()

    def close(self):
        self.log.close()

    def summary(self, step, log_dict):
        if USE_TENSORBOARD:
            for k, v in log_dict.items():
                self.writer.add_scalar(k, v, step)

        logtxt = f"<<< [{step}]: " + ", ".join(
            [f"{k}: {float(v):.8f}" for k, v in log_dict.items()]
        )
        self.write(logtxt)
