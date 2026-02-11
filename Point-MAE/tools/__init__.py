# from .runner import run_net
from .runner import test_net
from .runner_pretrain import run_net as pretrain_run_net
from .runner_finetune import run_net as finetune_run_net
from .runner_finetune import test_net as test_run_net
from .runner_iter import test_net as test_net_iter
from .runner_infer import test_net as test_net_infer