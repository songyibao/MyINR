import platform
import torch
import subprocess


def get_best_gpu():
    def get_gpu_utilization():
        """获取所有GPU的GPU-Util利用率"""
        result = subprocess.run(['nvidia-smi', '--query-gpu=utilization.gpu', '--format=csv,noheader,nounits'],
                                stdout=subprocess.PIPE, text=True)
        gpu_utilization = list(map(int, result.stdout.split()))
        return gpu_utilization

    def get_least_utilized_gpu():
        """获取利用率最低的GPU"""
        gpu_utilization = get_gpu_utilization()
        # 获取GPU-Util最低的GPU索引
        least_utilized_gpu = gpu_utilization.index(min(gpu_utilization))
        return least_utilized_gpu

    least_occupied_gpu = get_least_utilized_gpu()
    return torch.device(f'cuda:{least_occupied_gpu}')

def get_best_device():
    os_type = platform.system()
    if os_type == 'Windows':
        if torch.cuda.is_available():
            return torch.device('cuda')
        else:
            import torch_directml
            return torch_directml.device() if torch_directml.is_available() else torch.device('cpu')
    elif os_type == 'Linux':
        return get_best_gpu() if torch.cuda.is_available() else torch.device('cpu')
    elif os_type == 'Darwin':
        return torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
    else:
        return torch.device('cpu')


global_device = get_best_device()




