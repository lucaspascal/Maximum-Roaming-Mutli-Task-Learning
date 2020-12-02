from torch import nn



class SE(nn.Module):
    """
    Squeeze and Excitation Layer for multiple tasks (dict)
    """
    def __init__(self, channel, task_count, reduction=16):
        super(SE, self).__init__()
        self.task_count = task_count
        self.active_task = 0
        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        print('Initializing squeeze and excitation modules:')
        self.fc = nn.ModuleList()
        for k in range(self.task_count):
            print('SE for task: {}'.format(k))
            self.fc.append(nn.Sequential(nn.Linear(channel, channel // reduction),
                                         nn.ReLU(inplace=True),
                                         nn.Linear(channel // reduction, channel),
                                         nn.Sigmoid()))

            
    def set_active_task(self, active_task):
        self.active_task = active_task
        return active_task

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc[self.active_task](y).view(b, c, 1, 1)
        return x * y
