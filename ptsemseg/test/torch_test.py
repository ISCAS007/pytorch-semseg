# -*- coding: utf-8 -*-

import torch
import matplotlib.pyplot as plt

a=torch.rand(3,4)
print(a.shape)
a_np=a.numpy()
print(a_np.shape)
a_cpu=a.data.cpu()
a_gpu=a.data.cuda()
print(a_cpu.shape)
print(a_gpu.shape)
plt.imshow(a)
plt.show()