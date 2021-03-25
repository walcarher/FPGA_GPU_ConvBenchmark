import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Size of input tensor
#WH_in_list = [224, 32]
WH_in_list = np.linspace(5, 14, 10, dtype = int ).tolist()
#C_in_list = [3, 512]
C_in_list = np.linspace(1, 10, 10, dtype = int ).tolist()
# Vector list of multiple output tensor channels (number of filters)
#N_list = [32, 1024]
N_list = np.linspace(1, 10, 10, dtype = int ).tolist()
#step_size_convs = 100 # step size from start to maximum number of iterations
K_list = [5]
#n_iter = 5000  # Number of iterations on a single convolution run
	       # the average of results is reported in output file

input_names = ['input']
output_names = ['output']

#def iter_range(start, end, step):
#	while start <= end:
#		yield start
#		start += step

for hw in WH_in_list:
    for c in C_in_list:
        # Random input tensor or image with 1 batch, c channel and size 
        # hxw
        input = torch.randn(1,c,hw,hw)
        #for n in iter_range(start_n, max_n, step_size_convs):
        for n in N_list:

            print("Number of convolutions: %d" % n)

            for k in K_list:

                class Convkxk_Net(nn.Module):
                    def __init__(self):
                        super(Convkxk_Net,self).__init__()
                        # batch size, n conv output, kernel size kxk, stride 1-1
                        self.conv1 = nn.Conv2d(c, n, k)

                    def forward(self, x):
                        # Maxpooling 2x2 and ReLu activation function
                        #x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
                        x = F.relu(self.conv1(x))
                        return x

                # Convolution layer model
                convkxk_net = Convkxk_Net()
                print(convkxk_net)

                print("Now generating ...")

                onnx_file = 'vhdl_generated/conv%dx%d/hw_%d/c_%d/conv%dx%d_%d_%d_%d' % (k, k, hw, c, k, k, hw, c, n) + '.onnx'
                torch.onnx.export(convkxk_net, input, onnx_file, verbose=True, input_names=input_names, output_names=output_names)                

print("Generation Done")
