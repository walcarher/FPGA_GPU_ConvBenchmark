import torch 
import torch.nn as nn
import torch.nn.functional as F
import time

input_tensor = 64 # input image/tensor size i.e. 32x32
start_num_convs = 100 # starting number of filters or depth of the output tensor
max_num_convs = 3000 # max number of filters or depth of the output tensor
step_size_convs = 100 # step size from start to maximum number of iterations
n_iter = 5000  # Number of iterations on a single convolution run
	       # the average of results is reported in output file
time_delay = 0.2 # Pause between running tests

# Writing results to this file
csv_file = open('GPU_time_Conv1x1_64.csv', "wb")

# Random input tensor or image with 1 batch, 1 channel and size 
# input_tensorxinput_tensor
input = torch.randn(1,1,input_tensor,input_tensor).cuda()

def iter_range(start, end, step):
	while start <= end:
		yield start
		start += step

for num_convs in iter_range(start_num_convs, max_num_convs, step_size_convs):

	print("Number of convolutions: %d" % num_convs)

	class Conv1x1_Net(nn.Module):

		def __init__(self):
			super(Conv1x1_Net,self).__init__()
			#1 batch size, n conv output, kernel size 1x1, stride 1-1
			self.conv1 = nn.Conv2d(1, num_convs, 1).cuda()

		def forward(self, x):
			# Maxpooling 2x2 and ReLu activation function
			x = F.max_pool2d(F.relu(self.conv1(x)),(2,2)).cuda()
			return x


	# Convolution layer model
	conv1x1_net = Conv1x1_Net()
	print(conv1x1_net)

	print("Now running ...")
	i = 0
	csv_file.write(str(num_convs)+',')
	csv_file.write('Conv1x1'+',')
	csv_file.write(str(time.time())+',')
	while(i < n_iter):
		out = conv1x1_net(input).cuda()
		i += 1
	csv_file.write(str(time.time())+',')
	csv_file.write('\n')


print("Test Done")
