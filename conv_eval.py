import torch 
import torch.nn as nn
import torch.nn.functional as F
import time

input_tensor = 224 # input image/tensor size i.e. 224x224 for ImageNet
#start_num_convs = 100 # starting number of filters or depth of the output tensor
#max_num_convs = 3000 # max number of filters or depth of the output tensor
num_conv_list = [1, 8, 16, 32, 64, 128, 256, 512, 1024]
#step_size_convs = 100 # step size from start to maximum number of iterations
n_iter = 5000  # Number of iterations on a single convolution run
	       # the average of results is reported in output file
time_delay = 0.2 # Pause between running tests

# Writing results to this file
csv_file = open('GPU_time.csv', "wb")

# Random input tensor or image with 1 batch, 1 channel and size 
# input_tensorxinput_tensor
input = torch.randn(1,1,input_tensor,input_tensor).cuda()

#def iter_range(start, end, step):
#	while start <= end:
#		yield start
#		start += step

#for num_convs in iter_range(start_num_convs, max_num_convs, step_size_convs):
for num_convs in num_conv_list:

	print("Number of convolutions: %d" % num_convs)

	class Conv1x1_Net(nn.Module):

		def __init__(self):
			super(Conv1x1_Net,self).__init__()
			#1 batch size, n conv output, kernel size 1x1, stride 1-1
			self.conv1 = nn.Conv2d(1, num_convs, 1).cuda()

		def forward(self, x):
			# Maxpooling 2x2 and ReLu activation function
			#x = F.max_pool2d(F.relu(self.conv1(x)),(2,2)).cuda()
			x = self.conv1(x).cuda()
			return x

	class Conv3x3_Net(nn.Module):

		def __init__(self):
			super(Conv3x3_Net,self).__init__()
			#1 batch size, n conv output, kernel size 3x3, stride 1-1
			self.conv1 = nn.Conv2d(1, num_convs, 3).cuda()

		def forward(self, x):
			# Maxpooling 2x2 and ReLu activation function
			#x = F.max_pool2d(F.relu(self.conv1(x)),(2,2)).cuda()
			x = self.conv1(x).cuda()			
			return x

	class Conv5x5_Net(nn.Module):

		def __init__(self):
			super(Conv5x5_Net,self).__init__()
			#1 batch size, n conv output, kernel size 5x5, stride 1-1
			self.conv1 = nn.Conv2d(1, num_convs, 5).cuda()

		def forward(self, x):
			# Maxpooling 2x2 and ReLu activation function
			#x = F.max_pool2d(F.relu(self.conv1(x)),(2,2)).cuda()
			x = self.conv1(x).cuda()			
			return x

	class Conv7x7_Net(nn.Module):

		def __init__(self):
			super(Conv7x7_Net,self).__init__()
			#1 batch size, n conv output, kernel size 7x7, stride 1-1
			self.conv1 = nn.Conv2d(1, num_convs, 7).cuda()

		def forward(self, x):
			# Maxpooling 2x2 and ReLu activation function
			#x = F.max_pool2d(F.relu(self.conv1(x)),(2,2)).cuda()
			x = self.conv1(x).cuda()			
			return x

	class Conv11x11_Net(nn.Module):

		def __init__(self):
			super(Conv11x11_Net,self).__init__()
			#1 batch size, n conv output, kernel size 11x11, stride 1-1
			self.conv1 = nn.Conv2d(1, num_convs, 11).cuda()

		def forward(self, x):
			# Maxpooling 2x2 and ReLu activation function
			#x = F.max_pool2d(F.relu(self.conv1(x)),(2,2)).cuda()
			x = self.conv1(x).cuda()			
			return x

	# Convolution layer model
	conv1x1_net = Conv1x1_Net()
	print(conv1x1_net)
	time.sleep(time_delay)

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

	# Convolution layer model
	conv3x3_net = Conv3x3_Net()
	print(conv3x3_net)
	
	time.sleep(time_delay)
	print("Now running ...")
	i = 0
	csv_file.write(str(num_convs)+',')
	csv_file.write('Conv3x3'+',')
	csv_file.write(str(time.time())+',')
	while(i < n_iter):
		out = conv3x3_net(input).cuda()
		i += 1
	csv_file.write(str(time.time())+',')
	csv_file.write('\n')

	# Convolution layer model
	conv5x5_net = Conv5x5_Net()
	print(conv5x5_net)

	time.sleep(time_delay)
	print("Now running ...")
	i = 0
	csv_file.write(str(num_convs)+',')
	csv_file.write('Conv5x5'+',')
	csv_file.write(str(time.time())+',')
	while(i < n_iter):
		out = conv5x5_net(input).cuda()
		i += 1
	csv_file.write(str(time.time())+',')
	csv_file.write('\n')

	# Convolution layer model
	conv7x7_net = Conv7x7_Net()
	print(conv7x7_net)

	time.sleep(time_delay)
	print("Now running ...")
	i = 0
	csv_file.write(str(num_convs)+',')
	csv_file.write('Conv7x7'+',')
	csv_file.write(str(time.time())+',')
	while(i < n_iter):
		out = conv7x7_net(input).cuda()
		i += 1
	csv_file.write(str(time.time())+',')
	csv_file.write('\n')

	# Convolution layer model
	conv11x11_net = Conv11x11_Net()
	print(conv11x11_net)

	time.sleep(time_delay)
	print("Now running ...")
	i = 0
	csv_file.write(str(num_convs)+',')
	csv_file.write('Conv11x11'+',')
	csv_file.write(str(time.time())+',')
	while(i < n_iter):
		out = conv11x11_net(input).cuda()
		i += 1
	csv_file.write(str(time.time())+',')
	csv_file.write('\n')

print("Test Done")
