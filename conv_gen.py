import torch 
import torch.nn as nn
import torch.nn.functional as F
import time

input_tensor = 224 # input image/tensor size i.e. 224x224 for ImageNet
#start_num_convs = 100 # starting number of filters or depth of the output tensor
#max_num_convs = 3000 # max number of filters or depth of the output tensor
num_conv_list = [2, 8, 16, 32, 64, 128, 256, 512, 1024]
#step_size_convs = 100 # step size from start to maximum number of iterations
#n_iter = 5000  # Number of iterations on a single convolution run
	       # the average of results is reported in output file


# Random input tensor or image with 1 batch, 1 channel and size 
# input_tensorxinput_tensor
input = torch.randn(1,1,input_tensor,input_tensor)
input_names = ['input']
output_names = ['output']

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
			self.conv1 = nn.Conv2d(1, num_convs, 1)

		def forward(self, x):
			# Maxpooling 2x2 and ReLu activation function
			#x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
			x = self.conv1(x)
			return x

	class Conv3x3_Net(nn.Module):

		def __init__(self):
			super(Conv3x3_Net,self).__init__()
			#1 batch size, n conv output, kernel size 3x3, stride 1-1
			self.conv1 = nn.Conv2d(1, num_convs, 3)

		def forward(self, x):
			# Maxpooling 2x2 and ReLu activation function
			#x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
			x = self.conv1(x)			
			return x

	class Conv5x5_Net(nn.Module):

		def __init__(self):
			super(Conv5x5_Net,self).__init__()
			#1 batch size, n conv output, kernel size 5x5, stride 1-1
			self.conv1 = nn.Conv2d(1, num_convs, 5)

		def forward(self, x):
			# Maxpooling 2x2 and ReLu activation function
			#x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
			x = self.conv1(x)			
			return x

	class Conv7x7_Net(nn.Module):

		def __init__(self):
			super(Conv7x7_Net,self).__init__()
			#1 batch size, n conv output, kernel size 7x7, stride 1-1
			self.conv1 = nn.Conv2d(1, num_convs, 7)

		def forward(self, x):
			# Maxpooling 2x2 and ReLu activation function
			#x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
			x = self.conv1(x)			
			return x

	class Conv11x11_Net(nn.Module):

		def __init__(self):
			super(Conv11x11_Net,self).__init__()
			#1 batch size, n conv output, kernel size 11x11, stride 1-1
			self.conv1 = nn.Conv2d(1, num_convs, 11)

		def forward(self, x):
			# Maxpooling 2x2 and ReLu activation function
			#x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
			x = self.conv1(x)			
			return x

	# Convolution layer model
	conv1x1_net = Conv1x1_Net()
	print(conv1x1_net)

	print("Now generating ...")

	onnx_file = 'vhdl_generated/conv1x1/conv1x1_%d' % num_convs + '.onnx'
	torch.onnx.export(conv1x1_net, input, onnx_file, verbose=True, input_names=input_names, output_names=output_names)

	# Convolution layer model
	conv3x3_net = Conv3x3_Net()
	print(conv3x3_net)
	
	print("Now generating ...")

	onnx_file = 'vhdl_generated/conv3x3/conv3x3_%d' % num_convs + '.onnx'
	torch.onnx.export(conv3x3_net, input, onnx_file, verbose=True, input_names=input_names, output_names=output_names)

	# Convolution layer model
	conv5x5_net = Conv5x5_Net()
	print(conv5x5_net)

	print("Now generating ...")

	onnx_file = 'vhdl_generated/conv5x5/conv5x5_%d' % num_convs + '.onnx'
	torch.onnx.export(conv5x5_net, input, onnx_file, verbose=True, input_names=input_names, output_names=output_names)

	# Convolution layer model
	conv7x7_net = Conv7x7_Net()
	print(conv7x7_net)

	print("Now generating ...")

	onnx_file = 'vhdl_generated/conv7x7/conv7x7_%d' % num_convs + '.onnx'
	torch.onnx.export(conv7x7_net, input, onnx_file, verbose=True, input_names=input_names, output_names=output_names)

	# Convolution layer model
	conv11x11_net = Conv11x11_Net()
	print(conv11x11_net)

	print("Now generating ...")

	onnx_file = 'vhdl_generated/conv11x11/conv11x11_%d' % num_convs + '.onnx'
	torch.onnx.export(conv11x11_net, input, onnx_file, verbose=True, input_names=input_names, output_names=output_names)

print("Generation Done")
