import torch 
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Size of input tensor
#WH_in_list = [224, 32]
WH_in_list = np.linspace(11, 100, 10, dtype = int ).tolist()
#C_in_list = [3, 512]
C_in_list = np.linspace(11, 100, 10, dtype = int ).tolist()
# Vector list of multiple output tensor channels (number of filters)
#N_list = [32, 1024]
N_list = np.linspace(11, 100, 10, dtype = int ).tolist()
#step_size_convs = 100 # step size from start to maximum number of iterations
#n_iter = 5000  # Number of iterations on a single convolution run
	       # the average of results is reported in output file

print(WH_in_list)
print(C_in_list)
print(N_list)

input_names = ['input']
output_names = ['output']

#def iter_range(start, end, step):
#	while start <= end:
#		yield start
#		start += step

for hw in HW_in_list:
	for c in C_in_list:
		# Random input tensor or image with 1 batch, c channel and size 
		# hxw
		input = torch.randn(1,c,hw,hw)
		#for n in iter_range(start_n, max_n, step_size_convs):
		for n in N_list:

			print("Number of convolutions: %d" % n)

			class Conv1x1_Net(nn.Module):

				def __init__(self):
					super(Conv1x1_Net,self).__init__()
					#1 batch size, n conv output, kernel size 1x1, stride 1-1
					self.conv1 = nn.Conv2d(c, n, 1)

				def forward(self, x):
					# Maxpooling 2x2 and ReLu activation function
					#x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
					x = F.relu(self.conv1(x))
					return x

			class Conv3x3_Net(nn.Module):

				def __init__(self):
					super(Conv3x3_Net,self).__init__()
					#1 batch size, n conv output, kernel size 3x3, stride 1-1
					self.conv1 = nn.Conv2d(c, n, 3)

				def forward(self, x):
					# Maxpooling 2x2 and ReLu activation function
					#x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
					x = F.relu(self.conv1(x))			
					return x

			class Conv5x5_Net(nn.Module):

				def __init__(self):
					super(Conv5x5_Net,self).__init__()
					#1 batch size, n conv output, kernel size 5x5, stride 1-1
					self.conv1 = nn.Conv2d(c, n, 5)

				def forward(self, x):
					# Maxpooling 2x2 and ReLu activation function
					#x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
					x = F.relu(self.conv1(x))			
					return x

			class Conv7x7_Net(nn.Module):

				def __init__(self):
					super(Conv7x7_Net,self).__init__()
					#1 batch size, n conv output, kernel size 7x7, stride 1-1
					self.conv1 = nn.Conv2d(c, n, 7)

				def forward(self, x):
					# Maxpooling 2x2 and ReLu activation function
					#x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
					x = F.relu(self.conv1(x))			
					return x

			class Conv11x11_Net(nn.Module):

				def __init__(self):
					super(Conv11x11_Net,self).__init__()
					#1 batch size, n conv output, kernel size 11x11, stride 1-1
					self.conv1 = nn.Conv2d(c, n, 11)

				def forward(self, x):
					# Maxpooling 2x2 and ReLu activation function
					#x = F.max_pool2d(F.relu(self.conv1(x)),(2,2))
					x = F.relu(self.conv1(x))			
					return x

			# Convolution layer model
			conv1x1_net = Conv1x1_Net()
			print(conv1x1_net)

			print("Now generating ...")

			onnx_file = 'vhdl_generated/conv1x1/hw_%d/c_%d/conv1x1_%d_%d_%d' % (hw, c, hw, c, n) + '.onnx'
			torch.onnx.export(conv1x1_net, input, onnx_file, verbose=True, input_names=input_names, output_names=output_names)

			# Convolution layer model
			conv3x3_net = Conv3x3_Net()
			print(conv3x3_net)
	
			print("Now generating ...")

			onnx_file = 'vhdl_generated/conv3x3/hw_%d/c_%d/conv3x3_%d_%d_%d' % (hw, c, hw, c, n) + '.onnx'
			torch.onnx.export(conv3x3_net, input, onnx_file, verbose=True, input_names=input_names, output_names=output_names)

			# Convolution layer model
			conv5x5_net = Conv5x5_Net()
			print(conv5x5_net)

			print("Now generating ...")

			onnx_file = 'vhdl_generated/conv5x5/hw_%d/c_%d/conv5x5_%d_%d_%d' % (hw, c, hw, c, n) + '.onnx'
			torch.onnx.export(conv5x5_net, input, onnx_file, verbose=True, input_names=input_names, output_names=output_names)

			# Convolution layer model
			conv7x7_net = Conv7x7_Net()
			print(conv7x7_net)

			print("Now generating ...")

			onnx_file = 'vhdl_generated/conv7x7/hw_%d/c_%d/conv7x7_%d_%d_%d' % (hw, c, hw, c, n) + '.onnx'
			torch.onnx.export(conv7x7_net, input, onnx_file, verbose=True, input_names=input_names, output_names=output_names)

			# Convolution layer model
			conv11x11_net = Conv11x11_Net()
			print(conv11x11_net)

			print("Now generating ...")

			onnx_file = 'vhdl_generated/conv11x11/hw_%d/c_%d/conv11x11_%d_%d_%d' % (hw, c, hw, c, n) + '.onnx'
			torch.onnx.export(conv11x11_net, input, onnx_file, verbose=True, input_names=input_names, output_names=output_names)

print("Generation Done")
