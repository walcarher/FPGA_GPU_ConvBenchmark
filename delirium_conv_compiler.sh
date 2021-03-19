# Create a root directory for VHDL generated files as a tree structure
mkdir "vhdl_generated"

# CNN Hyperparameters listing for Delirium VHDL code generator. ATTENTION: This must match the parameters of conv_gen.py file!
# Future modificatios: Add [args] arguments to conv_gen.py
# Operations: Conv kxk
# Input tensor size: HeightxWeight HxW
# Input channel depth: C
# Number of filters/output channels: N
CONV=("conv1x1" "conv3x3" "conv5x5" "conv7x7" "conv11x11")
#HW=(224 112 56 28 14)
#C=(3 16 32 64 128 256)
HW=(11 20 30 40 50 60 70 80 90 100)
C=(11 20 30 40 50 60 70 80 90 100)
N=(11 20 30 40 50 60 70 80 90 100)
#N=(16 32 64 128 256)

# Generate an iterated folder for each case OperationsxHWxCxN
for conv in "${CONV[@]}";
do
	mkdir "vhdl_generated/"$conv
	for hw in "${HW[@]}";
	do
		mkdir "vhdl_generated/"$conv"/hw_"$hw
		for c in "${C[@]}";
		do 
			mkdir "vhdl_generated/"$conv"/hw_"$hw"/c_"$c
		done
	done
done

# Create .onnx files on each folder vhdl_generated/conkxk/hw_HW/c_C for all n cases in N
python3 conv_gen.py;

# Using Delirium to automatically generate .vhdl files (weights and compilers) from .onnx files. ATTENTION: Delirium is not included!
# Create a folder vhdl_generated/conkxk/hw_HW/c_C/convkxk_hw_c_n_vhdl_generated with all the output .vhdl files
for conv in "${CONV[@]}";
do
	for hw in "${HW[@]}";
	do
		for c in "${C[@]}";
		do 
			for n in "${N[@]}"
			do
				python3 "../delirium_dist/delirium.py" --onnx "vhdl_generated/"$conv"/hw_"$hw"/c_"$c"/"$conv"_"$hw"_"$c"_"$n".onnx" --out  "vhdl_generated/"$conv"/hw_"$hw"/c_"$c"/"$conv"_"$hw"_"$c"_"$n"_vhdl_generated" --nbits 8;
			done	
		done
	done
done

