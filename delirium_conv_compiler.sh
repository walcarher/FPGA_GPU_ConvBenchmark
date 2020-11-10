mkdir "vhdl_generated"

CONV=("conv1x1" "conv3x3" "conv5x5" "conv7x7" "conv11x11")
HW=(224 112 56 28 14)
C=(3 16 32 64 128 256)
N=(16 32 64 128 256)

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

python3 conv_gen.py

for n in "${N[@]}"
do
	mkdir "vhdl_generated/conv1x1/conv1x1_"$n"_vhdl_generated";
	python3 "../delirium_dist/delirium.py" --onnx "vhdl_generated/conv1x1/conv1x1_"$n".onnx" --out "vhdl_generated/conv1x1/conv1x1_"$n"_vhdl_generated" --nbits 8;
	mkdir "vhdl_generated/conv3x3/conv3x3_"$n"_vhdl_generated";
	python3 "../delirium_dist/delirium.py" --onnx "vhdl_generated/conv3x3/conv3x3_"$n".onnx" --out "vhdl_generated/conv3x3/conv3x3_"$n"_vhdl_generated" --nbits 8;
	mkdir "vhdl_generated/conv5x5/conv5x5_"$n"_vhdl_generated";
	python3 "../delirium_dist/delirium.py" --onnx "vhdl_generated/conv5x5/conv5x5_"$n".onnx" --out "vhdl_generated/conv5x5/conv5x5_"$n"_vhdl_generated" --nbits 8;
	mkdir "vhdl_generated/conv7x7/conv7x7_"$n"_vhdl_generated";
	python3 "../delirium_dist/delirium.py" --onnx "vhdl_generated/conv7x7/conv7x7_"$n".onnx" --out "vhdl_generated/conv7x7/conv7x7_"$n"_vhdl_generated" --nbits 8;
	mkdir "vhdl_generated/conv11x11/conv11x11_"$n"_vhdl_generated";
	python3 "../delirium_dist/delirium.py" --onnx "vhdl_generated/conv11x11/conv11x11_"$n".onnx" --out "vhdl_generated/conv11x11/conv11x11_"$n"_vhdl_generated" --nbits 8;
done
