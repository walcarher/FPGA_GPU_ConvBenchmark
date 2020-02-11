mkdir "vhdl_generated"
mkdir "vhdl_generated/conv1x1"
mkdir "vhdl_generated/conv3x3"
mkdir "vhdl_generated/conv5x5"
mkdir "vhdl_generated/conv7x7"
mkdir "vhdl_generated/conv11x11"
python conv_gen.py
for i in 2 8 16 32 64 128 256 512 1024;
do
mkdir "vhdl_generated/conv1x1/conv1x1_"$i"_vhdl_generated";
python3 "../delirium_dist/delirium.py" --onnx "vhdl_generated/conv1x1/conv1x1_"$i".onnx" --out "vhdl_generated/conv1x1/conv1x1_"$i"_vhdl_generated" --nbits 8;
mkdir "vhdl_generated/conv3x3/conv3x3_"$i"_vhdl_generated";
python3 "../delirium_dist/delirium.py" --onnx "vhdl_generated/conv3x3/conv3x3_"$i".onnx" --out "vhdl_generated/conv3x3/conv3x3_"$i"_vhdl_generated" --nbits 8;
mkdir "vhdl_generated/conv5x5/conv5x5_"$i"_vhdl_generated";
python3 "../delirium_dist/delirium.py" --onnx "vhdl_generated/conv5x5/conv5x5_"$i".onnx" --out "vhdl_generated/conv5x5/conv5x5_"$i"_vhdl_generated" --nbits 8;
mkdir "vhdl_generated/conv7x7/conv7x7_"$i"_vhdl_generated";
python3 "../delirium_dist/delirium.py" --onnx "vhdl_generated/conv7x7/conv7x7_"$i".onnx" --out "vhdl_generated/conv7x7/conv7x7_"$i"_vhdl_generated" --nbits 8;
mkdir "vhdl_generated/conv11x11/conv11x11_"$i"_vhdl_generated";
python3 "../delirium_dist/delirium.py" --onnx "vhdl_generated/conv11x11/conv11x11_"$i".onnx" --out "vhdl_generated/conv11x11/conv11x11_"$i"_vhdl_generated" --nbits 8;
done
