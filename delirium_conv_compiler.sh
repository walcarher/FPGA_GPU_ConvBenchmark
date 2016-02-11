for i in 1 8 16 32 64 128 256 512 1024;
do
mkdir "conv1x1/conv1x1_"$i"_vhdl_generated";
python3 "../delirium_dist/delirium.py" --onnx "conv1x1/conv1x1_"$i".onnx" --out "conv1x1/conv1x1_"$i"_vhdl_generated" --nbits 8;
mkdir "conv3x3/conv3x3_"$i"_vhdl_generated";
python3 "../delirium_dist/delirium.py" --onnx "conv3x3/conv3x3_"$i".onnx" --out "conv3x3/conv3x3_"$i"_vhdl_generated" --nbits 8;
mkdir "conv5x5/conv5x5_"$i"_vhdl_generated";
python3 "../delirium_dist/delirium.py" --onnx "conv5x5/conv5x5_"$i".onnx" --out "conv5x5/conv5x5_"$i"_vhdl_generated" --nbits 8;
mkdir "conv7x7/conv7x7_"$i"_vhdl_generated";
python3 "../delirium_dist/delirium.py" --onnx "conv7x7/conv7x7_"$i".onnx" --out "conv7x7/conv7x7_"$i"_vhdl_generated" --nbits 8;
mkdir "conv11x11/conv11x11_"$i"_vhdl_generated";
python3 "../delirium_dist/delirium.py" --onnx "conv11x11/conv11x11_"$i".onnx" --out "conv11x11/conv11x11_"$i"_vhdl_generated" --nbits 8;
done
