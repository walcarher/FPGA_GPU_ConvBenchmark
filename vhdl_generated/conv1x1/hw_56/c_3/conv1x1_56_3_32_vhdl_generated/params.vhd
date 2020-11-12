--------------------------------------------------------
-- This file is generated by Delirium - Sma-RTY SAS 
-- Wed Nov 11 12:27:23 2020
--------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
library work;
    use work.Delirium.all;

package params is
--------------------------------------------------------
-- INPUT
constant INPUT_CHANNELS    : integer := 3;
constant INPUT_IMAGE_WIDTH : integer := 56;
--------------------------------------------------------
--Conv_0
constant Conv_0_IMAGE_WIDTH  :  integer := 56;
constant Conv_0_IN_SIZE      :  integer := 3;
constant Conv_0_OUT_SIZE     :  integer := 32;
constant Conv_0_MULT_STYLE   :  string  := "dsp";
constant Conv_0_PIPELINE     :  boolean := True;
constant Conv_0_KERNEL_SIZE  :  integer := 1;
constant Conv_0_PADDING      :  boolean := FALSE;
constant Conv_0_STRIDE       :  positive:= 1;
constant Conv_0_BIAS_VALUE   :  data_array  (0 to Conv_0_OUT_SIZE - 1) := 
    (x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00");
constant Conv_0_KERNEL_VALUE :  data_matrix (0 to Conv_0_OUT_SIZE - 1 ,  0 to Conv_0_IN_SIZE * Conv_0_KERNEL_SIZE * Conv_0_KERNEL_SIZE - 1) := (
    (x"d9",x"ec",x"3e"),
    (x"02",x"42",x"e6"),
    (x"d9",x"17",x"17"),
    (x"d2",x"35",x"c9"),
    (x"c0",x"c1",x"fb"),
    (x"ee",x"ba",x"e3"),
    (x"41",x"f1",x"d1"),
    (x"13",x"dd",x"15"),
    (x"14",x"b8",x"c7"),
    (x"f3",x"40",x"d4"),
    (x"d1",x"27",x"41"),
    (x"b7",x"dc",x"18"),
    (x"f1",x"1e",x"f5"),
    (x"2d",x"23",x"16"),
    (x"f9",x"48",x"e1"),
    (x"f6",x"47",x"1b"),
    (x"c9",x"c9",x"26"),
    (x"e3",x"18",x"27"),
    (x"fc",x"be",x"e9"),
    (x"31",x"07",x"ed"),
    (x"e4",x"0c",x"eb"),
    (x"25",x"41",x"2f"),
    (x"2d",x"e7",x"df"),
    (x"2e",x"fe",x"d0"),
    (x"29",x"2c",x"32"),
    (x"23",x"ed",x"2e"),
    (x"ba",x"c2",x"ea"),
    (x"36",x"47",x"38"),
    (x"45",x"e6",x"fe"),
    (x"cd",x"be",x"06"),
    (x"17",x"1e",x"16"),
    (x"f9",x"1e",x"27")
);
--------------------------------------------------------
--Relu_1
constant Relu_1_OUT_SIZE     :  integer := 32;
--------------------------------------------------------
constant OUT_WIDTH     :  integer := 32;
end package;