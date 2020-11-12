--------------------------------------------------------
-- This file is generated by Delirium - Sma-RTY SAS 
-- Wed Nov 11 12:28:39 2020
--------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
library work;
    use work.Delirium.all;

package params is
--------------------------------------------------------
-- INPUT
constant INPUT_CHANNELS    : integer := 3;
constant INPUT_IMAGE_WIDTH : integer := 14;
--------------------------------------------------------
--Conv_0
constant Conv_0_IMAGE_WIDTH  :  integer := 14;
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
    (x"b8",x"48",x"ee"),
    (x"c5",x"d3",x"0e"),
    (x"0c",x"d8",x"09"),
    (x"b9",x"09",x"15"),
    (x"e8",x"ff",x"10"),
    (x"dd",x"25",x"03"),
    (x"14",x"3f",x"cc"),
    (x"e5",x"30",x"04"),
    (x"f4",x"02",x"47"),
    (x"b7",x"00",x"b7"),
    (x"e9",x"cb",x"ba"),
    (x"35",x"b7",x"17"),
    (x"20",x"e2",x"d1"),
    (x"08",x"05",x"e1"),
    (x"24",x"e3",x"f0"),
    (x"39",x"b8",x"28"),
    (x"d6",x"dd",x"27"),
    (x"27",x"ba",x"de"),
    (x"19",x"f7",x"40"),
    (x"d8",x"2b",x"d4"),
    (x"46",x"12",x"d0"),
    (x"f7",x"f3",x"02"),
    (x"d8",x"f7",x"0f"),
    (x"1f",x"c1",x"ed"),
    (x"d5",x"3d",x"d8"),
    (x"f0",x"37",x"bb"),
    (x"3a",x"ba",x"d2"),
    (x"07",x"c3",x"e4"),
    (x"c6",x"e6",x"19"),
    (x"f7",x"c4",x"25"),
    (x"c5",x"ca",x"30"),
    (x"0f",x"d1",x"1d")
);
--------------------------------------------------------
--Relu_1
constant Relu_1_OUT_SIZE     :  integer := 32;
--------------------------------------------------------
constant OUT_WIDTH     :  integer := 32;
end package;