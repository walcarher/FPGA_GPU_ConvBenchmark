--------------------------------------------------------
-- This file is generated by Delirium - Sma-RTY SAS 
-- Wed Nov 11 05:57:36 2020
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
    (x"ee",x"40",x"bc"),
    (x"3b",x"d9",x"f0"),
    (x"f5",x"bf",x"f4"),
    (x"07",x"e4",x"c2"),
    (x"d6",x"2d",x"35"),
    (x"0e",x"d7",x"c1"),
    (x"39",x"d4",x"c6"),
    (x"42",x"f5",x"ec"),
    (x"35",x"21",x"e1"),
    (x"39",x"1d",x"d1"),
    (x"1a",x"3f",x"3c"),
    (x"d4",x"db",x"d7"),
    (x"29",x"15",x"df"),
    (x"33",x"c0",x"3e"),
    (x"42",x"c3",x"04"),
    (x"24",x"3d",x"33"),
    (x"fb",x"0f",x"f3"),
    (x"ec",x"cf",x"bf"),
    (x"38",x"29",x"0c"),
    (x"f0",x"cc",x"27"),
    (x"05",x"ee",x"38"),
    (x"b9",x"cc",x"fe"),
    (x"44",x"28",x"27"),
    (x"24",x"0e",x"41"),
    (x"f1",x"d4",x"11"),
    (x"e5",x"10",x"f2"),
    (x"02",x"f1",x"32"),
    (x"e9",x"0d",x"f8"),
    (x"e1",x"02",x"2d"),
    (x"3c",x"fc",x"e7"),
    (x"0a",x"f1",x"c9"),
    (x"e1",x"ff",x"28")
);
--------------------------------------------------------
constant OUT_WIDTH     :  integer := 32;
end package;