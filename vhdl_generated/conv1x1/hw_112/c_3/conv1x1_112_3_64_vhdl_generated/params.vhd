--------------------------------------------------------
-- This file is generated by Delirium - Sma-RTY SAS 
-- Wed Nov 11 05:55:41 2020
--------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
library work;
    use work.Delirium.all;

package params is
--------------------------------------------------------
-- INPUT
constant INPUT_CHANNELS    : integer := 3;
constant INPUT_IMAGE_WIDTH : integer := 112;
--------------------------------------------------------
--Conv_0
constant Conv_0_IMAGE_WIDTH  :  integer := 112;
constant Conv_0_IN_SIZE      :  integer := 3;
constant Conv_0_OUT_SIZE     :  integer := 64;
constant Conv_0_MULT_STYLE   :  string  := "dsp";
constant Conv_0_PIPELINE     :  boolean := True;
constant Conv_0_KERNEL_SIZE  :  integer := 1;
constant Conv_0_PADDING      :  boolean := FALSE;
constant Conv_0_STRIDE       :  positive:= 1;
constant Conv_0_BIAS_VALUE   :  data_array  (0 to Conv_0_OUT_SIZE - 1) := 
    (x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00");
constant Conv_0_KERNEL_VALUE :  data_matrix (0 to Conv_0_OUT_SIZE - 1 ,  0 to Conv_0_IN_SIZE * Conv_0_KERNEL_SIZE * Conv_0_KERNEL_SIZE - 1) := (
    (x"cd",x"31",x"08"),
    (x"28",x"df",x"e5"),
    (x"f3",x"2a",x"12"),
    (x"cc",x"42",x"d4"),
    (x"1f",x"31",x"15"),
    (x"12",x"2a",x"e4"),
    (x"49",x"1f",x"30"),
    (x"ed",x"0e",x"17"),
    (x"d6",x"dc",x"2c"),
    (x"32",x"ba",x"f7"),
    (x"ed",x"f0",x"ef"),
    (x"27",x"1d",x"02"),
    (x"12",x"cb",x"d8"),
    (x"0d",x"dd",x"d6"),
    (x"fb",x"00",x"20"),
    (x"f7",x"44",x"bb"),
    (x"d5",x"c9",x"be"),
    (x"3d",x"36",x"49"),
    (x"cd",x"48",x"3d"),
    (x"d5",x"0c",x"21"),
    (x"3e",x"d6",x"ec"),
    (x"12",x"0a",x"00"),
    (x"30",x"3a",x"e8"),
    (x"e4",x"dd",x"f3"),
    (x"c4",x"27",x"e4"),
    (x"2c",x"f4",x"cc"),
    (x"24",x"11",x"05"),
    (x"1f",x"22",x"35"),
    (x"0e",x"1f",x"2a"),
    (x"e5",x"3c",x"f2"),
    (x"cc",x"e3",x"40"),
    (x"0f",x"08",x"0f"),
    (x"f2",x"03",x"c0"),
    (x"48",x"bc",x"cb"),
    (x"45",x"31",x"38"),
    (x"b9",x"3a",x"10"),
    (x"c2",x"20",x"41"),
    (x"39",x"38",x"eb"),
    (x"2c",x"45",x"d9"),
    (x"d0",x"bc",x"2d"),
    (x"e5",x"be",x"3a"),
    (x"db",x"d6",x"e5"),
    (x"3f",x"c7",x"07"),
    (x"c0",x"c3",x"13"),
    (x"fc",x"46",x"ce"),
    (x"d2",x"ed",x"cf"),
    (x"ff",x"bc",x"23"),
    (x"0b",x"38",x"23"),
    (x"ff",x"1a",x"29"),
    (x"03",x"d2",x"3f"),
    (x"29",x"01",x"d8"),
    (x"d3",x"22",x"0c"),
    (x"ea",x"02",x"e6"),
    (x"0a",x"0d",x"2a"),
    (x"11",x"d2",x"37"),
    (x"00",x"07",x"ce"),
    (x"49",x"12",x"37"),
    (x"18",x"d6",x"c1"),
    (x"ea",x"16",x"36"),
    (x"f4",x"0c",x"fd"),
    (x"41",x"ce",x"06"),
    (x"c3",x"3f",x"d7"),
    (x"d9",x"47",x"b8"),
    (x"dc",x"bc",x"fc")
);
--------------------------------------------------------
constant OUT_WIDTH     :  integer := 64;
end package;