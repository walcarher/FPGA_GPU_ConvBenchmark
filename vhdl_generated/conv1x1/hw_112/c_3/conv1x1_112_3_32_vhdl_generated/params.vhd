--------------------------------------------------------
-- This file is generated by Delirium - Sma-RTY SAS 
-- Wed Nov 11 05:55:40 2020
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
constant Conv_0_OUT_SIZE     :  integer := 32;
constant Conv_0_MULT_STYLE   :  string  := "dsp";
constant Conv_0_PIPELINE     :  boolean := True;
constant Conv_0_KERNEL_SIZE  :  integer := 1;
constant Conv_0_PADDING      :  boolean := FALSE;
constant Conv_0_STRIDE       :  positive:= 1;
constant Conv_0_BIAS_VALUE   :  data_array  (0 to Conv_0_OUT_SIZE - 1) := 
    (x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00");
constant Conv_0_KERNEL_VALUE :  data_matrix (0 to Conv_0_OUT_SIZE - 1 ,  0 to Conv_0_IN_SIZE * Conv_0_KERNEL_SIZE * Conv_0_KERNEL_SIZE - 1) := (
    (x"be",x"2c",x"22"),
    (x"f5",x"d3",x"2f"),
    (x"21",x"d4",x"cd"),
    (x"f4",x"33",x"ef"),
    (x"46",x"d5",x"df"),
    (x"30",x"38",x"28"),
    (x"2c",x"c6",x"01"),
    (x"bb",x"47",x"0b"),
    (x"f0",x"19",x"c8"),
    (x"c7",x"11",x"bf"),
    (x"32",x"36",x"32"),
    (x"3e",x"c4",x"bf"),
    (x"fe",x"10",x"f1"),
    (x"27",x"fd",x"b9"),
    (x"d5",x"cb",x"39"),
    (x"10",x"e3",x"32"),
    (x"0a",x"d9",x"22"),
    (x"35",x"f2",x"fc"),
    (x"07",x"d0",x"e5"),
    (x"1c",x"20",x"d4"),
    (x"ba",x"e9",x"1c"),
    (x"d7",x"dc",x"24"),
    (x"2b",x"42",x"e1"),
    (x"14",x"22",x"34"),
    (x"fe",x"fc",x"f6"),
    (x"3d",x"2f",x"c0"),
    (x"24",x"36",x"dc"),
    (x"3d",x"ce",x"23"),
    (x"d6",x"e0",x"21"),
    (x"b8",x"1f",x"c8"),
    (x"c9",x"40",x"e8"),
    (x"3a",x"be",x"3e")
);
--------------------------------------------------------
constant OUT_WIDTH     :  integer := 32;
end package;