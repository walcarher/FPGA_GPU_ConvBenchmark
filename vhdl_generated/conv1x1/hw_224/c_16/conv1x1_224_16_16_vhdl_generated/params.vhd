--------------------------------------------------------
-- This file is generated by Delirium - Sma-RTY SAS 
-- Wed Nov 11 05:55:05 2020
--------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
library work;
    use work.Delirium.all;

package params is
--------------------------------------------------------
-- INPUT
constant INPUT_CHANNELS    : integer := 16;
constant INPUT_IMAGE_WIDTH : integer := 224;
--------------------------------------------------------
--Conv_0
constant Conv_0_IMAGE_WIDTH  :  integer := 224;
constant Conv_0_IN_SIZE      :  integer := 16;
constant Conv_0_OUT_SIZE     :  integer := 16;
constant Conv_0_MULT_STYLE   :  string  := "dsp";
constant Conv_0_PIPELINE     :  boolean := True;
constant Conv_0_KERNEL_SIZE  :  integer := 1;
constant Conv_0_PADDING      :  boolean := FALSE;
constant Conv_0_STRIDE       :  positive:= 1;
constant Conv_0_BIAS_VALUE   :  data_array  (0 to Conv_0_OUT_SIZE - 1) := 
    (x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00");
constant Conv_0_KERNEL_VALUE :  data_matrix (0 to Conv_0_OUT_SIZE - 1 ,  0 to Conv_0_IN_SIZE * Conv_0_KERNEL_SIZE * Conv_0_KERNEL_SIZE - 1) := (
    (x"ec",x"fb",x"14",x"05",x"1f",x"17",x"fd",x"0d",x"02",x"10",x"1a",x"06",x"ff",x"e1",x"1d",x"02"),
    (x"0a",x"09",x"11",x"fe",x"08",x"03",x"03",x"10",x"03",x"f4",x"18",x"14",x"fc",x"e9",x"19",x"1c"),
    (x"fa",x"fc",x"ec",x"f8",x"f4",x"15",x"e6",x"0e",x"14",x"1f",x"e5",x"0a",x"1f",x"f4",x"ff",x"fc"),
    (x"0e",x"04",x"eb",x"e5",x"f7",x"17",x"06",x"1a",x"0d",x"f0",x"fa",x"0e",x"f1",x"09",x"fa",x"1f"),
    (x"0c",x"0c",x"fa",x"ea",x"18",x"08",x"fa",x"e1",x"1c",x"01",x"e8",x"f7",x"02",x"f6",x"f6",x"04"),
    (x"14",x"e3",x"0d",x"e4",x"1b",x"f3",x"f6",x"15",x"11",x"e2",x"e2",x"f3",x"1c",x"09",x"08",x"08"),
    (x"e3",x"04",x"08",x"17",x"e6",x"1b",x"ff",x"ff",x"1a",x"0e",x"ff",x"0c",x"e5",x"e5",x"17",x"ff"),
    (x"12",x"e5",x"08",x"ff",x"1a",x"f4",x"fe",x"1f",x"17",x"18",x"e8",x"18",x"e9",x"13",x"ff",x"0d"),
    (x"eb",x"e6",x"08",x"e5",x"04",x"1b",x"ef",x"f5",x"ea",x"e4",x"f4",x"e9",x"1c",x"f7",x"0f",x"09"),
    (x"ff",x"e5",x"0b",x"0e",x"17",x"1d",x"e0",x"ed",x"f0",x"09",x"19",x"e5",x"07",x"ea",x"e2",x"e1"),
    (x"08",x"1a",x"18",x"f8",x"fb",x"0c",x"15",x"1f",x"0e",x"19",x"06",x"08",x"e9",x"12",x"fc",x"e6"),
    (x"fd",x"f1",x"f7",x"1a",x"f9",x"e8",x"00",x"0d",x"f5",x"f5",x"0d",x"ee",x"f3",x"e1",x"1a",x"10"),
    (x"ec",x"1f",x"1e",x"10",x"03",x"ee",x"0a",x"f9",x"e4",x"e6",x"f5",x"e5",x"13",x"10",x"e3",x"0a"),
    (x"eb",x"05",x"e7",x"fe",x"ff",x"09",x"e1",x"ea",x"e4",x"17",x"14",x"0e",x"14",x"ec",x"13",x"e8"),
    (x"f1",x"fb",x"fc",x"03",x"04",x"00",x"fe",x"e1",x"f6",x"f4",x"fe",x"09",x"08",x"e3",x"f8",x"12"),
    (x"fa",x"10",x"fe",x"04",x"07",x"ea",x"fc",x"e1",x"0a",x"04",x"08",x"e8",x"f6",x"04",x"1b",x"1d")
);
--------------------------------------------------------
constant OUT_WIDTH     :  integer := 16;
end package;