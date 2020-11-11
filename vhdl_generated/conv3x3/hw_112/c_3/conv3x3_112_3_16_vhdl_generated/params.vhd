--------------------------------------------------------
-- This file is generated by Delirium - Sma-RTY SAS 
-- Wed Nov 11 05:59:26 2020
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
constant Conv_0_OUT_SIZE     :  integer := 16;
constant Conv_0_MULT_STYLE   :  string  := "dsp";
constant Conv_0_PIPELINE     :  boolean := True;
constant Conv_0_KERNEL_SIZE  :  integer := 3;
constant Conv_0_PADDING      :  boolean := FALSE;
constant Conv_0_STRIDE       :  positive:= 1;
constant Conv_0_BIAS_VALUE   :  data_array  (0 to Conv_0_OUT_SIZE - 1) := 
    (x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00");
constant Conv_0_KERNEL_VALUE :  data_matrix (0 to Conv_0_OUT_SIZE - 1 ,  0 to Conv_0_IN_SIZE * Conv_0_KERNEL_SIZE * Conv_0_KERNEL_SIZE - 1) := (
    (x"ea",x"04",x"0b",x"fd",x"0f",x"ff",x"f4",x"f8",x"f1",x"11",x"0d",x"ee",x"fc",x"0a",x"ee",x"fe",x"07",x"0c",x"f0",x"f1",x"0e",x"f5",x"ff",x"ec",x"16",x"ea",x"14"),
    (x"f8",x"09",x"ec",x"0c",x"f4",x"05",x"f9",x"ef",x"ec",x"0e",x"0a",x"08",x"f6",x"13",x"f9",x"02",x"e9",x"ec",x"f4",x"0c",x"f9",x"fa",x"f7",x"ed",x"ee",x"f5",x"12"),
    (x"18",x"04",x"18",x"11",x"f8",x"0c",x"17",x"0f",x"ea",x"14",x"fb",x"f9",x"f9",x"f3",x"07",x"0d",x"16",x"14",x"f2",x"fd",x"07",x"12",x"ec",x"fe",x"06",x"0e",x"08"),
    (x"ea",x"f8",x"00",x"fe",x"fa",x"05",x"fb",x"14",x"09",x"03",x"f7",x"01",x"fe",x"fe",x"0f",x"03",x"f4",x"15",x"02",x"fd",x"f2",x"05",x"15",x"14",x"03",x"18",x"fb"),
    (x"e8",x"0a",x"ed",x"fc",x"13",x"f6",x"0d",x"f3",x"00",x"0c",x"02",x"f2",x"10",x"fe",x"f9",x"07",x"f9",x"0e",x"13",x"f7",x"ff",x"01",x"f5",x"00",x"14",x"08",x"04"),
    (x"fa",x"04",x"e9",x"eb",x"13",x"ea",x"15",x"ff",x"ec",x"10",x"f2",x"06",x"03",x"f9",x"10",x"01",x"0f",x"01",x"05",x"f3",x"01",x"fb",x"11",x"0c",x"0c",x"05",x"fc"),
    (x"06",x"f6",x"fa",x"fb",x"0b",x"09",x"01",x"f7",x"fa",x"e8",x"e9",x"f9",x"fc",x"ed",x"0e",x"00",x"02",x"02",x"fd",x"12",x"eb",x"e8",x"08",x"11",x"f9",x"e9",x"f9"),
    (x"ed",x"eb",x"ee",x"f0",x"0f",x"f7",x"f2",x"eb",x"fb",x"eb",x"ef",x"05",x"02",x"ea",x"f4",x"18",x"fe",x"fa",x"12",x"f0",x"09",x"08",x"14",x"0a",x"f5",x"f6",x"f1"),
    (x"17",x"13",x"ee",x"03",x"13",x"18",x"17",x"f4",x"04",x"12",x"04",x"0f",x"f9",x"ea",x"09",x"f9",x"f0",x"04",x"0b",x"fb",x"e8",x"f0",x"11",x"fd",x"13",x"08",x"08"),
    (x"11",x"06",x"0d",x"02",x"ef",x"ef",x"0f",x"f0",x"12",x"17",x"f0",x"11",x"12",x"f4",x"12",x"13",x"15",x"ea",x"07",x"06",x"0e",x"f0",x"17",x"f7",x"ee",x"17",x"ed"),
    (x"10",x"f8",x"f3",x"16",x"fe",x"17",x"ee",x"03",x"01",x"04",x"03",x"0c",x"fd",x"f0",x"05",x"e9",x"ec",x"f3",x"09",x"0c",x"f8",x"09",x"eb",x"f9",x"f5",x"06",x"eb"),
    (x"07",x"f8",x"fe",x"ec",x"04",x"12",x"01",x"0b",x"0a",x"0a",x"18",x"fd",x"05",x"11",x"f8",x"11",x"01",x"02",x"f9",x"fb",x"e9",x"15",x"ee",x"0f",x"f6",x"05",x"f1"),
    (x"f5",x"12",x"f3",x"f2",x"02",x"e9",x"01",x"07",x"ec",x"f6",x"f3",x"00",x"f3",x"04",x"0e",x"0a",x"0d",x"f0",x"ef",x"fd",x"ee",x"f5",x"0d",x"fa",x"f0",x"18",x"e8"),
    (x"03",x"f1",x"fe",x"f5",x"f6",x"0a",x"09",x"f7",x"15",x"f5",x"12",x"13",x"0b",x"12",x"f3",x"09",x"f4",x"f8",x"eb",x"0e",x"f1",x"09",x"06",x"ea",x"f9",x"ff",x"f9"),
    (x"01",x"e9",x"f8",x"16",x"08",x"05",x"f5",x"09",x"ec",x"04",x"11",x"fd",x"fb",x"fc",x"15",x"0a",x"fe",x"00",x"f5",x"05",x"f8",x"ee",x"10",x"e8",x"11",x"11",x"f8"),
    (x"0d",x"f0",x"09",x"ea",x"f0",x"0a",x"0f",x"15",x"e9",x"07",x"15",x"10",x"ee",x"fc",x"ee",x"ec",x"08",x"07",x"10",x"f0",x"ff",x"15",x"ec",x"06",x"11",x"07",x"08")
);
--------------------------------------------------------
constant OUT_WIDTH     :  integer := 16;
end package;