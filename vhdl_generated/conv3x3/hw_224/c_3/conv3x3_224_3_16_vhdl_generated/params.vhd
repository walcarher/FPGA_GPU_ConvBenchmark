--------------------------------------------------------
-- This file is generated by Delirium - Sma-RTY SAS 
-- Wed Nov 11 12:29:15 2020
--------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
library work;
    use work.Delirium.all;

package params is
--------------------------------------------------------
-- INPUT
constant INPUT_CHANNELS    : integer := 3;
constant INPUT_IMAGE_WIDTH : integer := 224;
--------------------------------------------------------
--Conv_0
constant Conv_0_IMAGE_WIDTH  :  integer := 224;
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
    (x"fa",x"15",x"0d",x"08",x"fb",x"eb",x"f0",x"ed",x"0b",x"f2",x"f9",x"ea",x"f0",x"01",x"16",x"f2",x"12",x"e9",x"f8",x"0c",x"12",x"fb",x"eb",x"08",x"f4",x"f7",x"eb"),
    (x"10",x"18",x"fc",x"fd",x"03",x"ea",x"fd",x"0e",x"07",x"13",x"ff",x"00",x"0d",x"eb",x"ed",x"03",x"f9",x"f4",x"15",x"14",x"fc",x"f2",x"ed",x"03",x"05",x"e9",x"07"),
    (x"f1",x"f8",x"f6",x"f7",x"f9",x"ed",x"00",x"f9",x"01",x"f4",x"04",x"07",x"eb",x"05",x"eb",x"18",x"01",x"fa",x"f9",x"05",x"08",x"eb",x"11",x"f8",x"11",x"f5",x"03"),
    (x"ee",x"01",x"02",x"01",x"07",x"ea",x"ef",x"15",x"10",x"0d",x"0f",x"11",x"13",x"f1",x"09",x"f0",x"0f",x"f1",x"17",x"14",x"0f",x"fa",x"ef",x"0f",x"ea",x"16",x"0f"),
    (x"12",x"e9",x"14",x"01",x"10",x"0c",x"18",x"0c",x"10",x"0b",x"01",x"16",x"04",x"02",x"04",x"16",x"f6",x"eb",x"08",x"f2",x"0f",x"f3",x"ff",x"f5",x"07",x"0b",x"fa"),
    (x"e8",x"16",x"15",x"fe",x"ee",x"e9",x"00",x"04",x"eb",x"08",x"f6",x"0a",x"0c",x"f1",x"f6",x"e9",x"fe",x"06",x"18",x"f5",x"ed",x"03",x"fa",x"f1",x"16",x"f3",x"08"),
    (x"01",x"16",x"03",x"0b",x"0c",x"fb",x"14",x"fc",x"fb",x"09",x"01",x"15",x"13",x"0c",x"ea",x"06",x"09",x"f2",x"17",x"0c",x"ec",x"ef",x"ee",x"0f",x"fe",x"e8",x"13"),
    (x"f2",x"e8",x"f2",x"fb",x"06",x"f7",x"ed",x"fb",x"ec",x"ff",x"0c",x"0f",x"f2",x"fe",x"ed",x"00",x"fe",x"04",x"17",x"01",x"0d",x"ee",x"f7",x"07",x"f3",x"18",x"f6"),
    (x"0c",x"00",x"12",x"e9",x"09",x"15",x"00",x"07",x"f1",x"12",x"fe",x"fe",x"ec",x"ea",x"18",x"eb",x"04",x"10",x"03",x"07",x"0b",x"02",x"06",x"f5",x"17",x"fe",x"0e"),
    (x"f5",x"f0",x"f9",x"05",x"02",x"13",x"ea",x"fe",x"18",x"18",x"0c",x"17",x"ed",x"00",x"0c",x"10",x"09",x"04",x"f7",x"f9",x"03",x"13",x"15",x"00",x"f6",x"ef",x"f9"),
    (x"04",x"f9",x"03",x"f3",x"ec",x"0a",x"f0",x"17",x"11",x"ee",x"f3",x"f8",x"f1",x"00",x"fc",x"0b",x"ec",x"fc",x"fa",x"ef",x"02",x"0d",x"02",x"ee",x"06",x"06",x"f4"),
    (x"fd",x"fa",x"06",x"f8",x"fd",x"fa",x"e9",x"f1",x"fb",x"18",x"f3",x"04",x"eb",x"eb",x"12",x"ff",x"fc",x"15",x"0d",x"12",x"02",x"f0",x"18",x"0e",x"ec",x"0c",x"ec"),
    (x"f0",x"e9",x"12",x"10",x"06",x"e9",x"ed",x"0b",x"f2",x"ff",x"0c",x"f3",x"17",x"08",x"17",x"0f",x"12",x"f3",x"fd",x"eb",x"13",x"14",x"00",x"0f",x"fe",x"04",x"13"),
    (x"05",x"f7",x"0e",x"09",x"10",x"fb",x"ef",x"10",x"17",x"ee",x"0d",x"09",x"f5",x"12",x"ef",x"06",x"04",x"fd",x"08",x"0a",x"fd",x"18",x"03",x"fd",x"18",x"0e",x"fe"),
    (x"15",x"f2",x"18",x"06",x"fa",x"ec",x"07",x"04",x"00",x"11",x"06",x"0b",x"f4",x"10",x"0b",x"f4",x"07",x"f1",x"18",x"01",x"0f",x"f3",x"fb",x"ff",x"fb",x"fb",x"f0"),
    (x"0a",x"eb",x"16",x"0b",x"fb",x"08",x"ff",x"f5",x"0b",x"08",x"07",x"0d",x"0b",x"f9",x"07",x"0f",x"ec",x"0e",x"f6",x"f4",x"0b",x"04",x"ff",x"fb",x"eb",x"15",x"f2")
);
--------------------------------------------------------
--Relu_1
constant Relu_1_OUT_SIZE     :  integer := 16;
--------------------------------------------------------
constant OUT_WIDTH     :  integer := 16;
end package;