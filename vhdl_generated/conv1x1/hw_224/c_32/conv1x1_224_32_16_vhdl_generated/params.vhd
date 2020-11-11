--------------------------------------------------------
-- This file is generated by Delirium - Sma-RTY SAS 
-- Wed Nov 11 05:55:11 2020
--------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
library work;
    use work.Delirium.all;

package params is
--------------------------------------------------------
-- INPUT
constant INPUT_CHANNELS    : integer := 32;
constant INPUT_IMAGE_WIDTH : integer := 224;
--------------------------------------------------------
--Conv_0
constant Conv_0_IMAGE_WIDTH  :  integer := 224;
constant Conv_0_IN_SIZE      :  integer := 32;
constant Conv_0_OUT_SIZE     :  integer := 16;
constant Conv_0_MULT_STYLE   :  string  := "dsp";
constant Conv_0_PIPELINE     :  boolean := True;
constant Conv_0_KERNEL_SIZE  :  integer := 1;
constant Conv_0_PADDING      :  boolean := FALSE;
constant Conv_0_STRIDE       :  positive:= 1;
constant Conv_0_BIAS_VALUE   :  data_array  (0 to Conv_0_OUT_SIZE - 1) := 
    (x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00");
constant Conv_0_KERNEL_VALUE :  data_matrix (0 to Conv_0_OUT_SIZE - 1 ,  0 to Conv_0_IN_SIZE * Conv_0_KERNEL_SIZE * Conv_0_KERNEL_SIZE - 1) := (
    (x"f1",x"11",x"ff",x"fe",x"05",x"ee",x"13",x"f9",x"fe",x"02",x"f3",x"05",x"f3",x"0a",x"f6",x"f4",x"02",x"f8",x"15",x"09",x"07",x"16",x"f1",x"ff",x"f7",x"f2",x"f3",x"0a",x"14",x"f7",x"fd",x"f5"),
    (x"eb",x"ec",x"0c",x"0f",x"0f",x"f2",x"06",x"f0",x"11",x"eb",x"f8",x"f7",x"f4",x"03",x"fb",x"fa",x"0a",x"f0",x"f3",x"00",x"12",x"ec",x"04",x"08",x"01",x"f6",x"02",x"0d",x"ee",x"02",x"03",x"ef"),
    (x"12",x"14",x"ff",x"00",x"f3",x"f7",x"04",x"fb",x"f5",x"11",x"06",x"15",x"00",x"f6",x"f7",x"ff",x"00",x"f5",x"02",x"16",x"07",x"01",x"ec",x"05",x"f5",x"fd",x"0f",x"08",x"05",x"08",x"0a",x"f3"),
    (x"15",x"f6",x"16",x"eb",x"f1",x"fa",x"12",x"01",x"12",x"0f",x"0c",x"00",x"01",x"f8",x"ea",x"12",x"f2",x"fa",x"fb",x"08",x"fc",x"fe",x"09",x"06",x"ea",x"ff",x"0c",x"0d",x"ed",x"f9",x"f4",x"eb"),
    (x"07",x"fe",x"16",x"f5",x"fe",x"08",x"fb",x"11",x"ea",x"f2",x"05",x"16",x"f0",x"f8",x"01",x"f1",x"06",x"ec",x"04",x"04",x"0a",x"08",x"14",x"0a",x"f3",x"fa",x"f2",x"fd",x"01",x"0c",x"0a",x"0e"),
    (x"07",x"07",x"eb",x"ec",x"0c",x"13",x"14",x"f9",x"15",x"ee",x"12",x"ef",x"02",x"ff",x"0e",x"07",x"0c",x"07",x"0f",x"11",x"f3",x"0a",x"fe",x"eb",x"f0",x"0d",x"ff",x"fc",x"16",x"fc",x"14",x"f2"),
    (x"09",x"15",x"f0",x"f0",x"00",x"f4",x"0c",x"f9",x"fb",x"f7",x"0f",x"fc",x"ec",x"02",x"ec",x"03",x"fb",x"f0",x"0f",x"f5",x"f8",x"06",x"01",x"ea",x"03",x"eb",x"0d",x"f6",x"10",x"0b",x"07",x"16"),
    (x"f9",x"0d",x"16",x"fc",x"fd",x"13",x"16",x"11",x"04",x"fd",x"f7",x"13",x"02",x"0b",x"10",x"f8",x"ee",x"09",x"05",x"04",x"07",x"05",x"ef",x"01",x"ff",x"13",x"0f",x"fb",x"fb",x"03",x"0f",x"fa"),
    (x"16",x"fe",x"f2",x"ed",x"0f",x"ec",x"fa",x"f9",x"f6",x"f7",x"ef",x"14",x"fe",x"14",x"fb",x"eb",x"f9",x"12",x"f8",x"14",x"f9",x"03",x"09",x"0e",x"ef",x"fd",x"ff",x"03",x"0e",x"f8",x"02",x"11"),
    (x"09",x"0e",x"f3",x"f6",x"12",x"fa",x"05",x"f2",x"07",x"fb",x"06",x"06",x"03",x"03",x"08",x"f6",x"f5",x"ee",x"08",x"fb",x"07",x"0c",x"10",x"12",x"fa",x"0d",x"f5",x"ec",x"f5",x"fe",x"0a",x"14"),
    (x"05",x"06",x"16",x"ea",x"01",x"0f",x"0e",x"07",x"13",x"f2",x"f4",x"00",x"fe",x"f9",x"01",x"f4",x"07",x"05",x"fb",x"16",x"0f",x"08",x"eb",x"f7",x"14",x"0e",x"ed",x"13",x"0e",x"14",x"02",x"05"),
    (x"f0",x"f8",x"11",x"11",x"fd",x"f8",x"12",x"12",x"05",x"04",x"f3",x"ec",x"0b",x"f0",x"06",x"fc",x"15",x"f9",x"ee",x"10",x"ff",x"0d",x"ea",x"01",x"f7",x"06",x"f0",x"f6",x"16",x"02",x"f9",x"13"),
    (x"16",x"08",x"0f",x"04",x"fa",x"10",x"f1",x"07",x"09",x"fe",x"06",x"06",x"f1",x"15",x"13",x"f1",x"0b",x"fe",x"f6",x"02",x"fd",x"0d",x"16",x"04",x"f2",x"f2",x"f5",x"f2",x"04",x"15",x"f5",x"eb"),
    (x"0a",x"ed",x"0b",x"f6",x"ef",x"fe",x"f5",x"fa",x"ff",x"f9",x"15",x"0d",x"10",x"0c",x"f5",x"f5",x"03",x"ec",x"0e",x"0c",x"eb",x"f7",x"12",x"f6",x"03",x"10",x"f1",x"08",x"fa",x"f0",x"fd",x"06"),
    (x"ec",x"13",x"fa",x"ee",x"09",x"f2",x"ec",x"ef",x"ec",x"ed",x"03",x"05",x"01",x"09",x"08",x"08",x"fc",x"f3",x"0c",x"14",x"ec",x"ff",x"f2",x"04",x"10",x"eb",x"04",x"16",x"f4",x"0a",x"02",x"0b"),
    (x"0c",x"fd",x"09",x"f6",x"f8",x"10",x"03",x"11",x"eb",x"ff",x"0e",x"13",x"0f",x"f8",x"eb",x"01",x"11",x"04",x"0f",x"01",x"ea",x"f6",x"f3",x"09",x"13",x"f0",x"fe",x"f3",x"f4",x"14",x"01",x"0b")
);
--------------------------------------------------------
constant OUT_WIDTH     :  integer := 16;
end package;