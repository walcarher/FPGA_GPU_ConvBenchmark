--------------------------------------------------------
-- This file is generated by Delirium - Sma-RTY SAS 
-- Wed Nov 11 12:27:34 2020
--------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
library work;
    use work.Delirium.all;

package params is
--------------------------------------------------------
-- INPUT
constant INPUT_CHANNELS    : integer := 32;
constant INPUT_IMAGE_WIDTH : integer := 56;
--------------------------------------------------------
--Conv_0
constant Conv_0_IMAGE_WIDTH  :  integer := 56;
constant Conv_0_IN_SIZE      :  integer := 32;
constant Conv_0_OUT_SIZE     :  integer := 32;
constant Conv_0_MULT_STYLE   :  string  := "dsp";
constant Conv_0_PIPELINE     :  boolean := True;
constant Conv_0_KERNEL_SIZE  :  integer := 1;
constant Conv_0_PADDING      :  boolean := FALSE;
constant Conv_0_STRIDE       :  positive:= 1;
constant Conv_0_BIAS_VALUE   :  data_array  (0 to Conv_0_OUT_SIZE - 1) := 
    (x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00");
constant Conv_0_KERNEL_VALUE :  data_matrix (0 to Conv_0_OUT_SIZE - 1 ,  0 to Conv_0_IN_SIZE * Conv_0_KERNEL_SIZE * Conv_0_KERNEL_SIZE - 1) := (
    (x"fc",x"f3",x"ef",x"07",x"f4",x"16",x"ed",x"ec",x"13",x"f5",x"01",x"f9",x"fb",x"0d",x"02",x"f0",x"13",x"06",x"03",x"ef",x"07",x"fc",x"f3",x"f8",x"fb",x"05",x"02",x"ff",x"f3",x"16",x"0b",x"0b"),
    (x"0c",x"0c",x"ef",x"15",x"09",x"02",x"eb",x"f9",x"ff",x"13",x"ef",x"0c",x"f4",x"0a",x"15",x"03",x"05",x"ea",x"09",x"f8",x"ff",x"09",x"ea",x"0d",x"0f",x"eb",x"00",x"0f",x"0b",x"0d",x"0b",x"ed"),
    (x"fb",x"13",x"15",x"f0",x"ea",x"03",x"ef",x"13",x"13",x"0e",x"ff",x"16",x"03",x"09",x"f5",x"02",x"09",x"10",x"14",x"f9",x"0f",x"0b",x"14",x"05",x"0e",x"15",x"00",x"0e",x"0f",x"f9",x"0d",x"0b"),
    (x"05",x"f4",x"07",x"f8",x"fb",x"f9",x"15",x"f3",x"0c",x"f0",x"04",x"0a",x"f8",x"ff",x"0a",x"0a",x"f7",x"00",x"03",x"10",x"02",x"f0",x"09",x"fc",x"0e",x"fc",x"12",x"fe",x"04",x"13",x"13",x"08"),
    (x"fe",x"f1",x"07",x"0a",x"f8",x"04",x"f2",x"0c",x"f6",x"fc",x"0d",x"16",x"f5",x"ec",x"fd",x"05",x"ff",x"13",x"ef",x"02",x"0d",x"f7",x"08",x"0e",x"ed",x"ef",x"fe",x"f0",x"0b",x"09",x"0e",x"f8"),
    (x"f1",x"f2",x"f3",x"10",x"12",x"eb",x"ed",x"10",x"09",x"ee",x"0e",x"0c",x"f0",x"ea",x"12",x"14",x"0b",x"fa",x"f8",x"f9",x"ec",x"f9",x"12",x"ff",x"0e",x"ef",x"f4",x"ea",x"f5",x"0e",x"ee",x"f6"),
    (x"f8",x"fc",x"fc",x"03",x"12",x"ec",x"ec",x"03",x"0e",x"14",x"11",x"13",x"02",x"0e",x"0b",x"11",x"f6",x"02",x"08",x"ee",x"0a",x"fb",x"ee",x"ff",x"fc",x"ef",x"09",x"f2",x"f7",x"11",x"01",x"0f"),
    (x"12",x"0b",x"09",x"0a",x"13",x"f2",x"03",x"f8",x"fc",x"14",x"09",x"06",x"01",x"f6",x"04",x"11",x"07",x"05",x"ec",x"ea",x"f8",x"f0",x"eb",x"16",x"fe",x"f2",x"10",x"06",x"f3",x"16",x"fd",x"f3"),
    (x"0e",x"fa",x"0c",x"0d",x"10",x"06",x"ed",x"04",x"ee",x"0e",x"ef",x"f6",x"f4",x"05",x"ee",x"06",x"ee",x"05",x"ec",x"0f",x"ec",x"f1",x"fe",x"0b",x"13",x"0f",x"0c",x"07",x"f6",x"f1",x"05",x"f9"),
    (x"f3",x"f6",x"16",x"f6",x"07",x"0e",x"f9",x"fb",x"f0",x"ee",x"0e",x"f5",x"eb",x"0f",x"fe",x"f0",x"f5",x"08",x"ef",x"13",x"08",x"ff",x"eb",x"f1",x"f0",x"f5",x"01",x"fc",x"f2",x"03",x"ee",x"f4"),
    (x"f1",x"0f",x"ff",x"f8",x"ea",x"ea",x"fe",x"16",x"14",x"13",x"0c",x"15",x"f7",x"f5",x"0f",x"fc",x"f9",x"f6",x"07",x"f5",x"0b",x"10",x"11",x"02",x"11",x"10",x"01",x"fb",x"0e",x"f2",x"05",x"07"),
    (x"eb",x"ea",x"00",x"04",x"09",x"f2",x"0f",x"10",x"0c",x"08",x"0a",x"f6",x"02",x"f4",x"13",x"f4",x"07",x"f4",x"14",x"08",x"0b",x"ec",x"f3",x"14",x"f0",x"f6",x"ef",x"04",x"0e",x"0e",x"f3",x"08"),
    (x"ef",x"0b",x"fb",x"04",x"ed",x"fa",x"0d",x"fe",x"ef",x"0c",x"f6",x"f8",x"16",x"0b",x"11",x"eb",x"ec",x"ed",x"f5",x"0f",x"ff",x"fc",x"f1",x"ff",x"f1",x"f1",x"f7",x"0d",x"f8",x"02",x"01",x"11"),
    (x"10",x"0a",x"0d",x"0a",x"16",x"07",x"ff",x"15",x"16",x"eb",x"f5",x"f2",x"13",x"ea",x"0e",x"ec",x"f1",x"02",x"10",x"ee",x"fd",x"eb",x"10",x"ec",x"f5",x"f7",x"02",x"00",x"02",x"ec",x"09",x"f5"),
    (x"16",x"f1",x"f3",x"ea",x"15",x"f2",x"ea",x"ec",x"f2",x"f2",x"08",x"fb",x"ec",x"04",x"fa",x"ec",x"03",x"14",x"f8",x"f6",x"07",x"09",x"fe",x"f1",x"00",x"f3",x"f1",x"06",x"01",x"0a",x"0f",x"f2"),
    (x"fc",x"11",x"f8",x"02",x"0b",x"fa",x"ee",x"ef",x"f7",x"fb",x"01",x"10",x"fd",x"0b",x"00",x"ec",x"05",x"0e",x"13",x"ee",x"f8",x"ea",x"10",x"02",x"eb",x"11",x"0e",x"ed",x"f0",x"ea",x"15",x"ee"),
    (x"0b",x"0f",x"f0",x"ef",x"07",x"11",x"0a",x"05",x"11",x"ed",x"02",x"eb",x"f8",x"04",x"fb",x"02",x"ff",x"fd",x"f2",x"ec",x"ed",x"f4",x"f2",x"fa",x"f8",x"09",x"f3",x"fe",x"0f",x"f2",x"fb",x"ef"),
    (x"f9",x"ec",x"05",x"f9",x"0f",x"fe",x"13",x"f5",x"fe",x"16",x"ed",x"f2",x"ff",x"0a",x"ee",x"16",x"fc",x"09",x"ec",x"08",x"02",x"fd",x"12",x"f8",x"05",x"08",x"fe",x"00",x"fe",x"05",x"02",x"02"),
    (x"03",x"00",x"12",x"fd",x"09",x"fe",x"00",x"00",x"0e",x"f5",x"0d",x"02",x"f7",x"0b",x"ef",x"f4",x"15",x"01",x"15",x"fe",x"fb",x"06",x"f6",x"f3",x"f6",x"15",x"ee",x"ef",x"04",x"00",x"06",x"04"),
    (x"ee",x"f0",x"10",x"fd",x"16",x"ed",x"08",x"fd",x"0e",x"13",x"0c",x"00",x"f3",x"05",x"ff",x"0c",x"14",x"16",x"f5",x"f1",x"02",x"09",x"12",x"ff",x"f3",x"f3",x"16",x"09",x"f0",x"f9",x"0e",x"f2"),
    (x"04",x"0e",x"f6",x"01",x"04",x"fa",x"ed",x"f2",x"ed",x"0a",x"0a",x"0c",x"ef",x"10",x"ee",x"f2",x"08",x"fc",x"07",x"f5",x"00",x"f4",x"f0",x"f0",x"07",x"ef",x"f8",x"0c",x"f4",x"f4",x"01",x"eb"),
    (x"fb",x"f6",x"0a",x"f5",x"f0",x"f7",x"ef",x"f4",x"0d",x"ea",x"04",x"0a",x"f5",x"f7",x"06",x"11",x"15",x"eb",x"02",x"00",x"00",x"f9",x"06",x"0e",x"08",x"f4",x"08",x"01",x"0b",x"eb",x"fd",x"eb"),
    (x"fc",x"04",x"14",x"0b",x"00",x"f4",x"03",x"04",x"f4",x"00",x"07",x"f4",x"0c",x"ee",x"07",x"15",x"04",x"01",x"ff",x"07",x"fe",x"02",x"04",x"f6",x"eb",x"01",x"f1",x"06",x"15",x"fa",x"f4",x"08"),
    (x"fd",x"0f",x"14",x"13",x"fc",x"f8",x"07",x"f7",x"ff",x"f3",x"f9",x"08",x"f9",x"13",x"0f",x"0a",x"06",x"04",x"fa",x"16",x"0f",x"16",x"0d",x"10",x"ea",x"0b",x"ee",x"f2",x"f9",x"f9",x"09",x"04"),
    (x"ef",x"ec",x"fc",x"ed",x"fd",x"0a",x"f3",x"ed",x"0a",x"fe",x"11",x"15",x"f5",x"0f",x"0b",x"ec",x"f5",x"f2",x"f7",x"08",x"13",x"f9",x"0a",x"fb",x"f7",x"16",x"f3",x"06",x"f6",x"03",x"ea",x"0b"),
    (x"0c",x"04",x"07",x"f8",x"09",x"11",x"ea",x"fb",x"f6",x"14",x"f5",x"ea",x"f3",x"0a",x"15",x"ec",x"0a",x"ed",x"ed",x"08",x"f0",x"00",x"0b",x"01",x"01",x"f3",x"ec",x"fb",x"09",x"00",x"0e",x"eb"),
    (x"05",x"03",x"eb",x"03",x"fe",x"fb",x"fe",x"13",x"f9",x"0b",x"00",x"02",x"eb",x"fc",x"ed",x"08",x"0a",x"f4",x"15",x"f1",x"ef",x"f4",x"15",x"ec",x"09",x"f6",x"02",x"ee",x"ee",x"02",x"f6",x"00"),
    (x"eb",x"00",x"fb",x"10",x"10",x"fb",x"ef",x"ee",x"ed",x"08",x"f4",x"0e",x"16",x"fc",x"0b",x"08",x"04",x"16",x"fb",x"0c",x"00",x"fb",x"10",x"11",x"f3",x"f1",x"06",x"ec",x"f1",x"0a",x"f1",x"12"),
    (x"02",x"fa",x"09",x"02",x"eb",x"fc",x"06",x"03",x"0e",x"0d",x"08",x"0f",x"fc",x"f1",x"f9",x"09",x"11",x"f6",x"13",x"ec",x"02",x"f5",x"08",x"fe",x"04",x"f1",x"13",x"0c",x"01",x"0a",x"f1",x"f6"),
    (x"06",x"15",x"10",x"f4",x"02",x"10",x"0c",x"eb",x"ed",x"01",x"16",x"01",x"f7",x"0e",x"08",x"06",x"ec",x"f6",x"01",x"fa",x"12",x"04",x"0c",x"0a",x"ed",x"f3",x"ec",x"ea",x"f9",x"0c",x"06",x"08"),
    (x"ef",x"0a",x"0f",x"f8",x"15",x"fa",x"f0",x"0e",x"0c",x"fa",x"14",x"f0",x"f5",x"f8",x"10",x"16",x"fe",x"0b",x"07",x"09",x"10",x"13",x"f6",x"15",x"05",x"0c",x"0f",x"ef",x"01",x"fb",x"10",x"04"),
    (x"09",x"f9",x"f8",x"01",x"f2",x"ea",x"f8",x"0f",x"f3",x"0c",x"f7",x"fa",x"fe",x"fb",x"f9",x"0c",x"04",x"f7",x"ff",x"ec",x"03",x"12",x"f6",x"f9",x"12",x"02",x"08",x"f4",x"15",x"fe",x"ec",x"eb")
);
--------------------------------------------------------
--Relu_1
constant Relu_1_OUT_SIZE     :  integer := 32;
--------------------------------------------------------
constant OUT_WIDTH     :  integer := 32;
end package;