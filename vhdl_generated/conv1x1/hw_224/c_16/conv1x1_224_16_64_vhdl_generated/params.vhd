--------------------------------------------------------
-- This file is generated by Delirium - Sma-RTY SAS 
-- Wed Nov 11 12:26:14 2020
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
constant Conv_0_OUT_SIZE     :  integer := 64;
constant Conv_0_MULT_STYLE   :  string  := "dsp";
constant Conv_0_PIPELINE     :  boolean := True;
constant Conv_0_KERNEL_SIZE  :  integer := 1;
constant Conv_0_PADDING      :  boolean := FALSE;
constant Conv_0_STRIDE       :  positive:= 1;
constant Conv_0_BIAS_VALUE   :  data_array  (0 to Conv_0_OUT_SIZE - 1) := 
    (x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00");
constant Conv_0_KERNEL_VALUE :  data_matrix (0 to Conv_0_OUT_SIZE - 1 ,  0 to Conv_0_IN_SIZE * Conv_0_KERNEL_SIZE * Conv_0_KERNEL_SIZE - 1) := (
    (x"01",x"0c",x"f9",x"04",x"f5",x"14",x"ff",x"e0",x"0f",x"f5",x"1a",x"1f",x"fe",x"fa",x"f8",x"f2"),
    (x"08",x"f4",x"fc",x"f9",x"ef",x"ed",x"f2",x"f1",x"02",x"1e",x"04",x"ff",x"f4",x"0b",x"e5",x"07"),
    (x"02",x"19",x"ec",x"04",x"fd",x"11",x"e2",x"16",x"01",x"14",x"fb",x"f6",x"06",x"fd",x"05",x"1b"),
    (x"19",x"15",x"00",x"fe",x"18",x"e8",x"05",x"02",x"15",x"1f",x"fe",x"0b",x"f2",x"18",x"0f",x"f2"),
    (x"eb",x"02",x"11",x"08",x"10",x"e1",x"0c",x"f6",x"02",x"1d",x"08",x"f5",x"ef",x"13",x"e5",x"f8"),
    (x"0c",x"18",x"05",x"e0",x"13",x"fa",x"ef",x"1d",x"0d",x"1b",x"09",x"0c",x"ed",x"e3",x"e8",x"0b"),
    (x"00",x"12",x"1c",x"ed",x"f1",x"0d",x"f1",x"09",x"1c",x"ec",x"f7",x"e1",x"f8",x"f1",x"0d",x"ed"),
    (x"ef",x"0a",x"ea",x"1b",x"14",x"eb",x"e6",x"ee",x"fb",x"03",x"f1",x"04",x"11",x"fd",x"01",x"12"),
    (x"01",x"e5",x"10",x"1b",x"ec",x"ff",x"eb",x"01",x"0a",x"e9",x"1c",x"13",x"e1",x"1b",x"12",x"16"),
    (x"f7",x"e6",x"fd",x"e5",x"f4",x"15",x"e3",x"17",x"fd",x"f0",x"08",x"e4",x"e5",x"ee",x"f6",x"04"),
    (x"1f",x"f9",x"06",x"02",x"fd",x"f1",x"13",x"ff",x"16",x"e7",x"00",x"fb",x"0e",x"fb",x"1c",x"e1"),
    (x"1b",x"1d",x"0e",x"e5",x"f6",x"ef",x"12",x"fd",x"f2",x"f2",x"e7",x"1e",x"09",x"1d",x"fc",x"f9"),
    (x"08",x"02",x"14",x"ef",x"0c",x"04",x"e2",x"f3",x"fa",x"1a",x"0a",x"1d",x"10",x"f4",x"09",x"08"),
    (x"f5",x"1d",x"1f",x"e8",x"1c",x"eb",x"05",x"f9",x"1e",x"11",x"e5",x"e3",x"20",x"e2",x"09",x"11"),
    (x"1e",x"fa",x"06",x"fa",x"0e",x"15",x"f1",x"ef",x"f5",x"ff",x"f9",x"07",x"e5",x"0c",x"14",x"e5"),
    (x"e0",x"eb",x"e3",x"f8",x"f1",x"fc",x"f4",x"ea",x"e0",x"02",x"05",x"fc",x"09",x"09",x"e8",x"e7"),
    (x"05",x"1a",x"e9",x"02",x"03",x"e7",x"f7",x"1a",x"1e",x"12",x"f3",x"0f",x"e9",x"e5",x"e8",x"04"),
    (x"10",x"15",x"e0",x"17",x"12",x"f3",x"1a",x"e3",x"eb",x"03",x"09",x"ef",x"1f",x"0a",x"eb",x"19"),
    (x"20",x"f1",x"fd",x"15",x"13",x"0a",x"ff",x"e7",x"18",x"f7",x"02",x"14",x"04",x"e1",x"0d",x"19"),
    (x"e7",x"eb",x"e1",x"ec",x"ef",x"00",x"0c",x"09",x"ea",x"ec",x"14",x"1a",x"13",x"1c",x"08",x"11"),
    (x"0d",x"04",x"ed",x"06",x"e2",x"f2",x"01",x"fc",x"1e",x"1a",x"f6",x"16",x"fe",x"fa",x"fa",x"0b"),
    (x"fc",x"e4",x"f4",x"1d",x"f5",x"e7",x"1b",x"e6",x"f6",x"0b",x"0a",x"1f",x"f9",x"f8",x"fd",x"11"),
    (x"06",x"07",x"f9",x"19",x"ee",x"05",x"07",x"11",x"1f",x"fa",x"13",x"18",x"f5",x"19",x"fd",x"0f"),
    (x"1e",x"f8",x"ff",x"fc",x"17",x"1e",x"fe",x"fd",x"fe",x"f5",x"08",x"fd",x"e7",x"fb",x"0e",x"eb"),
    (x"1a",x"f0",x"f9",x"ec",x"e9",x"19",x"0b",x"f1",x"ed",x"1a",x"1b",x"fd",x"17",x"fd",x"13",x"fd"),
    (x"03",x"1d",x"f3",x"e6",x"09",x"fc",x"10",x"e0",x"15",x"fa",x"fb",x"fa",x"ee",x"ff",x"e8",x"1c"),
    (x"0f",x"19",x"e9",x"0a",x"f7",x"1f",x"1f",x"f2",x"0f",x"f7",x"18",x"e3",x"1f",x"ed",x"e2",x"fc"),
    (x"e8",x"0e",x"fd",x"0d",x"1c",x"f3",x"19",x"04",x"e1",x"e1",x"13",x"1c",x"0e",x"f9",x"0a",x"16"),
    (x"f4",x"12",x"f2",x"08",x"e1",x"0b",x"17",x"18",x"f1",x"fc",x"ff",x"eb",x"18",x"eb",x"1e",x"1b"),
    (x"06",x"ed",x"f3",x"15",x"06",x"01",x"f0",x"ec",x"f9",x"fc",x"fe",x"ff",x"16",x"05",x"1f",x"f5"),
    (x"0f",x"16",x"0b",x"fe",x"00",x"07",x"0d",x"0e",x"ee",x"e8",x"e6",x"13",x"17",x"05",x"f7",x"fb"),
    (x"ea",x"0a",x"18",x"ef",x"19",x"02",x"e8",x"16",x"1a",x"0f",x"fd",x"16",x"05",x"f5",x"1e",x"ec"),
    (x"ee",x"04",x"0f",x"f8",x"e7",x"e3",x"f0",x"10",x"eb",x"e2",x"fb",x"e4",x"ec",x"e1",x"0e",x"02"),
    (x"10",x"ee",x"fe",x"f6",x"f6",x"fe",x"1b",x"0e",x"ef",x"ee",x"e1",x"1e",x"fd",x"ef",x"0a",x"14"),
    (x"12",x"ea",x"e4",x"11",x"01",x"eb",x"14",x"0f",x"06",x"f6",x"f1",x"15",x"f3",x"05",x"fa",x"00"),
    (x"08",x"12",x"19",x"ec",x"06",x"e7",x"e1",x"16",x"e5",x"e8",x"1f",x"f6",x"fd",x"e3",x"f2",x"f7"),
    (x"f4",x"e7",x"e2",x"f2",x"e8",x"f6",x"e9",x"03",x"04",x"15",x"f8",x"0a",x"e2",x"0d",x"1d",x"e9"),
    (x"02",x"f1",x"13",x"01",x"14",x"05",x"e8",x"16",x"e9",x"0c",x"11",x"19",x"f6",x"e3",x"e6",x"1f"),
    (x"00",x"0d",x"e4",x"e1",x"fc",x"0a",x"05",x"1a",x"fd",x"f6",x"16",x"06",x"1c",x"1d",x"1c",x"ed"),
    (x"01",x"fd",x"ed",x"e2",x"f9",x"fd",x"f0",x"1f",x"e7",x"0a",x"09",x"0a",x"01",x"f4",x"e7",x"f9"),
    (x"18",x"1d",x"eb",x"0f",x"f4",x"09",x"e9",x"11",x"04",x"16",x"f2",x"1a",x"fd",x"09",x"0f",x"0f"),
    (x"e1",x"10",x"f4",x"e4",x"09",x"15",x"1f",x"13",x"fa",x"0c",x"0b",x"1c",x"0d",x"e7",x"08",x"fa"),
    (x"ed",x"09",x"15",x"00",x"13",x"f8",x"1b",x"ed",x"02",x"f4",x"00",x"ec",x"fb",x"ec",x"0b",x"15"),
    (x"19",x"e5",x"f4",x"f2",x"fd",x"15",x"f0",x"1d",x"ff",x"ed",x"fc",x"03",x"04",x"f2",x"07",x"1d"),
    (x"ed",x"e7",x"ec",x"e6",x"08",x"13",x"fd",x"1b",x"ea",x"14",x"18",x"e4",x"12",x"f0",x"ee",x"f0"),
    (x"f8",x"fb",x"f5",x"04",x"1e",x"f6",x"0b",x"e8",x"1d",x"ea",x"f7",x"11",x"09",x"15",x"fd",x"e4"),
    (x"fe",x"f1",x"17",x"1e",x"17",x"12",x"eb",x"1f",x"18",x"1a",x"1d",x"14",x"f5",x"e7",x"1c",x"1d"),
    (x"09",x"fe",x"e8",x"02",x"06",x"0e",x"fc",x"f0",x"00",x"e1",x"f3",x"f6",x"f9",x"f7",x"ea",x"ec"),
    (x"01",x"14",x"f3",x"05",x"e6",x"0c",x"1b",x"12",x"eb",x"f5",x"1d",x"e1",x"f0",x"f2",x"1b",x"0f"),
    (x"18",x"e8",x"14",x"09",x"e1",x"e3",x"f7",x"0b",x"02",x"0a",x"02",x"f5",x"f0",x"f9",x"0d",x"1d"),
    (x"e9",x"e6",x"fd",x"e1",x"fe",x"ff",x"eb",x"17",x"e2",x"1b",x"0d",x"f8",x"eb",x"12",x"0e",x"07"),
    (x"00",x"12",x"f4",x"08",x"03",x"14",x"1a",x"fc",x"f0",x"fd",x"e3",x"0d",x"f7",x"eb",x"15",x"04"),
    (x"e6",x"e6",x"e5",x"f1",x"04",x"14",x"ed",x"1a",x"10",x"00",x"07",x"f0",x"ec",x"e8",x"08",x"ee"),
    (x"f4",x"16",x"0d",x"19",x"20",x"eb",x"18",x"ee",x"0e",x"f4",x"1a",x"18",x"e2",x"20",x"05",x"f2"),
    (x"ea",x"ff",x"09",x"0f",x"19",x"e0",x"09",x"eb",x"17",x"e8",x"12",x"04",x"e5",x"0a",x"e2",x"08"),
    (x"1a",x"16",x"19",x"ef",x"1c",x"f7",x"0b",x"19",x"15",x"13",x"09",x"eb",x"e5",x"e6",x"e2",x"e8"),
    (x"f0",x"1c",x"1c",x"04",x"f6",x"1a",x"14",x"1e",x"f3",x"f1",x"0f",x"0f",x"e3",x"ed",x"e3",x"00"),
    (x"09",x"ea",x"0a",x"04",x"f0",x"0b",x"00",x"16",x"f9",x"1c",x"12",x"e3",x"f9",x"01",x"18",x"ff"),
    (x"f7",x"12",x"18",x"05",x"e5",x"0a",x"0e",x"19",x"ef",x"13",x"e6",x"06",x"1a",x"15",x"06",x"e2"),
    (x"ed",x"e1",x"02",x"e2",x"15",x"ef",x"16",x"0d",x"ec",x"f0",x"e6",x"0e",x"13",x"1f",x"f6",x"e9"),
    (x"0a",x"ff",x"11",x"f5",x"12",x"07",x"0b",x"18",x"0c",x"f8",x"fc",x"ff",x"fe",x"e3",x"17",x"ec"),
    (x"e3",x"0e",x"13",x"f8",x"e1",x"f9",x"f5",x"14",x"e8",x"0b",x"06",x"0a",x"fb",x"0e",x"15",x"f2"),
    (x"e3",x"eb",x"04",x"e5",x"e1",x"fb",x"f9",x"fc",x"e2",x"ee",x"03",x"ea",x"f4",x"e3",x"05",x"07"),
    (x"e5",x"fb",x"ef",x"f4",x"fe",x"fb",x"11",x"0d",x"ea",x"e0",x"0e",x"ff",x"ed",x"13",x"e9",x"19")
);
--------------------------------------------------------
--Relu_1
constant Relu_1_OUT_SIZE     :  integer := 64;
--------------------------------------------------------
constant OUT_WIDTH     :  integer := 64;
end package;