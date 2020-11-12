--------------------------------------------------------
-- This file is generated by Delirium - Sma-RTY SAS 
-- Wed Nov 11 12:26:15 2020
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
constant Conv_0_OUT_SIZE     :  integer := 128;
constant Conv_0_MULT_STYLE   :  string  := "dsp";
constant Conv_0_PIPELINE     :  boolean := True;
constant Conv_0_KERNEL_SIZE  :  integer := 1;
constant Conv_0_PADDING      :  boolean := FALSE;
constant Conv_0_STRIDE       :  positive:= 1;
constant Conv_0_BIAS_VALUE   :  data_array  (0 to Conv_0_OUT_SIZE - 1) := 
    (x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00");
constant Conv_0_KERNEL_VALUE :  data_matrix (0 to Conv_0_OUT_SIZE - 1 ,  0 to Conv_0_IN_SIZE * Conv_0_KERNEL_SIZE * Conv_0_KERNEL_SIZE - 1) := (
    (x"0c",x"0d",x"e1",x"eb",x"01",x"09",x"09",x"f3",x"1c",x"14",x"1f",x"ea",x"ea",x"0e",x"f3",x"10"),
    (x"f5",x"16",x"f7",x"f0",x"e8",x"fe",x"ec",x"20",x"f6",x"17",x"f1",x"01",x"e7",x"e8",x"ed",x"15"),
    (x"fd",x"1b",x"ea",x"e3",x"0e",x"ff",x"f1",x"e6",x"0a",x"fe",x"17",x"1c",x"1f",x"0b",x"e4",x"17"),
    (x"e1",x"ed",x"09",x"ee",x"05",x"16",x"0a",x"f6",x"e4",x"e7",x"f6",x"13",x"13",x"03",x"f5",x"fa"),
    (x"05",x"f2",x"0d",x"19",x"1a",x"15",x"f6",x"e6",x"1c",x"02",x"e4",x"e2",x"f2",x"e7",x"1b",x"17"),
    (x"fb",x"1d",x"e1",x"09",x"0e",x"fe",x"18",x"17",x"09",x"17",x"05",x"f1",x"f0",x"fd",x"1c",x"f6"),
    (x"e3",x"fe",x"ff",x"1f",x"f1",x"f9",x"fc",x"0d",x"e9",x"e2",x"fd",x"18",x"04",x"e1",x"1c",x"f3"),
    (x"02",x"13",x"1c",x"f3",x"13",x"02",x"0f",x"0a",x"16",x"0b",x"e1",x"14",x"0f",x"08",x"e2",x"17"),
    (x"1d",x"ea",x"10",x"ea",x"e5",x"ec",x"f1",x"ec",x"e1",x"f7",x"1c",x"01",x"1e",x"e2",x"fd",x"1c"),
    (x"1b",x"ee",x"ec",x"e9",x"e6",x"0a",x"1b",x"0b",x"04",x"ea",x"18",x"1d",x"15",x"04",x"1d",x"f0"),
    (x"e2",x"01",x"ff",x"f3",x"e6",x"0c",x"1b",x"05",x"ef",x"f8",x"0f",x"00",x"fc",x"1b",x"e8",x"03"),
    (x"f3",x"e2",x"03",x"0a",x"07",x"eb",x"0e",x"e1",x"ed",x"04",x"17",x"0e",x"03",x"fa",x"10",x"01"),
    (x"1e",x"0a",x"fa",x"1d",x"f1",x"18",x"f5",x"e2",x"eb",x"15",x"fb",x"f6",x"10",x"e9",x"ed",x"ef"),
    (x"15",x"12",x"f9",x"f7",x"e2",x"ed",x"13",x"12",x"04",x"fa",x"07",x"13",x"19",x"18",x"f3",x"00"),
    (x"19",x"e3",x"f7",x"e8",x"e4",x"f6",x"12",x"f8",x"01",x"1b",x"00",x"04",x"ec",x"03",x"e4",x"e2"),
    (x"0d",x"fb",x"08",x"e9",x"06",x"07",x"ed",x"02",x"1a",x"ec",x"fd",x"0f",x"1a",x"ff",x"1f",x"08"),
    (x"04",x"0f",x"03",x"f9",x"10",x"f9",x"06",x"01",x"0f",x"04",x"ea",x"03",x"00",x"1e",x"f6",x"ec"),
    (x"11",x"02",x"1b",x"f7",x"04",x"fc",x"f1",x"00",x"e1",x"18",x"09",x"07",x"05",x"1d",x"fe",x"12"),
    (x"f0",x"02",x"10",x"16",x"e6",x"0b",x"19",x"01",x"1c",x"09",x"fc",x"e5",x"f6",x"1d",x"17",x"ec"),
    (x"f5",x"18",x"fb",x"e8",x"fe",x"ee",x"ea",x"06",x"07",x"ec",x"e5",x"ee",x"e3",x"1a",x"00",x"f7"),
    (x"05",x"ec",x"e4",x"13",x"1f",x"15",x"17",x"18",x"00",x"01",x"1c",x"f4",x"03",x"07",x"1d",x"0d"),
    (x"ff",x"20",x"00",x"e7",x"1f",x"01",x"e2",x"fb",x"16",x"0f",x"0d",x"f5",x"02",x"e7",x"0e",x"ee"),
    (x"e6",x"0f",x"01",x"01",x"fb",x"e0",x"05",x"13",x"1b",x"f2",x"11",x"ff",x"11",x"07",x"14",x"fb"),
    (x"e6",x"18",x"02",x"e4",x"02",x"f9",x"e8",x"01",x"fc",x"13",x"f1",x"11",x"02",x"fc",x"0a",x"f3"),
    (x"f5",x"f4",x"f6",x"f0",x"e9",x"16",x"11",x"04",x"fd",x"ed",x"0b",x"fd",x"13",x"e2",x"1e",x"18"),
    (x"e1",x"fc",x"fa",x"19",x"f1",x"17",x"fa",x"19",x"03",x"1b",x"0a",x"e9",x"1b",x"0d",x"e2",x"06"),
    (x"e8",x"10",x"10",x"09",x"11",x"0b",x"e3",x"fa",x"0d",x"fd",x"05",x"ee",x"09",x"15",x"0d",x"1d"),
    (x"e7",x"f5",x"1e",x"f1",x"e1",x"1b",x"0a",x"1f",x"10",x"07",x"1f",x"18",x"1c",x"0d",x"f6",x"0d"),
    (x"04",x"1b",x"1d",x"f4",x"ea",x"f5",x"08",x"13",x"0c",x"03",x"e6",x"f6",x"19",x"07",x"f5",x"f9"),
    (x"17",x"08",x"17",x"06",x"e1",x"f5",x"ef",x"fd",x"1b",x"1c",x"0d",x"14",x"09",x"f0",x"12",x"08"),
    (x"20",x"18",x"f8",x"1a",x"e7",x"03",x"eb",x"ee",x"0f",x"e6",x"ef",x"ee",x"e4",x"01",x"17",x"ff"),
    (x"e9",x"e2",x"f4",x"ed",x"f6",x"15",x"e6",x"12",x"f8",x"e4",x"1f",x"0a",x"fb",x"04",x"15",x"14"),
    (x"ff",x"f3",x"e7",x"09",x"ff",x"0d",x"f1",x"f4",x"ee",x"f3",x"ec",x"17",x"fa",x"e9",x"ff",x"01"),
    (x"e7",x"0f",x"0b",x"e8",x"1a",x"01",x"f5",x"f7",x"e9",x"08",x"ee",x"02",x"fc",x"fd",x"eb",x"1b"),
    (x"e8",x"0c",x"05",x"1e",x"e8",x"f1",x"17",x"20",x"07",x"fc",x"f2",x"f8",x"e8",x"15",x"05",x"07"),
    (x"e3",x"1f",x"fe",x"0a",x"18",x"18",x"ec",x"16",x"ec",x"ea",x"f4",x"17",x"fa",x"1f",x"0a",x"fa"),
    (x"06",x"01",x"ef",x"fa",x"0c",x"1d",x"0e",x"f1",x"19",x"fa",x"fe",x"f2",x"0e",x"13",x"f3",x"0c"),
    (x"04",x"1a",x"f0",x"f9",x"05",x"1d",x"f6",x"16",x"f6",x"f2",x"ee",x"1d",x"0f",x"fd",x"09",x"fb"),
    (x"00",x"ef",x"e8",x"e0",x"09",x"11",x"16",x"e2",x"1d",x"16",x"06",x"fb",x"18",x"fd",x"15",x"10"),
    (x"1d",x"03",x"0e",x"0d",x"12",x"ff",x"f7",x"14",x"ee",x"01",x"0d",x"01",x"11",x"e6",x"ed",x"17"),
    (x"0d",x"f3",x"1d",x"08",x"0a",x"e2",x"14",x"f1",x"08",x"ed",x"1f",x"10",x"fc",x"fb",x"fe",x"f2"),
    (x"09",x"0e",x"f0",x"e1",x"fe",x"ec",x"fd",x"fa",x"00",x"12",x"fc",x"19",x"f3",x"1b",x"e1",x"e6"),
    (x"e9",x"e5",x"17",x"e2",x"17",x"05",x"f9",x"f2",x"ee",x"f2",x"f2",x"07",x"e8",x"ff",x"e6",x"0e"),
    (x"e5",x"f9",x"13",x"e9",x"11",x"e5",x"1c",x"fe",x"fa",x"14",x"f3",x"00",x"f1",x"fe",x"e8",x"19"),
    (x"fa",x"fd",x"1c",x"e2",x"f3",x"ec",x"0f",x"f7",x"1a",x"e8",x"fd",x"0b",x"0f",x"0f",x"00",x"fa"),
    (x"0d",x"02",x"0c",x"f9",x"e8",x"ee",x"14",x"ec",x"0f",x"e3",x"0c",x"ed",x"1d",x"08",x"0b",x"06"),
    (x"f9",x"07",x"17",x"f5",x"fd",x"02",x"e9",x"f2",x"19",x"ed",x"fc",x"1d",x"e0",x"0a",x"e6",x"06"),
    (x"1e",x"e2",x"08",x"e8",x"19",x"eb",x"1a",x"0c",x"e5",x"02",x"e2",x"1d",x"06",x"13",x"f7",x"f1"),
    (x"00",x"12",x"e0",x"e5",x"1e",x"0b",x"11",x"e2",x"e8",x"15",x"fc",x"f1",x"e1",x"fb",x"e2",x"eb"),
    (x"0c",x"0f",x"00",x"fa",x"07",x"e7",x"04",x"fa",x"fe",x"16",x"f5",x"e4",x"15",x"e5",x"1e",x"e3"),
    (x"e8",x"07",x"0f",x"02",x"0a",x"fb",x"f0",x"10",x"0f",x"0f",x"06",x"0b",x"16",x"00",x"00",x"e8"),
    (x"17",x"f2",x"ea",x"e1",x"f0",x"00",x"0e",x"f9",x"17",x"12",x"1b",x"f9",x"1d",x"e5",x"1e",x"16"),
    (x"00",x"12",x"12",x"1a",x"e6",x"fc",x"f5",x"ff",x"08",x"fb",x"e8",x"e6",x"ee",x"14",x"e3",x"20"),
    (x"f1",x"f6",x"f9",x"15",x"11",x"e3",x"05",x"1f",x"1d",x"13",x"e1",x"eb",x"fd",x"10",x"10",x"14"),
    (x"e9",x"0a",x"11",x"fe",x"eb",x"06",x"03",x"f9",x"1a",x"07",x"e1",x"ed",x"f8",x"fd",x"1b",x"0b"),
    (x"03",x"f0",x"e6",x"ef",x"1e",x"ec",x"0a",x"ed",x"0c",x"1e",x"f9",x"e1",x"e4",x"16",x"e7",x"1b"),
    (x"ef",x"f7",x"07",x"1f",x"f4",x"16",x"13",x"ed",x"0f",x"f6",x"e4",x"15",x"f9",x"f4",x"1c",x"f3"),
    (x"fc",x"12",x"19",x"fe",x"05",x"eb",x"17",x"f6",x"ef",x"e3",x"1e",x"06",x"ef",x"e6",x"04",x"02"),
    (x"eb",x"f9",x"f3",x"05",x"02",x"ea",x"f6",x"fd",x"fc",x"e7",x"19",x"08",x"08",x"fa",x"0c",x"18"),
    (x"f2",x"e1",x"fd",x"0c",x"e9",x"f3",x"f5",x"fd",x"10",x"f7",x"ea",x"e3",x"05",x"10",x"e3",x"20"),
    (x"03",x"0d",x"13",x"1a",x"04",x"fa",x"00",x"ec",x"eb",x"17",x"ff",x"1d",x"f4",x"e8",x"04",x"f0"),
    (x"0d",x"0e",x"f9",x"fc",x"02",x"e9",x"19",x"1a",x"1a",x"1a",x"e9",x"fc",x"ee",x"18",x"13",x"f2"),
    (x"e1",x"fe",x"14",x"eb",x"19",x"fd",x"e9",x"13",x"06",x"ef",x"eb",x"1d",x"0e",x"04",x"06",x"00"),
    (x"13",x"0f",x"0a",x"ec",x"00",x"ee",x"ef",x"e3",x"ef",x"01",x"fa",x"e6",x"e6",x"e6",x"f4",x"e1"),
    (x"1a",x"e3",x"f6",x"ee",x"17",x"20",x"e1",x"e1",x"12",x"f8",x"f3",x"f8",x"1c",x"1a",x"12",x"fa"),
    (x"07",x"fa",x"0a",x"0c",x"02",x"f6",x"f0",x"1d",x"07",x"01",x"f1",x"12",x"e3",x"ec",x"12",x"19"),
    (x"f3",x"08",x"ed",x"fc",x"ea",x"fb",x"12",x"03",x"16",x"e6",x"08",x"e4",x"ff",x"1e",x"1d",x"1b"),
    (x"e4",x"1e",x"00",x"04",x"ed",x"f0",x"13",x"fd",x"1e",x"02",x"1d",x"e3",x"f1",x"06",x"fa",x"01"),
    (x"14",x"ec",x"13",x"fa",x"02",x"1f",x"0c",x"f8",x"e3",x"1e",x"1d",x"04",x"e8",x"14",x"15",x"0f"),
    (x"e8",x"f6",x"0a",x"09",x"03",x"11",x"1e",x"e6",x"1e",x"ef",x"1d",x"ff",x"f3",x"e9",x"18",x"0c"),
    (x"ee",x"18",x"0d",x"f6",x"03",x"f4",x"15",x"f4",x"0f",x"ec",x"ed",x"e1",x"02",x"0d",x"fc",x"e1"),
    (x"17",x"ea",x"04",x"16",x"ec",x"14",x"15",x"f4",x"1b",x"0d",x"fa",x"fc",x"e5",x"0e",x"ec",x"e4"),
    (x"0a",x"0f",x"e1",x"f6",x"e7",x"1c",x"1c",x"08",x"00",x"e4",x"04",x"f7",x"e4",x"fd",x"0a",x"04"),
    (x"f6",x"e1",x"00",x"f9",x"ee",x"16",x"e8",x"13",x"14",x"09",x"f3",x"fd",x"04",x"fd",x"1f",x"ed"),
    (x"f1",x"11",x"e1",x"f5",x"0e",x"e2",x"f1",x"1f",x"f3",x"f1",x"ea",x"01",x"e7",x"fb",x"fe",x"13"),
    (x"f0",x"f5",x"04",x"1a",x"00",x"e1",x"00",x"e9",x"13",x"0a",x"10",x"e3",x"f4",x"fd",x"03",x"1f"),
    (x"fb",x"fa",x"f3",x"ff",x"16",x"e3",x"02",x"f4",x"e0",x"05",x"f8",x"17",x"f4",x"fa",x"e2",x"17"),
    (x"13",x"13",x"f6",x"09",x"16",x"f0",x"01",x"e1",x"08",x"14",x"fb",x"1e",x"19",x"04",x"e7",x"04"),
    (x"e4",x"1c",x"0d",x"03",x"e4",x"08",x"f6",x"f4",x"f6",x"f4",x"fe",x"1a",x"18",x"fc",x"f4",x"11"),
    (x"07",x"0b",x"fb",x"03",x"1d",x"f9",x"0e",x"20",x"e7",x"f7",x"fe",x"1e",x"04",x"08",x"ef",x"e8"),
    (x"16",x"e2",x"03",x"fc",x"15",x"fa",x"03",x"0f",x"18",x"1c",x"f6",x"f1",x"10",x"09",x"f7",x"fe"),
    (x"1e",x"02",x"e9",x"e8",x"e7",x"03",x"16",x"e1",x"01",x"11",x"fd",x"e2",x"ee",x"15",x"18",x"ff"),
    (x"12",x"01",x"f1",x"eb",x"ec",x"e1",x"1f",x"09",x"19",x"10",x"08",x"00",x"f5",x"fa",x"f4",x"f8"),
    (x"0d",x"19",x"0d",x"ec",x"0a",x"e7",x"f2",x"e4",x"f1",x"06",x"f0",x"e3",x"08",x"fb",x"02",x"f6"),
    (x"14",x"03",x"03",x"f4",x"1c",x"0a",x"14",x"fe",x"f9",x"1f",x"f3",x"f3",x"1e",x"0c",x"fa",x"ec"),
    (x"eb",x"e3",x"fa",x"19",x"01",x"ff",x"fc",x"f5",x"09",x"f4",x"e8",x"06",x"10",x"fd",x"fd",x"12"),
    (x"fb",x"1f",x"ed",x"07",x"17",x"f7",x"10",x"e7",x"f9",x"f2",x"f1",x"11",x"0c",x"ed",x"1f",x"0b"),
    (x"0f",x"f9",x"ff",x"e2",x"00",x"1d",x"19",x"03",x"07",x"f2",x"f7",x"1d",x"f7",x"e8",x"e5",x"ee"),
    (x"07",x"ef",x"1c",x"1f",x"f4",x"03",x"f9",x"04",x"f3",x"18",x"0f",x"04",x"03",x"1f",x"02",x"f2"),
    (x"ee",x"03",x"1c",x"e8",x"ea",x"eb",x"f4",x"e5",x"1d",x"17",x"08",x"17",x"fb",x"f9",x"fa",x"e8"),
    (x"ec",x"0b",x"07",x"02",x"1f",x"15",x"04",x"f7",x"09",x"19",x"00",x"e2",x"13",x"f6",x"f1",x"eb"),
    (x"0b",x"02",x"01",x"e7",x"eb",x"12",x"16",x"05",x"09",x"0f",x"f9",x"fa",x"ff",x"fa",x"0f",x"e9"),
    (x"fe",x"e5",x"04",x"1a",x"06",x"e7",x"1b",x"15",x"1c",x"ff",x"f3",x"0f",x"08",x"08",x"11",x"19"),
    (x"15",x"ee",x"f0",x"ed",x"12",x"ec",x"f3",x"fa",x"fd",x"f6",x"17",x"0e",x"f6",x"09",x"1f",x"12"),
    (x"19",x"f8",x"03",x"13",x"04",x"f1",x"f7",x"1f",x"0e",x"06",x"eb",x"f9",x"0d",x"17",x"e2",x"ea"),
    (x"f1",x"15",x"e9",x"04",x"18",x"13",x"f2",x"e4",x"e0",x"e7",x"e8",x"03",x"e4",x"00",x"02",x"eb"),
    (x"ee",x"fe",x"e6",x"16",x"07",x"f7",x"fa",x"ee",x"f9",x"04",x"f1",x"fa",x"17",x"eb",x"17",x"e8"),
    (x"1f",x"e1",x"e5",x"05",x"e3",x"e8",x"0b",x"e8",x"ff",x"16",x"19",x"f7",x"18",x"ed",x"f0",x"18"),
    (x"11",x"00",x"07",x"1f",x"1f",x"ed",x"fa",x"e8",x"e4",x"ee",x"f7",x"11",x"f4",x"ea",x"16",x"ed"),
    (x"00",x"06",x"0e",x"f2",x"0e",x"0c",x"1a",x"16",x"05",x"20",x"ec",x"f3",x"f4",x"17",x"12",x"1f"),
    (x"ef",x"15",x"0d",x"1a",x"f1",x"e7",x"16",x"16",x"18",x"08",x"e6",x"ed",x"08",x"f7",x"fd",x"f4"),
    (x"f2",x"e7",x"1a",x"e6",x"1e",x"ea",x"e6",x"e4",x"ee",x"10",x"ec",x"18",x"00",x"fd",x"02",x"1f"),
    (x"e9",x"e4",x"0f",x"1a",x"fe",x"04",x"fb",x"fa",x"1b",x"ed",x"fc",x"0a",x"0f",x"0a",x"e8",x"01"),
    (x"e7",x"0a",x"0e",x"e7",x"e1",x"05",x"fe",x"e9",x"08",x"e3",x"09",x"ea",x"13",x"e1",x"ed",x"07"),
    (x"1a",x"1c",x"f0",x"04",x"fc",x"05",x"0d",x"ff",x"ff",x"06",x"1e",x"f0",x"07",x"10",x"f5",x"1c"),
    (x"05",x"06",x"1a",x"f7",x"00",x"e9",x"12",x"ef",x"ec",x"fb",x"12",x"e4",x"13",x"16",x"f2",x"ed"),
    (x"f7",x"1d",x"03",x"e9",x"1a",x"05",x"f7",x"fb",x"05",x"0e",x"e2",x"e4",x"1a",x"14",x"0d",x"f3"),
    (x"e2",x"15",x"fb",x"ec",x"f6",x"18",x"e5",x"1a",x"f4",x"0d",x"06",x"13",x"e4",x"15",x"fb",x"f8"),
    (x"eb",x"f3",x"1e",x"1f",x"ee",x"08",x"0b",x"1e",x"0e",x"19",x"f6",x"00",x"ed",x"04",x"09",x"01"),
    (x"1b",x"0f",x"ec",x"fb",x"10",x"07",x"14",x"04",x"f5",x"1b",x"0b",x"e9",x"e7",x"0c",x"f8",x"e7"),
    (x"05",x"e6",x"fd",x"03",x"17",x"00",x"fd",x"1a",x"0d",x"f3",x"0e",x"fb",x"1d",x"fb",x"15",x"14"),
    (x"12",x"0f",x"ff",x"f4",x"ed",x"16",x"0b",x"0a",x"1e",x"07",x"f3",x"1f",x"ec",x"e9",x"ed",x"ed"),
    (x"07",x"0e",x"1f",x"f2",x"fc",x"06",x"f9",x"0f",x"e6",x"1e",x"fe",x"17",x"fb",x"fd",x"ed",x"16"),
    (x"e1",x"f8",x"00",x"07",x"00",x"ee",x"fe",x"fc",x"ea",x"16",x"1a",x"04",x"11",x"06",x"02",x"e3"),
    (x"e2",x"13",x"f7",x"e3",x"1d",x"f0",x"01",x"fb",x"1c",x"18",x"08",x"e2",x"fe",x"e2",x"ff",x"07"),
    (x"0d",x"05",x"1c",x"f6",x"fe",x"02",x"e4",x"f1",x"0d",x"ee",x"f2",x"f0",x"f3",x"fe",x"e3",x"17"),
    (x"e6",x"0e",x"13",x"02",x"0f",x"e8",x"09",x"e5",x"f1",x"02",x"fd",x"e2",x"ec",x"1e",x"08",x"05"),
    (x"e3",x"e7",x"f8",x"0d",x"00",x"15",x"ef",x"e9",x"17",x"1a",x"04",x"e9",x"04",x"0d",x"e7",x"0e"),
    (x"1d",x"0d",x"09",x"17",x"e9",x"f0",x"06",x"0b",x"03",x"12",x"06",x"1e",x"ef",x"11",x"e6",x"05"),
    (x"ed",x"13",x"10",x"fc",x"02",x"f7",x"e1",x"ef",x"0d",x"ff",x"19",x"05",x"fe",x"ea",x"f1",x"18"),
    (x"f4",x"f4",x"19",x"e1",x"e4",x"e7",x"f1",x"e2",x"e4",x"f2",x"1a",x"0d",x"15",x"f6",x"01",x"1d"),
    (x"f7",x"09",x"fb",x"e2",x"fc",x"f4",x"fb",x"ec",x"02",x"13",x"19",x"f5",x"0e",x"0c",x"e5",x"14"),
    (x"f3",x"0f",x"e7",x"f4",x"1c",x"1d",x"f0",x"19",x"11",x"13",x"1d",x"ea",x"ec",x"0e",x"ef",x"07"),
    (x"00",x"eb",x"e1",x"ee",x"05",x"e6",x"ec",x"07",x"f9",x"ff",x"ea",x"fe",x"03",x"f8",x"e3",x"e1"),
    (x"15",x"0b",x"e4",x"e8",x"0f",x"e3",x"02",x"ee",x"fb",x"f6",x"ee",x"13",x"fb",x"eb",x"1c",x"04"),
    (x"16",x"f0",x"11",x"ef",x"e2",x"02",x"10",x"eb",x"fe",x"14",x"f7",x"f2",x"04",x"0d",x"00",x"08"),
    (x"ea",x"e1",x"12",x"1f",x"0d",x"10",x"14",x"f3",x"ed",x"ec",x"ef",x"08",x"f0",x"ee",x"f1",x"00"),
    (x"ef",x"0c",x"fd",x"1b",x"09",x"ec",x"06",x"fd",x"f2",x"e1",x"e2",x"1a",x"e8",x"12",x"1c",x"01")
);
--------------------------------------------------------
--Relu_1
constant Relu_1_OUT_SIZE     :  integer := 128;
--------------------------------------------------------
constant OUT_WIDTH     :  integer := 128;
end package;