--------------------------------------------------------
-- This file is generated by Delirium - Sma-RTY SAS 
-- Wed Nov 11 12:33:59 2020
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
constant Conv_0_OUT_SIZE     :  integer := 128;
constant Conv_0_MULT_STYLE   :  string  := "dsp";
constant Conv_0_PIPELINE     :  boolean := True;
constant Conv_0_KERNEL_SIZE  :  integer := 3;
constant Conv_0_PADDING      :  boolean := FALSE;
constant Conv_0_STRIDE       :  positive:= 1;
constant Conv_0_BIAS_VALUE   :  data_array  (0 to Conv_0_OUT_SIZE - 1) := 
    (x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00");
constant Conv_0_KERNEL_VALUE :  data_matrix (0 to Conv_0_OUT_SIZE - 1 ,  0 to Conv_0_IN_SIZE * Conv_0_KERNEL_SIZE * Conv_0_KERNEL_SIZE - 1) := (
    (x"0e",x"e8",x"f8",x"0f",x"f6",x"f0",x"ea",x"04",x"ff",x"01",x"09",x"f1",x"fc",x"0f",x"fe",x"ee",x"05",x"17",x"01",x"f0",x"0d",x"0a",x"03",x"14",x"f2",x"06",x"f3"),
    (x"f1",x"f0",x"0b",x"f5",x"0d",x"14",x"f0",x"01",x"e9",x"fc",x"e9",x"fd",x"ec",x"fe",x"04",x"ff",x"00",x"0f",x"ef",x"f9",x"f3",x"03",x"04",x"13",x"fb",x"04",x"fe"),
    (x"ed",x"12",x"fb",x"f4",x"f5",x"01",x"0b",x"01",x"0a",x"ef",x"0e",x"0f",x"ff",x"07",x"14",x"f2",x"13",x"fb",x"0c",x"fb",x"f7",x"f7",x"eb",x"0c",x"f7",x"fc",x"07"),
    (x"04",x"f3",x"f6",x"0f",x"f4",x"0b",x"07",x"0f",x"ed",x"07",x"ee",x"ea",x"0f",x"18",x"f3",x"00",x"ee",x"fe",x"f4",x"f9",x"ea",x"04",x"ff",x"f4",x"0a",x"07",x"e8"),
    (x"0d",x"f3",x"f5",x"eb",x"ed",x"ef",x"04",x"18",x"ec",x"f0",x"ea",x"fc",x"05",x"10",x"f2",x"0a",x"ea",x"16",x"13",x"04",x"f9",x"0e",x"ed",x"11",x"04",x"f1",x"fe"),
    (x"0c",x"11",x"14",x"00",x"0a",x"03",x"03",x"f8",x"0d",x"ee",x"f9",x"05",x"16",x"f2",x"17",x"13",x"04",x"08",x"12",x"03",x"05",x"e9",x"0e",x"13",x"10",x"0f",x"fb"),
    (x"fb",x"e9",x"11",x"07",x"0b",x"0d",x"f2",x"10",x"12",x"f9",x"0c",x"fe",x"0a",x"f3",x"03",x"0a",x"0f",x"15",x"f5",x"13",x"0b",x"18",x"04",x"11",x"04",x"ef",x"fe"),
    (x"06",x"00",x"09",x"06",x"07",x"0e",x"15",x"10",x"0f",x"01",x"00",x"11",x"05",x"07",x"10",x"02",x"06",x"ff",x"0d",x"03",x"05",x"15",x"fd",x"ed",x"f4",x"e8",x"0d"),
    (x"02",x"0a",x"0a",x"11",x"e9",x"fa",x"f9",x"16",x"17",x"17",x"f8",x"f0",x"05",x"14",x"08",x"f4",x"fb",x"ee",x"ed",x"ef",x"0b",x"ff",x"00",x"f1",x"11",x"17",x"0e"),
    (x"11",x"ee",x"f8",x"04",x"0f",x"ee",x"0e",x"fd",x"f0",x"f1",x"11",x"ef",x"f0",x"e9",x"fd",x"03",x"fa",x"0d",x"0c",x"18",x"f0",x"f6",x"0f",x"ed",x"10",x"18",x"09"),
    (x"f6",x"eb",x"12",x"0d",x"f5",x"f5",x"f2",x"fd",x"18",x"ec",x"ee",x"f9",x"ea",x"00",x"fd",x"ee",x"12",x"13",x"f7",x"01",x"fa",x"14",x"f6",x"08",x"ed",x"05",x"e8"),
    (x"ea",x"16",x"ec",x"f3",x"0d",x"f8",x"14",x"18",x"0f",x"17",x"0f",x"ff",x"fd",x"ef",x"f8",x"ef",x"f3",x"f1",x"0d",x"0e",x"ed",x"0e",x"ea",x"03",x"13",x"0b",x"f1"),
    (x"13",x"f4",x"0f",x"ee",x"ef",x"e8",x"f3",x"10",x"0e",x"e8",x"fa",x"01",x"18",x"04",x"0c",x"fb",x"f1",x"04",x"f0",x"ea",x"04",x"10",x"e9",x"10",x"09",x"fd",x"0a"),
    (x"18",x"e8",x"04",x"04",x"0c",x"17",x"0f",x"ea",x"f0",x"12",x"f2",x"00",x"10",x"fe",x"eb",x"16",x"f0",x"f6",x"10",x"f0",x"16",x"ee",x"05",x"f8",x"f8",x"13",x"06"),
    (x"f5",x"05",x"02",x"0c",x"f3",x"01",x"f2",x"17",x"fb",x"e8",x"f7",x"eb",x"12",x"fb",x"14",x"ff",x"0a",x"ff",x"fb",x"fa",x"fe",x"fd",x"eb",x"10",x"02",x"09",x"07"),
    (x"f1",x"10",x"11",x"f2",x"f5",x"eb",x"16",x"17",x"f8",x"00",x"f7",x"f6",x"15",x"04",x"fd",x"10",x"ff",x"00",x"09",x"05",x"f1",x"fb",x"0e",x"ef",x"ec",x"e8",x"f0"),
    (x"ff",x"0c",x"17",x"0d",x"10",x"fe",x"eb",x"16",x"fe",x"15",x"ed",x"f5",x"fa",x"04",x"f4",x"e8",x"06",x"fa",x"0c",x"ee",x"15",x"eb",x"f7",x"f1",x"0b",x"ea",x"0d"),
    (x"0d",x"ef",x"06",x"e8",x"ed",x"09",x"f5",x"f7",x"ee",x"17",x"04",x"13",x"ff",x"05",x"f9",x"f4",x"fc",x"f9",x"15",x"f6",x"eb",x"02",x"0f",x"f4",x"0a",x"02",x"f5"),
    (x"0f",x"ec",x"fc",x"0a",x"f5",x"0e",x"ef",x"08",x"e8",x"fa",x"0a",x"16",x"14",x"f4",x"f9",x"fe",x"f1",x"fe",x"fd",x"e9",x"06",x"02",x"06",x"ed",x"f0",x"eb",x"16"),
    (x"fa",x"0b",x"02",x"f5",x"04",x"03",x"17",x"fe",x"05",x"08",x"05",x"ef",x"e9",x"fc",x"f1",x"f1",x"09",x"0c",x"17",x"04",x"0f",x"f7",x"09",x"14",x"0f",x"0a",x"06"),
    (x"f8",x"16",x"08",x"ff",x"14",x"01",x"f5",x"f4",x"14",x"f5",x"12",x"0e",x"18",x"0e",x"15",x"fa",x"0d",x"08",x"ec",x"02",x"16",x"ea",x"f6",x"08",x"07",x"13",x"0a"),
    (x"ef",x"10",x"08",x"f8",x"f0",x"05",x"06",x"06",x"f1",x"18",x"01",x"f1",x"10",x"eb",x"ee",x"0c",x"f1",x"0b",x"e8",x"ec",x"f0",x"07",x"f6",x"0d",x"f2",x"13",x"ed"),
    (x"0e",x"ff",x"18",x"fd",x"11",x"ee",x"15",x"ee",x"06",x"fd",x"01",x"02",x"05",x"17",x"0e",x"fc",x"02",x"13",x"11",x"02",x"13",x"f2",x"06",x"05",x"06",x"05",x"17"),
    (x"08",x"11",x"0c",x"ff",x"13",x"ea",x"11",x"e9",x"f5",x"0f",x"18",x"ee",x"12",x"f3",x"18",x"17",x"0a",x"fb",x"16",x"ec",x"16",x"fa",x"0b",x"ee",x"06",x"fe",x"f3"),
    (x"02",x"f9",x"ff",x"f3",x"09",x"12",x"f6",x"ea",x"fe",x"ec",x"14",x"ff",x"0f",x"00",x"f0",x"0f",x"11",x"04",x"06",x"12",x"fd",x"03",x"ef",x"11",x"08",x"ea",x"f3"),
    (x"05",x"f1",x"0f",x"18",x"08",x"f4",x"f0",x"ff",x"11",x"05",x"f0",x"08",x"fd",x"f2",x"f5",x"12",x"15",x"f4",x"f9",x"0f",x"f2",x"05",x"06",x"0a",x"fb",x"0d",x"12"),
    (x"11",x"15",x"02",x"03",x"fa",x"f8",x"10",x"f1",x"12",x"13",x"ec",x"f2",x"01",x"09",x"fe",x"11",x"0a",x"f4",x"07",x"fa",x"fb",x"f1",x"f9",x"f7",x"f8",x"f2",x"0d"),
    (x"0e",x"f3",x"17",x"0d",x"f2",x"ff",x"05",x"f5",x"04",x"fa",x"f6",x"f6",x"05",x"08",x"fd",x"fc",x"18",x"05",x"f4",x"13",x"f7",x"fa",x"0f",x"fd",x"e9",x"f8",x"f8"),
    (x"f5",x"ea",x"0d",x"0c",x"0c",x"08",x"f3",x"ec",x"0c",x"0e",x"0c",x"06",x"16",x"07",x"ea",x"07",x"17",x"0e",x"fe",x"f3",x"f8",x"00",x"0c",x"eb",x"03",x"fc",x"ff"),
    (x"0c",x"08",x"17",x"ea",x"eb",x"0c",x"ea",x"f5",x"05",x"15",x"fd",x"07",x"15",x"0e",x"ed",x"0f",x"f2",x"ef",x"eb",x"04",x"04",x"02",x"f6",x"02",x"fd",x"f0",x"0f"),
    (x"09",x"ef",x"e9",x"05",x"17",x"f3",x"f6",x"ff",x"18",x"10",x"0b",x"0c",x"03",x"04",x"f4",x"18",x"f6",x"f6",x"0a",x"fa",x"17",x"fe",x"14",x"17",x"f9",x"ed",x"08"),
    (x"f9",x"fb",x"ed",x"11",x"17",x"f8",x"14",x"f6",x"14",x"f3",x"0b",x"01",x"f9",x"f2",x"fe",x"fd",x"0f",x"02",x"e9",x"10",x"09",x"e8",x"09",x"fe",x"fb",x"f9",x"f7"),
    (x"fa",x"f3",x"f3",x"f2",x"f0",x"09",x"f0",x"fe",x"fe",x"f9",x"16",x"06",x"f7",x"14",x"fe",x"e9",x"0e",x"fe",x"14",x"0c",x"f6",x"12",x"f0",x"16",x"f7",x"0f",x"02"),
    (x"f5",x"01",x"0a",x"00",x"fa",x"03",x"fa",x"01",x"11",x"12",x"eb",x"f9",x"12",x"fc",x"02",x"08",x"f7",x"0b",x"12",x"11",x"eb",x"fd",x"00",x"11",x"17",x"ef",x"ff"),
    (x"05",x"00",x"f9",x"0d",x"fb",x"f7",x"10",x"f2",x"fa",x"18",x"16",x"17",x"07",x"0f",x"05",x"17",x"03",x"ea",x"07",x"fe",x"0c",x"08",x"02",x"fb",x"e9",x"08",x"12"),
    (x"fd",x"06",x"fc",x"fc",x"06",x"14",x"06",x"05",x"f4",x"f8",x"f3",x"02",x"fe",x"05",x"08",x"f4",x"ed",x"f6",x"07",x"01",x"ff",x"e9",x"ea",x"17",x"f4",x"ed",x"08"),
    (x"f0",x"f0",x"ef",x"16",x"07",x"f0",x"17",x"0f",x"15",x"f6",x"fe",x"f7",x"01",x"17",x"fc",x"06",x"fa",x"0c",x"18",x"ff",x"eb",x"10",x"f2",x"f2",x"01",x"03",x"03"),
    (x"fb",x"f5",x"f2",x"09",x"f2",x"02",x"08",x"f0",x"03",x"0c",x"fa",x"f8",x"03",x"e8",x"fd",x"0f",x"ea",x"ee",x"f4",x"f5",x"14",x"02",x"05",x"fd",x"11",x"0c",x"f1"),
    (x"09",x"fb",x"05",x"18",x"ef",x"02",x"eb",x"01",x"ec",x"08",x"f8",x"09",x"f3",x"f7",x"0e",x"01",x"05",x"13",x"f5",x"13",x"04",x"ef",x"fa",x"f2",x"ee",x"0f",x"14"),
    (x"f2",x"0a",x"fd",x"ee",x"f1",x"06",x"ec",x"0a",x"ff",x"f6",x"0a",x"e9",x"06",x"00",x"f5",x"e8",x"eb",x"e8",x"ef",x"04",x"f3",x"10",x"17",x"fd",x"ed",x"0e",x"f4"),
    (x"fc",x"00",x"04",x"10",x"f7",x"eb",x"07",x"10",x"f7",x"fe",x"01",x"00",x"17",x"fe",x"ef",x"0a",x"f3",x"18",x"07",x"0e",x"01",x"f2",x"ef",x"fd",x"e8",x"eb",x"0f"),
    (x"00",x"12",x"ed",x"f2",x"18",x"0d",x"f5",x"e9",x"02",x"f7",x"04",x"03",x"0f",x"f7",x"e9",x"14",x"15",x"f3",x"fc",x"09",x"0f",x"ec",x"ee",x"06",x"e8",x"fa",x"ea"),
    (x"fa",x"ff",x"eb",x"e8",x"e8",x"11",x"13",x"02",x"f0",x"e8",x"0e",x"f5",x"fc",x"04",x"11",x"03",x"eb",x"f5",x"fe",x"eb",x"fb",x"fe",x"f7",x"0b",x"f3",x"ef",x"01"),
    (x"11",x"e9",x"f3",x"16",x"f2",x"f1",x"0b",x"15",x"10",x"e8",x"ec",x"ed",x"fc",x"11",x"10",x"10",x"15",x"f8",x"12",x"02",x"0a",x"18",x"0a",x"09",x"05",x"15",x"f9"),
    (x"08",x"f5",x"f3",x"0b",x"07",x"07",x"10",x"fa",x"f8",x"05",x"ec",x"ee",x"0e",x"f1",x"01",x"f2",x"fc",x"e8",x"0e",x"0d",x"fe",x"11",x"0a",x"00",x"02",x"fa",x"ed"),
    (x"12",x"f2",x"f8",x"15",x"ec",x"03",x"0b",x"08",x"07",x"15",x"0f",x"11",x"08",x"ff",x"0c",x"fc",x"04",x"f7",x"ee",x"0b",x"12",x"03",x"02",x"0d",x"06",x"12",x"14"),
    (x"0c",x"15",x"03",x"04",x"fc",x"fe",x"06",x"ff",x"17",x"0c",x"15",x"fc",x"f5",x"fb",x"0b",x"06",x"fd",x"00",x"0d",x"f7",x"10",x"0f",x"ea",x"15",x"18",x"f3",x"f5"),
    (x"00",x"12",x"0c",x"12",x"0b",x"fc",x"0a",x"03",x"fe",x"12",x"0b",x"ec",x"f6",x"eb",x"ea",x"ff",x"f6",x"f0",x"02",x"f3",x"ed",x"09",x"f3",x"f5",x"01",x"ed",x"01"),
    (x"ea",x"f1",x"e8",x"14",x"f6",x"f5",x"f6",x"f3",x"0e",x"ec",x"14",x"0d",x"0b",x"eb",x"fb",x"f6",x"f1",x"04",x"0a",x"0d",x"07",x"fb",x"01",x"f0",x"e9",x"06",x"f2"),
    (x"ee",x"0f",x"08",x"e9",x"01",x"fd",x"08",x"f0",x"0b",x"12",x"ed",x"0a",x"00",x"ed",x"fd",x"0a",x"f9",x"05",x"09",x"0e",x"f5",x"f1",x"f9",x"17",x"14",x"07",x"0a"),
    (x"fe",x"04",x"03",x"16",x"12",x"06",x"11",x"01",x"fb",x"ec",x"eb",x"17",x"ea",x"ea",x"15",x"fe",x"f4",x"f9",x"09",x"0f",x"ea",x"e8",x"f7",x"11",x"fb",x"0d",x"0f"),
    (x"f0",x"0b",x"07",x"10",x"11",x"f4",x"e9",x"16",x"13",x"fe",x"ee",x"0f",x"fa",x"0b",x"e8",x"f1",x"f6",x"e9",x"f3",x"05",x"ea",x"02",x"e9",x"e9",x"f7",x"ec",x"02"),
    (x"01",x"e9",x"12",x"fd",x"07",x"13",x"fc",x"17",x"f4",x"0f",x"0b",x"f1",x"f4",x"18",x"09",x"f6",x"10",x"f8",x"eb",x"0e",x"18",x"ed",x"e8",x"f6",x"17",x"ea",x"15"),
    (x"ff",x"ff",x"f7",x"f2",x"10",x"ef",x"ee",x"f4",x"0e",x"18",x"18",x"f5",x"fd",x"ee",x"07",x"04",x"ff",x"f2",x"f0",x"0c",x"f6",x"f2",x"ee",x"05",x"00",x"fd",x"f5"),
    (x"f1",x"0f",x"12",x"01",x"f7",x"11",x"fe",x"09",x"f1",x"f6",x"fe",x"ea",x"03",x"eb",x"f4",x"f0",x"fb",x"ea",x"07",x"0f",x"e8",x"0b",x"0f",x"fb",x"f4",x"ed",x"e8"),
    (x"0b",x"ed",x"e8",x"f2",x"17",x"02",x"05",x"00",x"fb",x"fb",x"f4",x"09",x"ed",x"06",x"f8",x"05",x"00",x"f8",x"f3",x"fb",x"17",x"07",x"02",x"ff",x"f5",x"ed",x"0b"),
    (x"13",x"13",x"12",x"05",x"f5",x"fa",x"fb",x"0f",x"0a",x"ea",x"16",x"fd",x"16",x"0b",x"15",x"03",x"08",x"13",x"12",x"fe",x"fc",x"ec",x"11",x"08",x"0a",x"fc",x"13"),
    (x"01",x"f9",x"07",x"f4",x"ed",x"ee",x"eb",x"14",x"f1",x"03",x"f5",x"11",x"ed",x"13",x"12",x"09",x"02",x"fb",x"0d",x"05",x"ed",x"00",x"08",x"f0",x"07",x"15",x"f6"),
    (x"17",x"ee",x"f6",x"f3",x"f7",x"0f",x"f1",x"12",x"0d",x"09",x"16",x"02",x"0a",x"11",x"0c",x"f8",x"fa",x"f9",x"f8",x"14",x"14",x"ed",x"14",x"00",x"fa",x"10",x"f4"),
    (x"0e",x"fe",x"08",x"0d",x"0c",x"f0",x"f9",x"f4",x"ed",x"fc",x"f0",x"fa",x"03",x"09",x"fe",x"f2",x"11",x"f1",x"06",x"f7",x"10",x"13",x"f4",x"16",x"fd",x"fd",x"f9"),
    (x"0a",x"0c",x"17",x"14",x"f9",x"01",x"f3",x"fe",x"f8",x"f1",x"11",x"11",x"07",x"14",x"ed",x"14",x"05",x"ed",x"f0",x"ec",x"fd",x"0f",x"14",x"fb",x"0c",x"16",x"09"),
    (x"0c",x"04",x"ef",x"0e",x"ed",x"f7",x"ee",x"02",x"11",x"06",x"e9",x"ec",x"fc",x"14",x"ed",x"17",x"f0",x"01",x"fd",x"0d",x"11",x"f8",x"f3",x"e9",x"f4",x"15",x"12"),
    (x"f7",x"0a",x"08",x"f3",x"17",x"fd",x"f2",x"13",x"0b",x"12",x"f5",x"04",x"0b",x"17",x"f9",x"eb",x"f7",x"f2",x"18",x"f2",x"01",x"07",x"04",x"00",x"08",x"f1",x"ef"),
    (x"00",x"f5",x"10",x"06",x"f4",x"13",x"e9",x"12",x"fd",x"f1",x"18",x"ec",x"e8",x"17",x"13",x"05",x"00",x"0e",x"fd",x"18",x"f6",x"0f",x"f9",x"18",x"fd",x"14",x"05"),
    (x"e8",x"f2",x"fe",x"10",x"12",x"14",x"11",x"f6",x"ea",x"02",x"08",x"0f",x"07",x"12",x"eb",x"f2",x"ee",x"ff",x"ed",x"08",x"01",x"0f",x"09",x"05",x"0b",x"f4",x"0f"),
    (x"e9",x"f1",x"f0",x"0b",x"f6",x"f0",x"0f",x"0f",x"01",x"ec",x"fb",x"f4",x"17",x"02",x"0e",x"06",x"f1",x"0a",x"17",x"18",x"16",x"09",x"f1",x"f6",x"09",x"18",x"0d"),
    (x"06",x"01",x"ee",x"ff",x"fa",x"f3",x"ee",x"09",x"13",x"f6",x"e8",x"f1",x"fc",x"fb",x"05",x"17",x"0b",x"11",x"ea",x"16",x"fd",x"ec",x"05",x"04",x"0c",x"04",x"0e"),
    (x"03",x"eb",x"07",x"12",x"06",x"fc",x"ff",x"fd",x"e9",x"16",x"01",x"0c",x"15",x"14",x"f7",x"0a",x"03",x"ef",x"f6",x"10",x"0e",x"0a",x"0e",x"f6",x"f8",x"ee",x"08"),
    (x"ed",x"17",x"f4",x"00",x"04",x"07",x"ff",x"fa",x"ef",x"06",x"fe",x"18",x"0e",x"0b",x"00",x"fc",x"0d",x"06",x"08",x"f4",x"ea",x"ec",x"f5",x"10",x"08",x"10",x"ff"),
    (x"eb",x"18",x"12",x"12",x"00",x"18",x"fc",x"13",x"0a",x"12",x"f5",x"ee",x"09",x"0e",x"e9",x"03",x"0f",x"f9",x"ec",x"ed",x"f9",x"0a",x"02",x"f2",x"13",x"e9",x"eb"),
    (x"f0",x"00",x"05",x"ff",x"12",x"17",x"16",x"02",x"17",x"0b",x"02",x"01",x"f8",x"f3",x"fd",x"16",x"11",x"e9",x"10",x"f8",x"03",x"13",x"0c",x"0c",x"10",x"00",x"f4"),
    (x"f8",x"13",x"fc",x"0f",x"fb",x"f5",x"0f",x"ea",x"ec",x"0f",x"13",x"f1",x"0b",x"e8",x"16",x"fd",x"15",x"f2",x"0e",x"03",x"15",x"03",x"ee",x"f6",x"f9",x"08",x"01"),
    (x"f7",x"f3",x"0d",x"ec",x"f7",x"09",x"14",x"08",x"05",x"03",x"05",x"ec",x"fe",x"ef",x"04",x"fe",x"04",x"01",x"f9",x"00",x"12",x"e9",x"07",x"e9",x"0e",x"f6",x"12"),
    (x"07",x"05",x"09",x"07",x"fd",x"f8",x"ee",x"fb",x"16",x"02",x"fa",x"fe",x"f9",x"16",x"05",x"00",x"f6",x"14",x"00",x"fd",x"ee",x"0b",x"0a",x"ec",x"ef",x"ec",x"13"),
    (x"03",x"f5",x"f8",x"12",x"0e",x"fe",x"17",x"09",x"08",x"f5",x"f3",x"f7",x"e8",x"12",x"10",x"16",x"f1",x"18",x"ef",x"00",x"fc",x"e8",x"08",x"f9",x"03",x"fd",x"eb"),
    (x"14",x"fc",x"fd",x"18",x"00",x"eb",x"16",x"0d",x"13",x"ef",x"05",x"ed",x"e9",x"fb",x"02",x"f5",x"fd",x"0a",x"13",x"f7",x"15",x"ff",x"f6",x"fe",x"fa",x"02",x"04"),
    (x"03",x"0d",x"13",x"0a",x"f7",x"f6",x"07",x"07",x"02",x"09",x"17",x"11",x"ed",x"f6",x"06",x"fe",x"0f",x"ef",x"04",x"e8",x"fa",x"0e",x"fe",x"fb",x"fc",x"13",x"f1"),
    (x"ff",x"09",x"07",x"0b",x"17",x"09",x"f8",x"fc",x"ff",x"ec",x"05",x"04",x"04",x"04",x"f2",x"03",x"f0",x"f4",x"17",x"f3",x"02",x"f4",x"f6",x"14",x"11",x"17",x"f1"),
    (x"f4",x"f0",x"0f",x"f9",x"f7",x"fd",x"ee",x"ff",x"0e",x"05",x"06",x"e8",x"12",x"e9",x"14",x"05",x"fe",x"15",x"16",x"04",x"01",x"07",x"01",x"17",x"e8",x"15",x"ee"),
    (x"f4",x"18",x"0e",x"17",x"15",x"fb",x"f5",x"0d",x"f0",x"f6",x"ec",x"f0",x"01",x"05",x"eb",x"08",x"f5",x"fb",x"fc",x"f1",x"00",x"05",x"e9",x"0d",x"f0",x"00",x"03"),
    (x"05",x"08",x"01",x"f1",x"02",x"fb",x"13",x"f5",x"e9",x"0f",x"09",x"18",x"0d",x"01",x"16",x"11",x"0f",x"00",x"0a",x"02",x"f1",x"f0",x"e9",x"f2",x"f8",x"f0",x"0a"),
    (x"f7",x"ee",x"f8",x"ea",x"f1",x"eb",x"ff",x"0e",x"e8",x"18",x"ea",x"0c",x"f4",x"15",x"ef",x"15",x"ef",x"08",x"0d",x"f8",x"ff",x"04",x"0d",x"f5",x"fc",x"09",x"f6"),
    (x"f4",x"ef",x"02",x"03",x"ee",x"15",x"0b",x"0e",x"e8",x"f7",x"16",x"fb",x"ef",x"01",x"01",x"fa",x"fa",x"f8",x"e9",x"ea",x"16",x"0e",x"0a",x"0a",x"16",x"15",x"03"),
    (x"f5",x"12",x"14",x"12",x"0f",x"fc",x"ef",x"fa",x"eb",x"07",x"06",x"fc",x"eb",x"00",x"fc",x"eb",x"06",x"07",x"09",x"f5",x"fe",x"13",x"0e",x"18",x"11",x"f7",x"11"),
    (x"13",x"f7",x"13",x"07",x"09",x"0e",x"16",x"f2",x"f2",x"00",x"f9",x"fb",x"ef",x"0b",x"f7",x"ee",x"0f",x"0d",x"f6",x"16",x"f1",x"ed",x"f0",x"ee",x"ff",x"f2",x"ef"),
    (x"10",x"f2",x"08",x"fa",x"ed",x"04",x"0f",x"09",x"13",x"0d",x"ec",x"06",x"ea",x"ea",x"14",x"17",x"13",x"04",x"0f",x"0c",x"0e",x"16",x"f0",x"ea",x"17",x"11",x"08"),
    (x"0d",x"02",x"f7",x"03",x"f6",x"02",x"ec",x"ff",x"fe",x"fd",x"f9",x"17",x"fb",x"fb",x"e9",x"f3",x"15",x"0e",x"f9",x"e9",x"f4",x"0a",x"f4",x"15",x"f2",x"f7",x"12"),
    (x"fd",x"ec",x"02",x"ef",x"ed",x"0a",x"15",x"ec",x"0f",x"04",x"00",x"ef",x"0a",x"05",x"e9",x"0b",x"f6",x"e9",x"17",x"e8",x"f4",x"ed",x"01",x"0e",x"f4",x"fe",x"f2"),
    (x"fe",x"fd",x"09",x"f2",x"ed",x"f9",x"11",x"fe",x"f1",x"fe",x"10",x"06",x"ec",x"ec",x"ed",x"15",x"f8",x"fc",x"f0",x"08",x"f5",x"f2",x"f2",x"f9",x"03",x"f1",x"11"),
    (x"eb",x"02",x"0d",x"f5",x"f0",x"12",x"fb",x"ed",x"0a",x"0b",x"e9",x"0f",x"ea",x"17",x"04",x"02",x"e9",x"0a",x"15",x"18",x"e9",x"12",x"f5",x"f0",x"0f",x"eb",x"ed"),
    (x"fd",x"06",x"f9",x"08",x"ff",x"f8",x"00",x"eb",x"ed",x"f3",x"fd",x"f5",x"14",x"fa",x"ef",x"f0",x"ed",x"f9",x"f7",x"ed",x"05",x"0a",x"e8",x"08",x"fd",x"05",x"fb"),
    (x"0a",x"ff",x"14",x"0f",x"10",x"e9",x"f0",x"fc",x"f4",x"18",x"ef",x"f3",x"14",x"f7",x"f6",x"fd",x"17",x"10",x"05",x"17",x"f9",x"f3",x"04",x"ec",x"09",x"f8",x"0c"),
    (x"eb",x"00",x"13",x"e9",x"f0",x"06",x"0a",x"11",x"e8",x"05",x"f8",x"ed",x"f6",x"14",x"06",x"ff",x"0f",x"e8",x"0a",x"ee",x"03",x"ed",x"f1",x"fa",x"f6",x"06",x"0e"),
    (x"18",x"f3",x"07",x"06",x"15",x"0d",x"ea",x"f7",x"0e",x"16",x"10",x"16",x"0f",x"f1",x"03",x"fe",x"0f",x"08",x"fb",x"f2",x"0c",x"fc",x"f8",x"01",x"eb",x"10",x"11"),
    (x"13",x"f8",x"0f",x"f2",x"03",x"06",x"0a",x"0c",x"07",x"eb",x"f5",x"f3",x"ff",x"fc",x"fe",x"16",x"ef",x"fe",x"0f",x"12",x"f1",x"16",x"f8",x"10",x"f7",x"ed",x"f3"),
    (x"f8",x"07",x"f0",x"eb",x"ea",x"03",x"fa",x"13",x"f0",x"17",x"fd",x"f0",x"f4",x"f4",x"01",x"fc",x"15",x"e9",x"14",x"14",x"11",x"03",x"fd",x"05",x"03",x"fa",x"08"),
    (x"ff",x"01",x"0b",x"fa",x"f2",x"f7",x"03",x"03",x"00",x"10",x"07",x"f7",x"0a",x"ed",x"f8",x"14",x"eb",x"f9",x"04",x"ea",x"fd",x"08",x"07",x"15",x"0f",x"11",x"f8"),
    (x"ee",x"0b",x"0d",x"00",x"05",x"fd",x"f8",x"f2",x"15",x"17",x"ea",x"01",x"00",x"00",x"0b",x"07",x"fc",x"f9",x"ff",x"14",x"15",x"fd",x"0c",x"04",x"f3",x"0b",x"14"),
    (x"0b",x"0e",x"f1",x"06",x"0e",x"fc",x"06",x"ff",x"eb",x"0d",x"00",x"11",x"f7",x"e8",x"0a",x"fb",x"15",x"f1",x"fa",x"05",x"13",x"ea",x"06",x"f2",x"15",x"f4",x"ef"),
    (x"0e",x"f4",x"f0",x"fb",x"f3",x"ef",x"fb",x"fa",x"01",x"09",x"14",x"02",x"ff",x"10",x"06",x"0d",x"12",x"ff",x"f5",x"02",x"e9",x"f9",x"08",x"f0",x"04",x"fd",x"e8"),
    (x"f0",x"0a",x"eb",x"08",x"f2",x"03",x"ea",x"fb",x"e8",x"12",x"fd",x"fe",x"05",x"ff",x"fd",x"16",x"f5",x"fe",x"13",x"01",x"f3",x"14",x"e8",x"fd",x"f2",x"ed",x"ff"),
    (x"07",x"03",x"18",x"ff",x"fd",x"07",x"ed",x"0b",x"eb",x"07",x"17",x"f8",x"13",x"01",x"f5",x"16",x"17",x"09",x"15",x"0c",x"ed",x"fc",x"f6",x"f6",x"f1",x"f3",x"fe"),
    (x"f4",x"09",x"f1",x"0e",x"ff",x"f6",x"00",x"f0",x"ec",x"16",x"fd",x"e9",x"17",x"15",x"f2",x"0c",x"06",x"11",x"01",x"0e",x"ed",x"fd",x"0c",x"0f",x"11",x"04",x"01"),
    (x"fe",x"e8",x"ff",x"fe",x"08",x"10",x"ef",x"16",x"f4",x"05",x"0e",x"fc",x"00",x"f3",x"f9",x"0d",x"ef",x"f7",x"ee",x"0e",x"14",x"05",x"15",x"17",x"f3",x"0c",x"17"),
    (x"0d",x"0b",x"ed",x"f0",x"10",x"10",x"ee",x"00",x"fd",x"16",x"18",x"ec",x"0a",x"0e",x"f6",x"03",x"09",x"16",x"ed",x"f2",x"0e",x"f0",x"10",x"f3",x"e9",x"12",x"ea"),
    (x"16",x"01",x"12",x"18",x"ee",x"11",x"02",x"ef",x"07",x"08",x"fa",x"f6",x"f6",x"fb",x"17",x"11",x"0e",x"0e",x"fe",x"01",x"09",x"04",x"0b",x"fa",x"03",x"f6",x"03"),
    (x"f3",x"ee",x"fe",x"06",x"10",x"f6",x"15",x"0f",x"eb",x"ed",x"11",x"00",x"12",x"14",x"01",x"e9",x"12",x"05",x"fa",x"14",x"f0",x"07",x"0c",x"f3",x"12",x"0a",x"06"),
    (x"0d",x"e9",x"ff",x"f7",x"f8",x"f3",x"fe",x"f1",x"17",x"ea",x"0d",x"06",x"17",x"0b",x"ed",x"00",x"08",x"fd",x"09",x"05",x"ee",x"ef",x"f3",x"12",x"09",x"ea",x"ee"),
    (x"ff",x"f0",x"10",x"0a",x"ef",x"fc",x"f0",x"16",x"01",x"f2",x"ff",x"ef",x"13",x"0e",x"f7",x"03",x"0d",x"10",x"f6",x"03",x"fc",x"0e",x"ea",x"e9",x"fc",x"f8",x"0f"),
    (x"14",x"fa",x"0c",x"01",x"f9",x"0d",x"14",x"ed",x"0e",x"0e",x"02",x"04",x"06",x"0c",x"0c",x"ee",x"14",x"fc",x"f3",x"0d",x"f9",x"15",x"fc",x"09",x"f5",x"07",x"e9"),
    (x"08",x"0f",x"07",x"f8",x"ed",x"18",x"15",x"fd",x"0b",x"ea",x"f5",x"01",x"fc",x"ff",x"17",x"0b",x"f2",x"15",x"01",x"11",x"07",x"fd",x"f7",x"0e",x"01",x"eb",x"16"),
    (x"e9",x"05",x"16",x"0e",x"0f",x"e8",x"09",x"11",x"f8",x"f2",x"ee",x"e8",x"f8",x"ea",x"12",x"e8",x"fe",x"0e",x"0e",x"ff",x"04",x"e9",x"15",x"12",x"f1",x"05",x"f6"),
    (x"ef",x"12",x"0e",x"ee",x"0c",x"15",x"0a",x"03",x"fe",x"ee",x"01",x"09",x"0c",x"15",x"ff",x"ee",x"f7",x"ef",x"18",x"ee",x"0a",x"0c",x"eb",x"0d",x"08",x"06",x"14"),
    (x"07",x"15",x"ef",x"ec",x"f0",x"0c",x"fd",x"ec",x"f6",x"f6",x"f0",x"fd",x"f9",x"00",x"fe",x"fc",x"fb",x"0a",x"ed",x"09",x"17",x"00",x"02",x"0d",x"ea",x"0e",x"10"),
    (x"fe",x"ef",x"07",x"eb",x"07",x"f4",x"02",x"f5",x"13",x"e8",x"fb",x"15",x"12",x"0a",x"f1",x"09",x"f3",x"0f",x"ea",x"f0",x"ed",x"fa",x"ee",x"04",x"e8",x"ec",x"f7"),
    (x"0e",x"03",x"e8",x"e9",x"0d",x"09",x"0b",x"f1",x"fb",x"fe",x"f6",x"13",x"16",x"fe",x"ec",x"14",x"02",x"e9",x"15",x"14",x"03",x"f1",x"f2",x"f7",x"11",x"ed",x"ec"),
    (x"14",x"15",x"f5",x"ea",x"f8",x"fa",x"06",x"03",x"0f",x"15",x"ff",x"16",x"f1",x"f4",x"eb",x"f2",x"0f",x"04",x"04",x"15",x"ed",x"05",x"05",x"14",x"fb",x"16",x"fe"),
    (x"0d",x"06",x"f8",x"0a",x"fb",x"ee",x"04",x"03",x"10",x"f1",x"13",x"01",x"ed",x"fa",x"ff",x"08",x"ea",x"13",x"06",x"0b",x"15",x"0f",x"04",x"0d",x"09",x"13",x"ef"),
    (x"f5",x"06",x"f8",x"ef",x"fe",x"fb",x"fc",x"fa",x"10",x"0f",x"09",x"f4",x"03",x"10",x"05",x"eb",x"f4",x"f1",x"17",x"17",x"05",x"08",x"ee",x"18",x"05",x"08",x"14"),
    (x"16",x"f1",x"f4",x"02",x"03",x"eb",x"f2",x"e9",x"07",x"f6",x"f8",x"01",x"0a",x"16",x"11",x"0b",x"0c",x"0c",x"14",x"0a",x"00",x"0d",x"05",x"fa",x"15",x"ef",x"0f"),
    (x"06",x"0d",x"09",x"0b",x"08",x"15",x"f6",x"e8",x"07",x"03",x"0f",x"14",x"fc",x"f1",x"f5",x"ea",x"02",x"01",x"f8",x"02",x"16",x"0d",x"01",x"04",x"f7",x"16",x"0c"),
    (x"fd",x"18",x"f8",x"fc",x"ea",x"16",x"0c",x"0e",x"f7",x"f3",x"02",x"ea",x"fc",x"0a",x"08",x"fc",x"08",x"0e",x"0b",x"07",x"f7",x"0d",x"eb",x"f6",x"f3",x"12",x"f6"),
    (x"0c",x"01",x"0f",x"0c",x"f4",x"fb",x"10",x"e9",x"18",x"ef",x"ec",x"01",x"fb",x"fd",x"02",x"e8",x"02",x"02",x"fa",x"09",x"f8",x"f8",x"f9",x"ed",x"05",x"03",x"04"),
    (x"ff",x"05",x"e8",x"12",x"ef",x"f9",x"10",x"03",x"0a",x"15",x"f3",x"05",x"fe",x"0d",x"06",x"ff",x"18",x"ec",x"00",x"0d",x"fd",x"08",x"e9",x"ec",x"10",x"e8",x"10"),
    (x"fb",x"15",x"0c",x"f9",x"18",x"f4",x"f4",x"18",x"ee",x"f2",x"0d",x"0d",x"05",x"ff",x"f7",x"0e",x"e8",x"f6",x"0e",x"16",x"12",x"ea",x"0f",x"f6",x"0b",x"f9",x"f9"),
    (x"0b",x"12",x"01",x"15",x"e8",x"ff",x"0a",x"e8",x"eb",x"17",x"11",x"12",x"f6",x"04",x"ee",x"17",x"ed",x"16",x"f2",x"f5",x"04",x"f5",x"00",x"ed",x"07",x"f4",x"14"),
    (x"07",x"0a",x"f7",x"fd",x"0c",x"e8",x"04",x"0f",x"0c",x"f9",x"f0",x"ed",x"f1",x"12",x"08",x"0f",x"13",x"ee",x"00",x"ee",x"12",x"10",x"02",x"02",x"03",x"ff",x"16"),
    (x"f4",x"03",x"f4",x"0c",x"03",x"0f",x"f8",x"f6",x"ec",x"e8",x"0e",x"15",x"12",x"10",x"11",x"01",x"05",x"05",x"e9",x"05",x"03",x"ec",x"ea",x"fd",x"f3",x"13",x"07")
);
--------------------------------------------------------
--Relu_1
constant Relu_1_OUT_SIZE     :  integer := 128;
--------------------------------------------------------
constant OUT_WIDTH     :  integer := 128;
end package;