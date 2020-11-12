--------------------------------------------------------
-- This file is generated by Delirium - Sma-RTY SAS 
-- Wed Nov 11 12:26:57 2020
--------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
library work;
    use work.Delirium.all;

package params is
--------------------------------------------------------
-- INPUT
constant INPUT_CHANNELS    : integer := 32;
constant INPUT_IMAGE_WIDTH : integer := 112;
--------------------------------------------------------
--Conv_0
constant Conv_0_IMAGE_WIDTH  :  integer := 112;
constant Conv_0_IN_SIZE      :  integer := 32;
constant Conv_0_OUT_SIZE     :  integer := 64;
constant Conv_0_MULT_STYLE   :  string  := "dsp";
constant Conv_0_PIPELINE     :  boolean := True;
constant Conv_0_KERNEL_SIZE  :  integer := 1;
constant Conv_0_PADDING      :  boolean := FALSE;
constant Conv_0_STRIDE       :  positive:= 1;
constant Conv_0_BIAS_VALUE   :  data_array  (0 to Conv_0_OUT_SIZE - 1) := 
    (x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00");
constant Conv_0_KERNEL_VALUE :  data_matrix (0 to Conv_0_OUT_SIZE - 1 ,  0 to Conv_0_IN_SIZE * Conv_0_KERNEL_SIZE * Conv_0_KERNEL_SIZE - 1) := (
    (x"ee",x"eb",x"09",x"12",x"fa",x"f3",x"ea",x"07",x"fe",x"ff",x"0a",x"fe",x"07",x"06",x"f6",x"00",x"f7",x"0c",x"12",x"f0",x"ff",x"f0",x"ef",x"16",x"fe",x"f8",x"15",x"0d",x"fa",x"f5",x"15",x"fb"),
    (x"0e",x"fb",x"03",x"07",x"05",x"fd",x"0f",x"13",x"0a",x"ed",x"fb",x"f4",x"06",x"ed",x"16",x"f4",x"fa",x"10",x"fd",x"ec",x"f9",x"fd",x"f8",x"f6",x"15",x"fc",x"ec",x"f1",x"ed",x"01",x"13",x"fd"),
    (x"0d",x"fc",x"07",x"ee",x"f4",x"04",x"0a",x"08",x"04",x"13",x"12",x"06",x"05",x"fd",x"ef",x"ff",x"16",x"ff",x"fe",x"f3",x"12",x"f9",x"ed",x"fb",x"fe",x"fd",x"f3",x"0b",x"fd",x"f8",x"00",x"02"),
    (x"f3",x"08",x"11",x"0c",x"f8",x"07",x"0a",x"06",x"f0",x"f4",x"fa",x"fb",x"f6",x"07",x"fd",x"03",x"14",x"f0",x"f1",x"f7",x"fd",x"ee",x"0f",x"fb",x"f8",x"ea",x"07",x"fc",x"ee",x"09",x"0a",x"0a"),
    (x"f5",x"01",x"fa",x"f8",x"f9",x"fa",x"13",x"00",x"0f",x"ea",x"12",x"00",x"fd",x"15",x"f0",x"11",x"05",x"f7",x"02",x"16",x"ef",x"03",x"f3",x"fc",x"ec",x"eb",x"f5",x"07",x"08",x"07",x"0d",x"0e"),
    (x"06",x"0b",x"f8",x"0d",x"12",x"08",x"06",x"00",x"f0",x"f6",x"f8",x"06",x"00",x"0e",x"fc",x"11",x"fa",x"f6",x"f4",x"0f",x"fb",x"ed",x"00",x"fb",x"02",x"f9",x"f1",x"fe",x"16",x"11",x"15",x"ff"),
    (x"eb",x"ec",x"f5",x"f5",x"0d",x"05",x"f2",x"04",x"ea",x"0c",x"05",x"0d",x"15",x"01",x"f8",x"f8",x"f7",x"02",x"fd",x"04",x"07",x"08",x"f5",x"ef",x"02",x"f6",x"0e",x"16",x"00",x"16",x"00",x"fe"),
    (x"0e",x"f9",x"ee",x"0c",x"11",x"ef",x"07",x"fb",x"f5",x"00",x"03",x"03",x"01",x"12",x"15",x"09",x"0b",x"0b",x"02",x"f9",x"eb",x"0a",x"15",x"fc",x"02",x"fe",x"02",x"0f",x"0f",x"15",x"ff",x"f6"),
    (x"0c",x"ef",x"0f",x"f3",x"ef",x"05",x"06",x"ef",x"10",x"0f",x"f5",x"09",x"f7",x"f6",x"f7",x"0b",x"fa",x"f3",x"f8",x"ed",x"f0",x"0c",x"13",x"ea",x"03",x"f9",x"eb",x"04",x"0c",x"ec",x"f1",x"0c"),
    (x"ee",x"f6",x"0f",x"ef",x"fb",x"02",x"fe",x"ee",x"fc",x"f4",x"05",x"04",x"05",x"fa",x"ee",x"f2",x"10",x"07",x"0d",x"00",x"0c",x"03",x"04",x"f0",x"01",x"0e",x"0a",x"f8",x"06",x"0b",x"00",x"14"),
    (x"f0",x"07",x"14",x"f2",x"03",x"13",x"11",x"ea",x"f7",x"0a",x"f6",x"f4",x"06",x"eb",x"04",x"10",x"ef",x"14",x"fc",x"ea",x"16",x"fc",x"0c",x"fd",x"15",x"10",x"ea",x"07",x"f3",x"ef",x"16",x"03"),
    (x"fe",x"0e",x"ea",x"ef",x"ed",x"01",x"0f",x"ec",x"ef",x"fe",x"f6",x"ea",x"12",x"04",x"f2",x"fe",x"fa",x"fb",x"03",x"06",x"fd",x"0f",x"f0",x"ef",x"05",x"00",x"05",x"11",x"fd",x"f0",x"fb",x"f1"),
    (x"0b",x"f4",x"12",x"fb",x"ea",x"ed",x"ea",x"0e",x"f1",x"f1",x"fa",x"0b",x"10",x"f2",x"10",x"10",x"08",x"03",x"12",x"05",x"f2",x"f2",x"14",x"11",x"ec",x"0d",x"16",x"eb",x"08",x"05",x"00",x"f2"),
    (x"ea",x"02",x"07",x"0c",x"01",x"03",x"02",x"00",x"ee",x"0f",x"06",x"0a",x"fb",x"fb",x"f4",x"16",x"16",x"f7",x"12",x"04",x"f9",x"03",x"0a",x"f6",x"10",x"ea",x"ee",x"01",x"0f",x"14",x"f1",x"f0"),
    (x"0c",x"f2",x"fb",x"00",x"fe",x"f0",x"fa",x"10",x"ec",x"13",x"fa",x"fe",x"ec",x"12",x"16",x"11",x"02",x"f5",x"ff",x"fa",x"05",x"ee",x"0f",x"ec",x"0b",x"eb",x"0b",x"0f",x"fc",x"ee",x"fe",x"f8"),
    (x"ee",x"12",x"fc",x"10",x"02",x"f6",x"00",x"05",x"fc",x"fb",x"0a",x"ff",x"f3",x"f9",x"15",x"09",x"f0",x"ea",x"02",x"ff",x"ed",x"15",x"0b",x"02",x"f7",x"04",x"14",x"ed",x"f7",x"ee",x"f3",x"f9"),
    (x"02",x"ff",x"07",x"11",x"06",x"0e",x"14",x"f6",x"0a",x"fc",x"06",x"ec",x"ee",x"14",x"07",x"eb",x"04",x"f8",x"09",x"03",x"04",x"04",x"ed",x"06",x"07",x"f5",x"0f",x"f9",x"09",x"16",x"f4",x"12"),
    (x"0d",x"14",x"13",x"12",x"13",x"05",x"0e",x"ec",x"f2",x"f4",x"f8",x"0a",x"16",x"11",x"f3",x"07",x"0d",x"0d",x"06",x"0c",x"0d",x"f9",x"15",x"14",x"fa",x"00",x"fc",x"08",x"f1",x"ec",x"15",x"01"),
    (x"01",x"eb",x"f4",x"08",x"f5",x"ee",x"fc",x"02",x"ee",x"ee",x"f4",x"fd",x"0f",x"04",x"0d",x"00",x"03",x"ef",x"0e",x"ef",x"f4",x"0f",x"13",x"f4",x"fe",x"04",x"fa",x"eb",x"fc",x"ee",x"fb",x"0e"),
    (x"0a",x"05",x"0f",x"f2",x"02",x"00",x"03",x"ed",x"0c",x"ed",x"f0",x"ec",x"0e",x"f7",x"fb",x"07",x"13",x"fa",x"04",x"14",x"13",x"f9",x"05",x"0b",x"f3",x"08",x"fe",x"04",x"13",x"05",x"f1",x"fb"),
    (x"0f",x"05",x"f7",x"07",x"f3",x"10",x"09",x"07",x"f6",x"f7",x"f0",x"ec",x"00",x"f4",x"0f",x"f8",x"14",x"ef",x"00",x"fb",x"0a",x"fb",x"0d",x"ef",x"fe",x"03",x"f4",x"15",x"07",x"12",x"03",x"08"),
    (x"0f",x"fb",x"01",x"fa",x"08",x"ea",x"07",x"f6",x"11",x"fd",x"01",x"eb",x"f8",x"f0",x"fd",x"fc",x"ff",x"02",x"ff",x"0f",x"ee",x"0f",x"0e",x"0d",x"fd",x"0c",x"f8",x"15",x"ef",x"fb",x"f7",x"0f"),
    (x"ea",x"f5",x"14",x"0e",x"f9",x"01",x"eb",x"04",x"0b",x"14",x"08",x"f3",x"08",x"0c",x"04",x"14",x"f4",x"f0",x"06",x"ee",x"09",x"03",x"f0",x"00",x"14",x"16",x"10",x"0d",x"f8",x"f2",x"12",x"0b"),
    (x"14",x"fd",x"03",x"ef",x"13",x"07",x"ec",x"f3",x"09",x"0e",x"fe",x"12",x"00",x"0d",x"06",x"09",x"fb",x"f2",x"16",x"f3",x"ee",x"16",x"f1",x"04",x"ea",x"0e",x"f3",x"f9",x"16",x"fb",x"fe",x"05"),
    (x"11",x"f7",x"00",x"13",x"f3",x"fd",x"10",x"fe",x"0b",x"04",x"ff",x"15",x"01",x"ee",x"0f",x"ff",x"10",x"16",x"06",x"fc",x"13",x"16",x"13",x"ec",x"ed",x"14",x"fc",x"11",x"04",x"01",x"ff",x"07"),
    (x"13",x"12",x"ea",x"0a",x"01",x"f2",x"01",x"09",x"fb",x"ec",x"ef",x"05",x"fc",x"ea",x"f6",x"05",x"15",x"01",x"f5",x"13",x"05",x"11",x"ff",x"15",x"f4",x"05",x"f9",x"f6",x"f5",x"12",x"fa",x"05"),
    (x"06",x"f9",x"06",x"f7",x"f7",x"02",x"f3",x"fb",x"13",x"f1",x"09",x"02",x"fd",x"0f",x"ef",x"ec",x"0b",x"fd",x"ed",x"ed",x"f9",x"03",x"0e",x"f5",x"fc",x"01",x"fb",x"00",x"f2",x"ed",x"ee",x"ed"),
    (x"0a",x"f2",x"04",x"06",x"03",x"f0",x"15",x"03",x"ff",x"15",x"eb",x"fe",x"0a",x"06",x"10",x"f8",x"03",x"15",x"0b",x"eb",x"f5",x"0d",x"ee",x"06",x"0e",x"fc",x"09",x"f5",x"0e",x"16",x"f9",x"fd"),
    (x"fa",x"0a",x"ea",x"02",x"f9",x"f0",x"00",x"f2",x"0c",x"0d",x"03",x"f7",x"04",x"f1",x"07",x"fd",x"04",x"f5",x"12",x"fb",x"15",x"0d",x"02",x"f9",x"08",x"04",x"f3",x"ef",x"ee",x"f4",x"ef",x"09"),
    (x"0a",x"f8",x"10",x"fc",x"16",x"0b",x"f4",x"f2",x"ed",x"08",x"15",x"11",x"0c",x"fb",x"f4",x"0d",x"f8",x"0b",x"0e",x"ea",x"00",x"ea",x"f8",x"ee",x"fb",x"f3",x"f7",x"f5",x"0e",x"06",x"f2",x"ed"),
    (x"07",x"0e",x"04",x"fa",x"12",x"15",x"0c",x"eb",x"11",x"13",x"f1",x"fa",x"03",x"05",x"11",x"fe",x"0a",x"0f",x"15",x"f3",x"ed",x"03",x"14",x"0e",x"15",x"10",x"fe",x"ff",x"f1",x"fe",x"fd",x"ec"),
    (x"10",x"f7",x"00",x"f6",x"01",x"fd",x"fb",x"ff",x"03",x"0a",x"f8",x"fc",x"15",x"fa",x"ff",x"05",x"f8",x"00",x"13",x"07",x"fc",x"f6",x"ef",x"0c",x"08",x"16",x"ec",x"f4",x"ed",x"04",x"ff",x"fb"),
    (x"f4",x"0d",x"11",x"06",x"01",x"f5",x"08",x"0d",x"0f",x"fe",x"00",x"04",x"15",x"05",x"10",x"f4",x"fd",x"12",x"0d",x"0e",x"11",x"03",x"f4",x"f4",x"fa",x"ed",x"f7",x"fc",x"06",x"0f",x"f9",x"ef"),
    (x"f6",x"09",x"ed",x"fd",x"fb",x"0d",x"15",x"06",x"ef",x"f4",x"10",x"f4",x"02",x"f1",x"12",x"00",x"03",x"fc",x"fc",x"07",x"06",x"f2",x"12",x"14",x"0a",x"0a",x"12",x"0d",x"fe",x"08",x"06",x"f8"),
    (x"11",x"0e",x"f1",x"f4",x"00",x"01",x"0b",x"0b",x"07",x"00",x"ff",x"fd",x"03",x"12",x"f1",x"f2",x"05",x"01",x"05",x"f2",x"07",x"ee",x"09",x"07",x"fd",x"f6",x"fb",x"fd",x"ea",x"12",x"eb",x"f8"),
    (x"fd",x"fe",x"ed",x"f9",x"ec",x"09",x"11",x"f3",x"16",x"eb",x"f3",x"f8",x"0b",x"0a",x"16",x"0e",x"f6",x"0c",x"09",x"11",x"15",x"fc",x"0e",x"ec",x"00",x"03",x"11",x"ed",x"15",x"12",x"08",x"09"),
    (x"15",x"eb",x"f9",x"08",x"00",x"eb",x"fa",x"16",x"0c",x"13",x"10",x"0c",x"11",x"f4",x"13",x"f5",x"02",x"01",x"fd",x"16",x"ea",x"0e",x"f4",x"10",x"14",x"ea",x"fd",x"fc",x"05",x"09",x"08",x"fc"),
    (x"ed",x"f5",x"03",x"fb",x"f6",x"f6",x"eb",x"14",x"0f",x"06",x"fd",x"12",x"00",x"f6",x"f2",x"04",x"f9",x"ed",x"f4",x"0f",x"05",x"f7",x"f0",x"13",x"13",x"11",x"03",x"ee",x"0d",x"f0",x"0b",x"14"),
    (x"fd",x"0d",x"00",x"0e",x"02",x"02",x"f0",x"08",x"f2",x"0b",x"05",x"12",x"fc",x"03",x"05",x"ed",x"ee",x"13",x"eb",x"fd",x"f7",x"ef",x"f9",x"0d",x"08",x"f0",x"f3",x"0a",x"07",x"14",x"fb",x"f8"),
    (x"fb",x"fc",x"15",x"0b",x"0e",x"04",x"12",x"0d",x"ef",x"f2",x"f9",x"0f",x"16",x"f3",x"12",x"0a",x"eb",x"0b",x"06",x"ec",x"f7",x"fe",x"eb",x"0d",x"14",x"09",x"ee",x"ef",x"06",x"05",x"03",x"ed"),
    (x"fa",x"09",x"fb",x"f4",x"fd",x"eb",x"0d",x"f8",x"00",x"f3",x"f8",x"04",x"f9",x"fd",x"fe",x"0c",x"0d",x"01",x"15",x"f4",x"ef",x"f5",x"fe",x"04",x"13",x"ef",x"11",x"fb",x"ff",x"ea",x"f6",x"fd"),
    (x"15",x"06",x"f4",x"ea",x"0f",x"13",x"f0",x"f1",x"ef",x"f0",x"f4",x"ec",x"f2",x"09",x"f6",x"13",x"03",x"ef",x"fb",x"f4",x"eb",x"03",x"f6",x"0a",x"14",x"fe",x"f8",x"f9",x"08",x"ed",x"0b",x"10"),
    (x"f4",x"eb",x"f5",x"f6",x"16",x"14",x"f0",x"13",x"12",x"ed",x"ea",x"11",x"ea",x"f9",x"f4",x"f0",x"ff",x"07",x"f7",x"04",x"f9",x"ee",x"01",x"0a",x"f5",x"fa",x"fd",x"eb",x"ec",x"ef",x"15",x"eb"),
    (x"12",x"10",x"09",x"ea",x"ea",x"fe",x"f5",x"08",x"f5",x"ee",x"0f",x"f6",x"06",x"f5",x"04",x"fb",x"f2",x"0a",x"12",x"fc",x"02",x"0b",x"f4",x"14",x"13",x"eb",x"12",x"06",x"0b",x"f2",x"07",x"13"),
    (x"f8",x"fb",x"0c",x"f3",x"10",x"10",x"06",x"ea",x"ec",x"14",x"00",x"00",x"f8",x"00",x"f2",x"04",x"13",x"ef",x"ee",x"f3",x"0a",x"ed",x"f0",x"00",x"16",x"eb",x"0e",x"ec",x"ed",x"f6",x"14",x"fd"),
    (x"ff",x"09",x"ff",x"11",x"fa",x"f6",x"f4",x"ef",x"ef",x"0b",x"10",x"04",x"f9",x"ea",x"f9",x"04",x"f4",x"0e",x"f9",x"00",x"14",x"f0",x"ed",x"0a",x"05",x"10",x"eb",x"0b",x"05",x"0b",x"02",x"fe"),
    (x"10",x"15",x"13",x"ed",x"fd",x"ef",x"06",x"fd",x"ec",x"14",x"05",x"16",x"fd",x"0e",x"f8",x"01",x"11",x"0b",x"08",x"f7",x"00",x"f1",x"f2",x"fe",x"f4",x"ea",x"f5",x"03",x"15",x"05",x"eb",x"f2"),
    (x"04",x"eb",x"07",x"fb",x"ea",x"16",x"fa",x"03",x"09",x"02",x"ed",x"0b",x"f7",x"f5",x"0b",x"0b",x"ff",x"fd",x"fa",x"00",x"fe",x"eb",x"f1",x"0b",x"f7",x"fc",x"f9",x"f0",x"0b",x"fb",x"ed",x"14"),
    (x"fa",x"f2",x"ee",x"ee",x"12",x"f2",x"06",x"03",x"16",x"ee",x"07",x"f6",x"01",x"f0",x"ef",x"ff",x"f7",x"ef",x"fb",x"ec",x"08",x"0e",x"fb",x"0a",x"f1",x"14",x"16",x"ed",x"14",x"eb",x"09",x"fe"),
    (x"f7",x"fe",x"0f",x"01",x"04",x"10",x"08",x"f3",x"14",x"0f",x"ee",x"f1",x"0c",x"0a",x"fe",x"0e",x"eb",x"f8",x"15",x"0d",x"15",x"09",x"fd",x"00",x"07",x"eb",x"fd",x"eb",x"f6",x"0a",x"ee",x"fa"),
    (x"fd",x"0d",x"07",x"16",x"fe",x"12",x"02",x"f8",x"fe",x"ef",x"16",x"f4",x"0c",x"0f",x"ed",x"fb",x"f7",x"11",x"14",x"f1",x"04",x"13",x"0b",x"12",x"04",x"11",x"08",x"ee",x"03",x"f9",x"fe",x"07"),
    (x"ed",x"f4",x"09",x"0a",x"ee",x"fa",x"fb",x"ec",x"f9",x"0d",x"12",x"fa",x"f8",x"0c",x"f1",x"0f",x"ee",x"0c",x"0d",x"08",x"f6",x"15",x"08",x"09",x"09",x"15",x"0b",x"f6",x"f2",x"00",x"15",x"fc"),
    (x"00",x"0b",x"ed",x"12",x"02",x"03",x"f0",x"f8",x"f7",x"11",x"0e",x"eb",x"04",x"0a",x"fe",x"f6",x"0c",x"02",x"09",x"f7",x"fd",x"15",x"03",x"f6",x"11",x"f9",x"fb",x"f9",x"13",x"0e",x"0e",x"0e"),
    (x"06",x"0d",x"0f",x"03",x"ed",x"ed",x"f9",x"fc",x"04",x"01",x"f3",x"ff",x"04",x"12",x"14",x"fe",x"0b",x"f5",x"06",x"fc",x"0a",x"fd",x"16",x"03",x"f6",x"ef",x"0d",x"fb",x"fd",x"13",x"0b",x"14"),
    (x"0b",x"0e",x"ff",x"fe",x"0f",x"04",x"ff",x"01",x"fa",x"06",x"10",x"fe",x"02",x"15",x"f0",x"04",x"14",x"0a",x"fa",x"f3",x"01",x"0f",x"06",x"09",x"12",x"fe",x"fa",x"ef",x"eb",x"00",x"15",x"09"),
    (x"ed",x"0e",x"ee",x"12",x"ef",x"14",x"04",x"fe",x"fd",x"ef",x"04",x"11",x"f8",x"0c",x"0a",x"0c",x"f0",x"fb",x"0a",x"01",x"03",x"f1",x"01",x"04",x"0b",x"f9",x"ec",x"10",x"fb",x"0e",x"14",x"ef"),
    (x"08",x"0d",x"16",x"08",x"ff",x"ea",x"ff",x"ff",x"00",x"f2",x"f0",x"f7",x"0f",x"0a",x"13",x"07",x"08",x"06",x"05",x"fb",x"f1",x"ef",x"f8",x"ee",x"f7",x"01",x"01",x"16",x"0b",x"f2",x"08",x"04"),
    (x"ff",x"ea",x"f9",x"f7",x"0d",x"01",x"08",x"16",x"05",x"07",x"0c",x"16",x"0a",x"f8",x"05",x"eb",x"05",x"ed",x"09",x"ee",x"11",x"14",x"ee",x"0f",x"15",x"f1",x"ef",x"f1",x"f7",x"0d",x"0e",x"02"),
    (x"f3",x"ee",x"0d",x"fe",x"0c",x"0e",x"f6",x"08",x"09",x"01",x"eb",x"15",x"0e",x"12",x"11",x"ee",x"13",x"f1",x"ef",x"0f",x"fb",x"05",x"0a",x"0e",x"11",x"f1",x"f6",x"05",x"09",x"fb",x"f9",x"ef"),
    (x"ed",x"0b",x"ef",x"09",x"0a",x"0a",x"f7",x"07",x"ee",x"fc",x"f8",x"f3",x"02",x"f1",x"f4",x"05",x"16",x"0e",x"06",x"10",x"f5",x"00",x"f1",x"11",x"06",x"ea",x"15",x"08",x"11",x"fa",x"09",x"fe"),
    (x"02",x"f8",x"f3",x"fa",x"fb",x"04",x"fd",x"07",x"01",x"ee",x"ef",x"f8",x"06",x"0f",x"ef",x"f8",x"f7",x"01",x"09",x"0e",x"ea",x"10",x"16",x"10",x"14",x"12",x"07",x"f7",x"13",x"f2",x"eb",x"f6"),
    (x"f2",x"ea",x"07",x"11",x"03",x"16",x"f9",x"ed",x"16",x"f8",x"ef",x"15",x"04",x"fd",x"fa",x"02",x"15",x"00",x"ee",x"12",x"eb",x"00",x"fa",x"f6",x"10",x"06",x"03",x"0c",x"f2",x"0e",x"11",x"f6"),
    (x"ec",x"ef",x"03",x"03",x"fd",x"ec",x"0f",x"0e",x"f0",x"06",x"00",x"f1",x"02",x"f7",x"fc",x"fe",x"06",x"0b",x"f2",x"15",x"fc",x"0b",x"f8",x"f8",x"13",x"0c",x"fb",x"03",x"fe",x"07",x"f7",x"f8"),
    (x"f8",x"ed",x"0c",x"0c",x"07",x"07",x"f3",x"fe",x"01",x"0a",x"f7",x"07",x"01",x"10",x"f0",x"f3",x"f0",x"09",x"0d",x"ef",x"f3",x"fa",x"09",x"ea",x"06",x"0f",x"f1",x"16",x"ef",x"12",x"f4",x"fa")
);
--------------------------------------------------------
--Relu_1
constant Relu_1_OUT_SIZE     :  integer := 64;
--------------------------------------------------------
constant OUT_WIDTH     :  integer := 64;
end package;