--------------------------------------------------------
-- This file is generated by Delirium - Sma-RTY SAS 
-- Wed Nov 11 05:56:26 2020
--------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
library work;
    use work.Delirium.all;

package params is
--------------------------------------------------------
-- INPUT
constant INPUT_CHANNELS    : integer := 16;
constant INPUT_IMAGE_WIDTH : integer := 56;
--------------------------------------------------------
--Conv_0
constant Conv_0_IMAGE_WIDTH  :  integer := 56;
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
    (x"f0",x"fb",x"00",x"1a",x"16",x"ff",x"08",x"0e",x"fd",x"19",x"09",x"14",x"e8",x"16",x"fe",x"e7"),
    (x"f8",x"11",x"f6",x"f4",x"1a",x"f7",x"18",x"08",x"e8",x"14",x"01",x"e3",x"0d",x"eb",x"f9",x"01"),
    (x"0b",x"e6",x"1e",x"1d",x"f2",x"03",x"f8",x"03",x"02",x"e5",x"1f",x"0d",x"1d",x"fb",x"ff",x"08"),
    (x"ed",x"07",x"14",x"09",x"1d",x"f1",x"ea",x"ec",x"18",x"16",x"f7",x"f5",x"0a",x"0a",x"16",x"f0"),
    (x"ea",x"15",x"1f",x"ee",x"0e",x"ee",x"f4",x"12",x"13",x"17",x"12",x"13",x"08",x"02",x"1e",x"1a"),
    (x"0f",x"14",x"14",x"ec",x"f3",x"12",x"e4",x"01",x"fd",x"19",x"06",x"20",x"f2",x"ef",x"0c",x"ff"),
    (x"03",x"0c",x"f7",x"06",x"1e",x"f7",x"ff",x"18",x"14",x"f8",x"1b",x"16",x"f9",x"fd",x"16",x"f8"),
    (x"fe",x"fe",x"0e",x"f3",x"f8",x"07",x"e6",x"f3",x"11",x"fa",x"0e",x"fa",x"13",x"f8",x"01",x"e5"),
    (x"f1",x"fe",x"09",x"15",x"07",x"1b",x"06",x"01",x"1f",x"f6",x"18",x"e4",x"ec",x"13",x"04",x"08"),
    (x"1e",x"ea",x"10",x"12",x"1f",x"1b",x"11",x"e7",x"18",x"ee",x"eb",x"11",x"f1",x"f1",x"ee",x"0c"),
    (x"20",x"09",x"03",x"20",x"eb",x"f5",x"f7",x"ff",x"f3",x"15",x"f8",x"ec",x"e5",x"11",x"e4",x"1f"),
    (x"fc",x"e7",x"ff",x"e7",x"00",x"0b",x"02",x"f2",x"0a",x"ed",x"e8",x"fb",x"fb",x"f8",x"1e",x"17"),
    (x"e1",x"f7",x"ef",x"e1",x"e5",x"eb",x"18",x"e1",x"e5",x"02",x"f6",x"f4",x"ea",x"0b",x"0f",x"f5"),
    (x"f7",x"1e",x"07",x"08",x"15",x"05",x"01",x"f8",x"04",x"07",x"f0",x"08",x"00",x"f4",x"e5",x"14"),
    (x"18",x"e7",x"12",x"e0",x"09",x"0a",x"fe",x"1b",x"fb",x"01",x"20",x"09",x"ed",x"fd",x"fb",x"10"),
    (x"ee",x"ee",x"1e",x"f6",x"06",x"f1",x"f3",x"ed",x"e9",x"e2",x"19",x"eb",x"f3",x"fb",x"10",x"1e"),
    (x"04",x"f0",x"e1",x"0f",x"0e",x"eb",x"f6",x"05",x"14",x"e1",x"f8",x"ed",x"f4",x"1a",x"17",x"e2"),
    (x"00",x"14",x"08",x"fd",x"1b",x"04",x"11",x"f8",x"1c",x"0b",x"f3",x"1b",x"13",x"0b",x"04",x"1c"),
    (x"fa",x"fa",x"f0",x"1f",x"18",x"1a",x"ed",x"18",x"03",x"ee",x"f2",x"05",x"f0",x"18",x"0a",x"14"),
    (x"04",x"00",x"14",x"19",x"e2",x"f7",x"fd",x"e2",x"17",x"07",x"fc",x"0e",x"1a",x"10",x"f1",x"fb"),
    (x"fd",x"ee",x"06",x"04",x"ea",x"e7",x"0e",x"00",x"ef",x"19",x"19",x"ed",x"14",x"f3",x"10",x"01"),
    (x"02",x"ed",x"01",x"0c",x"00",x"18",x"ed",x"ef",x"e3",x"05",x"03",x"ed",x"e4",x"fb",x"03",x"06"),
    (x"f3",x"02",x"e6",x"07",x"0a",x"e9",x"fc",x"e3",x"1a",x"ec",x"1a",x"fc",x"0b",x"0e",x"00",x"1e"),
    (x"09",x"e8",x"1e",x"fd",x"0c",x"f1",x"f7",x"e8",x"01",x"e6",x"03",x"e2",x"f4",x"eb",x"04",x"f2"),
    (x"f5",x"08",x"e5",x"03",x"16",x"eb",x"ea",x"1c",x"0e",x"e2",x"0d",x"11",x"f3",x"0d",x"1a",x"09"),
    (x"f5",x"11",x"18",x"15",x"f7",x"01",x"14",x"e6",x"18",x"1f",x"0d",x"f9",x"0a",x"e6",x"1f",x"fc"),
    (x"12",x"f1",x"16",x"fd",x"f9",x"f3",x"ef",x"e2",x"01",x"ea",x"e5",x"f9",x"04",x"f3",x"f1",x"ef"),
    (x"1e",x"eb",x"1f",x"18",x"04",x"11",x"e6",x"eb",x"0b",x"18",x"ec",x"07",x"e3",x"04",x"1a",x"f8"),
    (x"10",x"19",x"fc",x"07",x"e7",x"1b",x"f6",x"15",x"e7",x"0b",x"02",x"fe",x"e6",x"0e",x"1c",x"19"),
    (x"ec",x"e2",x"1a",x"18",x"ec",x"fc",x"13",x"06",x"0a",x"f3",x"0c",x"1e",x"e7",x"e6",x"e8",x"ff"),
    (x"1e",x"e3",x"01",x"e2",x"e9",x"f9",x"0c",x"e7",x"e3",x"07",x"e7",x"11",x"1f",x"e3",x"f0",x"e1"),
    (x"ea",x"f1",x"1b",x"f7",x"0d",x"e5",x"e6",x"fe",x"14",x"fa",x"f4",x"ed",x"ee",x"fa",x"fc",x"f9"),
    (x"0c",x"e8",x"fd",x"06",x"13",x"00",x"fd",x"09",x"14",x"ea",x"e4",x"16",x"fe",x"05",x"13",x"0f"),
    (x"04",x"10",x"e7",x"20",x"f0",x"e2",x"e7",x"07",x"e5",x"05",x"1a",x"fe",x"fc",x"11",x"09",x"fe"),
    (x"07",x"ea",x"1d",x"05",x"e8",x"1c",x"ef",x"17",x"ec",x"1e",x"13",x"0c",x"e4",x"fa",x"e7",x"e1"),
    (x"ee",x"1e",x"f9",x"e7",x"04",x"f3",x"15",x"f2",x"f8",x"12",x"07",x"12",x"fe",x"03",x"e6",x"f0"),
    (x"f0",x"06",x"15",x"e4",x"12",x"1d",x"14",x"1d",x"ed",x"e0",x"16",x"16",x"f8",x"ef",x"ea",x"14"),
    (x"e1",x"06",x"f5",x"e7",x"ec",x"17",x"1e",x"04",x"e9",x"19",x"ed",x"f5",x"fd",x"0e",x"f2",x"f0"),
    (x"eb",x"02",x"ef",x"ec",x"f2",x"06",x"f2",x"ee",x"1f",x"08",x"07",x"09",x"12",x"11",x"fb",x"02"),
    (x"f6",x"09",x"05",x"f7",x"e9",x"f5",x"ef",x"f4",x"0e",x"f9",x"f7",x"19",x"f0",x"16",x"03",x"14"),
    (x"ef",x"ed",x"1e",x"18",x"e8",x"ea",x"08",x"1b",x"06",x"f4",x"fb",x"0e",x"fb",x"09",x"00",x"ff"),
    (x"ff",x"f4",x"e2",x"02",x"0c",x"e7",x"08",x"e8",x"e9",x"fb",x"1d",x"1a",x"e8",x"14",x"e5",x"e7"),
    (x"f1",x"f0",x"f9",x"05",x"1b",x"e5",x"f7",x"ee",x"1a",x"01",x"f4",x"ff",x"e5",x"fb",x"ee",x"f6"),
    (x"0f",x"e5",x"0a",x"17",x"02",x"05",x"06",x"f6",x"e6",x"10",x"e1",x"1c",x"fb",x"f7",x"04",x"f3"),
    (x"06",x"1c",x"10",x"00",x"ec",x"0d",x"fd",x"1f",x"e3",x"08",x"f3",x"fa",x"ed",x"09",x"f4",x"e8"),
    (x"fc",x"0f",x"f0",x"f9",x"f9",x"f1",x"fd",x"ea",x"0f",x"10",x"17",x"1d",x"03",x"f6",x"ed",x"14"),
    (x"f9",x"1c",x"1f",x"09",x"1d",x"12",x"e3",x"e2",x"f7",x"e3",x"16",x"01",x"12",x"0a",x"1c",x"17"),
    (x"10",x"00",x"0a",x"17",x"10",x"fe",x"08",x"17",x"fb",x"04",x"08",x"e6",x"e1",x"fe",x"f6",x"11"),
    (x"08",x"fc",x"ec",x"e1",x"f0",x"f5",x"ff",x"0a",x"f1",x"12",x"f6",x"ff",x"0b",x"03",x"f3",x"f7"),
    (x"1d",x"00",x"1c",x"0f",x"0e",x"0e",x"16",x"00",x"14",x"0d",x"fa",x"f5",x"09",x"08",x"01",x"18"),
    (x"09",x"19",x"f1",x"fe",x"14",x"f8",x"03",x"0f",x"13",x"f8",x"e2",x"19",x"17",x"e6",x"eb",x"1f"),
    (x"f4",x"18",x"f4",x"17",x"f9",x"12",x"fa",x"00",x"19",x"0d",x"f0",x"10",x"01",x"15",x"09",x"00"),
    (x"f0",x"e6",x"e5",x"fa",x"10",x"eb",x"ef",x"00",x"fd",x"06",x"07",x"1f",x"15",x"fc",x"1a",x"e6"),
    (x"e6",x"f3",x"fb",x"13",x"ee",x"13",x"e1",x"fb",x"04",x"13",x"08",x"0a",x"07",x"ef",x"e3",x"07"),
    (x"15",x"09",x"08",x"f4",x"14",x"f4",x"e6",x"1e",x"16",x"1a",x"13",x"04",x"15",x"f4",x"14",x"1c"),
    (x"e5",x"fc",x"ff",x"f6",x"e1",x"ee",x"e9",x"e6",x"f5",x"14",x"fa",x"0d",x"06",x"10",x"1c",x"05"),
    (x"ea",x"ec",x"1c",x"03",x"12",x"e6",x"09",x"f4",x"1e",x"04",x"08",x"f0",x"e6",x"02",x"0e",x"f6"),
    (x"ef",x"fc",x"e7",x"09",x"11",x"17",x"16",x"f5",x"f5",x"ff",x"fd",x"e9",x"12",x"fc",x"0a",x"e5"),
    (x"0d",x"02",x"f6",x"fb",x"fb",x"f6",x"ed",x"f7",x"ed",x"05",x"e4",x"f8",x"0a",x"1c",x"0a",x"fe"),
    (x"fd",x"05",x"e3",x"e2",x"0c",x"1d",x"0f",x"e5",x"eb",x"e2",x"10",x"ec",x"0e",x"f8",x"f5",x"f0"),
    (x"06",x"05",x"13",x"e3",x"ec",x"1d",x"04",x"e6",x"f9",x"ed",x"f4",x"1f",x"f3",x"1e",x"e5",x"fe"),
    (x"f5",x"18",x"19",x"f6",x"f9",x"0a",x"04",x"fe",x"09",x"19",x"ec",x"ec",x"e5",x"e0",x"0e",x"e1"),
    (x"17",x"ee",x"ef",x"02",x"1c",x"10",x"0e",x"00",x"18",x"f8",x"14",x"0e",x"10",x"04",x"05",x"14"),
    (x"e4",x"ea",x"18",x"1f",x"f9",x"ef",x"05",x"e8",x"f8",x"fd",x"f7",x"ff",x"0d",x"0f",x"fb",x"0f")
);
--------------------------------------------------------
constant OUT_WIDTH     :  integer := 64;
end package;