--------------------------------------------------------
-- This file is generated by Delirium - Sma-RTY SAS 
-- Wed Nov 11 05:56:24 2020
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
constant Conv_0_OUT_SIZE     :  integer := 32;
constant Conv_0_MULT_STYLE   :  string  := "dsp";
constant Conv_0_PIPELINE     :  boolean := True;
constant Conv_0_KERNEL_SIZE  :  integer := 1;
constant Conv_0_PADDING      :  boolean := FALSE;
constant Conv_0_STRIDE       :  positive:= 1;
constant Conv_0_BIAS_VALUE   :  data_array  (0 to Conv_0_OUT_SIZE - 1) := 
    (x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00");
constant Conv_0_KERNEL_VALUE :  data_matrix (0 to Conv_0_OUT_SIZE - 1 ,  0 to Conv_0_IN_SIZE * Conv_0_KERNEL_SIZE * Conv_0_KERNEL_SIZE - 1) := (
    (x"eb",x"eb",x"f8",x"19",x"15",x"1c",x"e7",x"14",x"e6",x"f9",x"1b",x"ed",x"07",x"1d",x"e2",x"1c"),
    (x"e7",x"ed",x"03",x"0b",x"f8",x"01",x"03",x"05",x"f8",x"e3",x"1a",x"0f",x"ea",x"0d",x"f9",x"0c"),
    (x"e6",x"1c",x"04",x"18",x"e4",x"19",x"19",x"0b",x"1e",x"05",x"0e",x"19",x"ec",x"1d",x"08",x"e8"),
    (x"16",x"e8",x"1f",x"ff",x"0c",x"ec",x"0e",x"09",x"e1",x"04",x"13",x"06",x"ee",x"eb",x"e5",x"e3"),
    (x"e4",x"0c",x"17",x"f8",x"f6",x"18",x"15",x"07",x"06",x"0e",x"f5",x"f6",x"f0",x"14",x"1d",x"1d"),
    (x"15",x"f0",x"f1",x"f8",x"00",x"02",x"1b",x"05",x"1b",x"ec",x"0e",x"eb",x"eb",x"0d",x"05",x"e4"),
    (x"f7",x"e2",x"e4",x"05",x"e4",x"02",x"0c",x"e9",x"ff",x"16",x"0a",x"e7",x"e2",x"13",x"1d",x"15"),
    (x"05",x"e9",x"f2",x"11",x"e2",x"06",x"08",x"0d",x"03",x"05",x"1a",x"20",x"f9",x"f3",x"e5",x"fa"),
    (x"17",x"e1",x"1e",x"f1",x"0e",x"f4",x"f6",x"01",x"fd",x"15",x"ed",x"ec",x"e1",x"0b",x"13",x"07"),
    (x"e8",x"0b",x"04",x"0f",x"18",x"e1",x"1b",x"04",x"eb",x"f3",x"1f",x"f1",x"15",x"04",x"ff",x"12"),
    (x"fa",x"e5",x"f6",x"12",x"0b",x"06",x"f2",x"00",x"13",x"fb",x"03",x"ea",x"05",x"fd",x"07",x"f8"),
    (x"1b",x"14",x"13",x"fd",x"f0",x"ed",x"ed",x"0c",x"00",x"05",x"ea",x"0e",x"ee",x"07",x"f0",x"03"),
    (x"09",x"f6",x"0d",x"12",x"f7",x"17",x"0f",x"12",x"01",x"06",x"0b",x"fd",x"0a",x"18",x"e3",x"ea"),
    (x"e1",x"19",x"07",x"f3",x"f1",x"f4",x"19",x"fe",x"ec",x"06",x"0f",x"19",x"15",x"18",x"19",x"ed"),
    (x"fe",x"f5",x"fe",x"1c",x"f7",x"0a",x"fb",x"16",x"ed",x"01",x"12",x"e5",x"eb",x"f7",x"0f",x"0f"),
    (x"e6",x"e5",x"f7",x"10",x"08",x"0b",x"1c",x"e9",x"ed",x"f5",x"ed",x"e2",x"fe",x"e4",x"e5",x"06"),
    (x"f9",x"01",x"1a",x"f0",x"fb",x"f4",x"ec",x"1c",x"1d",x"f1",x"fa",x"0b",x"f8",x"fd",x"ea",x"fa"),
    (x"1a",x"ec",x"e8",x"f6",x"00",x"e2",x"11",x"03",x"f6",x"01",x"0c",x"f6",x"1c",x"fe",x"0b",x"e5"),
    (x"1b",x"f5",x"04",x"03",x"f0",x"1d",x"e3",x"0c",x"f8",x"1f",x"02",x"1e",x"1a",x"1b",x"07",x"1e"),
    (x"15",x"1f",x"fe",x"f3",x"1c",x"05",x"fe",x"e5",x"ed",x"f7",x"f6",x"08",x"1d",x"14",x"ff",x"08"),
    (x"0b",x"fb",x"f0",x"1a",x"0a",x"f8",x"e2",x"0f",x"1d",x"f4",x"1f",x"fb",x"14",x"08",x"0d",x"03"),
    (x"f0",x"f4",x"f8",x"08",x"ec",x"1e",x"f1",x"f0",x"f8",x"e7",x"ee",x"e1",x"e5",x"ea",x"f4",x"f8"),
    (x"1c",x"13",x"ee",x"1a",x"f0",x"ed",x"05",x"11",x"fb",x"03",x"ed",x"1f",x"0d",x"f5",x"17",x"02"),
    (x"eb",x"e1",x"16",x"0d",x"fb",x"f3",x"19",x"18",x"ee",x"1f",x"fd",x"f5",x"01",x"fb",x"fb",x"19"),
    (x"06",x"f4",x"f5",x"20",x"18",x"e6",x"fa",x"ed",x"e7",x"04",x"ec",x"11",x"1e",x"07",x"e1",x"16"),
    (x"11",x"e7",x"f7",x"1b",x"f4",x"e3",x"ed",x"ee",x"10",x"f7",x"ed",x"f8",x"f9",x"f7",x"15",x"11"),
    (x"ee",x"fd",x"e3",x"fb",x"17",x"05",x"04",x"1a",x"05",x"f9",x"1d",x"e4",x"ef",x"f4",x"eb",x"e6"),
    (x"1b",x"0a",x"fb",x"ea",x"eb",x"00",x"1a",x"10",x"e4",x"e3",x"e3",x"18",x"19",x"12",x"10",x"0f"),
    (x"e4",x"ff",x"17",x"ed",x"17",x"f2",x"0d",x"f0",x"1f",x"e4",x"0d",x"f5",x"0d",x"0c",x"fb",x"06"),
    (x"08",x"0e",x"01",x"f0",x"1d",x"14",x"f9",x"10",x"f0",x"fc",x"f1",x"0c",x"0c",x"e1",x"e4",x"e9"),
    (x"0c",x"fd",x"fe",x"0a",x"f4",x"13",x"f7",x"08",x"08",x"ff",x"fd",x"e2",x"e9",x"16",x"e7",x"0d"),
    (x"fe",x"0b",x"fa",x"16",x"e4",x"f3",x"f4",x"17",x"1a",x"eb",x"0d",x"fe",x"e1",x"0c",x"1b",x"18")
);
--------------------------------------------------------
constant OUT_WIDTH     :  integer := 32;
end package;