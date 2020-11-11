--------------------------------------------------------
-- This file is generated by Delirium - Sma-RTY SAS 
-- Wed Nov 11 05:55:45 2020
--------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
library work;
    use work.Delirium.all;

package params is
--------------------------------------------------------
-- INPUT
constant INPUT_CHANNELS    : integer := 16;
constant INPUT_IMAGE_WIDTH : integer := 112;
--------------------------------------------------------
--Conv_0
constant Conv_0_IMAGE_WIDTH  :  integer := 112;
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
    (x"f5",x"e3",x"ea",x"e4",x"0b",x"06",x"18",x"f8",x"10",x"02",x"e5",x"1a",x"eb",x"11",x"13",x"e4"),
    (x"0b",x"1c",x"09",x"0e",x"ff",x"fe",x"09",x"fb",x"0c",x"06",x"08",x"e7",x"0b",x"04",x"fb",x"10"),
    (x"ff",x"1b",x"fc",x"08",x"f8",x"00",x"fa",x"03",x"17",x"ee",x"18",x"0a",x"13",x"f2",x"e1",x"f4"),
    (x"11",x"1a",x"e4",x"02",x"e4",x"eb",x"17",x"00",x"1f",x"09",x"e7",x"14",x"00",x"e8",x"1a",x"13"),
    (x"01",x"f7",x"1f",x"fc",x"1c",x"0e",x"e7",x"1b",x"04",x"00",x"e4",x"fc",x"ec",x"e8",x"0e",x"e2"),
    (x"f8",x"01",x"12",x"e8",x"08",x"02",x"07",x"07",x"f1",x"19",x"15",x"1d",x"f7",x"18",x"e0",x"f9"),
    (x"15",x"ef",x"07",x"11",x"0e",x"15",x"ee",x"f5",x"1a",x"0c",x"04",x"e1",x"05",x"f4",x"19",x"18"),
    (x"03",x"16",x"14",x"0b",x"1a",x"fb",x"05",x"ed",x"12",x"f7",x"f9",x"fd",x"f0",x"fa",x"f3",x"04"),
    (x"e2",x"13",x"20",x"0a",x"fd",x"18",x"e3",x"ea",x"0a",x"15",x"04",x"e8",x"f7",x"0f",x"07",x"e2"),
    (x"10",x"1b",x"15",x"e7",x"f2",x"1a",x"1a",x"ed",x"ee",x"e1",x"05",x"0e",x"f9",x"e8",x"fa",x"11"),
    (x"1c",x"09",x"f1",x"f5",x"ef",x"ff",x"08",x"ed",x"16",x"f0",x"01",x"1a",x"f6",x"f9",x"1f",x"e7"),
    (x"ec",x"11",x"ef",x"e9",x"e6",x"e7",x"09",x"1b",x"08",x"e9",x"f7",x"19",x"f2",x"f3",x"08",x"f7"),
    (x"08",x"ea",x"f0",x"f8",x"fa",x"0d",x"f8",x"1a",x"e9",x"03",x"e8",x"ea",x"07",x"03",x"0b",x"19"),
    (x"08",x"1f",x"f2",x"f2",x"00",x"00",x"fe",x"f2",x"f7",x"01",x"e6",x"18",x"1f",x"e2",x"f7",x"fa"),
    (x"16",x"0c",x"e1",x"0a",x"fc",x"0d",x"e7",x"02",x"09",x"eb",x"04",x"03",x"e3",x"0e",x"1a",x"0f"),
    (x"fc",x"07",x"1f",x"15",x"11",x"05",x"1a",x"ec",x"05",x"01",x"f3",x"01",x"07",x"0d",x"fb",x"eb"),
    (x"fd",x"0f",x"02",x"0c",x"f5",x"f6",x"e9",x"1c",x"0c",x"09",x"14",x"e9",x"03",x"18",x"eb",x"f7"),
    (x"e1",x"19",x"04",x"ee",x"ea",x"1f",x"f6",x"13",x"e9",x"fd",x"f0",x"f3",x"e3",x"f1",x"18",x"e2"),
    (x"03",x"fb",x"ee",x"1c",x"1c",x"18",x"02",x"fc",x"19",x"1f",x"16",x"1a",x"f8",x"0c",x"15",x"e4"),
    (x"03",x"fa",x"e5",x"0a",x"08",x"e4",x"03",x"f1",x"1e",x"fc",x"e4",x"fd",x"07",x"17",x"15",x"13"),
    (x"fa",x"0b",x"19",x"17",x"0a",x"13",x"1a",x"f4",x"e2",x"01",x"e4",x"e9",x"f3",x"04",x"fa",x"fc"),
    (x"e1",x"ee",x"05",x"06",x"ed",x"07",x"16",x"04",x"11",x"04",x"e8",x"f8",x"10",x"fe",x"fc",x"10"),
    (x"e4",x"04",x"0e",x"13",x"e6",x"0d",x"f8",x"f7",x"03",x"13",x"f7",x"e4",x"ea",x"09",x"f1",x"ee"),
    (x"e3",x"16",x"f7",x"0d",x"e9",x"ee",x"01",x"f7",x"0a",x"e0",x"ff",x"e2",x"10",x"e3",x"02",x"f3"),
    (x"f7",x"f7",x"f1",x"ee",x"0a",x"08",x"fa",x"17",x"14",x"e3",x"0d",x"16",x"14",x"1d",x"17",x"fd"),
    (x"07",x"00",x"f1",x"18",x"1a",x"e7",x"f1",x"12",x"fb",x"18",x"0e",x"1f",x"ef",x"e6",x"1e",x"11"),
    (x"ef",x"16",x"f9",x"fb",x"17",x"f3",x"12",x"02",x"13",x"0f",x"02",x"1a",x"0d",x"0e",x"09",x"06"),
    (x"e6",x"0f",x"fe",x"03",x"fa",x"0a",x"fa",x"f2",x"17",x"f7",x"17",x"01",x"09",x"0e",x"20",x"1a"),
    (x"0b",x"f1",x"eb",x"1f",x"16",x"f7",x"0b",x"16",x"fd",x"ec",x"fe",x"19",x"15",x"07",x"08",x"f2"),
    (x"e0",x"f3",x"02",x"0a",x"10",x"02",x"ef",x"02",x"eb",x"1f",x"07",x"1c",x"1b",x"04",x"f0",x"f8"),
    (x"08",x"19",x"19",x"fb",x"08",x"f6",x"19",x"12",x"ff",x"f0",x"f1",x"1a",x"f7",x"04",x"02",x"0a"),
    (x"14",x"f2",x"ff",x"f1",x"1e",x"14",x"fd",x"13",x"10",x"03",x"ee",x"1d",x"1a",x"e4",x"f5",x"ff")
);
--------------------------------------------------------
constant OUT_WIDTH     :  integer := 32;
end package;