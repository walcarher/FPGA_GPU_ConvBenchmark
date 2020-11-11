--------------------------------------------------------
-- This file is generated by Delirium - Sma-RTY SAS 
-- Wed Nov 11 05:56:35 2020
--------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
library work;
    use work.Delirium.all;

package params is
--------------------------------------------------------
-- INPUT
constant INPUT_CHANNELS    : integer := 64;
constant INPUT_IMAGE_WIDTH : integer := 56;
--------------------------------------------------------
--Conv_0
constant Conv_0_IMAGE_WIDTH  :  integer := 56;
constant Conv_0_IN_SIZE      :  integer := 64;
constant Conv_0_OUT_SIZE     :  integer := 16;
constant Conv_0_MULT_STYLE   :  string  := "dsp";
constant Conv_0_PIPELINE     :  boolean := True;
constant Conv_0_KERNEL_SIZE  :  integer := 1;
constant Conv_0_PADDING      :  boolean := FALSE;
constant Conv_0_STRIDE       :  positive:= 1;
constant Conv_0_BIAS_VALUE   :  data_array  (0 to Conv_0_OUT_SIZE - 1) := 
    (x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00");
constant Conv_0_KERNEL_VALUE :  data_matrix (0 to Conv_0_OUT_SIZE - 1 ,  0 to Conv_0_IN_SIZE * Conv_0_KERNEL_SIZE * Conv_0_KERNEL_SIZE - 1) := (
    (x"01",x"fd",x"f4",x"0a",x"fe",x"fe",x"0b",x"0e",x"0c",x"f6",x"f0",x"f8",x"01",x"fa",x"0a",x"fe",x"fa",x"0f",x"fb",x"f8",x"10",x"0e",x"07",x"fd",x"f8",x"f8",x"06",x"f3",x"04",x"09",x"fc",x"fe",x"09",x"f7",x"0f",x"f8",x"01",x"fb",x"0c",x"03",x"07",x"0f",x"07",x"f1",x"f9",x"10",x"0c",x"f5",x"f8",x"fa",x"00",x"05",x"0a",x"f3",x"f8",x"09",x"f6",x"0c",x"f7",x"02",x"07",x"0f",x"f3",x"f4"),
    (x"f8",x"fc",x"09",x"0f",x"f3",x"f0",x"fd",x"07",x"fc",x"06",x"fd",x"f0",x"0f",x"0c",x"fe",x"0e",x"fa",x"0b",x"0d",x"fa",x"f4",x"0e",x"f9",x"03",x"f5",x"07",x"fe",x"0e",x"06",x"08",x"0a",x"06",x"f5",x"f4",x"ff",x"06",x"f1",x"f9",x"f5",x"0f",x"09",x"fb",x"00",x"f2",x"f9",x"f7",x"fc",x"0c",x"f7",x"f0",x"f8",x"fd",x"01",x"0e",x"04",x"0a",x"0b",x"f5",x"fc",x"04",x"fd",x"0d",x"01",x"f0"),
    (x"00",x"fc",x"01",x"f9",x"00",x"f4",x"05",x"f8",x"f7",x"f7",x"f5",x"0f",x"0e",x"f2",x"f6",x"02",x"f9",x"05",x"f6",x"0a",x"0f",x"07",x"ff",x"fb",x"f8",x"f6",x"06",x"01",x"0c",x"fb",x"0f",x"f9",x"06",x"04",x"fa",x"05",x"f6",x"fb",x"0a",x"f8",x"04",x"f7",x"02",x"f2",x"00",x"0b",x"09",x"f9",x"fe",x"05",x"0f",x"09",x"01",x"f7",x"06",x"02",x"09",x"0f",x"09",x"f6",x"0f",x"f6",x"08",x"01"),
    (x"f3",x"0b",x"f5",x"f2",x"fc",x"f4",x"f8",x"f3",x"09",x"0b",x"0c",x"05",x"f1",x"08",x"0b",x"f2",x"10",x"0c",x"fe",x"02",x"f5",x"08",x"fc",x"f1",x"fb",x"10",x"fb",x"08",x"f9",x"f1",x"fa",x"0b",x"f8",x"ff",x"f9",x"05",x"00",x"0e",x"04",x"01",x"fd",x"fd",x"f3",x"ff",x"00",x"0d",x"0a",x"f5",x"06",x"0c",x"00",x"0a",x"fb",x"fc",x"03",x"fd",x"0f",x"f1",x"ff",x"fc",x"f4",x"07",x"f1",x"f4"),
    (x"0c",x"f8",x"fc",x"f8",x"f6",x"01",x"fe",x"fa",x"00",x"00",x"03",x"0d",x"f2",x"fb",x"06",x"0d",x"fb",x"07",x"08",x"02",x"00",x"fd",x"0a",x"f7",x"fc",x"0b",x"fb",x"f3",x"00",x"0a",x"fd",x"0a",x"ff",x"f6",x"04",x"01",x"f0",x"06",x"0e",x"05",x"f7",x"0c",x"fb",x"10",x"0e",x"04",x"00",x"04",x"02",x"08",x"08",x"00",x"0c",x"00",x"f7",x"06",x"ff",x"0c",x"fd",x"f4",x"07",x"0e",x"fd",x"f4"),
    (x"00",x"f1",x"01",x"0d",x"08",x"0d",x"ff",x"fd",x"08",x"fe",x"0d",x"fb",x"10",x"10",x"f0",x"f5",x"f1",x"00",x"05",x"f7",x"f3",x"fd",x"0b",x"f7",x"09",x"0c",x"fb",x"fb",x"ff",x"f6",x"fe",x"fc",x"0f",x"04",x"01",x"0e",x"f2",x"0b",x"fa",x"fd",x"0a",x"f0",x"07",x"01",x"09",x"f2",x"00",x"f0",x"0d",x"01",x"f5",x"f0",x"06",x"fe",x"f1",x"f6",x"0a",x"0a",x"fd",x"f8",x"f6",x"0d",x"0c",x"04"),
    (x"fb",x"f8",x"0a",x"f3",x"02",x"0c",x"f4",x"f2",x"fb",x"f3",x"00",x"03",x"02",x"07",x"01",x"0b",x"f6",x"00",x"f4",x"0e",x"06",x"ff",x"0f",x"06",x"f5",x"f4",x"ff",x"f7",x"f3",x"fc",x"03",x"0d",x"01",x"0e",x"0c",x"f5",x"09",x"fe",x"09",x"0f",x"f9",x"02",x"f7",x"02",x"01",x"f9",x"ff",x"f1",x"04",x"f7",x"f9",x"fb",x"f9",x"f8",x"f0",x"0a",x"09",x"05",x"06",x"02",x"ff",x"f3",x"f4",x"0d"),
    (x"06",x"fd",x"07",x"0f",x"ff",x"fe",x"0a",x"fa",x"f4",x"06",x"00",x"0d",x"fc",x"fd",x"fc",x"f9",x"fe",x"f3",x"0a",x"f7",x"f3",x"08",x"ff",x"06",x"05",x"00",x"00",x"07",x"0c",x"09",x"f2",x"f4",x"fc",x"09",x"f4",x"f3",x"fe",x"f4",x"f3",x"05",x"01",x"05",x"00",x"05",x"07",x"f9",x"f9",x"fa",x"fe",x"0e",x"f0",x"03",x"f8",x"00",x"fe",x"f4",x"f4",x"0d",x"0b",x"05",x"fe",x"f4",x"f3",x"f5"),
    (x"02",x"04",x"0d",x"f8",x"f4",x"00",x"f9",x"f5",x"f0",x"07",x"0d",x"fc",x"fd",x"02",x"fc",x"00",x"fe",x"0d",x"08",x"09",x"02",x"03",x"09",x"f5",x"08",x"f4",x"00",x"f9",x"0c",x"ff",x"f9",x"f1",x"f5",x"09",x"08",x"05",x"06",x"06",x"f1",x"0c",x"ff",x"f0",x"01",x"f5",x"f8",x"07",x"f8",x"06",x"08",x"f2",x"f6",x"f1",x"f8",x"f3",x"03",x"f2",x"00",x"f2",x"0f",x"01",x"04",x"02",x"f6",x"0b"),
    (x"f4",x"fc",x"07",x"00",x"fe",x"fc",x"f2",x"f1",x"0d",x"0b",x"f2",x"10",x"fc",x"f5",x"f9",x"02",x"04",x"07",x"f8",x"fc",x"00",x"07",x"fb",x"01",x"08",x"fa",x"ff",x"08",x"f8",x"f9",x"0a",x"fe",x"fd",x"f9",x"00",x"fe",x"f8",x"00",x"08",x"0a",x"08",x"f8",x"0f",x"fb",x"02",x"09",x"05",x"0d",x"f2",x"fc",x"f9",x"f9",x"f9",x"fa",x"00",x"08",x"f9",x"0f",x"ff",x"00",x"fd",x"fa",x"07",x"f3"),
    (x"04",x"fd",x"f9",x"03",x"fb",x"fa",x"03",x"f6",x"06",x"09",x"01",x"08",x"fb",x"0e",x"05",x"f0",x"0c",x"0a",x"f7",x"0e",x"0a",x"09",x"fc",x"f9",x"02",x"f6",x"f3",x"f2",x"0c",x"0c",x"0c",x"01",x"04",x"fe",x"08",x"f9",x"02",x"fa",x"03",x"ff",x"02",x"02",x"00",x"01",x"f3",x"0b",x"f3",x"f3",x"09",x"f3",x"08",x"03",x"01",x"f6",x"fc",x"07",x"f6",x"fc",x"07",x"0f",x"04",x"00",x"06",x"f1"),
    (x"fb",x"f5",x"fa",x"f8",x"07",x"ff",x"0e",x"05",x"f5",x"f3",x"09",x"f7",x"04",x"0b",x"f2",x"f3",x"0c",x"ff",x"02",x"f9",x"f3",x"f9",x"f4",x"0d",x"0b",x"f7",x"0b",x"fc",x"0f",x"0f",x"01",x"06",x"02",x"f8",x"02",x"0a",x"09",x"f3",x"fe",x"fa",x"01",x"f6",x"05",x"f4",x"06",x"0a",x"fa",x"f3",x"f3",x"fb",x"f9",x"0c",x"01",x"f3",x"f9",x"f6",x"f8",x"f0",x"f1",x"fe",x"f0",x"f6",x"04",x"f3"),
    (x"0b",x"f3",x"0c",x"f7",x"fd",x"0e",x"f4",x"0a",x"06",x"f2",x"00",x"0a",x"01",x"05",x"f6",x"0d",x"ff",x"0d",x"f5",x"ff",x"fc",x"00",x"0e",x"f9",x"0b",x"f1",x"06",x"f2",x"0b",x"fa",x"09",x"f2",x"f7",x"06",x"fd",x"0b",x"f2",x"0f",x"04",x"0e",x"f7",x"04",x"f6",x"07",x"08",x"06",x"04",x"0b",x"0b",x"f4",x"f5",x"fe",x"f2",x"f3",x"0f",x"f5",x"f8",x"f3",x"0f",x"07",x"0e",x"00",x"0d",x"f6"),
    (x"f0",x"08",x"f6",x"f2",x"0b",x"fd",x"0f",x"fa",x"03",x"fb",x"09",x"06",x"0c",x"fc",x"fc",x"02",x"f9",x"f4",x"0c",x"06",x"fa",x"f6",x"fe",x"f5",x"03",x"08",x"08",x"00",x"00",x"06",x"00",x"f4",x"f6",x"03",x"0e",x"ff",x"fd",x"09",x"f5",x"0d",x"01",x"f1",x"f2",x"04",x"04",x"f8",x"0d",x"08",x"fb",x"08",x"f8",x"f4",x"f0",x"fc",x"00",x"0e",x"0d",x"fa",x"f7",x"f4",x"f9",x"01",x"f2",x"02"),
    (x"0f",x"05",x"0a",x"0a",x"fc",x"0a",x"0b",x"03",x"f4",x"f7",x"f9",x"0d",x"08",x"f4",x"00",x"ff",x"ff",x"f1",x"00",x"07",x"0d",x"f1",x"f2",x"04",x"0c",x"f4",x"0f",x"08",x"03",x"00",x"f9",x"f5",x"ff",x"0b",x"0a",x"07",x"07",x"07",x"0a",x"fa",x"fd",x"01",x"f5",x"f6",x"fc",x"0b",x"04",x"fa",x"10",x"09",x"08",x"08",x"0b",x"08",x"f9",x"06",x"f9",x"04",x"04",x"01",x"f5",x"fa",x"fd",x"fc"),
    (x"08",x"09",x"fc",x"03",x"ff",x"0a",x"06",x"09",x"00",x"fc",x"0a",x"0a",x"f6",x"f5",x"f3",x"0d",x"f4",x"0c",x"02",x"f2",x"ff",x"07",x"04",x"fb",x"fe",x"0f",x"08",x"f6",x"01",x"0e",x"fe",x"06",x"0f",x"07",x"f6",x"0a",x"07",x"05",x"03",x"f7",x"fb",x"f8",x"09",x"f3",x"fe",x"fd",x"0e",x"fa",x"0b",x"06",x"0f",x"fc",x"0d",x"fe",x"fc",x"0d",x"0b",x"f6",x"04",x"0d",x"fa",x"fb",x"00",x"fa")
);
--------------------------------------------------------
constant OUT_WIDTH     :  integer := 16;
end package;