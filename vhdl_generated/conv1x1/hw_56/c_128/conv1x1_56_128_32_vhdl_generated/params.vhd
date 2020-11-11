--------------------------------------------------------
-- This file is generated by Delirium - Sma-RTY SAS 
-- Wed Nov 11 05:56:43 2020
--------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
library work;
    use work.Delirium.all;

package params is
--------------------------------------------------------
-- INPUT
constant INPUT_CHANNELS    : integer := 128;
constant INPUT_IMAGE_WIDTH : integer := 56;
--------------------------------------------------------
--Conv_0
constant Conv_0_IMAGE_WIDTH  :  integer := 56;
constant Conv_0_IN_SIZE      :  integer := 128;
constant Conv_0_OUT_SIZE     :  integer := 32;
constant Conv_0_MULT_STYLE   :  string  := "dsp";
constant Conv_0_PIPELINE     :  boolean := True;
constant Conv_0_KERNEL_SIZE  :  integer := 1;
constant Conv_0_PADDING      :  boolean := FALSE;
constant Conv_0_STRIDE       :  positive:= 1;
constant Conv_0_BIAS_VALUE   :  data_array  (0 to Conv_0_OUT_SIZE - 1) := 
    (x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00",x"00");
constant Conv_0_KERNEL_VALUE :  data_matrix (0 to Conv_0_OUT_SIZE - 1 ,  0 to Conv_0_IN_SIZE * Conv_0_KERNEL_SIZE * Conv_0_KERNEL_SIZE - 1) := (
    (x"03",x"02",x"08",x"ff",x"02",x"fb",x"01",x"f7",x"09",x"00",x"06",x"fd",x"08",x"05",x"08",x"04",x"04",x"fe",x"06",x"ff",x"0a",x"f6",x"f9",x"ff",x"fb",x"fa",x"f6",x"fd",x"f9",x"f6",x"f8",x"03",x"02",x"f8",x"06",x"07",x"f8",x"fc",x"01",x"f5",x"fe",x"01",x"01",x"0a",x"06",x"fe",x"fc",x"00",x"03",x"f9",x"02",x"08",x"fa",x"06",x"fc",x"09",x"fd",x"00",x"02",x"f8",x"0a",x"f8",x"f6",x"fc",x"09",x"00",x"fd",x"f8",x"05",x"f9",x"04",x"f6",x"fa",x"fb",x"00",x"01",x"f7",x"0a",x"05",x"fb",x"f9",x"04",x"f9",x"fb",x"09",x"03",x"07",x"f7",x"f6",x"00",x"fa",x"04",x"04",x"07",x"02",x"fc",x"02",x"f6",x"06",x"fa",x"fb",x"01",x"f6",x"fa",x"fd",x"f9",x"fa",x"fa",x"f8",x"f5",x"fb",x"00",x"fe",x"05",x"f6",x"01",x"fd",x"f7",x"07",x"02",x"02",x"02",x"06",x"05",x"ff",x"03",x"09",x"0a"),
    (x"08",x"fd",x"06",x"02",x"0b",x"fd",x"fc",x"fa",x"f7",x"fe",x"01",x"f5",x"01",x"fc",x"f8",x"08",x"08",x"fc",x"07",x"08",x"f8",x"fa",x"07",x"05",x"fd",x"fc",x"ff",x"0b",x"fb",x"fb",x"fa",x"07",x"fe",x"05",x"01",x"f5",x"06",x"fb",x"0a",x"00",x"09",x"03",x"09",x"f8",x"f9",x"04",x"fd",x"ff",x"04",x"09",x"fb",x"0a",x"03",x"02",x"09",x"00",x"08",x"ff",x"02",x"08",x"ff",x"fa",x"f6",x"0b",x"0a",x"f7",x"03",x"f5",x"ff",x"05",x"0a",x"fe",x"f7",x"0b",x"08",x"02",x"06",x"f6",x"0b",x"f5",x"f6",x"fd",x"fa",x"fa",x"00",x"f7",x"fc",x"fb",x"0a",x"04",x"02",x"05",x"f6",x"06",x"09",x"07",x"0b",x"ff",x"06",x"f7",x"f5",x"f9",x"fe",x"08",x"fc",x"08",x"0b",x"fc",x"05",x"f7",x"08",x"fe",x"07",x"f8",x"fb",x"ff",x"0b",x"01",x"08",x"f9",x"03",x"09",x"fd",x"fe",x"f7",x"00",x"0a",x"06"),
    (x"f6",x"00",x"0a",x"06",x"ff",x"04",x"07",x"f9",x"03",x"08",x"f9",x"f9",x"01",x"fd",x"fb",x"fd",x"fb",x"f7",x"fc",x"fd",x"f7",x"f5",x"fd",x"02",x"fd",x"08",x"07",x"04",x"f7",x"09",x"f7",x"f9",x"09",x"08",x"f8",x"04",x"05",x"08",x"f9",x"0a",x"ff",x"f6",x"f7",x"00",x"07",x"fa",x"fe",x"03",x"02",x"f8",x"03",x"05",x"fc",x"00",x"01",x"f5",x"f9",x"f6",x"01",x"fa",x"fd",x"f7",x"fb",x"f8",x"06",x"fd",x"f5",x"07",x"ff",x"ff",x"02",x"05",x"fc",x"00",x"04",x"ff",x"fc",x"fa",x"fc",x"fd",x"fa",x"ff",x"04",x"f8",x"fb",x"05",x"fc",x"0a",x"06",x"02",x"07",x"09",x"f7",x"0a",x"f6",x"08",x"04",x"f7",x"fa",x"06",x"06",x"01",x"fc",x"fd",x"0a",x"06",x"f5",x"f5",x"f9",x"ff",x"07",x"00",x"03",x"fc",x"07",x"02",x"04",x"08",x"04",x"fa",x"f6",x"03",x"05",x"05",x"08",x"fc",x"0b",x"03"),
    (x"ff",x"0a",x"0b",x"f5",x"03",x"fb",x"02",x"fa",x"fd",x"f8",x"fd",x"fe",x"fd",x"f8",x"02",x"ff",x"0a",x"0a",x"0b",x"0a",x"00",x"05",x"0a",x"01",x"ff",x"f8",x"fd",x"03",x"05",x"0b",x"fb",x"01",x"fa",x"f8",x"07",x"f6",x"f5",x"08",x"03",x"f7",x"fc",x"06",x"04",x"f6",x"08",x"f7",x"fa",x"fa",x"f6",x"f6",x"05",x"ff",x"02",x"f5",x"fd",x"fb",x"01",x"f9",x"fd",x"f5",x"01",x"00",x"f7",x"fd",x"fc",x"f5",x"08",x"0a",x"04",x"fc",x"f7",x"f9",x"05",x"fb",x"f6",x"05",x"fa",x"f6",x"fa",x"01",x"fe",x"03",x"00",x"fe",x"00",x"fa",x"08",x"05",x"0b",x"f5",x"f6",x"fb",x"f8",x"02",x"ff",x"0b",x"09",x"03",x"00",x"fe",x"f5",x"08",x"00",x"03",x"fd",x"04",x"03",x"fa",x"06",x"02",x"f9",x"09",x"0a",x"07",x"02",x"04",x"f9",x"0b",x"02",x"ff",x"01",x"03",x"f8",x"04",x"02",x"fc",x"0a",x"00"),
    (x"00",x"fa",x"f5",x"fa",x"03",x"00",x"f5",x"f7",x"f7",x"06",x"02",x"08",x"02",x"0b",x"f8",x"f6",x"f9",x"08",x"fa",x"fa",x"fd",x"f7",x"f7",x"08",x"00",x"f5",x"fe",x"09",x"f8",x"f7",x"fd",x"07",x"0a",x"fc",x"fa",x"fa",x"f7",x"f6",x"00",x"06",x"03",x"fc",x"ff",x"fa",x"05",x"06",x"00",x"01",x"09",x"f5",x"05",x"07",x"fc",x"02",x"fd",x"f6",x"f8",x"03",x"f8",x"f5",x"05",x"00",x"09",x"f9",x"fd",x"f7",x"f8",x"f9",x"09",x"01",x"fb",x"fc",x"fe",x"f6",x"f5",x"fb",x"08",x"02",x"f6",x"fa",x"03",x"08",x"fe",x"00",x"fd",x"05",x"01",x"05",x"02",x"fc",x"f9",x"06",x"00",x"f7",x"fb",x"fa",x"05",x"08",x"fb",x"ff",x"02",x"fe",x"06",x"05",x"f7",x"fb",x"fa",x"04",x"f7",x"f6",x"fa",x"f7",x"07",x"01",x"fd",x"07",x"ff",x"fb",x"0b",x"fd",x"03",x"08",x"05",x"0b",x"08",x"01",x"fc",x"04"),
    (x"f6",x"f9",x"00",x"fe",x"fa",x"f8",x"f9",x"06",x"08",x"f6",x"fa",x"0a",x"04",x"09",x"0b",x"01",x"fe",x"f6",x"08",x"0a",x"f9",x"03",x"0b",x"01",x"08",x"05",x"f9",x"fb",x"06",x"06",x"fb",x"09",x"f7",x"f8",x"f9",x"07",x"08",x"ff",x"f6",x"fd",x"fc",x"ff",x"ff",x"02",x"fd",x"03",x"f7",x"f6",x"07",x"08",x"f7",x"05",x"03",x"f5",x"03",x"f6",x"04",x"f7",x"05",x"08",x"07",x"f9",x"f9",x"f6",x"00",x"fe",x"02",x"f8",x"f7",x"f8",x"ff",x"00",x"fe",x"07",x"fb",x"fc",x"fd",x"07",x"f5",x"05",x"0b",x"03",x"05",x"f9",x"fd",x"fe",x"09",x"05",x"fa",x"f8",x"fd",x"0b",x"09",x"ff",x"03",x"05",x"03",x"fd",x"00",x"fb",x"f8",x"02",x"f8",x"0a",x"f6",x"f7",x"01",x"01",x"f5",x"0b",x"0b",x"08",x"09",x"02",x"fb",x"fe",x"09",x"05",x"01",x"f9",x"07",x"f6",x"01",x"f7",x"f9",x"00",x"ff",x"f6"),
    (x"f7",x"09",x"0a",x"fb",x"02",x"0b",x"0b",x"06",x"0a",x"07",x"0a",x"05",x"f7",x"fd",x"08",x"06",x"01",x"03",x"0a",x"fc",x"ff",x"f8",x"f5",x"02",x"05",x"ff",x"05",x"02",x"08",x"fb",x"fa",x"0b",x"03",x"02",x"08",x"06",x"08",x"fd",x"0b",x"ff",x"f6",x"04",x"01",x"fa",x"00",x"05",x"fa",x"fe",x"06",x"04",x"fe",x"f9",x"fb",x"ff",x"09",x"03",x"fc",x"0a",x"ff",x"ff",x"fa",x"04",x"f7",x"ff",x"04",x"05",x"05",x"fd",x"ff",x"fc",x"02",x"06",x"fb",x"fa",x"03",x"0b",x"fc",x"fb",x"fe",x"ff",x"fd",x"07",x"fa",x"f8",x"fe",x"f9",x"fc",x"04",x"f8",x"f6",x"f6",x"05",x"fc",x"fb",x"01",x"f8",x"fe",x"f5",x"f5",x"fb",x"f7",x"fe",x"02",x"f9",x"fa",x"f9",x"f6",x"0b",x"f8",x"07",x"fb",x"09",x"f8",x"fd",x"08",x"0a",x"f6",x"0a",x"f6",x"f9",x"fc",x"f8",x"f8",x"f6",x"04",x"fe",x"02",x"09"),
    (x"0a",x"f6",x"01",x"07",x"02",x"f7",x"01",x"f7",x"ff",x"f6",x"fb",x"f8",x"f5",x"08",x"07",x"08",x"fe",x"f8",x"f9",x"0a",x"fb",x"f8",x"f6",x"fa",x"00",x"0a",x"07",x"0b",x"fc",x"f8",x"06",x"07",x"0b",x"fc",x"f6",x"f7",x"fb",x"07",x"f8",x"f9",x"ff",x"fe",x"05",x"01",x"08",x"f6",x"fc",x"08",x"09",x"fc",x"09",x"f9",x"04",x"ff",x"f7",x"0b",x"0b",x"01",x"f9",x"09",x"fc",x"04",x"f6",x"fa",x"fa",x"00",x"f6",x"fc",x"01",x"ff",x"04",x"01",x"06",x"07",x"0b",x"fc",x"0b",x"f5",x"03",x"f8",x"0a",x"fb",x"f7",x"f8",x"02",x"fd",x"05",x"07",x"ff",x"fc",x"05",x"05",x"f8",x"08",x"f9",x"ff",x"00",x"0b",x"06",x"00",x"fe",x"f6",x"04",x"02",x"f5",x"fd",x"0a",x"ff",x"f8",x"f8",x"03",x"02",x"0a",x"01",x"07",x"fb",x"03",x"fd",x"05",x"fa",x"04",x"01",x"f9",x"f8",x"06",x"03",x"04",x"0a"),
    (x"fc",x"00",x"04",x"fd",x"ff",x"02",x"08",x"f7",x"04",x"07",x"fb",x"fb",x"fc",x"f8",x"09",x"ff",x"fa",x"03",x"0b",x"fb",x"05",x"04",x"f8",x"fc",x"ff",x"f6",x"03",x"03",x"fa",x"fd",x"fb",x"fd",x"fd",x"fb",x"01",x"09",x"fa",x"05",x"00",x"04",x"0b",x"f7",x"f9",x"fb",x"0a",x"f7",x"fb",x"0a",x"fe",x"fc",x"08",x"fd",x"f8",x"f9",x"09",x"f9",x"0b",x"fa",x"04",x"f7",x"f5",x"fb",x"02",x"f7",x"fa",x"00",x"02",x"fd",x"fb",x"06",x"f7",x"0b",x"0a",x"04",x"fc",x"05",x"0a",x"ff",x"0b",x"f8",x"f9",x"08",x"f9",x"f5",x"f8",x"f6",x"f7",x"fa",x"08",x"fd",x"fc",x"f6",x"f5",x"fd",x"f9",x"03",x"fa",x"fc",x"05",x"ff",x"ff",x"fc",x"fb",x"0a",x"00",x"06",x"01",x"f9",x"05",x"05",x"00",x"f7",x"fa",x"07",x"fa",x"fc",x"f9",x"08",x"09",x"fa",x"0a",x"00",x"ff",x"fc",x"08",x"0a",x"f6",x"07"),
    (x"01",x"0a",x"f7",x"05",x"0b",x"fd",x"fc",x"fc",x"fe",x"05",x"fd",x"08",x"02",x"ff",x"09",x"fa",x"06",x"01",x"05",x"06",x"fd",x"fe",x"fa",x"fd",x"ff",x"f9",x"05",x"09",x"03",x"f9",x"fc",x"05",x"01",x"04",x"f8",x"0a",x"08",x"ff",x"03",x"05",x"09",x"fe",x"f9",x"0b",x"f6",x"ff",x"05",x"f7",x"f7",x"06",x"06",x"f6",x"04",x"ff",x"fd",x"f6",x"07",x"fb",x"0b",x"03",x"f8",x"01",x"00",x"04",x"f6",x"fc",x"07",x"04",x"ff",x"05",x"06",x"08",x"0b",x"06",x"09",x"fa",x"0a",x"09",x"02",x"fe",x"fc",x"fe",x"f8",x"f7",x"02",x"0a",x"07",x"02",x"00",x"ff",x"fc",x"07",x"f7",x"fd",x"05",x"00",x"fe",x"fb",x"00",x"0a",x"01",x"f7",x"06",x"ff",x"fc",x"f6",x"fc",x"06",x"02",x"f5",x"01",x"03",x"fc",x"fb",x"f8",x"07",x"f7",x"f6",x"f8",x"f8",x"f8",x"f8",x"fe",x"fb",x"01",x"fb",x"0b",x"07"),
    (x"01",x"08",x"0a",x"02",x"fe",x"05",x"fc",x"fd",x"f6",x"fc",x"f8",x"ff",x"f7",x"05",x"fb",x"fb",x"04",x"f9",x"04",x"f8",x"fe",x"fa",x"fd",x"00",x"fa",x"f6",x"05",x"fb",x"06",x"08",x"01",x"06",x"00",x"05",x"fa",x"04",x"f8",x"09",x"04",x"fd",x"08",x"00",x"09",x"09",x"00",x"03",x"03",x"09",x"0b",x"fe",x"01",x"fc",x"f7",x"08",x"08",x"00",x"fe",x"0a",x"03",x"fb",x"ff",x"02",x"ff",x"0b",x"03",x"f5",x"09",x"fd",x"fd",x"fe",x"f8",x"05",x"f6",x"0b",x"fc",x"05",x"05",x"f8",x"f9",x"f8",x"fb",x"f9",x"0b",x"02",x"08",x"05",x"03",x"06",x"00",x"00",x"f9",x"f6",x"ff",x"03",x"05",x"fd",x"05",x"fc",x"04",x"00",x"09",x"03",x"fb",x"02",x"04",x"00",x"f7",x"02",x"04",x"ff",x"f8",x"fc",x"0a",x"09",x"f6",x"fb",x"0a",x"07",x"04",x"fa",x"06",x"f5",x"f8",x"f9",x"fd",x"09",x"05",x"08"),
    (x"fd",x"07",x"06",x"fb",x"05",x"05",x"fb",x"0b",x"fc",x"ff",x"f9",x"f8",x"f6",x"fc",x"fa",x"03",x"f9",x"0b",x"09",x"05",x"03",x"fb",x"06",x"0a",x"02",x"ff",x"06",x"03",x"07",x"08",x"00",x"fa",x"01",x"fb",x"0a",x"fd",x"0b",x"05",x"f9",x"f7",x"f5",x"0b",x"fc",x"f9",x"fa",x"fc",x"fc",x"01",x"01",x"f6",x"01",x"fa",x"f6",x"0b",x"08",x"04",x"fd",x"07",x"02",x"01",x"05",x"0b",x"05",x"04",x"09",x"f6",x"06",x"09",x"fd",x"fc",x"05",x"02",x"02",x"06",x"f6",x"f5",x"f7",x"0a",x"08",x"05",x"f8",x"06",x"f5",x"03",x"f9",x"0a",x"06",x"fb",x"0b",x"09",x"01",x"fe",x"f8",x"01",x"f5",x"0a",x"0b",x"f7",x"03",x"07",x"ff",x"fb",x"fc",x"08",x"fa",x"01",x"f8",x"ff",x"03",x"f9",x"02",x"fb",x"fc",x"f5",x"0a",x"fd",x"06",x"fe",x"03",x"ff",x"fb",x"fd",x"f6",x"f5",x"f5",x"06",x"0b",x"fd"),
    (x"fa",x"02",x"fd",x"fe",x"05",x"03",x"fc",x"f7",x"01",x"00",x"fd",x"02",x"06",x"03",x"09",x"f6",x"ff",x"fc",x"02",x"04",x"06",x"f8",x"03",x"02",x"fc",x"fb",x"02",x"08",x"05",x"07",x"07",x"fe",x"07",x"f9",x"fe",x"04",x"00",x"07",x"fd",x"fb",x"0a",x"08",x"f7",x"07",x"f6",x"f5",x"f8",x"06",x"fc",x"fd",x"fe",x"06",x"07",x"06",x"f6",x"f8",x"fa",x"f6",x"09",x"fa",x"fe",x"08",x"f7",x"02",x"fd",x"01",x"f9",x"ff",x"fe",x"0a",x"f6",x"09",x"06",x"fa",x"09",x"f7",x"09",x"f9",x"01",x"f6",x"03",x"f5",x"fc",x"fb",x"09",x"00",x"04",x"01",x"ff",x"03",x"01",x"fc",x"00",x"fe",x"ff",x"fc",x"fc",x"02",x"fe",x"09",x"fc",x"f8",x"ff",x"04",x"f6",x"f5",x"0a",x"fd",x"07",x"00",x"fc",x"07",x"01",x"05",x"fd",x"06",x"fd",x"06",x"02",x"05",x"ff",x"06",x"f8",x"f7",x"ff",x"01",x"08",x"fa"),
    (x"f9",x"09",x"fc",x"0b",x"01",x"fb",x"03",x"fd",x"fa",x"07",x"02",x"f7",x"00",x"fd",x"03",x"fb",x"03",x"f7",x"07",x"01",x"fc",x"fc",x"06",x"03",x"ff",x"00",x"fa",x"02",x"06",x"03",x"fc",x"ff",x"08",x"09",x"f8",x"fc",x"fc",x"07",x"09",x"0a",x"fc",x"09",x"f9",x"09",x"fe",x"f7",x"fd",x"f7",x"06",x"f6",x"f5",x"04",x"ff",x"f6",x"f8",x"f5",x"0a",x"f9",x"06",x"f8",x"05",x"f8",x"fb",x"fb",x"fe",x"09",x"06",x"fb",x"f5",x"05",x"01",x"08",x"fb",x"0b",x"fc",x"08",x"06",x"fe",x"03",x"fc",x"fc",x"00",x"04",x"09",x"f6",x"f6",x"fe",x"fd",x"f9",x"08",x"ff",x"fb",x"04",x"04",x"0a",x"f7",x"f9",x"09",x"fa",x"fe",x"09",x"07",x"f9",x"04",x"f7",x"01",x"08",x"07",x"07",x"fe",x"06",x"f7",x"01",x"fe",x"0b",x"09",x"04",x"fa",x"07",x"f5",x"ff",x"fb",x"09",x"01",x"fc",x"fa",x"0b",x"0a"),
    (x"04",x"f8",x"03",x"02",x"0b",x"04",x"fc",x"0a",x"04",x"fc",x"f9",x"fb",x"04",x"f7",x"f6",x"02",x"05",x"06",x"05",x"00",x"0a",x"05",x"fc",x"09",x"f7",x"f6",x"f7",x"fd",x"05",x"0a",x"f8",x"03",x"04",x"ff",x"07",x"0b",x"00",x"01",x"fb",x"f5",x"f6",x"02",x"fe",x"04",x"0a",x"05",x"0b",x"03",x"04",x"f6",x"09",x"04",x"00",x"f9",x"09",x"01",x"09",x"02",x"04",x"08",x"0a",x"09",x"07",x"fc",x"f6",x"06",x"0a",x"fc",x"f9",x"06",x"08",x"08",x"00",x"05",x"00",x"0a",x"0a",x"05",x"f6",x"02",x"f9",x"fe",x"01",x"01",x"fa",x"05",x"08",x"f8",x"f7",x"fc",x"08",x"02",x"fd",x"fa",x"fb",x"f9",x"fb",x"03",x"fb",x"fc",x"01",x"09",x"03",x"fc",x"02",x"0a",x"f5",x"fd",x"09",x"06",x"0a",x"f8",x"fa",x"09",x"0a",x"fd",x"f7",x"fa",x"fe",x"fc",x"09",x"f5",x"fe",x"06",x"07",x"fb",x"0b",x"f5"),
    (x"fd",x"04",x"0a",x"0b",x"03",x"fb",x"00",x"07",x"fd",x"f6",x"f9",x"f6",x"fd",x"07",x"f6",x"f9",x"02",x"ff",x"fb",x"fd",x"07",x"09",x"04",x"fa",x"07",x"fb",x"ff",x"01",x"ff",x"0b",x"fd",x"fe",x"08",x"07",x"ff",x"ff",x"fb",x"f9",x"02",x"fe",x"02",x"f5",x"f5",x"fa",x"02",x"08",x"f8",x"00",x"0b",x"f7",x"06",x"ff",x"08",x"f5",x"07",x"f9",x"f9",x"fc",x"04",x"f9",x"fe",x"fa",x"f6",x"04",x"07",x"05",x"ff",x"0a",x"09",x"fc",x"f8",x"fc",x"f6",x"fb",x"03",x"08",x"0b",x"f8",x"03",x"09",x"f8",x"fd",x"06",x"04",x"07",x"02",x"f6",x"07",x"05",x"08",x"00",x"0a",x"fa",x"09",x"08",x"fa",x"fc",x"f7",x"07",x"f8",x"01",x"09",x"f6",x"06",x"f5",x"fe",x"02",x"f9",x"08",x"03",x"01",x"06",x"fd",x"06",x"ff",x"f7",x"03",x"05",x"05",x"08",x"f5",x"fe",x"f8",x"f8",x"ff",x"01",x"fa",x"01"),
    (x"fd",x"00",x"00",x"fa",x"ff",x"fd",x"05",x"03",x"07",x"0b",x"0b",x"f5",x"01",x"fb",x"01",x"f5",x"04",x"fa",x"0b",x"f6",x"fe",x"03",x"0b",x"f9",x"fc",x"01",x"01",x"fd",x"07",x"02",x"f6",x"fd",x"00",x"f8",x"03",x"07",x"fa",x"09",x"f5",x"f6",x"01",x"02",x"f5",x"03",x"fa",x"03",x"01",x"ff",x"09",x"09",x"fe",x"fd",x"fc",x"fc",x"f9",x"05",x"f9",x"f9",x"09",x"f7",x"f8",x"f6",x"0a",x"fd",x"fb",x"f6",x"f5",x"09",x"02",x"f5",x"0b",x"08",x"08",x"fc",x"0a",x"f5",x"f5",x"fb",x"f5",x"07",x"08",x"02",x"06",x"f5",x"f8",x"fd",x"fa",x"05",x"08",x"fc",x"05",x"03",x"00",x"f8",x"fa",x"0a",x"f8",x"09",x"02",x"f5",x"09",x"f5",x"f8",x"fd",x"09",x"ff",x"fe",x"05",x"f7",x"08",x"00",x"f6",x"fb",x"ff",x"ff",x"fb",x"03",x"fe",x"01",x"08",x"02",x"00",x"03",x"08",x"fa",x"06",x"0b",x"02"),
    (x"00",x"fb",x"06",x"fe",x"0a",x"f7",x"f6",x"fe",x"02",x"f6",x"09",x"08",x"ff",x"fa",x"05",x"01",x"00",x"0a",x"fd",x"00",x"f6",x"fd",x"03",x"fe",x"0a",x"fa",x"0a",x"fe",x"f8",x"fe",x"fd",x"fb",x"02",x"fd",x"f6",x"f7",x"0b",x"f6",x"f7",x"ff",x"00",x"0b",x"f9",x"fd",x"00",x"f7",x"fe",x"06",x"02",x"03",x"04",x"f8",x"04",x"09",x"09",x"fe",x"fc",x"08",x"04",x"f5",x"f5",x"ff",x"07",x"07",x"06",x"f8",x"f8",x"f5",x"f5",x"fa",x"01",x"05",x"07",x"05",x"0a",x"00",x"f7",x"04",x"0a",x"04",x"fe",x"f8",x"02",x"fd",x"02",x"f6",x"f7",x"0a",x"fc",x"04",x"fd",x"08",x"fc",x"03",x"05",x"06",x"07",x"fc",x"f6",x"f7",x"01",x"f9",x"f7",x"f6",x"fa",x"f5",x"fe",x"f9",x"f8",x"01",x"01",x"07",x"f7",x"04",x"00",x"06",x"04",x"02",x"ff",x"03",x"f6",x"00",x"fd",x"fa",x"f8",x"fa",x"08",x"fe"),
    (x"fa",x"0a",x"00",x"fe",x"fd",x"08",x"00",x"fe",x"05",x"0a",x"05",x"fa",x"09",x"fe",x"fc",x"fa",x"fd",x"02",x"fc",x"f9",x"f5",x"05",x"04",x"04",x"08",x"01",x"fc",x"01",x"01",x"f6",x"ff",x"02",x"01",x"fc",x"0b",x"03",x"00",x"fb",x"0a",x"0a",x"0a",x"f5",x"05",x"06",x"f6",x"f6",x"03",x"ff",x"f7",x"05",x"04",x"0b",x"f5",x"01",x"fc",x"fa",x"f8",x"03",x"fe",x"01",x"03",x"0b",x"00",x"f7",x"f8",x"01",x"ff",x"f7",x"fe",x"fe",x"f9",x"fe",x"08",x"03",x"00",x"fe",x"fa",x"09",x"06",x"02",x"09",x"02",x"f8",x"f9",x"f7",x"f6",x"04",x"f8",x"06",x"06",x"fb",x"01",x"fc",x"00",x"02",x"06",x"09",x"f7",x"f7",x"fa",x"0b",x"0b",x"04",x"fd",x"07",x"f9",x"f8",x"fa",x"f8",x"00",x"0a",x"02",x"0a",x"fa",x"f9",x"04",x"ff",x"00",x"ff",x"07",x"0b",x"fb",x"fa",x"00",x"ff",x"fe",x"06",x"05"),
    (x"f8",x"09",x"07",x"00",x"01",x"f6",x"f5",x"08",x"fc",x"09",x"0b",x"06",x"03",x"f7",x"fe",x"fb",x"f5",x"f6",x"fe",x"0b",x"02",x"f7",x"fc",x"f5",x"f8",x"02",x"06",x"0b",x"fa",x"fc",x"f8",x"0b",x"06",x"01",x"fd",x"01",x"fd",x"fe",x"f9",x"ff",x"fa",x"f8",x"f7",x"ff",x"ff",x"08",x"0b",x"00",x"f5",x"fc",x"f9",x"01",x"f6",x"f9",x"0b",x"01",x"07",x"fd",x"0a",x"fe",x"f8",x"08",x"f8",x"fb",x"fd",x"08",x"07",x"03",x"f7",x"0a",x"fa",x"01",x"05",x"01",x"f6",x"01",x"f8",x"fb",x"fb",x"01",x"fc",x"04",x"00",x"f7",x"09",x"07",x"05",x"05",x"07",x"04",x"07",x"fa",x"f9",x"0a",x"fa",x"00",x"00",x"fe",x"09",x"07",x"03",x"09",x"fe",x"03",x"f6",x"f8",x"07",x"f8",x"f6",x"f5",x"05",x"fd",x"05",x"08",x"f6",x"ff",x"0a",x"f6",x"03",x"04",x"f6",x"f7",x"f5",x"f6",x"fc",x"f7",x"0a",x"fb"),
    (x"04",x"0a",x"00",x"f5",x"f6",x"04",x"0a",x"04",x"f5",x"02",x"00",x"f9",x"09",x"0a",x"fb",x"07",x"fb",x"09",x"04",x"09",x"07",x"fe",x"05",x"f5",x"00",x"fe",x"0a",x"08",x"0b",x"f9",x"fd",x"f8",x"01",x"fe",x"f7",x"01",x"fb",x"07",x"08",x"fa",x"fa",x"0b",x"fa",x"06",x"08",x"f9",x"fb",x"f9",x"fe",x"ff",x"f7",x"01",x"0a",x"09",x"09",x"05",x"09",x"f5",x"fa",x"04",x"00",x"06",x"f7",x"ff",x"fb",x"08",x"03",x"0b",x"f7",x"07",x"02",x"fc",x"07",x"04",x"f7",x"fa",x"09",x"fa",x"fe",x"fd",x"06",x"fd",x"03",x"09",x"03",x"fb",x"f8",x"fb",x"f9",x"0a",x"0b",x"00",x"fe",x"ff",x"09",x"fc",x"f9",x"fc",x"0b",x"f5",x"fe",x"08",x"09",x"06",x"fe",x"09",x"f9",x"fc",x"04",x"03",x"ff",x"f9",x"fb",x"fc",x"05",x"05",x"ff",x"f8",x"fd",x"09",x"00",x"ff",x"fb",x"03",x"0b",x"f8",x"fd",x"05"),
    (x"fe",x"0b",x"0b",x"06",x"04",x"ff",x"00",x"07",x"ff",x"09",x"f5",x"ff",x"fa",x"fb",x"02",x"00",x"f7",x"fc",x"fb",x"f9",x"01",x"f5",x"01",x"f7",x"0b",x"fb",x"00",x"08",x"f8",x"06",x"01",x"09",x"f5",x"0b",x"fd",x"09",x"ff",x"00",x"00",x"09",x"06",x"f6",x"fe",x"02",x"fb",x"fa",x"f6",x"f9",x"0a",x"f9",x"f9",x"f7",x"0a",x"07",x"f6",x"f7",x"03",x"02",x"08",x"fe",x"04",x"fb",x"09",x"07",x"f7",x"fb",x"08",x"07",x"05",x"fc",x"fc",x"08",x"fb",x"f9",x"07",x"f9",x"f6",x"04",x"02",x"fc",x"06",x"01",x"fb",x"fc",x"00",x"02",x"fc",x"fd",x"02",x"f8",x"fa",x"06",x"08",x"fe",x"f6",x"fa",x"05",x"f5",x"09",x"00",x"fe",x"0b",x"f5",x"fb",x"08",x"06",x"f9",x"04",x"f7",x"f8",x"07",x"01",x"02",x"00",x"04",x"f8",x"02",x"05",x"02",x"f7",x"08",x"02",x"fd",x"03",x"0b",x"06",x"02",x"04"),
    (x"05",x"fb",x"ff",x"ff",x"03",x"02",x"08",x"04",x"06",x"f7",x"f7",x"f5",x"fd",x"fc",x"08",x"f5",x"00",x"fb",x"02",x"05",x"0a",x"02",x"fc",x"0a",x"07",x"fa",x"06",x"02",x"09",x"fa",x"01",x"f5",x"f8",x"f9",x"fc",x"01",x"fa",x"fc",x"fb",x"f8",x"07",x"fc",x"ff",x"f6",x"05",x"fb",x"f9",x"f7",x"f6",x"fc",x"fc",x"fa",x"fc",x"03",x"04",x"06",x"f6",x"0a",x"05",x"04",x"00",x"fe",x"07",x"fe",x"05",x"f5",x"f9",x"fb",x"03",x"f8",x"fc",x"05",x"09",x"f7",x"0a",x"fb",x"08",x"fc",x"f6",x"fe",x"f6",x"fd",x"f7",x"fc",x"f5",x"05",x"03",x"ff",x"fa",x"fd",x"02",x"08",x"06",x"f9",x"fa",x"fc",x"fd",x"00",x"0a",x"04",x"00",x"05",x"07",x"fb",x"05",x"05",x"04",x"07",x"09",x"fe",x"f5",x"0b",x"f6",x"f9",x"09",x"f8",x"06",x"fd",x"00",x"f8",x"03",x"fc",x"06",x"09",x"f6",x"fb",x"fd",x"fa"),
    (x"fc",x"f9",x"ff",x"0b",x"04",x"03",x"07",x"07",x"06",x"fc",x"09",x"01",x"fc",x"f5",x"fe",x"04",x"fd",x"f6",x"06",x"05",x"02",x"07",x"fa",x"f8",x"fc",x"fc",x"00",x"03",x"ff",x"f5",x"08",x"ff",x"01",x"03",x"02",x"0b",x"08",x"07",x"f9",x"08",x"05",x"f7",x"f7",x"fc",x"08",x"04",x"fc",x"06",x"03",x"04",x"01",x"f6",x"f9",x"f9",x"02",x"07",x"04",x"fa",x"0a",x"00",x"f8",x"05",x"fb",x"fd",x"fa",x"f9",x"f9",x"f5",x"07",x"08",x"ff",x"0a",x"f7",x"fb",x"f6",x"fd",x"00",x"03",x"03",x"02",x"fd",x"0a",x"05",x"07",x"f6",x"01",x"00",x"01",x"f7",x"fc",x"fc",x"05",x"fc",x"fa",x"00",x"fb",x"04",x"fa",x"f8",x"fb",x"05",x"04",x"fe",x"0a",x"fb",x"0a",x"09",x"f7",x"08",x"f7",x"00",x"03",x"fd",x"fd",x"f7",x"02",x"03",x"fc",x"09",x"05",x"06",x"04",x"f7",x"f8",x"fc",x"f7",x"08",x"0a"),
    (x"f7",x"f8",x"00",x"fd",x"05",x"fa",x"01",x"09",x"0a",x"08",x"01",x"08",x"fa",x"03",x"f6",x"05",x"f8",x"fb",x"03",x"fd",x"f9",x"0b",x"02",x"03",x"fc",x"0a",x"08",x"07",x"f8",x"08",x"fe",x"fc",x"fd",x"f7",x"fb",x"fa",x"00",x"f8",x"0a",x"03",x"09",x"06",x"02",x"fc",x"f6",x"fa",x"fd",x"ff",x"08",x"f6",x"00",x"08",x"03",x"04",x"09",x"fc",x"fa",x"01",x"03",x"fe",x"09",x"f8",x"fa",x"f7",x"f8",x"fb",x"01",x"fd",x"0a",x"00",x"06",x"fc",x"f8",x"f6",x"fc",x"03",x"05",x"00",x"fa",x"fb",x"01",x"07",x"03",x"ff",x"f9",x"09",x"03",x"02",x"f9",x"0a",x"00",x"06",x"ff",x"f8",x"f7",x"f8",x"fd",x"f6",x"07",x"f8",x"fd",x"04",x"0b",x"f7",x"0b",x"04",x"fa",x"f6",x"f8",x"ff",x"f6",x"04",x"f5",x"04",x"01",x"ff",x"f9",x"09",x"06",x"ff",x"09",x"f5",x"00",x"f8",x"fb",x"02",x"04",x"0b"),
    (x"06",x"04",x"fb",x"fc",x"07",x"09",x"fd",x"0a",x"f6",x"f6",x"0b",x"fc",x"fc",x"fd",x"04",x"09",x"07",x"03",x"0b",x"fc",x"fe",x"0a",x"00",x"f6",x"fa",x"f8",x"0a",x"f6",x"fb",x"01",x"05",x"f5",x"09",x"02",x"fa",x"0a",x"09",x"fb",x"01",x"ff",x"01",x"f8",x"05",x"05",x"08",x"f6",x"05",x"fa",x"09",x"00",x"08",x"f8",x"fb",x"fd",x"ff",x"f9",x"f7",x"00",x"03",x"02",x"fe",x"02",x"f5",x"0b",x"07",x"02",x"0b",x"07",x"05",x"f5",x"fe",x"f9",x"09",x"fc",x"0b",x"fa",x"fd",x"fe",x"f7",x"0a",x"f7",x"fe",x"0a",x"fd",x"0b",x"04",x"07",x"f6",x"f5",x"fa",x"fb",x"05",x"f9",x"08",x"08",x"00",x"fb",x"0a",x"04",x"fb",x"03",x"03",x"0a",x"f5",x"05",x"fe",x"fd",x"00",x"02",x"fa",x"f9",x"04",x"fa",x"f9",x"f8",x"fe",x"f9",x"03",x"04",x"f8",x"fd",x"0a",x"0a",x"01",x"05",x"04",x"09",x"fc"),
    (x"fc",x"fe",x"07",x"03",x"08",x"04",x"00",x"0a",x"fc",x"fc",x"07",x"fe",x"f6",x"08",x"00",x"f9",x"02",x"07",x"02",x"fc",x"00",x"fc",x"f5",x"f8",x"ff",x"08",x"fe",x"0a",x"05",x"03",x"fe",x"ff",x"08",x"03",x"f8",x"fe",x"fb",x"fc",x"02",x"f9",x"ff",x"f6",x"f5",x"0a",x"01",x"f6",x"05",x"00",x"fd",x"03",x"00",x"0b",x"05",x"00",x"fe",x"f7",x"fe",x"f6",x"07",x"04",x"08",x"fd",x"fa",x"01",x"f6",x"ff",x"05",x"fa",x"07",x"fb",x"ff",x"f7",x"09",x"05",x"fa",x"07",x"04",x"06",x"0a",x"04",x"07",x"01",x"06",x"f9",x"fe",x"02",x"f7",x"fe",x"fc",x"f8",x"ff",x"05",x"01",x"06",x"f6",x"ff",x"02",x"00",x"07",x"fe",x"07",x"fd",x"f6",x"f8",x"fe",x"0a",x"f7",x"f9",x"02",x"03",x"06",x"00",x"f6",x"09",x"fa",x"01",x"fb",x"08",x"f5",x"0b",x"ff",x"09",x"f8",x"f5",x"07",x"05",x"08",x"01"),
    (x"05",x"01",x"09",x"fb",x"fb",x"08",x"03",x"fc",x"fc",x"0b",x"09",x"ff",x"fc",x"fb",x"fc",x"04",x"f6",x"f7",x"fd",x"f9",x"03",x"04",x"f5",x"00",x"fc",x"0a",x"f9",x"fe",x"f8",x"04",x"0b",x"fa",x"fe",x"04",x"01",x"fd",x"fa",x"00",x"09",x"fc",x"fe",x"01",x"f9",x"f5",x"fa",x"fb",x"fc",x"00",x"fd",x"08",x"f6",x"fe",x"fe",x"f9",x"05",x"f8",x"f7",x"fd",x"05",x"02",x"f8",x"f9",x"f9",x"01",x"0a",x"08",x"f9",x"0a",x"f5",x"05",x"f5",x"08",x"fb",x"05",x"fb",x"01",x"ff",x"fc",x"f6",x"f8",x"05",x"02",x"08",x"fa",x"f8",x"06",x"ff",x"f5",x"03",x"0a",x"08",x"04",x"02",x"02",x"05",x"09",x"05",x"f7",x"09",x"08",x"01",x"f7",x"f9",x"fa",x"fe",x"f6",x"0a",x"08",x"02",x"06",x"03",x"f7",x"fa",x"09",x"fb",x"ff",x"f6",x"01",x"f9",x"fa",x"fd",x"ff",x"fd",x"00",x"02",x"fb",x"f9",x"fa"),
    (x"fc",x"05",x"03",x"05",x"07",x"01",x"04",x"ff",x"f8",x"fc",x"fb",x"07",x"f7",x"04",x"fd",x"06",x"01",x"06",x"07",x"fc",x"fc",x"04",x"fa",x"0a",x"f6",x"0b",x"0b",x"01",x"fa",x"07",x"fd",x"04",x"f8",x"fc",x"f9",x"fa",x"fc",x"06",x"fd",x"08",x"07",x"f7",x"05",x"f7",x"f7",x"f6",x"f8",x"07",x"f6",x"fd",x"fc",x"0b",x"03",x"07",x"fc",x"f5",x"04",x"fe",x"09",x"07",x"fc",x"f7",x"f7",x"0b",x"0b",x"f6",x"0a",x"f6",x"ff",x"04",x"f7",x"07",x"02",x"02",x"f6",x"f9",x"f5",x"ff",x"0b",x"05",x"0a",x"f5",x"f8",x"fb",x"ff",x"0a",x"0a",x"fa",x"ff",x"fa",x"02",x"ff",x"f7",x"fb",x"fa",x"fb",x"fd",x"f6",x"fd",x"0a",x"fc",x"0a",x"09",x"01",x"fd",x"fb",x"fc",x"08",x"06",x"07",x"0a",x"08",x"fe",x"08",x"fa",x"0a",x"f7",x"fd",x"f7",x"06",x"09",x"07",x"00",x"05",x"fa",x"02",x"01",x"02"),
    (x"04",x"f9",x"07",x"fa",x"fe",x"0a",x"00",x"05",x"08",x"fd",x"fc",x"fd",x"02",x"fe",x"02",x"01",x"0a",x"07",x"ff",x"02",x"f7",x"04",x"06",x"01",x"0b",x"07",x"03",x"f6",x"02",x"ff",x"fa",x"f7",x"f7",x"07",x"03",x"f8",x"08",x"fd",x"00",x"f6",x"06",x"fa",x"fb",x"04",x"ff",x"0b",x"0b",x"f8",x"f5",x"09",x"08",x"f6",x"f9",x"fd",x"04",x"f5",x"fd",x"ff",x"02",x"fc",x"0b",x"05",x"06",x"fb",x"f7",x"ff",x"fc",x"04",x"01",x"f7",x"04",x"02",x"f8",x"0a",x"ff",x"0a",x"fd",x"f8",x"fb",x"f9",x"04",x"05",x"01",x"f5",x"09",x"0b",x"ff",x"ff",x"04",x"01",x"01",x"00",x"f7",x"0a",x"0a",x"03",x"02",x"f6",x"07",x"fd",x"09",x"0b",x"03",x"02",x"08",x"06",x"08",x"08",x"04",x"04",x"fc",x"00",x"06",x"f9",x"fe",x"fb",x"00",x"fd",x"07",x"08",x"ff",x"04",x"08",x"fa",x"f9",x"04",x"01",x"09"),
    (x"04",x"02",x"09",x"05",x"02",x"fa",x"07",x"fa",x"f6",x"08",x"04",x"07",x"05",x"fd",x"f9",x"02",x"f9",x"0a",x"ff",x"fd",x"f7",x"0b",x"f9",x"03",x"04",x"08",x"fe",x"07",x"04",x"09",x"01",x"f8",x"fd",x"fc",x"ff",x"02",x"ff",x"fa",x"fc",x"fc",x"03",x"fa",x"05",x"0a",x"00",x"0a",x"f8",x"02",x"09",x"08",x"00",x"ff",x"0b",x"00",x"f9",x"f7",x"06",x"fe",x"03",x"00",x"fe",x"07",x"ff",x"02",x"f6",x"01",x"09",x"f8",x"f9",x"03",x"0a",x"f8",x"08",x"05",x"0b",x"fa",x"01",x"02",x"ff",x"0b",x"05",x"01",x"f9",x"05",x"f5",x"0b",x"06",x"01",x"f7",x"fd",x"00",x"00",x"08",x"00",x"f8",x"f9",x"06",x"fe",x"03",x"f5",x"f7",x"04",x"f7",x"05",x"ff",x"08",x"00",x"06",x"09",x"09",x"fd",x"fb",x"fc",x"02",x"02",x"0a",x"03",x"f6",x"fb",x"08",x"fa",x"07",x"03",x"04",x"03",x"f6",x"00",x"03"),
    (x"03",x"02",x"fd",x"f5",x"fe",x"00",x"01",x"fa",x"05",x"f9",x"fe",x"09",x"01",x"f7",x"01",x"08",x"05",x"f6",x"fe",x"07",x"07",x"f8",x"fd",x"08",x"ff",x"09",x"fc",x"00",x"fc",x"0b",x"04",x"fa",x"00",x"fc",x"04",x"05",x"01",x"07",x"f6",x"00",x"fb",x"fe",x"00",x"00",x"04",x"f6",x"fb",x"09",x"02",x"fa",x"0a",x"f8",x"08",x"f6",x"01",x"fe",x"fe",x"03",x"02",x"f6",x"f9",x"f6",x"fc",x"fc",x"06",x"0a",x"04",x"09",x"fc",x"08",x"02",x"f7",x"02",x"fd",x"f8",x"07",x"f5",x"f8",x"f5",x"06",x"f6",x"fa",x"f7",x"09",x"07",x"06",x"fb",x"fa",x"07",x"fb",x"fe",x"f7",x"03",x"05",x"f9",x"0a",x"fb",x"00",x"f9",x"f6",x"01",x"07",x"06",x"07",x"0b",x"f5",x"fa",x"00",x"0b",x"f7",x"05",x"00",x"08",x"fd",x"fb",x"06",x"f5",x"02",x"f6",x"07",x"f6",x"ff",x"fe",x"f7",x"03",x"04",x"f7",x"05")
);
--------------------------------------------------------
constant OUT_WIDTH     :  integer := 32;
end package;