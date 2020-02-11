--------------------------------------------------------
-- This file is generated by Delirium - Sma-RTY SAS 
-- Mon Feb 10 17:17:52 2020
--------------------------------------------------------

library ieee;
    use ieee.std_logic_1164.all;
library work;
    use work.Delirium.all;

package params is
--------------------------------------------------------
-- INPUT
constant INPUT_CHANNELS    : integer := 1;
constant INPUT_IMAGE_WIDTH : integer := 224;
--------------------------------------------------------
--CONV1
constant CONV1_IMAGE_WIDTH  :  integer := 224;
constant CONV1_IN_SIZE      :  integer := 1;
constant CONV1_OUT_SIZE     :  integer := 8;
constant CONV1_MULT_STYLE   :  string  := "dsp";
constant CONV1_PIPELINE     :  boolean := True;
constant CONV1_KERNEL_SIZE  :  integer := 1;
constant CONV1_PADDING      :  boolean := FALSE;
constant CONV1_STRIDE       :  positive:= 1;
constant CONV1_BIAS_VALUE   :  data_array  (0 to CONV1_OUT_SIZE - 1) := 
    (x"14",x"8e",x"e8",x"7d",x"eb",x"30",x"fe",x"44");
constant CONV1_KERNEL_VALUE :  data_matrix (0 to CONV1_OUT_SIZE - 1 ,  0 to CONV1_IN_SIZE * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE - 1) := (
    (x"a5"),
    (x"b8"),
    (x"77"),
    (x"db"),
    (x"c2"),
    (x"0c"),
    (x"b1"),
    (x"7f")
);
--------------------------------------------------------
constant OUT_WIDTH     :  integer := 8;
end package;