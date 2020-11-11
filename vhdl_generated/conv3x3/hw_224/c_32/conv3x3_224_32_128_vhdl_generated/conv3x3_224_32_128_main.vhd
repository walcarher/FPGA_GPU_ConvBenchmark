--------------------------------------------------------
-- This file is generated by Delirium - Sma-RTY SAS 
-- Wed Nov 11 05:58:31 2020
--------------------------------------------------------

library ieee;
  use ieee.std_logic_1164.all;
  use ieee.numeric_std.all;
library work;
  use work.bitwidths.all;
  use work.Delirium.all;
  use work.params.all;
entity conv3x3_224_32_128_main is
generic(
  BITWIDTH  : integer := DATA_BITWIDTH;
  IMAGE_WIDTH : integer := INPUT_IMAGE_WIDTH
);
port(
  clk      : in std_logic;
  reset_n  : in std_logic;
  enable   : in std_logic;
  select_i : in std_logic_vector(31 downto 0);
  in_data  : in std_logic_vector(PIXEL_BITWIDTH -1 downto 0);
  in_dv    : in std_logic;
  in_fv    : in std_logic;
  out_data : out std_logic_vector(BITWIDTH-1 downto 0);
  out_dv   : out std_logic;
  out_fv   : out std_logic
  );
end entity;

architecture STRUCTURAL of conv3x3_224_32_128_main is
 -- Signals
signal input_data: data_array  (0 to INPUT_CHANNELS - 1);
signal input_dv	: std_logic;
signal input_fv	: std_logic;
signal output_data: accu_array (0 to Conv_0_OUT_SIZE - 1);
signal output_dv	: std_logic;
signal output_fv	: std_logic;

 -- Instances
begin
InputLayer_i : InputLayer
generic map (
  BITWIDTH        => BITWIDTH,
  PIXEL_BITWIDTH  => PIXEL_BITWIDTH,
  NB_OUT_FLOWS    => INPUT_CHANNELS
)
port map (
  clk             => clk,
  reset_n         => reset_n,
  enable          => enable,
  in_data         => in_data,
  in_dv           => in_dv,
  in_fv           => in_fv,
  out_data        => input_data,
  out_dv          => input_dv,
  out_fv          => input_fv
  );

Conv_0 : ConvLayer
generic map (
  BITWIDTH     => BITWIDTH,
  ACCU_BITWIDTH=> ACCU_BITWIDTH,
  MULT_STYLE   => Conv_0_MULT_STYLE,
  PIPELINE     => Conv_0_PIPELINE,
  IMAGE_WIDTH  => Conv_0_IMAGE_WIDTH,
  NB_OUT_FLOWS => Conv_0_OUT_SIZE,
  NB_IN_FLOWS  => Conv_0_IN_SIZE,
  KERNEL_SIZE  => Conv_0_KERNEL_SIZE,
  PADDING      => Conv_0_PADDING,
  STRIDE       => Conv_0_STRIDE,
  KERNEL_VALUE => Conv_0_KERNEL_VALUE,
  BIAS_VALUE   => Conv_0_BIAS_VALUE
)
port map (
  clk      => clk,
  reset_n  => reset_n,
  enable   => enable,
  in_data  => input_data,
  in_dv    => input_dv,
  in_fv    => input_fv,
  out_data => output_data,
  out_dv   => output_dv,
  out_fv   => output_fv
);

DisplayLayer_i: DisplayLayer
  generic map(
  BITWIDTH    => BITWIDTH,
  NB_IN_FLOWS => Conv_0_OUT_SIZE
  )
  port map(
  in_data     => output_data,
  in_dv       => output_dv,
  in_fv       => output_fv,
  sel         => select_i,
  out_data    => out_data,
  out_dv      => out_dv,
  out_fv      => out_fv
);
end architecture;
