--------------------------------------------------------
-- This file is generated by Delirium - Sma-RTY SAS 
-- Mon Feb 10 14:45:23 2020
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
constant CONV1_OUT_SIZE     :  integer := 512;
constant CONV1_MULT_STYLE   :  string  := "dsp";
constant CONV1_PIPELINE     :  boolean := True;
constant CONV1_KERNEL_SIZE  :  integer := 1;
constant CONV1_PADDING      :  boolean := FALSE;
constant CONV1_STRIDE       :  positive:= 1;
constant CONV1_BIAS_VALUE   :  data_array  (0 to CONV1_OUT_SIZE - 1) := 
    (x"cd",x"4b",x"7f",x"62",x"fe",x"e3",x"e9",x"71",x"da",x"03",x"f3",x"f8",x"ea",x"8d",x"27",x"c2",x"70",x"92",x"92",x"34",x"d9",x"d0",x"4b",x"29",x"33",x"41",x"67",x"67",x"a0",x"41",x"50",x"b0",x"6f",x"52",x"c0",x"ec",x"59",x"74",x"25",x"7b",x"d8",x"e9",x"ee",x"9b",x"b7",x"04",x"c6",x"47",x"7b",x"c6",x"89",x"a9",x"ef",x"4a",x"a9",x"70",x"b3",x"0e",x"6e",x"e6",x"91",x"a8",x"64",x"1e",x"c6",x"33",x"f7",x"ba",x"3d",x"b7",x"11",x"4c",x"a0",x"ff",x"28",x"0c",x"37",x"a2",x"76",x"22",x"8a",x"2a",x"3f",x"43",x"6b",x"0e",x"71",x"93",x"02",x"d6",x"a1",x"7a",x"e5",x"8c",x"53",x"d9",x"0e",x"97",x"3a",x"9f",x"c5",x"38",x"c7",x"a6",x"b3",x"55",x"40",x"87",x"39",x"24",x"4d",x"eb",x"fa",x"a0",x"eb",x"53",x"0d",x"9f",x"af",x"0c",x"b9",x"6a",x"c9",x"6b",x"02",x"b6",x"54",x"c2",x"4c",x"af",x"1f",x"18",x"82",x"9c",x"12",x"2f",x"05",x"77",x"b3",x"c5",x"43",x"f6",x"43",x"a4",x"ec",x"6e",x"2c",x"ef",x"8d",x"f2",x"74",x"9b",x"5e",x"42",x"df",x"0e",x"09",x"0d",x"8e",x"00",x"de",x"0d",x"11",x"fd",x"fc",x"ab",x"6d",x"92",x"c5",x"ee",x"63",x"a5",x"ad",x"de",x"1b",x"be",x"be",x"46",x"c1",x"32",x"8a",x"9a",x"ce",x"6f",x"46",x"0c",x"fc",x"a3",x"3d",x"42",x"de",x"23",x"24",x"fa",x"1d",x"f1",x"ea",x"cc",x"c6",x"44",x"59",x"84",x"22",x"b6",x"2d",x"76",x"bd",x"9f",x"16",x"c2",x"d0",x"00",x"9f",x"5f",x"1f",x"b6",x"39",x"dc",x"d2",x"e8",x"8c",x"f0",x"2f",x"82",x"7c",x"07",x"de",x"11",x"f5",x"5e",x"f4",x"04",x"17",x"3b",x"9d",x"23",x"55",x"db",x"a0",x"d0",x"a0",x"c6",x"17",x"9b",x"6d",x"20",x"20",x"88",x"f5",x"fa",x"c6",x"3b",x"e0",x"28",x"85",x"1d",x"83",x"64",x"26",x"57",x"1d",x"7c",x"7e",x"0e",x"22",x"be",x"ad",x"b0",x"c6",x"ec",x"09",x"23",x"2e",x"eb",x"79",x"38",x"70",x"79",x"b5",x"3e",x"8d",x"7a",x"0d",x"43",x"6d",x"8b",x"dd",x"11",x"ad",x"0e",x"ab",x"0a",x"9f",x"2e",x"62",x"47",x"6b",x"ef",x"e6",x"50",x"a6",x"5f",x"bb",x"15",x"51",x"d4",x"b1",x"44",x"ce",x"7f",x"c5",x"47",x"f1",x"24",x"45",x"c1",x"f6",x"df",x"38",x"ef",x"68",x"44",x"b2",x"39",x"27",x"6e",x"21",x"39",x"1c",x"09",x"e6",x"ae",x"97",x"da",x"c3",x"10",x"c6",x"bc",x"4f",x"2c",x"32",x"b5",x"00",x"0c",x"25",x"44",x"75",x"6a",x"9e",x"e3",x"32",x"d1",x"fd",x"18",x"23",x"8f",x"69",x"45",x"34",x"d5",x"9f",x"ed",x"f1",x"9b",x"fe",x"a6",x"8f",x"77",x"ef",x"4c",x"4a",x"7b",x"30",x"8e",x"ed",x"f8",x"9a",x"c0",x"7a",x"3f",x"a3",x"60",x"ef",x"05",x"69",x"a2",x"fe",x"5e",x"83",x"d9",x"20",x"07",x"40",x"2d",x"c5",x"db",x"54",x"0b",x"b7",x"52",x"bf",x"1c",x"c7",x"f5",x"e5",x"cc",x"a9",x"35",x"c7",x"07",x"ff",x"dc",x"32",x"3e",x"ef",x"a5",x"6e",x"6d",x"8e",x"95",x"72",x"68",x"d5",x"d9",x"c3",x"f2",x"56",x"d3",x"22",x"e3",x"76",x"d6",x"62",x"56",x"4b",x"e7",x"1e",x"54",x"a5",x"b4",x"5f",x"79",x"a1",x"40",x"b6",x"81",x"06",x"47",x"9b",x"07",x"64",x"43",x"95",x"c2",x"fe",x"70",x"7b",x"61",x"d6",x"fa",x"8f",x"f6",x"f9",x"14",x"f2",x"29",x"04",x"2a",x"36",x"57",x"fc",x"96",x"ca",x"84",x"9d",x"88",x"43",x"15",x"b7",x"6b",x"ff",x"f1",x"c5",x"35",x"65",x"dd",x"e5",x"2d",x"49",x"72",x"58",x"8b",x"ca",x"73",x"97",x"f6",x"d0",x"26",x"44",x"66",x"26",x"7b",x"65",x"ed",x"93",x"d8",x"46",x"7e",x"1f",x"5a",x"a7",x"40");
constant CONV1_KERNEL_VALUE :  data_matrix (0 to CONV1_OUT_SIZE - 1 ,  0 to CONV1_IN_SIZE * CONV1_KERNEL_SIZE * CONV1_KERNEL_SIZE - 1) := (
    (x"f7"),
    (x"35"),
    (x"4b"),
    (x"98"),
    (x"67"),
    (x"79"),
    (x"05"),
    (x"bf"),
    (x"55"),
    (x"b7"),
    (x"00"),
    (x"ad"),
    (x"2c"),
    (x"38"),
    (x"d7"),
    (x"67"),
    (x"3f"),
    (x"01"),
    (x"3f"),
    (x"f4"),
    (x"da"),
    (x"60"),
    (x"e6"),
    (x"f3"),
    (x"1b"),
    (x"0e"),
    (x"ef"),
    (x"68"),
    (x"ea"),
    (x"fb"),
    (x"7c"),
    (x"4d"),
    (x"70"),
    (x"a6"),
    (x"4e"),
    (x"71"),
    (x"2a"),
    (x"d7"),
    (x"34"),
    (x"92"),
    (x"eb"),
    (x"fa"),
    (x"ef"),
    (x"56"),
    (x"28"),
    (x"5c"),
    (x"df"),
    (x"cc"),
    (x"84"),
    (x"6a"),
    (x"2f"),
    (x"8e"),
    (x"a2"),
    (x"71"),
    (x"95"),
    (x"82"),
    (x"fc"),
    (x"c2"),
    (x"c8"),
    (x"a3"),
    (x"c2"),
    (x"40"),
    (x"fe"),
    (x"9d"),
    (x"d3"),
    (x"49"),
    (x"2e"),
    (x"05"),
    (x"1c"),
    (x"97"),
    (x"dc"),
    (x"88"),
    (x"00"),
    (x"ca"),
    (x"5d"),
    (x"0b"),
    (x"df"),
    (x"a8"),
    (x"f5"),
    (x"74"),
    (x"d4"),
    (x"60"),
    (x"1e"),
    (x"37"),
    (x"8b"),
    (x"7f"),
    (x"77"),
    (x"ac"),
    (x"8b"),
    (x"1d"),
    (x"35"),
    (x"cd"),
    (x"42"),
    (x"df"),
    (x"29"),
    (x"cb"),
    (x"7c"),
    (x"68"),
    (x"3c"),
    (x"15"),
    (x"d6"),
    (x"83"),
    (x"26"),
    (x"4e"),
    (x"ae"),
    (x"1d"),
    (x"55"),
    (x"ba"),
    (x"93"),
    (x"f2"),
    (x"28"),
    (x"7e"),
    (x"9d"),
    (x"a1"),
    (x"d9"),
    (x"bd"),
    (x"b0"),
    (x"0c"),
    (x"a8"),
    (x"a8"),
    (x"ff"),
    (x"a1"),
    (x"b2"),
    (x"34"),
    (x"19"),
    (x"9f"),
    (x"ff"),
    (x"a0"),
    (x"a8"),
    (x"77"),
    (x"75"),
    (x"e0"),
    (x"52"),
    (x"9b"),
    (x"60"),
    (x"59"),
    (x"41"),
    (x"1c"),
    (x"e3"),
    (x"8d"),
    (x"a7"),
    (x"96"),
    (x"5e"),
    (x"ee"),
    (x"f8"),
    (x"8e"),
    (x"fc"),
    (x"ab"),
    (x"d9"),
    (x"c6"),
    (x"1a"),
    (x"3c"),
    (x"ea"),
    (x"a3"),
    (x"d8"),
    (x"2a"),
    (x"40"),
    (x"bc"),
    (x"0f"),
    (x"1e"),
    (x"d1"),
    (x"b9"),
    (x"68"),
    (x"a0"),
    (x"8c"),
    (x"18"),
    (x"10"),
    (x"9b"),
    (x"b2"),
    (x"e3"),
    (x"bc"),
    (x"2d"),
    (x"c6"),
    (x"5a"),
    (x"38"),
    (x"f5"),
    (x"0b"),
    (x"9e"),
    (x"29"),
    (x"a5"),
    (x"d6"),
    (x"ee"),
    (x"c6"),
    (x"81"),
    (x"a7"),
    (x"0f"),
    (x"9f"),
    (x"d1"),
    (x"ed"),
    (x"3f"),
    (x"df"),
    (x"c4"),
    (x"a0"),
    (x"86"),
    (x"3d"),
    (x"bf"),
    (x"52"),
    (x"e0"),
    (x"a7"),
    (x"50"),
    (x"29"),
    (x"b5"),
    (x"dd"),
    (x"38"),
    (x"b8"),
    (x"63"),
    (x"56"),
    (x"65"),
    (x"96"),
    (x"b5"),
    (x"9a"),
    (x"82"),
    (x"ab"),
    (x"fa"),
    (x"43"),
    (x"c4"),
    (x"4a"),
    (x"3a"),
    (x"c9"),
    (x"48"),
    (x"77"),
    (x"97"),
    (x"e7"),
    (x"b0"),
    (x"72"),
    (x"30"),
    (x"e7"),
    (x"a1"),
    (x"1e"),
    (x"ee"),
    (x"19"),
    (x"9e"),
    (x"d2"),
    (x"1c"),
    (x"76"),
    (x"02"),
    (x"c7"),
    (x"d3"),
    (x"ed"),
    (x"a7"),
    (x"ea"),
    (x"ee"),
    (x"53"),
    (x"3a"),
    (x"c2"),
    (x"c4"),
    (x"72"),
    (x"e0"),
    (x"6e"),
    (x"44"),
    (x"fc"),
    (x"b6"),
    (x"65"),
    (x"bc"),
    (x"10"),
    (x"fd"),
    (x"f4"),
    (x"ea"),
    (x"9e"),
    (x"62"),
    (x"4f"),
    (x"76"),
    (x"c9"),
    (x"c5"),
    (x"1e"),
    (x"a3"),
    (x"8c"),
    (x"a8"),
    (x"81"),
    (x"3b"),
    (x"78"),
    (x"ee"),
    (x"4a"),
    (x"0b"),
    (x"6c"),
    (x"97"),
    (x"39"),
    (x"0d"),
    (x"b1"),
    (x"35"),
    (x"6b"),
    (x"e4"),
    (x"5c"),
    (x"25"),
    (x"5b"),
    (x"86"),
    (x"12"),
    (x"27"),
    (x"45"),
    (x"bc"),
    (x"16"),
    (x"65"),
    (x"10"),
    (x"96"),
    (x"31"),
    (x"bb"),
    (x"b9"),
    (x"20"),
    (x"d6"),
    (x"be"),
    (x"55"),
    (x"37"),
    (x"cf"),
    (x"a7"),
    (x"74"),
    (x"c0"),
    (x"9c"),
    (x"35"),
    (x"85"),
    (x"95"),
    (x"fa"),
    (x"21"),
    (x"ad"),
    (x"53"),
    (x"d4"),
    (x"99"),
    (x"d5"),
    (x"a1"),
    (x"f3"),
    (x"ec"),
    (x"63"),
    (x"b3"),
    (x"00"),
    (x"18"),
    (x"63"),
    (x"d9"),
    (x"de"),
    (x"3c"),
    (x"f4"),
    (x"2b"),
    (x"28"),
    (x"ac"),
    (x"5f"),
    (x"fa"),
    (x"2c"),
    (x"0e"),
    (x"55"),
    (x"fa"),
    (x"fc"),
    (x"c7"),
    (x"70"),
    (x"90"),
    (x"2e"),
    (x"c4"),
    (x"95"),
    (x"37"),
    (x"53"),
    (x"42"),
    (x"12"),
    (x"9a"),
    (x"6c"),
    (x"d1"),
    (x"a0"),
    (x"0e"),
    (x"fd"),
    (x"11"),
    (x"22"),
    (x"fd"),
    (x"a9"),
    (x"e6"),
    (x"ed"),
    (x"56"),
    (x"c7"),
    (x"0b"),
    (x"bc"),
    (x"d6"),
    (x"61"),
    (x"b2"),
    (x"9c"),
    (x"87"),
    (x"9e"),
    (x"f1"),
    (x"fa"),
    (x"83"),
    (x"f0"),
    (x"e9"),
    (x"9f"),
    (x"f1"),
    (x"d9"),
    (x"74"),
    (x"55"),
    (x"28"),
    (x"01"),
    (x"d7"),
    (x"95"),
    (x"35"),
    (x"d9"),
    (x"d7"),
    (x"53"),
    (x"da"),
    (x"c4"),
    (x"4e"),
    (x"9a"),
    (x"09"),
    (x"8a"),
    (x"e0"),
    (x"11"),
    (x"ab"),
    (x"31"),
    (x"4c"),
    (x"e1"),
    (x"0b"),
    (x"82"),
    (x"85"),
    (x"28"),
    (x"f3"),
    (x"69"),
    (x"ca"),
    (x"bf"),
    (x"69"),
    (x"11"),
    (x"cc"),
    (x"05"),
    (x"08"),
    (x"c1"),
    (x"cc"),
    (x"59"),
    (x"51"),
    (x"bb"),
    (x"64"),
    (x"33"),
    (x"22"),
    (x"23"),
    (x"77"),
    (x"e7"),
    (x"78"),
    (x"0c"),
    (x"71"),
    (x"90"),
    (x"69"),
    (x"8f"),
    (x"ff"),
    (x"c5"),
    (x"f7"),
    (x"3b"),
    (x"78"),
    (x"b7"),
    (x"ef"),
    (x"06"),
    (x"5e"),
    (x"b9"),
    (x"64"),
    (x"31"),
    (x"26"),
    (x"34"),
    (x"fc"),
    (x"2f"),
    (x"d2"),
    (x"1c"),
    (x"c8"),
    (x"b4"),
    (x"4a"),
    (x"ea"),
    (x"bf"),
    (x"d4"),
    (x"c1"),
    (x"db"),
    (x"3b"),
    (x"18"),
    (x"52"),
    (x"5c"),
    (x"32"),
    (x"07"),
    (x"7e"),
    (x"74"),
    (x"6b"),
    (x"7f"),
    (x"b0"),
    (x"63"),
    (x"32"),
    (x"c3"),
    (x"a3"),
    (x"a4"),
    (x"df"),
    (x"f9"),
    (x"fa"),
    (x"19"),
    (x"89"),
    (x"a8"),
    (x"27"),
    (x"06"),
    (x"f5"),
    (x"d3"),
    (x"81"),
    (x"16"),
    (x"a3"),
    (x"57"),
    (x"53"),
    (x"b0"),
    (x"ce"),
    (x"90"),
    (x"c3"),
    (x"54"),
    (x"84"),
    (x"74"),
    (x"71"),
    (x"61"),
    (x"e6"),
    (x"b8"),
    (x"c2"),
    (x"99"),
    (x"73"),
    (x"4b"),
    (x"bf"),
    (x"a5"),
    (x"36"),
    (x"89"),
    (x"44"),
    (x"98"),
    (x"6a"),
    (x"f3"),
    (x"8c")
);
--------------------------------------------------------
constant OUT_WIDTH     :  integer := 512;
end package;