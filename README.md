This is a Hand-Made project

This program searches for Bitcoin Puzzles (or any other BTC address) on any range

if you want to search with a bloomfile - take this one https://github.com/neoman513-cloud/Bloom-Searcher

THIS IS PURE TRUE UNBIASED RANDOM, with real speed, not fake

it uses a Cuda specialized random distribution that is Curand + CPU crypto secure random combined and millions of threads, no overlaps

it has a very good random distribution that is carefully tested

tough the range is huge, but with a good GPU hardware this program can be very useful

Run (dont include puzzle number in the command, its just for info):

`main.exe 100000 1fffff 29a78213caa9eea824acf08022ab9dfc83414f56` - puzzle 21\
`main.exe 1000000 1ffffff 2f396b29b27324300d0c59b17c3abc1835bd3dbb` - puzzle 25\
`main.exe 10000000 1fffffff 5a416cc9148f4a377b672c8ae5d3287adaafadec` - puzzle 29\
`main.exe 20000000 3fffffff d39c4704664e1deb76c9331e637564c257d68a08` - puzzle 30\
`main.exe 100000000 1ffffffff 4e15e5189752d1eaf444dfd6bff399feb0443977` - puzzle 33\
`main.exe 200000000 3ffffffff f6d67d7983bf70450f295c9cb828daab265f1bfa` - puzzle 34\
`main.exe 1000000000000000000 1ffffffffffffffffff 105b7f253f0ebd7843adaebbd805c944bfb863e4` - puzzle 73

you can convert any bitcoin address to hash160 using this site: https://cointools.org/address-to-hash

just in case if you will edit this code further, you can compile it with the command:

`nvcc -o main main.cu`

you can set the DEVICE ID as the last parameter as 0, 1, 2, 3 etc...when you run the program it will display available gpus and their IDS

also in secp256k1.cuh there are few parameters at top of the file that you can change for your preference:

`#define BIGINT_WORDS 8`\
`#define WINDOW_SIZE 16`\
`#define NUM_BASE_POINTS 16`\
`#define BATCH_SIZE 128`\
`#define MOD_EXP 4`

BTC for donations:

`bc1q8n38pk3urztlt4vceq0h089l9jdcw58l2c0e80`
