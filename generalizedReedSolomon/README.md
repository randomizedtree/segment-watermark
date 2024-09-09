# Generalized Reed Solomon Code
This is an implementation of a special kind of Reed Solomon Code called Generalized Reed Solomon code according to the paper "A Class of Generalized RS-Codes with Faster
Encoding and Decoding Algorithms" by Amin Shokrollahi. A version of the paper can be found at https://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.306.4767&rep=rep1&type=pdf.

This special code can be sped up by a factor of p, since the code can be parallelized over p components. Additionally it allows to construct codes of arbitrary length as long as n%p == 0 is fulfilled and a primitive root of unity p can be found in the corresponding field over which the code is constructed. Whereas the standard Reed Solomon code imposes n= q^m -1. The code can be constructed over all Galois Fields.

## Documentation
n = k + r <br/>
k = message symbols sent <br/>
r = number of parity symbols <br/>
p = speed up factor <br/>

Currently the code can also handle codes where n %p != 0 however only if r is still divisble by p.
### Class Generalized_Reed_Solomon
```
class Generalized_Reed_Solomon(
field_size:int, #q characterisitc of the field
message_length:int, # n
payload_length:int, # k
symbol_size:int, #m exponent of field. Order of field is q^m
p_factor:int, #p speedup factor
irr_poly=None:array, # optional in case that the galois libary has not a precomputed irreducible polynominal
multi_processing = False, #optional when True certain parts of the code will be parallelized this is however currently only efficent when operating with long messages
debug=False # can be set to true in order to have the steps printed to the console
)
```
### Generalized_Reed_Solomon.encode()
```
Generalized_Reed_Solomon.encode(self,array)
# Takes as an argument a list and appends parity symbols to it
```
### Generalized_Reed_Solomon.decode()
```
Generalized_Reed_Solomon.decode(self, recieved_msg)
# Takes as an argument a list of a encoded message. Detects and corrects errors and then returns the k info symbols
```
### Generalized_Reed_Solomon.convert_to_symbol_array()
```
Generalized_Reed_Solomon.convert_to_symbol_array(self,array)
# Takes as an argument a list values and converts them to galois elements. If the degree of field > 1, the list will be´a list of list
# where each list represents the coefficents of a Galois element of the corresponding Field
```

### Generalized_Reed_Solomon.symbol_array_to_array()
```
Generalized_Reed_Solomon.symbol_array_to_array(self,array)
# Takes a list of ints as an argument. Performs the reverse of convert_to_symbol_array(). Transforms each Galois field element in the list
# into its coefficent representation.
```
## Usage
```
import generalizedReedSolomon
import numpy as np

org_info = [30, 20, 20, 9, 25, 1, 7, 2, 0, 18, 17, 22, 11,0,0]
org_compare = np.copy(org_info)
test_normal_rs = generalizedReedSolomon.Generalized_Reed_Solomon(field_size=31,
                                                                 message_length=27,
                                                                 payload_length=15,
                                                                 symbol_size=1,
                                                                 p_factor=3)
# encode message
normal_msg = test_normal_rs.encode(org_info)
#corrupt encode message
normal_msg[6] = 10
normal_msg[9] = 12
normal_msg[10] = 0
normal_msg[16] = 4
normal_msg[23] = 10

#fix encoded message
dec_test_normal = test_normal_rs.decode(normal_msg)
print("dec",dec_test_normal)
print("org",org_info)
print("equal",org_info == dec_test_normal)

#dec [30, 20, 20, 9, 25, 1, 7, 2, 0, 18, 17, 22, 11, 0, 0]
#org [30, 20, 20, 9, 25, 1, 7, 2, 0, 18, 17, 22, 11, 0, 0]
#equal True
```
```
 info_symbols = [
        [20,5],
        [35,10],
        [5,10],
        [35,20],
        [35,10],
        [3,10],
        [35,10],
        [35,2],
        [5,10],
        [35,10]
    ]
test_normal_rs =reedSolomon.Generalized_Reed_Solomon(field_size=47,
                                                                 message_length=14,
                                                                 payload_length=10,
                                                                 symbol_size=2,
                                                                 p_factor=2,
                                                                debug=True)
#convert to galois elements
symbols = test_normal_rs.convert_to_symbol_array(info_symbols)
print("symbols",symbols)
symbols_encoded = test_normal_rs.encode(symbols)
symbols_encoded[0] =1
#convert Galois Elements back to coefficent representation
symbols_decoded = test_normal_rs.decode(symbols_encoded)
conv_back = test_normal_rs.symbol_array_to_array(symbols_decoded)
    
print(conv_back)
print(info_symbols == conv_back)
#symbols [945, 1655, 245, 1665, 1655, 151, 1655, 1647, 245, 1655]
#[[20, 5], [35, 10], [5, 10], [35, 20], [35, 10], [3, 10], [35, 10], [35, 2], [5, 10], [35, 10]]
#True
```
## Dependencies
galois == 0.0.28 <br/>
numpy == 1.19.5

## TODO
1. Add case for r % p != 0 <br/>
2. Add parallelization(done for fft)
