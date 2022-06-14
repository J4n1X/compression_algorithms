from ast import Lambda
from dataclasses import dataclass
from pprint import pprint
from typing import Tuple
from bitarray import bitarray
import bitarray.util as bitutil
import sys
import argparse

from pkg_resources import require

def load_file(file_path: str) -> bytearray:
    with open(file_path, 'rb') as f:
        return bytearray(f.read())

def print_lzsymbol(symbol, buffer):
    if symbol.match:
        [offset, length] = symbol.reference
        ptr = buffer[-offset:][:length]
        # print types of arguments
        print(f'Reference: {symbol.reference} -> {ptr}')
    else:
        bytes = symbol.literal.to_bytes(1, 'big')
        print(f'Literal: {bytes}')

@dataclass(frozen=True)
class LZSymbol:
    match: bool
    _value: int | Tuple[int, int]

    @property
    def literal(self) -> int:
        if not self.match:
            return self._value
        else:
            raise ValueError('LZSymbol is not a literal')
    
    @property
    def reference(self) -> Tuple[int, int]:
        if self.match:
            return self._value
        else:
            raise ValueError('LZSymbol is not a reference')
    
    # also manipulates the bits
    def from_bits(bits: bitarray, lookahead_bits: int, search_bits: int, next_sym_bits: int) -> 'LZSymbol':
        match = bits[0]
        if match:
            offset = bitutil.ba2int(bits[1:lookahead_bits+1])
            length = bitutil.ba2int(bits[lookahead_bits+1:lookahead_bits+search_bits+1])
            del bits[:lookahead_bits+search_bits+1]
            return LZSymbol(match, (offset, length))
        else:
            symbol = bitutil.ba2int(bits[1:next_sym_bits+1])
            del bits[:next_sym_bits+1]
            return LZSymbol(match,symbol)
    
    def as_bits(self, lookahead_bits: int, search_bits: int, next_sym_bits: int) -> bitarray:
        ret_bits = bitarray()
        if self.match:
            ret_bits.append(self.match)
            ret_bits.extend(bitutil.int2ba(self.reference[0], lookahead_bits))
            ret_bits.extend(bitutil.int2ba(self.reference[1], search_bits))
        else:
            ret_bits.append(self.match)
            ret_bits.extend(bitutil.int2ba(self.literal, next_sym_bits))
        return ret_bits
    
class LZEncoder:
    def __init__(self, lookahead_size: int, search_size: int, next_sym_bits: int):
        self.lookahead_size = lookahead_size
        self.search_size = search_size
        self.next_sym_bits = next_sym_bits
        self.search_buffer = bytearray()
        self.lookahead_buffer = bytearray()
        self.reference_bits = self.lookahead_size.bit_length() + self.search_size.bit_length() + 1
        self.literal_bits = self.next_sym_bits + 1

    def _get_header_bytes(self) -> bytearray:
        return self.next_sym_bits.to_bytes(1, 'big') + self.lookahead_size.to_bytes(4, 'big') + self.search_size.to_bytes(4, 'big')

    def _push_search_char(self, byte: int) -> None:
        self.search_buffer.append(byte)
        if len(self.search_buffer) > self.search_size:
            del self.search_buffer[0]

    def _push_lookahead_char(self, byte: int) -> None:
        self.lookahead_buffer.append(byte)
        if len(self.lookahead_buffer) > self.lookahead_size:
            del self.lookahead_buffer[0]

    def _push_search_bytes(self, bytes: bytearray) -> None:
        for byte in bytes:
            self._push_search_char(byte)

    def _push_lookahead_bytes(self, bytes: bytearray) -> None:
        for byte in bytes:
            self._push_lookahead_char(byte)

    

    def _search_match(self, check_elements, elements):
        i = 0
        offset = 0
        for element in elements:
            if len(check_elements) <= offset:
                # All of the elements in check_elements are in elements
                return i - len(check_elements)

            if check_elements[offset] == element:
                offset += 1
            else:
                offset = 0

            i += 1
        return -1

    def encode(self, data: bytearray) -> bytearray:
        symbols: list[LZSymbol] = []
        i = 0
        for char in data:
            index = self._search_match(self.lookahead_buffer, self.search_buffer)

            # if no match or at end of file
            if self._search_match(self.lookahead_buffer + char.to_bytes(1, 'big'), self.search_buffer) == -1 or len(data) -1 == i:
                if len(data) - 1 == i:
                    self._push_lookahead_char(char)

                # if more than one character has no match
                if len(self.lookahead_buffer) > 1 and index != -1:
                    # calculate offset in search buffer
                    offset = len(self.search_buffer) - index
                    length = len(self.lookahead_buffer)
                    # if it would use less space to just store the literals
                    if self.reference_bits > length * self.literal_bits:
                        print(f'Using literal for {self.lookahead_buffer}')
                        # push the literals
                        for literal in self.lookahead_buffer:
                            new_sym = LZSymbol(False, literal)
                            #print_lzsymbol(new_sym, self.search_buffer)
                            symbols.append(new_sym)
                    # if it would use more space to just store the literals
                    else:
                        # print the reference first
                        # push the reference
                        new_sym = LZSymbol(True, (offset, length))
                        #print_lzsymbol(new_sym, self.search_buffer)
                        symbols.append(new_sym)

                    # transfer the lookahead buffer to the search buffer
                    self._push_search_bytes(self.lookahead_buffer)

                # if only one character has no match
                else: 
                    for literal in self.lookahead_buffer:
                        new_sym = LZSymbol(False, literal)
                        #print_lzsymbol(new_sym, self.search_buffer)
                        symbols.append(new_sym)
                        self._push_search_bytes(self.lookahead_buffer)

                self.lookahead_buffer = bytearray()

            self._push_lookahead_char(char)
            i+=1

        ret_bits = bitarray(endian='big')
        for i, symbol in enumerate(symbols):
            ret_bits.extend(symbol.as_bits(self.lookahead_size.bit_length(), self.search_size.bit_length(), self.next_sym_bits))
        return self._get_header_bytes() + bitutil.serialize(ret_bits)

class LZDecoder:
    def __init__(self, data: bytearray):
        self.next_sym_bits = data[0]
        self.lookahead_size = int.from_bytes(data[1:5], 'big')
        self.search_size = int.from_bytes(data[5:9], 'big')
        self._data = bitutil.deserialize(data[9:])
        self._search_buffer = bytearray()
    
    def _push_search_char(self, byte: int) -> None:
        self._search_buffer.append(byte)
        if len(self._search_buffer) > self.search_size:
            del self._search_buffer[0]

    def _push_search_bytes(self, bytes: bytearray) -> None:
        self._search_buffer.extend(bytes)
        if len(self._search_buffer) > self.search_size:
            # make sure we don't go over the search size, and remove the first elements if so
            self._search_buffer = self._search_buffer[len(self._search_buffer) - self.search_size:]


    def decode(self) -> bytearray:
        ret_data = bytearray()
        while len(self._data) > 0:
            sym: LZSymbol = LZSymbol.from_bits(self._data, self.lookahead_size.bit_length(), self.search_size.bit_length(), self.next_sym_bits) # also deletes the used bits
            #print_lzsymbol(sym, self._search_buffer)
            if sym.match:
                offset, length = sym.reference
                if offset > len(self._search_buffer):
                    raise ValueError(f'Offset {offset} is larger than the search buffer size {len(self._search_buffer)}. Search buffer contained {self._search_buffer}')
                if -offset+length > 0:
                    raise ValueError(f'Offset {offset} is larger than the search buffer size {len(self._search_buffer)}. Search buffer contained {self._search_buffer}')
                dec_bytes = self._search_buffer[-offset:][:length]
                self._push_search_bytes(dec_bytes)
                ret_data.extend(dec_bytes)
            else:
                ret_data.append(sym.literal)
                self._push_search_char(sym.literal)
        return ret_data


ARGS_DEFAULT_LOOKAHEAD_SIZE = 32
ARGS_DEFAULT_SEARCH_SIZE = 512

COMPRESS_ACTIVE = lambda: '-c' in sys.argv or '--compress' in sys.argv

# argparser
argparser = argparse.ArgumentParser(description='LZ Compression program. Created by J4n1X 2022', add_help=True)
mode_mutex = argparser.add_mutually_exclusive_group(required=True)
mode_mutex.add_argument('-c', '--compress', action='store_true', dest='compress_mode', help='Compress a file')
mode_mutex.add_argument('-d', '--decompress', action='store_true', dest='decompress_mode', help='Decompress a file')
argparser.add_argument('-s', '--search-size', type=int, default=ARGS_DEFAULT_SEARCH_SIZE, help='Size of the search buffer')
argparser.add_argument('-w', '--word_size', type=int, default=ARGS_DEFAULT_LOOKAHEAD_SIZE, help='Size of the word')
argparser.add_argument('-i', '--input', type=str, dest='input', required=True, help='Input file')
argparser.add_argument('-o', '--output', type=str, dest='output', required=True, help='Output file')

# argparser -> mode_mutex


def main():
    args = argparser.parse_args()
    if args.compress_mode:
        print(f'Compressing with window size of {args.search_size} and word size of {args.word_size}')
        args.search_size -= 1 
        args.word_size -= 1
        encoder = LZEncoder(args.search_size, args.word_size, 8)
        with open(args.output, 'wb') as f:
            f.write(encoder.encode(load_file(args.input)))
    elif args.decompress_mode:
        decoder = LZDecoder(load_file(args.input))
        with open(args.output, 'wb') as f:
            f.write(decoder.decode())
    else:
        raise ValueError('Invalid mode')
    

main()