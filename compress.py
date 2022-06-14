from pprint import pprint
from dataclasses import dataclass;
import argparse
import sys

import cProfile

import bitarray.util as bitutil
from bitarray import bitarray, bits2bytes;

MAX_CODE_COUNT = 65534
MAX_CODE_VAL = 65535
TREE_DICT_TERM = bitarray('11111111111111111', endian='little')

def load_file(file_path: str) -> bytearray:
    """
    Loads a file into a bytearray.
    """
    # check if exists
#    if not os.path.exists(file_path):
#        raise FileNotFoundError("File not found: " + file_path)
    with open(file_path, 'rb') as f:
        return bytearray(f.read())

def get_blocks(data: bytearray, block_size: int) -> list[bytearray]:
    """
    Returns a list of blocks of size block_size.
    CONSUMES data.
    """
    block_data = bytearray(data)
    blocks: list[bytearray] = []
    while len(block_data) > block_size:
        blocks.append(bytes(block_data[:block_size]))
        del block_data[:block_size]
    # add remaining data
    if len(data) > 0:
        blocks.append(bytes(block_data) + bytes(b'\x00' * (block_size - len(block_data))))
        del block_data[:]    

    print(len(blocks), "blocks created")
    return blocks

@dataclass(frozen=True)
class HuffmanSymbol:
    block: bytes
    freq: int
    def as_bits(self) -> bitarray:
        ret = bitarray(endian='little')
        ret.frombytes(self.block)
        return ret
    def __add__(self, other: 'HuffmanSymbol') -> 'HuffmanSymbol':
        return HuffmanSymbol(self.block + other.block, self.freq + other.freq)

# A Huffman Tree Node
@dataclass(init=True, repr=True, eq=True, order=False)
class HuffmanNode:
    symbol: HuffmanSymbol
    left: 'HuffmanNode' = None
    right: 'HuffmanNode' = None
    code: bitarray = bitarray(endian='little')

@dataclass(repr=True)
class HuffmanCode:
    value: int
    depth: int
    def __eq__(self, other):
        if isinstance(other, HuffmanCode):
            return self.value == other.value and self.depth == other.depth
        else:
            raise TypeError("Cannot compare for equality on a HuffmanCode to a non-HuffmanCode")
    def __lt__(self, other):
        if isinstance(other, HuffmanCode):
            return self.value < other.value
        return self.value < other
    def __gt__(self, other):
        if isinstance(other, HuffmanCode):
            return self.value > other.value
        return self.value > other
    def __le__(self, other):
        if isinstance(other, HuffmanCode):
            return self.value <= other.value
        return self.value <= other
    def __ge__(self, other):
        if isinstance(other, HuffmanCode):
            return self.value >= other.value
        return self.value >= other
    def __ne__(self, other):
        if isinstance(other, HuffmanCode):
            return not (self == other)
        else:
            raise TypeError("Cannot compare for inequality on a HuffmanCode to a non-HuffmanCode")

@dataclass(init=False)
class HuffmanTree:
    root: HuffmanNode
    codes: dict[bytes, HuffmanCode]
    block_size: int
    def __init__(self, root: HuffmanNode, codes: dict[bytes, HuffmanCode], block_size: int) -> None:
        self.root = root
        self.codes = codes
        self.block_size = block_size

# returns a sorted list of HuffmanSymbols
def huffman_create_freq_list(data: bytearray, symbol_len: int) -> list[HuffmanSymbol]:
    # get unique symbols
    freqs: dict[bytes, int] = {}
    for block in get_blocks(data, symbol_len):
        if block not in freqs.keys():
            freqs[block] = 1
        else:
            freqs[block] += 1
    print(f"Found a total of {len(freqs)} unique symbols")
    return [HuffmanSymbol(block, freq) for block, freq in freqs.items()]

def huffman_generate_tree(symbols: list[HuffmanSymbol], paren_node: HuffmanNode = None) -> HuffmanNode:
    """
    Generates a Huffman tree from a bytearray.
    """
    if len(symbols) == 0:
        raise Exception("No symbols to generate tree from")
    if len(symbols) == 1:
        return HuffmanNode(symbols[0])

    nodes: list[HuffmanNode] = []
    for symbol in symbols:
        if symbol.freq == 0:
            raise Exception("Symbol frequency is 0")
        nodes.append(HuffmanNode(symbol))

    while len(nodes) > 1:
        # sort by frequency
        nodes = sorted(nodes, key=lambda x: x.symbol.freq)
        right = nodes[0]
        left = nodes[1] if len(nodes) > 1 else None

        left.code = bitarray('0', endian='little')
        right.code = bitarray('1', endian='little')
        parent = HuffmanNode(HuffmanSymbol(bytearray(), left.symbol.freq + right.symbol.freq), left, right)
        nodes.remove(left)
        nodes.remove(right)
        nodes.append(parent)
    return nodes[0]

def huffman_get_tree_node_count(tree: HuffmanTree) -> int:
    """
    Returns the number of nodes in a Huffman tree.
    """
    count = 0
    def count_nodes(node: HuffmanNode):
        nonlocal count
        count += 1
        if node.left is not None:
            count_nodes(node.left)
        if node.right is not None:
            count_nodes(node.right)
    count_nodes(tree)
    return count

# todo: Make more efficient by replacing bytes with int
def huffman_get_symbol_codes(node: HuffmanNode, value: bitarray = bitarray()) -> dict[bytes, bitarray]:
    """
    Returns a dictionary of symbol codes.
    """
    new_value = bitarray()
    new_value.extend(value)
    new_value.extend(node.code)
    codes: dict[bytes, bitarray] = {}
    if node.left == None and node.right == None:
        #print("Leaf node", node.symbol.block, node.code)
        codes[node.symbol.block] = new_value
    else:
        #print("Branch node", node.symbol.block, node.code)
        if node.left is not None:
            codes.update(huffman_get_symbol_codes(node.left, new_value))
        if node.right is not None:
            codes.update(huffman_get_symbol_codes(node.right, new_value))
    
    return codes
    

def huffman_serialize_node(node: HuffmanNode, bits: bitarray = bitarray(endian='little')) -> bitarray:
    """
    Serializes a node to a bytearray.
    """
    node_bits = bitarray(endian='little')
    if node.left == None and node.right == None:
        node_bits.append(1)
        node_bits.frombytes(node.symbol.block)
    else:
        node_bits.append(0)
        node_bits.extend(huffman_serialize_node(node.left))
        node_bits.extend(huffman_serialize_node(node.right))
    return node_bits

def huffman_serialize_tree(tree: HuffmanTree) -> bitarray:
    """
    Serializes a tree to a bytearray.
    """

    bits = bitarray(endian='little')

    # how big a block is
    bits.extend(bitutil.int2ba(tree.block_size, length=32, endian='little'))
    # how many symbols are in the tree
    node_count = huffman_get_tree_node_count(tree.root)
    bits.extend(bitutil.int2ba(node_count, length=32, endian='little'))
    bits.extend(huffman_serialize_node(tree.root))
    return bits

node_count = 0
node_cur = 0
def huffman_deserialize_node(bits: bitarray, sym_len: int, freq: int = 0) -> HuffmanNode:
    """
    Deserializes a node from a bitarray.
    """
    global node_count, node_cur

    if node_count == node_cur:
        return None
    if len(bits) == 0:
        return None
        #raise Exception("No bits left to deserialize")

    if bits[0] == 0:
        #print("Deserializing branch node")
        # the huffman_deserialize_node call manipulates the bits array, so we don't need to manipulate it here
        # except for the deletion of the first bit
        del bits[0]
        left = huffman_deserialize_node(bits, sym_len, freq + 1)
        right = huffman_deserialize_node(bits, sym_len, freq + 1)
        if left is not None:
            left.code = bitarray('0', endian='little')
        if right is not None:
            right.code = bitarray('1', endian='little')

        node_cur += 1
        new_freq = left.symbol.freq if left is not None else 0 + right.symbol.freq if right is not None else 0

        return HuffmanNode(HuffmanSymbol(bytearray(), new_freq), left, right)
    elif bits[0] == 1:
        del bits[0]
        sym = bits[:(sym_len * 8)].tobytes()
        del bits[:sym_len * 8]
        #print("Leaf node", sym.hex(), freq)

        node_cur += 1

        return HuffmanNode(HuffmanSymbol(sym, freq), code=1)
        
def huffman_deserialize_tree(bits: bitarray) -> HuffmanTree:
    """
    Deserializes a tree from a bitarray.
    """
    global node_count, node_cur
    node_cur = 0
    
    sym_len = bitutil.ba2int(bits[:32],signed=False)
    node_count = bitutil.ba2int(bits[32:64], signed=False)
    print("Symbol Length:", sym_len)
    print("Amount of Nodes:", node_count)
    del bits[:64]

    root = huffman_deserialize_node(bits, sym_len)
    codes = huffman_get_symbol_codes(root)
    return HuffmanTree(root, codes, sym_len)
        
def huffman_build_system(data: bytearray, symbol_len: int) -> HuffmanTree:
    """
    Builds a Huffman tree from a bytearray.
    """
    symbols = huffman_create_freq_list(data, symbol_len)
    root = huffman_generate_tree(symbols)
    codes = huffman_get_symbol_codes(root)
    return HuffmanTree(root, codes, symbol_len)

# a huffman encoding generator, takes a system and data and yields a bitarray for the encoded value
def huffman_encode(data: bytearray, huffman_system: HuffmanTree) -> bitarray:
    """
    Encodes a bytearray using a Huffman tree.
    """
    for i in range(0, len(data), huffman_system.block_size):
        block = bytes(data[i:i+huffman_system.block_size])
        if len(block) < huffman_system.block_size:
            block += bytes(huffman_system.block_size - len(block))
        yield huffman_system.codes[block]

# decode system; takes a system and data and yields a bytearray for the decoded value
def huffman_decode(data: bitarray, huffman_system: HuffmanTree) -> bytearray:
    """
    Decodes a bitarray using a Huffman tree.
    """
    dec_data = bytearray()
    cur_node = huffman_system.root
    while(len(data) > 0):
        bit = data.pop(0)
        if bit == 0:
            cur_node = cur_node.left
        elif bit == 1:
            cur_node = cur_node.right
        if cur_node.left is None and cur_node.right is None:
            dec_data.extend(cur_node.symbol.block)
            cur_node = huffman_system.root
    return dec_data

@dataclass(repr=True, init=False)
class HuffmanProgram:
    """
    A class to hold the program parameters.
    """
    input_file: str
    output_file: str
    encode_mode: bool
    decode_mode: bool
    symbol_len: int

    def __init__(self, args: list[str]):
        parser = argparse.ArgumentParser(prog= "Huffman Encoder/Decoder" \
                                         , description="Encodes and decodes data using a Huffman tree. Written by: Janick Eicher" \
                                         , epilog= \
                                             """
                                             If no output is provided, then the data will be written into stdout.
                                             If no input is provided, then the data will be read from stdin.
                                             """)

        mode_group = parser.add_mutually_exclusive_group(required=True)
        mode_group.add_argument("-e", "--encode", action="store_true", help="Encode data.")
        mode_group.add_argument("-d", "--decode", action="store_true", help="Decode data.")

        parser.add_argument("-s", "--symbol-length", type=int, default=1, help="The length of the symbols in the data.")
        parser.add_argument("-i", "--input", dest="input_file", help="The file to encode/decode.")
        parser.add_argument("-o", "--output", dest="output_file", help="The file to write the encoded/decoded data to.")

        params = parser.parse_args(args)
        self.input_file = params.input_file
        self.output_file = params.output_file
        self.encode_mode = params.encode
        self.decode_mode = params.decode
        self.symbol_len = params.symbol_length

    def read_input(self) -> bytearray:
        """
        Reads from the input file, if specified, otherwise from stdin.
        """
        if self.input_file is None:
            return sys.stdin.buffer.read()
        else:
            with open(self.input_file, 'rb') as f:
                return bytearray(f.read())
    def write_output(self, data: bytearray) -> None:
        """
        Writes to the output file, if specified, otherwise to stdout.
        """
        if self.output_file is None:
            sys.stdout.buffer.write(data)
            sys.stdout.buffer.flush()
        else:
            with open(self.output_file, 'wb') as f:
                f.write(data)


def main():
    program = HuffmanProgram(sys.argv[1:])

    if program.encode_mode:
        data = program.read_input()
        print("File uses", len(data)*8, "bits")
        tree = huffman_build_system(data, program.symbol_len)
        bits = huffman_serialize_tree(tree)
        for b in huffman_encode(data, tree):
            bits.extend(b)
        print()
        #bits.extend(huffman_serialize_data(tree, data))
        #pprint(tree, 
        #   compact=True, 
        #   sort_dicts=True, 
        #   width=120, 
        #   stream=sys.stdout)
        program.write_output(bitutil.serialize(bits))

        exit(0)
    elif program.decode_mode:
        data = bitutil.deserialize(program.read_input())
        tree = huffman_deserialize_tree(data)
        dec_data = huffman_decode(data, tree)
        program.write_output(bytes(dec_data))
    else:
        print("Invalid mode.")
        exit(1)

main()