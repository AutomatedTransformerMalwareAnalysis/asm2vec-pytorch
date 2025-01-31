import argparse
import hashlib
import os
from pathlib import Path
import r2pipe
import re
from tqdm import tqdm
from typing import Dict

class Bin2Asm(object):
    '''
    
    '''
    def __init__(self, inputDir: str, outputDir: str, maxDisasmLen: int = 0) -> None:
        '''

        '''
        self.inputDir = inputDir
        self.outputDir = outputDir
        self.maxLength = maxDisasmLen


    def processDirFiles(self) -> None:
        # Ensure output directory exists
        if not os.path.exists(self.outputDir):
            Path(self.outputDir).mkdir(parents=True, exist_ok=True)
        # Iterate through files in source directory
        for inode in tqdm(os.listdir(self.inputDir)):
            fullPath = os.path.join(self.inputDir, inode)
            if os.path.isfile(fullPath):
                try:
                    # Open a pipeline for radare2
                    pipe = r2pipe.open(fullPath, flags=["-2"])
                    pipe.cmd("aaaa")
                    # Extract asm code function-by-function
                    for fnCall in pipe.cmdj("aflj"):
                        pipe.cmd(f"s {fnCall['offset']}")
                        asm = self.fnToAsm(pipe.cmdj("pdfj"))
                        if asm:
                            # Prepare data for writing
                            fhash = hashlib.sha3_256(inode.encode()).hexdigest()
                            uid = hashlib.sha3_256(asm.encode()).hexdigest()
                            asm = f" .name {fnCall['name']}\n .offset {fnCall['offset']:016x}\n .file{fullPath}\n{asm}"
                            # Create subdirectory for a given executable
                            if not os.path.exists(os.path.join(self.outputDir, fhash)):
                                os.mkdir(os.path.join(self.outputDir, fhash))
                            # Write asm to file
                            with open(os.path.join(self.outputDir, fhash, uid), 'w') as asmWrite:
                                asmWrite.write(asm)
                except Exception as e:
                    print(f"[ERR] Could not disassemble file {inode}\n{e}")
    

    def fnToAsm(self, r2Output: Dict) -> str | None:
        '''
        
        '''
        # Error checking
        if(r2Output is None) or ("invalid" in [op["type"] for op in r2Output["ops"]]):
            return None
        # Set labels
        ops = r2Output["ops"]
        labels, scope = {}, [op["offset"] for op in ops]
        if(None in scope):
            return None
        for i, op in enumerate(ops):
            if op.get("jump") in scope:
                labels.setdefault(op.get("jump"), i)
        # Dump output
        output = ""
        count = 0
        for op in ops:
            # add label
            if labels.get(op.get('offset')) is not None:
                output += f'LABEL{labels[op["offset"]]}:\n'
            # add instruction
            if labels.get(op.get('jump')) is not None:
                output += f' {op["type"]} LABEL{labels[op["jump"]]}\n'
            else:
                output += f' {normalize(op["opcode"])}\n'
            count += 1
            # Early stop
            if(self.maxLength > 0) and (count >= self.maxLength):
                break
        return output


def normalize(opcode):
    opcode = opcode.replace(' - ', ' + ')
    opcode = re.sub(r'0x[0-9a-f]+', 'CONST', opcode)
    opcode = re.sub(r'\*[0-9]', '*CONST', opcode)
    opcode = re.sub(r' [0-9]', ' CONST', opcode)
    return opcode

# --- Script Code ------------------------------------------------------------------------------------------------------
def parseArgv() -> argparse.Namespace:
    '''
    
    '''
    parser = argparse.ArgumentParser(prog="Binary to Assembly Script",
                                     description="Extract assembly calls from the executables contained within a given directory.",
                                     formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument("--input",
                        help="Directory containing executables to be processed.",
                        type=str,
                        required=True)
    parser.add_argument("--output",
                        help="Destination directory to contain processed function calls.",
                        type=str,
                        required=True)
    return parser.parse_args()


# def main(argc, argv):
if __name__  == "__main__":
    argv = parseArgv()
    b2a = Bin2Asm(argv.input, argv.output)
    b2a.processDirFiles()
