#!/usr/bin/env python3
import re
import os
import r2pipe
import hashlib
from pathlib import Path

def sha3(data):
    return hashlib.sha3_256(data.encode()).hexdigest()

def validEXE(filename):
    magics = [bytes.fromhex('7f454c46')]
    with open(filename, 'rb') as f:
        header = f.read(4)
        return header in magics

def normalize(opcode):
    opcode = opcode.replace(' - ', ' + ')
    opcode = re.sub(r'0x[0-9a-f]+', 'CONST', opcode)
    opcode = re.sub(r'\*[0-9]', '*CONST', opcode)
    opcode = re.sub(r' [0-9]', ' CONST', opcode)
    return opcode

def fn2asm(pdf, minlen, maxlen):
    # check
    if pdf is None:
        return
    if len(pdf['ops']) < minlen:
        return
    if 'invalid' in [op['type'] for op in pdf['ops']]:
        return

    ops = pdf['ops']

    # set label
    labels, scope = {}, [op['offset'] for op in ops]
    assert(None not in scope)
    for i, op in enumerate(ops):
        if op.get('jump') in scope:
            labels.setdefault(op.get('jump'), i)
    
    # dump output
    output = ''
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
        if((maxlen is not None) and (count >= maxlen)):
            break

    return output

def bin2asm(filename, opath, minlen: int, maxlen: int | None) -> int:
    r = r2pipe.open(str(filename))
    r.cmd('aaaa')

    count = 0

    for fn in r.cmdj('aflj'):
        r.cmd(f's {fn["offset"]}')
        asm = fn2asm(r.cmdj('pdfj'), minlen, maxlen)
        if asm:
            uid = sha3(asm)
            # I'm not touching that multiline string, it looks important and I don't want to mess up formatting
            asm = f''' .name {fn["name"]}
 .offset {fn["offset"]:016x}
 .file {filename.name}
''' + asm
            with open(opath / uid, 'w') as f:
                f.write(asm)
                count += 1

    print(f'[+] {filename}')

    return count


def cli(ipath: str, opath: str, minlen: int = 0, maxlen: int | None = None) -> None:
    # Check asm length params
    if((minlen > maxlen) or (maxlen <= 0)):
        raise ValueError("Incompatible disassembly limits {minlen} | {maxlen}")
    '''
    Extract assembly functions from binary executable
    '''
    ipath = Path(ipath)
    opath = Path(opath)

    # create output directory
    if not os.path.exists(opath):
        os.mkdir(opath)

    fcount, bcount = 0, 0

    # directory
    if os.path.isdir(ipath):
        for f in os.listdir(ipath):
            try:
                if not os.path.islink(ipath / f) and not os.path.isdir(ipath / f):
                    fcount += bin2asm(ipath / f, opath, minlen, maxlen)
                    bcount += 1
            except Exception as e:
                print(f"[ERR] Unable to process file {f}\n{e}")
    # file
    elif os.path.exists(ipath):
        try:
            fcount += bin2asm(ipath, opath, minlen, maxlen)
            bcount += 1
        except Exception as e:
            print(f"[ERR] Unable to process {ipath}\n{e}")
    else:
        print(f'[Error] No such file or directory: {ipath}')

if __name__ == '__main__':
    cli()
