
from pycparser import CParser, c_ast
import sys
from typing import Any
import ctypes


def parse_file(filename):
    with open(filename, 'r') as f:
        code = f.readlines()
    code = [line for line in code if not line.startswith('#')]
    code = [line.split('//')[0] for line in code]
    code = [line.strip() for line in code if line.strip()]
    code = '\n'.join(code)
    parser = CParser()
    return parser.parse(code, filename=filename)

import random
class Memory:
    pointer_size = 4
    def __init__(self):
        self.memory = {}
        self.functions = {}
    
    def get(self, addr: int, size: int):
        offset = addr % 2**16
        src = addr - offset
        # print(hex(src), hex(offset), size)
        # print(self.memory[src])
        if src in self.memory:
            return self.memory[src][offset:offset+size]
        raise KeyError("Segmentation fault")
    
    def set(self, addr: int, size: int, value: int|bytes):
        offset = addr % 2**16
        src = addr - offset
        if src not in self.memory:
            raise KeyError("Segmentation fault")
        if isinstance(value, int):
            self.memory[src][offset:offset+size] = value.to_bytes(size, 'little')
        else:
            self.memory[src][offset:offset+size] = value
    
    def malloc(self, size: int):
        addr = random.randint(0, 2**16)
        while addr in self.memory:
            addr = random.randint(0, 2**16)
        addr *= 2**16
        self.memory[addr] = bytearray(size)
        return addr

    def free(self, addr):
        del self.memory[addr]


class Variable:
    def __init__(self, ptr: int, vartype: str, count: int):
        self.ptr = ptr
        self.vartype = vartype
        self.count = count

class Address(int):
    def __new__(cls, value, size):
        obj = super().__new__(cls, value)
        obj.size = size
        return obj
    
    def __init__(self, value, size):
        self.size = size
    
    def __add__(self, other):
        if isinstance(other, int):
            return Address(super().__add__(other * self.size), self.size)
        return NotImplemented

    def __str__(self):
        return hex(self)
    
    def __repr__(self):
        return f"Address(0x{self:x}, size={self.size})"
        

import struct

class Context:
    def __init__(self, parent=None):
        self.parent = parent
        self.memory = Memory() if parent is None else parent.memory
        self.vars: dict[str, Variable] = {}
    
    def __getitem__(self, key: str|Address) -> Any:
        if isinstance(key, Address):
            return int.from_bytes(self.memory.get(key, key.size), 'little')
        if key in self.vars:
            var = self.vars[key]
            if '*' in var.vartype:
                return Address(int.from_bytes(self.memory.get(var.ptr, Memory.pointer_size), 'little'), Memory.pointer_size)
            elif var.count > 1:
                size = self.get_type_size(var.vartype)
                return Address(var.ptr, size)
            elif var.vartype in ['int', 'long', 'short', 'char']:
                size = self.get_type_size(var.vartype)
                return int.from_bytes(self.memory.get(var.ptr, size), 'little', signed=True)
            elif var.vartype in ['unsigned int', 'unsigned long', 'unsigned short', 'unsigned char']:
                size = self.get_type_size(var.vartype[9:])
                return int.from_bytes(self.memory.get(var.ptr, size), 'little', signed=False)
            elif var.vartype == 'float':
                return struct.unpack('f', self.memory.get(var.ptr, 4))[0]
            elif var.vartype == 'double':
                return struct.unpack('d', self.memory.get(var.ptr, 8))[0]
            elif var.vartype == 'function':
                return self.memory.functions[var.ptr]
            else:
                raise NotImplementedError(var.vartype)
        if self.parent is not None:
            return self.parent[key]
        raise KeyError('Segmentation fault')
    
    def __setitem__(self, key: str|Address, value: Any):
        if isinstance(key, Address):
            self.memory.set(key, key.size, value)
        else:
            if key in self.vars:
                var = self.vars[key]
                if '*' in var.vartype:
                    self.memory.set(var.ptr, Memory.pointer_size, value)
                elif var.vartype in ['int', 'long', 'short', 'char']:
                    size = self.get_type_size(var.vartype)
                    self.memory.set(var.ptr, size, value)
                elif var.vartype in ['unsigned int', 'unsigned long', 'unsigned short', 'unsigned char']:
                    size = self.get_type_size(var.vartype[9:])
                    self.memory.set(var.ptr, size, value.to_bytes(size, 'little', signed=False))
                elif var.vartype == 'float':
                    self.memory.set(var.ptr, 4, struct.pack('f', value))
                elif var.vartype == 'double':
                    self.memory.set(var.ptr, 8, struct.pack('d', value))
                elif var.vartype == 'function':
                    self.memory.functions[var.ptr] = value
                else:
                    raise NotImplementedError(var.vartype)
            elif self.parent is not None:
                self.parent[key] = value
            else:
                raise KeyError('Segmentation fault')

    def __del__(self):
        for key in self.vars:
            self.memory.free(self.vars[key].ptr)
    
    def assign(self, key: str, vartype: str, count: int) -> Address:
        size = self.get_type_size(vartype) * count
        ptr = self.memory.malloc(size)
        if vartype == 'function':
            self.memory.functions[ptr] = None
        addr = Address(ptr, self.get_type_size(vartype))
        self.vars[key] = Variable(addr, vartype, count)
        return addr

    def get_type_size(self, vartype: str) -> int:
        if "*" in vartype:
            return Memory.pointer_size
        if vartype == "function":
            return 1
        return ctypes.sizeof(getattr(ctypes, 'c_' + vartype))
    
    def free(self, key: str):
        var = self.vars[key]
        self.memory.free(var.ptr)
        del self.vars[key]

    def get_pointer(self, key: str) -> Address:
        if key not in self.vars:
            if self.parent is not None:
                return self.parent.get_pointer(key)
            raise KeyError('Segmentation fault')
        return Address(self.vars[key].ptr, Memory.pointer_size)

    def get_from_pointer(self, key: str) -> Any:
        if key not in self.vars:
            if self.parent is not None:
                return self.parent.get_from_pointer(key)
            raise KeyError('Segmentation fault')
        addr = self[key]
        size = self.get_type_size(self.vars[key].vartype)
        return self[Address(addr, size)]
    
    def create_child(self) -> 'Context':
        return Context(self)
    
    def __str__(self):
        # return str({
        total = {}
        target = self
        while target is not None:
            total.update({key: (target[key] if target.vars[key].vartype != 'function' else f'<function \'{key}>\' at {target.vars[key].ptr}') for key in target.vars})
            target = target.parent
        return str(total)


class Executer:
    def __init__(self, ast: c_ast.Node):
        self.ast = ast
        self.external_functions = ['printf']
    
    def read_string(self, addr: Address|int, context: Context):
        res = ""
        i = 0
        while i < 100:
            v = context[addr + i]
            res += chr(v)
            if v == 0:
                break
            i += 1
        return res

    def call_external(self, name: str, context: Context, args: list):
        if name == 'printf':
            fmt = self.read_string(args[0], context)
            arg_value = args[1:]
            for i, v in enumerate(arg_value):
                if isinstance(v, Address):
                    arg_value[i] = self.read_string(v, context)
            print(fmt % tuple(arg_value))

        else:
            raise NotImplementedError(name)

    def execute_FileAST(self, node: c_ast.FileAST, context: Context):
        for ext in node.ext:
            self.execute(ext, context)
    
    def execute_IdentifierType(self, node: c_ast.IdentifierType, context: Context):
        return node.names[0]

    def execute_TypeDecl(self, node: c_ast.TypeDecl, context: Context):
        return self.execute(node.type, context)
    
    def execute_PtrDecl(self, node: c_ast.PtrDecl, context: Context):
        return self.execute(node.type, context) + '*'

    def execute_ID(self, node: c_ast.ID, context: Context):
        return node.name

    def execute_Decl(self, node: c_ast.Decl, context: Context):
        res = None
        if isinstance(node.type, c_ast.TypeDecl):
            context.assign(node.name, self.execute(node.type, context), 1)
            if node.init:
                res = self.execute(node.init, context)
                context[node.name] = res
        elif isinstance(node.type, c_ast.PtrDecl):
            name = node.name
            type_name = self.execute(node.type, context)
            context.assign(name, type_name, 1)
            if node.init:
                res = self.execute(node.init, context)
                context[name] = res
        elif isinstance(node.type, c_ast.ArrayDecl):
            name = node.name
            length = self.execute(node.type.dim, context)
            type_name = self.execute(node.type.type, context)
            addr = context.assign(name, type_name, length)
            if node.init:
                res = self.execute(node.init, context)
                i = 0
                while i < 5:
                    context[addr + i] = context[res + i]
                    if context[res + i] == 0:
                        break
                    i += 1
        else:
            raise NotImplementedError(node.type.__class__.__name__)

        return res

    def execute_DeclList(self, node: c_ast.DeclList, context: Context):
        for decl in node.decls:
            self.execute(decl, context)

    def execute_ArrayRef(self, node: c_ast.ArrayRef, context: Context, left=False):
        if left:
            addr = self.evaluate(node.name, context)
            index = self.evaluate(node.subscript, context)
            new_addr = addr + index
            return new_addr
        else:
            addr = self.evaluate(node.name, context, left=True)
            index = self.evaluate(node.subscript, context)
            new_addr = addr + index
            return context[new_addr]

    def execute_FuncDef(self, node: c_ast.FuncDef, context: Context):
        name = node.decl.name
        context.assign(name, 'function', 1)
        context[name] = node
    
    def execute_InitList(self, node: c_ast.InitList, context: Context):
        return [self.execute(n, context) for n in node.exprs]

    def execute_UnaryOp(self, node: c_ast.UnaryOp, context: Context, left=False):
        if left and not node.op == '*':
            raise SyntaxError("Invalid left operand")
        if node.op == '&':
            if isinstance(node.expr, c_ast.ID):
                return context.get_pointer(node.expr.name)
            else:
                raise NotImplementedError(node.expr.__class__.__name__)
        elif node.op == '*':
            if isinstance(node.expr, c_ast.ID):
                if left:
                    return context[node.expr.name]
                else:
                    return context.get_from_pointer(node.expr.name)
            else:
                raise NotImplementedError(node.expr.__class__.__name__)
        val = self.evaluate(node.expr, context)
        if node.op == '+':
            return val
        elif node.op == '-':
            return -val
        elif node.op == '!':
            return not val
        elif node.op == '~':
            return ~val
        elif node.op == 'p++':
            val_left = self.evaluate(node.expr, context, left=True)
            context[val_left] += 1
            return val
        elif node.op == 'p--':
            val_left = self.evaluate(node.expr, context, left=True)
            context[val_left] -= 1
            return val
        elif node.op == '++':
            val_left = self.evaluate(node.expr, context, left=True)
            context[val_left] += 1
            return context[val_left]
        elif node.op == '--':
            val_left = self.evaluate(node.expr, context, left=True)
            context[val_left] -= 1
            return context[val_left]
        else:
            raise NotImplementedError(node.op)
    
    def execute_BinaryOp(self, node: c_ast.BinaryOp, context: Context):
        left = self.evaluate(node.left, context)
        right = self.evaluate(node.right, context)
        if node.op == '+':
            return left + right
        elif node.op == '-':
            return left - right
        elif node.op == '*':
            return left * right
        elif node.op == '/':
            return left / right
        elif node.op == '%':
            return left % right
        elif node.op == '<<':
            return left << right
        elif node.op == '>>':
            return left >> right
        elif node.op == '&':
            return left & right
        elif node.op == '|':
            return left | right
        elif node.op == '^':
            return left ^ right
        elif node.op == '&&':
            return left and right
        elif node.op == '||':
            return left or right
        elif node.op == '==':
            return left == right
        elif node.op == '!=':
            return left != right
        elif node.op == '<':
            return left < right
        elif node.op == '>':
            return left > right
        elif node.op == '<=':
            return left <= right
        elif node.op == '>=':
            return left >= right
        else:
            raise NotImplementedError(node.op)
    
    def execute_Compound(self, node: c_ast.Compound, context: Context):
        if node.block_items is None:
            return
        for decl in node.block_items:
            if isinstance(decl, c_ast.Return):
                return self.execute(decl, context)
            self.execute(decl, context)
    
    def execute_Constant(self, node: c_ast.Constant, context: Context):
        if node.type == 'int':
            return int(node.value)
        elif node.type == 'float':
            return float(node.value)
        elif node.type == 'double':
            return float(node.value)
        elif node.type == 'char':
            return ord(node.value[1])
        elif node.type == 'string':
            val = node.value[1:-1]
            str_hash = hash(val)
            if f'str_{str_hash}' not in context.vars:
                addr = context.assign(f'str_{str_hash}', 'char', len(val))
                for i, c in enumerate(val):
                    context[addr + i] = ord(c)
            return context.vars[f'str_{str_hash}'].ptr
        else:
            raise NotImplementedError(node.type)
    
    def execute_Assignment(self, node: c_ast.Assignment, context: Context):
        val = self.evaluate(node.rvalue, context)
        if node.op == '=':
            addr = self.evaluate(node.lvalue, context, left=True)
            context[addr] = val
        elif node.op == '+=':
            addr = self.evaluate(node.lvalue, context, left=True)
            context[addr] += val
        elif node.op == '-=':
            addr = self.evaluate(node.lvalue, context, left=True)
            context[addr] -= val
        else:
            raise NotImplementedError(node.op)

    def execute_FuncCall(self, node: c_ast.FuncCall, context: Context):
        args = [self.evaluate(arg, context) for arg in node.args.exprs]
        if node.name.name in self.external_functions:
            return self.call_external(node.name.name, context, args)
        ftn = context[node.name.name]
        new_context = context.create_child()
        for i, arg in enumerate(args):
            new_context.assign(ftn.decl.type.args.params[i].name, self.execute(ftn.decl.type.args.params[i].type, context), 1)
            new_context[ftn.decl.type.args.params[i].name] = arg
        return self.execute(ftn.body, new_context)
    
    def execute_Return(self, node: c_ast.Return, context: Context):
        return self.evaluate(node.expr, context)
    
    def execute_If(self, node: c_ast.If, context: Context):
        new_context = context.create_child()
        if self.evaluate(node.cond, new_context):
            return self.execute(node.iftrue, new_context)
        else:
            if node.iffalse is not None:
                return self.execute(node.iffalse, new_context)
    
    def execute_For(self, node: c_ast.For, context: Context):
        new_context = context.create_child()
        self.execute(node.init, new_context)
        while self.evaluate(node.cond, new_context):
            res = self.execute(node.stmt, new_context)
            if res is not None:
                return res
            self.execute(node.next, new_context)
        return res

    def evaluate(self, node, context: Context, left=False):
        if isinstance(node, c_ast.ID):
            if left:
                return context.get_pointer(node.name)
            else:
                return context[node.name]
        else:
            return self.execute(node, context, left)

    def execute(self, node, context: Context, left=False):
        node_name = node.__class__.__name__
        method_name = 'execute_' + node_name
        if hasattr(self, method_name):
            if left:
                return getattr(self, method_name)(node, context, left)
            else:
                return getattr(self, method_name)(node, context)
        else:
            print('No method for', node_name)
            print(node)
            print("Children:", "  ".join([t.__class__.__name__ for n, t in node.children()]))
            sys.exit(1)

    def run(self):
        context = Context()
        self.execute(self.ast, context)
        
        ftncall = c_ast.FuncCall(c_ast.ID('main'), c_ast.ExprList([]))
        ret = self.execute(ftncall, context)
        print(ret)

        print(context)


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: %s <filename>" % sys.argv[0])
        sys.exit(1)
        
    filename = sys.argv[1]
    ast = parse_file(filename)

    executer = Executer(ast)
    executer.run()

