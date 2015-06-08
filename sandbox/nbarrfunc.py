from numba import types, compiler
from dynd import nd, ndt

# Shove all compilation results in a global list to prevent them getting deleted!
compile_results = []

def arrfunc(f):
    res = compiler.compile_isolated(f, (types.int32, types.int32))
    compile_results.append(res)
    ptr = res.library.get_pointer_to_function(res.fndesc.llvm_func_name)
    proto = ndt.type('(int, int) -> int')
    return nd.arrfunc(ptr, proto)

@arrfunc
def add(a, b):
    return a + b

print(add(1, 2))

@arrfunc
def sumsq(a, b):
    return a * a + b * b

print(sumsq(4, 5))
print(add(sumsq(4, 5), 6))
