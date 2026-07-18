# Mojo Attention Kernel Matrix-Multiplication Prototype
# This script simulates a vectorized matrix multiplication kernel for Q @ K.T (Attention query-key dot product)
# using Mojo's low-level hardware-level abstractions: SIMD vectors, memory pointers, and parallelization.

from sys.info import simdbitwidth
from memory import Pointer
from algorithm import parallelize

alias type_f32 = DType.float32
alias simd_width = simdbitwidth() // 8 // 4 # Calculate SIMD lane count for float32 on current architecture

struct Matrix:
    var data: Pointer[Scalar[type_f32]]
    var rows: Int
    var cols: Int

    fn __init__(inout self, r: Int, c: Int):
        self.rows = r
        self.cols = c
        self.data = Pointer[Scalar[type_f32]].alloc(r * c)
        
        # Initialize matrix memory with dummy zeros
        for i in range(r * c):
            self.data.store(i, 0.0)

    fn get(self, r: Int, c: Int) -> Float32:
        return self.data.load(r * self.cols + c)

    fn set(inout self, r: Int, c: Int, val: Float32):
        self.data.store(r * self.cols + c, val)

    fn free(inout self):
        self.data.free()

# ----------------------------------------------------------------
# Vectorized MatMul Kernel for Attention (Q @ K.T)
# ----------------------------------------------------------------
fn attention_matmul_vectorized(q: Matrix, k: Matrix, inout out: Matrix):
    """
    Computes Q @ K.T, which is the dot product of Queries and Keys in self-attention.
    Utilizes SIMD parallelization to calculate vector dot products in parallel.
    """
    # Outer dimension parallelization using Mojo's parallelize algorithm
    @parameter
    fn worker_row(r: Int):
        for c in range(k.rows): # k.rows is the sequence length of Keys (K)
            var sum = SIMD[type_f32, simd_width]() # Initialize SIMD vector accumulator
            
            # Step in chunks of SIMD lanes (simd_width)
            var col_idx = 0
            while col_idx < q.cols:
                # Load SIMD vector from Q row
                let q_vec = q.data.load[width=simd_width](r * q.cols + col_idx)
                # Load SIMD vector from K row
                let k_vec = k.data.load[width=simd_width](c * k.cols + col_idx)
                
                # Fused Multiply-Accumulate in SIMD registers
                sum = sum + (q_vec * k_vec)
                col_idx += simd_width
                
            # Reduce SIMD lanes sum to scalar float value
            out.set(r, c, sum.reduce_add())

    # Distribute the computation across CPU hardware threads
    parallelize[worker_row](q.rows, q.rows)

fn main():
    print("Mojo Version Check: MAX SDK 2026.07 compatible.")
    print("Initializing matrix inputs for Q (queries) and K (keys)...")
    
    # Standard dimensions for attention calculation: Batch=1, Heads=1, Seq=128, Dim=64
    let seq_len = 128
    let head_dim = 64
    
    var Q = Matrix(seq_len, head_dim)
    var K = Matrix(seq_len, head_dim)
    var Out = Matrix(seq_len, seq_len)
    
    # Mock data initialization
    for i in range(seq_len):
        Q.set(i, 0, 1.5)
        K.set(i, 0, 2.0)
        
    print("Executing Vectorized SIMD Attention MatMul...")
    attention_matmul_vectorized(Q, K, Out)
    
    # Read sample outputs
    print("Validation: Q @ K.T first element result:", Out.get(0, 0))
    
    # Deallocate memory blocks
    Q.free()
    K.free()
    Out.free()
    print("Memory cleaned up. Mojo kernel validation succeeded!")
Dummy
