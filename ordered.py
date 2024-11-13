#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import argparse
import aiofiles
import json

TEMPLATE = r"""
#ifndef type
#define type double
#endif

#ifndef N
#define N 16
#endif

#ifndef M
#define M 16
#endif

volatile type A[M * N];
volatile type B[N * M];
volatile type C[M * M];

#define I_LOOP for (int i = 0; i < M; ++i)
#define J_LOOP for (int j = 0; j < M; ++j)
#define K_LOOP for (int k = 0; k < N; ++k)

#ifndef LOOP0 
#define LOOP0 I_LOOP
#endif

#ifndef LOOP1
#define LOOP1 J_LOOP
#endif

#ifndef LOOP2
#define LOOP2 K_LOOP
#endif

[[gnu::noinline]]
[[noreturn]]
[[gnu::leaf]]
void _start() {
  LOOP0 
  LOOP1
  LOOP2
  {
    C[i * M + j] += A[i * N + k] * B[k * M + j];
    asm volatile("" ::: "memory");
  }

  asm(
    "mov $0, %rdi\n\t"
    "mov $60, %rax\n\t"
    "syscall"
  );

    __builtin_unreachable();
}
"""



async def compile_matrix(tmpdir, M=16, N=16, type="double", order=('I_LOOP', 'J_LOOP', 'K_LOOP')):
    async with aiofiles.tempfile.NamedTemporaryFile(suffix='.c', mode='w', dir=tmpdir) as f:
        await f.write(TEMPLATE)
        await f.flush()
        compilation = await asyncio.create_subprocess_exec(
                'clang', '-static', '-nostdlib', '-fno-stack-protector', 
                '-fno-pic', '-O3', '-Dtype=' + type, 
                '-DM=' + str(M), '-DN=' + str(N), 
                '-DLOOP0=' + order[0], '-DLOOP1=' + order[1], '-DLOOP2=' + order[2],
                f.name, '-o', 'matmul.exe', cwd=tmpdir)
        await compilation.wait()

async def run_valgrind(tmpdir, cache_size, block_size, assoc=None):
    if assoc is None:
        assoc = cache_size // block_size
    simulation = await asyncio.create_subprocess_exec(
                'valgrind', '--tool=cachegrind', '--cache-sim=yes', 
                '--D1=' + str(cache_size) + ',' + str(assoc) + ',' + str(block_size), 
                './matmul.exe', cwd=tmpdir, 
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
    _, stderr = await simulation.communicate()
    drefs = 0
    d1_miss = 0
    for line in stderr.decode().split('\n'):
        if 'D1  misses' in line:
            d1_miss = int(line.split()[3].replace(',', ''))
        elif 'D refs' in line:
            drefs = int(line.split()[3].replace(',', ''))
    if drefs == 0:
        return 0
    return d1_miss / drefs

def order_generator():
    orders = ('I_LOOP', 'J_LOOP', 'K_LOOP')
    for i in range(3):
        for j in range(3):
            if i != j:
                yield (orders[i], orders[j], orders[3 - i - j])

def order_to_name(order):
    return ''.join([i.split('_')[0] for i in order])

def data_collect_tasks(order, cache_size, block_size, assoc=None, 
                          m_range=(8, 256, 1), n_range=(8, 256, 1)):
    async def _task(m, n):
        async with aiofiles.tempfile.TemporaryDirectory() as tmpdir:
            await compile_matrix(tmpdir, m, n, order=order)
            mr = await run_valgrind(tmpdir, cache_size, block_size, assoc)
            return (m, n, mr)
    for m in range(*m_range):
        for n in range(*n_range):
            yield _task(m, n)

async def batched_execute(order, batch_size=128, **kwargs):
    tasks = [t for t in data_collect_tasks(order, **kwargs)]
    results = []
    for i in range(0, len(tasks), batch_size):
        results += await asyncio.gather(*tasks[i:i+batch_size])
    return results

def name_to_order(name):
    orders = ('I_LOOP', 'J_LOOP', 'K_LOOP')
    return [orders['IJK'.index(i)] for i in name]


async def main():
    parser = argparse.ArgumentParser(description='Matrix multiplication with cache simulation')
    parser.add_argument('--type', type=str, default='double', help='data type of matrix')
    parser.add_argument('--size', type=int, default=128, help='size of matrix')
    parser.add_argument('--cache', type=int, default=1024, help='cache size')
    parser.add_argument('--block', type=int, default=32, help='block size')
    parser.add_argument('--assoc', type=int, default=None, help='associativity')
    parser.add_argument('--m-range', type=int, nargs=3, default=(8, 257, 8), help='range of M')
    parser.add_argument('--n-range', type=int, nargs=3, default=(8, 257, 8), help='range of N')
    parser.add_argument('--batch', type=int, default=128, help='batch size')
    parser.add_argument('--order', type=str, default='IJK', help='order of loop')
    parser.add_argument('--output', type=str, help='save to file')
    args = parser.parse_args()
    data = {}
    orders = []
    if args.order == 'ALL':
        orders = [o for o in order_generator()]
    else:
        orders = [name_to_order(o) for o in args.order.split(',')]
    
    for order in orders:
        results = await batched_execute(
            order, cache_size=args.cache, block_size=args.block, assoc=args.assoc,
            m_range=args.m_range, n_range=args.n_range, batch_size=args.batch)
        name = order_to_name(order)
        data[name] = results
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(data, f)
    else:
        json_str = json.dumps(data)
        print(json_str)

if __name__ == "__main__":
    asyncio.run(main())