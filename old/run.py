#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import asyncio
import argparse
import tempfile
import copy
import aiofiles

SRC=r"""
#ifndef type
#define type double
#endif
[[gnu::noinline]]
void matmul(type volatile * A, type volatile * B, type volatile * C, int n) {
        for (int i = 0; i < n; ++i)
                for (int j = 0; j < N; ++j)
                        for (int k = 0; k < n; ++k) {
                                C[i * n + j] += A[i * n + k] * B[k * n + j];
                                asm volatile("":::"memory");
        }
}

#ifndef N
#define N 16
#endif

type A[N * N];
type B[N * N];
type C[N * N];

extern "C" void _start()
{
    int argc;
    char **argv, **envp;
    int ret = 0;
    asm volatile(
        "mov %%edi, %0\n\t"
        "mov %%esi, %1\n\t"
        "mov %%edx, %2\n\t"
        : "=m"(argc), "=m"(argv), "=m"(envp)
    );
    asm volatile(
        "mov %0, %%edi\n\t"
        "mov %1, %%esi\n\t"
        "mov %2, %%edx\n\t"
        "call main\n\t"
        "mov %%eax, %%ebx\n\t"
        :
        : "g"(argc), "g"(argv), "g"(envp), "g"(ret)
        : "%edi", "%esi", "%edx", "%ebx"
    );
    asm volatile(
        "mov %0, %%edi\n\t"
        "mov $60, %%eax\n\t"
        "syscall"
        :
        : "g"(ret)
    );
}

int main() {
    matmul(A, B, C, N);
    return 0;
}
"""

# arguments
# type: data type of matrix
# size: size of matrix
# cache: cache size
# block: block size
# assoc: associativity, if unspecified, use full associativity (cache size/block size)

async def execute(args):
    async with aiofiles.tempfile.TemporaryDirectory() as dir:
        async with aiofiles.open(dir + '/matmul.cpp', 'w') as f:
            await f.write(SRC)
            await f.flush()
            compilation = await asyncio.create_subprocess_exec(
                'clang++', '-static', '-nostdlib', '-fno-stack-protector', '-fno-pic', '-O3', '-Dtype=' + args.type, '-DN=' + str(args.size), f.name, '-o', f.name + '.exe', cwd=dir)
            await compilation.wait()
            simulation = await asyncio.create_subprocess_exec(
                'valgrind', '--tool=cachegrind', '--cache-sim=yes', '--D1=' + str(args.cache) + ',' + str(args.assoc) + ',' + str(args.block), f.name + '.exe', cwd=dir, 
                stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE)
            _, stderr = await simulation.communicate()
            drefs = 0
            d1_miss = 0
            if args.remove_cold:
                cold_miss = args.size * args.size * 3 / args.block
            else:
                cold_miss = 0
            for line in stderr.decode().split('\n'):
                if 'D1  misses' in line:
                    d1_miss = int(line.split()[3].replace(',', ''))
                elif 'D refs' in line:
                    drefs = int(line.split()[3].replace(',', ''))
            if drefs == 0:
                return 0
            return (d1_miss - cold_miss) / drefs

async def batch_execute(args):
    tasks = {}
    res = {}
    for i in range(args.batch_start, args.batch_end+1, args.batch_step):
        my_args = copy.deepcopy(args)
        if args.batch == 'cache':
            my_args.cache = i
            my_args.assoc = my_args.cache // my_args.block
        elif args.batch == 'size':
            my_args.size = i
        tasks[i] = asyncio.create_task(execute(my_args))
    for k, v in tasks.items():
        res[k] = await v
    return res
        

def main():
    parser = argparse.ArgumentParser(description='Matrix multiplication with cache simulation')
    parser.add_argument('--type', type=str, default='double', help='data type of matrix')
    parser.add_argument('--size', type=int, default=128, help='size of matrix')
    parser.add_argument('--cache', type=int, default=1024, help='cache size')
    parser.add_argument('--block', type=int, default=32, help='block size')
    parser.add_argument('--assoc', type=int, default=None, help='associativity')
    parser.add_argument('--remove-cold', type=bool, default=False, help='remove cold misses')
    # batch cache size or data size, value can only be 'cache' or 'size' or ''
    parser.add_argument('--batch', type=str, default='', help='batch size')
    parser.add_argument('--batch-start', type=int, default=0, help='batch start value')
    parser.add_argument('--batch-end', type=int, default=0, help='batch end value')
    parser.add_argument('--batch-step', type=int, default=1, help='batch step value')
    args = parser.parse_args()
    if args.assoc is None:
        args.assoc = args.cache // args.block
    if args.batch:
        for k, v in asyncio.run(batch_execute(args)).items():
            print(k, v, sep=',')
    else:
        print(asyncio.run(execute(args)))

if __name__ == '__main__':
    main()
