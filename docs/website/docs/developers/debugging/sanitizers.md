---
icon: material/broom
---

# Sanitizers (ASan/MSan/TSan)

[AddressSanitizer](https://clang.llvm.org/docs/AddressSanitizer.html),
[MemorySanitizer](https://clang.llvm.org/docs/MemorySanitizer.html) and
[ThreadSanitizer](https://clang.llvm.org/docs/ThreadSanitizer.html) are tools
provided by `clang` to detect certain classes of errors in C/C++ programs. They
consist of compiler instrumentation (so your program's executable code is
modified) and runtime libraries (so e.g. the `malloc` function may get
replaced).

They are abbreviated as "ASan", "MSan" and "TSan" respectively.

They all incur large overhead, so only enable them while debugging.

Tool   | Detects | Helps debug what? | Slowdown | Memory overhead | Android support
------ | ------- | ----------------- | -------- | --------------- | ---------------
ASan   | Out-of-bounds accesses, use-after-free, use-after-return, memory leaks | Crashes, non-deterministic results, memory leaks | 2x | 3x | Yes
MSan   | Uninitialized memory reads | Non-deterministic results | 3x | ? | Yes
TSan   | Data races | Many bugs in multi-thread code | 5x-15x | 5x-10x | [No](https://github.com/android/ndk/issues/1171)

!!! note

    See
    [this documentation](https://clang.llvm.org/docs/AddressSanitizer.html#memory-leak-detection)
    on leak detection. It is only enabled by default on some platforms.

## Support status and how to enable each sanitizer

### ASan (AddressSanitizer)

To enable ASan:

```shell
cmake -DIREE_ENABLE_ASAN=ON ...
```

Several `_asan` tests like
`iree/tests/e2e/stablehlo_ops/check_llvm-cpu_local-task_asan_abs.mlir` are
also defined when using this configuration. These tests include AddressSanitizer
in compiled CPU code as well by using these `iree-compile` flags:

```shell
--iree-llvmcpu-link-embedded=false
--iree-llvmcpu-sanitize=address
```

#### Linking to the dynamic ASan runtime

You may want to use ASan when using the python bindings.
One way to achieve this is to build Python (or whatever executable that is
going to use IREE as a shared library) with Asan.
Another option is to link to the ASan runtime dynamically instead of
linking it statically into an executable.

Using clang-12 (other versions should also work) as a example, configure IREE
with something like:

```shell
cmake \
  -DIREE_ENABLE_ASAN=ON \
  -DCMAKE_EXE_LINKER_FLAGS=-shared-libasan \
  -DCMAKE_SHARED_LINKER_FLAGS=-shared-libasan \
  -DCMAKE_C_COMPILER=clang-12 \
  -DCMAKE_CXX_COMPILER=clang++-12 \
  ...
```

Then when running things the ASan runtime will have to be preloaded.

```shell
LD_PRELOAD=/usr/lib/llvm-12/lib/clang/12.0.0/lib/linux/libclang_rt.asan-x86_64.so \
ASAN_SYMBOLIZER_PATH=/usr/lib/llvm-12/bin/llvm-symbolizer \
  python ...
```

On Ubuntu the corresponding ASan runtime is provided by a package like
`libclang-common-12-dev` depending on your Clang version.
E.g.

```shell
sudo apt install libclang-common-12-dev llvm-12 clang-12
```

Note that during building would also need to preload the ASan runtime, since
the build executes its own binaries that are linked against the runtime.

```shell
LD_PRELOAD=/usr/lib/llvm-12/lib/clang/12.0.0/lib/linux/libclang_rt.asan-x86_64.so \
ASAN_OPTIONS=detect_leaks=0 \
ASAN_SYMBOLIZER_PATH=/usr/lib/llvm-12/bin/llvm-symbolizer \
  cmake --build ...
```

!!! tip "Tip - ASan stack traces from Python"

    When properly configured, you should see stack trace symbols from ASan
    reports, even when running Python code.

    If you see stack traces pointing at `site-packages`, you are using an
    installed package from pip and _not_ your source build with ASan!

    ```
    #0 0x7fbffd1712d8  (/.venv/lib/python3.11/site-packages/iree/_runtime_libs/_runtime.cpython-311-x86_64-linux-gnu.so+0xae2d8) (BuildId: 32e87a22f20d0241)
    #1 0x7fbffd1e5d78  (/.venv/lib/python3.11/site-packages/iree/_runtime_libs/_runtime.cpython-311-x86_64-linux-gnu.so+0x122d78) (BuildId: 32e87a22f20d0241)
    #2 0x7fbffd1e5b86  (/.venv/lib/python3.11/site-packages/iree/_runtime_libs/_runtime.cpython-311-x86_64-linux-gnu.so+0x122b86) (BuildId: 32e87a22f20d0241)
    #3 0x7fbffd11882d  (/.venv/lib/python3.11/site-packages/iree/_runtime_libs/_runtime.cpython-311-x86_64-linux-gnu.so+0x5582d) (BuildId: 32e87a22f20d0241)
    #4 0x5af471  (/usr/bin/python3.11+0x5af471) (BuildId: ead95fcf0410547669743f801bc8c549efbdf7ce)
    ```

    To fix this, uninstall the packages and ensure that you have your
    `PYTHONPATH` pointing at your build directory:

    ```shell hl_lines="14-18"
    python -m pip uninstall iree-base-runtime
    python -m pip uninstall iree-base-compiler
    source iree-build/.env && export PYTHONPATH

    LD_PRELOAD=/usr/lib/llvm-12/lib/clang/12.0.0/lib/linux/libclang_rt.asan-x86_64.so \
    ASAN_SYMBOLIZER_PATH=/usr/lib/llvm-12/bin/llvm-symbolizer \
    ASAN_OPTIONS="detect_leaks=0" \
      python ...

    AddressSanitizer:DEADLYSIGNAL
    =================================================================
    ==229852==ERROR: AddressSanitizer: SEGV on unknown address 0x7f66510ff050 (pc 0x7f66efa5f25e bp 0x7fff9db6e9d0 sp 0x7fff9db6e950 T0)
    ==229852==The signal is caused by a READ memory access.
        #0 0x7f66efa5f25e in __flatbuffers_soffset_read iree/third_party/flatcc/include/flatcc/flatcc_endian.h:89:2
        #1 0x7f66efa5f25e in __flatbuffers_soffset_read_from_pe iree/third_party/flatcc/include/flatcc/flatcc_endian.h:89:2
        #2 0x7f66efa5f25e in iree_vm_BytecodeModuleDef_exported_functions iree-build/runtime/src/iree/schemas/bytecode_module_def_reader.h:693:1
        #3 0x7f66efa5f25e in iree_vm_bytecode_module_lookup_function iree/runtime/src/iree/vm/bytecode/module.c:292:9
        #4 0x7f66efb5b497 in iree_vm_context_run_function iree/runtime/src/iree/vm/context.c:77:26
    ```

!!! tip "Tip - Using the CUDA driver with ASan from Python"

    If you want to run the IREE CUDA runtime driver it is likely you would
    need.

    ```shell
    ASAN_OPTIONS="protect_shadow_gap=0"
    ```

    Like this

    ```shell
    LD_PRELOAD=/usr/lib/llvm-12/lib/clang/12.0.0/lib/linux/libclang_rt.asan-x86_64.so \
    ASAN_SYMBOLIZER_PATH=/usr/lib/llvm-12/bin/llvm-symbolizer \
    ASAN_OPTIONS="protect_shadow_gap=0" \
      python ...
    ```

### TSan (ThreadSanitizer)

To enable TSan:

```shell
cmake -DIREE_ENABLE_TSAN=ON ...
```

You may also need:

* Depending on your system (see
  <https://github.com/google/benchmark/issues/773#issuecomment-616067912>):

    ```shell
    -DRUN_HAVE_STD_REGEX=0 \
    -DRUN_HAVE_POSIX_REGEX=0 \
    -DCOMPILE_HAVE_GNU_POSIX_REGEX=0 \
    ```

* For clang < 18.1.0 on system with `vm.mmap_rnd_bits` > 28 (see
  <https://stackoverflow.com/a/77856955>):

    ```shell
    sudo sysctl vm.mmap_rnd_bits=28
    ```

    TSan in LLVM >= 18.1.0 supports 30 bits of ASLR entropy. If the layout is
    unsupported, TSan will automatically re-execute without ASLR.

* If running under Docker, add `--privileged` to your `docker run` command

#### C++ Standard Library with TSan support

For best results to avoid false positives/negatives TSan needs all userspace
code to be compiled with Tsan.
This includes `libstdc++` or `libc++`.
libstdc++ is usually the default C++ runtime on Linux.

Building GCC's 12 libstdc++ on Ubuntu 22.04 with Clang has build errors.
It seems that GCC and Clang shared their
[TSan implementation](https://github.com/google/sanitizers).
They may be interoperable, but to avoid problems we should build everything
with GCC.
This means using GCC both as a compiler and linker.

##### Build libstdc++ with TSan support

Get GCC 12.3 source code.

```bash
git clone --depth 1 --branch releases/gcc-12.3.0 \
  https://github.com/gcc-mirror/gcc.git
```

```bash
SRC_DIR=$PWD/gcc
BIN_DIR=$PWD/gcc/build
```

Building all dependencies of libstdc++ with TSan has errors during linking of
`libgcc`.
libgcc is a dependency of libstdc++.
It is desirable to build everything with TSan, but it seems this excludes
libgcc, as the TSan runtime `libtsan` has it as a dependency.
We build it without TSan.
We do that to make libstdc++'s configuration find `gthr-default.h`, which
is generated during building of libgcc.
If not found C++ threads will silently have missing symbols.

```bash
LIBGCC_BIN_DIR=$BIN_DIR/libgcc
mkdir -p $LIBGCC_BIN_DIR
cd $LIBGCC_BIN_DIR

$SRC_DIR/configure \
  CC=gcc-12 \
  CXX=g++-12 \
  --disable-multilib \
  --disable-bootstrap \
  --enable-languages=c,c++

make -j$(nproc) --keep-going all-target-libgcc
```

Now build libstdc++.

```bash
LIBSTDCXX_BIN_DIR=$BIN_DIR/libstdc++
mkdir -p $LIBSTDCXX_BIN_DIR
LIBSTDCXX_INSTALL_DIR=$BIN_DIR/install/libstdc++
mkdir -p $LIBSTDCXX_INSTALL_DIR

GTHREAD_INCLUDE_DIR=$LIBGCC_BIN_DIR/x86_64-pc-linux-gnu/libgcc
CXX_AND_C_FLAGS="-I$GTHREAD_INCLUDE_DIR -g -fno-omit-frame-pointer -fsanitize=thread"

cd $LIBSTDCXX_BIN_DIR
$SRC_DIR/libstdc++-v3/configure \
  CC=gcc-12 \
  CXX=g++-12 \
  CFLAGS="$CXX_AND_C_FLAGS" \
  CXXFLAGS="$CXX_AND_C_FLAGS" \
  LDFLAGS="-fsanitize=thread" \
  --prefix=$LIBSTDCXX_INSTALL_DIR \
  --disable-multilib \
  --disable-libstdcxx-pch \
  --enable-libstdcxx-threads=yes \
  --with-default-libstdcxx-abi=new

make -j$(nproc)
make install
```

When running programs you would need to use the sanitized version of libstdc++.

```bash
LD_LIBRARY_PATH="$LIBSTDCXX_INSTALL_DIR/lib" \
  my-program ...
```

#### IREE with TSan support

To enable TSan:

```shell
cmake -DIREE_ENABLE_TSAN=ON ...
```

Several `_tsan` tests like
`iree/tests/e2e/stablehlo_ops/check_llvm-cpu_local-task_tsan_abs.mlir` are
also defined when using this configuration. These tests include ThreadSanitizer
in compiled CPU code as well by using these `iree-compile` flags:

```shell
--iree-llvmcpu-link-embedded=false
--iree-llvmcpu-sanitize=address
```

Note that a IREE runtime built with TSan cannot load a IREE compiled LLVM/CPU
module unless those flags are used, so other tests are excluded using the
`notsan` label.

### MSan (MemorySanitizer)

In theory that should be a simple matter of

```shell
-DIREE_ENABLE_MSAN=ON
```

However, that requires making and using a custom
build of libc++ with MSan as explained in
[this documentation](https://github.com/google/sanitizers/wiki/MemorySanitizerLibcxxHowTo).

As of April 2022, all of IREE's tests succeeded with MSan on Linux/x86-64,
provided that the `vulkan` driver was disabled (due to lack of MSan
instrumentation in the NVIDIA Vulkan driver).

### UBSan (UndefinedBehaviorSanitizer)

Enabling UBSan in the IREE build is a simple matter of setting the
`IREE_ENABLE_UBSAN` CMake option:

```shell
cmake -DIREE_ENABLE_UBSAN=ON ...
```

Note that both ASan and UBSan can be enabled in the same build.

## Symbolizing the reports

### Desktop platforms

On desktop platforms, getting nicely symbolized reports is covered in [this
documentation](https://clang.llvm.org/docs/AddressSanitizer.html#symbolizing-the-reports).
The gist of it is make sure that `llvm-symbolizer` is in your `PATH`, or make
the `ASAN_SYMBOLIZER_PATH` environment variable point to it.

### Android

On Android it's more complicated due to
[this](https://github.com/android/ndk/issues/753) Android NDK issue.
Fortunately, we have a script to perform the symbolization. Copy the raw output
from the sanitizer and feed it into the `stdin` of the
`build_tools/scripts/android_symbolize.sh` script, with the `ANDROID_NDK`
environment variable pointing to the NDK root directory, like this:

```shell
ANDROID_NDK=~/android-ndk-r21d ./build_tools/scripts/android_symbolize.sh < /tmp/asan.txt
```

Where `/tmp/asan.txt` is where you've pasted the raw sanitizer report.

!!! tip

    This script will happily just echo any line that isn't a stack frame.
    That means you can feed it the whole `ASan` report at once, and it will
    output a symbolized version of it. DO NOT run it on a single stack at a
    time! That is unlike the symbolizer tool that's being added in NDK r22, and
    one of the reasons why we prefer to keep our own script. For more details
    see
    [this comment](https://github.com/android/ndk/issues/753#issuecomment-719719789).
