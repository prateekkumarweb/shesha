# Shesha

Shesha is an incomplete MaxSAT solver for GPU.

## Compile and Run

### Build

To compile glucose

```sh
$ cd glucose-syrup-4.1/simp
$ make libr
$ cd ../..
```

To compile shesha

```sh
$ mkdir build
$ cd build
$ cmake -DCMAKE_BUILD_TYPE=Release ..
$ make
$ cd ..
```

### Run

```sh
$ ./build/shesha
```

---

LICENSE: MIT
