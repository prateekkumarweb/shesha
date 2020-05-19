# Shesha

Shesha is an incomplete MaxSAT solver for GPU.

## Compile and Run with decimation

```sh
nvcc src/* -I include --std=c++11 -O3 -o shesha
```

### Run

```sh
./shesha file.cnf
```

## Old readme

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
$ cmake ..
$ make
$ cd ..
```

### Run

```sh
$ ./build/shesha
```

---

LICENSE: MIT
