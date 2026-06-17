# Cerebrics

## about
Cerebrics aims to be a minimal deeplearning framework written in C with a python interface and optional CUDA acceleration. For now it is only in the first stages of development.

## install

The C library doesn't use any libraries so project can be installed simply by cloning the repo. It you want to contribute code you will also need `clang-format` and `clang-tidy` in order to format / lint the code.

## running

You can use `make list-examples` to discover the example scripts and `make run EXAMPLE=<EXAMPLE NAME>` to run an example program. For instance: `make run EXAMPLE=tensor_basic` to run the most simple example.

# testing
Tests can be ran with `make test`.