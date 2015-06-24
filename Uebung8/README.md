# Uebung 8

## Run programm on TurboTUC (Xeon Phi)

* ssh turbotuc-6

### load module for compiler

* module load intel/15.0.2

### compile with icc

* icc helloworld.c -o helloworld --mmic

### run on mic coprocessor 

* either with micnativeloadex
 * SINK_LD_LIBRARY_PATH=/opt/intel/composerxe/lib/mic/ micnativeloadex helloworld
* or without, directly on the coprocessor
 * ssh mic<id>
 * export LD_LIBRARY_PATH=/opt/intel/lib/mic (gives you some useful libraries)
