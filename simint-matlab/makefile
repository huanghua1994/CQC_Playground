# make sure this includes the architectures you care about
SYS = $(shell uname -s)
ARCH = $(shell uname -m)
ifeq ($(SYS), Linux)
  MEX = mex
  ifeq ($(ARCH), x86_64)
    MEXSUFFIX = mexa64
  endif
  ifeq ($(ARCH), i686)
    MEXSUFFIX = mexglx
  endif
else # CYGWIN
  MEX = mex.bat
  MEXSUFFIX = mexw64
endif

# set sources explicitly if not all your .c files are mex programs
#SOURCES = $(wildcard *.c)
SOURCES = readmol.c calculate_eri.c calculate_kei.c calculate_ovlpi.c calculate_nai.c
TARGETS = $(SOURCES:.c=.$(MEXSUFFIX))

SIMINT_INSTALL_DIR = ./simint-avx

all: $(TARGETS)

%.$(MEXSUFFIX): %.c
	$(MEX) $(SIMINT_INSTALL_DIR)/lib/libsimint.so -O -largeArrayDims CFLAGS='$$CFLAGS -fopenmp -std=c99 -mavx -Wno-unused-result -I$(SIMINT_INSTALL_DIR)/include -fPIC' LDFLAGS='$$LDFLAGS -fopenmp' $<

clean:
	rm -f *.$(MEXSUFFIX)

.PHONY: clean
