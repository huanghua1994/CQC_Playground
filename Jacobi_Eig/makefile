CC      = icc
CFLAGS  = -Wall -g -O3 -xHost -qopenmp -mkl -std=c99
LDFLAGS = $(CFLAGS)

OBJS = Jacobi_dsyev.o test_Jacobi_dsyev.o

all: test_Jacobi_dsyev.x

Jacobi_dsyev.o: Jacobi_dsyev.h Jacobi_dsyev.c
	$(CC) $(CFLAGS) -c Jacobi_dsyev.c
    
test_Jacobi_dsyev.o: Jacobi_dsyev.h test_Jacobi_dsyev.c
	$(CC) $(CFLAGS) -c test_Jacobi_dsyev.c

test_Jacobi_dsyev.x: $(OBJS)
	$(CC) $(CFLAGS) -o test_Jacobi_dsyev.x $(OBJS)
    
.PHONY: clean
clean:
	rm $(OBJS) test_Jacobi_dsyev.x