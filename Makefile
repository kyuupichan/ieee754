CXXFLAGS=-O2

all: test-special test-normal test-print test-speed test-libc

test-special: APFloat.o APInt.o test-special.o APFloat.h
	g++ ${>} -o $@

test-normal: APFloat.o APInt.o test-normal.o APFloat.h
	g++ ${>} -o $@

test-print: APFloat.o APInt.o test-print.o APFloat.h
	g++ ${>} -o $@

test-speed: APFloat.o APInt.o test-speed.o APFloat.h
	g++ ${>} -o $@

test-libc: test-libc.o
	g++ ${>} -o $@

.cpp.o: APFloat.h
	g++ ${CXXFLAGS} -Wall -Werror -long-long -g  -c $< -o $@

clean:
	rm -f test-special test-normal test-print test-speed *.o *~ *.core *.orig

git:	clean
	git commit -a
	make all
