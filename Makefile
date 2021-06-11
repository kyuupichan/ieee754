CXXFLAGS=

all: test-special test-normal test-print test-speed

test-special: APFloat.o APInt.o test-special.o
	g++ ${>} -o $@

test-normal: APFloat.o APInt.o test-normal.o
	g++ ${>} -o $@

test-print: APFloat.o APInt.o test-print.o
	g++ ${>} -o $@

test-speed: APFloat.o APInt.o test-speed.o
	g++ ${>} -o $@

.cpp.o: APFloat.h
	g++ ${CXXFLAGS} -Wall -Werror -g  -c $< -o $@

clean:
	rm -f test-special test-normal test-print test-speed *.o *~ *.core *.orig

git:	clean
	git commit -a
	make all
