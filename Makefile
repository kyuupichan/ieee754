all: test-special test-normal test-print

test-special: APFloat.o APInt.o test-special.o APFloat.h
	g++ ${>} -o $@

test-normal: APFloat.o APInt.o test-normal.o APFloat.h
	g++ ${>} -o $@

test-print: APFloat.o APInt.o test-print.o APFloat.h
	g++ ${>} -o $@

.cpp.o: APFloat.h
	g++ -Wall -Werror -long-long -g  -c $< -o $@

clean:
	rm -f test-special test-normal test-print *.o *~ *.core *.orig

git:	clean
	git commit -a
	make all
