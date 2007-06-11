all: test-special test-normal

test-special: float.o integer.o test-special.o
	g++ ${>} -o $@

test-normal: float.o integer.o test-normal.o
	g++ ${>} -o $@

.cpp.o: ${ALL_H}
	g++ -g  -c $< -o $@

clean:
	rm -f test-special test-normal *.o *~ *.core