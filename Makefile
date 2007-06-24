all: test-special test-normal

test-special: APFloat.o APInt.o test-special.o APFloat.h
	g++ ${>} -o $@

test-normal: APFloat.o APInt.o test-normal.o APFloat.h
	g++ ${>} -o $@

.cpp.o: APFloat.h
	g++ -g  -c $< -o $@

clean:
	rm -f test-special test-normal *.o *~ *.core *.orig

git:	clean
	git commit -a
	make all
