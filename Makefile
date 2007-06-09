test-float: float.o integer.o test.o
	g++ ${>} -o $@

.cpp.o: ${ALL_H}
	g++ -g  -c $< -o $@

clean:
	rm -f test-float *.o *~