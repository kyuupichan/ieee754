This is a self-contained (separately in both Python and C++) host- and target-independent
arbitrary-precision IEEE-754 binary floating point software implementation.

The code is written for clarity and correctness rather than speed.  The C++ code is
written with a view to its use in the front-end of a cross compiler so that target
arithmetic can be correctly performed on the host.  Nevertheless the code is no slouch and
performance should be reasonable even in Python.

In 2007 I contributed the initial APFloat.cpp implementation to the LLVM project; this was
a hastily-converted C++ translation of C code I had written as part of an unreleased C
compiler front end.  I have been disapointed in the direction the code evolved as a mostly
free-standing part of LLVM.  So I have been motivated to resurrect my original code, add
missing features, and bring it up to date with the IEEE-754 standards released in 2008 and
2019 - the original code was written in 2007 with drafts of the 2008 standard which was
still under active discussion.

In 2007 I hadn't learnt Python, which I now enjoy and am proficient in.  Accordingly my
approach to this project has been to apply the few bug fixes to the C++ code that were
found whilst being part of LLVM, and then write a Python implementation which I can more
easily refactor and test.  This has already led to algorithmic improvements and
simplifications.  I have written a comprehensive testsuite mostly based on
language-independent testcase files.  I then plan to rewrite the Python code back into
C++, reusing those test cases, and in this way the C++ version should be very quick to
re-implement.  I have been unable to find any serious library in Python attempting
arbitrary precision IEEE arithmetic, so if nothing else this fills a gap in the market.

Proper support for transcendental operations in arbitrary precision is a research topic
and so I consider it out-of-scope, at least initially.  I will support all other
operations in the standard including the augmented functions added in the 2019 revision.

The library currently implements the following:

  - correct rounding of all implemented operations in any rounding mode
  - all IEEE basic and extended binary formats
  - user-definable binary formats with arbitrary exponent range and precision
  - support for x87 extended precision, including where resuls are rounded to double
    or single precision, for all operations
  - operations, where relevant, take a floating point context, which defaults to a thread-local
    one if none is specified.  The context specifies:
     - the rounding mode (seven are supported)
     - whether tininess is detected before or after rounding
     - complete control over how floating point exceptions are handled
     - floating point exception flags
  - full support for quiet and signalling NaNs and their payloads
  - where meaningful, operations to permit operands and the result to have any format,
    which can all be different.  So, for example, you can do a fused-multiply-add
    operation, multiplying an IEEEhalf by an IEEEdouble, adding IEEE quad, and place the
    correctly-rounded result for the operation as a whole in an IEEE single
  - the following operations on all formats:
     - negate, abs, copy_sign, classify_number, is_negative, is_zero, is_infinite, is_NaN,
       is_normal, is_finite, is_subnormal, is_signalling, is_canonical, radix
     - add, subtract
     - multiply
     - divide
     - fused-multiply-add
     - comparison operations
     - conversions between floating point types
     - conversion from any integer format to any floating point format
     - conversion from any floating point format to any integer format
     - correctly-rounded conversions from strings with decimal and hexadecimal significands
     - correctly-rounded conversions to strings with decimal and hexadecimal significands,
       with full control of output precision and format
     - rounding to "integral" (i.e. to an integer in the same floating point format)
     - total_order, total_order_mag
     - next_up, next_down
     - remainder, fmod (as per C99), and Python's divmod, mod, and floordiv operations
     - scaleB, logB
     - reading from and writing to arbitrary binary interchange formats
     - max, max_num, min, min_num, max_mag, min_mag, max_mag_num, min_mag_num
     - payload, set_payload, set_payload_signalling
   - Python support:
     - the numbers can be used with standard operators (such as +, *= and /), mixing with
       other Python types where meaningful, much like the Decimal type does
     - a number hashes correctly, so it has the same hash as other Python types with the
       same arithmetic value.

  - I intend to support the following but they are not yet imlpemented:
     * augmented addition, subtraction and multiplication
     * square root and reciprocal square root (these will be a bit tricky)
