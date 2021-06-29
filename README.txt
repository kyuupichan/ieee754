This is a self-contained (separately in both Python and C++) host- and target-independent
arbitrary-precision binary floating point software implementation.

The code is written for clarity and correctness rather than speed; the C++ code in
particular with a view to its use in the front-end of a cross compiler so that target
arithmetic can be correctly performed on the host.  Nevertheless the code is no slouch and
performance should be reasonable even in Python.  For its intended use in compiler front
ends, performance should not be an issue.

In 2007 I contributed the initial APFloat.cpp implementation to the LLVM project; this was
a hastily-converted C++ translation of C code I had written as part of an unreleased C
compiler front end.  I did not contribute the tests I had as these were hacky and not fit
for purpose.  Writing this README 14 years since the initial contribution, and reviewing
the LLVM commits in that time, I believe just six bugs have been found in the code I
contributed.  As a broad and ambitious project, I condsider this a respectable record
particularly given the notoriously intricate nature of floating point arithmetic and the
rushed rewrite from C.  Many dozens of bugs have been fixed in APFloat.cpp over the years
in LLVM, but almost all were introduced in code modified or added by other contributors.

I have long been disappointed in the direction the code evolved as a mostly free-standing
part of LLVM.  Developers added features and functionality but didn't seem to fully
appreciate the existing code and its features.  For example, my initial code only
implemented quiet NaNs, and did not handle NaN payloads of any kind.  Apple added support
for these, but in what can only be described as a very awkward way.  Later, various new
operations were added, but the implementations were generally inefficient or reinvented
existing primitives, and invariably had several bugs in their initial commit.

As a result I have been motivated to resurrect my original code, add missing features, and
bring it up to date with the IEEE-754 standards released in 2008 and 2019.  I wrote the
original code in 2007 just with drafts of the 2008 standard which was still under active
discussion.  I now have ideas on how to simplify and improve the performance of the
original code.  I intend to support all of the core non-transcendental operations in the
standard including the augmented functions added in the 2019 revision (however the square
root operation in arbitrary precision is difficult and uncertain).

In 2007 I hadn't learnt Python, which I now enjoy and am proficient in.  Accordingly my
approach to this project is to first apply bug fixes to the C++ code that were found
whilst being part of LLVM, and then write a Python implementation which I can more easily
refactor and test.  This has already led to several algorithmic improvements and
simplifications.  I am writing a comprehensive testsuite mostly based on
language-independent testcase files.  I then plan to rewrite the Python code in C++,
reusing those test cases, and in this way the C++ version should only take a few days to
implement.  As a motivating bonus I have been unable to find any serious IEEE library in
Python attempting a similar goal, so if nothing else this fills a gap in the market.

Once complete this library, in both Python and C++, should implement the following (*
indicates new features not present in the original C++ code):

  - all relevant operations to take a floating point context and be correctly
    rounded.  The context specifies:
     - the rounding mode (all five IEEE rounding modes are supported)
     * whether tininess is detected before or after rounding
     * whether underflow is flagged if tiny, or only if tiny and inexact
  - reporting of status flags as per IEEE-754 as modified by the context
  - quiet and signalling NaNs and their payloads
  - arbitrary floating point formats, i.e. arbitrary exponent widths and significand
    precisions, for all operations
  - all the following operations:
     - copy, negate, abs, is_negative, is_zero, is_infinite, is_NaN
     - add, subtract
     - multiply
     - divide
     - fused-multiply-add
     - comparison operations
     - conversions between floating point types
     - conversion from any integer format to any floating point format
     - conversion from any floating point format to any integer format
     - correctly-rounded conversions from strings with decimal significands
     - correctly-rounded conversions from strings with hexadecimal significands
     - conversions to strings with hexadecimal significands with fine control on output
       precision and format
     * conversions to string with decimal significands with fine control on output precision
       and format
     * conversion from floating point to "integral" (i.e. rounding with destination format
       matching source format)
     * class
     * is_normal, is_finite, is_subnormal, is_signalling, is_canonical, radix
     * totalOrder, totalOrderMag
     * copySign
     * nextUp, nextDown
     * remainder and fmod (as per C99)
     * scaleB, logB
     * reading from and writing to arbitrary binary interchange formats
     * maximum, maximumNumber, minimum, minimumNumber, maximumMagitudeNumber,
       minimumMagnitudeNumber
     * getPayload, setPayload, setPayloadSignaling
     * proper support for x87 extended operations with double and single precision results
     * augmented addition, subtraction and multiplication
     * square root and reciprocal square root (if I have time; these will be tricky)
