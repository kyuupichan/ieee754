:mod:`ieee754` - Binary floating point arithmetic
=================================================

.. module:: ieee754
   :synopsis: Arbitrary-precision binary floating point arithmetic
.. moduleauthor:: Neil Booth <kyuupichan@gmail.com>

**Source code:** `ieee754.py <https://github.com/kyuupichan/ieee754/blob/master/ieee754/>`_.

--------------

The :mod:`ieee754` module provides support for arbitrary-precision binary floating point
arithmetic.

Its design revolves around four concepts: binary numbers, binary formats, the context for
arithmetic, and signals.

A binary number is immutable.  It has a format, sign, exponent and significand, and can
hold special values such as :const:`infinity`, :const:`NaN`, :const:`sNaN`.  It also
distinguishes :const:`-0` from :const:`+0`.

A binary format controls the exponent range and precision that the result of a calculation
is delivered to.

Arithmetic is done under the control of an environment called a `context`.  It specifies
rounding rules, when tininess is detected, holds flags indicating what arithmetic
exceptions have occurred, and offers fine-grained control over signal handling.  Rounding
options are :const:`ROUND_CEILING` (towards positive infinity), :const:`ROUND_FLOOR`
(towards negative infinity), :const:`ROUND_DOWN` (towards zero), :const:`ROUND_UP` (away
from zero), :const:`ROUND_HALF_EVEN` (to nearest, ties towards even),
:const:`ROUND_HALF_DOWN` (to nearest, ties towards zero), :const:`ROUND_HALF_UP` (to
nearest, ties away from zero).

Signals are exceptional conditions that can arise during the course of a computation.
Depending on the needs of the applicaiton, signals may be handled in various ways
including ignoring them, noting them with flags, recording their details, substitute a
result, or raising an exception.  The signals are those specified by the **IEEE-754**
standard, namely :const:`Invalid`, :const:`DivisionByZero`, :const:`Inexact`,
:const:`Overflow`, and :const:`Underflow`.

Each of the five major signals has its own flag which normally is set in the controlling
`context` object when it occurs.  Flags are sticky, so the user needs to reset them when
wanting to detect them in a fresh calculation.  Many signals have subcategories, organised
as an exception hierarchy, and the context controls what happens when each is detected.
The user can specify how each exception or sub-exception in the hierarchy is handled.  If
nothing is specified for the specific exception that occurred, handling is delegated to
the parent exception, recursively.


Quick-start Tutorial
====================

To be written.


BinaryFormat objects
====================

A binary format describes the exponent range and precision of a floating point number.
Many operations on floating point numbers are defined as methods on the `BinaryFormat`
class, with the instance being the desired format of the floating point result.

Several binary formats are predefined, including the four specified in the IEEE-754
standard: :const:`IEEEhalf`, :const:`IEEEsingle`, :const:`IEEEdouble` and
:const:`IEEEquad`.  The user can also create his own formats with control of the minimum
and maximum exponents of normalized numbers, and the precision in bits.


Binary objects
==============

A Binary object represents a binary floating-point value.  They should not be constructed
directly, but through helper class methods on the :class:`BinaryFormat` class.


Context objects
===============

Arithmetic is done under the control of an environment called a `context`.  It specifies
rounding rules, when tininess is detected, holds flags indicating what arithmetic
exceptions have occurred, and offers fine-grained control over signal handling.

Each thread has its own current context which can be accessed or changed using the
:func:`get_context()` and :func:`set_context()` functions.


TextFormat objects
==================

.. class:: TextFormat

  Binary values can be converted to hexadecimal and decimal text form with the functions
  :meth:`to_string` and :meth:`to_hex_string`.  Both functions accept a
  :class:`TextFormat` object which controls the precise form of the strings produced.  If
  no output format is specified the methods use :data:`DefaultHexFormat` or
  :data:`DefaultDecFormat` respectively.

  A :class:`TextFormat` object has the following attributes and defaults, which you can
  pass as keyword arguments to the constructor.

  .. attribute:: exp_digits

    The minimum number of digits to output in the exponent of a finite number.  Defaults
    to :const:`1`.

    For decimal output :const:`0` suppresses the exponent by adding leading or trailing
    zeroes to the significand as needed (as for the :func:`printf` **f** format specifier
    in the **C** programming language).  If negative, apply the rule for the
    :func:`printf` **g** format specifier to decide whether to display an exponent or not,
    in which case the minimum number of digits in the exponent is the absolute value.

    Hexadecimal output always has an exponent so the absolute value is used with a minimum
    of :const:`1`.

  .. attribute:: force_exp_sign

     Defaults to :const:`True`.  If :const:`True` output positive exponents with a leading '+'.

  .. attribute:: force_leading_sign

     Defaults to :const:`False`.  If :const:`True` output values with a clear sign with a
     leading '+'.

  .. attribute:: force_point

     Defaults to :const:`False`.  If :const:`True` the output for a floating point number
     includes a floating point and a single zero even when none is needed.  For example
     '5', '1e2' and '0x1p2' would instead be output as '5.0', '1.0e2' and '0x1.0p2'
     respectively.

  .. attribute:: rstrip_zeroes

     Defaults to :const:`False`.  If :const:`True` suppress trailing insignificant zeroes
     in significands.  This does not affect the :const:`force_point` attribute which takes
     precedence.

  .. attribute:: upper_case

     Defaults to :const:`False`.  If :const:`True` output the exponent character ('p' or
     'e') in upper case and output :const:`NaN` hexadecimal payloads (when
     :attr:`nan_payload` is 'X') in upper case.  For hexadecimal strings output the hex
     indicator '0x' and hexadecimal digits in upper case.

     This attribute does not affect :attr:`inf`, :attr:`qNaN` and :attr:`sNaN` which
     appear in the output string verbatim.

  .. attribute:: inf

     The string to output for infinities.  The default is 'Infinity'.

  .. attribute:: qNaN

     The string to output for a quiet :const:`NaN`.  The default is 'NaN'.

  .. attribute:: sNaN

     The string to output for a signalling :const:`NaN`.  The default is 'sNaN'.

  .. attribute:: nan_payload

     This attribute controls the display of :const:`NaN` payloads and defaults to 'X'.
     'N' suppresses them, 'X' outputs them in hexadecimal, and 'D' outputs them in
     decimal.  Examples of all three formats for a :const:`NaN` payload of :const:`255`
     are 'NaN', 'NaN255' and 'NaN0xff' respectively.


.. data:: DefaultHexFormat

     This instance controls hexadecimal output when no object is explicitly passed to
     :meth:`Binary.to_string`.  It is intended to match the output of Python's
     :func:`float.hex` method for finite numbers, and to match Python's :class:`Decimal`
     string conversion for non-finite numbers::

       DefaultHexFormat = TextFormat(force_point=True, nan_payload='N')


.. data:: DefaultDecFormat

     This instance controls decimal output when no object is explicitly passed to
     :meth:`Binary.to_decimal_string`.  It is intended to match the way Python displays
     :class:`float` values::

       DefaultDecFormat = TextFormat(exp_digits=-2, force_point=True,
                                     inf='inf', qNaN='nan', sNaN='snan', nan_payload='N')


.. data:: Dec_G_Format

     This instance is intended to match the output of Python's **g** format specifier when
     the specified precisions are the same::

       DefaultDecFormat = TextFormat(exp_digits=-2, rstrip_zeroes=True,
                                     inf='inf', qNaN='nan', sNaN='snan', nan_payload='N')
