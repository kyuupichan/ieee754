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
hold special values such as :const:`Infinity`, :const:`NaN`, :const:`sNaN`.  It also
distinguishes :const:`-0` from :const:`+0`.

A binary format controls the exponent range and precision that the result of a calculation
is delivered to.

Arithmetic is done under the control of an environment called a `context`.  It specifies
rounding rules, when tininess is detected, holds flags indicating what arithmetic
exceptions have occurred, and offers fine-grained control over signal handling.  Rounding
options are :const:`ROUND_CEILING` (towards :const:`+Infinity`), :const:`ROUND_FLOOR`
(towards :const:`-Infinity`), :const:`ROUND_DOWN` (towards zero), :const:`ROUND_UP` (away
from zero), :const:`ROUND_HALF_EVEN` (to nearest, ties towards even),
:const:`ROUND_HALF_DOWN` (to nearest, ties towards zero), :const:`ROUND_HALF_UP` (to
nearest, ties away from zero).

`Signals`_ are exceptional conditions that can arise during the course of a computation.
Depending on the needs of the applicaiton, signals may be handled in various ways
including ignoring them, noting them with flags, recording their details, substitute a
result, or raising an exception.  The signals are those specified by the **IEEE-754**
standard, namely :exc:`Invalid`, :exc:`DivisionByZero`, :exc:`Inexact`, :exc:`Overflow`,
and :exc:`Underflow`.

Each of the five major signals has its own flag which normally is set in the controlling
`context` object when it occurs.  Flags are sticky, so the user needs to reset them when
wanting to detect them in a fresh calculation.  Many signals have subcategories, organised
as an exception hierarchy, and the context controls what happens when each is detected.
The user can specify how each exception or sub-exception in the hierarchy is handled.  If
nothing is specified for the specific exception that occurred, handling is delegated to
the parent exception, recursively.

Several of the classes described in this documentation have attributes and methods that
are not documented.  Consider these as implementation details that are subject to change
or removal.


Quick-start Tutorial
====================

To be written.


.. _interchange format:

BinaryFormat objects
====================

A binary format is specified by its exponent range and precision.  Some binary formats,
called :dfn:`interchange formats`, have a well-defined encoding as byte strings, which
enables the exchange of floating point data between implementations with a common
understanding of byte-order.

Several formats are predefined, including the four specified in the IEEE-754 standard.
You can also create your own binary formats with the constructors
:meth:`BinaryFormat.from_triple`, :meth:`BinaryFormat.from_pair`,
:meth:`BinaryFormat.from_precision` and :meth:`BinaryFormat.from_IEEE`.


.. data:: IEEEhalf

   The IEEE-754 half-precision format is 16-bit binary `interchange format`_ with a
   precision of 11 bits and an exponent width of 5 bits.


.. data:: IEEEsingle

   The IEEE-754 single-precision format is a 32-bit binary `interchange format`_ with a
   precision of 24 bits and an exponent width of 8 bits.


.. data:: IEEEdouble

   The IEEE-754 double-precision format is a 64-bit binary `interchange format`_ with a
   precision of 53 bits and an exponent width of 11 bits.


.. data:: IEEEquad

   The IEEE-754 quadruple-precision format is a 128-bit binary `interchange format`_ with
   a precision of 113 bits and an exponent width of 15 bits.


.. data:: x87extended

   This is the full-precision format used by Intel x87 compatible CPUs.  It is an 80-bit
   binary `interchange format`_ with a precision of 64 bits and an exponent width of 15
   bits.

   Since the format's integer bit is explicit, it admits extra non-canonical encodings
   (which Intel termed pseudo-NaNs, pseudo-infinities, pseudo-denormals and unnormals)
   beyond those specified in IEEE-754.  The :meth:`Binary.pack` method never returns such
   encodings and :meth:`BinaryFormat.unpack` automatically canonicalizes them.


.. data:: x87double

   This simulates the operation of x87 compatible CPUs when in round-to-double-precision
   mode.  It has a precision of 53 bits and an exponent width of 15 bits, and is not a
   binary interchange format.


.. data:: x87single

   This simulates the operation of x87 compatible CPUs when in round-to-single-precision
   mode.  It has a precision of 24 bits and an exponent width of 15 bits, and is not a
   binary interchange format.


.. class:: BinaryFormat

  Represents a binary format.  Binary formats are immutable.

  .. attribute:: precision

     The precision in bits.

  .. attribute:: e_max

     The maximum exponent of normalized numbers.

  .. attribute:: e_min

     The minimum exponent of normalized numbers.

  .. attribute:: fmt_width

     For a binary `interchange format`_, the format width in bits, otherwise :const:`0`.

  .. attribute:: logb_zero
  .. attribute:: logb_nan
  .. attribute:: logb_inf

     These values are returned by the :meth:`Binary.logb_integral` operation on zero,
     :const:`NaN` and :const:`infinity` values in this format, respectively.

  .. classmethod:: from_triple(precision, e_max, e_min)

     This constructor directly initializes the three defining attributes of a binary
     format.

     *precision* must be at least :const:`3`, *e_max* at least :const:`2` and *e_min*
     negative.

  .. classmethod:: from_pair(precision, e_width)

      With this constructor you specify the *precision* and *e_width* - the width of the
      exponent field in bits.  :attr:`e_max` is set to ``2^(e_width - 1) - 1`` and
      :attr:`e_min` to ``1 - e_max``.

  .. classmethod:: from_precision(precision)

     Only *precision* is specified.  A reasonable exponent range for that precision is
     chosen.  The exponent range chosen may change in future versions.

  .. classmethod:: from_IEEE(fmt_width)

     The IEEE standard defines binary formats for specific format widths.  This returns
     the format for *fmt_width*, which must be 16, 32, 64 or a multiple of 32 that is at
     least 128.

  A binary format offers several methods to conveniently and efficiently create common
  values in that format.  These methods are quiet.

  .. method:: make_zero(sign)

     Return a zero of the specified sign.

  .. method:: make_one(sign)

     Return a value of one with the specified sign.

  .. method:: make_infinity(sign)

     Return an :const:`Infinity` with the specified sign.

  .. method:: make_largest_finite(sign)

     Return the largest finite value with the specified sign.

  .. method:: make_smallest_subnormal(sign)

     Return the smallest subnormal number with the specified sign.

  .. method:: make_smallest_normal(sign)

     Return the smallest normal number with the specified sign.

  .. method:: make_nan(sign, is_signalling, payload)

     Return a :const:`NaN` with the specified sign and payload.  *is_signalling*
     indicates if the :const:`NaN` is signalling or quiet.

     *payload* must be a non-negative integer.  It is truncated to fit the format if it is
     too large.  Signalling NaNs cannot represent payloads of 0 so 1 is used instead.

  You can convert various datatypes to a binary format via the following constructors.
  Most take a *context* which determines the rounding mode, and they signal
  :exc:`Overflow`, :exc:`Underflow` and :exc:`Inexact` as appropriate.

  .. method:: from_value(value, context=None)

     Convert from an arbitrary value.  This function passes *value* on to :meth:`convert`,
     :meth:`from_string`, :meth:`from_int`, :meth:`from_float`, :meth:`from_decimal`,
     :meth:`from_fraction` or :meth:`unpack_value` depending on its type.

     If you already know the type of *value* it is more efficient to call the specific
     method directly.

  .. method:: convert(value, context=None)

     Convert from a :class:`Binary` object.

  .. method:: from_string(string, context=None)

     Convert from a Python string.  Strings representing floating point values encoded in
     decimal or hexadecimal form, as per C99, are accepted.  See `String Syntax`_ for a
     detailed specification.

  .. method:: from_int(value, context=None)

     Convert from a Python :class:`int` object.

  .. method:: from_float(value, context=None)

     Convert from a Python :class:`float` object.

  .. method:: from_decimal(value, context=None)

     Convert from a :class:`Decimal` object of the :mod:`decimal` module.

  .. method:: from_fraction(value, context=None)

     Convert from a :class:`Fraction` object of the :mod:`fractions` module.

  .. method:: unpack_value(raw, endianness=None, context=None)

     Convert from a packed binary encoding *raw* of a value of this `interchange format`_.
     *endiannness* is the byte order of the encoding, valid values are 'little', 'big' and
     :const:`None` which will use the native endianness of the host machine.  sNaNs are
     preserved and Conversion is necessarily exact so only :exc:`UnderflowExact` can be
     signalled.

  The following operations take operands of arbitrary binary formats, and deliver a result
  in this format.  The *context* parameter controls the rounding and exception handling,
  as described by the documentation of :class:`Context`.  Each operation signals at most
  one exception.

  .. method:: add(lhs, rhs, context=None)

     Return the sum of *lhs* and *rhs*.

  .. method:: subtract(lhs, rhs, context=None)

     Return the result of subtracting *rhs* from *lhs*.

  .. method:: multiply(lhs, rhs, context=None)

     Return the product of *lhs* and *rhs*.

  .. method:: divide(lhs, rhs, context=None)

     Return the result of dividing *lhs* by *rhs*.

  .. method:: fma(lhs, rhs, addend, context=None)

     Return the result of multiplying *lhs* and *rhs* and then adding *addend*, with a
     single rounding operation at the end.  This is called a :dfn:`fused-multiply-add`
     operation.

  .. method:: sqrt(value, context=None)

     Return the square root of *value*.

  The following two methods convert to and from binary encodings and are only applicable
  if the format is an `interchange format`_.

  .. method:: pack(sign, exponent, significand, endianness=None)

     Encode the three parts of a floating point number to `bytes`.  *endiannness* is the
     byte order of the encoding, valid values are 'little', 'big' and :const:`None` which
     will use the native endianness of the host machine.  *exponent* is the biased
     exponent in the IEEE sense, i.e., it is zero for zeroes and subnormals and e_max *
     2 + 1 for NaNs and infinites.  *significand* must not include the integer bit.

  .. method:: unpack(raw, endianness=None)

     *raw* is a a binary encoding of a value; decode it and return a ``(sign, exponent,
     significand)`` tuple.

     *endiannness* is the byte order of the encoding, valid values are 'little', 'big' and
     :const:`None` which will use the native endianness of the host machine.  *exponent*
     is the biased exponent in the IEEE sense, i.e., it is zero for zeroes and subnormals
     and e_max * 2 + 1 for NaNs and infinites.  *significand* does not include the integer
     bit.


Binary objects
==============

A Binary object represents a binary floating-point value.  They should not be constructed
directly, but through helper methods or class methods on the :class:`BinaryFormat` class.
Once constructed, :class:`Binary` objects are immutable.

Binary objects share many properties with other built-in numeric types such as `float` and
`int`.  The usual mathematical operations and special methods apply; the thread's default
context is used as the context.  Likewise Binary objects can be copied, pickled, printed,
used as dictionary keys, used as set elements, compared, sorted and coerced to another
type (such as `float` and `int`).  Conversion to `bool` is a quiet operation.

Binary objects behave the same as `float` object for the ``%`` and ``//`` operators::

  >>> -7.0 % 4.0
  1.0
  >>> -7.0 // 4.0
  -2.0
  >>> IEEEhalf.from_int(-7) % IEEEhalf.from_int(4)
  0x1.000p+0
  >>> IEEEhalf.from_int(-7) // IEEEhalf.from_int(4)
  -0x1.000p+1

Binary objects cannot generally be combined with instances of `decimal.Decimal` or
`fractions.Fraction`, but they can be combined with instances of type `int` and `float`.
However it is possible to use Python's comparison operators to compare a :class:`Binary`
instance with any other numeric type.

Unless noted otherwise :const:`NaN` operands are propagated as descibed in the section
`NaN propagation`_.

Quiet Operations
----------------

The following operations are *quiet* - they do not raise signals and no context affects or
is affected by them.

.. method:: number_class()

   Return a string describing the *class* of the operand, which is one of the following
   ten strings:

      * ``"-Infinity"`` when the operand is negative infinity.
      * ``"-Normal"`` when the operand is a negative normal number.
      * ``"-Subnormal"`` when the operand is negative and subnormal.
      * ``"-Zero"`` when the operand is negative zero.
      * ``"+Zero"`` when the operand is positive zero.
      * ``"+Subnormal"`` when the operand is positive and subnormal.
      * ``"+Normal"`` when the operand is a positive normal number.
      * ``"+Infinity"`` when the operand is positive infinity.
      * ``"NaN"`` when the operand is a quiet NaN (Not a Number).
      * ``"sNaN"`` when the operand is a signalling NaN.

.. method:: is_negative()

   Return :const:`True` if the sign bit is set (including for :const:`NaN` values).

.. method:: is_zero()

   Return :const:`True` if the value is a zero of either sign.

.. method:: is_subnormal()

   Return :const:`True` if the value is subnormal.

.. method:: is_normal()

   Return :const:`True` if the value is finite, non-zero and not subnormal.

.. method:: is_finite()

   Return :const:`True` if the value is finite.  A finite number is precisely one of zero,
   subnormal or normal.

.. method:: is_finite_non_zero()

   Return :const:`True` if the value is subnormal or normal.

.. method:: is_infinite()

   Return :const:`True` if the value is an :const:`Infinity` of either sign.

.. method:: is_nan()

   Return :const:`True` if the value is a quiet or signalling :const:`NaN`.

.. method:: is_qnan()

   Return :const:`True` if the value is a quiet :const:`NaN`.

.. method:: is_snan()

   Return :const:`True` if the value is a signalling :const:`NaN`.

.. method:: is_canonical()

   Return :const:`True`.

.. method:: radix()

   Return :const:`2`.

.. method:: set_sign(sign)

   Return this value with the given sign (including for :const:`NaN` values).

.. method:: abs_quiet()

   Return this value with sign :const:`False` (including for :const:`NaN` values).

.. method:: pack(endianness=None)

   Encode the three parts of the floating point value as `bytes`.  *endiannness* is the
   byte order of the encoding, valid values are 'little', 'big' and :const:`None` which
   will use the native endianness of the host machine.

.. method:: nan_payload()

   Return the payload of a :const:`NaN` as a Python `int`.  If the value is not a
   :const:`NaN` raise a :exc:`RuntimeError`.


Context objects
===============

Arithmetic is done under the control of an environment called a `context`.  It specifies
rounding rules, when tininess is detected, holds flags indicating what arithmetic
exceptions have occurred, and offers fine-grained control over signal handling.

Each thread has its own context which can be accessed or changed using the
:func:`get_context()` and :func:`set_context()` functions.

.. function:: get_context()

   Return the context for the current thread.

.. function:: set_context(context)

   Set the context for the current thread to *context*.  A copy is not made, a reference
   is held.

You can also use the `with` statement and the :func:`local_context` function to
temporarily replace the active context for a block of code.

.. function:: local_context(context=None)

   Return a context manager that will replace the active thread's context with a copy of
   *context* on entry to the `with` statement, and restore the previous context on exit.
   If *context* is :const:`None`, a copy of the current context is used instead.

   For example, the following code inherits the ambient context, sets the rounding mode to
   :const:`ROUND_CEILING`, performs a calculation, and then automatically restores the
   previous context::

      from ieee754 import local_context

      with local_context() as ctx:
          ctx.rounding = ROUND_CEILING
          result = some_calculation()

   On exiting the with block above, the original context will be effective with its
   original rounding mode, and its flags and other attributes will be unaffected by the
   arithmetic done within the block.

New contexts can be created with the :class:`Context` constructor described below.  In
addition the :mod:`ieee754` module provides a predefined context.

.. data:: DefaultContext

   This context is used as the default context in effect when a new thread is started.
   The module initializes it as follows::

        DefaultContext = Context()
        DefaultContext.set_handler((Invalid, DivisionByZero, Overflow), HandlerKind.RAISE)

   In words, rounding is :const:`ROUND_HALF_EVEN`, tininess is detected after rounding,
   and no flags are set.  :exc:`Underflow` and :exc:`Inexact` receive default handling,
   and :exc:`Invalid`, :exc:`DivisionByZero` and :exc:`Overflow` raise Python exceptions.

   Except perhaps to modify it at program startup, it is preferable to not use
   :data:`DefaultContext` at all.

.. class:: Context(*, rounding=ROUND_HALF_EVEN, flags=0, tininess_after=True)

    Create a new :class:`Context` object and initialize the three attibutes.

    .. attribute:: rounding

      The rounding mode.  One of the constants listed in the section `Rounding Modes`_.

    .. attribute:: flags

      Which flags have been raised; see `Context Flags`_.  Flags can be set or cleared
      directly at any time.

    .. attribute:: tininess_after

      If :const:`True` tininess is detected after rounding, otherwise before rounding.

    .. attribute:: exceptions

       A list of exceptions recorded as specified by the
       :attr:`HandlerKind.RECORD_EXCEPTION` alternative exception handling attribute.  The
       exceptions are ordered, earliest first.  This list is never cleared by the library
       so the user should clear it when done with the exceptions.

    .. method:: copy()

       Return a deep copy of the context.

    .. method:: set_handler(exc_classes, kind, handler=None)

       Specify alternate exception handling for one or more exception classes.

       *exc_classes* is an exception class, or an iterable of one or more exception
       classes.  Each exception class must be a subclass of :exc:`IEEEError`.  If *kind*
       is :attr:`ABRUPT_UNDERFLOW`, each exception class must be a subclass of
       :exc:`Underflow`.

       *kind* is one of the :class:`HandlerKind` constants.

       *handler* is the handler to call.  Some kinds require a handler to be specified,
       the rest require no handler be given.

       See `Alternate Exception Handling`_ for more information.

    .. method:: handler(exc_class)

       Return how an exception is handled, as ``(kind, handler)`` pair.  *kind* and
       *handler* are as for :meth:`set_handler`.


Rounding Modes
--------------

When the infinitely precise result of an operation cannot be represented in the
destination format, the rounding mode of the operation's *context* determines the result
it will deliver to the default exception handler whilst raising an :exc:`Overflow`,
:exc:`Underflow` or :exc:`Inexact` signal as appropriate.  Inexact results always have the
same sign as the infinitely precise result.

Additionally, the rounding mode affects the sign of an exact-zero sum, and the threshold
beyond which an operation signals :exc:`Overflow`, and the :exc:`Underflow` threshold when
tininess is detected after rounding.


.. data:: ROUND_CEILING

   Round towards :const:`Infinity`.

.. data:: ROUND_FLOOR

   Round towards :const:`-Infinity`.

.. data:: ROUND_UP

   Round away from zero.

.. data:: ROUND_DOWN

   Round towards zero.

.. data:: ROUND_HALF_EVEN

   Round to nearest with ties going to the value whose significand has a *least
   significant bit* of zero.

.. data:: ROUND_HALF_DOWN

   Round to nearest with ties going towards zero.

.. data:: ROUND_HALF_UP

   Round to nearest with ties going away from zero.


Context Flags
-------------

The :class:`Flags` class is derived from :class:`IntFlag` so the flags form a bitmask,
with one flag for each of the IEEE-754 signals.  Each flag (with the possible exception of
:attr:`Flags.UNDERFLOW`; see :exc:`UnderflowExact`) is raised when its associated signal
is handled by the default exception handler.  Flags are never cleared once raised, so the
user must clear them when appropriate by directly updating the :attr:`Context.flags`
attribute of the context object.

.. class:: Flags

    .. attribute:: INVALID

       Corresponds to an :exc:`Invalid` exception.

    .. attribute:: DIV_BY_ZERO

       Corresponds to a :exc:`DivisionByZero` exception.

    .. attribute:: INEXACT

       Corresponds to an :exc:`Inexact` exception.

    .. attribute:: OVERFLOW

       Corresponds to an :exc:`Overflow` exception.

    .. attribute:: UNDERFLOW

       Corresponds to an :exc:`UnderflowInexact` exception.


Signals
=======

Most operations, called **signalling** operations, can signal during their calculation
depending on the values of their arguments; other **quiet** operations never raise
signals.  Some signals indicate unusual conditions, such as :exc:`DivisionByZero`, others
like :exc:`Inexact` are very common.

The IEEE-754 standard specifies five signal categories, namely :exc:`Invalid`,
:exc:`DivisionByZero`, :exc:`Inexact`, :exc:`Overflow` and :exc:`Underflow`.  Each of
these categories is associated with a flag in context objects which is normally set when
it occurs.  Operations only set flags and never clear them; application code must decide
if and when to clear these flags.

The :mod:`ieee754` module defines several sub-categories of signal as a hierarchy of
exceptions.  Signal handling can be controlled for each exception class separately or in
groups, offering very fine-grained control.  For example, you might want to specify that
invalid operation signals arising from multiplication of zeroes and infinities -
represented by the :exc:`InvalidMultiply` sub-exception of the :exc:`Invalid` exception
category - raise a Python exception when they occur, and that all other :exc:`Invalid`
signals should be handled by default and not interrupt program execution.

Each exception carries the name and operands of the operation that caused it, and the
default result that should be delivered by default exception handling.  The exception
class hierarchy as follows::

  IEEEError(ArithmeticError)
      Invalid
          SignallingNaNOperand
          InvalidAdd
          InvalidMultiply
          InvalidDivide
          InvalidFMA
          InvalidRemainder
          InvalidSqrt
          InvalidToString
          InvalidConvertToInteger
          InvalidComparison
          InvalidLogBIntegral
      DivisionByZero(IEEEError, ZeroDivisionError)
          DivideByZero
          LogBZero
      Inexact
      Overflow
      Underflow
          UnderflowExact
          UnderflowInexact


.. exception:: IEEEError(op_tuple, result)

    All exceptions defined in the `ieee754` module derive from :exc:`IEEEError`.  The
    constructor initializes the object's attributes as indicated:

    .. attribute:: op_tuple

    A tuple of the :ref:`operation name<Operation Names>` and its operands that caused the
    signal, for example this tuple indciates that the :meth:`divide` operation raised the
    signal when *x* and *y* were passed as operands::

         op_tuple = (OP_DIVIDE, x, y)

    .. attribute:: result

    The result that default exception handling should deliver.  This can be inspected to
    determine the appropriate destination format for the operation.


.. exception:: Invalid

    The class representing all invalid operations specified in the IEEE-754 standard.
    Operations signal invalid when there is no usefully defineable result, and set the
    default result to a quiet :const:`NaN`.

    Its constructor has the same signature as that of :exc:`IEEEError`, but if *result* is
    a :class:`BinaryFormat` instance, then *result* is converted to a quiet :const:`NaN`
    of that format with zero payload and clear sign bit.

    ::exc::`Invalid` has many sub-exceptions which indicate more precisely what happened.


.. exception:: SignallingNaNOperand

    This signal is raised when any signalling operation (with the possible exception of
    conversions to string) is passed a signalling :const:`NaN` operand.  The default
    result will be a quiet :const:`NaN` in the destination format, see `NaN Propagation`_
    for more details.


.. exception:: InvalidAdd

    The signal is raised when :meth:`add` is given two differently-signed infinities, or
    :meth:`subtract` is given two like-signed infinities.


.. exception:: InvalidMultiply

    This signal is raised when the :meth:`multiply` operation is passed a zero and an
    infinity.


.. exception:: InvalidDivide

    This signal is raised when the :meth:`divide` operation is passed two zeros or two
    infinities.


.. exception:: InvalidFMA

    This signal is raised when the fused-multiply-add operation :meth:`fma` multiplies a
    zero and an infinity.


.. exception:: InvalidRemainder

    This signal is raised when a remainder operation has an infinite dividend or a zero
    divisor, and neither operand is a :const:`NaN`.

    The remainder operations are :meth:`remainder`, :meth:`fmod`, :meth:`mod`,
    :meth:`floordiv` and :meth:`divmod`.  Of these only :meth:`remainder` is described in
    the IEEE standard.


.. exception:: InvalidSqrt

    This signal is raised when the square root operation :meth:`sqrt` is passed an operand
    less than zero.


.. exception:: InvalidToString

    This signal is raised when a decimal or hexadecimal string conversion operation is
    passed a signalling :const:`NaN` and the :class:`TextFormat` :attr:`snan` attribute is
    empty, indicating to output it as a quiet :const:`NaN`.


.. exception:: InvalidConvertToInteger

    Raised during the conversion of a Binary value to an integer format when the result
    cannot be represented in that format.

    This happens when the result would be too large or too small, or the source value is
    an infinity or :const:`NaN`.


.. exception:: InvalidComparison

   Raised when a comparison is done on two values that would return :const:`Unordered`
   (i.e., at least one of the operands is a :const:`NaN`) and the comparison predicate
   indicates that unordered comparisons should raise an invalid operation signal.


.. exception:: InvalidLogBIntegral

   This signal is raised when the operand of :meth:`logb_integral` is a zero, infinity or
   :const:`NaN`.


.. exception:: DivisionByZero

   This class is the base class of all division-by-zero exceptions specified in the
   **IEEE-754** standard.  Division by zero is signalled when an operation on finite
   operands delivers an exact infinite result.

   This class has two sub-exceptions.


.. exception:: DivideByZero

   Raised when the :meth:`divide` operation was passed a zero divisor.


.. exception:: LogBZero

   Raised when :meth:`logb` operates on a a zero value.


.. exception:: Inexact

   One of the five IEEE-754 signals, this is raised when the infinitely precise result
   cannot be represented in the destination format.  This is perhaps the most common
   signal.

   The default result is the precise result rounded according to the rounding mode to fit
   the destination format.

   This class has no sub-exceptions.

   Under default exception handling this signal raises the :attr:`INEXACT` flag.


.. exception:: Overflow

   Overflow is one of the five IEEE-754 signals.  It is raised, after rounding, when the
   result would have an exponent exceeding the destination format's :attr:`e_max`
   attribute.

   The default result is either infinity, or the finite value of the greatest magnitude,
   depending on *rounding* and *sign*.

   This class has no sub-exceptions.

   Under default exception handling this signal raises the :attr:`OVERFLOW` flag and
   signals :exc:`Inexact`.


.. exception:: Underflow

    The last of the five IEEE-754 signals, :exc:`Underflow` is signalled when a tiny
    non-zero result is detected.  Tininess means the precise non-zero result computed as
    though with unbounded exponent range would lie strictly between ``± 2^e_min`` where
    :attr:`e_min` is the minimum normalized exponent of the destination format.

    Tininess can be detected before or after rounding, as determined by the operation's
    *context* argument.

    This exception must not be raised directly; instead one of its two sub-exceptions
    should be raised depending on whether the result is exact.


.. exception:: UnderflowExact

   This exception is signalled when the result it tiny but exact.  Since the result is
   exact it was necessarily tiny before and after rounding.

   Under default exception handling, as per IEEE-754, this signal does *not* raise the
   :attr:`UNDERFLOW` flag and it does *not* signal :exc:`Inexact`.  It is the only signal
   to not raise its associated flag.


.. exception:: UnderflowInexact

   This exception is signalled when tininess was detected and rounding of the precise
   result was necessary.

   A tiny rounded result was necessarily tiny before rounding, however an infinitely
   precise result that was tiny might round to be the smallest finite non-tiny number.
   Hence it matters whether tininess is detected before or after rounding; this is
   controlled by the :attr:`tininess_after` attribute of the *context* of the operation.

   The method of detecting tininess has no effect on the rounded result delivered, which
   might be any of zero, a subnormal number, or the smallest finite normal number.

   Under default exception handling this signal raises the :attr:`UNDERFLOW` flag and
   signals :exc:`Inexact`.


Alternate Exception Handling
============================

When a signal is raised during a computation, it is sometimes desirable to handle the
signal in a different way to the `default exception handling` that the IEEE-754 standard
specifies.

Alternate means of handling an exception that occurs in a block of code can be categorised
as follows:

  * :dfn:`resuming` ones handle the exception immediately, taking some action which
    delivers a result, and then execution of the block of code continues normally.

  * :dfn:`immediate` ones immediately abandon the block of code, call an alternative block
    of code to handle the exception condition, and then resume control after the end of
    the code block.

  * :dfn:`immediate with transfer` ones immediately abandon the block of code and transfer
    control to an alternative block of code to handle the exception condition, with no
    return possible.

  * :dfn:`delayed` ones deliver a default result and resume execution of the code block.
    The actual exception handling takes place when the block of code ends.

  * :dfn:`delayed with transfer` ones deliver a default result and transfer control at the
    end of the block of code to an alternative block of code to handle the exception, with
    no return possible.

This module supports several resuming exception handling methods.  The `immediate` form is
supported by specifying the signal should raise a Python exception and placing a ``try
... except ...`` construct around the block of code.  The `immediate with transfer` form
is similarly supported by placing the ``try ... except ...`` construct at a higher place
in the call stack than the block of code in question.

At present no support is implemented for the delayed forms of exception handling.


.. class:: HandlerKind

    Values of the :class:`HandlerKind` enumeration can be associated with a signal via
    :meth:`Context.set_handler` to specify alternate means of handling exceptions.  The
    :attr:`SUBSTITUTE_VALUE` and :attr:`SUBSTITUTE_VALUE_XOR` kinds require a handler to
    be given; the others take no handler.

    .. attribute:: DEFAULT

       The associated exception is handled with default exception handling.

    .. attribute:: NO_FLAG

       The associated exception is handled with default exception handling but the
       associated flag is not raised in the context object.

    .. attribute:: MAYBE_FLAG

       The associated exception is handled with default exception handling and the
       associated flag might be raised in the context object.  It is imagined that
       determining whether operations should raise a flag, such as :attr:`Flags.INEXACT`,
       or not might in some cases be expensive.  If the user has indicated with
       :attr:`MAYBE_FLAG` a lack of interest in the accurate signalling of this condition,
       then the implementation can take advantage of this fact to not perform the
       expensive computations required.

       At present no operations in the module take advantage of this leeway, but new ones
       might do so in future.

    .. attribute:: RECORD_EXCEPTION

       The associated exception is handled with default exception handling, and details of
       the exception condition are recorded in the :attr:`exceptions` attribute of the
       context obejct if default exception handling raises a flag.

    .. attribute:: SUBSTITUTE_VALUE

       The associated exception is handled with default exception handling, but a
       different value is delivered as the result.  The value to deliver is returned by
       the handler function passed to :meth:`Context.set_handler`, which takes two
       arguments: the *exception* that has been signalled, and the *context* of the
       operation.  If the handler does not return a value of the correct type and format
       the behaviour is undefined.

    .. attribute:: SUBSTITUTE_VALUE_XOR

       This associated exception is handled with default exception handling unless it
       arises from a multiply or divide operation.

       Behaviour is as for :attr:`SUBSTITUTE_VALUE` except the sign of the value to
       substitue is ignored and instead replaced with the correct sign of the multiply or
       divide operation (i.e., the exclusive or of the signs of the two operands).  Sign
       substitution does not happend for :const:`NaN` values.

    .. attribute:: ABRUPT_UNDERFLOW

       This kind can only be associated with exceptions derived from :exc:`Underflow`.
       When the associated exception is signalled, replace the default result with a zero
       of the same sign, or the minimum **normal** of the same sign, according to the
       applicable rounding mode.  Then raise the :attr:`Flag.UNDERFLOW` flag and signal
       the inexact exception.

    .. attribute:: RAISE

       Immediately raise the associated exception as a Python exception.


Here is a silly but illustrative example::

  >>> from ieee754 import *
  >>> def handler(exception, context):
  ...     # A generic handler would use exception.result.fmt instead of IEEEdouble
  ...     return IEEEdouble.from_string('1.25')
  ...
  >>> context = get_context()
  >>> context.set_handler(DivideByZero, HandlerKind.SUBSTITUTE_VALUE_XOR, handler)
  >>> lhs = IEEEdouble.from_float(1.34)
  >>> (lhs / -0.0).to_decimal_string()
  '-1.25'
  >>> context
  <Context rounding=ROUND_HALF_EVEN flags=<Flags.DIV_BY_ZERO: 2> tininess_after=True>

The example installs a handler for divison by zero which substitutes with appropriate sign
the value 1.25.  When a division by negative zero then happens, the substitution results
in the value -1.25 and the context's flag is raised.


NaN Propagation
===============

So-called `general computational operations` return a quiet :const:`NaN` when any operand
is a :const:`NaN`.  If any operand is a signalling :const:`NaN` then instead the operation
signals :exc:`SignallingNaNOperand` with default result the quiet :const:`NaN`.  The only
exception to this principle is string conversion.  If the :class:`TextFormat` does not
require conversion of signalling NaNs to quiet ones, then string conversion does not raise
a signal as the signalling status is not lost.

The **IEEE-754** standard specifies that under default exception handling the delivered
:const:`NaN` shall be quiet and preserve as much of the payload as possible from the
operand NaNs.  It does not specify which :const:`NaN` operand provides the payload of the
delivered :const:`NaN` if there are two or more, nor the means of payload truncation and
extension to narrower and wider formats.  It does specify that a :const:`NaN` payload
converted to a wider format, and then back again to the original format, should not
change.

This implementation behaves as follows:

  * payloads are viewed as unsigned integer values.  For binary interchange formats, the
    least significant bit of the significand forms the least significant bit of the
    payload.
  * when a :const:`NaN` is converted to a narrower format, the payload is truncated by
    losing its most significant bits.  When converted to a wider format leading zero bits
    are added.
  * one *source* :const:`NaN` is chosen from among the operand NaNs.  Its sign provides
    the sign of the delivered quiet :const:`NaN`, and its payload is converted to the
    destination format as described above.
  * the leftmost :const:`NaN` in the list of operands, whether signalling or quiet, whose
    payload preserves its value when converted to the destination format, is first chosen
    as the source.
  * if no :const:`NaN` would preserve its payload value on conversion, then the leftmost
    :const:`NaN` is chosen as the source.


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

     This attribute does not affect :attr:`inf`, :attr:`qnan` and :attr:`snan` which
     appear in the output string verbatim.

  .. attribute:: inf

     The string to output for infinities.  The default is 'Infinity'.

  .. attribute:: qnan

     The string to output for a quiet :const:`NaN`.  The default is 'NaN'.

  .. attribute:: snan

     The string to output for a signalling :const:`NaN`.  An empty string means output it
     as a quiet :const:`NaN` instead, which signals an :exc:`InvalidToString` exception.
     The default is 'sNaN'.

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
                                     inf='inf', qnan='nan', snan='snan', nan_payload='N')


.. data:: Dec_G_Format

     This instance is intended to match the output of Python's **g** format specifier when
     the specified precisions are the same::

       DefaultDecFormat = TextFormat(exp_digits=-2, rstrip_zeroes=True,
                                     inf='inf', qnan='nan', snan='snan', nan_payload='N')


String Syntax
=============

:meth:`BinaryFormat.from_string` accepts a broad class of strings.  After removing leading
and trailing whitespace, and underscores throughout, it should conform to the following
syntax::

      sign            ::=  '+' | '-'
      digit           ::=  '0' | '1' | '2' | '3' | '4' | '5' | '6' | '7' | '8' | '9'
      hex-digit       ::=  digit | 'a' | 'b' | 'c' | 'd' | 'e' | 'f'
      dec-exp-char    ::=  'e'
      hex-exp-char    ::=  'p'
      hex-specifier   ::=  '0x'
      nan-specifier   ::=  'nan' | 'snan'
      infinity        ::=  'infinity' | 'inf'
      exponent        ::=  [sign] digits
      digits          ::=  digit [digit]...
      hex-digits      ::=  hex-digit [hex-digit]...
      dec-significand ::=  digits '.' [digits] | ['.'] digits
      hex-significand ::=  hex-digits '.' [hex-digits] | ['.'] hex-digits
      dec-exponent    ::=  dec-exp-char exponent
      hex-exponent    ::=  hex-exp-char exponent
      nan-payload     ::=  hex-specifier hex-digits | digits
      nan             ::=  nan-specifier [nan-payload]
      dec-value       ::=  dec-significand [dec-exponent]
      hex-value       ::=  hex-significand hex-exponent
      numeric-value   ::=  hex-specifier hex-value | dec-value
      numeric-string  ::=  [sign] numeric-value | [sign] nan | [sign] infinity

Case is ignored and other Unicode decimal digits are permitted where ASCII digits appear
above.


.. _Operation Names:

Operation Names
===============

The following constants are defined in the module and form the first element of the
:attr:`op_tuple` attribute of exceptions.  Each is a string which is the method name that
performs the operation.  For example, :data:`OP_DIVIDE` is "divide".


.. data:: OP_ABS

   '__abs__' representing Python's builtin :func:`abs`.

.. data:: OP_ADD
.. data:: OP_SUBTRACT
.. data:: OP_MULTIPLY
.. data:: OP_DIVIDE
.. data:: OP_FMA
.. data:: OP_REMAINDER
.. data:: OP_FMOD
.. data:: OP_MOD
.. data:: OP_DIVMOD
.. data:: OP_FLOORDIV
.. data:: OP_SQRT
.. data:: OP_SCALEB
.. data:: OP_LOGB
.. data:: OP_LOGB_INTEGRAL
.. data:: OP_NEXT_UP
.. data:: OP_NEXT_DOWN
.. data:: OP_COMPARE
.. data:: OP_CONVERT
.. data:: OP_ROUND_TO_INTEGRAL
.. data:: OP_ROUND_TO_INTEGRAL_EXACT
.. data:: OP_CONVERT_TO_INTEGER
.. data:: OP_CONVERT_TO_INTEGER_EXACT
.. data:: OP_FROM_FLOAT
.. data:: OP_FROM_INT
.. data:: OP_FROM_STRING
.. data:: OP_TO_STRING
.. data:: OP_TO_DECIMAL_STRING
.. data:: OP_MAX
.. data:: OP_MAX_NUM
.. data:: OP_MIN
.. data:: OP_MIN_NUM
.. data:: OP_MAX_MAG_NUM
.. data:: OP_MAX_MAG
.. data:: OP_MIN_MAG_NUM
.. data:: OP_MIN_MAG
.. data:: OP_MINUS

   '__neg__' representing Python's built-in unary minus.

.. data:: OP_PLUS

   '__pos__' representing Python's built-in unary plus.
