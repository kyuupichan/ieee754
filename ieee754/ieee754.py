#
# An implementation of many operations of generic binary floating-point arithmetic
#
# (c) Neil Booth 2007-2021.  All rights reserved.
#

import copy
import re
import threading
from collections import namedtuple
from decimal import Decimal
from enum import IntFlag, IntEnum
from fractions import Fraction
from math import ceil, floor, log2, sqrt
from typing import NamedTuple
from struct import Struct

import attr

__all__ = ('Context', 'DefaultContext', 'get_context', 'set_context', 'local_context',
           'DefaultDecFormat', 'DefaultHexFormat', 'Dec_g_Format', 'DecimalToBinary',
           'Flags', 'Compare', 'HandlerKind',
           'BinaryFormat', 'Binary', 'TextFormat',
           'IEEEError', 'Invalid', 'DivisionByZero', 'Inexact', 'Overflow', 'Underflow',
           'SignallingNaNOperand', 'InvalidAdd', 'InvalidMultiply', 'InvalidDivide',
           'InvalidSqrt', 'InvalidFMA', 'InvalidRemainder', 'InvalidLogBIntegral',
           'InvalidConvertToInteger', 'InvalidComparison', 'InvalidToString',
           'DivideByZero', 'LogBZero', 'UnderflowExact', 'UnderflowInexact',
           'ROUND_CEILING', 'ROUND_FLOOR', 'ROUND_DOWN', 'ROUND_UP',
           'ROUND_HALF_EVEN', 'ROUND_HALF_UP', 'ROUND_HALF_DOWN',
           'OP_ABS', 'OP_ADD', 'OP_SUBTRACT', 'OP_MULTIPLY', 'OP_DIVIDE', 'OP_FMA', 'OP_SQRT',
           'OP_REMAINDER', 'OP_FMOD', 'OP_MOD', 'OP_DIVMOD', 'OP_FLOORDIV',
           'OP_LOGB', 'OP_LOGB_INTEGRAL', 'OP_SCALEB',
           'OP_NEXT_UP', 'OP_NEXT_DOWN', 'OP_COMPARE',
           'OP_CONVERT', 'OP_CONVERT_TO_INTEGER', 'OP_CONVERT_TO_INTEGER_EXACT',
           'OP_ROUND_TO_INTEGRAL', 'OP_ROUND_TO_INTEGRAL_EXACT',
           'OP_FROM_INT', 'OP_FROM_FLOAT', 'OP_FROM_STRING',
           'OP_TO_STRING', 'OP_TO_DECIMAL_STRING',
           'OP_MAX', 'OP_MAX_NUM', 'OP_MIN', 'OP_MIN_NUM', 'OP_MAX_MAG_NUM', 'OP_MAX_MAG',
           'OP_MIN_MAG_NUM', 'OP_MIN_MAG',
           'OP_PLUS', 'OP_MINUS',
           'IEEEhalf', 'IEEEsingle', 'IEEEdouble', 'IEEEquad',
           'x87extended', 'x87double', 'x87single')


# Rounding modes
ROUND_CEILING   = 'ROUND_CEILING'       # Towards +infinity
ROUND_FLOOR     = 'ROUND_FLOOR'         # Towards -infinity
ROUND_DOWN      = 'ROUND_DOWN'          # Torwards zero
ROUND_UP        = 'ROUND_UP'            # Away from zero
ROUND_HALF_EVEN = 'ROUND_HALF_EVEN'     # To nearest with ties towards even
ROUND_HALF_DOWN = 'ROUND_HALF_DOWN'     # To nearest with ties towards zero
ROUND_HALF_UP   = 'ROUND_HALF_UP'       # To nearest with ties away from zero


# Operation names
OP_ABS = '__abs__'
OP_ADD = 'add'
OP_SUBTRACT = 'subtract'
OP_MULTIPLY = 'multiply'
OP_DIVIDE = 'divide'
OP_FMA = 'fma'
OP_REMAINDER = 'remainder'
OP_FMOD = 'fmod'
OP_MOD = 'mod'                # Python
OP_DIVMOD = 'divmod'          # Python
OP_FLOORDIV = 'floordiv'      # Python
OP_SQRT = 'sqrt'
OP_SCALEB = 'scaleb'
OP_LOGB = 'logb'
OP_LOGB_INTEGRAL = 'logb_integral'
OP_NEXT_UP = 'next_up'
OP_NEXT_DOWN = 'next_down'
OP_COMPARE = 'compare'
OP_CONVERT = 'convert'
OP_ROUND_TO_INTEGRAL = 'round_to_integral'
OP_ROUND_TO_INTEGRAL_EXACT = 'round_to_integral_exact'
OP_CONVERT_TO_INTEGER = 'convert_to_integer'
OP_CONVERT_TO_INTEGER_EXACT = 'convert_to_integer_exact'
OP_FROM_FLOAT = 'from_float'
OP_FROM_INT = 'from_int'
OP_FROM_STRING = 'from_string'
OP_TO_STRING = 'to_string'
OP_TO_DECIMAL_STRING = 'to_decimal_string'
OP_MAX = 'max'
OP_MAX_NUM = 'max_num'
OP_MIN = 'min'
OP_MIN_NUM = 'min_num'
OP_MAX_MAG_NUM = 'max_mag_num'
OP_MAX_MAG = 'max_mag'
OP_MIN_MAG_NUM = 'min_mag_num'
OP_MIN_MAG = 'min_mag'
OP_MINUS = '__neg__'
OP_PLUS = '__pos__'


class MinMaxFlags(IntFlag):
    MIN = 0x00
    MAX = 0x01
    MAG = 0x02
    NUM = 0x04

    def op_name(self):
        max_min = 'max' if self & MinMaxFlags.MAX else 'min'
        mag = '_mag' if self & MinMaxFlags.MAG else ''
        num = '_num' if self & MinMaxFlags.NUM else ''
        return f'{max_min}{mag}{num}'


# Four-way result of the compare() operation.
class Compare(IntEnum):
    LESS_THAN = 0
    EQUAL = 1
    GREATER_THAN = 2
    UNORDERED = 3


# Operation status flags.
class Flags(IntFlag):
    INVALID     = 0x01
    DIV_BY_ZERO = 0x02
    OVERFLOW    = 0x04
    UNDERFLOW   = 0x08
    INEXACT     = 0x10


BinaryTuple = namedtuple('BinaryTuple', 'sign exponent significand')
pack_double = Struct('=d').pack
unpack_double = Struct('=d').unpack


@attr.s(slots=True, kw_only=True, cmp=False)
class TextFormat:
    '''Controls the output of conversion to decimal and hexadecimal strings.'''

    # The minimum number of digits to output in the exponent of a finite number.  Defaults
    # to 1.  For decimal output 0 suppresses the exponent by adding leading or trailing
    # zeroes to the significand as needed (as for the printf 'f' format specifier in the C
    # programming language).  If negative, apply the rule for the printf 'g' format
    # specifier to decide whether to display an exponent or not, in which case the minimum
    # number of digits in the exponent is the absolute value.
    exp_digits = attr.ib(default=1)
    # If True positive exponents display a '+'.
    force_exp_sign = attr.ib(default=True)
    # If True, numbers with a clear sign bit are preceded with a '+'.
    force_leading_sign = attr.ib(default=False)
    # If True, display a floating point followed by a zero even though none is needed.  For
    # example, "5", "1e2" and "0x1p2" would display as "5.0", "1.0e2" and 0x1.0p2".
    force_point = attr.ib(default=False)
    # If True, the exponent character ('p' or 'e') is in upper case, and for hexadecimal
    # output, the hex indicator 'x' and hexadecimal digits are in upper case.  For NaN
    # output, hexadecimal payloads are in upper case.  This does not affect the text of
    # the inf, qnan and snan indicators below which are copied unmodified.
    upper_case = attr.ib(default=False)
    # If True, trailing insignificant zeroes are stripped
    rstrip_zeroes = attr.ib(default=False)
    # The string output for infinity
    inf = attr.ib(default='Infinity')
    # The string output for quiet NaNs
    qnan = attr.ib(default='NaN')
    # The string output for signalling NaNs.  The empty string means flag an invalid
    # operation and output as a quiet NaN instead.
    snan = attr.ib(default='sNaN')
    # Controls the display of NaN payloads.  If N, NaN payloads are omitted.  If X, they
    # are output in hexadecimal.  If 'D' in decimal.  Examples of all 3 formats are: nan,
    # nan255 and nan0xff for a quiet NaN with payload 255, respectively.
    nan_payload = attr.ib(default='X')

    def leading_sign(self, value):
        '''Return the leading sign string.'''
        return '-' if value.sign else '+' if self.force_leading_sign else ''

    def exponent_str(self, exponent):
        '''Return the formatted exponent.'''
        sign = '-' if exponent < 0 else '+' if self.force_exp_sign else ''
        main = str(abs(exponent))
        zeroes = '0' * (abs(self.exp_digits) - len(main))
        return f'{sign}{zeroes}{main}'

    def format_non_finite(self, value, op_tuple, context):
        '''Returns the output text for infinities and NaNs.

        Signals InvalidToString if a signalling NaN is output as a quiet NaN.'''
        lost_snan = False
        # Infinity
        if value.is_infinite():
            special = self.inf
        else:
            # NaNs
            special = self.qnan
            if value.is_snan():
                if self.snan:
                    special = self.snan
                else:
                    lost_snan = True
            if self.nan_payload == 'D':
                special += str(value.nan_payload())
            elif self.nan_payload == 'X':
                special += hex(value.nan_payload())

        result = self.leading_sign(value) + special
        if lost_snan:
            result = InvalidToString(op_tuple, result).signal(context)
        return result

    def format_decimal(self, sign, exponent, digits, precision=None):
        '''sign is True if the number has a negative sign.  sig_digits is a string of significant
        digits of a number converted to decimal.  exponent is the exponent of the leading
        digit, i.e. the decimal point appears exponent digits after the leading digit.
        '''
        precision = precision or len(digits)
        assert precision > 0

        if self.rstrip_zeroes:
            digits = digits.rstrip('0') or '0'

        parts = []
        if sign:
            parts.append('-')
        elif self.force_leading_sign:
            parts.append('+')

        exp_digits = self.exp_digits
        if exp_digits < 0:
            # Apply the fprintf 'g' format specifier rule
            if precision > exponent >= -4:
                exp_digits = 0

        if exp_digits:
            if len(digits) > 1:
                parts.extend((digits[0], '.', digits[1:]))
            elif self.force_point:
                parts.extend((digits, '.0'))
            else:
                parts.append(digits)
            parts.append('e')
            parts.append(self.exponent_str(exponent))
        else:
            point = exponent + 1
            if point <= 0:
                parts.extend(('0.', '0' * -point, digits))
            else:
                if point > len(digits):
                    digits += (point - len(digits)) * '0'
                if point < len(digits):
                    parts.extend((digits[:point], '.', digits[point:]))
                elif self.force_point:
                    parts.extend((digits, '.0'))
                else:
                    parts.append(digits)

        result = ''.join(parts)
        if self.upper_case:
            result = result.upper()

        return result

    def format_hex(self, value):
        '''Return the finite value formatted as a hexadecimal float.'''
        # The value has been converted and rounded to our format.  We output the unbiased
        # exponent, but have to shift the significand left up to 3 bits so that converting
        # the significand to hex has the integer bit as an MSB
        significand = value.significand
        if significand == 0:
            exponent = 0
        else:
            significand <<= (value.fmt.precision & 3) ^ 1
            exponent = value.exponent()

        # Treat zero specially only because Python does
        if value.is_zero():
            sig_digits = 1
        else:
            sig_digits = (value.fmt.precision + 6) // 4
        hex_sig = f'{significand:x}'
        # Prepend zeroes to get the full output precision
        hex_sig = '0' * (sig_digits - len(hex_sig)) + hex_sig
        # Strip trailing zeroes?
        if self.rstrip_zeroes:
            hex_sig = hex_sig.rstrip('0') or '0'
        # Insert the decimal point only if there are trailing digits
        if len(hex_sig) > 1:
            hex_sig = hex_sig[0] + '.' + hex_sig[1:]
        elif self.force_point:
            hex_sig += '.0'

        sign = self.leading_sign(value)
        result = f'{sign}0x{hex_sig}p{self.exponent_str(exponent)}'
        if self.upper_case:
            result = result.upper()
        return result


# Default format for decimal output
DefaultDecFormat = TextFormat(exp_digits=-2, force_point=True,
                              inf='inf', qnan='nan', snan='snan', nan_payload='N')

# Default format for hexadecimal output
DefaultHexFormat = TextFormat(force_point=True, nan_payload='N')


# This instance is intended to match the output of Python's **g** format specifier when
# the specified precisions are the same.
Dec_g_Format = TextFormat(exp_digits=-2, rstrip_zeroes=True,
                          inf='inf', qnan='nan', snan='snan', nan_payload='N')


#
# Signals
#

class IEEEError(ArithmeticError):
    '''All arithmetic exceptions signalled by this moduile subclass from this.

    IEEEError expects two arguments:

         def __init__(self, op_tuple, result):

    op_tuple is a tuple of the operation name and operands causing the signal. result is
    the result in the destination format that default exception handling should deliver.

    Derived classes may initialize differently but must initialize IEEEError in this
    way.

    Exceptions derived from IEEEError must have a linear inheritance from it and
    through the first base class if an exception has multiple base classes.  See, for
    example, DivisionByZero.
    '''

    flag_to_raise = 'Nope! Fix your bug.'

    @property
    def op_tuple(self):
        return self.args[0]

    @property
    def default_result(self):
        return self.args[1]

    def is_multiply_divide(self):
        return self.op_tuple[0] in {OP_MULTIPLY, OP_DIVIDE}

    def signal(self, context=None):
        '''Call to signal an exception.  This routine handles the exception according to
        default or alternative exception handling as specified in the context.'''
        context = context or get_context()
        kind, handler = context.handler(self.__class__)
        result = self.default_result

        if kind != HandlerKind.NO_FLAG:
            if kind == HandlerKind.ABRUPT_UNDERFLOW:
                result = result.fmt.make_underflow_value(context.rounding, result.sign, True)
                context.flags |= Flags.UNDERFLOW
                return Inexact(self.op_tuple, result).signal(context)

            # flag_to_raise can be zero for exact underflow
            context.flags |= self.flag_to_raise
            if kind == HandlerKind.RECORD_EXCEPTION and self.flag_to_raise:
                context.exceptions.append(self)

        if kind == HandlerKind.RAISE:
            raise self
        if kind == HandlerKind.SUBSTITUTE_VALUE:
            result = handler(self, context)
        elif kind == HandlerKind.SUBSTITUTE_VALUE_XOR and self.is_multiply_divide():
            result = handler(self, context)
            if not result.is_nan():
                result = result.set_sign(self.op_tuple[1].sign ^ self.op_tuple[2].sign)

        return result


#
# Invalid - many sub-exceptions
#

class Invalid(IEEEError):
    '''Invalid operation base class.  Signalled when an operation has no usefully defineable
    result.  The default result is a quiet NaN.
    '''

    flag_to_raise = Flags.INVALID

    def __init__(self, op_tuple, result):
        '''If result is a BinaryFormat its canonical quiet NaN is used.'''
        if isinstance(result, BinaryFormat):
            result = result.make_nan(False, False, 0)
        super().__init__(op_tuple, result)


class SignallingNaNOperand(Invalid):
    '''Signalled when an operand is a signalling NaN.'''


class InvalidAdd(Invalid):
    '''Signalled when adding two differently-signed infinities or subtracting two like-signed
    infinities.'''


class InvalidMultiply(Invalid):
    '''Signalled when multiplying a zero and an infinity.'''


class InvalidDivide(Invalid):
    '''Signalled when dividing two zeros or two infinities.'''


class InvalidFMA(Invalid):
    '''Signalled when an FMA operation multiplies a zero and an infinity.'''


class InvalidRemainder(Invalid):
    '''Signalled by remainder(x, y) when x is infinite, or y is zero, and neither is a NaN.'''


class InvalidSqrt(Invalid):
    '''Signalled if the sqrt operand is less than zero.'''


class InvalidToString(Invalid):
    '''Signalled if to_string() converts a singalling NaN to a quiet NaN.'''


class InvalidConvertToInteger(Invalid):
    '''Signalled if the result cannot be represented in the destination format.'''


class InvalidComparison(Invalid):
    '''Signalled on comparison of unordered operands with unordered signalling predicates.'''


class InvalidLogBIntegral(Invalid):
    '''Singalled when the operand of logb_integral is a zero, infinity or NaN.'''


#
# DivisionByZero - sub-exceptions are DivisionByZero, LogBZero
#

class DivisionByZero(IEEEError, ZeroDivisionError):
    '''Base class of division by zero errors.  Division by zero is signalled when an operation
    on finite operands delivers an exact infinite result.'''

    flag_to_raise = Flags.DIV_BY_ZERO


class DivideByZero(DivisionByZero):
    '''A divide operation with zero divisor.'''


class LogBZero(DivisionByZero):
    '''LogB on a zero.'''


#
# Inexact.  Not subclassed.
#

class Inexact(IEEEError):
    '''Signalled when the infinitely precise result cannot be represented.'''

    flag_to_raise = Flags.INEXACT


#
# Overflow.  Not subclassed.
#

class Overflow(IEEEError):
    '''Signalled when, after rounding, the result would have an exponent exceeding e_max.  The
       default result is either infinity, or the finite value of the greatest magnitude,
       depending on the rounding mode and sign.'''

    flag_to_raise = Flags.OVERFLOW

    def signal(self, context=None):
        '''Call to signal the overflow exception.  Defer to the base class for standard handling;
        then signal inexact.'''
        result = super().signal(context)
        return Inexact(self.op_tuple, result).signal(context)


#
# Underflow - sub-exceptions are UnderflowExact and UnderflowInexact.
#

class Underflow(IEEEError):
    '''Signalled when a tiny non-zero result is detected.  Tininess means the result computed
    as though with unbounded exponent range would lie strictly between ±2^e_min.  Tininess
    can be detected before or after rounding.'''


class UnderflowExact(Underflow):
    '''An exact underflow.  By default, raises no flag.'''

    flag_to_raise = 0


class UnderflowInexact(Underflow):
    '''An inexact underflow.'''

    flag_to_raise = Flags.UNDERFLOW

    def signal(self, context=None):
        '''Signal an inexact underflow.  Defer to the base class for standard handling; then
        signal inexact.'''
        result = super().signal(context)
        return Inexact(self.op_tuple, result).signal(context)


# Alternate exception handling

class HandlerKind(IntEnum):
    '''Indicates how a singalled exception should be handled.'''
    # Default exception handling as per IEEE-754.  This returns a default value, and
    # except possibly for underflow, raises a flag
    DEFAULT = 0

    # Default exception handling without raising the associated flag
    NO_FLAG = 1

    # Default exception handling but raise the associated flag if not expensive.
    MAYBE_FLAG = 2

    # Default exception handling but record the exception if that raises the associated flag
    RECORD_EXCEPTION = 3

    # Default exception handling but substitute a value for the default result.  A handler
    # must be provided with signature
    #
    #    def handler(exception, context):
    #
    # The value returned by the handler will become the operation's result.  The behaviour
    # is undefined if this is not a Binary object with format that of
    # exception.default_result.
    SUBSTITUTE_VALUE = 4

    # For exceptions arising from multiply and divide operations: default exception
    # handling but substitute a value for the default result, giving it the sign bit the
    # XOR of the signs of the operands if it is not a NaN.  A handler must be provided as
    # described for SUBSTITUTE_VALUE.
    SUBSTITUTE_VALUE_XOR = 5

    # When underflow is signalled, replace the default result with a zero of the same
    # sign, or a minimum *normal* rounded result of the same sign, raise the underflow
    # flag, and signal the inexact exception.
    ABRUPT_UNDERFLOW = 6

    # Raise the exception immediately
    RAISE = 7

    def requires_handler(self):
        return self in {HandlerKind.SUBSTITUTE_VALUE, HandlerKind.SUBSTITUTE_VALUE_XOR}


class Context:
    '''The execution context for operations.  Carries the rounding mode, status flags,
    whether tininess is detected before or after rounding, and traps.'''

    __slots__ = ('rounding', 'flags', 'tininess_after', 'handlers', 'exceptions')

    def __init__(self, *, rounding=ROUND_HALF_EVEN, flags=0, tininess_after=True):
        '''rounding is one of the ROUND_ constants and (mostly) controls the rounding of inexact
        results.  flags represents the initially raised flags.  tininess_after indicates
        if tininess is detected before or after rounding.
        '''
        self.rounding = rounding
        self.flags = flags
        self.tininess_after = tininess_after
        self.handlers = {}
        self.exceptions = []

    def copy(self):
        '''Return a (deep) copy of the context.'''
        # A deep copy is needed because handlers and exceptions are mutable containers
        return copy.deepcopy(self)

    def set_handler(self, exc_classes, kind, handler=None):
        classes = (exc_classes, ) if not isinstance(exc_classes, (tuple, list)) else exc_classes
        base = Underflow if kind == HandlerKind.ABRUPT_UNDERFLOW else IEEEError
        if not all (issubclass(exc_class, base) for exc_class in classes):
            raise TypeError(f'all exception classes must be subclasses of {base.__name__}')
        if not isinstance(kind, HandlerKind):
            raise TypeError('kind must be a HandlerKind instance')
        if (handler is not None) ^ kind.requires_handler():
            if handler is None:
                raise ValueError(f'handler not given for kind {kind!r}')
            else:
                raise ValueError(f'handler given for kind {kind!r}')
        pair = (kind, handler)
        for exc_class in classes:
            self.handlers[exc_class] = pair

    def handler(self, exc_class):
        '''Return a (handler_kind, callback) pair for a signal class.'''
        if not issubclass(exc_class, IEEEError):
            raise TypeError('exc_class must be a subclass of IEEEError')

        for cls in exc_class.mro():
            handler = self.handlers.get(cls)
            if handler:
                return handler

        return HandlerKind.DEFAULT, None

    def round_to_nearest(self):
        '''Return True if the rounding mode rounds to nearest (ignoring ties).'''
        return self.rounding in {ROUND_HALF_EVEN, ROUND_HALF_DOWN, ROUND_HALF_UP}

    def __repr__(self):
        return (f'<Context rounding={self.rounding} flags={self.flags!r} '
                f'tininess_after={self.tininess_after}>')


# When precision is lost during a calculation these indicate what fraction of the LSB the
# lost bits represented.  It essentially combines the roles of 'guard' and 'sticky' bits.
LF_EXACTLY_ZERO = 0           # 000000
LF_LESS_THAN_HALF = 1	      # 0xxxxx  x's not all zero
LF_EXACTLY_HALF = 2           # 100000
LF_MORE_THAN_HALF = 3         # 1xxxxx  x's not all zero


class BinaryFormat(NamedTuple):
    '''An IEEE-754 binary floating point arithmetic format.  Only instantiate indirectly
    through the from_ contructors.

    precision is the number of bits in the significand including an explicit integer bit.

    e_max is largest e such that 2^e is representable; the largest representable
    number is then 2^e_max * (2 - 2^(1 - precision)) when the significand is all ones.

    e_min is the smallest e such that 2^e is not a subnormal number.  The smallest
    subnormal number is then 2^(e_min - (precision - 1)).

    It is not required that e_min = 1 - e_max.

    A binary format is permitted to be an interchange format if all the following are true:

        a) e_min = 1 - e_max

        b) e_max + 1 is a power of 2.  This means that IEEE-754 interchange format
           exponents use all values in an exponent field of with e_width bits.

        c) The number of bits (1 + e_width + precision), i.e. including the sign bit, is a
           multiple of 16, so that the format can be written as an even number of bytes.
           Alternatively if (2 + e_width + precision) is a multiple of 16, then the format
           is presumed to have an explicit integer bit (as per Intel x87 extended
           precision numbers).
    '''

    # These three attributes determine the rest, which are pre-calculated for efficiency
    precision: int
    e_max: int
    e_min: int

    # All a function of the 3 values above
    e_bias: int
    int_bit: int
    quiet_bit: int
    max_significand: int
    fmt_width: int
    decimal_precision: int
    logb_inf: int

    _converters = {}

    @classmethod
    def from_triple(cls, precision, e_max, e_min):
        '''Make a BinaryFormat with pre-calculated values.   All constructors ultimately
        call this one.'''
        if not all(isinstance(arg, int) for arg in (precision, e_max, e_min)):
            raise TypeError('precision, e_max and e_min must be integers')
        if precision < 3:
            raise ValueError('precision must be at least 3 bits')
        if e_max < 2:
            raise ValueError('e_max must be at least 2')
        if e_min > -1:
            raise ValueError('e_min must be negative')
        e_bias = 1 - e_min
        int_bit = 1 << (precision - 1)
        quiet_bit = 1 << (precision - 2)
        max_significand = (1 << precision) - 1

        # This least number of significant digits to convert to and from decimal correctly
        decimal_precision = 2 + floor(precision / log2_10)

        # What integer value logb(inf) returns.  IEEE-754 requires this and the values for
        # logb(0) and logb(NaN) be values "outside the range ±2 * (emax + p - 1)".
        logb_inf = 2 * (max(e_max, abs(e_min)) + precision - 1) + 1

        # Are we an interchange format?  If not, fmt_width is zero, otherwise it is the
        # format width in bits (the sign bit, the exponent and the significand including
        # the integer bit).  Only interchange formats can be packed to and unpacked from
        # binary bytes.
        fmt_width = 0
        if e_min == 1 - e_max and (e_max + 1) & e_max == 0:
            e_width = e_max.bit_length() + 1
            test_fmt_width = 1 + e_width + precision   # With integer bit
            if test_fmt_width % 8 == 1 or (precision == 64 and e_width == 15):
                fmt_width = test_fmt_width
        return cls(precision, e_max, e_min, e_bias, int_bit, quiet_bit, max_significand,
                   fmt_width, decimal_precision, logb_inf)

    @classmethod
    def from_pair(cls, precision, e_width):
        '''Construct from the specified precision and exponent width.'''
        e_max = (1 << (e_width - 1)) - 1
        return cls.from_triple(precision, e_max, 1 - e_max)

    @classmethod
    def from_precision(cls, precision):
        '''Construct an extended precison format from the specified precision.'''
        if precision >= 128:
            e_width = round(4 * log2(precision)) - 11
        else:
            e_width = round(4 * log2(precision) + 0.5) - 9
        return cls.from_pair(precision, e_width)

    @classmethod
    def from_IEEE(cls, fmt_width):
        '''The IEEE-754 required format for the given width.'''
        if fmt_width == 16:
            precision = 11
        elif fmt_width == 32:
            precision = 24
        elif fmt_width == 64 or (fmt_width >= 128 and fmt_width % 32 == 0):
            precision = fmt_width - round(4 * log2(fmt_width)) + 13
        else:
            raise ValueError('IEEE-754 does not define a standard format for width {fmt_width}')
        return BinaryFormat.from_pair(precision, fmt_width - precision)

    @property
    def logb_zero(self):
        return -self.logb_inf

    @property
    def logb_nan(self):
        return -self.logb_inf - 1

    def __repr__(self):
        return f'BinaryFormat(precision={self.precision}, e_max={self.e_max}, e_min={self.e_min})'

    def __eq__(self, other):
        '''Return True if two formats are equal.'''
        return (isinstance(other, BinaryFormat) and
                (self.precision, self.e_max, self.e_min)
                == (other.precision, other.e_max, other.e_min))

    def make_zero(self, sign):
        '''Return a zero of the given sign.'''
        return Binary(self, sign, 1, 0)

    def make_one(self, sign):
        '''Return a one of the given sign.'''
        return Binary(self, sign, self.e_bias, self.int_bit)

    def make_infinity(self, sign):
        '''Return an infinity of the given sign.'''
        return Binary(self, sign, 0, 0)

    def make_largest_finite(self, sign):
        '''Return the finite number of maximal magnitude with the given sign.'''
        return Binary(self, sign, self.e_max + self.e_bias, self.max_significand)

    def make_smallest_subnormal(self, sign):
        '''Return the smallest subnormal number with the given sign.'''
        return Binary(self, sign, 1, 1)

    def make_smallest_normal(self, sign):
        '''Return the smallest normal number with the given sign.'''
        return Binary(self, sign, 1, self.int_bit)

    def make_nan(self, sign, is_signalling, payload):
        '''Return a NaN with the given sign and payload and signalling status.

        Payload changes, either through loss of most significand bits, or because the payload
        is invalid for a signalling NaN, are silent.'''
        if payload < 0:
            raise ValueError(f'NaN payload cannot be negative: {payload}')
        # If necessary, drop most significant bits
        payload &= self.quiet_bit - 1
        if is_signalling:
            payload = max(payload, 1)
        else:
            payload |= self.quiet_bit
        return Binary(self, sign, 0, payload)

    def make_overflow_value(self, rounding, sign):
        '''Return the value to deliver when an overflow occurs, because the exponent would be too
        large, delivering a result to this format with the given sign.  rounding is the rounding
        mode to apply.'''
        if round_up(rounding, LF_MORE_THAN_HALF, sign, False):
            return self.make_infinity(sign)
        return self.make_largest_finite(sign)

    def make_underflow_value(self, rounding, sign, force_normal):
        '''Returns the value to deliver when a calculation can be determined, typically early, to
        underflow to zero with the given sign.  rounding is the rounding mode to apply.'''
        if round_up(rounding, LF_LESS_THAN_HALF, sign, False):
            if force_normal:
                return self.make_smallest_normal(sign)
            else:
                return self.make_smallest_subnormal(sign)
        else:
            return self.make_zero(sign)

    def _propagate_nan(self, op_tuple, context=None):
        '''Return the result of an operation with at least one NaN.

        This implementation returns the leftmost NaN whose payload can fit in the
        destination format (or the leftmost one if none can), as a quiet NaN.  Signal
        SignallingNaNOperand if any NaNs are signalling.
        '''
        nans = [item for item in op_tuple if isinstance(item, Binary) and item.is_nan()]

        for nan in nans:
            if nan.nan_payload() < self.quiet_bit:
                break
        else:
            nan = nans[0]

        result = self.make_nan(nan.sign, False, nan.nan_payload())
        if any(nan.is_snan() for nan in nans):
            result = SignallingNaNOperand(op_tuple, result).signal(context)
        return result

    def _normalize(self, sign, exponent, significand, op_tuple, context):
        '''Return a normalized floating point number that is the correctly-rounded (by the
        context) value of the infinitely precise result

           ± 2^exponent * significand
        '''
        if significand == 0:
            return self.make_zero(sign)

        context = context or get_context()
        size = significand.bit_length()

        # Shifting the significand so the MSB is one gives us the natural shift.  There t
        # is followed by a decimal point, so the exponent must be adjusted to compensate.
        # However we cannot fully shift if the exponent would fall below e_min.
        exponent += self.precision - 1
        rshift = max(size - self.precision, self.e_min - exponent)

        # Shift the significand and update the exponent
        significand, lost_fraction = shift_right(significand, rshift)
        exponent += rshift

        is_tiny = significand < self.int_bit

        # Round
        if round_up(context.rounding, lost_fraction, sign, bool(significand & 1)):
            # Increment the significand
            significand += 1
            # If the significand now overflows, halve it and increment the exponent
            if significand > self.max_significand:
                significand >>= 1
                exponent += 1

        # If the new exponent would be too big, then signal overflow.
        if exponent > self.e_max:
            value = self.make_overflow_value(context.rounding, sign)
            return Overflow(op_tuple, value).signal(context)

        if context.tininess_after:
            is_tiny = significand < self.int_bit

        is_inexact = lost_fraction != LF_EXACTLY_ZERO

        result = Binary(self, sign, exponent + self.e_bias, significand)
        if is_tiny:
            cls = UnderflowInexact if is_inexact else UnderflowExact
            result = cls(op_tuple, result).signal(context)
        elif is_inexact:
            result = Inexact(op_tuple, result).signal(context)
        return result

    def _next_up(self, value, context, flip_sign):
        '''Return the smallest floating point value (unless operating on a positive infinity or
        NaN) that compares greater than the one whose parts are given.

        Signs are flipped on read and write if flip_sign is True.  next_up() has flip_sign
        False, next_down() as True.
        '''
        assert value.fmt is self
        context = context or get_context()
        sign = value.sign ^ flip_sign
        e_biased = value.e_biased
        significand = value.significand
        op_tuple = (OP_NEXT_DOWN if flip_sign else OP_NEXT_UP, value)

        if e_biased:
            # Increment the significand of positive numbers, decrement the significand of
            # negative numbers.  Negative zero is the only number whose sign flips.
            if sign and significand:
                if significand == self.int_bit and e_biased > 1:
                    significand = self.max_significand
                    e_biased -= 1
                else:
                    significand -= 1
            else:
                sign = False
                significand += 1
                if significand > self.max_significand:
                    significand >>= 1
                    e_biased += 1
                    # Overflow to infinity?
                    if e_biased - self.e_bias > self.e_max:
                        e_biased = 0
                        significand = 0
        else:
            # Negative infinity becomes largest negative number; positive infinity unchanged
            if significand == 0:
                if sign:
                    significand = self.max_significand
                    e_biased = self.e_max + self.e_bias
            # Signalling NaNs are converted to quiet.
            elif significand < self.quiet_bit:
                return self._propagate_nan(op_tuple, context)

        result = Binary(self, sign ^ flip_sign, e_biased, significand)
        if result.is_subnormal():
            result = UnderflowExact(op_tuple, result).signal(context)
        return result

    def convert(self, value, context=None):
        '''Return the value converted to this format, rounding if necessary.

        If value is a signalling NaN, signal SignallingNaNOperand and return a quiet NaN.'''
        op_tuple = (OP_CONVERT, value)
        return self._convert(value, op_tuple, context, False)

    def _convert(self, value, op_tuple, context, preserve_snan):
        '''Return the value converted to this format, rounding if necessary.

        If value is a signalling NaN and preserve_snan is false, signal
        SignallingNaNOperand and return a quiet NaN.'''
        context = context or get_context()

        if value.e_biased:
            # Avoid expensive normalisation to same format; copy and check for subnormals
            if value.fmt is self:
                if value.is_subnormal():
                    value = UnderflowExact(op_tuple, value).signal(context)
                return value

            if value.significand:
                return self._normalize(value.sign, value.exponent_int(), value.significand,
                                       op_tuple, context)

            # Zeroes
            return self.make_zero(value.sign)

        if value.significand:
            # NaNs
            if preserve_snan:
                return self.make_nan(value.sign, value.is_snan(), value.nan_payload())

            result = self.make_nan(value.sign, False, value.nan_payload())
            if value.is_snan():
                result = SignallingNaNOperand(op_tuple, result).signal(context)
            return result

        # Infinities
        return self.make_infinity(value.sign)

    def convert_for_arith(self, value):
        '''Convert value to something capable of doing arithmetic with this format.

        Binary values are returned unmodified.  Python floats are returned as an
        IEEEdouble.  Python ints are converted to this format.  Otherwise None is
        returned.
        '''
        if isinstance(value, Binary):
            return value
        if isinstance(value, float):
            return IEEEdouble_from_float(value)
        if isinstance(value, int):
            return self.from_int(value)
        return None

    def pack(self, sign, exponent, significand, endianness=None):
        '''Packs the IEEE parts of a floating point number as bytes of the given endianness.

        Endianness can be 'big' or 'little'.  If None, host-native endianness is used.

        exponent is the biased exponent in the IEEE sense, i.e., it is zero for zeroes and
        subnormals and e_max * 2 + 1 for NaNs and infinites.  significand must not include
        the integer bit.
        '''
        if not self.fmt_width:
            raise RuntimeError('not an interchange format')
        if not 0 <= significand < self.int_bit:
            raise ValueError('significand out of range')
        if not 0 <= exponent <= self.e_max * 2 + 1:
            raise ValueError('biased exponent out of range')

        # If the format has an explicit integer bit, add it to the significand for normal
        # numbers.
        explicit_integer_bit = self.fmt_width % 8 == 0
        lshift = self.precision - 1
        if explicit_integer_bit:
            lshift += 1
            # NaNs and infinities have the integer bit set
            if 1 <= exponent <= self.e_max * 2 + 1:
                significand += self.int_bit

        # Build up the encoding from the parts
        value = exponent
        if sign:
            value += (self.e_max + 1) * 2
        value = (value << lshift) + significand
        return value.to_bytes(self.fmt_width // 8, endianness or host_endianness)

    def unpack(self, raw, endianness=None):
        '''Decode a binary encoding and return a BinaryTuple.

        Endianness can be 'big' or 'little'.  If None, host-native endianness is used.

        Exponent is the biased exponent in the IEEE sense, i.e., it is zero for zeroes and
        subnormals and e_max * 2 + 1 for NaNs and infinites.  significand does not include
        the integer bit.
        '''
        if not self.fmt_width:
            raise RuntimeError('not an interchange format')
        size = self.fmt_width // 8
        if len(raw) != size:
            raise ValueError(f'expected {size} bytes to unpack; got {len(raw)}')

        value = int.from_bytes(raw, endianness or host_endianness)

        # Extract the parts from the encoding
        implicit_integer_bit = self.fmt_width % 8 == 1
        significand = value & (self.int_bit - 1)
        value >>= self.precision - implicit_integer_bit
        exponent = value & (self.e_max * 2 + 1)
        sign = value != exponent

        return BinaryTuple(sign, exponent, significand)

    def unpack_value(self, raw, endianness=None):
        '''Decode a binary encoding and return a Binary floating point value.

        Endianness can be 'big' or 'little'.  If None, host-native endianness is used.'''
        sign, exponent, significand = self.unpack(raw, endianness)
        if exponent == 0:
            exponent = 1
        elif exponent == self.e_max * 2 + 1:
            exponent = 0
        else:
            significand += self.int_bit

        return Binary(self, sign, exponent, significand)

    def _unpack_value(self, binary, _context):
        '''unpack_value but takes a context for from_value dispatch.'''
        return self.unpack_value(binary)

    ##
    ## General computational operations.  The operand(s) can be different formats;
    ## the destination format is self.
    ##

    def from_value(self, value, context=None):
        '''Return a floating point value derived from value.  Values of type int, float and string
        are accepted, ass passed on to from_int, from_float and from_string respectively.'''
        converter = BinaryFormat._converters.get(type(value))
        if not converter:
            raise TypeError(f'from_value cannot convert values of type {type(value)}')
        return converter(self, value, context)

    def from_decimal(self, value, context=None):
        '''Return the decimal converted to this format, rounding if necessary.'''
        if not isinstance(value, Decimal):
            raise TypeError('from_decimal requires a Decimal instance')
        return self.from_string(str(value), context)

    def from_fraction(self, value, context=None):
        '''Return the decimal converted to this format, rounding if necessary.'''
        if not isinstance(value, Fraction):
            raise TypeError('from_fraction requires a Fraction instance')
        numerator = self._from_int_exact(value.numerator)
        denominator = self._from_int_exact(value.denominator)
        return self.divide(numerator, denominator, context)

    @classmethod
    def _from_int_exact(cls, value):
        '''Returns an integer as a floating point value.  A wide format is used if necessary
        so that this is exact.  If an IEEEdouble suffices, that format is used.'''
        size = value.bit_length()
        fmt = IEEEdouble if size <= IEEEdouble.precision else cls.from_precision(size)
        return fmt.from_int(value)

    def from_int(self, value, context=None):
        '''Return the integer value converted to this format, rounding if necessary.'''
        if not isinstance(value, int):
            raise TypeError('from_int requires an integer')
        op_tuple = (OP_FROM_INT, value)
        return self._normalize(value < 0, 0, abs(value), op_tuple, context)

    def from_float(self, value, context=None):
        '''Return the float value converted to this format, rounding if necessary.'''
        if not isinstance(value, float):
            raise TypeError('from_float requires a float')
        result = IEEEdouble_from_float(value)
        if self is IEEEdouble:
            return result
        op_tuple = (OP_FROM_FLOAT, value)
        return self._convert(result, op_tuple, context, True)

    def from_string(self, string, context=None):
        '''Convert a string to a rounded floating number of this format.'''
        if not isinstance(string, str):
            raise TypeError('from_string requires a string')

        if HEX_SIGNIFICAND_PREFIX.match(string):
            return self._from_hex_significand_string(string, context)
        return decimal_to_binary.convert(self, string, context)

    def _from_hex_significand_string(self, string, context):
        '''Convert a string with hexadecimal significand to a rounded floating number of this
        format.
        '''
        op_tuple = (OP_FROM_STRING, string)
        match = HEX_SIGNIFICAND_REGEX.match(string)
        if match is None:
            raise SyntaxError(f'invalid hexadecimal float: {string}')

        sign = string[0] == '-'
        groups = match.groups()
        exponent = int(groups[4])

        # If a fraction was specified, the integer and fraction parts are in groups[1],
        # groups[2].  If no fraction was specified the integer is in groups[3].
        assert groups[1] is not None or groups[3]
        if groups[1] is None:
            significand = int(groups[3], 16)
        else:
            fraction = groups[2].rstrip('0')
            significand = int((groups[1] + fraction) or '0', 16)
            exponent -= len(fraction) * 4

        return self._normalize(sign, exponent, significand, op_tuple, context)

    def to_string(self, value, text_format=None, context=None):
        '''Return text, with a hexadecimal significand for finite numbers, that is a
        representation of the floating point value converted to this format.  See
        TextFormat docstring for output control.  All signals are raised as appropriate,
        and zeroes are output with an exponent of 0.
        '''
        context = context or get_context()
        text_format = text_format or DefaultHexFormat
        op_tuple = (OP_TO_STRING, value)

        # Do a convert() operation but preserve sNaNs.  This step signals inexact if appropriate.
        value = self._convert(value, op_tuple, context, True)

        # Now output the value.
        if not value.is_finite():
            # This step signals if sNaNs are lost.
            return text_format.format_non_finite(value, op_tuple, context)
        return text_format.format_hex(value)

    def add(self, lhs, rhs, context=None):
        '''Return the sum LHS + RHS in this format.'''
        return self._add_sub((OP_ADD, lhs, rhs), lhs, rhs, False, context)

    def subtract(self, lhs, rhs, context=None):
        '''Return the difference LHS - RHS in this format.'''
        return self._add_sub((OP_SUBTRACT, lhs, rhs), lhs, rhs, True, context)

    def _add_sub(self, op_tuple, lhs, rhs, is_subtract, context):
        context = context or get_context()

        # Handle either being non-finite
        if lhs.e_biased == 0 or rhs.e_biased == 0:
            # Put the non-finite in LHS
            flipped = lhs.e_biased != 0
            if flipped:
                lhs, rhs = rhs, lhs

            if lhs.significand == 0:
                # infinity + finite -> infinity
                if rhs.is_finite():
                    return self.make_infinity(lhs.sign ^ (is_subtract and flipped))
                if rhs.significand == 0:
                    if is_subtract == (lhs.sign == rhs.sign):
                        # Subtraction of like-signed infinities is an invalid op
                        return InvalidAdd(op_tuple, self).signal(context)
                    # Addition of like-signed infinites preserves its sign
                    return self.make_infinity(lhs.sign)

            # Propagate the NaN in the LHS
            return self._propagate_nan(op_tuple, context)

        # Both operations are finite.  Determine if the operation on the absolute values
        # is effectively an addition or subtraction of shifted significands.
        is_sub = is_subtract ^ lhs.sign ^ rhs.sign
        sign = lhs.sign

        # How much the LHS significand needs to be shifted left for exponents to match
        lshift = lhs.exponent_int() - rhs.exponent_int()

        if is_sub:
            # Shift the significand with the greater exponent left until its effective
            # exponent is equal to the smaller exponent.  Then subtract them.
            if lshift >= 0:
                significand = (lhs.significand << lshift) - rhs.significand
                exponent = rhs.exponent_int()
            else:
                significand = (rhs.significand << -lshift) - lhs.significand
                exponent = lhs.exponent_int()
                sign = not sign
            # If the result is negative then we must flip the sign and significand
            if significand < 0:
                sign = not sign
                significand = -significand
        else:
            # Shift the significand with the greater exponent left until its effective
            # exponent is equal to the smaller exponent, then add them.  The sign is the
            # sign of the lhs.
            if lshift >= 0:
                significand = (lhs.significand << lshift) + rhs.significand
                exponent = rhs.exponent_int()
            else:
                significand = (rhs.significand << -lshift) + lhs.significand
                exponent = lhs.exponent_int()

        # If two numbers add exactly to zero, IEEE 754 decrees it is a positive zero
        # unless rounding to minus infinity.  However, regardless of rounding mode, adding
        # two like-signed zeroes (or subtracting opposite-signed ones) gives the sign of
        # the left hand zero.
        if not significand and (lhs.significand or rhs.significand or is_sub):
            sign = context.rounding == ROUND_FLOOR

        return self._normalize(sign, exponent, significand, op_tuple, context)

    def multiply(self, lhs, rhs, context=None):
        '''Returns the product of LHS and RHS in this format.'''
        context = context or get_context()
        op_tuple = (OP_MULTIPLY, lhs, rhs)

        if lhs.e_biased == 0 or rhs.e_biased == 0:
            # Ensure the LHS is special
            if lhs.e_biased:
                lhs, rhs = rhs, lhs

            if lhs.significand == 0:
                # infinity * zero -> invalid op
                if rhs.is_zero():
                    return InvalidMultiply(op_tuple, self).signal(context)
                # infinity * infinity -> infinity
                # infinity * finite-non-zero -> infinity
                if not rhs.is_nan():
                    return self.make_infinity(lhs.sign ^ rhs.sign)

            return self._propagate_nan(op_tuple, context)

        return self._multiply_finite(lhs, rhs, op_tuple, context)

    def _multiply_finite(self, lhs, rhs, op_tuple, context):
        '''Returns the product of two finite floating point numbers in this format.'''
        sign = lhs.sign ^ rhs.sign
        exponent = lhs.exponent_int() + rhs.exponent_int()
        return self._normalize(sign, exponent, lhs.significand * rhs.significand,
                               op_tuple, context)

    def divide(self, lhs, rhs, context=None):
        '''Return lhs / rhs in this format.'''
        context = context or get_context()
        op_tuple = (OP_DIVIDE, lhs, rhs)
        sign = lhs.sign ^ rhs.sign

        if lhs.e_biased:
            # LHS is finite.  Is RHS finite too?
            if rhs.e_biased:
                # Both are finite.  Handle zeroes.
                if rhs.significand == 0:
                    # 0 / 0 -> NaN
                    if lhs.significand == 0:
                        return InvalidDivide(op_tuple, self).signal(context)
                    # Finite / 0 -> Infinity
                    return DivideByZero(op_tuple, self.make_infinity(sign)).signal(context)

                return self._divide_finite(lhs, rhs, op_tuple, context)

            # RHS is NaN or infinity
            if rhs.significand == 0:
                # finite / infinity -> zero
                return self.make_zero(sign)

            # finite / NaN propagates the NaN.
        elif lhs.significand == 0:
            # LHS is infinity
            # infinity / finite -> infinity
            if rhs.is_finite():
                return self.make_infinity(sign)
            # infinity / infinity is an invalid op
            if rhs.significand == 0:
                return InvalidDivide(op_tuple, self).signal(context)
            # infinity / NaN propagates the NaN.

        return self._propagate_nan(op_tuple, context)

    def _divide_finite(self, lhs, rhs, op_tuple, context):
        '''Calculate LHS / RHS, where both are finite and RHS is non-zero.

        if operation is 'D' for division, returns the correctly-rounded result.  If the
        operation is 'R' for IEEE remainder, the quotient is rounded to the nearest
        integer (ties to even) and the remainder returned.
        '''
        sign = lhs.sign ^ rhs.sign

        lhs_sig = lhs.significand
        if lhs_sig == 0:
            return self.make_zero(sign)

        rhs_sig = rhs.significand
        assert rhs_sig
        lhs_exponent = lhs.exponent_int()
        rhs_exponent = rhs.exponent_int()

        # Shift the lhs significand left until it is greater than the significand of the RHS
        lshift = rhs_sig.bit_length() - lhs_sig.bit_length()
        if lshift >= 0:
            lhs_sig <<= lshift
            lhs_exponent -= lshift
        else:
            rhs_sig <<= -lshift
            rhs_exponent += lshift

        if lhs_sig < rhs_sig:
            lhs_sig <<= 1
            lhs_exponent -= 1

        assert lhs_sig >= rhs_sig

        # Long division.  Note by construction the quotient significand will have a
        # leading 1, i.e., it will contain precisely precision significant bits,
        # representing a value in [1, 2).
        bits = self.precision
        quot_sig = 0
        for _n in range(bits):
            if _n:
                lhs_sig <<= 1
            quot_sig <<= 1
            if lhs_sig >= rhs_sig:
                lhs_sig -= rhs_sig
                quot_sig |= 1

        assert 0 <= lhs_sig < rhs_sig

        exponent = lhs_exponent - rhs_exponent - (bits - 1)
        if lhs_sig == 0:              # LF_EXACTLY_ZERO
            pass
        elif lhs_sig * 2 < rhs_sig:   # LF_LESS_THAN_HALF
            quot_sig = (quot_sig << 2) + 1
            exponent -= 2
        elif lhs_sig * 2 == rhs_sig:  # LF_EXACTLY_HALF
            quot_sig = (quot_sig << 1) + 1
            exponent -= 1
        else:                         # LF_MORE_THAN_HALF
            quot_sig = (quot_sig << 2) + 3
            exponent -= 2

        return self._normalize(sign, exponent, quot_sig, op_tuple, context)

    def sqrt(self, value, context=None):
        '''Return sqrt(value) in this format.  It has a positive sign for all operands >= 0,
        except that sqrt(-0) shall be -0.'''
        context = context or get_context()
        op_tuple = (OP_SQRT, value)

        if value.e_biased == 0:
            if value.significand:
                # Propagate NaNs
                return self._propagate_nan(op_tuple, context)
            # -Inf -> invalid operation
            if value.sign:
                return InvalidSqrt(op_tuple, self).signal(context)
            return self.make_infinity(False)
        if not value.significand:
            # +0 -> +0, -0 -> -0
            return self.make_zero(value.sign)
        if value.sign:
            return InvalidSqrt(op_tuple, self).signal(context)

        # Value is non-zero, finite and positive.  Try and find a good initial estimate
        # with significand the square root of the target significand, and half the
        # exponent.  Shift the significand to improve the initial approximation (adjusting
        # the exponent to compensate) and ensure the exponent is even.
        nearest_context = Context()
        exponent = value.exponent_int()
        sig = value.significand
        precision_bump = max(0, 63 - sig.bit_length())
        if (exponent - precision_bump) & 1:
            precision_bump += 1
        exponent -= precision_bump
        sig <<= precision_bump

        # Newton-Raphson loop
        est = self._normalize(False, exponent // 2, floor(sqrt(sig)), op_tuple, nearest_context)
        n = 1
        while True:
            print(n, "EST", est.as_tuple(), est)
            assert est.significand

            nearest_context.flags = 0
            div = self.divide(value, est, nearest_context)
            print(n, "DIV", div.as_tuple(), nearest_context)

            if est.e_biased == div.e_biased:
                if abs(div.significand - est.significand) <= 1:
                    break
            elif (div.next_up(nearest_context).as_tuple() == est.as_tuple()
                    or est.next_up(nearest_context).as_tuple() == div.as_tuple()):
                break

            # est <- (est + div) / 2
            est = self.add(est, div, nearest_context)
            est = est.scaleb(-1, nearest_context)
            n += 1

        down_context = Context(rounding=ROUND_DOWN)
        # Ensure that the precise answer lies in [est, est.next_up()).
        if div.significand == est.significand:
            # Do we have an exact square root?
            if not (nearest_context.flags & Flags.INEXACT):
                if est.is_subnormal():
                    est = UnderflowExact(op_tuple, est).signal(context)
                return est

            est_squared = value.fmt.multiply(est, est, down_context)
            if est_squared.compare_ge(value, down_context):
                est = est.next_down(down_context)
        elif est.compare_ge(div, down_context):
            est = div

        print("EST", est.as_tuple(), est)

        # EST and EST.next_up() now bound the precise result.  Decide if we need to round
        # it up.  This is only difficult with non-directed roundings.
        if context.round_to_nearest():
            temp_fmt = BinaryFormat.from_triple(self.precision + 1, self.e_max, self.e_min)
            nearest_context.flags = 0
            half = temp_fmt.add(est, est.next_up(nearest_context), nearest_context)
            half = half.scaleb(-1, nearest_context)
            assert not (nearest_context.flags & Flags.INEXACT)
            down_context.flags = 0
            half_squared = value.fmt.multiply(half, half, down_context)
            compare = half_squared.compare(value, down_context)
            if compare == Compare.LESS_THAN:
                lost_fraction = LF_MORE_THAN_HALF
            elif compare == Compare.EQUAL:
                if down_context.flags & Flags.INEXACT:
                    lost_fraction = LF_LESS_THAN_HALF
                else:
                    lost_fraction = LF_EXACTLY_HALF
            else:
                lost_fraction = LF_LESS_THAN_HALF
        else:
            # Anything as long as non-zero
            lost_fraction = LF_EXACTLY_HALF

        # Now round with the lost fraction.  FIXME: next_up context and signals.
        result = est
        is_tiny = result.is_subnormal()
        if round_up(context.rounding, lost_fraction, False, bool(result.significand & 1)):
            result = result.next_up(nearest_context)

        if context.tininess_after:
            is_tiny = result.is_subnormal()

        return (UnderflowInexact if is_tiny else Inexact)(op_tuple, result).signal(context)

    def fma(self, lhs, rhs, addend, context=None):
        '''Return a fused multiply-add operation.  The result is lhs * rhs + addend correctly
        rounded to this format.'''
        op_tuple = (OP_FMA, lhs, rhs, addend)
        context = context or get_context()

        # FMA must signal no more than once for the entire operation; this precludes a
        # naive multiply followed by add.  It seems simplest to handle cases where
        # multiplication might signal an invalid operation early.  There are two: any
        # (signalling) NaN, and if the first two operands are some combination of zero and
        # infinity.
        if any(value.is_nan() for value in (lhs, rhs, addend)):
            return self._propagate_nan(op_tuple, context)

        # Note we cannot, in general, defer to the multiply below, because any substituted
        # result would then be added.  The FMA as a whole must be a single atomic
        # operation.
        if (lhs.is_zero() and rhs.is_infinite()) or (rhs.is_zero() and lhs.is_infinite()):
            return InvalidFMA(op_tuple, self).signal(context)

        # Perform the multiplication in a format where it is exact and there are no
        # subnormals.  Then there can be no signals.
        product_fmt = BinaryFormat.from_triple(lhs.fmt.precision + rhs.fmt.precision,
                                               lhs.fmt.e_max + rhs.fmt.e_max + 1,
                                               lhs.fmt.e_min - (lhs.fmt.precision - 1)
                                               + rhs.fmt.e_min - (rhs.fmt.precision - 1))
        product = product_fmt.multiply(lhs, rhs, context)
        return self._add_sub(op_tuple, product, addend, False, context)


BinaryFormat._converters = {
    int: BinaryFormat.from_int,
    float: BinaryFormat.from_float,
    str: BinaryFormat.from_string,
    bytes: BinaryFormat._unpack_value,
    bytearray: BinaryFormat._unpack_value,
    memoryview: BinaryFormat._unpack_value,
    Decimal: BinaryFormat.from_decimal,
    Fraction: BinaryFormat.from_fraction,
}


class Binary(namedtuple('Binary', 'fmt sign e_biased significand')):
    '''Internal Representation
       -----------------------

    Our internal representation stores the exponent as a non-negative number.  This
    requires adding a bias, which like in IEEE-754 is 1 - e_min.  Then the actual binary
    exponent of e_min is stored internally as 1.

    Subnormal numbers and zeroes are stored with an exponent of 1 (whereas IEEE-754 uses
    an exponent of zero).  They can be distinguished from normal numbers with an exponent
    of e_min because the integer bit is not set in the significand.  A significand of zero
    represents a zero floating point number.

    Our internal representation uses an exponent of 0 to represent infinities and NaNs
    (note IEEE interchange formats use e_max + 1).  The integer bit is always cleared and
    a payload of zero represents infinity.  The quiet NaN bit is the bit below the integer
    bit.

    The advantage of this representation is that the following holds for all finite
    numbers wthere significand is understood as having a single leading integer bit:

            value = (-1)^sign * significand * 2^(exponent-bias).

    and NaNs and infinites are easily tested for by comparing the exponent with zero.
    '''

    def  __new__(cls, fmt, sign, e_biased, significand):
        '''Validate and create a floating point number with the given format, sign, biased
        exponent and significand.
        '''
        if not isinstance(e_biased, int):
            raise TypeError('e_biased must be an integer')
        if not isinstance(significand, int):
            raise TypeError('significand must be an integer')
        if not 0 <= e_biased <= fmt.e_max + fmt.e_bias:
            raise ValueError(f'biased exponent {e_biased:,d} out of range')
        # The significand is an unsigned integer, or payload plus quiet bit for NaNs
        if e_biased == 0:
            if not 0 <= significand < fmt.int_bit:
                raise ValueError(f'significand {significand:,d} out of range for non-finite')
        else:
            if not 0 <= significand <= fmt.max_significand:
                raise ValueError(f'significand {significand:,d} out of range')
        return super().__new__(cls, fmt, sign, e_biased, significand)


    ##
    ## Non-computational operations.  These are never exceptional so simply return their
    ## non-floating-point results.
    ##

    def number_class(self):
        '''Return a string describing the class of the number.'''
        # Finite?
        if self.e_biased:
            if self.significand:
                if self.e_biased == 1 and self.significand < self.fmt.int_bit:
                    return '-Subnormal' if self.sign else '+Subnormal'
                # Normal
                return '-Normal' if self.sign else '+Normal'

            return '-Zero' if self.sign else '+Zero'

        if self.significand:
            return 'NaN' if self.significand & self.fmt.quiet_bit else 'sNaN'

        return '-Infinity' if self.sign else '+Infinity'

    def as_tuple(self):
        '''Returns a BinaryTuple: (sign, exponent, significand).

        Finite non-zero numbers have the magniture 2^exponent * significand (an integer).

        Zeroes have an exponent and significand of 0.  Infinities have an exponent of 'I'
        with significand zero, quiet NaNs an exponent of 'Q' and signalling NaNs an
        exponent of 'S' and in either case the significand is the payload (without the
        quiet bit).
        '''
        significand = self.significand
        if self.e_biased == 0:
            # NaNs and infinities
            if significand == 0:
                exponent = 'I'
            elif significand & self.fmt.quiet_bit:
                exponent = 'Q'
                significand -= self.fmt.quiet_bit
            else:
                exponent = 'S'
        elif significand == 0:
            # Zeroes
            exponent = 0
        else:
            exponent = self.exponent_int()

        return BinaryTuple(self.sign, exponent, significand)

    def pack(self, endianness=None):
        '''Packs this value to bytes of the given endianness.

        Endianness can be 'big' or 'little'.  If None, host-native endianness is used.'''
        significand = self.significand
        if self.e_biased == 0:
            ieee_exponent = self.fmt.e_max * 2 + 1
        elif significand < self.fmt.int_bit:
            ieee_exponent = 0
        else:
            ieee_exponent = self.e_biased
            significand -= self.fmt.int_bit
        return self.fmt.pack(self.sign, ieee_exponent, significand, endianness)

    def nan_payload(self):
        '''Returns the NaN payload.  Raises RuntimeError if the value is not a NaN.'''
        if not self.is_nan():
            raise RuntimeError('nan_payload called on non-NaN')
        return self.significand & (self.fmt.quiet_bit - 1)

    def is_negative(self):
        '''Return True if the sign bit is set.'''
        return self.sign

    def is_normal(self):
        '''Return True if the value is finite, non-zero and not denormal.'''
        return self.significand & self.fmt.int_bit

    def is_finite(self):
        '''Return True if the value is finite.'''
        return bool(self.e_biased)

    def is_zero(self):
        '''Return True if the value is zero regardless of sign.'''
        return self.e_biased == 1 and not self.significand

    def is_subnormal(self):
        '''Return True if the value is subnormal.'''
        return self.e_biased == 1 and 0 < self.significand < self.fmt.int_bit

    def is_infinite(self):
        '''Return True if the value is infinite.'''
        return self.e_biased == 0 and not self.significand

    def is_nan(self):
        '''Return True if this is a NaN of any kind.'''
        return self.e_biased == 0 and self.significand

    def is_qnan(self):
        '''Return True if and only if this is a quiet NaN.'''
        return self.e_biased == 0 and (self.significand & self.fmt.quiet_bit)

    def is_snan(self):
        '''Return True if and only if this is a signalling NaN.'''
        return self.is_nan() and not (self.significand & self.fmt.quiet_bit)

    def is_canonical(self):
        '''We only have canonical values.'''
        return True

    # Not in IEEE-754
    def is_finite_non_zero(self):
        '''Return True if the value is finite and non-zero.'''
        return self.e_biased and self.significand

    def radix(self):
        '''We're binary!'''
        return 2

    def exponent_int(self):
        '''Return the arithmetic exponent of our significand interpreted as an integer.'''
        return self.exponent() - (self.fmt.precision - 1)

    def exponent(self):
        '''Return the arithmetic exponent of our significand interpreted as a binary floating
        point number with a decimal point after the MSB.
        '''
        assert self.is_finite()
        return self.e_biased - self.fmt.e_bias

    ##
    ## Quiet computational operations
    ##

    def set_sign(self, sign):
        '''Retuns a copy of this number with the given sign.'''
        if self.sign is sign:
            return self
        return Binary(self.fmt, sign, self.e_biased, self.significand)

    def copy_abs(self):
        '''Return this value with sign False (positive), including for NaNs.'''
        return self.set_sign(False)

    def copy_negate(self):
        '''Return this value with the opposite sign, including for NaNs.'''
        return self.set_sign(not self.sign)

    def copy_sign(self, y):
        '''Return this value but with the sign of y.'''
        return self.set_sign(y.sign)

    def as_integer_ratio(self):
        '''Return a pair (n, d) of integers that represent the floating point value as a fraction
        in lowest terms and with a positive denominator.'''
        if self.e_biased == 0:
            if self.significand:
                raise ValueError('cannot convert a NaN to an integer ratio')
            raise OverflowError('cannot convert an infinity to an integer ratio')
        if self.significand == 0:
            return (0, 1)
        exp = self.exponent_int()
        significand = self.significand
        while exp < 0 and not (significand & 1):
            significand >>= 1
            exp += 1

        if exp >= 0:
            n, d = significand << exp, 1
        else:
            n, d = significand, 1 << -exp
        return (-n if self.sign else n), d

    def payload(self):
        '''Return the payload of a NaN as a non-negative floating point integer, or -1
        if not a NaN.'''
        try:
            payload = self.nan_payload()
        except RuntimeError:
            return self.fmt.make_one(True)
        bits = payload.bit_length()
        payload <<= self.fmt.precision - bits
        return Binary(self.fmt, False, self.fmt.e_bias + (bits - 1), payload)

    def _set_payload(self, is_signalling):
        '''Common implementation for set_payload and set_payload_signalling.'''
        def payload_from_int():
            # Sign must be positive and value finite
            if self.sign or self.e_biased == 0:
                return -1
            # Handle zero; it has a special e_biased value
            if self.significand == 0:
                return 0
            # Need to check if e_biased is in-range, and if so, if the integer is
            # in-range.
            sig_bits = 1 + self.e_biased - self.fmt.e_bias
            shift = self.fmt.precision - sig_bits
            if 0 <= shift < self.fmt.precision:
                payload = self.significand >> shift
                if self.significand == payload << shift:
                    return payload
            return -1

        payload = payload_from_int()
        if is_signalling <= payload < self.fmt.quiet_bit:
            return self.fmt.make_nan(False, is_signalling, payload)
        return self.fmt.make_zero(False)

    def set_payload(self):
        '''Return a NaN with our value as payload, if it is a non-negative floating point
        integer and in-range.  Otherwise return +0.'''
        return self._set_payload(False)

    def set_payload_signalling(self):
        '''Return a signalling NaN with our value as payload, if it is a non-negative floating
        point integer and in-range.  Otherwise return +0.'''
        return self._set_payload(True)

    ##
    ## General homogeneous computational operations.  Operands and result have the same
    ## format.
    ##

    def remainder(self, rhs, context=None):
        '''Set to the remainder when divided by rhs.

        If rhs != 0, the remainder is defined for finite operands as r = x - y * n, where
        n is the integer nearest the exact number x / y rounded with ROUND_HALF_EVEN.  The
        result is always exact, and if r is zero its sign shall be that of lhs.

        If rhs is zero or lhs is infinite, an invalid operation is returned if neither
        operand is a NaN.  If rhs is infinite then the result is lhs if it is finite.
        '''
        op_tuple = (OP_REMAINDER, self, rhs)
        _quot, rem = self._divmod(op_tuple, rhs, ROUND_HALF_EVEN, context)
        return rem

    def fmod(self, rhs, context=None):
        '''C99's fmod operation.  As for remainder, but rounding is with ROUND_DOWN.'''
        op_tuple = (OP_FMOD, self, rhs)
        _quot, rem = self._divmod(op_tuple, rhs, ROUND_DOWN, context)
        return rem

    def mod(self, rhs, context=None):
        '''Python's mod.  As for remainder, but rounding is with ROUND_FLOOR.'''
        op_tuple = (OP_MOD, self, rhs)
        _quot, rem = self._divmod(op_tuple, rhs, ROUND_FLOOR, context)
        return rem

    def floordiv(self, rhs, context=None):
        '''Python's floordiv operation.  The quotient of mod().'''
        # TODO - suppress remainder signals?
        op_tuple = (OP_FLOORDIV, self, rhs)
        quot, _rem = self._divmod(op_tuple, rhs, ROUND_FLOOR, context)
        return self.fmt.from_int(quot)

    def divmod(self, rhs, context=None):
        '''Python's divmod operation - floordiv and mod combined.  The quotient is returned as a
        floating point number of the same format.
        '''
        # TODO - are two sets of signals possible?
        quot, rem = self._divmod(OP_DIVMOD, rhs, ROUND_FLOOR, context)
        quot = self.fmt.from_int(quot)
        return quot, rem

    def _divmod(self, op_tuple, rhs, rounding, context):
        '''Common implementation of the various remainder operations.

        Returns a (quot, remainder) pair where quot is a Python int or a NaN.
        '''
        if self.fmt != rhs.fmt:
            raise ValueError(f'{op_tuple[0]} requires operands of the same format')

        context = context or get_context()
        quot_sign = self.sign ^ rhs.sign     # Sign of the quotient
        rem_sign = self.sign                 # Rounding up always flips rem_sign

        # NaNs?
        if self.is_nan() or rhs.is_nan():
            result = self.fmt._propagate_nan(op_tuple, context)
            return result, result

        # remainder (infinity, non-NaN) is an invalid operation
        if self.e_biased == 0:
            result = InvalidRemainder(op_tuple, self.fmt).signal(context)
            return result, result

        # remainder(finite, infinity) is the LHS and exact.  remainder(subnormal,
        # infinity) is the LHS and signals underflow
        if rhs.e_biased == 0:
            rem = self
            quot = 0
            if round_up(rounding, LF_LESS_THAN_HALF, quot_sign, False):
                rem = self.fmt.make_infinity(not rem_sign)
                quot = 1
            elif rem.is_subnormal():
                rem = UnderflowExact(op_tuple, rem).signal(context)
            return -quot if quot_sign else quot, rem

        # remainder (finite, zero) is an invalid operation
        if rhs.significand == 0:
            result = InvalidRemainder(op_tuple, self.fmt).signal(context)
            return result, result

        lhs_sig = self.significand
        if not lhs_sig:
            # Both operations return a zero of the same sign.  As this is a homogeneous
            # operation formats match so just return self.
            return 0, self
        rhs_sig = rhs.significand

        # Like divide, shift significands until rhs_sig <= lhs_sig < rhs_sig * 2 keeping
        # track of implied exponents.
        lhs_exponent = self.exponent_int()
        rhs_exponent = rhs.exponent_int()

        lshift = rhs_sig.bit_length() - lhs_sig.bit_length()
        if lshift >= 0:
            lhs_sig <<= lshift
            lhs_exponent -= lshift
        else:
            rhs_sig <<= -lshift
            rhs_exponent += lshift

        if lhs_sig < rhs_sig:
            lhs_sig <<= 1
            lhs_exponent -= 1

        # Now the quotient is lhs_sig / rhs_sig, which has integer bit of 1 and exponent
        # lhs_exponent - rhs_exponent.  However we only want to do the bit-by-bit division
        # operation until we've found the integer result, when we must stop to keep the
        # remainder.
        quot = 0
        bits = (lhs_exponent - rhs_exponent) + 1
        if bits <= 0:
            if bits == 0:
                if lhs_sig > rhs_sig:
                    lost_fraction = LF_MORE_THAN_HALF
                else:
                    lost_fraction = LF_EXACTLY_HALF
            else:
                lost_fraction = LF_LESS_THAN_HALF
            rem = self
            if round_up(rounding, lost_fraction, quot_sign, False):
                rem = self.fmt.subtract(self, rhs, context)
                quot = 1
            return -quot if quot_sign else quot, rem

        # Perform the division
        while True:
            quot <<= 1
            if lhs_sig >= rhs_sig:
                quot += 1
                lhs_sig -= rhs_sig
            bits -= 1
            if not (bits and lhs_sig):
                break
            lhs_sig <<= 1

        if lhs_sig * 2 > rhs_sig:
            lost_fraction = LF_MORE_THAN_HALF
        elif lhs_sig * 2 == rhs_sig:
            lost_fraction = LF_EXACTLY_HALF
        elif lhs_sig:
            lost_fraction = LF_LESS_THAN_HALF
        else:
            lost_fraction = LF_EXACTLY_ZERO

        # IEEE-754 decrees a remainder of zero shall have the sign of the LHS.  A
        # remainder of zero cannot round up, so this is satisfied.
        if round_up(rounding, lost_fraction, quot_sign, bool(quot & 1)):
            lhs_sig = rhs_sig - lhs_sig
            rem_sign = not rem_sign
            quot += 1

        # This is exact; signals underflow if appropriate
        rem = self.fmt._normalize(rem_sign, rhs_exponent, lhs_sig, op_tuple, context)
        return -quot if quot_sign else quot, rem

    ##
    ## logBFormat operations (logBFormat is an integer format)
    ##

    def scaleb(self, N, context=None):
        '''Return x * 2^N for integral values N, correctly-rounded. and in the format of
        x.  For non-zero values of N, scalb(±0, N) is ±0 and scalb(±∞) is ±∞.  For zero values
        of N, scalb(x, N) is x.
        '''
        if not isinstance(N, int):
            raise TypeError('scaleb requires an integer')
        context = context or get_context()
        op_tuple = (OP_SCALEB, self, N)

        if self.e_biased == 0:
            # NaNs and infinities are unchanged (but NaNs are made quiet)
            if self.significand:
                return self.fmt._propagate_nan(op_tuple, context)
            return self
        return self.fmt._normalize(self.sign, self.exponent_int() + N, self.significand,
                                   op_tuple, context)

    def _logb(self):
        '''A private helper function.'''
        if self.e_biased == 0:
            if self.significand == 0:
                return 'Inf'
            return 'NaN'

        if self.significand == 0:
            return 'Zero'

        return self.exponent() + self.significand.bit_length() - self.fmt.precision

    def logb_integral(self, context=None):
        '''Return the exponent e of x, a signed integer, when represented with infinite range and
        minimum exponent.  Thus 1 <= scalb(x, -logb(x)) < 2 when x is positive and finite.
        logb(1) is +0.  logb(NaN), logb(∞) and logb(0) return implementation-defined
        values outside the range ±2 * (emax + p - 1) and flag an invalid operation.
        '''
        result = self._logb()
        if isinstance(result, int):
            return result

        op_tuple = (OP_LOGB_INTEGRAL, self)
        context = context or get_context()

        if result == 'Inf':
            result = self.fmt.logb_inf
        elif result == 'NaN':
            result = self.fmt.logb_nan
        else:
            result = self.fmt.logb_zero

        return InvalidLogBIntegral(op_tuple, result).signal(context)

    def logb(self, context=None):
        '''Return the exponent e of x, a signed integer, when represented with infinite range and
        minimum exponent.  Thus 1 <= scalb(x, -logb(x)) < 2 when x is positive and finite.
        logb(1) is +0.  logb(NaN) is a NaN, logb(∞) is +∞, and logb(0) is -∞ and signals
        the divide-by-zero exception.
        '''
        context = context or get_context()
        op_tuple = (OP_LOGB, self)

        result = self._logb()
        if isinstance(result, int):
            return self.fmt._normalize(result < 0, 0, abs(result), op_tuple, context)

        if result == 'NaN':
            return self.fmt._propagate_nan(op_tuple, context)
        if result == 'Zero':
            return LogBZero(op_tuple, self.fmt.make_infinity(True)).signal(context)
        return self.fmt.make_infinity(False)

    ##
    ## General computational operations
    ##

    def next_up(self, context=None):
        '''Set to the smallest floating point value (unless operating on a positive infinity or
        NaN) that compares greater.
        '''
        return self.fmt._next_up(self, context, False)

    def next_down(self, context=None):
        '''Set to the greatest floating point value (unless operating on a negative infinity or
        NaN) that compares less.

        As per IEEE-754 next_down(x) = -next_up(-x)
        '''
        return self.fmt._next_up(self, context, True)

    def _to_int(self, rounding):
        '''Round our finite value to an integer, rounding as specified.

        Return a (abs_result, is_exact) pair.  The caller needs to apply our sign to the
        result.
        '''
        assert self.e_biased

        if self.significand == 0:
            return 0, True

        # Strategy: truncate the significand, capture the lost fraction, and round.
        is_exact = True
        value = self.significand
        rshift = -self.exponent_int()

        if rshift <= 0:
            # We're already a (large) integer if rshift is <= 0
            value <<= -rshift
        else:
            value, lost_fraction = shift_right(value, rshift)
            if lost_fraction != LF_EXACTLY_ZERO:
                is_exact = False
                if round_up(rounding, lost_fraction, self.sign, bool(value & 1)):
                    value += 1
        return value, is_exact

    def _round_to_integral(self, operation, rounding, context=None):
        '''Round this value to an integer, as per rounding.  The result is a floating point value
        of the same format.

        If the operation is OP_ROUND_TO_INTEGRAL_EXACT, and the operand was finite and not
        an integer, signal Inexact.
        '''
        if operation == OP_ROUND_TO_INTEGRAL:
            op_tuple = (operation, self, rounding)
        else:
            op_tuple = (operation, self)

        if self.e_biased == 0:
            # NaNs and infinities are unchanged (but NaNs are made quiet)
            if self.is_nan():
                return self.fmt._propagate_nan(op_tuple, context)
            return self

        result, is_exact = self._to_int(rounding)

        # This should not raise any signals - FIXME
        result = self.fmt._normalize(self.sign, 0, result, op_tuple, context)
        if operation == OP_ROUND_TO_INTEGRAL_EXACT and not is_exact:
            result = Inexact(op_tuple, result).signal(context)
        return result

    def round_to_integral(self, rounding, context=None):
        '''Return the value rounded to the nearest integer in the same format.  The rounding
        method is given explicitly; that in context is ignored.

        This only signal is SignallingNaNOperand.
        '''
        return self._round_to_integral(OP_ROUND_TO_INTEGRAL, rounding, context)

    def round_to_integral_exact(self, context=None):
        '''Return the value rounded to the nearest integer in the same BinaryFormat.  The rounding
        method is taken from context.

        This function signals SignallingNaNOperand, or INEXACT if the result is not the
        original value (i.e. it was not an integer).
        '''
        return self._round_to_integral(OP_ROUND_TO_INTEGRAL_EXACT, context.rounding, context)

    def _to_integer(self, operation, min_int, max_int, rounding, context=None):
        '''Convert this to an integer in the given inclusive range, as per rounding.  min_int and
        max_int are the range of the target integer format and must include zero.  If
        min_int == max_int == 0 then the target integer format is unbounded (as in
        Python).

        If the value is a NaN or infinity, or the result is out of the integer range,
        InvalidConvertToInteger is signalled.  For NaNs the default result is 0, for
        infinites max_int or min_int as appropriate.

        If operation is OP_CONVERT_TO_INTEGER_EXACT, and the operand was finite and not an
        integer, signal Inexact.
        '''
        if not (isinstance(min_int, int) and isinstance(max_int, int)):
            raise TypeError('min_int and max_int must be integers')
        if not min_int <= 0 <= max_int:
            raise ValueError('zero must lie between min_int and max_int')

        op_tuple = (operation, self, min_int, max_int, rounding)
        if self.e_biased == 0:
            is_invalid = True
            if self.significand == 0:
                result = min_int if self.sign else max_int
            else:
                result = 0   # NaN
        else:
            result, is_exact = self._to_int(rounding)
            result = -result if self.sign else result
            if max_int == min_int:
                is_invalid = False
            else:
                result, unclamped_result = min(max(result, min_int), max_int), result
                is_invalid = result != unclamped_result

        if is_invalid:
            return InvalidConvertToInteger(op_tuple, result).signal(context)
        if operation == OP_CONVERT_TO_INTEGER_EXACT and not is_exact:
            return Inexact(op_tuple, result).signal(context)
        return result

    def convert_to_integer(self, min_int, max_int, rounding, context=None):
        '''Return the value rounded to the nearest integer in the given range.  The rounding
        method is given explicitly; that in context is ignored.

        if max_int <= min_int then the integers are unbounded.

        NaNs, infinities and out-of-range values signal InvalidConvertToInteger.
        Inexact is not signalled.
        '''
        return self._to_integer(OP_CONVERT_TO_INTEGER, min_int, max_int, rounding, context)

    def convert_to_integer_exact(self, min_int, max_int, rounding, context=None):
        '''As for convert_to_integer, except that Inexact is signaled if the result is
        in-range and not numerically equal to the original value (i.e. it was not an
        integer).
        '''
        return self._to_integer(OP_CONVERT_TO_INTEGER_EXACT, min_int, max_int, rounding, context)

    ##
    ## Signalling computational operations
    ##

    def _compare_quiet(self, rhs, order_zeroes):
        '''Return LHS vs RHS as one of the four comparison constants.

        If order_zeroes is True then -0 compares less than +0.
        '''
        if self.e_biased == 0:
            # LHS is a NaN?
            if not self.significand:
                # LHS is infinity.
                if rhs.e_biased == 0:
                    # RHS Infinity?
                    if not rhs.significand:
                        # Comparing two infinities
                        if self.sign == rhs.sign:
                            return Compare.EQUAL
                        # RHS is finite or a differently-signed infinity
                        return Compare.LESS_THAN if self.sign else Compare.GREATER_THAN
                    # Fall through for NaN comparisons
                else:
                    # RHS finite
                    return Compare.LESS_THAN if self.sign else Compare.GREATER_THAN
            # Fall through for NaN comparisons
        elif rhs.e_biased == 0:
            # LHS is finite, RHS is not
            if not rhs.significand:
                # Finite vs infinity
                return Compare.GREATER_THAN if rhs.sign else Compare.LESS_THAN
            # Finite vs NaN - Fall through
        elif self.significand == rhs.significand == 0:
            # Zero vs Zero
            if order_zeroes and self.sign != rhs.sign:
                return Compare.LESS_THAN if self.sign else Compare.GREATER_THAN
            return Compare.EQUAL
        else:
            # Finite vs Finite, at least one non-zero.  If signs differ it's easy.  Also
            # get the either-is-a-zero case out the way as zeroes cannot have their
            # exponents compared.
            if self.sign != rhs.sign or not rhs.significand:
                return Compare.LESS_THAN if self.sign else Compare.GREATER_THAN
            if not self.significand:
                return Compare.GREATER_THAN if rhs.sign else Compare.LESS_THAN

            # Finally, two non-zero finite numbers with equal signs.  Compare the unbiased
            # exponents of their explicit integer-bit significands.
            exponent_diff = self.exponent() - rhs.exponent()
            if exponent_diff:
                if (exponent_diff > 0) ^ (self.sign):
                    return Compare.GREATER_THAN
                else:
                    return Compare.LESS_THAN

            # Exponents are the same.  We need to make their significands comparable.
            lhs_sig, rhs_sig = self.significand, rhs.significand
            length_diff = lhs_sig.bit_length() - rhs_sig.bit_length()
            if length_diff > 0:
                rhs_sig <<= length_diff
            elif length_diff < 0:
                lhs_sig <<= -length_diff

            # At last we have an apples-for-apples comparison
            if lhs_sig == rhs_sig:
                return Compare.EQUAL
            if (lhs_sig > rhs_sig) ^ (self.sign):
                return Compare.GREATER_THAN
            else:
                return Compare.LESS_THAN

        # Comparisons involving at least one NaN.
        return Compare.UNORDERED

    def _compare(self, rhs, context, signalling):
        '''Return LHS vs RHS as one of the four comparison constants.

        If signalling is True, comparisons of NaNs signal InvalidComparison if either is a
        NaN.  Otherwise SignallingNaNOperand may be signalled.
        '''
        result = self._compare_quiet(rhs, False)
        # Comparisons involving at least one NaN.
        if result == Compare.UNORDERED and (signalling or self.is_snan() or rhs.is_snan()):
            context = context or get_context()
            op_tuple = (OP_COMPARE, self, rhs)
            cls = InvalidComparison if signalling else SignallingNaNOperand
            result = cls(op_tuple, result).signal(context)
        return result

    def compare(self, rhs, context=None):
        '''Return LHS vs RHS as one of the four comparison constants.  A signalling NaN signals
        SignallingNaNOperand.'''
        return self._compare(rhs, context, False)

    def compare_signal(self, rhs, context=None):
        '''This operation is identical to the compare() method, except that any NaN operand
        signals InvalidComparison.'''
        return self._compare(rhs, context, True)

    def compare_eq(self, rhs, context=None):
        return self._compare(rhs, context, False) == Compare.EQUAL

    def compare_ne(self, rhs, context=None):
        return self._compare(rhs, context, False) != Compare.EQUAL

    def compare_gt(self, rhs, context=None):
        return self._compare(rhs, context, False) == Compare.GREATER_THAN

    def compare_ng(self, rhs, context=None):
        return self._compare(rhs, context, False) != Compare.GREATER_THAN

    def compare_ge(self, rhs, context=None):
        return self._compare(rhs, context, False) in (Compare.EQUAL, Compare.GREATER_THAN)

    def compare_lu(self, rhs, context=None):
        return self._compare(rhs, context, False) not in (Compare.EQUAL, Compare.GREATER_THAN)

    def compare_lt(self, rhs, context=None):
        return self._compare(rhs, context, False) == Compare.LESS_THAN

    def compare_nl(self, rhs, context=None):
        return self._compare(rhs, context, False) != Compare.LESS_THAN

    def compare_le(self, rhs, context=None):
        return self._compare(rhs, context, False) in (Compare.EQUAL, Compare.LESS_THAN)

    def compare_gu(self, rhs, context=None):
        return self._compare(rhs, context, False) not in (Compare.EQUAL, Compare.LESS_THAN)

    def compare_un(self, rhs, context=None):
        return self._compare(rhs, context, False) == Compare.UNORDERED

    def compare_or(self, rhs, context=None):
        return self._compare(rhs, context, False) != Compare.UNORDERED

    def compare_eq_signal(self, rhs, context=None):
        return self._compare(rhs, context, True) == Compare.EQUAL

    def compare_ne_signal(self, rhs, context=None):
        return self._compare(rhs, context, True) != Compare.EQUAL

    def compare_gt_signal(self, rhs, context=None):
        return self._compare(rhs, context, True) == Compare.GREATER_THAN

    def compare_ge_signal(self, rhs, context=None):
        return self._compare(rhs, context, True) in (Compare.EQUAL, Compare.GREATER_THAN)

    def compare_lt_signal(self, rhs, context=None):
        return self._compare(rhs, context, True) == Compare.LESS_THAN

    def compare_le_signal(self, rhs, context=None):
        return self._compare(rhs, context, True) in (Compare.EQUAL, Compare.LESS_THAN)

    def compare_ng_signal(self, rhs, context=None):
        return self._compare(rhs, context, True) != Compare.GREATER_THAN

    def compare_lu_signal(self, rhs, context=None):
        return self._compare(rhs, context, True) not in (Compare.EQUAL, Compare.GREATER_THAN)

    def compare_nl_signal(self, rhs, context=None):
        return self._compare(rhs, context, True) != Compare.LESS_THAN

    def compare_gu_signal(self, rhs, context=None):
        return self._compare(rhs, context, True) not in (Compare.EQUAL, Compare.LESS_THAN)

    def compare_total(self, rhs):
        '''The total order relation on a format.  A loose equivalent of <=.'''
        if self.fmt != rhs.fmt:
            raise ValueError('total order requires both operands have the same format')
        comp = self._compare_quiet(rhs, True)
        if comp == Compare.UNORDERED:
            # At least one is a NaN
            if rhs.is_nan():
                if self.is_nan():
                    if self.sign != rhs.sign:
                        return self.sign
                    if self.is_snan() != rhs.is_snan():
                        return self.is_snan() ^ self.sign
                    if self.nan_payload() < rhs.nan_payload():
                        return not self.sign
                    if self.nan_payload() > rhs.nan_payload():
                        return self.sign
                    return True
                return not rhs.sign
            return self.sign
        return comp != Compare.GREATER_THAN

    def compare_total_mag(self, rhs):
        '''Per IEEE-754, totalOrderMag(x, y) is totalOrder(abs(x), abs(y)).'''
        return self.copy_abs().compare_total(rhs.copy_abs())

    def _max_min(self, flags, rhs, context):
        '''Implementation of IEEE-754 2019's maximum and maximum_number operations.'''
        if self.fmt != rhs.fmt:
            raise ValueError(f'{flags.op_name()} requires both operands have the same format')

        if flags & MinMaxFlags.MAG:
            comp = self.copy_abs()._compare_quiet(rhs.copy_abs(), True)
            # If magnitudes are equal, compare without magnitudes
            if comp == Compare.EQUAL:
                comp = self._compare_quiet(rhs, True)
        else:
            comp = self._compare_quiet(rhs, True)

        if comp in {Compare.GREATER_THAN, Compare.EQUAL}:
            return self if flags & MinMaxFlags.MAX else rhs
        if comp == Compare.LESS_THAN:
            return rhs if flags & MinMaxFlags.MAX else self

        # Propagate NaNs if it not a NUM operation
        op_tuple = (flags.op_name(), self, rhs)
        if not flags & MinMaxFlags.NUM:
            return self.fmt._propagate_nan(op_tuple, context)

        # The result is the number; if both are NaNs return the leftmost.  Signals invalid
        # if either is signalling.
        if rhs.is_nan():
            result = self
        else:
            result = rhs
        if self.is_snan() or rhs.is_snan():
            if result.is_snan():
                result = result.fmt.make_nan(result.sign, False, result.nan_payload())
            result = SignallingNaNOperand(op_tuple, result).signal(context)
        return result

    def max(self, rhs, context):
        '''Implementation of IEEE-754 2019's maximum operation.'''
        return self._max_min(MinMaxFlags.MAX, rhs, context)

    def max_num(self, rhs, context):
        '''Implementation of IEEE-754 2019's maximum_number operation.'''
        return self._max_min(MinMaxFlags.MAX | MinMaxFlags.NUM, rhs, context)

    def max_mag(self, rhs, context):
        '''Implementation of IEEE-754 2019's maximum_magnitude operation.'''
        return self._max_min(MinMaxFlags.MAX | MinMaxFlags.MAG, rhs, context)

    def max_mag_num(self, rhs, context):
        '''Implementation of IEEE-754 2019's maximum_magnitude_number operation.'''
        return self._max_min(MinMaxFlags.MAX | MinMaxFlags.MAG | MinMaxFlags.NUM, rhs, context)

    def min(self, rhs, context):
        '''Implementation of IEEE-754 2019's minimum operation.'''
        return self._max_min(MinMaxFlags.MIN, rhs, context)

    def min_num(self, rhs, context):
        '''Implementation of IEEE-754 2019's minimum_number operation.'''
        return self._max_min(MinMaxFlags.MIN | MinMaxFlags.NUM, rhs, context)

    def min_mag(self, rhs, context):
        '''Implementation of IEEE-754 2019's minimum_magnitude operation.'''
        return self._max_min(MinMaxFlags.MIN | MinMaxFlags.MAG, rhs, context)

    def min_mag_num(self, rhs, context):
        '''Implementation of IEEE-754 2019's minimum_magnitude_number operation.'''
        return self._max_min(MinMaxFlags.MIN | MinMaxFlags.MAG | MinMaxFlags.NUM, rhs, context)

    def __repr__(self):
        return self.to_string()

    def __str__(self):
        return self.to_string()

    def to_string(self, text_format=None, context=None):
        '''Return text, with a hexadecimal significand for finite numbers, that is a
        representation of the floating point value.  See TextFormat docstring for output
        control.  All signals are raised as appropriate, and zeroes are output with an
        exponent of 0.
        '''
        return self.fmt.to_string(self, text_format, context)

    def to_decimal_string(self, precision=0, text_format=None, context=None):
        '''Return text, with a hexadecimal significand for finite numbers, that is a
        representation of the floating point value.  See TextFormat docstring for output
        control.  All signals are raised as appropriate, and zeroes are output with an
        exponent of 0.

        precision is the number of significant digits to output.  0 returns the floating
        point number printed in the shortest posible number of digits such that, when
        reading back with round-to-nearest it shall give the original floating point
        number.  -1 outputs as many digits as necessary to give the precise value.
        '''
        context = context or get_context()
        text_format = text_format or DefaultDecFormat
        op_tuple = (OP_TO_DECIMAL_STRING, self, precision)

        if self.is_finite():
            exponent, digits, is_inexact = self._to_decimal_parts(precision, context.rounding)
            if precision == 0:
                precision = self.fmt.decimal_precision - 1
            else:
                precision = len(digits)
            result = text_format.format_decimal(self.sign, exponent, digits, precision)
            if is_inexact:
                result = Inexact(op_tuple, result).signal(context)
            return result
        else:
            return text_format.format_non_finite(self, op_tuple, context)

    def _to_decimal_parts(self, precision, rounding):
        '''Returns a tuple (exponent, digits, is_inexact) for finite numbers.

        See "How to Print Floating-Point Numbers Accurately" by Steele and White, in
        particular Table 3.  This is an optimized implementation of their algorithm.
        '''
        if self.is_zero():
            return 0, '0', False

        if not self.is_finite():
            raise RuntimeError('value must be finite')

        e_p = self.exponent_int()
        R = self.significand << max(0, e_p)
        M = 1 << max(0, e_p)
        S = 1 << max(0, -e_p)

        # This loop is for negative exponents H. It scales R until divmod() delivers the
        # first significant digit.
        exponent = -1
        while R * 10 < S:
            exponent -= 1
            R *= 10
            M *= 10

        # This loop is for positive exponents H.  It scales S until digits can be reliably
        # delivered.  This addition of M here seems entirely superfluous.
        while 2 * R + M >= 2 * S:
            S *= 10
            exponent += 1

        if precision:
            # Precision is fixed or infinite.  Generate the digits.
            def gen_digits(count):
                nonlocal R, S
                while R and count:
                    U, R = divmod(R * 10, S)
                    yield U + 48
                    count -= 1

            digits = bytearray(gen_digits(precision))
            # Rounding
            if R:
                R *= 2
                if R < S:
                    lost_fraction = LF_LESS_THAN_HALF
                elif R == S:
                    lost_fraction = LF_EXACTLY_HALF
                else:
                    lost_fraction = LF_MORE_THAN_HALF

                # Handle rounding by bumping
                if round_up(rounding, lost_fraction, self.sign, bool(digits[-1] & 1)):
                    pos = len(digits)
                    while True:
                        pos -= 1
                        digits[pos] = (digits[pos] - 48 + 1) % 10 + 48
                        if digits[pos] != 48:
                            break
                        if pos == 0:
                            digits[pos] = 49
                            exponent += 1
                            break
        else:
            # Now the arithmetic value is R / S.  M is the value of one high-ulp, and
            # hence is always scaled alongside the significand remainder R.  A low-ulp is
            # almost always the same size as a high-ulp; the exception is when our value
            # is on an exponent boundary in which case a low-ulp is half the size of a
            # high-ulp.  M is used to detect when we can stop generating digits.  When the
            # remainder R is strictly less than half a low-ulp, or when it is strictly
            # greater than S less half a high-ulp we can stop generating digits.  At that
            # point round-to-nearest of the output is guaranteed to give our target value.
            # The 'strictly' condition can be removed if we are even, because then
            # round-to-even will round correctly.
            low_shift = 2 if self.significand == self.fmt.int_bit else 1
            is_even = (self.significand & 1) == 0

            # If precision is zero we need the sophisticated algorithm
            digits = bytearray()

            while True:
                U, R = divmod(R * 10, S)
                M *= 10
                # If we have equality with M then the decimal we output is exactly
                # half-an-ulp from the target value.  If the target value is even then
                # it will be rounded to and we can stop.
                low = (R << low_shift) < M + is_even
                high = 2 * (S - R) < M + is_even
                if low or high:
                    break
                digits.append(U + 48)

            # All code paths here are exercised by the to_decimal tests
            if low and not high:
                pass
            elif high and not low:
                U += 1
            elif 2 * R < S:
                pass
            elif 2 * R > S:
                U += 1
            else:
                U += (U & 1)
            digits.append(U + 48)

        digits = digits.decode()
        digits += '0' * (precision - len(digits))

        return exponent, digits, R != 0

    ##
    ## Python support - make it feel like a Python numeric data type.
    ##
    ## These operations are not quiet.
    ##

    def __abs__(self):
        '''Return this value with sign False (positive).'''
        if self.is_snan():
            op_tuple = (OP_ABS, self)
            return self.fmt._propagate_nan(op_tuple)
        return self.set_sign(False)

    def __neg__(self):
        '''Return this value with the opposite sign (unary minus).'''
        if self.is_snan():
            op_tuple = (OP_MINUS, self)
            return self.fmt._propagate_nan(op_tuple)
        return self.set_sign(not self.sign)

    def __pos__(self):
        '''Return this value (unary plus).'''
        if self.is_snan():
            op_tuple = (OP_PLUS, self)
            return self.fmt._propagate_nan(op_tuple)
        return self

    def __eq__(self, other):
        compare = compare_any_eq(self, other)
        if compare is None:
            return NotImplemented
        return compare == Compare.EQUAL

    def __ne__(self, other):
        compare = compare_any_eq(self, other)
        if compare is None:
            return NotImplemented
        return compare != Compare.EQUAL

    def __lt__(self, other):
        compare = compare_any(self, other)
        if compare is None:
            return NotImplemented
        return compare == Compare.LESS_THAN

    def __le__(self, other):
        compare = compare_any(self, other)
        if compare is None:
            return NotImplemented
        return compare in (Compare.EQUAL, Compare.LESS_THAN)

    def __ge__(self, other):
        compare = compare_any(self, other)
        if compare is None:
            return NotImplemented
        return compare in (Compare.EQUAL, Compare.GREATER_THAN)

    def __gt__(self, other):
        compare = compare_any(self, other)
        if compare is None:
            return NotImplemented
        return compare == Compare.GREATER_THAN

    def __bool__(self):
        return not self.is_zero()

    def __int__(self):
        # Same as __trunc__
        return self._to_integer(OP_CONVERT_TO_INTEGER, 0, 0, ROUND_DOWN)

    def __float__(self):
        value = IEEEdouble.convert(self)
        result, = unpack_double(value.pack())
        return result

    def __trunc__(self):
        return self._to_integer(OP_CONVERT_TO_INTEGER, 0, 0, ROUND_DOWN)

    def __floor__(self):
        return self._to_integer(OP_CONVERT_TO_INTEGER, 0, 0, ROUND_FLOOR)

    def __ceil__(self):
        return self._to_integer(OP_CONVERT_TO_INTEGER, 0, 0, ROUND_CEILING)

    def __round__(self, ndigits=None):
        '''If ndigits is None, round to an integer under ROUND_HALF_EVEN.  Otherwise round after
        ndigits binary digits with ROUND_HALF_EVEN and the result is floating-point.
        '''
        if ndigits is None:
            return self._to_integer(OP_CONVERT_TO_INTEGER, 0, 0, ROUND_HALF_EVEN)
        if not isinstance(ndigits, int):
            raise TypeError('ndigits must be an integer')

        if self.e_biased == 0:
            return self._round_to_integral(OP_ROUND_TO_INTEGRAL, ROUND_HALF_EVEN)
        ndigits = min(ndigits, -self.exponent_int())
        result = self.scaleb(ndigits)
        result = result._round_to_integral(OP_ROUND_TO_INTEGRAL, ROUND_HALF_EVEN)
        return result.scaleb(-ndigits)

    def __add__(self, other):
        other = self.fmt.convert_for_arith(other)
        if other is None:
            return NotImplemented
        return self.fmt.add(self, other)

    def __sub__(self, other):
        other = self.fmt.convert_for_arith(other)
        if other is None:
            return NotImplemented
        return self.fmt.subtract(self, other)

    def __mul__(self, other):
        other = self.fmt.convert_for_arith(other)
        if other is None:
            return NotImplemented
        return self.fmt.multiply(self, other)

    def __truediv__(self, other):
        other = self.fmt.convert_for_arith(other)
        if other is None:
            return NotImplemented
        return self.fmt.divide(self, other)

    def __mod__(self, other):
        other = self.fmt.convert_for_arith(other)
        if other is None:
            return NotImplemented
        return self.mod(self, other)

    def __floordiv__(self, other):
        other = self.fmt.convert_for_arith(other)
        if other is None:
            return NotImplemented
        return self.floordiv(self, other)

    def __divmod__(self, other):
        other = self.fmt.convert_for_arith(other)
        if other is None:
            return NotImplemented
        return self.divmod(other)

    def __radd__(self, other):
        return self.__add__(other)

    def __rsub__(self, other):
        other = self.fmt.convert_for_arith(other)
        if other is None:
            return NotImplemented
        return self.fmt.subtract(other, self)

    def __rmul__(self, other):
        return self.__mul__(other)

    def __rtruediv__(self, other):
        other = self.fmt.convert_for_arith(other)
        if other is None:
            return NotImplemented
        return self.fmt.divide(other, self)

    def __rmod__(self, other):
        other = self.fmt.convert_for_arith(other)
        if other is None:
            return NotImplemented
        return other.mod(self)

    def __rfloordiv__(self, other):
        other = self.fmt.convert_for_arith(other)
        if other is None:
            return NotImplemented
        return other.floordiv(self)

    def __rdivmod__(self, other):
        other = self.fmt.convert_for_arith(other)
        if other is None:
            return NotImplemented
        return other.divmod(self)

    def __hash__(self):
        '''Python hash.  Must hash equally to other types with the same value.'''
        if self.e_biased == 0:
            # Follow behaviour of the Decimal package for non-finite values
            if self.significand == 0:
                return -314159 if self.sign else 314159
            if self.significand & self.fmt.quiet_bit:
                return 0
            raise TypeError('cannot hash a signalling NaN')

        return hash(Fraction(*self.as_integer_ratio()))

#
# Core decimal-to-binary conversion logic
#

class DecimalToBinary:
    '''Core decimal-to-binary conversion logic.  Parameterize some values to allow finding
    of corner cases that require them.'''

    def __init__(self, debug=False, short_sig_err=1, scaling_err_pos=1, scaling_err_neg=1,
                 pow5_err=1, pow5_recip_err=2):
        self.debug = debug
        self.short_sig_err = short_sig_err
        self.scaling_err_pos = scaling_err_pos
        self.scaling_err_neg = scaling_err_neg
        self.pow5_err = pow5_err
        self.pow5_recip_err = pow5_recip_err

    def convert(self, fmt, string, context):
        '''Converts a string with a hexadecimal significand to a floating number of the
        required format.

        A quiet NaN with no specificed payload has payload 0, a signalling NaN has payload
        1.  If the specified payload is too wide to be stored without loss, the most
        significant bits are dropped.  If the resulting signalling NaN payload is 0 it
        becomes 1.  If the payload of the returned NaN does not equal the given payload
        INEXACT is flagged.
        '''
        match = DEC_FLOAT_REGEX.match(string)
        if match is None:
            raise SyntaxError(f'invalid floating point number: {string}')

        sign = string[0] == '-'
        groups = match.groups()
        context = context or get_context()
        op_tuple = (OP_FROM_STRING, string)

        # Decimal float?
        if groups[1] is not None:
            # Read the optional exponent first.  It is in groups[6].
            exponent = 0 if groups[6] is None else int(groups[6])

            # If a fraction was specified, the integer and fraction parts are in
            # groups[2], groups[3].  If no fraction was specified the integer is in
            # groups[4].
            assert groups[2] is not None or groups[4]

            # Get the integer and fraction strings
            if groups[2] is None:
                int_str, frac_str = groups[4], ''
            else:
                int_str, frac_str = groups[2], groups[3]

            # Combine them into sig_str removing all insignificant zeroes.  Viewing that
            # as an integer, calculate the exponent adjustment to the true decimal point.
            sig_str = int_str + frac_str.rstrip('0')
            exponent += len(int_str) - len(sig_str)
            sig_str = sig_str.lstrip('0') or '0'

            # Now the value is significand * 10^exponent.
            result, exc = self.try_many(sign, exponent, sig_str, fmt, context)
            if exc:
                return exc(op_tuple, result).signal(context)
            return result

        # groups[7] matches infinities
        if groups[7] is not None:
            return fmt.make_infinity(sign)

        # groups[9] matches NaNs.  groups[10] is 's', 'S' or ''; the s indicates a
        # signalling NaN.  The payload is in groups[11], and duplicated in groups[12] if
        # hex or groups[13] if decimal
        assert groups[9] is not None
        is_signalling = groups[10] != ''
        if groups[12] is not None:
            payload = int(groups[12], 16)
        elif groups[13] is not None:
            payload = int(groups[13])
        else:
            payload = int(is_signalling)
        return fmt.make_nan(sign, is_signalling, payload)

    def try_many(self, sign, exponent, sig_str, fmt, context):
        '''Return a pair(result, exc).  Result is the correctly-rounded binary value of

             (-1)^sign * int(sig_str) * 10^exponent

        and exc is the exception to signal (or None).'''
        # Exponent doesn't matter if zero
        if sig_str == '0':
            return fmt.make_zero(sign), None

        # Test for obviously over-large exponents
        frac_exp = exponent + len(sig_str)
        if (frac_exp - 1) * log2_10 >= fmt.e_max + 1:
            return fmt.make_overflow_value(context.rounding, sign), Overflow

        # Test for obviously over-small exponents
        if frac_exp * log2_10 <= fmt.e_min - fmt.precision:
            return fmt.make_underflow_value(context.rounding, sign, False), UnderflowInexact

        # Figure out what exponent range we must have for our intermediate calculations
        # Each is sig_conversion, final_result, pow5_conversion
        e_max = ceil(max(max(len(sig_str), frac_exp) * log2_10,
                         abs(exponent) * (log2_10) - 1)) - 1
        # Need to be able to represent the smallest number as normal
        e_min = min(-2, floor((frac_exp - 1) * log2_10))

        # The loops are expensive; optimistically the above starts with a low precision
        # and iteratively increases it until we can guarantee the answer is correctly
        # rounded.  Use precision a multiple of 64 bits with some room over the format
        # precision, and always an exponent range at least 1 larger - we eliminate
        # obviously out-of-range exponents above.  Intermediate calculations must not
        # overflow nor use subnormal numbers.
        precision = fmt.precision
        while True:
            if not self.debug:
                precision = (precision + 63) & ~63

            calc_fmt = BinaryFormat.from_triple(precision, e_max, e_min)

            result, exc = self.try_once(sign, exponent, sig_str, fmt, calc_fmt, context)
            if result is None:
                # The result and/or the signal to raise cannot be determined.  Increase
                # precision and retry.
                precision += 1 if self.debug else precision // 2
                continue

            return result, exc

    def try_once(self, sign, exponent, sig_str, fmt, calc_fmt, context):
        # We have done a calculation in whose lowest bits will be rounded.  We want to
        # know how far away the value of theese rounded bits is, in ULPs, from the
        # rounding boundary - the boundary is where the rounding changes the value, so
        # that we can determine if it is safe to round now.  Directed rounding never
        # changes so the boundary has all zeroes with the next MSB 0 or 1.
        # Round-to-nearest has a boundary of a half - i.e. 1 followed by bits-1 zeroes.
        def ulps_from_boundary(significand, bits, context):
            assert bits >= 0
            boundary = 1 << bits
            rounded_bits = significand & (boundary - 1)
            if context.round_to_nearest():
                boundary >>= 1
                return abs(boundary - rounded_bits)
            else:
                return min(rounded_bits, boundary - rounded_bits)

        # Numbers that will be subnormal round more bits
        bits_to_round = calc_fmt.precision - fmt.precision

        # With this many digits, an increment of one is strictly less than one ULP in
        # the binary format.
        digit_count = min(calc_fmt.decimal_precision, len(sig_str))

        # We want to calculate significand * 10^sig_exponent.  sig_exponent may differ
        # from exponent because not all sig_str digits are used.
        significand = int(sig_str[:digit_count])
        sig_exponent = exponent + (len(sig_str) - digit_count)

        # All err variables are upper bounds and in half-ULPs
        if digit_count < len(sig_str):
            # The error is strictly less than a half-ULP if we round based on the next digit
            if int(sig_str[digit_count]) >= 5:
                significand += 1
            sig_err = self.short_sig_err
        else:
            sig_err = 0

        calc_context = Context(rounding=ROUND_HALF_EVEN)
        sig = calc_fmt._normalize(sign, 0, significand, None, calc_context)
        if calc_context.flags & Flags.INEXACT:
            sig_err += 1
        calc_context.flags &= ~Flags.INEXACT

        pow5_int = pow(5, abs(sig_exponent))
        pow5 = calc_fmt._normalize(False, 0, pow5_int, None, calc_context)
        if calc_context.flags & Flags.INEXACT:
            pow5_err = self.pow5_err
        else:
            pow5_err = 0
        calc_context.flags &= ~Flags.INEXACT

        # Call scaleb() since we scaled by 5^n and actually want 10^n
        if sig_exponent >= 0:
            scaled_sig = calc_fmt._multiply_finite(sig, pow5, None, calc_context)
            scaled_sig = scaled_sig.scaleb(sig_exponent, calc_context)
            if calc_context.flags & Flags.INEXACT:
                scaling_err = self.scaling_err_pos
            else:
                scaling_err = 0
        else:
            scaled_sig = calc_fmt._divide_finite(sig, pow5, None, calc_context)
            scaled_sig = scaled_sig.scaleb(sig_exponent, calc_context)
            if calc_context.flags & Flags.INEXACT:
                scaling_err = self.scaling_err_neg
            else:
                scaling_err = 0
            # Lemma 2 requires a normal number
            assert not scaled_sig.is_subnormal()
            # Numbers that will be subnormal round more bits
            bits_to_round += max(0, fmt.e_min - scaled_sig.exponent())
            # An extra half-ulp is lost in reciprocal of pow5.  FIXME: prove.  No test
            # fails if this condition is removed.
            if pow5_err or scaling_err:
                pow5_err = self.pow5_recip_err

        assert calc_context.flags & Flags.OVERFLOW == 0

        # The error from the true value, in half-ulps, on multiplying two floating point
        # numbers, which differ from the value they approximate by at most HUE1 and HUE2
        # half-ulps, is strictly less than err (when non-zero).
        #
        # See Lemma 2 in "How to Read Floating Point Numbers Accurately" by William D
        # Clinger.  It assumes the product of normalized numbers with a normalized result.
        if sig_err + pow5_err == 0:
            # If there is a scaling error it is at most 1 half-ULP, which is < 2
            err = scaling_err * 2
        else:
            # Note that 0 <= sig_err <= 2, 0 <= pow5_err <= 2, and 0 <= scaling_err <=
            # 1.  If sig_err is 2 it is actually strictly less than 2.  Hance per the
            # lemma the error is strictly less than this.
            err = scaling_err + 2 * (sig_err + pow5_err)

        rounding_distance = 2 * ulps_from_boundary(scaled_sig.significand, bits_to_round,
                                                   context)

        # If we round now are we guaranteed to round correctly?
        if err <= rounding_distance:
            # Convert with context's rounding and tininess detection
            convert_context = Context(rounding=context.rounding,
                                      tininess_after=context.tininess_after)
            result = fmt.convert(scaled_sig, convert_context)

            # It remains to determine exactness.  Overflow is inexact; exit early.
            if convert_context.flags & Flags.OVERFLOW:
                return result, Overflow

            # Work out distance of the result from our estimate; if too close we need
            # more precision to determine exactness.  It's simplest to look at the
            # rounded bits than at the shifted significand which can change, e.g. from
            # 0xffff to 0x8000, when rounding.
            rounded_bits = scaled_sig.significand & ((1 << bits_to_round) - 1)
            exact_distance = min(rounded_bits, (1 << bits_to_round) - rounded_bits)

            # Guaranteed inexact?
            if err < exact_distance:
                if convert_context.flags & Flags.UNDERFLOW:
                    return result, UnderflowInexact
                return result, Inexact

            # Guaranteed exact?
            if err == 0:
                if result.is_subnormal():
                    return result, UnderflowExact
                return result, None

            # We can't be sure as to exactness - loop again with more precision

        # We can't be sure as to the result
        return None, None


#
# Useful internal helper routines
#

def lost_bits_from_rshift(significand, bits):
    '''Return what the lost bits would be were the significand shifted right the given number
    of bits (negative is a left shift).
    '''
    if bits <= 0:
        return LF_EXACTLY_ZERO
    # Prevent over-large shifts consuming memory
    bits = min(bits, significand.bit_length() + 2)
    bit_mask = 1 << (bits - 1)
    first_bit = bool(significand & bit_mask)
    second_bit = bool(significand & (bit_mask - 1))
    return first_bit * 2 + second_bit


def shift_right(significand, bits):
    '''Return the significand shifted right a given number of bits (left if bits is negative),
    and the fraction that is lost doing so.
    '''
    if bits <= 0:
        result = significand << -bits
    else:
        result = significand >> bits

    return result, lost_bits_from_rshift(significand, bits)


def round_up(rounding, lost_fraction, sign, is_odd):
    '''Return True if, when an operation is inexact, the result should be rounded up (i.e.,
    away from zero by incrementing the significand).

    sign is the sign of the number, and is_odd indicates if the LSB of the new
    significand is set, which is needed for ties-to-even rounding.
    '''
    if lost_fraction == LF_EXACTLY_ZERO:
        return False

    if rounding == ROUND_HALF_EVEN:
        if lost_fraction == LF_EXACTLY_HALF:
            return is_odd
        else:
            return lost_fraction == LF_MORE_THAN_HALF
    elif rounding == ROUND_CEILING:
        return not sign
    elif rounding == ROUND_FLOOR:
        return sign
    elif rounding == ROUND_DOWN:
        return False
    elif rounding == ROUND_UP:
        return True
    elif rounding == ROUND_HALF_DOWN:
        return lost_fraction == LF_MORE_THAN_HALF
    else:
        return lost_fraction != LF_LESS_THAN_HALF


def IEEEdouble_from_float(value):
    '''Return an IEEEdouble converted from a Python float value.'''
    return IEEEdouble.unpack_value(pack_double(value))


def compare_any_eq(value, other):
    '''LHS is a Binary.  RHS is any type.  Accept complex comparisons for == and !=.'''
    if isinstance(other, complex):
        if not other.imag:
            return compare_any(value, other.real)
        # Only used for == and != so this works.
        return Compare.UNORDERED

    return compare_any(value, other)


def compare_any(value, other):
    '''LHS is a Binary.  RHS is any type.'''
    if isinstance(other, Binary):
        return value.compare(other)
    if isinstance(other, float):
        return value.compare(IEEEdouble_from_float(other))
    if isinstance(other, Decimal):
        # Deal with infinites and NaNs only here.  Let finite decimals fall through.
        if other.is_nan():
            return Compare.UNORDERED
        if other.is_infinite():
            return value.compare(IEEEdouble.make_infinity(other.is_signed()))
    if isinstance(other, (Decimal, Fraction, int)):
        # other is finite.  Handle non-finite cases for us; they compare the same as to zero.
        if value.e_biased == 0:
            return value.compare(_positive_zero)

        # Both are finite.  We want an infinitely precise comparison so we must not
        # convert to Binary.  Compare both sides as fractions.
        a, b = value.as_integer_ratio()
        if isinstance(other, Decimal):
            c, d = other.as_integer_ratio()
        elif isinstance(other, int):
            c, d = other, 1
        else:
            c, d = other.numerator, other.denominator
        diff = a * d - b * c
        if diff > 0:
            return Compare.GREATER_THAN
        if diff < 0:
            return Compare.LESS_THAN
        return Compare.EQUAL

    return None


#
# Exported functions
#

DefaultContext = Context()
DefaultContext.set_handler((Invalid, DivisionByZero, Overflow), HandlerKind.RAISE)
tls = threading.local()


def get_context():
    try:
        return tls.context
    except AttributeError:
        tls.context = DefaultContext.copy()
        return tls.context


def set_context(context):
    '''Sets the current thread's context to context (not a copy of it).'''
    tls.context = context


class LocalContext:
    '''A context manager that will set the current context for the active thread to a copy of
    context on entry to the with-statement and restore the previous context on exit.  If
    no context is specified a copy of the current context is taken instead.
    '''

    def __init__(self, context=None):
        self.saved_context = None
        self.context_to_set = context

    def __enter__(self):
        self.saved_context = get_context()
        context = (self.context_to_set or self.saved_context).copy()
        set_context(context)
        return context

    def __exit__(self, etype, value, traceback):
        set_context(self.saved_context)


local_context = LocalContext

#
# Constants are predefined formats.
#

log2_10 = log2(10)
host_endianness = 'little' if Struct('<d').pack(-0.0)[0] == 0 else 'big'

IEEEhalf = BinaryFormat.from_IEEE(16)
IEEEsingle = BinaryFormat.from_IEEE(32)
IEEEdouble = BinaryFormat.from_IEEE(64)
IEEEquad = BinaryFormat.from_IEEE(128)

# 80387 floating point takes place with a wide exponent range but rounds to single, double
# or extended precision.  It also has an explicit integer bit.
x87extended = BinaryFormat.from_pair(64, 15)
x87double = BinaryFormat.from_pair(53, 15)
x87single = BinaryFormat.from_pair(24, 15)

decimal_to_binary = DecimalToBinary()
_positive_zero = IEEEdouble.make_zero(False)

HEX_SIGNIFICAND_PREFIX = re.compile('[-+]?0x', re.ASCII | re.IGNORECASE)
HEX_SIGNIFICAND_REGEX = re.compile(
    # sign[opt] hex-sig-prefix
    '[-+]?0x'
    # (hex-integer[opt].fraction or hex-integer.[opt])
    '(([0-9a-f]*)\\.([0-9a-f]+)|([0-9a-f]+)\\.?)'
    # p exp-sign[opt]dec-exponent
    'p([-+]?[0-9]+)$',
    re.ASCII | re.IGNORECASE
)
DEC_FLOAT_REGEX = re.compile(
    # sign[opt]
    '[-+]?('
    # (dec-integer[opt].fraction or dec-integer.[opt])
    '(([0-9]*)\\.([0-9]+)|([0-9]+)\\.?)'
    # e sign[opt]dec-exponent   [opt]
    '(e([-+]?[0-9]+))?|'
    # inf or infinity
    '(inf(inity)?)|'
    # nan-or-snan hex-payload-or-dec-payload[opt]
    '((s?)nan((0x[0-9a-f]+)|([0-9]+))?))$',
    re.ASCII | re.IGNORECASE
)
