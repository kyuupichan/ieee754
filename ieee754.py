#
# An implementation of many operations of generic binary floating-point arithmetic
#
# (c) Neil Booth 2007-2021.  All rights reserved.
#

import math
import re
from collections import namedtuple
from enum import IntFlag, IntEnum

import attr


__all__ = ('Context', 'BinaryFormat', 'IntegerFormat', 'Flags', 'Binary', 'TextFormat', 'Compare',
           'DivisionByZero', 'Inexact', 'InvalidOperation', 'InvalidOperationInexact',
           'Overflow', 'Subnormal', 'SubnormalExact', 'SubnormalInexact', 'Underflow',
           'ROUND_CEILING', 'ROUND_FLOOR', 'ROUND_DOWN', 'ROUND_UP',
           'ROUND_HALF_EVEN', 'ROUND_HALF_UP', 'ROUND_HALF_DOWN',
           'IEEEhalf', 'IEEEsingle', 'IEEEdouble', 'IEEEquad',
           'x87extended', 'x87double', 'x87single',)


# Rounding modes
ROUND_CEILING   = 'ROUND_CEILING'       # Towards +infinity
ROUND_FLOOR     = 'ROUND_FLOOR'         # Towards -infinity
ROUND_DOWN      = 'ROUND_DOWN'          # Torwards zero
ROUND_UP        = 'ROUND_UP'            # Away from zero
ROUND_HALF_EVEN = 'ROUND_HALF_EVEN'     # To nearest with ties towards even
ROUND_HALF_DOWN = 'ROUND_HALF_DOWN'     # To nearest with ties towards zero
ROUND_HALF_UP   = 'ROUND_HALF_UP'       # To nearest with ties away from zero


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


# Operation status flags.
class Flags(IntFlag):
    INVALID     = 0x01
    DIV_BY_ZERO = 0x02
    OVERFLOW    = 0x04
    SUBNORMAL   = 0x08
    UNDERFLOW   = 0x10
    INEXACT     = 0x20


@attr.s(slots=True)
class TextFormat:
    '''Controls the output of conversion to decimal and hexadecimal strings.'''
    # The minimum number of digits to show in the exponent of a finite number.  For
    # decimal output, 0 suppresses the exponent (adding trailing or leading zeroes if
    # necessary as per the printf 'f' format specifier), and if negative applies the rule
    # for the printf 'g' format specifier to decide whether to display an exponent or not,
    # in which case it has at minimum the absolute value number of digits.  Hexadecimal
    # output always has an exponent so negative values are treated as 1.
    exp_digits = attr.ib(default=1)
    # If True positive exponents display a '+'.
    force_exp_sign = attr.ib(default=False)
    # If True, numbers with a clear sign bit are preceded with a '+'.
    force_leading_sign = attr.ib(default=False)
    # If True, display a floating point even if there are no digits after it.  For
    # example, "5.", "1.e2" and "0x1.p2".
    force_point = attr.ib(default=False)
    # If True, the exponent character ('p' or 'e') is in upper case, and for hexadecimal
    # output, the hex indicator 'x' and hexadecimal digits are in upper case.  For NaN
    # output, hexadecimal payloads are in upper case.  This does not affect the text of
    # the inf, qNaN and sNaN indicators below which are copied unmodified.
    upper_case = attr.ib(default=False)
    # If True, trailing insignificant zeroes are stripped
    rstrip_zeroes = attr.ib(default=False)
    # The string output for infinity
    inf = attr.ib(default='Inf')
    # The string output for quiet NaNs
    qNaN = attr.ib(default='NaN')
    # The string output for signalling NaNs.  The empty string means flag an invalid
    # operation and output as a quiet NaN instead.
    sNaN = attr.ib(default='sNaN')
    # Controls the display of NaN payloads.  If N, NaN payloads are omitted.  If X, they
    # are output in hexadecimal.  If 'D' in decimal.  Examples of all 3 formats are: nan,
    # nan255 and nan0xff for a quiet NaN with payload 255, respectively.
    nan_payload = attr.ib(default='X')

    def leading_sign(self, value):
        '''Returns the leading sign string to output for value.'''
        return '-' if value.sign else '+' if self.force_leading_sign else ''

    def non_finite_string(self, value):
        '''Returns the output text for infinities and NaNs without a sign.'''
        assert value.e_biased == 0

        # Infinity
        if value.significand == 0:
            special = self.inf
        else:
            # NaNs
            special = self.sNaN if value.is_signalling() else self.qNaN
            if self.nan_payload == 'D':
                special += str(value.NaN_payload())
            elif self.nan_payload == 'X':
                special += hex(value.NaN_payload())

        return self.leading_sign(value) + special

    def format_decimal(self, sign, sig_digits, exponent):
        '''sign is True if the number has a negative sign.  sig_digits is a string of significant
        digits of a number converted to decimal.  exponent is the exponent of the leading
        digit, i.e. the decimal point appears exponent digits after the leading digit.
        '''
        if self.rstrip_zeroes:
            sig_digits = sig_digits.rstrip('0') or '0'

        assert len(sig_digits) > 0

        parts = []
        if sign:
            parts.append('-')
        elif self.force_leading_sign:
            parts.append('+')

        exp_digits = self.exp_digits
        if exp_digits < 0:
            # Apply the fprintf 'g' format specifier rule
            if max(len(sig_digits), 4) > exponent >= -4:
                exp_digits = 0
            else:
                exp_digits = -exp_digits

        if exp_digits:
            if len(sig_digits) > 1 or self.force_point:
                parts.extend((sig_digits[0], '.', sig_digits[1:]))
            else:
                parts.append(sig_digits)
            parts.append('e')
            if exponent < 0:
                parts.append('-')
            elif self.force_exp_sign:
                parts.append('+')
            exp_str = f'{abs(exponent):d}'
            parts.append('0' * (exp_digits - len(exp_str)))
            parts.append(exp_str)
        else:
            point = exponent + 1
            if point <= 0:
                parts.extend(('0.', '0' * -point, sig_digits))
            else:
                if point > len(sig_digits):
                    sig_digits += (point - len(sig_digits)) * '0'
                if point < len(sig_digits) or self.force_point:
                    parts.extend((sig_digits[:point], '.', sig_digits[point:]))
                else:
                    parts.append(sig_digits)

        result = ''.join(parts)
        if self.upper_case:
            result = result.upper()

        return result

    def to_hex(self, value):
        if not value.is_finite():
            return self.non_finite_string(value)

        # The value has been converted and rounded to our format.  We output the
        # unbiased exponent, but have to shift the significand left up to 3 bits so
        # that converting the significand to hex has the integer bit as an MSB
        significand = value.significand
        if significand == 0:
            exponent = 0
        else:
            significand <<= (value.fmt.precision & 3) ^ 1
            exponent = value.exponent()

        output_digits = (value.fmt.precision + 6) // 4
        hex_sig = f'{significand:x}'
        # Prepend zeroes to get the full output precision
        hex_sig = '0' * (output_digits - len(hex_sig)) + hex_sig
        # Strip trailing zeroes?
        if self.rstrip_zeroes:
            hex_sig = hex_sig.rstrip('0') or '0'
        # Insert the decimal point only if there are trailing digits
        if len(hex_sig) > 1:
            hex_sig = hex_sig[0] + '.' + hex_sig[1:]

        sign = self.leading_sign(value)
        exp_sign = '+' if exponent >= 0 and self.force_exp_sign else ''
        return f'{sign}0x{hex_sig}p{exp_sign}{exponent:d}'


BinaryTuple = namedtuple('BinaryTuple', 'sign exponent significand')


# When precision is lost during a calculation this indicates what fraction of the LSB the
# lost bits represented.  It essentially combines the roles of 'guard' and 'sticky' bits.
class LostFraction(IntEnum):   # Example of truncated bits:
    EXACTLY_ZERO = 0           # 000000
    LESS_THAN_HALF = 1	       # 0xxxxx  x's not all zero
    EXACTLY_HALF = 2           # 100000
    MORE_THAN_HALF = 3         # 1xxxxx  x's not all zero


# Four-way result of the compare() operation.
class Compare(IntEnum):
    LESS_THAN = 0
    EQUAL = 1
    GREATER_THAN = 2
    UNORDERED = 3


def lost_bits_from_rshift(significand, bits):
    '''Return what the lost bits would be were the significand shifted right the given number
    of bits (negative is a left shift).
    '''
    if bits <= 0:
        return LostFraction.EXACTLY_ZERO
    # Prevent over-large shifts consuming memory
    bits = min(bits, significand.bit_length() + 2)
    bit_mask = 1 << (bits - 1)
    first_bit = bool(significand & bit_mask)
    second_bit = bool(significand & (bit_mask - 1))
    return LostFraction(first_bit * 2 + second_bit)


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
    if lost_fraction == LostFraction.EXACTLY_ZERO:
        return False

    if rounding == ROUND_HALF_EVEN:
        if lost_fraction == LostFraction.EXACTLY_HALF:
            return is_odd
        else:
            return lost_fraction == LostFraction.MORE_THAN_HALF
    elif rounding == ROUND_CEILING:
        return not sign
    elif rounding == ROUND_FLOOR:
        return sign
    elif rounding == ROUND_DOWN:
        return False
    elif rounding == ROUND_UP:
        return True
    elif rounding == ROUND_HALF_DOWN:
        return lost_fraction == LostFraction.MORE_THAN_HALF
    else:
        assert rounding == ROUND_HALF_UP
        return lost_fraction != LostFraction.LESS_THAN_HALF


class BinaryError(ArithmeticError):
    '''All traps subclass from this.'''


class DivisionByZero(BinaryError, ZeroDivisionError):
    '''Division by zero.'''


class Inexact(BinaryError):
    '''The result cannot be represented precisely.'''


class InvalidOperation(BinaryError):
    '''Invalid operation, e.g. 0 / 0 or any non-quiet operation on a signallying NaN.'''


class InvalidOperationInexact(InvalidOperation, Inexact):
    '''An operation on a signalling NaN results in a quiet NaN with a different payload.'''


class Overflow(Inexact):
    '''If after rounding the result would have an exponent exceeding e_max.  The result is
    either infinity or the signed finite value of greatest magnitude, depending on the
    rounding mode and sign.'''


class Subnormal(BinaryError):
    '''Before rounding the result had an exponent less than e_min.'''


class SubnormalExact(Subnormal):
    '''The result is exact and has an exponent less than e_min.'''


class SubnormalInexact(Inexact, Subnormal):
    '''Before rounding the result was inexact and had an exponent less than e_min.'''


class Underflow(SubnormalInexact):
    '''The result is inexact and has an exponent less than e_min, or is zero, after rounding.'''


flag_map = {
    Flags.INEXACT: Inexact,
    Flags.INVALID: InvalidOperation,
    Flags.INVALID | Flags.INEXACT: InvalidOperationInexact,
    Flags.DIV_BY_ZERO: DivisionByZero,
    Flags.OVERFLOW | Flags.INEXACT: Overflow,
    Flags.SUBNORMAL: SubnormalExact,
    Flags.SUBNORMAL | Flags.INEXACT: SubnormalInexact,
    Flags.UNDERFLOW | Flags.SUBNORMAL | Flags.INEXACT: Underflow,
}


class Context:
    '''The execution context for operations.  Carries the rounding mode, status flags and
    traps.
    '''

    __slots__ = ('rounding', 'flags', 'traps')

    def __init__(self, rounding=None, flags=None, traps=None):
        '''rounding is one of the ROUND_ constants and controls rounding of inexact results.'''
        self.rounding = rounding or ROUND_HALF_EVEN
        self.flags = flags or 0
        self.traps = traps or 0

    def clear_flags(self):
        self.flags = 0

    def clear_traps(self):
        self.traps = 0

    def copy(self):
        '''Return a copy of the context.'''
        return Context(self.rounding, self.flags, self.traps)

    def set_flags(self, flags):
        self.flags |= flags
        if self.traps & flags:
            raise flag_map[flags]

    def on_normalized_finite(self, is_tiny_before, is_tiny_or_zero_after, is_inexact):
        '''Call after normalisation of a finite number to set flags appropriately.'''
        if is_tiny_or_zero_after:
            if is_inexact:
                self.set_flags(Flags.UNDERFLOW | Flags.SUBNORMAL | Flags.INEXACT)
            else:
                self.set_flags(Flags.SUBNORMAL)
        elif is_tiny_before:
            assert is_inexact
            self.set_flags(Flags.SUBNORMAL | Flags.INEXACT)
        elif is_inexact:
            self.set_flags(Flags.INEXACT)

    def overflow_value(self, binary_format, sign):
        '''Call when an overflow occurs on a number of the given format and sign, because
        the rounded result would have too large an exponent.

        Flags overflow and returns the result, which is signed and either the largest
        finite number or infinity depending on the rounding mode.'''
        self.set_flags(Flags.OVERFLOW | Flags.INEXACT)

        rounding = self.rounding
        if rounding in {ROUND_HALF_EVEN, ROUND_HALF_DOWN, ROUND_HALF_UP, ROUND_UP}:
            is_up = True
        elif rounding == ROUND_CEILING:
            is_up = not sign
        elif rounding == ROUND_FLOOR:
            is_up = sign
        else: # ROUND_DOWN
            is_up = False

        if is_up:
            return binary_format.make_infinity(sign)
        return binary_format.make_largest_finite(sign)

    def round_to_nearest(self):
        '''Return True if the rounding mode rounds to nearest (ignoring ties).'''
        return self.rounding in {ROUND_HALF_EVEN, ROUND_HALF_DOWN, ROUND_HALF_UP}

    def __repr__(self):
        return f'Context(rounding={self.rounding}, flags={self.flags!r} traps={self.traps!r})'


class IntegerFormat:
    '''A two's-complement integer, either signed or unsigned.'''

    __slots__ = ('width', 'is_signed', 'min_int', 'max_int')

    def __init__(self, width, is_signed):
        if not isinstance(width, int):
            raise TypeError('width must be an integer')
        if width < 1:
            raise TypeError('width must be at least 1')
        self.width = width
        self.is_signed = is_signed
        if is_signed:
            self.min_int = -(1 << (width - 1))
            self.max_int = (1 << (width - 1)) - 1
        else:
            self.min_int = 0
            self.max_int = (1 << width) - 1

    def clamp(self, value):
        '''Clamp the value to being in-range.  Return a (clamped_value, was_clamped) pair.'''
        assert isinstance(value, int)
        if value > self.max_int:
            return self.max_int, True
        if value < self.min_int:
            return self.min_int, True
        return value, False


class BinaryFormat:
    '''An IEEE-754 binary floating point format.  Does not require e_min = 1 - e_max.'''

    __slots__ = ('precision', 'e_max', 'e_min', 'e_bias',
                 'int_bit', 'quiet_bit', 'max_significand', 'fmt_width',
                 'logb_inf', 'logb_zero', 'logb_NaN')

    def __init__(self, precision, e_max, e_min):
        '''precision is the number of bits in the significand including an explicit integer bit.

        e_max is largest e such that 2^e is representable; the largest representable
        number is then 2^e_max * (2 - 2^(1 - precision)) when the significand is all ones.

        e_min is the smallest e such that 2^e is not a subnormal number.  The smallest
        subnormal number is then 2^(e_min - (precision - 1)).

        Internal Representation
        -----------------------

        Our internal representation stores the exponent as a non-negative number.  This
        requires adding a bias, which like in IEEE-754 is 1 - e_min.  Then the actual
        binary exponent of e_min is stored internally as 1.

        Subnormal numbers and zeroes are stored with an exponent of 1 (whereas IEEE-754
        uses an exponent of zero).  They can be distinguished from normal numbers with an
        exponent of e_min because the integer bit is not set in the significand.  A
        significand of zero represents a zero floating point number.

        Our internal representation uses an exponent of 0 to represent infinities and NaNs
        (note IEEE interchange formats use e_max + 1).  The integer bit is always cleared
        and a payload of zero represents infinity.  The quiet NaN bit is the bit below the
        integer bit.

        The advantage of this representation is that the following holds for all finite
        numbers wthere significand is understood as having a single leading integer bit:

                  value = (-1)^sign * significand * 2^(exponent-bias).

        and NaNs and infinites are easily tested for by comparing the exponent with zero.

        A binary format is permitted to be an interchange format if all the following are
        true:

           a) e_min = 1 - e_max

           b) e_max + 1 is a power of 2.  This means that IEEE-754 interchange format
              exponents use all values in an exponent field of with e_width bits.

           c) The number of bits (1 + e_width + precision), i.e. including the sign bit,
              is a multiple of 16, so that the format can be written as an even number of
              bytes.  Alternatively if (2 + e_width + precision) is a multiple of 16, then
              the format is presumed to have an explicit integer bit (as per Intel x87
              extended precision numbers).
        '''
        self.precision = precision
        self.e_max = e_max
        self.e_min = e_min

        # Store these as handy pre-calculated values
        self.e_bias = 1 - e_min
        self.int_bit = 1 << (precision - 1)
        self.quiet_bit = 1 << (precision - 2)
        self.max_significand = (1 << precision) - 1

        # What integer value logb(inf) returns.  IEEE-754 requires this and the values for
        # logb(0) and logb(NaN) be values "outside the range ±2 * (emax + p - 1)".
        self.logb_inf = 2 * (max(e_max, abs(e_min)) + precision - 1) + 1
        self.logb_zero = -self.logb_inf
        self.logb_NaN = self.logb_zero - 1

        # Are we an interchange format?  If not, fmt_width is zero, otherwise it is the
        # format width in bits (the sign bit, the exponent and the significand including
        # the integer bit).  Only interchange formats can be packed to and unpacked from
        # binary bytes.
        self.fmt_width = 0
        if e_min == 1 - e_max and (e_max + 1) & e_max == 0:
            e_width = e_max.bit_length() + 1
            fmt_width = 1 + e_width + precision   # With integer bit
            if 0 <= fmt_width % 16 <= 1:
                self.fmt_width = fmt_width

    @classmethod
    def from_exponent_width(cls, precision, e_width):
        e_max = (1 << (e_width - 1)) - 1
        return cls(precision, e_max, 1 - e_max)

    def make_zero(self, sign):
        '''Returns a zero of the given sign.'''
        return Binary(self, sign, 1, 0)

    def make_infinity(self, sign):
        '''Returns an infinity of the given sign.'''
        return Binary(self, sign, 0, 0)

    def make_largest_finite(self, sign):
        '''Returns the finite number of maximal magnitude with the given sign.'''
        return Binary(self, sign, self.e_max + self.e_bias, self.max_significand)

    def make_NaN(self, sign, is_quiet, payload, context, flag_invalid):
        '''Return a NaN with the given sign and payload; the NaN is quiet iff is_quiet.

        If flag_invalid is true, invalid operation is flagged.  If payload bits are lost,
        or payload is 0 and is_quiet is False, then INEXACT is flagged.
        '''
        if payload < 0:
            raise ValueError(f'NaN payload cannot be negative: {payload}')
        mask = self.quiet_bit - 1
        adj_payload = payload & mask
        if is_quiet:
            adj_payload |= self.quiet_bit
        else:
            adj_payload = max(adj_payload, 1)
        flags = Flags.INEXACT if adj_payload & mask != payload else 0
        if flag_invalid:
            flags |= Flags.INVALID
        context.set_flags(flags)
        return Binary(self, sign, 0, adj_payload)

    def _propagate_NaN(self, nan, other, context):
        '''Call to get the result of a binary operation with at least one NaN.  The NaN is
        converted to a quiet one for this format, and invalid_op is flagged if either
        this, or the other operand, is a signalling NaN.
        '''
        assert nan.is_NaN()

        is_invalid = nan.is_signalling() or other.is_signalling()
        return self.make_NaN(nan.sign, True, nan.NaN_payload(), context, is_invalid)

    def _invalid_op_NaN(self, context):
        '''Call when an invalid operation happens.  Returns a canonical quiet NaN.'''
        return self.make_NaN(False, True, 0, context, True)

    def make_real(self, sign, exponent, significand, context,
                  lost_fraction=LostFraction.EXACTLY_ZERO):
        '''Return a floating point number that is the correctly-rounded (by the context) value of
        the infinitely precise result

           ± 2^exponent * (significand plus lost_fraction)

        of the given sign, where lost_fraction ULPs have been lost.

        For example,

           make_real(IEEEsingle, True, -3, 2) -> -0.75
           make_real(IEEEsingle, True, 1, -200) -> +0.0 and sets the UNDERFLOW and INEXACT flags.
        '''
        # Return a correctly-signed zero
        if significand == 0:
            return self.make_zero(sign)

        size = significand.bit_length()

        # Shifting the significand so the MSB is one gives us the natural shift.  There t
        # is followed by a decimal point, so the exponent must be adjusted to compensate.
        # However we cannot fully shift if the exponent would fall below e_min.
        exponent += self.precision - 1
        rshift = max(size - self.precision, self.e_min - exponent)

        # Shift the significand and update the exponent
        significand, shift_lf = shift_right(significand, rshift)
        exponent += rshift

        # Santity check - a lost fraction loses information if we needed to shift left
        assert rshift >= 0 or lost_fraction == LostFraction.EXACTLY_ZERO

        # If we shifted, combine the lost fractions; shift_lf is the more significant
        if rshift:
            lost_fraction = LostFraction(shift_lf | (lost_fraction != LostFraction.EXACTLY_ZERO))

        is_tiny_before = significand < self.int_bit
        is_inexact = lost_fraction != LostFraction.EXACTLY_ZERO

        # Round
        if round_up(context.rounding, lost_fraction, sign, bool(significand & 1)):
            # Increment the significand
            significand += 1
            # If the significand now overflows, halve it and increment the exponent
            if significand > self.max_significand:
                significand >>= 1
                exponent += 1

        # If the new exponent would be too big, then we overflow.  The result is either
        # infinity or the format's largest finite value depending on the rounding mode.
        if exponent > self.e_max:
            return context.overflow_value(self, sign)

        is_tiny_after = significand < self.int_bit

        context.on_normalized_finite(is_tiny_before, is_tiny_after, is_inexact)

        return Binary(self, sign, exponent + self.e_bias, significand)

    def _next_up(self, value, context, flip_sign):
        '''Return the smallest floating point value (unless operating on a positive infinity or
        NaN) that compares greater than the one whose parts are given.

        Signs are flipped on read and write if flip_sign is True.
        '''
        assert value.fmt is self
        sign = value.sign ^ flip_sign
        e_biased = value.e_biased
        significand = value.significand

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
            if 0 < significand < self.int_bit:
                context.set_flags(Flags.SUBNORMAL)
        else:
            # Negative infinity becomes largest negative number; positive infinity unchanged
            if significand == 0:
                if sign:
                    significand = self.max_significand
                    e_biased = self.e_max + self.e_bias
            # Signalling NaNs are converted to quiet.
            elif significand < self.quiet_bit:
                significand |= self.quiet_bit
                context.set_flags(Flags.INVALID)

        return Binary(self, sign ^ flip_sign, e_biased, significand)

    def convert(self, value, context):
        '''Return the value converted to this format and rounding if necessary.

        If value is a signalling NaN it is converted to a quiet NaN and invalid operation
        is flagged.
        '''
        if value.e_biased:
            # Avoid expensive normalisation to same format; copy and check for subnormals
            if value.fmt is self:
                if value.is_subnormal():
                    context.set_flags(Flags.SUBNORMAL)
                return value.copy()

            if value.significand:
                return self.make_real(value.sign, value.exponent_int(), value.significand, context)

            # Zeroes
            return self.make_zero(value.sign)

        if value.significand:
            # NaNs
            return self._propagate_NaN(value, value, context)

        # Infinities
        return self.make_infinity(value.sign)

    def pack(self, sign, exponent, significand, endianness='little'):
        '''Packs the IEEE parts of a floating point number as bytes of the given endianness.

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
        return value.to_bytes(self.fmt_width // 8, endianness)

    def unpack(self, binary, endianness='little'):
        '''Decode a binary encoding and return a BinaryTuple.

        Exponent is the biased exponent in the IEEE sense, i.e., it is zero for zeroes and
        subnormals and e_max * 2 + 1 for NaNs and infinites.  significand does not include
        the integer bit.
        '''
        if not self.fmt_width:
            raise RuntimeError('not an interchange format')
        size = self.fmt_width // 8
        if len(binary) != size:
            raise ValueError(f'expected {size} bytes to unpack; got {len(binary)}')
        value = int.from_bytes(binary, endianness)

        # Extract the parts from the encoding
        implicit_integer_bit = self.fmt_width % 8 == 1
        significand = value & (self.int_bit - 1)
        value >>= self.precision - implicit_integer_bit
        exponent = value & (self.e_max * 2 + 1)
        sign = value != exponent

        return BinaryTuple(sign, exponent, significand)

    def unpack_value(self, binary, endianness='little'):
        '''Decode a binary encoding and return a Binary floating point value.'''
        sign, exponent, significand = self.unpack(binary, endianness)
        if exponent == 0:
            exponent = 1
        elif exponent == self.e_max * 2 + 1:
            exponent = 0
        else:
            significand += self.int_bit

        return Binary(self, sign, exponent, significand)

    def from_string(self, string, context):
        '''Convert a string to a rounded floating number of this format.'''
        if HEX_SIGNIFICAND_PREFIX.match(string):
            return self._from_hex_significand_string(string, context)
        return self._from_decimal_string(string, context)

    def _from_hex_significand_string(self, string, context):
        '''Convert a string with hexadecimal significand to a rounded floating number of this
        format.
        '''
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

        return self.make_real(sign, exponent, significand, context)

    def _from_decimal_string(self, string, context):
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
            int_str = int_str.lstrip('0')
            sig_str = (int_str + frac_str).rstrip('0') or '0'
            exponent += len(int_str) - len(sig_str)

            # Now the value is significand * 10^exponent.
            return self._decimal_to_binary(sign, exponent, sig_str, context)

        # groups[7] matches infinities
        if groups[7] is not None:
            return self.make_infinity(sign)

        # groups[9] matches NaNs.  groups[10] is 's', 'S' or ''; the s indicates a
        # signalling NaN.  The payload is in groups[11], and duplicated in groups[12] if
        # hex or groups[13] if decimal
        assert groups[9] is not None
        is_quiet = not groups[10]
        if groups[12] is not None:
            payload = int(groups[12], 16)
        elif groups[13] is not None:
            payload = int(groups[13])
        else:
            payload = 1 - is_quiet
        return self.make_NaN(sign, is_quiet, payload, context, False)

    def _decimal_to_binary(self, sign, exponent, sig_str, context):
        '''Return a correctly-rounded binary value of

             (-1)^sign * int(sig_str) * 10^exponent
        '''
        # We have done a calculation in whose lowest bits will be rounded.  We want to
        # know how far away the value of theese rounded bits is, in ULPs, from the
        # rounding boundary - the boundary is where the rounding changes the value, so
        # that we can determine if it is safe to round now.  Directed rounding never
        # changes so the boundary has all zeroes with the next MSB 0 or 1.
        # Round-to-nearest has a boundary of a half - i.e. 1 followed by bits-1 zeroes.
        def ulps_from_boundary(significand, bits, context):
            assert bits > 0
            boundary = 1 << bits
            rounded_bits = significand & (boundary - 1)
            if context.round_to_nearest():
                boundary >>= 1
                return abs(boundary - rounded_bits)
            else:
                return min(rounded_bits, boundary - rounded_bits)

        # Test for obviously over-large exponents
        frac_exp = exponent + len(sig_str)
        if frac_exp - 1 >= (self.e_max + 1) / math.log2(10):
            return context.overflow_value(self, sign)

        # Test for obviously over-small exponents
        if frac_exp < (self.e_min - self.precision) / math.log2(10):
            context.set_flags(Flags.UNDERFLOW | Flags.SUBNORMAL | Flags.INEXACT)
            return self.make_zero(sign)

        # Start with a precision a multiple of 64 bits with some room over the format
        # precision, and always an exponent range 1 bit larger - we eliminate obviously
        # out-of-range exponents above.  Our intermediate calculations must not overflow
        # nor use subnormal numbers.
        parts_count = (self.precision + 10) // 64 + 1
        e_width = (max(self.e_max, abs(self.e_min)) * 2 + 2).bit_length() + 1

        while True:
            # The loops are expensive; optimistically the above starts with a low
            # precision and iteratively increases it until we can guarantee the answer is
            # correctly rounded.  Perform this loop in this format.
            calc_fmt = BinaryFormat.from_exponent_width(parts_count * 64, e_width)
            bits_to_round = calc_fmt.precision - self.precision

            # With this many digits, an increment of one is strictly less than one ULP in
            # the binary format.
            digit_count = min(math.floor(calc_fmt.precision / math.log2(10)) + 2, len(sig_str))

            # We want to calculate significand * 10^sig_exponent.  sig_exponent may differ
            # from exponent because not all sig_str digits are used.
            significand = int(sig_str[:digit_count])
            sig_exponent = exponent + (len(sig_str) - digit_count)

            # All err variables are upper bounds and in half-ULPs
            if digit_count < len(sig_str):
                # The error is strictly less than a half-ULP if we round based on the next digit
                if int(sig_str[digit_count]) >= 5:
                    significand += 1
                sig_err = 1
            else:
                sig_err = 0

            calc_context = Context(ROUND_HALF_EVEN)
            sig = calc_fmt.make_real(sign, 0, significand, calc_context)
            if calc_context.flags & Flags.INEXACT:
                sig_err += 1
            calc_context.flags &= ~Flags.INEXACT

            pow5_int = pow(5, abs(sig_exponent))
            pow5 = calc_fmt.make_real(False, 0, pow5_int, calc_context)
            pow5_err = 1 if calc_context.flags & Flags.INEXACT else 0
            calc_context.flags &= ~Flags.INEXACT

            # Call scaleb() since we scaled by 5^n and actually want 10^n
            if sig_exponent >= 0:
                scaled_sig = calc_fmt._multiply_finite(sig, pow5, calc_context)
                scaled_sig = scaled_sig.scaleb(sig_exponent, calc_context)
                scaling_err = 1 if calc_context.flags & Flags.INEXACT else 0
            else:
                scaled_sig = calc_fmt._divide_finite(sig, pow5, 'D', calc_context)
                scaled_sig = scaled_sig.scaleb(sig_exponent, calc_context)
                scaling_err = 1 if calc_context.flags & Flags.INEXACT else 0
                # If the exponent is below our e_min, the number is subnormal, and so
                # during convert() more bits are rounded
                bits_to_round += max(0, self.e_min - scaled_sig.exponent())
                # An extra half-ulp is lost in reciprocal of pow5.  FIXME: verify
                if pow5_err or scaling_err:
                    pow5_err = 2

            assert calc_context.flags & (Flags.SUBNORMAL | Flags.OVERFLOW) == 0

            # The error from the true value, in half-ulps, on multiplying two floating
            # point numbers, which differ from the value they approximate by at most HUE1
            # and HUE2 half-ulps, is strictly less than err (when non-zero).
            #
            # See Lemma 2 in "How to Read Floating Point Numbers Accurately" by William D
            # Clinger.
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
                convert_context = context.copy()
                result = self.convert(scaled_sig, convert_context)

                # Now work out distance of the result from our estimate; if it's too
                # close we need to try harder to determine if the result is exact
                exact_distance = abs((result.significand << bits_to_round)
                                     - scaled_sig.significand)

                # Guaranteed inexact?
                if err < exact_distance:
                    convert_context.flags |= Flags.INEXACT
                    break

                # Guaranteed exact?
                if err == 0:
                    convert_context.flags &= ~Flags.INEXACT
                    break

                # We can't be sure as to exactness - loop again with more precision

            # Increase precision and try again
            parts_count *= 2

        context.set_flags(convert_context.flags)
        return result

    ##
    ## General computational operations.  The operand(s) can be different formats;
    ## the destination format is self.
    ##

    def to_hex(self, value, text_format, context):
        '''Return text, with a hexadecimal significand for finite numbers, that is a
        representation of the floating point value converted to this format.  See the
        docstring of OutputSpec for output control.

        After rounding, normal numbers have a hex integer digit of 1, subnormal numbers 0.
        Zeroes are output with an exponent of 0.

        All flags / traps are raised as appropriate.
        '''
        # convert() converts NaNs to quiet; avoid that if we preserve them
        if value.is_NaN():
            # Rather ugly code to handle the myriad of cases
            src_payload = value.NaN_payload()
            src_signalling = value.is_signalling()

            if text_format.nan_payload == 'N':
                payload = 0
            else:
                payload = src_payload & (self.quiet_bit - 1)

            if src_signalling and text_format.sNaN:
                payload = max(payload, 1)
                flags = 0 if payload == src_payload else Flags.INEXACT
            else:
                flags = 0 if payload == src_payload else Flags.INEXACT
                payload |= self.quiet_bit
                if src_signalling:
                    flags |= Flags.INVALID

            context.set_flags(flags)
            value = Binary(self, value.sign, 0, payload)
        else:
            value = self.convert(value, context)

        return text_format.to_hex(value)

    def from_int(self, x, context):
        '''Return the integer x converted to this floating point format, rounding if necessary.'''
        return self.make_real(x < 0, 0, abs(x), context)

    def add(self, lhs, rhs, context):
        '''Return the sum LHS + RHS in this format.'''
        return self._add_sub(lhs, rhs, False, context)

    def subtract(self, lhs, rhs, context):
        '''Return the difference LHS - RHS in this format.'''
        return self._add_sub(lhs, rhs, True, context)

    def _add_sub(self, lhs, rhs, is_sub, context):
        if lhs.e_biased == 0:
            return self._add_sub_special(lhs, rhs, is_sub, False, context)
        if rhs.e_biased == 0:
            return self._add_sub_special(rhs, lhs, is_sub, True, context)
        return self._add_sub_finite(lhs, rhs, is_sub, context)

    def _add_sub_special(self, lhs, rhs, is_sub, flipped, context):
        '''Return a lhs * rhs where the LHS is a NaN or infinity.'''
        assert lhs.e_biased == 0

        if lhs.significand == 0:
            # infinity + finite -> infinity
            if rhs.is_finite():
                return self.make_infinity(lhs.sign ^ (is_sub and flipped))
            if rhs.significand == 0:
                if is_sub == (lhs.sign == rhs.sign):
                    # Subtraction of like-signed infinities is an invalid op
                    return self._invalid_op_NaN(context)
                # Addition of like-signed infinites preserves its sign
                return self.make_infinity(lhs.sign)
            # infinity +- NaN propagates the NaN
            lhs, rhs = rhs, lhs

        # Propagate the NaN in the LHS
        return self._propagate_NaN(lhs, rhs, context)

    def _add_sub_finite(self, lhs, rhs, is_sub, context):
        # Determine if the operation on the absolute values is effectively an addition
        # or subtraction of shifted significands.
        is_sub ^= lhs.sign ^ rhs.sign
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

        return self.make_real(sign, exponent, significand, context)

    def multiply(self, lhs, rhs, context):
        '''Returns the product of LHS and RHS in this format.'''
        # Multiplication is commutative
        if lhs.e_biased == 0:
            return self._multiply_special(lhs, rhs, context)
        if rhs.e_biased == 0:
            return self._multiply_special(rhs, lhs, context)
        return self._multiply_finite(lhs, rhs, context)

    def _multiply_finite(self, lhs, rhs, context):
        '''Returns the product of two finite floating point numbers in this format.'''
        sign = lhs.sign ^ rhs.sign
        exponent = lhs.exponent_int() + rhs.exponent_int()
        return self.make_real(sign, exponent, lhs.significand * rhs.significand, context)

    def _multiply_special(self, lhs, rhs, context):
        '''Return a lhs * rhs where the LHS is a NaN or infinity.'''
        if lhs.significand == 0:
            # infinity * zero -> invalid op
            if rhs.is_zero():
                return self._invalid_op_NaN(context)
            # infinity * infinity -> infinity
            # infinity * finite-non-zero -> infinity
            if not rhs.is_NaN():
                return self.make_infinity(lhs.sign ^ rhs.sign)
            # infinity * NaN propagates the NaN
            lhs, rhs = rhs, lhs

        # Propagate the NaN in the LHS
        return self._propagate_NaN(lhs, rhs, context)

    def divide(self, lhs, rhs, context):
        '''Return lhs / rhs in this format.'''
        sign = lhs.sign ^ rhs.sign

        if lhs.e_biased:
            # LHS is finite.  Is RHS finite too?
            if rhs.e_biased:
                # Both are finite.  Handle zeroes.
                if rhs.significand == 0:
                    # 0 / 0 -> NaN
                    if lhs.significand == 0:
                        return self._invalid_op_NaN(context)
                    # Finite / 0 -> Infinity
                    context.set_flags(Flags.DIV_BY_ZERO)
                    return self.make_infinity(sign)

                return self._divide_finite(lhs, rhs, 'D', context)

            # RHS is NaN or infinity
            if rhs.significand == 0:
                # finite / infinity -> zero
                return self.make_zero(sign)

            # finite / NaN propagates the NaN.
            lhs, rhs = rhs, lhs
        elif lhs.significand == 0:
            # LHS is infinity
            # infinity / finite -> infinity
            if rhs.is_finite():
                return self.make_infinity(sign)
            # infinity / infinity is an invalid op
            if rhs.significand == 0:
                return self._invalid_op_NaN(context)
            # infinity / NaN propagates the NaN.
            lhs, rhs = rhs, lhs

        # Propagate the NaN in the LHS
        return self._propagate_NaN(lhs, rhs, context)

    def _divide_finite(self, lhs, rhs, operation, context):
        '''Calculate LHS / RHS, where both are finite and RHS is non-zero.

        if operation is 'D' for division, returns the correctly-rounded result.  If the
        operation is 'R' for IEEE remainder, the quotient is rounded to the nearest
        integer (ties to even) and the remainder returned.
        '''
        assert operation in ('D', 'R')

        sign = lhs.sign ^ rhs.sign if operation == 'D' else lhs.sign

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

        exponent_diff = lhs_exponent - rhs_exponent
        if operation == 'D':
            bits = self.precision
        else:
            bits = min(max(exponent_diff + 1, 0), self.precision)

        # Long division.  Note by construction the quotient significand will have a
        # leading 1, i.e., it will contain precisely precision significant bits,
        # representing a value in [1, 2).
        quot_sig = 0
        for _n in range(bits):
            if _n:
                lhs_sig <<= 1
            quot_sig <<= 1
            if lhs_sig >= rhs_sig:
                lhs_sig -= rhs_sig
                quot_sig |= 1

        assert 0 <= lhs_sig < rhs_sig or bits == 0

        if lhs_sig == 0 or bits == 0:
            lost_fraction = LostFraction.EXACTLY_ZERO
        elif lhs_sig * 2 < rhs_sig:
            lost_fraction = LostFraction.LESS_THAN_HALF
        elif lhs_sig * 2 == rhs_sig:
            lost_fraction = LostFraction.EXACTLY_HALF
        else:
            lost_fraction = LostFraction.MORE_THAN_HALF

        # At present both quotient (which is truncated) and remainder have sign lhs.sign.
        # The exponent when viewing quot_sig as an integer.
        quotient_exponent = exponent_diff - (bits - 1)
        if operation == 'D':
            return self.make_real(sign, quotient_exponent, quot_sig, context, lost_fraction)

        # IEEE remainder.  We need to figure out if the quotient rounds up under
        # nearest-ties-to-even, to do another deduction of rhs_sig.

        # As an integer it would drop the significand's least significant
        # -quotient_exponent bits.
        rshift = -quotient_exponent
        shift_lf = lost_bits_from_rshift(quot_sig, rshift)

        # If we shifted, combine the lost fractions; shift_lf is the more significant
        if rshift > 0:
            lost_fraction = LostFraction(shift_lf | (lost_fraction != LostFraction.EXACTLY_ZERO))
        elif rshift < 0:
            lost_fraction = LostFraction(lost_fraction != LostFraction.EXACTLY_ZERO)

        if lost_fraction == LostFraction.EXACTLY_HALF:
            rounds_up = bool(quot_sig & (1 << rshift))
        else:
            rounds_up = lost_fraction == LostFraction.MORE_THAN_HALF

        # If we incremented the quotient, we must correspondingly deduct the significands
        if rounds_up:
            lhs_sig -= rhs_sig
            sign = not sign
            lhs_sig = -lhs_sig

        # IEEE-754 decrees a remainder of zero shall have the sign of the LHS
        if lhs_sig == 0:
            sign = lhs.sign

        # Return the remainder
        return self.make_real(sign, lhs_exponent - (bits - 1), lhs_sig, context)

    def sqrt(self, value, context):
        '''Return sqrt(value) in this format.  It has a positive sign for all operands >= 0,
        except that sqrt(-0) shall be -0.'''
        if value.e_biased == 0:
            if value.significand:
                # Propagate NaNs
                return self._propagate_NaN(value, value, context)
            # -Inf -> invalid operation
            if value.sign:
                return self._invalid_op_NaN(context)
        if not value.significand:
            # +0 -> +0, -0 -> -0, +Inf -> +Inf
            return value.copy()
        if value.sign:
            return self._invalid_op_NaN(context)

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
        est = self.make_real(False, exponent // 2, math.floor(math.sqrt(sig)), nearest_context)
        n = 1
        while True:
            print(n, "EST", est.as_tuple(), est.to_hex(TextFormat(), Context()))
            assert est.significand

            nearest_context.clear_flags()
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

        down_context = Context(ROUND_DOWN)
        # Ensure that the precise answer lies in [est, est.next_up()).
        if div.significand == est.significand:
            # Do we have an exact square root?
            if not (nearest_context.flags & Flags.INEXACT):
                if est.is_subnormal():
                    context.set_flags(Flags.SUBNORMAL)
                return est

            est_squared = value.fmt.multiply(est, est, down_context)
            if est_squared.compare_quiet_ge(value, down_context):
                est = est.next_down(down_context)
        elif est.compare_quiet_ge(div, down_context):
            est = div

        print("EST", est.as_tuple(), est.to_hex(TextFormat(), Context()))

        # EST and EST.next_up() now bound the precise result.  Decide if we need to round
        # it up.  This is only difficult with non-directed roundings.
        if context.round_to_nearest():
            temp_fmt = BinaryFormat(self.precision + 1, self.e_max, self.e_min)
            nearest_context.clear_flags()
            half = temp_fmt.add(est, est.next_up(nearest_context), nearest_context)
            half = half.scaleb(-1, nearest_context)
            assert not (nearest_context.flags & Flags.INEXACT)
            down_context.clear_flags()
            half_squared = value.fmt.multiply(half, half, down_context)
            compare = half_squared.compare(value, down_context, False)
            if compare == Compare.LESS_THAN:
                lost_fraction = LostFraction.MORE_THAN_HALF
            elif compare == Compare.EQUAL:
                if down_context.flags & Flags.INEXACT:
                    lost_fraction = LostFraction.LESS_THAN_HALF
                else:
                    lost_fraction = LostFraction.EXACTLY_HALF
            else:
                lost_fraction = LostFraction.LESS_THAN_HALF
        else:
            # Anything as long as non-zero
            lost_fraction = LostFraction.EXACTLY_HALF

        # Now round est with the lost fraction
        is_tiny_before = est.is_subnormal()
        if round_up(context.rounding, lost_fraction, False, bool(est.significand & 1)):
            est = est.next_up(nearest_context)
        context.on_normalized_finite(is_tiny_before, est.is_subnormal(), True)
        return est

    def fma(self, lhs, rhs, addend, context):
        '''Return a fused multiply-add operation.  The result is lhs * rhs + addend correctly
        rounded to this format.
        '''
        # Perform the multiplication in a format where it is exact and there are no
        # subnormals.  Then the only signal that can be raised is INVALID.
        product_fmt = BinaryFormat(lhs.fmt.precision + rhs.fmt.precision,
                                   lhs.fmt.e_max + rhs.fmt.e_max + 1,
                                   lhs.fmt.e_min - (lhs.fmt.precision - 1)
                                   + rhs.fmt.e_min - (rhs.fmt.precision - 1))
        # FIXME: when tests are complete, use context not product_context
        product_context = Context()
        product = product_fmt.multiply(lhs, rhs, product_context)
        assert product_context.flags & ~Flags.INVALID == 0
        context.set_flags(product_context.flags)
        return self.add(product, addend, context)


IEEEhalf = BinaryFormat.from_exponent_width(11, 5)
IEEEsingle = BinaryFormat.from_exponent_width(24, 8)
IEEEdouble = BinaryFormat.from_exponent_width(53, 11)
IEEEquad = BinaryFormat.from_exponent_width(113, 15)
#IEEEoctuple = BinaryFormat.from_exponent_width(237, 18)
# 80387 floating point takes place with a wide exponent range but rounds to single, double
# or extended precision.  It also has an explicit integer bit.
x87extended = BinaryFormat.from_exponent_width(64, 15)
x87double = BinaryFormat.from_exponent_width(53, 15)
x87single = BinaryFormat.from_exponent_width(24, 15)


class Binary:
    def __init__(self, fmt, sign, biased_exponent, significand):
        '''Create a floating point number with the given format, sign, biased exponent and
        significand.  For NaNs - saturing exponents with non-zero signficands - we interpret
        the significand as the binary payload below the quiet bit.
        '''
        if not isinstance(biased_exponent, int):
            raise TypeError('biased exponent must be an integer')
        if not isinstance(significand, int):
            raise TypeError('significand must be an integer')
        if not 0 <= biased_exponent <= fmt.e_max + fmt.e_bias:
            raise ValueError(f'biased exponent {biased_exponent:,d} out of range')
        if biased_exponent == 0:
            if not 0 <= significand < fmt.int_bit:
                raise ValueError(f'significand {significand:,d} out of range for non-finite')
        else:
            if not 0 <= significand <= fmt.max_significand:
                raise ValueError(f'significand {significand:,d} out of range')
        self.fmt = fmt
        self.sign = bool(sign)
        self.e_biased = biased_exponent
        # The significand as an unsigned integer, or payload including quiet bit for NaNs
        self.significand = significand

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

    def pack(self, endianness='little'):
        '''Packs this value to bytes of the given endianness.'''
        significand = self.significand
        if self.e_biased == 0:
            ieee_exponent = self.fmt.e_max * 2 + 1
        elif significand < self.fmt.int_bit:
            ieee_exponent = 0
        else:
            ieee_exponent = self.e_biased
            significand -= self.fmt.int_bit
        return self.fmt.pack(self.sign, ieee_exponent, significand, endianness)

    def NaN_payload(self):
        '''Returns the NaN payload.  Raises RuntimeError if the value is not a NaN.'''
        assert self.is_NaN()
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

    def is_NaN(self):
        '''Return True if this is a NaN of any kind.'''
        return self.e_biased == 0 and self.significand

    def is_signalling(self):
        '''Return True if and only if this is a signalling NaN.'''
        return self.is_NaN() and not (self.significand & self.fmt.quiet_bit)

    def is_canonical(self):
        '''We only have canonical values.'''
        # FIXME: how to extend this to packed formats
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

    def copy(self):
        '''Returns a copy of this number.'''
        return Binary(self.fmt, self.sign, self.e_biased, self.significand)

    def copy_sign(self, y):
        '''Retuns a copy of this number but with the sign of y.'''
        return Binary(self.fmt, y.sign, self.e_biased, self.significand)

    def copy_negate(self):
        '''Returns a copy of this number with the opposite sign.'''
        return Binary(self.fmt, not self.sign, self.e_biased, self.significand)

    def copy_abs(self):
        '''Returns a copy of this number with sign False (positive).'''
        return Binary(self.fmt, False, self.e_biased, self.significand)

    def take_sign(self, y):
        '''Sets the sign of this number to that of y.  copy_sign without a copy.'''
        self.sign = y.sign

    def negate(self):
        '''Negates this number in-place.'''
        self.sign = not self.sign

    def abs(self):
        '''Sets the sign to False (positive).'''
        self.sign = False

    ##
    ## General homogeneous computational operations.
    ##
    ## Format is preserved and the result is in-place.
    ##

    def remainder(self, rhs, context):
        '''Set to the remainder when divided by rhs.

        If rhs != 0, the remainder is defined for finite operands as r = lhs - rhs * n,
        where n is the integer nearest the exact number lhs / rhs with round-to-even
        semantics.  The result is always exact, and if r is zero its sign shall be that of
        lhs.

        If rhs is zero or lhs is infinite, an invalid operation is returned if neither
        operand is a NaN.

        If rhs is infinite then the result is lhs if it is finite.
        '''
        if self.fmt != rhs.fmt:
            raise ValueError('remainder requires an operand of the same format')

        # Are we an infinity or NaN?
        if self.e_biased == 0:
            if self.significand:
                return self.fmt._propagate_NaN(self, rhs, context)
            elif rhs.is_NaN():
                return self.fmt._propagate_NaN(rhs, rhs, context)
            else:
                # remainder (infinity, non-NaN) is an invalid operation
                return self.fmt._invalid_op_NaN(context)
        elif rhs.e_biased == 0:
            if rhs.significand:
                return self.fmt._propagate_NaN(rhs, rhs, context)
            else:
                # remainder(finite, infinity) is the LHS exact remainder(subnormal,
                # infinity) is the LHS and signals underflow, which because the result is
                # exact, sets a subnormal flag.
                if self.is_subnormal():
                    context.set_flags(Flags.SUBNORMAL)
                return self.copy()
        elif rhs.significand == 0:
            # remainder (finite, zero) is an invalid operation
            return self.fmt._invalid_op_NaN(context)

        # At last, we're talking the remainder of two finite numbers with RHS non-zero.
        return self.fmt._divide_finite(self, rhs, 'R', context)

    ##
    ## logBFormat operations (logBFormat is an integer format)
    ##

    def scaleb(self, N, context):
        '''Return x * 2^N for integral values N, correctly-rounded. and in the format of
        x.  For non-zero values of N, scalb(±0, N) is ±0 and scalb(±∞) is ±∞.  For zero values
        of N, scalb(x, N) is x.
        '''
        # NaNs and infinities are unchanged (but NaNs are made quiet)
        if self.e_biased == 0:
            return self.to_quiet(context)
        return self.fmt.make_real(self.sign, self.exponent_int() + N, self.significand, context)

    def _logb(self):
        '''A private helper function.'''
        if self.e_biased == 0:
            if self.significand == 0:
                return 'Inf'
            return 'NaN'

        if self.significand == 0:
            return 'Zero'

        return self.exponent() + self.significand.bit_length() - self.fmt.precision

    def logb_integral(self, context):
        '''Return the exponent e of x, a signed integer, when represented with infinite range and
        minimum exponent.  Thus 1 <= scalb(x, -logb(x)) < 2 when x is positive and finite.
        logb(1) is +0.  logb(NaN), logb(∞) and logb(0) return implementation-defined
        values outside the range ±2 * (emax + p - 1) and flag an invalid operation.
        '''
        result = self._logb()
        if isinstance(result, int):
            return result

        context.set_flags(Flags.INVALID)
        if result == 'Inf':
            return self.fmt.logb_inf
        if result == 'NaN':
            return self.fmt.logb_NaN
        return self.fmt.logb_zero

    def logb(self, context):
        '''Return the exponent e of x, a signed integer, when represented with infinite range and
        minimum exponent.  Thus 1 <= scalb(x, -logb(x)) < 2 when x is positive and finite.
        logb(1) is +0.  logb(NaN) is a NaN, logb(∞) is +∞, and logb(0) is -∞ and signals
        the divide-by-zero exception.
        '''
        result = self._logb()
        if isinstance(result, int):
            return self.fmt.make_real(result < 0, 0, abs(result), context)

        if result == 'NaN':
            return self.to_quiet(context)
        if result == 'Zero':
            context.set_flags(Flags.DIV_BY_ZERO)
            return self.fmt.make_infinity(True)
        return self.fmt.make_infinity(False)

    ##
    ## General computational operations
    ##

    def to_hex(self, text_format, context):
        '''Return text that is a representation of the floating point value that is hexadecimal if
        the number is finite.  See the docstring of OutputSpec for output control.

        Normal numbers are returned with a hex integer digit of 1, subnormal numbers 0.
        Zeroes are output with an exponent of 0.

        Only conversion of signalling NaNs to quiet NaNs can set flags (INVALID and
        possibly INEXACT).
        '''
        return self.fmt.to_hex(self, text_format, context)

    def to_decimal(self, text_format, precision, context):
        '''Convert a binary floating point value to a decimal string.'''
        if not self.is_finite():
            return text_format.non_finite_string(self.to_quiet(context))

        sig_digits, exponent, is_inexact = self.decimal_digits(precision)
        if is_inexact:
            context.set_flags(Flags.INEXACT)

        return text_format.format_decimal(self.sign, sig_digits, exponent)

    def decimal_digits(self, _precision):
        '''Returns a tuple:

            (digits, exp, inexact)

        digits is a string of digits (most significant first) with no decimal point.  exp
        is the base 10 exponent of the first digit (so the base 10 exponent of the last is
        N - (len(digits) - 1).  inexact is True if the conversion to decimal is not exact.

        See "How to Print Floating-Point Numbers Accurately" by Steele and White, in
        particular Table 3.  This is an optimized implementation of their algorithm.
        '''
        if not self.is_finite():
            raise RuntimeError('decimal_digits() called on a non-finite number')

        # Special-case zero.
        if self.is_zero():
            return '0', 0, False

        e_p = self.exponent_int()
        R = self.significand << max(0, e_p)
        S = 1 << max(0, -e_p)

        # Now the arithmetic value is R / S.  M is the scaled value of an ulp upwards.  We
        # can stop generating digits when the remainder is within M of the boundary,
        # because then round-to-nearest of the output is guaranteed to give our target
        # value.
        M = 1 << max(0, e_p)
        low_shift = 2 if self.significand == self.fmt.int_bit else 1
        k = 0

        # This loop is needed for negative exponents H. It scales R until divmod()
        # delivers a non-zero digit.
        while R * 10 < S:
            k -= 1
            R *= 10
            M *= 10

        # This loop is needed for positive exponents H.  It scales S until digits can be
        # reliably delivered.  This condition is not strictly tested.  can remove M
        # entirely.
        while 2 * R + M >= 2 * S:
            S *= 10
            k += 1

        H = k - 1

        digits = bytearray()
        while True:
            k -= 1
            U, R = divmod(R * 10, S)
            M *= 10
            low = (R << low_shift) < M
            high = 2 * R > (2 * S) - M
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

        return digits.decode(), H, R != 0

    def to_quiet(self, context):
        '''Return a copy except that a signalling NaN becomes its quiet twin (in which case
        an invalid operation is flagged).
        '''
        if self.is_signalling():
            context.set_flags(Flags.INVALID)
            return Binary(self.fmt, self.sign, self.e_biased,
                             self.significand | self.fmt.quiet_bit)
        return self.copy()

    def next_up(self, context):
        '''Set to the smallest floating point value (unless operating on a positive infinity or
        NaN) that compares greater.
        '''
        return self.fmt._next_up(self, context, False)

    def next_down(self, context):
        '''Set to the greatest floating point value (unless operating on a negative infinity or
        NaN) that compares less.

        As per IEEE-754 next_down(x) = -next_up(-x)
        '''
        return self.fmt._next_up(self, context, True)

    def _to_integer(self, integer_fmt, rounding, signal_inexact, context):
        '''Round the value to an integer, as per rounding.  If integer_fmt is None, the result is
        a floating point value of the same format (a round_to_integral operation).
        Otherwise it is a convert_to_integer operation and an integer that fits in the
        given format is returned.

        If the value is a NaN or infinity or the rounded result cannot be represented in
        the integer format, invalid operation is signalled.  The result is zero if the
        source is a NaN, otherwise it is the smallest or largest value in the integer
        format, whichever is closest to the ideal result.

        If signal_inexact is True then if the result is not numerically equal to the
        original value, INEXACT is signalled, otherwise it is not.
        '''
        if not isinstance(integer_fmt, (type(None), IntegerFormat)):
            raise TypeError('integer_fmt must be an integer format or None')

        if self.e_biased == 0:
            if integer_fmt is None:
                # Quiet NaNs and infinites stay unchanged; signalling NaNs are converted to quiet.
                return self.to_quiet(context)
            context.set_flags(Flags.INVALID)
            if self.significand:
                return 0
            return integer_fmt.min_int if self.sign else integer_fmt.max_int

        # Zeroes return unchanged.
        if self.significand == 0:
            if integer_fmt is None:
                return self.copy()
            return 0

        # Strategy: truncate the significand, capture the lost fraction, and round.
        # Truncating the significand means clearing zero or more of its least-significant
        # bits.
        value = self.significand
        rshift = -self.exponent_int()

        # We're already a (large) integer if count is <= 0
        if rshift <= 0:
            if integer_fmt is None:
                return self.copy()
            value <<= -rshift
            value, was_clamped = integer_fmt.clamp(-value if self.sign else value)
            if was_clamped:
                context.set_flags(Flags.INVALID)
            return value

        value, lost_fraction = shift_right(value, rshift)

        if lost_fraction == LostFraction.EXACTLY_ZERO:
            signal_inexact = False
        elif round_up(rounding, lost_fraction, self.sign, bool(value & 1)):
            value += 1

        # must only do so after confirming the result was not clamped (consider a negative
        # subnormal rounding down to -1 with an unsigned format).
        if integer_fmt is None:
            if signal_inexact:
                context.set_flags(Flags.INEXACT)
            if value:
                lshift = self.fmt.precision - value.bit_length()
                e_biased = self.e_biased + rshift - lshift
                return Binary(self.fmt, self.sign, e_biased, value << lshift)
            return self.fmt.make_zero(self.sign)

        value, was_clamped = integer_fmt.clamp(-value if self.sign else value)
        if was_clamped:
            context.set_flags(Flags.INVALID)
        elif signal_inexact:
            context.set_flags(Flags.INEXACT)
        return value

    def round_to_integral(self, rounding, context):
        '''Return the value rounded to the nearest integer in the same BinaryFormat.  The rounding
        method is given explicitly; that in context is ignored.

        This only signal raised is INVALID on a signalling NaN input.
        '''
        return self._to_integer(None, rounding, False, context)

    def round_to_integral_exact(self, context):
        '''Return the value rounded to the nearest integer in the same BinaryFormat.  The rounding
        method is taken from context.

        This function signals INVALID on a signalling NaN input, or INEXACT if the
        result is not the original value (i.e. it was not an integer).
        '''
        return self._to_integer(None, context.rounding, True, context)

    def convert_to_integer(self, integer_fmt, rounding, context):
        '''Return the value rounded to the nearest integer in the same BinaryFormat.  The rounding
        method is given explicitly; that in context is ignored.

        NaNs, infinities and out-of-range values raise the INVALID signal.  INEXACT is not
        raised.
        '''
        return self._to_integer(integer_fmt, rounding, False, context)

    def convert_to_integer_exact(self, integer_fmt, rounding, context):
        '''As for convert_to_integer, except that INEXACT is signaled if the result is in-range
        and not numerically equal to the original value (i.e. it was not an integer).
        '''
        return self._to_integer(integer_fmt, rounding, True, context)

    ##
    ## Signalling computational operations
    ##

    def compare(self, rhs, context, signalling):
        '''Return LHS vs RHS as one of the four comparison constants.

        Comparisons of NaNs signal INVALID if either is signalling or if signalling is True.
        If context is None, nothing is signalled.
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
        if context and (signalling or self.is_signalling() or rhs.is_signalling()):
            context.set_flags(Flags.INVALID)
        return Compare.UNORDERED

    def compare_quiet_eq(self, rhs, context):
        return self.compare(rhs, context, False) == Compare.EQUAL

    def compare_quiet_ne(self, rhs, context):
        return self.compare(rhs, context, False) != Compare.EQUAL

    def compare_quiet_gt(self, rhs, context):
        return self.compare(rhs, context, False) == Compare.GREATER_THAN

    def compare_quiet_ng(self, rhs, context):
        return self.compare(rhs, context, False) != Compare.GREATER_THAN

    def compare_quiet_ge(self, rhs, context):
        return self.compare(rhs, context, False) in (Compare.EQUAL, Compare.GREATER_THAN)

    def compare_quiet_lu(self, rhs, context):
        return self.compare(rhs, context, False) not in (Compare.EQUAL, Compare.GREATER_THAN)

    def compare_quiet_lt(self, rhs, context):
        return self.compare(rhs, context, False) == Compare.LESS_THAN

    def compare_quiet_nl(self, rhs, context):
        return self.compare(rhs, context, False) != Compare.LESS_THAN

    def compare_quiet_le(self, rhs, context):
        return self.compare(rhs, context, False) in (Compare.EQUAL, Compare.LESS_THAN)

    def compare_quiet_gu(self, rhs, context):
        return self.compare(rhs, context, False) not in (Compare.EQUAL, Compare.LESS_THAN)

    def compare_quiet_un(self, rhs, context):
        return self.compare(rhs, context, False) == Compare.UNORDERED

    def compare_quiet_or(self, rhs, context):
        return self.compare(rhs, context, False) != Compare.UNORDERED

    def compare_signalling_eq(self, rhs, context):
        return self.compare(rhs, context, True) == Compare.EQUAL

    def compare_signalling_ne(self, rhs, context):
        return self.compare(rhs, context, True) != Compare.EQUAL

    def compare_signalling_gt(self, rhs, context):
        return self.compare(rhs, context, True) == Compare.GREATER_THAN

    def compare_signalling_ge(self, rhs, context):
        return self.compare(rhs, context, True) in (Compare.EQUAL, Compare.GREATER_THAN)

    def compare_signalling_lt(self, rhs, context):
        return self.compare(rhs, context, True) == Compare.LESS_THAN

    def compare_signalling_le(self, rhs, context):
        return self.compare(rhs, context, True) in (Compare.EQUAL, Compare.LESS_THAN)

    def compare_signalling_ng(self, rhs, context):
        return self.compare(rhs, context, True) != Compare.GREATER_THAN

    def compare_signalling_lu(self, rhs, context):
        return self.compare(rhs, context, True) not in (Compare.EQUAL, Compare.GREATER_THAN)

    def compare_signalling_nl(self, rhs, context):
        return self.compare(rhs, context, True) != Compare.LESS_THAN

    def compare_signalling_gu(self, rhs, context):
        return self.compare(rhs, context, True) not in (Compare.EQUAL, Compare.LESS_THAN)

    def total_order(self, rhs):
        '''The total order relation on a format.  A loose equivalent of <=.'''
        if self.fmt != rhs.fmt:
            raise ValueError('total order requires both operands have the same format')
        comp = self.compare(rhs, None, False)
        if comp == Compare.LESS_THAN:
            return True
        if comp == Compare.GREATER_THAN:
            return False
        if comp == Compare.EQUAL:
            return self.sign == rhs.sign or self.sign
        # At least one is a NaN
        if rhs.is_NaN():
            if self.is_NaN():
                if self.sign != rhs.sign:
                    return self.sign
                if self.is_signalling() != rhs.is_signalling():
                    return self.is_signalling() ^ self.sign
                if self.NaN_payload() < rhs.NaN_payload():
                    return not self.sign
                if self.NaN_payload() > rhs.NaN_payload():
                    return self.sign
                return True
            return not rhs.sign
        return self.sign

    def total_order_mag(self, rhs):
        '''Per IEEE-754, totalOrderMag(x, y) is totalOrder(abs(x), abs(y)).'''
        return self.copy_abs().total_order(rhs.copy_abs())
