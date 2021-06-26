#
# An implementation of many operations of generic binary floating-point arithmetic
#
# (c) Neil Booth 2007-2021.  All rights reserved.
#

import re
from abc import ABC, abstractmethod
from enum import IntFlag, IntEnum


__all__ = ('InterchangeKind', 'FloatClass', 'FloatEnv', 'FloatFormat', 'OpStatus', 'IEEEfloat',
           'RoundingMode', 'RoundTiesToEven', 'RoundTiesToAway', 'RoundTowardsPositive',
           'RoundTowardsNegative', 'RoundTowardsZero',
           'IEEEhalf', 'IEEEsingle', 'IEEEdouble', 'IEEEquad',
           'x87extended', 'x87double', 'x87single',)


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


class InterchangeKind(IntEnum):
    '''Descibes if and how floating point numbers in this format can be interchaged.'''
    NONE = 0            # Not an interchange format, values cannot be packed or unpacked
    IMPLICIT = 1        # Pack with an implicit integer bit (IEEE)
    EXPLICIT = 2        # Pack with an explicit integer bit (x87 extended precision)


class FloatClass(IntEnum):
    '''All floating point numbers belong to precisely one class.'''
    sNaN = 0          # Signalling NaN
    qNaN = 1          # Quiet NaN
    nInf = 2          # Negative infinity
    nNormal = 3       # Negative normal
    nSubnormal = 4    # Negative subnormal
    nZero = 5         # Negative zero
    pZero = 6         # Positive zero
    pSubnormal = 7    # Positive subnormal
    pNormal = 8       # Positive normal
    pInf = 9          # Positive infinity


# Operation status flags.  OVERFLOW is always returned with INEXACT.  UNDERFLOW, too, has
# INEXACT unless the environment always_flag_underflow bit is set, in which case UNDERFLOW
# may stand alone as per IEEE-754.
class OpStatus(IntFlag):
    OK          = 0
    INVALID     = 0x01
    DIV_BY_ZERO = 0x02
    OVERFLOW    = 0x04
    UNDERFLOW   = 0x08
    INEXACT     = 0x10


# When precision is lost during a calculation this indicates what fraction of the LSB the
# lost bits represented.  It essentially combines the roles of 'guard' and 'sticky' bits.
class LostFraction(IntEnum):   # Example of truncated bits:
    EXACTLY_ZERO = 0           # 000000
    LESS_THAN_HALF = 1	       # 0xxxxx  x's not all zero
    EXACTLY_HALF = 2           # 100000
    MORE_THAN_HALF = 3         # 1xxxxx  x's not all zero


def lost_bits_from_rshift(significand, bits):
    '''Return what the lost bits would be were the significand shifted right the given number
    of bits (negative is a left shift).
    '''
    if bits <= 0:
        return LostFraction.EXACTLY_ZERO
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


class RoundingMode(ABC):
    '''Rounding modes are implemented as derived classes.'''

    @classmethod
    @abstractmethod
    def rounds_away(cls, _lost_fraction, _sign, _is_odd):
        '''Return True if, when an operation results in a lost fraction, the rounding mode
        requires rounding away from zero (i.e. incrementing the significand).

        sign is the sign of the number, and is_odd indicates if the LSB of the new
        significand is set, which is needed for ties-to-even rounding.
        '''
        raise NotImplementedError

    @classmethod
    @abstractmethod
    def overflow_to_infinity(cls, sign):
        '''When the rounded result has an too high exponent, return True if the rounding mode
        requires rounding to infinity.

        sign is the sign of the number.'''
        raise NotImplementedError


class RoundTiesToEven(RoundingMode):
    '''Round-to-nearest with ties-to-even.'''

    @classmethod
    def rounds_away(cls, lost_fraction, _sign, is_odd):
        if lost_fraction == LostFraction.EXACTLY_HALF:
            return is_odd
        return lost_fraction == LostFraction.MORE_THAN_HALF

    @classmethod
    @abstractmethod
    def overflow_to_infinity(cls, _sign):
        return True


class RoundTiesToAway(RoundingMode):
    '''Round-to-nearest with ties-to-away.'''

    @classmethod
    def rounds_away(cls, lost_fraction, _sign, _is_odd):
        return lost_fraction in {LostFraction.EXACTLY_HALF, LostFraction.MORE_THAN_HALF}

    @classmethod
    @abstractmethod
    def overflow_to_infinity(cls, _sign):
        return True


class RoundTowardsPositive(RoundingMode):
    '''Round towards positive infinity.'''

    @classmethod
    def rounds_away(cls, lost_fraction, sign, _is_odd):
        return not sign and lost_fraction != LostFraction.EXACTLY_ZERO

    @classmethod
    @abstractmethod
    def overflow_to_infinity(cls, sign):
        return not sign


class RoundTowardsNegative(RoundingMode):
    '''Round towards negative infinity.'''

    @classmethod
    def rounds_away(cls, lost_fraction, sign, _is_odd):
        return sign and lost_fraction != LostFraction.EXACTLY_ZERO

    @classmethod
    @abstractmethod
    def overflow_to_infinity(cls, sign):
        return sign


class RoundTowardsZero(RoundingMode):

    @classmethod
    def rounds_away(cls, _lost_fraction, _sign, _is_odd):
        return False

    @classmethod
    @abstractmethod
    def overflow_to_infinity(cls, _sign):
        return False


class FloatEnv:

    __slots__ = ('rounding_mode', 'detect_tininess_after', 'always_flag_underflow')

    def __init__(self, rounding_mode, detect_tininess_after, always_flag_underflow):
        '''Floating point environment flags:

        rounding_mode is one of the RoundingMode enumerations and controls rounding of inexact
        results.

        If detect_tininess_after is True tininess is detected before rounding, otherwise after

        If always_flag_underflow is True then underflow is flagged whenever tininess is detected,
        otherwise it is only flagged if tininess is detected and the result is inexact.
        '''
        self.rounding_mode = rounding_mode
        self.detect_tininess_after = detect_tininess_after
        self.always_flag_underflow = always_flag_underflow


class IntFormat:
    '''A two's-complement signed integer.'''

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
        '''Returns a (result, status) pair.

        The result is value, but forced to the range.'''
        if not isinstance(value, int):
            raise TypeError('clamp takes an integer')
        if value > self.max_int:
            return self.max_int, OpStatus.OVERFLOW
        if value < self.min_int:
            return self.min_int, OpStatus.OVERFLOW
        return value, OpStatus.OK


class FloatFormat:
    '''An IEEE-754 floating point format.'''

    __slots__ = ('precision', 'e_width', 'interchange_kind', 'e_max', 'e_min', 'e_bias',
                 'e_saturated', 'size', 'int_bit', 'quiet_bit', 'max_significand')

    def __init__(self, e_width, precision, interchange_kind):
        '''e_width is the exponent width in bits.

        precision is the number of bits in the significand including an explicit integer
        bit.

        interchange_kind describes if this is an interchange format, and if so if the
        integer bit is implicit or explicit.
        '''
        self.precision = precision
        self.e_width = e_width
        self.interchange_kind = interchange_kind
        # The largest e such that 2^e is representable.  The largest representable number
        # is 2^e_max * (2 - 2^(1 - precision)) when the significand is all ones.  Unbibased.
        self.e_max = (1 << e_width - 1) - 1
        # The smallest e such that 2^e is a normalized number.  Unbiased.
        self.e_min = 1 - self.e_max
        # The exponent bias
        self.e_bias = self.e_max
        # The biased exponent for infinities and NaNs has all bits 1
        self.e_saturated = (1 << e_width) - 1
        # The number of bytes needed to encode an FP number.  A sign bit, the exponent,
        # and the significand.
        if interchange_kind == InterchangeKind.NONE:
            self.size = None
        else:
            self.size = (1 + e_width + (interchange_kind == InterchangeKind.EXPLICIT)
                         + (self.precision - 1) + 7) // 8
        # The integer bit (MSB) in the significand
        self.int_bit = 1 << self.precision - 1
        # The quiet bit in NaNs
        self.quiet_bit = self.int_bit >> 1
        # Significands are unsigned bitstrings of precision bits
        self.max_significand = (1 << self.precision) - 1

    def make_zero(self, sign):
        '''Returns a zero of the given sign.'''
        return IEEEfloat(self, sign, 0, 0)

    def make_infinity(self, sign):
        '''Returns an infinity of the given sign.'''
        return IEEEfloat(self, sign, self.e_saturated, 0)

    def make_largest_finite(self, sign):
        '''Returns the finite number of maximal magnitude with the given sign.'''
        return IEEEfloat(self, sign, self.e_saturated - 1, self.max_significand)

    def make_NaN(self, sign, is_quiet, payload, flag_invalid):
        '''Return a (value, status) pair.

        The value is a NaN with the given payload; the NaN is quiet iff is_quiet.
        If no payload bits are lost the status is OK, otherwise INEXACT.
        '''
        if payload < 0:
            raise ValueError(f'NaN payload cannot be negative: {payload}')
        mask = self.quiet_bit - 1
        adj_payload = payload & mask
        if is_quiet:
            adj_payload |= self.quiet_bit
        else:
            adj_payload = max(adj_payload, 1)
        status = OpStatus.OK if adj_payload & mask == payload else OpStatus.INEXACT
        if flag_invalid:
            status |= OpStatus.INVALID
        return IEEEfloat(self, sign, self.e_saturated, adj_payload), status

    def make_real(self, sign, exponent, significand, env):
        '''Return a (value, status) pair.

        The floating point number is the correctly-rounded (according to env) value of the
        infinitely precise result

           ± 2^exponent * significand

        The status indicates if overflow, underflow or inexact.

        For example,

           rounded_value(IEEEsingle, True, -3, 2)
               -> -0.75, OpStatus.OK
           rounded_value(IEEEsingle, True, 1, -200)
               -> +0.0, OpStatus.UNDERFLOW | OpStatus.INEXACT
        '''
        size = significand.bit_length()
        if size == 0:
            # Return a correctly-signed zero
            return IEEEfloat(self, sign, 0, 0), OpStatus.OK

        # Shifting the significand so the MSB is one gives us the natural shift.  There it
        # is followed by a decimal point, so the exponent must be adjusted to compensate.
        # However we cannot fully shift if the exponent would fall below e_min.
        exponent += self.precision - 1
        rshift = max(size - self.precision, self.e_min - exponent)

        # Shift the significand and update the exponent
        significand, lost_fraction = shift_right(significand, rshift)
        exponent += rshift

        # In case we detect tininess before rounding
        is_tiny = significand < self.int_bit

        # Round
        if env.rounding_mode.rounds_away(lost_fraction, sign, bool(significand & 1)):
            # Increment the significand
            significand += 1
            # If the significand now overflows, halve it and increment the exponent
            if significand > self.max_significand:
                significand >>= 1
                exponent += 1

        # If the new exponent would be too big, then we overflow.  The result is either
        # infinity or the format's largest finite value depnding on the rounding mode.
        if exponent > self.e_max:
            if env.rounding_mode.overflow_to_infinity(sign):
                result = self.make_infinity(sign)
            else:
                result = self.make_largest_finite(sign)
            return result, OpStatus.OVERFLOW | OpStatus.INEXACT

        # Detect tininess after rounding if appropriate
        if env.detect_tininess_after:
            is_tiny = significand < self.int_bit

        # Denormals require exponent of zero
        exponent -= significand < self.int_bit

        status = OpStatus.OK if lost_fraction == LostFraction.EXACTLY_ZERO else OpStatus.INEXACT

        if is_tiny and (env.always_flag_underflow or status == OpStatus.INEXACT):
            status |= OpStatus.UNDERFLOW

        return IEEEfloat(self, sign, exponent + self.e_bias, significand), status

    def next_up(self, value, flip_sign):
        '''Return the smallest floating point value (unless operating on a positive infinity or
        NaN) that compares greater than the one whose parts are given.

        Signs are flipped on read and write if flip_sign is True.
        '''
        assert value.fmt is self
        sign = value.sign ^ flip_sign
        e_biased = value.e_biased
        significand = value.significand
        status = OpStatus.OK

        if e_biased == self.e_saturated:
            # Negative infinity becomes largest negative number; positive infinity unchanged
            if significand == 0:
                if sign:
                    significand = self.max_significand
                    e_biased -= 1
            # Signalling NaNs are converted to quiet.
            elif significand < self.quiet_bit:
                significand |= self.quiet_bit
                status = OpStatus.INVALID
        else:
            # Increment the significand of positive numbers, decrement the significand of
            # negative numbers.  Negative zero is the only number whose sign flips.
            if sign and significand:
                significand -= 1
                if e_biased and significand < self.int_bit:
                    e_biased -= 1
                    if e_biased:
                        significand |= self.int_bit
            else:
                sign = False
                significand += 1
                if significand > self.max_significand:
                    significand >>= 1
                    e_biased += 1
                    if e_biased == self.e_saturated:
                        significand = 0

        return IEEEfloat(self, sign ^ flip_sign, e_biased, significand), status

    def convert(self, value, env):
        '''Return a (result, status) pair.

        Convert a floating point value to this format rounding according to env.'''
        if value.e_biased == value.fmt.e_saturated:
            # Infinities
            if value.significand == 0:
                return self.make_infinity(value.sign), OpStatus.OK

            # NaNs
            return self.make_NaN(value.sign, True, value.NaN_payload(), value.is_signalling())

        # Zeroes
        if value.significand == 0:
            return self.make_zero(value.sign), OpStatus.OK

        return self.make_real(value.sign, value.exponent(), value.significand, env)

    def pack(self, sign, biased_exponent, significand, endianness='little'):
        '''Returns a floating point value encoded as bytes of the given endianness.'''
        if self.size is None:
            raise RuntimeError('not an interchange format')
        if not 0 <= significand <= self.max_significand:
            raise ValueError('significand out of range')
        if not 0 <= biased_exponent <= self.e_saturated:
            raise ValueError('biased exponent out of range')
        # Build up the bit representation
        value = biased_exponent + (self.e_saturated + 1) if sign else 0
        if self.interchange_kind == InterchangeKind.EXPLICIT:
            # Remove integer bit for non-canonical infinities and NaNs
            if significand >= self.int_bit and biased_exponent in (0, self.e_saturated):
                significand -= self.int_bit
            shift = self.precision
        else:
            # Remove the integer bit if implicit
            if significand >= self.int_bit:
                if biased_exponent in (0, self.e_saturated):
                    raise ValueError('integer bit is set for infinity / NaN')
                significand -= self.int_bit
            shift = self.precision - 1
        value = (value << shift) + significand
        return value.to_bytes(self.size, endianness)

    def unpack(self, binary, endianness='little'):
        '''Decode a binary encoding to a (sign, biased_exponent, significand) tuple.'''
        if self.size is None:
            raise RuntimeError('not an interchange format')
        if len(binary) != self.size:
            raise ValueError(f'expected {self.size} bytes to unpack; got {len(binary)}')
        value = int.from_bytes(binary, endianness)

        significand = value & self.max_significand
        if self.interchange_kind == InterchangeKind.EXPLICIT:
            value >>= self.precision
        else:
            value >>= self.precision - 1
        biased_exponent = value & self.e_saturated
        # Ensure the implict integer bit is set for normal numbers and that it is cleared
        # for denormals, zeros, infinities and NaNs (Intel calls the weird values that
        # occur if we don't do this psuedo-denormals, psuedo-infinities, pseudo-NaNs and
        # unnormals).
        if 0 < biased_exponent < self.e_saturated:
            significand |= self.int_bit
        else:
            significand &= (self.int_bit - 1)
        sign = bool(value & (self.e_saturated + 1))
        return sign, biased_exponent, significand

    def from_string(self, string, env):
        '''Convert a string to a rounded floating number of this format.'''
        if HEX_SIGNIFICAND_PREFIX.match(string):
            return self._from_hex_significand_string(string, env)
        return self._from_decimal_string(string, env)

    def _from_hex_significand_string(self, string, env):
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

        return self.make_real(sign, exponent, significand, env)

    def _from_decimal_string(self, string, env):
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
            if groups[2] is None:
                significand = int(groups[4])
            else:
                fraction = groups[3].rstrip('0')
                significand = int((groups[2] + fraction) or '0', 10)
                exponent -= len(fraction)

            if exponent == 0:
                return self.make_real(sign, 0, significand, env)
            raise NotImplementedError

            # Now the value is significand * 10^exponent.
            return self._decimal_to_binary(sign, exponent, significand, env)

        # groups[7] matches infinities
        if groups[7] is not None:
            return self.make_infinity(sign), OpStatus.OK

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
        return self.make_NaN(sign, is_quiet, payload, False)

    def _decimal_to_binary(self, sign, exponent, significand, env):
        '''Return a correctly-rounded binary value of

             (-1)^sign * significand * 10^exponent
        '''
        pow5 = pow(5, abs(exponent))
        calc_env = FloatEnv(RoundTiesToEven, True, False)
        round_to_nearest = env.rounding_mode in {RoundTiesToEven, RoundTiesToAway}

        def ulps_from_boundary(a, b, c):
            raise NotImplementedError

        # The loops are expensive; optimistically start with a low precision and
        # iteratively increase it until we can guarantee the result was correctly rounded.
        # Start with a precision a multiple of 64 bits with some room over the format
        # precision.
        parts_count = (self.precision + 10) // 64 + 1

        while True:
            # FIXME: choice of e_width
            calc_fmt = FloatFormat(16, parts_count * 64, InterchangeKind.NONE)
            excess_precision = calc_fmt.precision - self.precision

            sig_status, sig_value = calc_fmt.make_real(sign, 0, significand, calc_env)
            pow5_status, pow5_value = calc_fmt.make_real(False, 0, pow5, calc_env)

            # Because 10^n = 5^n * 2^n.   FIXME: check for out-of-range exponents.
            sig_value.e_biased += exponent

            if exponent >= 0:
                scaled_value, lost_fraction = calc_fmt._multiply_finite(sig_value, pow5_value)
                pow_HU_err = pow5_status != OpStatus.OK
            else:
                scaled_value, lost_fraction = calc_fmt._divide_finite(sig_value, pow5_value)
                # Subnormals have less precision
                if scaled_value.e_biased == 0:
                    excess_precision += calc_fmt.precision - scaled_value.bit_length()
                    excess_precision = min(excess_precision, calc_fmt.precision)
                # An extra half-ulp is lost in reciprocal of pow5
                if pow5_status == OpStatus.OK and lost_fraction == LostFraction.EXACTLY_ZERO:
                    pow_HU_err = 0
                else:
                    pow_HU_err = 2

            # The error from the true value, in half-ulps, on multiplying two floating
            # point numbers, which differ from the value they approximate by at most HUE1
            # and HUE2 half-ulps, is strictly less than the returned value.
            #
            # See Lemma 2 in "How to Read Floating Point Numbers Accurately" by William D
            # Clinger.
            inexact_multiply = lost_fraction != LostFraction.EXACTLY_ZERO
            sig_HU_err = sig_status != OpStatus.OK
            if sig_HU_err + pow_HU_err == 0:
                HU_err = inexact_multiply * 2     # <= inexactMultiply half-ulps
            else:
                HU_err = inexact_multiply + 2 * (sig_HU_err + pow_HU_err)

            HU_distance = 2 * ulps_from_boundary(sig_value, excess_precision, round_to_nearest)

            # If we truncate now are we guaranteed to round correctly?
            if HU_distance >= HU_err:
                return self.convert(scaled_value, env)

            # Increase precision and try again
            parts_count += parts_count // 2 + 1

    def _multiply_finite(self, lhs, rhs):
        raise NotImplementedError

    def _divide_finite(self, lhs, rhs):
        raise NotImplementedError

    ##
    ## General computational operations.  The operands can be different formats;
    ## the destination format is self.
    ##

    def from_int(self, x, env):
        '''Convert the integer x to this floating point format, rounding if necessary.  Returns a
        (value, status) pair.
        '''
        raise NotImplementedError

    def add(self, lhs, rhs, env):
        '''Returns a (lhs + rhs, status) pair with dst_format as the format of the result.'''
        raise NotImplementedError

    def subtract(self, lhs, rhs, env):
        '''Returns a (lhs - rhs, status) pair with dst_format as the format of the result.'''
        raise NotImplementedError

    def multiply(self, lhs, rhs, env):
        '''Returns a (lhs * rhs, status) pair with dst_format as the format of the result.'''
        if lhs.e_biased == lhs.fmt.e_saturated:
            return self._multiply_special(lhs, rhs)
        if rhs.e_biased == rhs.fmt.e_saturated:
            return self._multiply_special(rhs, lhs)

        # Both numbers are finite.
        sign = lhs.sign ^ rhs.sign
        exponent = lhs.exponent() + rhs.exponent()
        return self.make_real(sign, exponent, lhs.significand * rhs.significand, env)

    def _multiply_special(self, lhs, rhs):
        '''Return a (lhs * rhs, status) pair with self as the format of the result.

        The LHS is a NaN or infinity.
        '''
        assert lhs.e_biased == lhs.fmt.e_saturated

        sign = lhs.sign ^ rhs.sign

        if lhs.significand == 0:
            # infinity * zero -> NaN
            if rhs.is_zero():
                return self.make_NaN(sign, True, 0, True)
            # infinity * NaN -> NaN
            if rhs.is_NaN():
                return self.make_NaN(sign, True, rhs.NaN_payload(), rhs.is_signalling())
            # infinity * infinity -> infinity
            # infinity * finite-non-zero -> infinity
            return self.make_infinity(sign), OpStatus.OK

        # NaN * anything -> NaN, but need to catch signalling NaNs
        return self.make_NaN(sign, True, lhs.NaN_payload(),
                             lhs.is_signalling() or rhs.is_signalling())

    def divide(self, lhs, rhs, env):
        '''Returns a (lhs / rhs, status) pair with dst_format as the format of the result.'''
        raise NotImplementedError

    def sqrt(self, x, env):
        '''Returns a (sqrt(x), status) pair with dst_format as the format of the result.'''
        raise NotImplementedError

    def fma(self, lhs, rhs, addend, env):
        '''Returns a (lhs * rhs + addend, status) pair with dst_format as the format of the
        result.'''
        raise NotImplementedError


IEEEhalf = FloatFormat(5, 11, InterchangeKind.IMPLICIT)
IEEEsingle = FloatFormat(8, 24, InterchangeKind.IMPLICIT)
IEEEdouble = FloatFormat(11, 53, InterchangeKind.IMPLICIT)
IEEEquad = FloatFormat(15, 113, InterchangeKind.IMPLICIT)
#IEEEoctuple = FloatFormat(18, 237, InterchangeKind.IMPLICIT)
# 80387 floating point takes place with a wide exponent range but rounds to single, double
# or extended precision.  It also has an explicit integer bit.
x87extended = FloatFormat(15, 64, InterchangeKind.EXPLICIT)
x87double = FloatFormat(15, 53, InterchangeKind.NONE)
x87single = FloatFormat(15, 24, InterchangeKind.NONE)


class IEEEfloat:
    # TODO: reconsider using an IEEE biased exponent internally.

    def __init__(self, fmt, sign, biased_exponent, significand):
        '''Create a floating point number with the given format, sign, biased exponent and
        significand.  For NaNs - saturing exponents with non-zero signficands - we interpret
        the significand as the binary payload below the quiet bit.
        '''
        if not isinstance(biased_exponent, int):
            raise TypeError('biased exponent must be an integer')
        if not isinstance(significand, int):
            raise TypeError('significand must be an integer')
        if not 0 <= biased_exponent <= fmt.e_saturated:
            raise ValueError(f'biased exponent {biased_exponent:,d} out of range')
        if biased_exponent in (0, fmt.e_saturated):
            if not 0 <= significand < fmt.int_bit:
                raise ValueError(f'significand {significand:,d} out of range for non-normal')
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

    def classify(self):
        '''Return which FloatClass this number is.'''
        # Zero or subnormal?
        if self.e_biased == 0:
            if self.significand:
                return FloatClass.nSubnormal if self.sign else FloatClass.pSubnormal
            return FloatClass.nZero if self.sign else FloatClass.pZero

        # Infinity or NaN?
        if self.e_biased == self.fmt.e_saturated:
            if self.significand:
                if self.significand & self.fmt.quiet_bit:
                    return FloatClass.qNaN
                else:
                    return FloatClass.sNaN
            else:
                return FloatClass.nInf if self.sign else FloatClass.pInf

        # Normal
        return FloatClass.nNormal if self.sign else FloatClass.pNormal

    def to_parts(self):
        '''Returns a triple: (sign, exponent, significand).

        Finite non-zero numbers have the magniture 2^exponent * significand (an integer).

        Zeroes have an exponent and significand of 0.  Infinities have an exponent of 'I'
        with significand zero, quiet NaNs an exponent of 'Q' and signalling NaNs an
        exponent of 'S' and in either case the significand is the payload (without the
        quiet bit).
        '''
        significand = self.significand
        if self.e_biased == self.fmt.e_saturated:
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
            exponent = self.exponent()

        return (self.sign, exponent, significand)

    def NaN_payload(self):
        '''Returns the NaN payload.  Raises RuntimeError if the value is not a NaN.'''
        if not self.is_NaN():
            raise RuntimeError(f'NaN_payload called on a non-NaN: {self.to_parts()}')
        return self.significand & self.fmt.quiet_bit - 1

    def is_negative(self):
        '''Return True if the sign bit is set.'''
        return self.sign

    def is_normal(self):
        '''Return True if the value is finite, non-zero and not denormal.'''
        return 0 < self.e_biased < self.fmt.e_saturated

    def is_finite(self):
        '''Return True if the value is finite.'''
        return self.e_biased < self.fmt.e_saturated

    def is_zero(self):
        '''Return True if the value is zero regardless of sign.'''
        return not self.e_biased and not self.significand

    def is_subnormal(self):
        '''Return True if the value is subnormal.'''
        return not self.e_biased and self.significand

    def is_infinite(self):
        '''Return True if the value is infinite.'''
        return self.e_biased == self.fmt.e_saturated and not self.significand

    def is_NaN(self):
        '''Return True if this is a NaN of any kind.'''
        return self.e_biased == self.fmt.e_saturated and self.significand

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
        return self.e_biased < self.fmt.e_saturated and (self.e_biased or self.significand)

    def radix(self):
        '''We're binary!'''
        return 2

    def exponent(self):
        '''Return the arithmetic exponent of our significand interpreted as an integer.

        Subnormal numbers mathematically have a biased exponent of 1 not 0.  Zero is only
        chosen so the interchange format can unambiguously omit the integer bit.  So here
        we must force it back to 1.
        '''
        assert self.is_finite()

        return max(1, self.e_biased) - self.fmt.e_bias - (self.fmt.precision - 1)

    def total_order(self, rhs):
        raise NotImplementedError

    def total_order_mag(self, rhs):
        raise NotImplementedError

    ##
    ## Quiet computational operations
    ##

    def copy(self):
        '''Returns a copy of this number.'''
        return IEEEfloat(self.fmt, self.sign, self.e_biased, self.significand)

    def copy_sign(self, y):
        '''Retuns a copy of this number but with the sign of y.'''
        return IEEEfloat(self.fmt, y.sign, self.e_biased, self.significand)

    def copy_negate(self):
        '''Returns a copy of this number with the opposite sign.'''
        return IEEEfloat(self.fmt, not self.sign, self.e_biased, self.significand)

    def copy_abs(self):
        '''Returns a copy of this number with sign False (positive).'''
        return IEEEfloat(self.fmt, False, self.e_biased, self.significand)

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

    def remainder(self, rhs):
        '''Set to the reaminder when divided by rhs.  Returns OpStatus.OK.

        If rhs != 0, the remainder is defined for finite operands as r = lhs - rhs * n,
        where n is the integer nearest the exact number lhs / rhs with round-to-even
        semantics.  The result is always exact, and if r is zero its sign shall be that of
        lhs.

        If rhs is zero or lhs is infinite, an invalid operation is returned if neither
        operand is a NaN.

        If rhs is infinite then the result is lhs if it is finite.
        '''
        raise NotImplementedError


    ##
    ## logBFormat operations (logBFormat is an integer format)
    ##

    def scalb(self, N, env):
        '''Returns a pair (scalb(x, N), status).

        The result is x * 2^N for integral values N, correctly-rounded. and in the format of
        x.  For non-zero values of N, scalb(±0, N) is ±0 and scalb(±∞) is ±∞.  For zero values
        of N, scalb(x, N) is x.
        '''
        raise NotImplementedError

    def logb(self, env):
        '''Return a pair (log2(x), status).

        The result is the exponent e of x, a signed integer, when represented with
        infinite range and minimum exponent.  Thus 1 <= scalb(x, -logb(x)) < 2 when x is
        positive and finite.  logb(1) is +0.  logb(NaN), logb(∞) and logb(0) return
        implementation-defined values outside the range ±2 * (emax + p - 1) and signal the
        invalid operation exception.
        '''
        raise NotImplementedError

    ##
    ## General computational operations
    ##

    def to_quiet(self):
        '''Return a pair (result, status).

        Result is a copy except that a signalling NaN becomes its quiet twin.  Status is
        OpStatus.INVALID in this case, otherwise OpStatus.OK.
        '''
        if self.is_signalling():
            return IEEEfloat(self.fmt, self.sign, self.e_biased,
                             self.significand | self.fmt.quiet_bit), OpStatus.INVALID
        return self.copy(), OpStatus.OK

    def next_up(self):
        '''Set to the smallest floating point value (unless operating on a positive infinity or
        NaN) that compares greater.
        '''
        return self.fmt.next_up(self, False)

    def next_down(self):
        '''Set to the greatest floating point value (unless operating on a negative infinity or
        NaN) that compares less.

        As per IEEE-754 next_down(x) = -next_up(-x)
        '''
        return self.fmt.next_up(self, True)

    def round(self, rounding_mode, exact=False):
        '''Round to the nearest integer retaining the input floating point format and return a
        (value, status) pair.

        This function flags INVALID on a signalling NaN input; if exact is True, the
        INEXACT flag is set in the status if the result does not have the same numerical
        value as the input.  In all other cases no flags are set.

        This function implmements all six functions whose names begin with "roundToIntegral"
        in the IEEE-754 standard.
        '''
        if self.e_biased == self.fmt.e_saturated:
            # Quiet NaNs and infinites stay unchanged; signalling NaNs are converted to quiet.
            return self.to_quiet()

        # Zeroes return unchanged.
        if self.significand == 0:
            return self.copy(), OpStatus.OK

        # Rounding-towards-zero is semantically equivalent to clearing zero or more of the
        # significand's least-significant bits.
        count = -self.exponent()

        # We're already an integer if count is <= 0
        if count <= 0:
            return self.copy(), OpStatus.OK

        # If count >= precision then we're a fraction and all bits are cleared, in which
        # case rounding away rounds to int_bit with exponent of 0.  Cap the count at
        # precision + 1, which still captures the lost fraction correctly.
        count = min(count, self.fmt.precision + 1)

        significand = self.significand
        lost_fraction = lost_bits_from_rshift(significand, count)

        # Clear the insignificant bits of the significand
        lsb = 1 << count
        significand &= ~(lsb - 1)

        # Round
        e_biased = self.e_biased
        if rounding_mode.rounds_away(lost_fraction, self.sign, bool(significand & lsb)):
            if lsb <= self.fmt.int_bit:
                significand += lsb
                # If the significand now overflows, halve it and increment the exponent
                if significand > self.fmt.max_significand:
                    significand >>= 1
                    e_biased += 1
            else:
                significand = self.fmt.int_bit
                e_biased = self.fmt.e_bias

        if exact and lost_fraction != LostFraction.EXACTLY_ZERO:
            status = OpStatus.INEXACT
        else:
            status = OpStatus.OK

        return IEEEfloat(self.fmt, self.sign, e_biased, significand), status

    def to_int(self, int_format, env):
        '''Returns a (int(lhs), status) pair correctly-rounded with int_format the integer format
        of the result.
        '''
        raise NotImplementedError

    def to_integral(self, flt_format, env):
        '''Returns a (lhs, status) pair correctly-rounded with flt_format the floating point
        format of the result.
        '''
        raise NotImplementedError

    def to_decimal_string(self, fmt_spec):
        '''Returns a (string, status) pair correctly-rounded with fmt_spec giving details of the
        output format required.
        '''
        raise NotImplementedError

    ##
    ## Signalling computational operations
    ##

    def compare_quiet_eq(self):
        raise NotImplementedError

    def compare_quiet_ne(self):
        raise NotImplementedError

    def compare_quiet_gt(self):
        raise NotImplementedError

    def compare_quiet_ge(self):
        raise NotImplementedError

    def compare_quiet_lt(self):
        raise NotImplementedError

    def compare_quiet_le(self):
        raise NotImplementedError

    def compare_quiet_un(self):
        raise NotImplementedError

    def compare_quiet_ng(self):
        raise NotImplementedError

    def compare_quiet_lu(self):
        raise NotImplementedError

    def compare_quiet_nl(self):
        raise NotImplementedError

    def compare_quiet_gu(self):
        raise NotImplementedError

    def compare_quiet_or(self):
        raise NotImplementedError

    def compare_signalling_eq(self):
        raise NotImplementedError

    def compare_signalling_ne(self):
        raise NotImplementedError

    def compare_signalling_gt(self):
        raise NotImplementedError

    def compare_signalling_ge(self):
        raise NotImplementedError

    def compare_signalling_lt(self):
        raise NotImplementedError

    def compare_signalling_le(self):
        raise NotImplementedError

    def compare_signalling_ng(self):
        raise NotImplementedError

    def compare_signalling_lu(self):
        raise NotImplementedError

    def compare_signalling_nl(self):
        raise NotImplementedError

    def compare_signalling_gu(self):
        raise NotImplementedError
