#
# An implementation of many operations of generic binary floating-point arithmetic
#
# (c) Neil Booth 2007-2021.  All rights reserved.
#

from enum import IntFlag, IntEnum

import attr


HEX_FLOAT_REGEX = re.compile(
    # sign[opt]
    '[-+]?'
    # (hex-integer[opt].fraction or hex-integer.[opt]) hex-exp-p exp-sign[opt]dec-exponent
    '(0x((([0-9a-f]*)\.([0-9a-f]+)|([0-9a-f]+)\.?)p?([-+]?[0-9]+))'
    # inf or infinity
    '|(inf(inity)?)'
    # nan-or-snan hex-payload-or-dec-payload[opt]
    '|(s?)nan((0x[0-9a-f]+)|([0-9]+))?)$',
    re.ASCII | re.IGNORECASE
)


class FloatClass(IntEnum):
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


# When bits of a floating point number are truncated, this is used to indicate what
# fraction of the LSB those bits represented.  It essentially combines the roles of guard
# and sticky bits.
class LostFraction(IntEnum):   # Example of truncated bits:
    EXACTLY_ZERO = 0           # 00000
    LESS_THAN_HALF = 1	       # 0xxxxx  x's not all zero
    EXACTLY_HALF = 2           # 100000
    MORE_THAN_HALF = 3         # 1xxxxx  x's not all zero


class RoundingMode:
    '''Rounding modes are implemented as derived classes.'''

    @classmethod
    def rounds_away(cls, _lost_fraction, _sign, _is_odd):
        '''Return True if, when an operation results in a lost fraction, the rounding mode
        requires rounding away from zero (i.e. incrementing the significand).

        sign is the sign of the number, and is_odd indicates if the LSB of the new
        significand is set, which is needed for ties-to-even rounding.
        '''
        raise NotImplementedError


class RoundTiesToEven(RoundingMode):
    '''Round-to-nearest with ties-to-even.'''

    @classmethod
    def rounds_away(cls, lost_fraction, _sign, is_odd):
        if lost_fraction == LostFraction.MORE_THAN_HALF:
            return True
        if lost_fraction == LostFraction.EXACTLY_HALF:
            return is_odd
        return False


class RoundTiesToAway(RoundingMode):
    '''Round-to-nearest with ties-to-away.'''

    @classmethod
    def rounds_away(cls, lost_fraction, _sign, _is_odd):
        return lost_fraction in {LostFraction.EXACTLY_HALF, LostFraction.MORE_THAN_HALF}


class RoundTowardsPositive(RoundingMode):
    '''Round towards positive infinity.'''

    @classmethod
    def rounds_away(cls, lost_fraction, sign, _is_odd):
        return not sign and lost_fraction != LostFraction.EXACTLY_ZERO


class RoundTowardsNegative(RoundingMode):
    '''Round towards negative infinity.'''

    @classmethod
    def rounds_away(cls, lost_fraction, sign, _is_odd):
        return sign and lost_fraction != LostFraction.EXACTLY_ZERO


class RoundTowardsZero(RoundingMode):

    @classmethod
    def rounds_away(cls, _lost_fraction, _sign, _is_odd):
        return False


class FloatEnv:

    __slots__ = ('rounding_mode', 'detect_tininess_before', 'always_flag_underflow')

    def __init__(self, rounding_mode, detect_tininess_before, always_flag_underflow):
        '''Floating point environment flags:

        rounding_mode is one of the RoundingMode enumerations and controls rounding of inexact
        results.

        If detect_tininess_before is True tininess is detected before rounding, otherwise after

        If always_flag_underflow is True then underflow is flagged whenever tininess is detected,
        otherwise it is only flagged if tininess is detected and the result is inexact.
        '''
        self.rounding_mode = rounding_mode
        self.detect_tininess_before = detect_tininess_before
        self.always_flag_underflow = always_flag_underflow


class FloatFormat:
    '''An IEEE-754 floating point format.'''

    __slots__ = ('precision', 'e_width', 'e_max', 'e_min', 'e_bias', 'e_saturated',
                 'size', 'ibit', 'max_significand')

    def __init__(self, e_width, precision, explicit_bit):
        '''e_width is the exponent width in bits.

        precision is the number of bits in the significand including an explicit integer
        bit.  The integer bit is implicit when a value is packed into interchange formats
        if explicit_bit is False.
        '''
        self.precision = precision
        self.e_width = e_width
        self.explicit_bit = explicit_bit
        # The largest e such that 2^e is representable.  The largest representable number
        # is 2^e_max * (2 - 2^(1 - precision)) when the significand is all ones.
        self.e_max = 1 << (e_width - 1) - 1
        # The smallest e such that 2^e is a normalized number
        self.e_min = 1 - e_max
        # The exponent bias
        self.e_bias = self.e_max
        # The biased exponent for infinities and NaNs has all bits 1
        self.e_saturated = (1 << e_width) - 1
        # The number of bytes needed to encode an FP number.  A sign bit, the exponent,
        # and the significand.
        self.size = (1 + e_width + explicit_bit + (self.precision - 1) + 7) // 8
        # The integer bit (MSB) in the significand
        self.int_bit = 1 << (self.precision - 1)
        # The quiet bit in NaNs
        self.quiet_bit = self.int_bit >> 1
        # Significands are unsigned bitstrings of precision bits
        self.max_significand = (1 << self.precision) - 1

    def make_zero(self, sign):
        '''Returns a zero of the given sign.'''
        return IEEEFloat(self, sign, 0, 0)

    def make_infinity(self, sign):
        '''Returns an infinity of the given sign.'''
        return IEEEFloat(self, sign, self.e_saturated, 0)

    def make_NaN(self, sign, is_quiet, payload):
        '''Returns a quiet NaN with the given payload.'''
        if is_quiet:
            if not 0 <= payload < self.quiet_bit:
                raise ValueError(f'invalid quiet NaN payload')
            payload += self.quiet_bit
        else:
            if not 1 <= payload < self.quiet_bit:
                raise ValueError(f'invalid signalling NaN payload')
        return IEEEFloat(self, sign, self.e_saturated, payload)

    def make_real(self, sign, exponent, significand, env):
        '''Return an (IEEEfloat, status) pair.

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

        # How many excess bits of precision do we have?
        excess = size - self.precision

        # Shift the significand and update the exponent
        significand, lost_fraction = self.shift_right(significand, excess)
        exponent += excess

        # Detect tininess now (we may overwrite later)
        is_tiny = exponent < self.e_min

        # Round
        if env.rounds_away(lost_fraction, sign, bool(significand & 1)):
            # Increment the significand
            significand += 1
            # If the significand now overflows, halve it and increment the exponent
            if significand > self.max_significand:
                significand >>= 1
                exponent += 1

        # If the new exponent would be too big, then we overflow
        if exponent > self.e_max:
            return self.make_infinity(sign), OpStatus.OVERFLOW | OpStatus.INEXACT

        # Otherwise detect tininess after rounding:
        if env.dt_after:
            is_tiny = exponent < self.e_min

        if lost_fraction == LostFraction.EXACTLY_ZERO:
            status = OpStatus.OK
        else:
            status = OpStatus.OP_INEXACT

        if is_tiny and (env.always_flag_underflow or status == OpStatus.OP_INEXACT):
            status |= OpStatus.UNDERFLOW

        return IEEEfloat(self, sign, exponent + self.e_bias, significand), status

    def from_string(cls, semantics, string, env):
        pass

    def pack(self, sign, biased_exponent, significand, endianness='little'):
        '''Returns a floating point value encoded as bytes of the given endianness.'''
        if not 0 <= significand <= self.max_significand:
            raise ValueError('significand out of range')
        if not 0 <= biased_exponent <= self.e_saturated:
            raise ValueError('biased exponent out of range')
        # Build up the bit representation
        value = biased_exponent + (self.e_saturated + 1) if sign else 0
        if self.explicit_bit:
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
        if len(binary) != self.size:
            raise ValueError(f'expected {self.size} bytes to unpack; got {len(binary)}')
        value = int.from_bytes(binary, endianness)

        significand = value & self.max_significand
        if self.explicit_bit:
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

    def hex_float(self, string, env):
        '''Converts a hexadecimal floating point string to a floating number of the
        required format.'''
        match = cls.HEX_FLOAT_REGEX.match(string)
        if match is None:
            raise ValueError(f'invalid hexadecimal float: {string}')
        sign = string[0] == '-'

        groups = match.groups
        if groups[2] is not None:
            # Floating point.  groups[3] is before the point and groups[4] after it.
            fraction = groups[4].rstrip('0')
            significand = int(groups[3] + fraction, 16)
            exponent = len(fraction) * -4
            return self.make_real(sign, exponent, significand, env):

        if groups[5] is not None:
            # Integer.  groups[3] is before the point and groups[4] after it.
            significand = int(groups[5], 16)
            return self.make_real(sign, 0, significand, env):

        if groups[7] is not None:
            # Infinity
            return self.make_infinity(sign)

        # NaN.  groups[9] is 's', 'S' or '' if this is a NaN; the s indicates signalling.
        assert groups[9] is not None
        # If it has a decimal payload groups[10] contains it
        # If it has a hex payload groups[11] contains it
        if groups[10] is not None:
            payload = int(groups[10])
        elif groups[11] is not None:
            payload = int(groups[11], 16)
        else:
            payload = 0
        if groups[9] == '':
            return make_quiet_NaN(self, sign, payload)
        return make_signalling_NaN(self, sign, payload)



IEEEhalf = FloatFormat(4, 11, False)
IEEEsingle = FloatFormat(7, 24, False)
IEEEdouble = FloatFormat(10, 53, False)
IEEEquad = FloatFormat(14, 113, False)
IEEEoctuple = FloatFormat(18, 237, False)
x87DoubleExtended = FloatFormat(14, 64, True)


# Operation status.  UNDERFLOW and OVERFLOW are always returned or-ed with INEXACT.
class OpStatus(IntFlag):
    OK          = 0
    INVALID_OP  = 0x01
    DIV_BY_ZERO = 0x02
    OVERFLOW    = 0x04
    UNDERFLOW   = 0x08
    INEXACT     = 0x10


class IEEEFloat:

    def __init__(self, format, sign, biased_exponent, significand):
        '''Create a floating point number with the given format, sign, biased exponent and
        significand.  For NaNs - saturing exponents with non-zero signficands - we interpret
        the significand as the binary payload below the quiet bit.
        '''
        self.format = format
        self.sign = is_negative
        self.e_biased = biased_exponent
        # The significand as an unsigned integer, or payload including quiet bit for NaNs
        self.significand = significand

    ##
    ## General non-computational operations.  They are never exceptional so simply return
    ## their results.
    ##

    def classify(self):
        '''Return which FloatClass this number is.'''
        # Zero or subnormal?
        if self.e_biased == 0:
            if self.significand:
                return FloatClass.nSubnormal if self.sign else FloatClass.pSubnormal
            return FloatClass.nZero if self.sign else FloatClass.pZero

        # Infinity or NaN?
        if self.e_biased == self.e_saturated:
            if self.significand:
                if self.significand & self.format.quiet_bit:
                    return FloatClass.qNaN
                else:
                    return FloatClass.sNaN
            else:
                return FloatClass.nInf if self.sign else FloatClass.pInf

        # Normal
        return FloatClass.nNormal is self.sign else FloatClass.pNormal

    def is_negative(self):
        '''Return True if the sign bit is set.'''
        return self.sign

    def is_normal(self):
        '''Return True if the value is finite, non-zero and not denormal.'''
        return 0 < self.exponent < self.semantics.e_saturated

    def is_finite(self):
        '''Return True if the value is finite.'''
        return self.exponent < self.semantics.e_saturated

    def is_zero(self):
        '''Return True if the value is zero regardless of sign.'''
        return not self.exponent and not self.significand

    def is_subnormal(self):
        '''Return True if the value is subnormal.'''
        return not self.exponent and self.significand

    def is_infinite(self):
        '''Return True if the value is infinite.'''
        return self.exponent == self.semantics.e_saturated and not self.significand

    def is_NaN(self):
        '''Return True if this is a NaN of any kind.'''
        return self.exponent == self.semantics.e_saturated and self.significand

    def is_signalling(self):
        '''Return True if and only if this is a signalling NaN.'''
        return self.is_NaN() and not (self.significand & self.format.quiet_bit)

    def is_canonical(self):
        '''We only have canonical values.'''
        # FIXME: how to extend this to packed formats
        return True

    # Not in IEEE-754
    def is_finite_non_zero(self):
        '''Return True if the value is finite and non-zero.'''
        return self.exponent < self.semantics.e_saturated and (self.exponent or self.significand)

    def radix(self):
        '''We're binary!'''
        return 2

    def total_order(self, rhs):
        raise NotImplementedError

    def total_order_mag(self, rhs):
        raise NotImplementedError

    def _quieten(self):
        assert self.category == FloatCategory.NAN
        if self.significand >= self.semantics.int_bit:
            return OpStatus.OK    # Already a quiet NaN
        self.significand += self.semantics.int_bit
        return OpStatus.INVALID_OP

    def next_up(self):
        if self.category == FloatCategory.FINITE_NON_ZERO:
            if self.sign:
                self.significand -= 1
                if self.significand == 0:
                    self.category = FloatCategory.ZERO
            else:
                self.significand += 1
                if self.significand > self.semantics.max_significand:
                    self.cateogry = FloatCategory.INFINITY
            return OpStatus.OK
        elif self.category == FloatCategory.ZERO:
            self.cateogry = FloatCategory.FINITE_NON_ZERO:
            self.sign = False
            self.significand = 1
            self.exponent = self.semantics.e_min
            return OpStatus.OK
        elif self.category == FloatCategory.INFINITY:
            if self.sign:
                self.category = FloatCategory.FINITE_NON_ZERO
                self.significand = self.semantics.max_significand
            return OpStatus.OK
        else:
            return _quieten(self)

    def next_down(self):
        self.sign = not self.sign
        status = self.next_up()
        self.sign = not self.sign
        return status

#     def _from_hex_string(string, index, rounding_mode):
#         '''Convert a hexadecimal floating point string to a floating point value.

#         char_iter returns characters after the leading 0x prefix.  The result is rounded
#         according to rounding_mode.  Return the operation status (OK, underflow, overflow,
#         inexact).
#         '''
#         significand = bytearray()
#         exponent = 0
#         dot_pos = None

#         # Skip leading zeroes and any decimal point
#         while
#   bitPos = partsCount * integerPartWidth;

#   /* Skip leading zeroes and any (hexa)decimal point.  */
#   p = skipLeadingZeroesAndAnyDot(p, &dot);
#   firstSigDigit = p;

#   lost_fraction = lfExactlyZero;
#   calced_lf = false;
#   for(;;) {
#     integerPart hex_value;

#     if(*p == '.') {
#       assert(dot == 0);
#       dot = p++;
#     }

#     hex_value = hexDigitValue(*p);
#     if(hex_value == -1U)
#       break;

#     p++;

#     /* Store the number whilst 4-bit nibbles remain.  */
#     if(bitPos) {
#       bitPos -= 4;
#       hex_value <<= bitPos % integerPartWidth;
#       significand[bitPos / integerPartWidth] |= hex_value;
#     } else if (!calced_lf) {
#       lost_fraction = trailingHexadecimalFraction(p, hex_value);
#       calced_lf = true;
#     }
#   }

#   /* Hex floats require an exponent but not a hexadecimal point.  */
#   assert(*p == 'p' || *p == 'P');

#   /* Ignore the exponent if we are zero.  */
#   if(p != firstSigDigit) {
#     int expAdjustment;

#     /* Implicit hexadecimal point?  */
#     if(!dot)
#       dot = p;

#     /* Calculate the exponent adjustment implicit in the number of
#        significant digits.  */
#     expAdjustment = dot - firstSigDigit;
#     if(expAdjustment < 0)
#       expAdjustment++;
#     expAdjustment = expAdjustment * 4 - 1;

#     /* Adjust for writing the significand starting at the most
#        significant nibble.  */
#     expAdjustment += semantics->precision;
#     expAdjustment -= partsCount * integerPartWidth;

#     /* Adjust for the given exponent.  */
#     readExponent(p + 1, expAdjustment, exponent);
#   }

#   return normalize(rounding_mode, lost_fraction);
# }
