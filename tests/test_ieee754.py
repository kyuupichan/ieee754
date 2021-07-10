import os
import re
from itertools import product

import pytest

from ieee754 import *
from ieee754 import HEX_SIGNIFICAND_PREFIX


all_IEEE_fmts = (IEEEhalf, IEEEsingle, IEEEdouble, IEEEquad)
all_roundings = (ROUND_CEILING, ROUND_FLOOR, ROUND_DOWN, ROUND_UP,
                 ROUND_HALF_EVEN, ROUND_HALF_UP, ROUND_HALF_DOWN)


def read_lines(filename):
    result = []
    with open(os.path.join('tests/data', filename)) as f:
        for line in f:
            hash_pos = line.find('#')
            if hash_pos != -1:
                line = line[:hash_pos]
            line = line.strip()
            if line:
                result.append(line)
    return result

def std_context():
    return Context(ROUND_HALF_EVEN)

def context_string_to_context(context):
    rounding = rounding_codes[context]
    return Context(rounding)

def read_significand(significand):
    if significand[:2] in ('0x', '0X'):
        return int(significand, 16)
    return int(significand)


def from_string(fmt, string):
    context = std_context()
    result = fmt.from_string(string, context)
    if HEX_SIGNIFICAND_PREFIX.match(string):
        assert context.flags & ~Flags.SUBNORMAL == 0
    else:
        assert context.flags & ~(Flags.UNDERFLOW | Flags.SUBNORMAL | Flags.INEXACT) == 0
    return result


boolean_codes = {
    'Y': True,
    'N': False,
}

compare_codes = {
    'L': Compare.LESS_THAN,
    'E': Compare.EQUAL,
    'G': Compare.GREATER_THAN,
    'U': Compare.UNORDERED,
}

format_codes = {
    'H': IEEEhalf,
    'S': IEEEsingle,
    'D': IEEEdouble,
    'Q': IEEEquad,
    'x': x87extended,
}

rounding_codes = {
    'E': ROUND_HALF_EVEN,
    'C': ROUND_CEILING,
    'F': ROUND_FLOOR,
    'D': ROUND_DOWN,
    'U': ROUND_UP,
    'u': ROUND_HALF_UP,
    'd': ROUND_HALF_DOWN,
}

status_codes = {
    'K': 0,
    'VI': Flags.OVERFLOW | Flags.INEXACT,
    'S': Flags.SUBNORMAL,
    'SI': Flags.SUBNORMAL | Flags.INEXACT,
    'U': Flags.UNDERFLOW | Flags.SUBNORMAL | Flags.INEXACT,
    'I': Flags.INEXACT,
    'Z': Flags.DIV_BY_ZERO,
    'X': Flags.INVALID,
    'XI': Flags.INVALID | Flags.INEXACT,
}

sNaN_codes = {
    'Y': 'sNaN',
    'N': '',
}

sign_codes = {
    '+': False,
    '-': True,
}

nan_payload_codes = {
    'N': 'N',
    'X': 'X',
    'D': 'D',
}


def to_text_format(hex_format):
    return TextFormat(
        force_leading_sign=boolean_codes[hex_format[0]],
        force_exp_sign=boolean_codes[hex_format[1]],
        rstrip_zeroes=boolean_codes[hex_format[2]],
        sNaN = sNaN_codes[hex_format[3]],
        nan_payload=nan_payload_codes[hex_format[4]],
    )


class TestTraps:

    def test_div_by_zero(self):
        context = Context(traps=Flags.DIV_BY_ZERO)
        lhs = IEEEsingle.make_real(False, 0, 1, context)
        rhs = IEEEsingle.make_zero(False)
        assert not context.flags
        assert issubclass(DivisionByZero, ZeroDivisionError)
        with pytest.raises(DivisionByZero):
            IEEEsingle.divide(lhs, rhs, context)
        assert context.flags == Flags.DIV_BY_ZERO
        context.clear_flags()
        assert not context.flags
        context.clear_traps()
        result = IEEEsingle.divide(lhs, rhs, context)
        assert result.is_infinite()
        assert context.flags == Flags.DIV_BY_ZERO

    def test_inexact(self):
        context = Context(traps=Flags.INEXACT)
        lhs = IEEEsingle.make_real(False, 0, 1, context)
        rhs = IEEEsingle.make_real(False, 0, 3, context)
        with pytest.raises(Inexact):
            IEEEsingle.divide(lhs, rhs, context)
        assert context.flags == Flags.INEXACT
        context.clear_flags()
        context.clear_traps()
        result = IEEEsingle.divide(lhs, rhs, context)
        assert result.is_finite()
        assert context.flags == Flags.INEXACT

    def test_invalid_operation(self):
        context = Context(traps=Flags.INVALID)
        lhs = IEEEsingle.make_zero(False)
        rhs = IEEEsingle.make_zero(False)
        with pytest.raises(InvalidOperation):
            IEEEsingle.divide(lhs, rhs, context)
        assert context.flags == Flags.INVALID
        context.clear_flags()
        context.clear_traps()
        result = IEEEsingle.divide(lhs, rhs, context)
        assert result.is_NaN()
        assert context.flags == Flags.INVALID

    def test_invalid_operation_inexact(self):
        context = Context(traps=Flags.INVALID)
        lhs = IEEEdouble.make_NaN(False, False, 0x123456789, context, False)
        assert not context.flags
        assert issubclass(InvalidOperationInexact, InvalidOperation)
        assert issubclass(InvalidOperationInexact, Inexact)
        with pytest.raises(InvalidOperationInexact):
            IEEEsingle.convert(lhs, context)
        assert context.flags == Flags.INVALID | Flags.INEXACT
        context.clear_flags()
        context.clear_traps()
        result = IEEEsingle.convert(lhs, context)
        assert result.is_NaN()
        assert context.flags == Flags.INVALID | Flags.INEXACT

    def test_overflow(self):
        context = Context(traps=Flags.OVERFLOW)
        lhs = IEEEsingle.make_real(False, 0, 1, context)
        rhs = IEEEsingle.make_real(False, -140, 1, context)
        assert context.flags == Flags.SUBNORMAL
        context.clear_flags()
        assert issubclass(Overflow, Inexact)
        with pytest.raises(Overflow):
            IEEEsingle.divide(lhs, rhs, context)
        assert context.flags == Flags.OVERFLOW | Flags.INEXACT
        context.clear_flags()
        context.clear_traps()
        result = IEEEsingle.divide(lhs, rhs, context)
        assert result.is_infinite()
        assert context.flags == Flags.OVERFLOW | Flags.INEXACT

    def test_subnormal_exact(self):
        context = Context(traps=Flags.SUBNORMAL)
        assert issubclass(SubnormalExact, Subnormal)
        assert not issubclass(SubnormalExact, Inexact)
        with pytest.raises(SubnormalExact):
            IEEEsingle.make_real(False, -140, 1, context)
        assert context.flags == Flags.SUBNORMAL
        context.clear_flags()
        context.clear_traps()
        result = IEEEsingle.make_real(False, -140, 1, context)
        assert result.is_subnormal()
        assert context.flags == Flags.SUBNORMAL

    def test_subnormal_inexact(self):
        # This is a rare case - the result must be rounded to normal, otherwise Underflow
        # would be raised
        context = Context(traps=Flags.SUBNORMAL)
        assert issubclass(SubnormalInexact, Subnormal)
        assert issubclass(SubnormalInexact, Inexact)
        with pytest.raises(SubnormalInexact):
            IEEEdouble.from_string('0x1.fffffffffffffp-1023', context)
        assert context.flags == Flags.SUBNORMAL | Flags.INEXACT
        context.clear_flags()
        context.clear_traps()
        result = IEEEdouble.from_string('0x1.fffffffffffffp-1023', context)
        assert result.is_normal()
        assert context.flags == Flags.SUBNORMAL | Flags.INEXACT

    def test_underflow_to_non_zero(self):
        context = Context(traps=Flags.UNDERFLOW)
        assert issubclass(Underflow, SubnormalInexact)
        with pytest.raises(Underflow):
            IEEEdouble.from_string('0x1.fffffffffffffp-1024', context)
        assert context.flags == Flags.UNDERFLOW | Flags.SUBNORMAL | Flags.INEXACT
        context.clear_flags()
        context.clear_traps()
        result = IEEEdouble.from_string('0x1.fffffffffffffp-1024', context)
        assert result.is_subnormal()
        assert context.flags == Flags.UNDERFLOW | Flags.SUBNORMAL | Flags.INEXACT

    def test_underflow_to_zero(self):
        context = Context(traps=Flags.UNDERFLOW)
        with pytest.raises(Underflow):
            IEEEdouble.from_string('0x1.fffffffffffffp-1200', context)
        assert context.flags == Flags.UNDERFLOW | Flags.SUBNORMAL | Flags.INEXACT
        context.clear_flags()
        context.clear_traps()
        result = IEEEdouble.from_string('0x1.fffffffffffffp-1200', context)
        assert result.is_zero()
        assert context.flags == Flags.UNDERFLOW | Flags.SUBNORMAL | Flags.INEXACT


class TestBinaryFormat:

    @pytest.mark.parametrize('fmt, is_if, precision, e_max', (
        (IEEEhalf, True, 11, 15),
        (IEEEsingle, True, 24, 127),
        (IEEEdouble, True, 53, 1023),
        (IEEEquad, True, 113, 16383),
        (x87extended, True, 64, 16383),
        (x87double, False, 53, 16383),
        (x87single, False, 24, 16383),
    ))
    def test_interchange(self, fmt, is_if, precision, e_max):
        assert bool(fmt.fmt_width) is is_if
        assert fmt.precision == precision
        assert fmt.e_max == e_max
        assert fmt.e_min == 1 - fmt.e_max


class TestIntegerFormat:

    @pytest.mark.parametrize('width, is_signed, min_int, max_int',(
        (8, True, -128, 127),
        (8, False, 0, 255),
        (32, True, -(1 << 31), (1 << 31) - 1),
        (32, False, 0, (1 << 32) - 1)
    ))
    def test_integer_format(self, width, is_signed, min_int, max_int):
        fmt = IntegerFormat(width, is_signed)
        assert fmt.min_int == min_int
        assert fmt.max_int == max_int
        assert fmt.width == width
        assert fmt.is_signed == is_signed


# Test basic class functions before reading test files
class TestGeneralNonComputationalOps:

    @pytest.mark.parametrize('fmt, sign',
                             product(all_IEEE_fmts,
                                     (False, True)
                             ))
    def test_make_zero(self, fmt, sign):
        value = fmt.make_zero(sign)
        if sign:
            assert value.number_class() == '-Zero'
            assert value.is_negative()
        else:
            assert value.number_class() == '+Zero'
            assert not value.is_negative()
        assert not value.is_normal()
        assert value.is_finite()
        assert not value.is_subnormal()
        assert not value.is_infinite()
        assert not value.is_NaN()
        assert not value.is_signalling()
        assert value.is_canonical()
        assert not value.is_finite_non_zero()
        assert value.radix() == 2

    @pytest.mark.parametrize('fmt, sign',
                             product(all_IEEE_fmts,
                                     (False, True)
                             ))
    def test_make_infinity(self, fmt, sign):
        value = fmt.make_infinity(sign)
        if sign:
            assert value.number_class() == '-Infinity'
            assert value.is_negative()
        else:
            assert value.number_class() == '+Infinity'
            assert not value.is_negative()
        assert not value.is_normal()
        assert not value.is_finite()
        assert not value.is_subnormal()
        assert value.is_infinite()
        assert not value.is_NaN()
        assert not value.is_signalling()
        assert value.is_canonical()
        assert not value.is_finite_non_zero()
        assert value.radix() == 2

    @pytest.mark.parametrize('fmt, sign, quiet, payload',
                             product(all_IEEE_fmts,
                                     (False, True),
                                     (False, True),
                                     (0, 1, 24),
                             ))
    def test_make_NaN(self, fmt, sign, quiet, payload):
        context = std_context()
        value = fmt.make_NaN(sign, quiet, payload, context, False)
        if payload == 0 and not quiet:
            assert context.flags == Flags.INEXACT
            payload = 1
        else:
            assert context.flags == 0
        if quiet:
            assert value.number_class() == 'NaN'
        else:
            assert value.number_class() == 'sNaN'
        if sign:
            assert value.is_negative()
        else:
            assert not value.is_negative()
        assert not value.is_normal()
        assert not value.is_finite()
        assert not value.is_subnormal()
        assert not value.is_infinite()
        assert value.is_NaN()
        assert value.is_signalling() is not quiet
        assert value.is_canonical()
        assert not value.is_finite_non_zero()
        assert value.radix() == 2
        assert value.as_tuple()[-1] == payload

    @pytest.mark.parametrize('fmt, sign',
                             product(all_IEEE_fmts,
                                     (False, True),
                             ))
    def test_make_NaN_quiet_payload(self, fmt, sign):
        context = std_context()
        fmt.make_NaN(sign, True, 0, context, False)
        fmt.make_NaN(sign, True, fmt.quiet_bit - 1, context, False)
        with pytest.raises(ValueError):
            fmt.make_NaN(sign, True, -1, context, False)
        with pytest.raises(TypeError):
            fmt.make_NaN(sign, True, 1.2, context, False)
        with pytest.raises(TypeError):
            fmt.make_NaN(sign, True, 1.2, context, False)

    @pytest.mark.parametrize('fmt, sign',
                             product(all_IEEE_fmts,
                                     (False, True),
                             ))
    def test_make_NaN_signalling_payload(self, fmt, sign):
        context = std_context()
        fmt.make_NaN(sign, False, fmt.quiet_bit - 1, context, False)
        with pytest.raises(ValueError):
            fmt.make_NaN(sign, False, -1, context, False)
        with pytest.raises(TypeError):
            fmt.make_NaN(sign, False, 1.2, context, False)
        with pytest.raises(TypeError):
            fmt.make_NaN(sign, False, 1.2, context, False)

    @pytest.mark.parametrize('fmt, sign',
                             product(all_IEEE_fmts, (False, True),
                             ))
    def test_make_real_MSB_set(self, fmt, sign):
        '''Test MSB set with various exponents.'''
        significand = 1
        for exponent in (
                fmt.e_min - (fmt.precision - 1),
                fmt.e_min - 1,
                fmt.e_min,
                -1,
                0,
                1,
                fmt.e_max
        ):
            context = Context(ROUND_HALF_EVEN)
            value = fmt.make_real(sign, exponent, significand, context)
            if exponent < fmt.e_min:
                assert context.flags == Flags.SUBNORMAL
                assert not value.is_normal()
                assert value.is_subnormal()
                if sign:
                    assert value.number_class() == '-Subnormal'
                else:
                    assert value.number_class() == '+Subnormal'
            else:
                assert context.flags == 0
                assert value.is_normal()
                assert not value.is_subnormal()
                if sign:
                    assert value.number_class() == '-Normal'
                else:
                    assert value.number_class() == '+Normal'
            if sign:
                assert value.is_negative()
            else:
                assert not value.is_negative()
            assert value.is_finite()
            assert not value.is_infinite()
            assert not value.is_NaN()
            assert not value.is_signalling()
            assert value.is_canonical()
            assert value.is_finite_non_zero()
            assert value.radix() == 2

    @pytest.mark.parametrize('fmt, sign, exponent',
                             product(all_IEEE_fmts,
                                     (False, True),
                                     (-1, 0, 1, (1 << 200)),
                             ))
    def test_make_real_zero_significand(self, fmt, sign, exponent):
        # Test that a zero significand gives a zero regardless of exponent
        context = std_context()
        value = fmt.make_real(sign, exponent, 0, context)
        assert context.flags == 0
        assert value.is_zero()
        assert value.sign is sign
        assert value.fmt is fmt

    @pytest.mark.parametrize('fmt, sign, two_bits, rounding',
                             product(all_IEEE_fmts,
                                     (False, True),
                                     (1, 2, 3, ),
                                     all_roundings,
                             ))
    def test_make_real_underflow_to_zero(self, fmt, sign, two_bits, rounding):
        # Test that a value that loses two bits of precision underflows correctly
        context = Context(rounding)
        value = fmt.make_real(sign, fmt.e_min - 2 - (fmt.precision - 1), two_bits, context)
        underflows_to_zero = (rounding in {ROUND_HALF_EVEN, ROUND_HALF_DOWN} and two_bits in (1, 2)
                              or (rounding == ROUND_HALF_UP and two_bits == 1)
                              or (rounding == ROUND_CEILING and sign)
                              or (rounding == ROUND_FLOOR and not sign)
                              or (rounding == ROUND_DOWN))
        assert context.flags == Flags.INEXACT | Flags.UNDERFLOW | Flags.SUBNORMAL
        if underflows_to_zero:
            assert value.is_zero()
        else:
            assert value.is_subnormal()
            assert value.significand == 1
        assert value.sign is sign
        assert value.fmt is fmt

    @pytest.mark.parametrize('fmt, sign, rounding',
                             product(all_IEEE_fmts,
                                     (False, True),
                                     all_roundings,
                             ))
    def test_make_overflow(self, fmt, sign, rounding):
        context = Context(rounding)
        # First test the exponent that doesn't overflow but that one more would
        exponent = fmt.e_max
        value = fmt.make_real(sign, exponent, 1, context)
        assert context.flags == 0
        assert value.is_normal()
        assert value.sign is sign
        assert value.fmt is fmt

        # Increment the exponent.  Overflow now depends on rounding mode
        exponent += 1
        value = fmt.make_real(sign, exponent, 1, context)
        assert context.flags == Flags.OVERFLOW | Flags.INEXACT
        if (rounding in {ROUND_HALF_EVEN, ROUND_HALF_DOWN, ROUND_HALF_UP, ROUND_UP}
                or (rounding == ROUND_CEILING and not sign)
                or (rounding == ROUND_FLOOR and sign)):
            assert value.is_infinite()
        else:
            assert value.is_normal()
        assert value.sign is sign
        assert value.fmt is fmt

    @pytest.mark.parametrize('fmt, sign, e_selector, two_bits, rounding',
                             product(all_IEEE_fmts,
                                     (False, True),
                                     range(0, 3),
                                     (0, 1, 2, 3, ),
                                     all_roundings,
                             ))
    def test_make_real_overflows_significand(self, fmt, sign, e_selector, two_bits, rounding):
        # Test cases where rounding away causes significand to overflow
        context = Context(rounding)
        # Minimimum good, maximum good, overflows to infinity
        exponent = [fmt.e_min - 2, fmt.e_max - 3, fmt.e_max - 2][e_selector]
        # two extra bits in the significand
        significand = two_bits + (fmt.max_significand << 2)
        value = fmt.make_real(sign, exponent - (fmt.precision - 1), significand, context)
        rounds_away = (two_bits and
                       ((rounding == ROUND_HALF_EVEN and two_bits in (2, 3))
                        or rounding == ROUND_UP
                        or (rounding == ROUND_HALF_DOWN and two_bits == 3)
                        or (rounding == ROUND_HALF_UP and two_bits in (2, 3))
                        or (rounding == ROUND_CEILING and not sign)
                        or (rounding == ROUND_FLOOR and sign)))
        if rounds_away:
            if e_selector == 2:
                assert context.flags == Flags.INEXACT | Flags.OVERFLOW
                assert value.is_infinite()
            else:
                assert context.flags == Flags.INEXACT
                assert value.is_normal()
                assert value.significand == fmt.int_bit
                assert value.e_biased == exponent + fmt.e_bias + 3
        else:
            assert context.flags == (Flags.INEXACT if two_bits else 0)
            assert value.is_normal()
            assert value.significand == fmt.max_significand
            assert value.e_biased == exponent + fmt.e_bias + 2
        assert value.sign is sign
        assert value.fmt is fmt

    @pytest.mark.parametrize('fmt, sign, two_bits, rounding',
                             product(all_IEEE_fmts,
                                     (False, True),
                                     (0, 1, 2, 3, ),
                                     all_roundings,
                             ))
    def test_make_real_subnormal_to_normal(self, fmt, sign, two_bits, rounding):
        # Test cases where rounding away causes a subnormal to normalize
        context = Context(rounding)
        # an extra bit in the significand with two LSBs varying
        significand = two_bits + ((fmt.max_significand >> 1) << 2)
        value = fmt.make_real(sign, fmt.e_min - 2 - (fmt.precision - 1), significand, context)
        rounds_away = (two_bits and
                       ((rounding == ROUND_HALF_EVEN and two_bits in (2, 3))
                        or rounding == ROUND_UP
                        or (rounding == ROUND_HALF_DOWN and two_bits == 3)
                        or (rounding == ROUND_HALF_UP and two_bits in (2, 3))
                        or (rounding == ROUND_CEILING and not sign)
                        or (rounding == ROUND_FLOOR and sign)))
        if rounds_away:
            assert context.flags == Flags.INEXACT | Flags.SUBNORMAL
            assert value.is_normal()
            assert value.significand == fmt.int_bit
            assert value.e_biased == 1
        else:
            if two_bits == 0:
                assert context.flags == Flags.SUBNORMAL
            else:
                assert context.flags == Flags.INEXACT | Flags.UNDERFLOW | Flags.SUBNORMAL
            assert value.is_subnormal()
            assert value.significand == fmt.int_bit - 1
            assert value.e_biased == 1
        assert value.sign is sign
        assert value.fmt is fmt


class TestUnaryOps:

    @pytest.mark.parametrize('line', read_lines('format_decimal.txt'))
    def test_format_decimal(self, line):
        parts = line.split()
        if len(parts) != 10:
            assert False, f'bad line: {line}'
        (exp_digits, force_exp_sign, force_leading_sign, force_point, upper_case,
         rstrip_zeroes, sign, digits, exponent, answer) = parts
        text_format = TextFormat(exp_digits=int(exp_digits),
                                 force_exp_sign=boolean_codes[force_exp_sign],
                                 force_leading_sign=boolean_codes[force_leading_sign],
                                 force_point=boolean_codes[force_point],
                                 upper_case=boolean_codes[upper_case],
                                 rstrip_zeroes=boolean_codes[rstrip_zeroes])
        sign = boolean_codes[sign]
        exponent = int(exponent)
        decimal_fmt = DecimalFormat(len(digits), 999, -999)
        decimal = Decimal(decimal_fmt, sign, exponent, digits)
        assert text_format.format_decimal(decimal) == answer

    @pytest.mark.parametrize('line', read_lines('from_string.txt'))
    def test_from_string(self, line):
        parts = line.split()
        if len(parts) == 1:
            hex_str, = parts
            with pytest.raises(SyntaxError):
                IEEEsingle.from_string(hex_str, std_context())
        elif len(parts) in (5, 7):
            fmt, context, test_str, status = parts[:4]
            fmt = format_codes[fmt]
            context = context_string_to_context(context)
            result = fmt.from_string(test_str, context)
            status = status_codes[status]

            if len(parts) == 5:
                answer = parts[-1]
                input_context = std_context()
                answer = fmt.from_string(answer, input_context)
                assert input_context.flags & ~Flags.SUBNORMAL == 0
                answer_tuple = answer.as_tuple()
            else:
                sign, exponent, significand = parts[-3:]
                sign = sign_codes[sign]
                try:
                    exponent = int(exponent)
                except ValueError:
                    pass
                significand = read_significand(significand)
                answer_tuple = (sign, exponent, significand)

            assert result.as_tuple() == answer_tuple
            assert context.flags == status
        else:
            assert False, f'bad line: {line}'

    @pytest.mark.parametrize('line, kind, exact', product(
        read_lines('round_to_integral.txt'),
        ('round', (8, True), (8, False), (64, True), (64, False)),
        (False, True),
    ))
    def test_to_integer(self, line, kind, exact):
        parts = line.split()
        if len(parts) != 5:
            assert False, f'bad line: {line}'
        fmt, rounding, value, status, answer_str = parts
        fmt = format_codes[fmt]
        value = from_string(fmt, value)
        status = status_codes[status]
        answer = from_string(fmt, answer_str)

        if kind == 'round':
            if exact:
                context = Context(rounding_codes[rounding])
                result = value.round_to_integral_exact(context)
                assert result.as_tuple() == answer.as_tuple()
                assert context.flags == status
            else:
                context = Context()
                result = value.round_to_integral(rounding_codes[rounding], context)
                assert result.as_tuple() == answer.as_tuple()
                assert context.flags == status & ~Flags.INEXACT
        else:
            integer_format = IntegerFormat(*kind)
            context = Context()
            if answer.is_NaN():
                status = Flags.INVALID
                answer = 0
            elif answer.is_infinite():
                status = Flags.INVALID
                answer = integer_format.min_int if answer.sign else integer_format.max_int
            else:
                answer = int(answer_str)
                if answer < integer_format.min_int:
                    answer, status = integer_format.min_int, Flags.INVALID
                elif answer > integer_format.max_int:
                    answer, status = integer_format.max_int, Flags.INVALID
            if exact:
                result = value.convert_to_integer_exact(integer_format, rounding_codes[rounding],
                                                        context)
                assert result == answer
                assert context.flags == status
            else:
                result = value.convert_to_integer(integer_format, rounding_codes[rounding],
                                                  context)
                assert result == answer
                assert context.flags == status & ~Flags.INEXACT


    @pytest.mark.parametrize('line', read_lines('convert.txt'))
    def test_convert(self, line):
        parts = line.split()
        if len(parts) != 6:
            assert False, f'bad line: {line}'
        src_fmt, context, src_value, dst_fmt, status, answer = parts
        src_fmt = format_codes[src_fmt]
        context = context_string_to_context(context)
        input_context = std_context()
        src_value = src_fmt.from_string(src_value, input_context)
        assert input_context.flags & ~Flags.SUBNORMAL == 0
        dst_fmt = format_codes[dst_fmt]
        status = status_codes[status]
        answer = dst_fmt.from_string(answer, input_context)
        assert input_context.flags & ~Flags.SUBNORMAL == 0

        result = dst_fmt.convert(src_value, context)
        assert result.as_tuple() == answer.as_tuple()
        assert context.flags == status

    @pytest.mark.parametrize('line', read_lines('from_int.txt'))
    def test_from_int(self, line):
        parts = line.split()
        if len(parts) != 5:
            assert False, f'bad line: {line}'
        dst_fmt, context, src_value, status, answer = parts

        dst_fmt = format_codes[dst_fmt]
        context = context_string_to_context(context)
        value = int(src_value)
        assert str(value) == src_value
        status = status_codes[status]
        answer = dst_fmt.from_string(answer, context)
        assert context.flags == 0

        result = dst_fmt.from_int(value, context)
        assert result.as_tuple() == answer.as_tuple()
        assert context.flags == status

    @pytest.mark.parametrize('line', read_lines('to_hex.txt'))
    def test_to_hex(self, line):
        parts = line.split()
        if len(parts) != 7:
            assert False, f'bad line: {line}'
        text_format, context, src_fmt, in_str, dst_fmt, status, answer = parts
        text_format = to_text_format(text_format)
        context = context_string_to_context(context)
        src_fmt = format_codes[src_fmt]
        input_context = std_context()
        in_value = src_fmt.from_string(in_str, input_context)
        assert input_context.flags & ~Flags.SUBNORMAL == 0
        dst_fmt = format_codes[dst_fmt]
        status = status_codes[status]

        result = dst_fmt.to_hex(in_value, text_format, context)
        assert result == answer
        assert context.flags == status

    @pytest.mark.parametrize('line', read_lines('to_decimal.txt'))
    def test_to_decimal(self, line):
        parts = line.split()
        if len(parts) != 5:
            assert False, f'bad line: {line}'
        context, fmt, in_str, status, answer = parts
        fmt = format_codes[fmt]
        context = context_string_to_context(context)
        value = from_string(fmt, in_str)
        status = status_codes[status]
        text_format = TextFormat(exp_digits=-2, force_exp_sign=True, rstrip_zeroes=True)

        decimal = value.to_decimal(context=context)
        result = text_format.format_decimal(decimal)
        assert result == answer
        assert context.flags == status

        # Confirm the round-trip: reading in the decimal value gives the same as the hex
        # value
        context.clear_flags()
        dec_value = fmt.from_string(answer, context)
        assert dec_value.as_tuple() == value.as_tuple()
        assert context.flags == status

        # Confirm Python prints the same.
        if '0x' in in_str:
            value = float.fromhex(in_str)
        else:
            value = float(in_str)
        py_str = str(value)
        if py_str.endswith('.0'):
            py_str = py_str[:-2]
        assert py_str == answer

    # def test_to_decimal_hard(self):
    #     text_format = TextFormat(exp_digits=-2, force_exp_sign=True)
    #     for exponent in range(-100, 101):
    #         hex_str = f'0x1p{exponent}'
    #         value = IEEEdouble.from_string(hex_str, Context())
    #         context = Context()
    #         dec_str = value.to_decimal(text_format, 0, context)
    #         flags = 'I' if context.flags & Flags.INEXACT else 'K'
    #         print(f'E D {hex_str} {flags} {dec_str}')
    #     assert False

    @pytest.mark.parametrize('line', read_lines('scaleb.txt'))
    def test_scaleb(self, line):
        parts = line.split()
        if len(parts) != 6:
            assert False, f'bad line: {line}'
        fmt, context, in_str, N_str, status, answer = parts
        fmt = format_codes[fmt]
        context = context_string_to_context(context)
        in_context = std_context()
        in_value = fmt.from_string(in_str, in_context)
        assert in_context.flags & ~Flags.SUBNORMAL == 0
        answer = fmt.from_string(answer, in_context)
        assert in_context.flags & ~Flags.SUBNORMAL == 0
        N = int(N_str)
        assert str(N) == N_str
        status = status_codes[status]

        result = in_value.scaleb(N, context)
        assert result.as_tuple() == answer.as_tuple()
        assert context.flags == status

    @pytest.mark.parametrize('fmt', all_IEEE_fmts)
    def test_logb_specials(self, fmt):
        # Test all 3 values are different
        values = {fmt.logb_zero, fmt.logb_inf, fmt.logb_NaN}
        assert len(values) == 3
        extremity = 2 * (max(fmt.e_max, abs(fmt.e_min)) + fmt.precision - 1)
        assert min(abs(value) for value in values) > extremity

    @pytest.mark.parametrize('line', read_lines('logb.txt'))
    def test_logb(self, line):
        parts = line.split()
        if len(parts) != 4:
            assert False, f'bad line: {line}'
        fmt, in_str, status, answer = parts
        fmt = format_codes[fmt]
        in_context = std_context()
        in_value = fmt.from_string(in_str, in_context)
        assert in_context.flags & ~Flags.SUBNORMAL == 0
        answer = fmt.from_string(answer, in_context)
        assert in_context.flags & ~Flags.SUBNORMAL == 0
        status = status_codes[status]

        context = std_context()
        result = in_value.logb(context)
        assert result.as_tuple() == answer.as_tuple()
        assert context.flags == status

        # Now test logb_integral
        context.clear_flags()
        result_integral = in_value.logb_integral(context)
        if result.is_finite():
            value = result.significand >> -result.exponent_int()
            if result.sign:
                value = -value
            assert value == result_integral
            assert context.flags == 0
        else:
            if result.is_infinite():
                if result.sign:
                    assert result_integral == fmt.logb_zero
                else:
                    assert result_integral == fmt.logb_inf
            else:
                assert result_integral == fmt.logb_NaN
            assert context.flags == Flags.INVALID

    @pytest.mark.parametrize('line', read_lines('next_up.txt'))
    def test_next(self, line):
        # Tests next_up and next_down
        parts = line.split()
        if len(parts) != 4:
            assert False, f'bad line: {line}'
        context = std_context()
        fmt, in_str, status, answer = parts
        fmt = format_codes[fmt]
        in_value = fmt.from_string(in_str, context)
        assert context.flags & ~Flags.SUBNORMAL == 0
        answer = fmt.from_string(answer, context)
        assert context.flags & ~Flags.SUBNORMAL == 0
        context.clear_flags()
        status = status_codes[status]

        result = in_value.next_up(context)
        assert result.as_tuple() == answer.as_tuple()
        assert context.flags == status

        # Now for next_down
        context.clear_flags()
        in_value = in_value.copy_negate()
        answer = answer.copy_negate()
        result = in_value.next_down(context)
        assert result.as_tuple() == answer.as_tuple()
        assert context.flags == status

    @pytest.mark.parametrize('line', read_lines('sqrt.txt'))
    def test_sqrt(self, line):
        # Tests next_up and next_down
        parts = line.split()
        if len(parts) != 6:
            assert False, f'bad line: {line}'
        context, fmt, value, dst_fmt, status, answer = parts
        fmt = format_codes[fmt]
        dst_fmt = format_codes[dst_fmt]
        context = context_string_to_context(context)
        value = from_string(fmt, value)
        status = status_codes[status]
        answer = from_string(dst_fmt, answer)

        result = dst_fmt.sqrt(value, context)
        assert result.as_tuple() == answer.as_tuple()
        assert context.flags == status

    @pytest.mark.parametrize('endianness', ('big', 'little'))
    def test_pack_unpack_round_trip(self, endianness):
        for value in range(0, 65536):
            binary = value.to_bytes(2, endianness)
            parts = IEEEhalf.unpack(binary, endianness)
            packed_value = IEEEhalf.pack(*parts, endianness)
            assert binary == packed_value

    def test_x87_pseudos(self):
        # 3fff9180000000000000 is the canonical representation of 0x1.23p0.  Clear its
        # integer bit (making it an unnormal) and check
        for hex_str in ('3fff9180000000000000', '3fff1180000000000000'):
            value = x87extended.unpack_value(bytes.fromhex(hex_str), 'big')
            assert value.to_hex(TextFormat(), Context()) == '0x1.2300000000000000p0'

        # 7fffc000000000000000 is the canonical representation of a NaN with integer bit
        # set.  Test clearing it (a pseudo-NaN) gives the right answer.
        for hex_str in ('7fffc000000000000000', '7fff4000000000000000'):
            value = x87extended.unpack_value(bytes.fromhex(hex_str), 'big')
            assert value.is_NaN()
            assert not value.is_signalling()
            assert value.NaN_payload() == 0

        # 7fff8000000000000000 is the canonical representation of +Inf with integer bit
        # set.  Test clearing it gives the right answer.
        for hex_str in ('7fff8000000000000000', '7fff0000000000000000'):
            value = x87extended.unpack_value(bytes.fromhex(hex_str), 'big')
            assert value.to_hex(TextFormat(), Context()) == 'Inf'

        # 00000000a03000000000 is the canonical representation of 0x1.85p-16400, or
        # 0x0.0000614p-16382, a subnormal with integer bit clear.  Test setting it gives
        # the right answer.
        answer = x87extended.from_string('0x1.85p-16400', Context())
        for hex_str in ('0000000030a000000000', '0000800030a000000000'):
            value = x87extended.unpack_value(bytes.fromhex(hex_str), 'big')
            assert value.as_tuple() == answer.as_tuple()

    def test_pack_bad(self):
        with pytest.raises(RuntimeError):
            x87double.pack(False, 0, 0)
        with pytest.raises(ValueError):
            IEEEhalf.pack(False, 0, -1)
        with pytest.raises(ValueError):
            IEEEhalf.pack(False, 0, 1 << 10)
        with pytest.raises(ValueError):
            IEEEhalf.pack(False, -1, 0)
        with pytest.raises(ValueError):
            IEEEhalf.pack(False, 32, 0)

    def test_unpack_bad(self):
        with pytest.raises(RuntimeError):
            x87double.unpack(bytes(8))
        with pytest.raises(ValueError):
            IEEEhalf.unpack(bytes(3))

    @pytest.mark.parametrize('line', read_lines('pack.txt'))
    def test_pack_file(self, line):
        parts = line.split()
        if len(parts) != 3:
            assert False, f'bad line: {line}'
        fmt, value, answer = parts
        fmt = format_codes[fmt]
        value = from_string(fmt, value)
        result = value.pack('big')

        # Test big-endian packing
        assert result.hex() == answer
        # Test little-endian packing
        le_packing = value.pack('little')
        assert bytes(reversed(result)) == le_packing
        # Test big-endian unpacking
        assert value.as_tuple() == value.fmt.unpack_value(result, 'big').as_tuple()
        # Test little-endian unpacking
        assert value.as_tuple() == value.fmt.unpack_value(le_packing, 'little').as_tuple()


def binary_operation(line, operation):
    parts = line.split()
    if len(parts) != 8:
        assert False, f'bad line: {line}'
    context, lhs_fmt, lhs, rhs_fmt, rhs, dst_fmt, status, answer = parts
    context = context_string_to_context(context)

    lhs_fmt = format_codes[lhs_fmt]
    input_context = std_context()
    lhs = lhs_fmt.from_string(lhs, input_context)
    assert input_context.flags & ~Flags.SUBNORMAL == 0

    rhs_fmt = format_codes[rhs_fmt]
    rhs = rhs_fmt.from_string(rhs, input_context)
    assert input_context.flags & ~Flags.SUBNORMAL == 0

    dst_fmt = format_codes[dst_fmt]
    status = status_codes[status]
    answer = dst_fmt.from_string(answer, input_context)
    assert input_context.flags & ~Flags.SUBNORMAL == 0

    context.clear_flags()
    operation = getattr(dst_fmt, operation)
    result = operation(lhs, rhs, context)
    assert result.as_tuple() == answer.as_tuple()
    assert context.flags == status


comparison_ops = {
    'eq': 'E',
    'ne': 'LGU',
    'gt': 'G',
    'ng': 'ELU',
    'ge': 'GE',
    'lu': 'LU',
    'lt': 'L',
    'nl': 'GEU',
    'le': 'LE',
    'gu': 'GU',
    'un': 'U',
    'or': 'LGE',
}

class TestBinaryOps:

    @pytest.mark.parametrize('line', read_lines('add.txt'))
    def test_add(self, line):
        binary_operation(line, 'add')

    @pytest.mark.parametrize('line', read_lines('subtract.txt'))
    def test_subtract(self, line):
        binary_operation(line, 'subtract')

    @pytest.mark.parametrize('line', read_lines('multiply.txt'))
    def test_multiply(self, line):
        binary_operation(line, 'multiply')

    @pytest.mark.parametrize('line', read_lines('divide.txt'))
    def test_divide(self, line):
        binary_operation(line, 'divide')

    # @pytest.mark.parametrize('line', read_lines('remainder.txt'))
    # def test_remainder(self, line):
    #     parts = line.split()
    #     if len(parts) != 5:
    #         assert False, f'bad line: {line}'
    #     fmt, lhs, rhs, status, answer = parts

    #     fmt = format_codes[fmt]
    #     lhs = from_string(fmt, lhs)
    #     rhs = from_string(fmt, rhs)
    #     status = status_codes[status]
    #     answer = from_string(fmt, answer)

    #     context = std_context()
    #     result = lhs.remainder(rhs, context)
    #     assert result.as_tuple() == answer.as_tuple()
    #     assert context.flags == status

    @pytest.mark.parametrize('line', read_lines('compare.txt'))
    def test_compare(self, line):
        parts = line.split()
        if len(parts) != 6:
            assert False, f'bad line: {line}'
        lhs_fmt, lhs, rhs_fmt, rhs, status, answer_code = parts

        lhs = from_string(format_codes[lhs_fmt], lhs)
        rhs = from_string(format_codes[rhs_fmt], rhs)
        status = status_codes[status]
        answer = compare_codes[answer_code]

        # Compare quietly
        context = std_context()
        result = lhs.compare(rhs, context, False)
        assert result == answer
        assert context.flags == status

        # Now check all the other comparison operations
        for op, true_set in comparison_ops.items():
            # Test the quiet form:
            op_name = f'compare_quiet_{op}'
            op_result = answer_code in true_set
            op_status = Flags.INVALID if lhs.is_signalling() or rhs.is_signalling() else 0
            context = std_context()
            result = getattr(lhs, op_name)(rhs, context)
            assert result == op_result
            assert context.flags == op_status

            # Test the singalling form:
            if op not in {'un', 'or'}:
                op_name = f'compare_signalling_{op}'
                op_status = Flags.INVALID if lhs.is_NaN() or rhs.is_NaN() else 0
                context = std_context()
                result = getattr(lhs, op_name)(rhs, context)
                assert result == op_result
                assert context.flags == op_status

    @pytest.mark.parametrize('line', read_lines('total_order.txt'))
    def test_total_order(self, line):
        parts = line.split()
        if len(parts) != 4:
            assert False, f'bad line: {line}'
        fmt, lhs, rhs, answer_code = parts

        lhs = from_string(format_codes[fmt], lhs)
        rhs = from_string(format_codes[fmt], rhs)
        answer = boolean_codes[answer_code]

        assert lhs.total_order(rhs) is answer
        lhs_abs = lhs.copy_abs()
        rhs_abs = rhs.copy_abs()

        assert lhs.total_order_mag(rhs) is lhs_abs.total_order(rhs_abs)


class TestFMA:

    @pytest.mark.parametrize('line', read_lines('fma.txt'))
    def test_fma(self, line):
        parts = line.split()
        if len(parts) != 10:
            assert False, f'bad line: {line}'
        context, lhs_fmt, lhs, rhs_fmt, rhs, add_fmt, addend, dst_fmt, status, answer = parts
        context = context_string_to_context(context)
        input_context = std_context()

        lhs_fmt = format_codes[lhs_fmt]
        lhs = lhs_fmt.from_string(lhs, input_context)
        assert input_context.flags & ~Flags.SUBNORMAL == 0

        rhs_fmt = format_codes[rhs_fmt]
        rhs = rhs_fmt.from_string(rhs, input_context)
        assert input_context.flags & ~Flags.SUBNORMAL == 0

        add_fmt = format_codes[add_fmt]
        addend = add_fmt.from_string(addend, input_context)
        assert input_context.flags & ~Flags.SUBNORMAL == 0

        dst_fmt = format_codes[dst_fmt]
        status = status_codes[status]
        answer = dst_fmt.from_string(answer, input_context)
        assert input_context.flags & ~Flags.SUBNORMAL == 0

        result = dst_fmt.fma(lhs, rhs, addend, context)
        assert result.as_tuple() == answer.as_tuple()
        assert context.flags == status
