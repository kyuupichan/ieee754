import os
import re
from itertools import product

import pytest

from ieee754 import *


std_context = Context(ROUND_HALF_EVEN, True, False)
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

def context_string_to_context(context):
    rounding = rounding_codes[context[0]]
    detect_tininess_after = not 'B' in context
    always_detect_underflow = 'U' in context
    return Context(rounding, detect_tininess_after, always_detect_underflow)

def read_significand(significand):
    if significand[:2] in ('0x', '0X'):
        return int(significand, 16)
    return int(significand)

boolean_codes = {
    'Y': True,
    'N': False,
}

format_codes = {
    'H': IEEEhalf,
    'S': IEEEsingle,
    'D': IEEEdouble,
    'Q': IEEEquad,
}

rounding_codes = {
    'E': ROUND_HALF_EVEN,
    'A': ROUND_HALF_UP,
    'P': ROUND_CEILING,
    'N': ROUND_FLOOR,
    'Z': ROUND_DOWN,
}

status_codes = {
    'K': OpStatus.OK,
    'VI': OpStatus.OVERFLOW | OpStatus.INEXACT,
    'U': OpStatus.UNDERFLOW,
    'UI': OpStatus.UNDERFLOW | OpStatus.INEXACT,
    'I': OpStatus.INEXACT,
    'Z': OpStatus.DIV_BY_ZERO,
    'X': OpStatus.INVALID,
    'XI': OpStatus.INVALID | OpStatus.INEXACT,
}

sign_codes = {
    '+': False,
    '-': True,
}


def to_hex_format(hex_format):
    return HexFormat(
        force_sign=boolean_codes[hex_format[0]],
        force_exp_sign=boolean_codes[hex_format[1]],
        nan_payload=hex_format[2],
        precision=int(hex_format[3:]),
    )


# Test basic class functions before reading test files
class TestGeneralNonComputationalOps:

    @pytest.mark.parametrize('fmt, sign',
                             product(all_IEEE_fmts,
                                     (False, True)
                             ))
    def test_make_zero(self, fmt, sign):
        value = fmt.make_zero(sign)
        if sign:
            assert value.classify() == FloatClass.nZero
            assert value.is_negative()
        else:
            assert value.classify() == FloatClass.pZero
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
            assert value.classify() == FloatClass.nInf
            assert value.is_negative()
        else:
            assert value.classify() == FloatClass.pInf
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
        value, status = fmt.make_NaN(sign, quiet, payload, False)
        if payload == 0 and not quiet:
            assert status == OpStatus.INEXACT
            payload = 1
        else:
            assert status == OpStatus.OK
        if quiet:
            assert value.classify() == FloatClass.qNaN
        else:
            assert value.classify() == FloatClass.sNaN
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
        assert value.to_parts()[-1] == payload

    @pytest.mark.parametrize('fmt, sign',
                             product(all_IEEE_fmts,
                                     (False, True),
                             ))
    def test_make_NaN_quiet_payload(self, fmt, sign):
        fmt.make_NaN(sign, True, 0, False)
        fmt.make_NaN(sign, True, fmt.quiet_bit - 1, False)
        with pytest.raises(ValueError):
            fmt.make_NaN(sign, True, -1, False)
        with pytest.raises(TypeError):
            fmt.make_NaN(sign, True, 1.2, False)
        with pytest.raises(TypeError):
            fmt.make_NaN(sign, True, 1.2, False)

    @pytest.mark.parametrize('fmt, sign',
                             product(all_IEEE_fmts,
                                     (False, True),
                             ))
    def test_make_NaN_signalling_payload(self, fmt, sign):
        fmt.make_NaN(sign, False, fmt.quiet_bit - 1, False)
        with pytest.raises(ValueError):
            fmt.make_NaN(sign, False, -1, False)
        with pytest.raises(TypeError):
            fmt.make_NaN(sign, False, 1.2, False)
        with pytest.raises(TypeError):
            fmt.make_NaN(sign, False, 1.2, False)

    @pytest.mark.parametrize('fmt, detect_tininess_after, always_flag_underflow, sign',
                             product(all_IEEE_fmts,
                                     (False, True),
                                     (False, True),
                                     (False, True),
                             ))
    def test_make_real_MSB_set(self, fmt, detect_tininess_after, always_flag_underflow, sign):
        '''Test MSB set with various exponents.'''
        context = Context(ROUND_HALF_EVEN, detect_tininess_after, always_flag_underflow)
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
            value, status = fmt.make_real(sign, exponent, significand, context)
            if exponent < fmt.e_min:
                assert status == (OpStatus.UNDERFLOW if always_flag_underflow else OpStatus.OK)
                assert not value.is_normal()
                assert value.is_subnormal()
                if sign:
                    assert value.classify() == FloatClass.nSubnormal
                else:
                    assert value.classify() == FloatClass.pSubnormal
            else:
                assert status == OpStatus.OK
                assert value.is_normal()
                assert not value.is_subnormal()
                if sign:
                    assert value.classify() == FloatClass.nNormal
                else:
                    assert value.classify() == FloatClass.pNormal
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
        value, status = fmt.make_real(sign, exponent, 0, std_context)
        assert status == OpStatus.OK
        assert value.is_zero()
        assert value.sign is sign
        assert value.fmt is fmt

    @pytest.mark.parametrize('fmt, sign, two_bits, rounding, dtar, afu',
                             product(all_IEEE_fmts,
                                     (False, True),
                                     (1, 2, 3, ),
                                     all_roundings,
                                     (False, True),
                                     (False, True),
                             ))
    def test_make_real_underflow_to_zero(self, fmt, sign, two_bits, rounding, dtar, afu):
        # Test that a value that loses two bits of precision underflows correctly
        context = Context(rounding, dtar, afu)
        value, status = fmt.make_real(sign, fmt.e_min - 2 - (fmt.precision - 1), two_bits, context)
        underflows_to_zero = (rounding in {ROUND_HALF_EVEN, ROUND_HALF_DOWN} and two_bits in (1, 2)
                              or (rounding == ROUND_HALF_UP and two_bits == 1)
                              or (rounding == ROUND_CEILING and sign)
                              or (rounding == ROUND_FLOOR and not sign)
                              or (rounding == ROUND_DOWN))
        if underflows_to_zero:
            assert status == OpStatus.INEXACT | OpStatus.UNDERFLOW
            assert value.is_zero()
        else:
            assert status == OpStatus.INEXACT | OpStatus.UNDERFLOW
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
        context = Context(rounding, True, False)
        # First test the exponent that doesn't overflow but that one more would
        exponent = fmt.e_max
        value, status = fmt.make_real(sign, exponent, 1, context)
        assert status == OpStatus.OK
        assert value.is_normal()
        assert value.sign is sign
        assert value.fmt is fmt

        # Increment the exponent.  Overflow now depends on rounding mode
        exponent += 1
        value, status = fmt.make_real(sign, exponent, 1, context)
        assert status == OpStatus.OVERFLOW | OpStatus.INEXACT
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
        context = Context(rounding, True, False)
        # Minimimum good, maximum good, overflows to infinity
        exponent = [fmt.e_min - 2, fmt.e_max - 3, fmt.e_max - 2][e_selector]
        # two extra bits in the significand
        significand = two_bits + (fmt.max_significand << 2)
        value, status = fmt.make_real(sign, exponent - (fmt.precision - 1), significand, context)
        rounds_away = (two_bits and
                       ((rounding == ROUND_HALF_EVEN and two_bits in (2, 3))
                        or rounding == ROUND_UP
                        or (rounding == ROUND_HALF_DOWN and two_bits == 3)
                        or (rounding == ROUND_HALF_UP and two_bits in (2, 3))
                        or (rounding == ROUND_CEILING and not sign)
                        or (rounding == ROUND_FLOOR and sign)))
        if rounds_away:
            if e_selector == 2:
                assert status == OpStatus.INEXACT | OpStatus.OVERFLOW
                assert value.is_infinite()
            else:
                assert status == OpStatus.INEXACT
                assert value.is_normal()
                assert value.significand == fmt.int_bit
                assert value.e_biased == exponent + fmt.e_bias + 3
        else:
            assert status == (OpStatus.INEXACT if two_bits else OpStatus.OK)
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
        context = Context(rounding, True, False)
        # an extra bit in the significand with two LSBs varying
        significand = two_bits + ((fmt.max_significand >> 1) << 2)
        value, status = fmt.make_real(sign, fmt.e_min - 2 - (fmt.precision - 1), significand,
                                      context)
        rounds_away = (two_bits and
                       ((rounding == ROUND_HALF_EVEN and two_bits in (2, 3))
                        or rounding == ROUND_UP
                        or (rounding == ROUND_HALF_DOWN and two_bits == 3)
                        or (rounding == ROUND_HALF_UP and two_bits in (2, 3))
                        or (rounding == ROUND_CEILING and not sign)
                        or (rounding == ROUND_FLOOR and sign)))
        if rounds_away:
            assert status == OpStatus.INEXACT
            assert value.is_normal()
            assert value.significand == fmt.int_bit
            assert value.e_biased == 1
        else:
            assert status == (OpStatus.INEXACT | OpStatus.UNDERFLOW if two_bits else OpStatus.OK)
            assert value.is_subnormal()
            assert value.significand == fmt.int_bit - 1
            assert value.e_biased == 0
        assert value.sign is sign
        assert value.fmt is fmt


class TestUnaryOps:

    @pytest.mark.parametrize('line', read_lines('from_string.txt'))
    def test_from_string(self, line):
        parts = line.split()
        if len(parts) == 1:
            hex_str, = parts
            with pytest.raises(SyntaxError):
                IEEEsingle.from_string(hex_str, std_context)
        elif len(parts) == 7:
            fmt, context, hex_str, status, sign, exponent, significand = parts
            fmt = format_codes[fmt]
            context = context_string_to_context(context)
            status = status_codes[status]
            sign = sign_codes[sign]
            try:
                exponent = int(exponent)
            except ValueError:
                pass
            significand = read_significand(significand)
            value, stat = fmt.from_string(hex_str, context)
            assert value.to_parts() == (sign, exponent, significand)
            assert stat == status
        else:
            assert False, f'bad line: {line}'

    @pytest.mark.parametrize('line', read_lines('round.txt'))
    def test_round(self, line):
        parts = line.split()
        if len(parts) != 5:
            assert False, f'bad line: {line}'
        fmt, context, value, status, answer = parts
        fmt = format_codes[fmt]
        context = Context(rounding_codes[context], True, False)
        value, stat = fmt.from_string(value, std_context)
        assert stat == OpStatus.OK
        status = status_codes[status]
        answer, stat = fmt.from_string(answer, std_context)
        assert stat == OpStatus.OK

        result, stat = value.round(context)
        assert result.to_parts() == answer.to_parts()
        assert stat == status

    @pytest.mark.parametrize('line', read_lines('convert.txt'))
    def test_convert(self, line):
        parts = line.split()
        if len(parts) != 6:
            assert False, f'bad line: {line}'
        src_fmt, context, src_value, dst_fmt, status, answer = parts
        src_fmt = format_codes[src_fmt]
        context = context_string_to_context(context)
        src_value, src_stat = src_fmt.from_string(src_value, std_context)
        assert src_stat == OpStatus.OK
        dst_fmt = format_codes[dst_fmt]
        status = status_codes[status]
        answer, ans_stat = dst_fmt.from_string(answer, std_context)
        assert ans_stat == OpStatus.OK

        result, stat = dst_fmt.convert(src_value, context)
        assert result.to_parts() == answer.to_parts()
        assert stat == status

    @pytest.mark.parametrize('line', read_lines('to_hex_format.txt'))
    def test_to_hex_format(self, line):
        parts = line.split()
        if len(parts) != 6:
            assert False, f'bad line: {line}'
        hex_format, context, dst_fmt, in_str, status, answer = parts
        hex_format = to_hex_format(hex_format)
        context = context_string_to_context(context)
        dst_fmt = format_codes[dst_fmt]
        in_value, in_stat = dst_fmt.from_string(in_str, std_context)
        assert in_stat == OpStatus.OK
        status = status_codes[status]

        result, stat = in_value.to_hex_format_string(hex_format, context)
        assert result == answer
        assert stat == status


def binary_operation(line, operation):
    parts = line.split()
    if len(parts) != 8:
        assert False, f'bad line: {line}'
    context, lhs_fmt, lhs, rhs_fmt, rhs, dst_fmt, status, answer = parts
    context = context_string_to_context(context)

    lhs_fmt = format_codes[lhs_fmt]
    lhs, lhs_stat = lhs_fmt.from_string(lhs, std_context)
    assert lhs_stat == OpStatus.OK

    rhs_fmt = format_codes[rhs_fmt]
    rhs, rhs_stat = rhs_fmt.from_string(rhs, std_context)
    assert rhs_stat == OpStatus.OK

    dst_fmt = format_codes[dst_fmt]
    status = status_codes[status]
    answer, ans_stat = dst_fmt.from_string(answer, std_context)
    assert ans_stat == OpStatus.OK

    operation = getattr(dst_fmt, operation)
    result, stat = operation(lhs, rhs, context)
    assert result.to_parts() == answer.to_parts()
    assert stat == status


class TestBinaryOps:

    @pytest.mark.parametrize('line', read_lines('multiply.txt'))
    def test_multiply(self, line):
        binary_operation(line, 'multiply')

    @pytest.mark.parametrize('line', read_lines('divide.txt'))
    def test_divide(self, line):
        binary_operation(line, 'divide')
