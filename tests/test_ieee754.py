import os
from itertools import product

import pytest

from ieee754 import *


std_env = FloatEnv(RoundTiesToEven, True, False)
all_IEEE_fmts = (IEEEhalf, IEEEsingle, IEEEdouble, IEEEquad)
all_rounding_modes = (RoundTiesToEven, RoundTiesToAway, RoundTowardsZero,
                      RoundTowardsPositive, RoundTowardsNegative)


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

def env_string_to_env(env_str):
    rounding = rounding_codes[env_str[0]]
    pos = 1
    detect_tininess_after = not 'B' in env_str
    always_detect_underflow = 'U' in env_str
    return FloatEnv(rounding, detect_tininess_after, always_detect_underflow)


format_codes = {
    'H': IEEEhalf,
    'S': IEEEsingle,
    'D': IEEEdouble,
    'Q': IEEEquad,
}

rounding_codes = {
    'E': RoundTiesToEven,
    'A': RoundTiesToAway,
    'P': RoundTowardsPositive,
    'N': RoundTowardsNegative,
    'Z': RoundTowardsZero,
}

status_codes = {
    'K': OpStatus.OK,
    'VI': OpStatus.OVERFLOW | OpStatus.INEXACT,
    'U': OpStatus.UNDERFLOW,
    'UI': OpStatus.UNDERFLOW | OpStatus.INEXACT,
    'I': OpStatus.INEXACT,
    'Z': OpStatus.DIV_BY_ZERO,
    'X': OpStatus.INVALID_OP,
}

sign_codes = {
    '+': False,
    '-': True,
}


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
                                     (1, 24),
                             ))
    def test_make_NaN(self, fmt, sign, quiet, payload):
        value = fmt.make_NaN(sign, quiet, payload)
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
        # FIXME: test get_payload

    @pytest.mark.parametrize('fmt, sign',
                             product(all_IEEE_fmts,
                                     (False, True),
                             ))
    def test_make_NaN_quiet_payload(self, fmt, sign):
        fmt.make_NaN(sign, True, 0)
        fmt.make_NaN(sign, True, fmt.quiet_bit - 1)
        with pytest.raises(ValueError):
            fmt.make_NaN(sign, True, -1)
        with pytest.raises(TypeError):
            fmt.make_NaN(sign, True, 1.2)
        with pytest.raises(TypeError):
            fmt.make_NaN(sign, True, 1.2)
        with pytest.raises(ValueError):
            fmt.make_NaN(sign, True, fmt.quiet_bit)

    @pytest.mark.parametrize('fmt, sign',
                             product(all_IEEE_fmts,
                                     (False, True),
                             ))
    def test_make_NaN_signalling_payload(self, fmt, sign):
        fmt.make_NaN(sign, False, fmt.quiet_bit - 1)
        with pytest.raises(ValueError):
            fmt.make_NaN(sign, False, 0)
        with pytest.raises(ValueError):
            fmt.make_NaN(sign, False, -1)
        with pytest.raises(TypeError):
            fmt.make_NaN(sign, False, 1.2)
        with pytest.raises(TypeError):
            fmt.make_NaN(sign, False, 1.2)
        with pytest.raises(ValueError):
            fmt.make_NaN(sign, False, fmt.quiet_bit)

    @pytest.mark.parametrize('fmt, detect_tininess_after, always_flag_underflow, sign',
                             product(all_IEEE_fmts,
                                     (False, True),
                                     (False, True),
                                     (False, True),
                             ))
    def test_make_real_MSB_set(self, fmt, detect_tininess_after, always_flag_underflow, sign):
        '''Test MSB set with various exponents.'''
        env = FloatEnv(RoundTiesToEven, detect_tininess_after, always_flag_underflow)
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
            value, status = fmt.make_real(sign, exponent, significand, env)
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
        value, status = fmt.make_real(sign, exponent, 0, std_env)
        assert status == OpStatus.OK
        assert value.is_zero()
        assert value.sign is sign
        assert value.fmt is fmt

    @pytest.mark.parametrize('fmt, sign, two_bits, rounding_mode, dtar, afu',
                             product(all_IEEE_fmts,
                                     (False, True),
                                     (1, 2, 3, ),
                                     all_rounding_modes,
                                     (False, True),
                                     (False, True),
                             ))
    def test_make_real_underflow_to_zero(self, fmt, sign, two_bits, rounding_mode, dtar, afu):
        # Test that a value that loses two bits of precision underflows correctly
        env = FloatEnv(rounding_mode, dtar, afu)
        value, status = fmt.make_real(sign, fmt.e_min - 2 - (fmt.precision - 1), two_bits, env)
        underflows_to_zero = ((rounding_mode is RoundTiesToEven and two_bits in (1, 2))
                              or (rounding_mode is RoundTiesToAway and two_bits == 1)
                              or (rounding_mode is RoundTowardsPositive and sign)
                              or (rounding_mode is RoundTowardsNegative and not sign)
                              or (rounding_mode is RoundTowardsZero))
        if underflows_to_zero:
            assert status == OpStatus.INEXACT | OpStatus.UNDERFLOW
            assert value.is_zero()
        else:
            assert status == OpStatus.INEXACT | OpStatus.UNDERFLOW
            assert value.is_subnormal()
            assert value.significand == 1
        assert value.sign is sign
        assert value.fmt is fmt

    @pytest.mark.parametrize('fmt, sign, rounding_mode',
                             product(all_IEEE_fmts,
                                     (False, True),
                                     all_rounding_modes,
                             ))
    def test_make_overflow(self, fmt, sign, rounding_mode):
        env = FloatEnv(rounding_mode, True, False)
        # First test the exponent that doesn't overflow but that one more would
        exponent = fmt.e_max
        value, status = fmt.make_real(sign, exponent, 1, env)
        assert status == OpStatus.OK
        assert value.is_normal()
        assert value.sign is sign
        assert value.fmt is fmt

        # Increment the exponent.  Overflow now depends on rounding mode
        exponent += 1
        value, status = fmt.make_real(sign, exponent, 1, env)
        if (rounding_mode is RoundTiesToEven or rounding_mode is RoundTiesToAway
                or (rounding_mode is RoundTowardsPositive and not sign)
                or (rounding_mode is RoundTowardsNegative and sign)):
            assert value.is_infinite()
            assert status == OpStatus.OVERFLOW | OpStatus.INEXACT
        else:
            assert value.is_normal()
            assert status == OpStatus.INEXACT
        assert value.sign is sign
        assert value.fmt is fmt

    @pytest.mark.parametrize('fmt, sign, e_selector, two_bits, rounding_mode',
                             product(all_IEEE_fmts,
                                     (False, True),
                                     range(0, 3),
                                     (0, 1, 2, 3, ),
                                     all_rounding_modes,
                             ))
    def test_make_real_rounding_overflows_significand(self, fmt, sign, e_selector, two_bits,
                                                      rounding_mode):
        # Test cases where rounding away causes significand to overflow
        env = FloatEnv(rounding_mode, True, False)
        # Minimimum good, maximum good, overflows to infinity
        exponent = [fmt.e_min - 2, fmt.e_max - 3, fmt.e_max - 2][e_selector]
        # two extra bits in the significand
        significand = two_bits + (fmt.max_significand << 2)
        value, status = fmt.make_real(sign, exponent - (fmt.precision - 1), significand, env)
        rounds_away = (two_bits and
                       ((rounding_mode is RoundTiesToEven and two_bits in (2, 3))
                        or (rounding_mode is RoundTiesToAway and two_bits in (2, 3))
                        or (rounding_mode is RoundTowardsPositive and not sign)
                        or (rounding_mode is RoundTowardsNegative and sign)))
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

    @pytest.mark.parametrize('fmt, sign, two_bits, rounding_mode',
                             product(all_IEEE_fmts,
                                     (False, True),
                                     (0, 1, 2, 3, ),
                                     all_rounding_modes,
                             ))
    def test_make_real_rounding_subnormal_to_normal(self, fmt, sign, two_bits, rounding_mode):
        # Test cases where rounding away causes a subnormal to normalize
        env = FloatEnv(rounding_mode, True, False)
        # an extra bit in the significand with two LSBs varying
        significand = two_bits + ((fmt.max_significand >> 1) << 2)
        value, status = fmt.make_real(sign, fmt.e_min - 2 - (fmt.precision - 1), significand, env)
        rounds_away = (two_bits and
                       ((rounding_mode is RoundTiesToEven and two_bits in (2, 3))
                        or (rounding_mode is RoundTiesToAway and two_bits in (2, 3))
                        or (rounding_mode is RoundTowardsPositive and not sign)
                        or (rounding_mode is RoundTowardsNegative and sign)))
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

    @pytest.mark.parametrize('line', read_lines('from_hex_significand_string.txt'))
    def test_from_hex_significand_string(self, line):
        parts = line.split()
        if len(parts) == 1:
            hex_str, = parts
            with pytest.raises(SyntaxError):
                IEEEsingle.from_string(hex_str, std_env)
        elif len(parts) == 7:
            fmt, env_str, hex_str, status, sign, exponent, significand = parts
            fmt = format_codes[fmt]
            env = env_string_to_env(env_str)
            status = status_codes[status]
            sign = sign_codes[sign]
            exponent = int(exponent)
            significand = int(significand)
            value, stat = fmt.from_string(hex_str, env)
            assert value.to_parts() == (sign, exponent, significand)
            assert stat == status
        else:
            assert False, f'bad line: {line}'

    @pytest.mark.parametrize('line', read_lines('round.txt'))
    def test_round(self, line):
        parts = line.split()
        if len(parts) == 5:
            fmt, env_str, value, status, answer = parts
            fmt = format_codes[fmt]
            rounding_mode = rounding_codes[env_str]
            value, stat = fmt.from_string(value, std_env)
            assert stat == OpStatus.OK
            status = status_codes[status]
            answer, stat = fmt.from_string(answer, std_env)
            assert stat == OpStatus.OK

            result, stat = value.round(rounding_mode)
            assert result.to_parts() == answer.to_parts()
            assert stat == status
        else:
            assert False, f'bad line: {line}'
