import pytest

from itertools import product

from ieee754 import *


std_env = FloatEnv(RoundTiesToEven, True, False)


# Test basic class functions before reading test files
class TestGeneralNonComputationalOps:

    @pytest.mark.parametrize('fmt, sign',
                             product((IEEEhalf, IEEEsingle, IEEEdouble, IEEEquad),
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
                             product((IEEEhalf, IEEEsingle, IEEEdouble, IEEEquad),
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
                             product((IEEEhalf, IEEEsingle, IEEEdouble, IEEEquad),
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
                             product((IEEEhalf, IEEEsingle, IEEEdouble, IEEEquad),
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
                             product((IEEEhalf, IEEEsingle, IEEEdouble, IEEEquad),
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
                             product((IEEEhalf, IEEEsingle, IEEEdouble, IEEEquad),
                                     (False, True),
                                     (False, True),
                                     (False, True),
                             ))
    def test_make_real(self, fmt, detect_tininess_after, always_flag_underflow, sign):
        env = FloatEnv(RoundTiesToEven, detect_tininess_after, always_flag_underflow)
        significand = 1 << (fmt.precision - 1)
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
                             product((IEEEhalf, IEEEsingle, IEEEdouble, IEEEquad),
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
