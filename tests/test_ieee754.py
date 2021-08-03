import os
import random
import re
import threading
from decimal import Decimal
from math import isfinite
from fractions import Fraction
from functools import partial
from itertools import product
from struct import pack

import pytest

from ieee754 import *


HEX_SIGNIFICAND_PREFIX = re.compile('[-+]?0x', re.ASCII | re.IGNORECASE)
all_IEEE_fmts = (IEEEhalf, IEEEsingle, IEEEdouble, IEEEquad)
all_roundings = (ROUND_CEILING, ROUND_FLOOR, ROUND_DOWN, ROUND_UP,
                 ROUND_HALF_EVEN, ROUND_HALF_UP, ROUND_HALF_DOWN)
native = 'little' if pack('d', -0.0)[-1] == 0x80 else 'big'


boolean_codes = {
    'Y': True,
    'N': False,
}

tininess_after_codes = {
    'A': True,
    'B': False
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
    'xd': x87double,
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
    'S': 0,
    'SI': Flags.INEXACT,
    'U': Flags.UNDERFLOW | Flags.INEXACT,
    'I': Flags.INEXACT,
    'Z': Flags.DIV_BY_ZERO,
    'X': Flags.INVALID,
}

snan_codes = {
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


def rounding_string_to_context(rounding):
    tininess_after = True
    if len(rounding) == 2:
        tininess_after = tininess_after_codes[rounding[1]]
        rounding = rounding[0]
    return Context(rounding=rounding_codes[rounding], tininess_after=tininess_after)


def read_significand(significand):
    if significand[:2] in ('0x', '0X'):
        return int(significand, 16)
    return int(significand)


def from_string(fmt, string):
    context = Context()
    result = fmt.from_string(string, context)
    if HEX_SIGNIFICAND_PREFIX.match(string):
        assert context.flags == 0
    else:
        assert context.flags & ~(Flags.UNDERFLOW | Flags.INEXACT) == 0
    return result


def floats_equal(lhs, rhs):
    return lhs.fmt == rhs.fmt and lhs.as_tuple() == rhs.as_tuple()


def values_equal(result, answer):
    if isinstance(answer, Binary):
        return floats_equal(result, answer)
    else:
        return result == answer


def to_text_format(hex_format):
    return TextFormat(
        force_leading_sign=boolean_codes[hex_format[0]],
        force_exp_sign=boolean_codes[hex_format[1]],
        rstrip_zeroes=boolean_codes[hex_format[2]],
        snan = snan_codes[hex_format[3]],
        nan_payload=nan_payload_codes[hex_format[4]],
    )

def substitute_plus_zero(exception, context):
    result = exception.default_result
    if isinstance(result, str):
        return '+0.0'
    elif isinstance(result, int):
        return 0
    else:
        return result.fmt.make_zero(False)


def substitute_plus_one(exception, context):
    result = exception.default_result
    if isinstance(result, str):
        return '+1.0'
    elif isinstance(result, int):
        return 1
    else:
        return result.fmt.make_one(False)


# Test functions with an explicit context and a None context
@pytest.fixture
def context():
    with local_context(DefaultContext) as context:
        yield context


# Test functions with an explicit context and a None context
@pytest.fixture
def quiet_context():
    with local_context(Context()) as context:
        yield context


class TestTextFormat:

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
        assert text_format.format_decimal(sign, exponent, digits) == answer

    @pytest.mark.parametrize('line', read_lines('format_hex.txt'))
    def test_format_hex(self, line):
        parts = line.split()
        if len(parts) != 8:
            assert False, f'bad line: {line}'
        (exp_digits, force_exp_sign, force_leading_sign, force_point, upper_case,
         rstrip_zeroes, value, answer) = parts
        text_format = TextFormat(exp_digits=int(exp_digits),
                                 force_exp_sign=boolean_codes[force_exp_sign],
                                 force_leading_sign=boolean_codes[force_leading_sign],
                                 force_point=boolean_codes[force_point],
                                 upper_case=boolean_codes[upper_case],
                                 rstrip_zeroes=boolean_codes[rstrip_zeroes])
        value = from_string(IEEEdouble, value)
        assert text_format.format_hex(value) == answer

    @pytest.mark.parametrize('value', (-1.0, -0.0, 0.0, 1.0, 123.456e12, 123.456e-12,
                                       float('inf'), float('-inf'), float('nan')))
    def test_hex_matches_python(self, value):
        if isfinite(value):
            assert value.hex() == IEEEdouble.from_float(value).to_string()
        else:
            assert str(Decimal(value)) == IEEEdouble.from_float(value).to_string()

    @pytest.mark.parametrize('value', (-1.0, -0.0, 0.0, 1.0, 123.456e12, 123.456e-12,
                                       1.256e3, 1.256e2,
                                       float('inf'), float('-inf'), float('nan')))
    def test_dec_matches_python(self, value):
        assert str(value) == IEEEdouble.from_float(value).to_decimal_string()
        assert f'{value:.3g}' == IEEEdouble.from_float(value).to_decimal_string(
            text_format=Dec_g_Format, precision=3)
        #assert f'{value:.2f}' == IEEEdouble.from_float(value).to_decimal_string(
        #    text_format=Dec_f_Format, precision=3)

    def test_snan(self):
        assert 'snan' == IEEEdouble.from_string('sNaN').to_decimal_string()


class TestContext:

    def test_default_context(self):
        assert DefaultContext.rounding == ROUND_HALF_EVEN
        assert DefaultContext.flags == 0
        assert DefaultContext.tininess_after is True

    def contexts_equal(self, lhs, rhs):
        return lhs.flags == rhs.flags and lhs.rounding == rhs.rounding

    def test_copy(self, context):
        context.exceptions = [1]
        context.flags = Flags.INEXACT
        c = context.copy()
        assert c.flags == context.flags
        assert c.rounding == context.rounding
        assert c.tininess_after == context.tininess_after
        assert c.handlers is not context.handlers
        assert c.handlers == context.handlers
        assert c.exceptions is not context.exceptions
        assert c.exceptions == context.exceptions

    def test_get_context(self):
        context = get_context()
        assert context is not DefaultContext
        assert get_context() is context

        def target():
            thread_context = get_context()
            assert self.contexts_equal(thread_context, DefaultContext)
            assert thread_context not in (context, DefaultContext)
            event.set()

        event = threading.Event()
        thread = threading.Thread(target=target)
        thread.start()
        event.wait()
        event.clear()

        # Now change DefaultContext
        thread = threading.Thread(target=target)
        DefaultContext.rounding = ROUND_DOWN
        try:
            thread.start()
            event.wait()
        finally:
            # Restore DefaultContext
            DefaultContext.rounding = ROUND_HALF_EVEN

    def test_set_context(self):
        context1 = get_context()
        context = Context(rounding=ROUND_CEILING, flags=Flags.INVALID)
        assert not self.contexts_equal(context, context1)

        set_context(context)
        context2 = get_context()

        assert context2 is context
        assert context2.rounding == ROUND_CEILING and context2.flags == Flags.INVALID

    def test_local_context_omitted(self):
        context = get_context()
        context_copy = context.copy()
        try:
            context.flags ^= Flags.INVALID

            with local_context() as ctx:
                assert get_context() is ctx
                assert ctx is not context
                assert self.contexts_equal(ctx, context)
                assert not self.contexts_equal(ctx, context_copy)

            assert get_context() is context
        finally:
            set_context(context_copy)

    def test_local_context(self):
        context = get_context()
        new_context = Context(rounding=ROUND_DOWN, flags=Flags.DIV_BY_ZERO)
        with local_context(new_context) as ctx:
            assert get_context() is ctx
            assert ctx not in (context, new_context)
            assert self.contexts_equal(ctx, new_context)
            assert self.contexts_equal(new_context, Context(rounding=ROUND_DOWN,
                                                            flags=Flags.DIV_BY_ZERO))

        assert get_context() is context

    def test_local_context_timing(self, context):
        # Want to check that the saved context is taken not on construction but on entry
        my_context = Context(rounding=ROUND_DOWN, flags=Flags.INVALID)
        manager = local_context(my_context)
        my_context.flags |= Flags.DIV_BY_ZERO
        set_context(my_context)
        with manager as ctx:
            assert get_context() is ctx
            assert ctx is not my_context   # must be a copy
            assert self.contexts_equal(ctx, my_context)
            assert not self.contexts_equal(ctx, context)
        assert get_context() is my_context

    def test_repr(self):
        c = Context(rounding=ROUND_UP, flags=Flags.INEXACT, tininess_after=True)
        assert repr(c) == (
            '<Context rounding=ROUND_UP flags=<Flags.INEXACT: 16> tininess_after=True>'
        )

    def test_handler_bad(self):
        context = Context()
        with pytest.raises(TypeError):
            context.handler(SyntaxError)
        with pytest.raises(TypeError):
            context.handler(ZeroDivisionError)

    def test_set_handler_single(self):
        context = Context()
        context.set_handler(DivisionByZero, HandlerKind.NO_FLAG)
        assert context.handler(DivisionByZero) == (HandlerKind.NO_FLAG, None)
        assert context.handler(DivideByZero) == (HandlerKind.NO_FLAG, None)
        assert context.handler(LogBZero) == (HandlerKind.NO_FLAG, None)
        assert context.handler(IEEEError) == (HandlerKind.DEFAULT, None)

    def test_set_handler_list(self):
        context = Context()
        context.set_handler([DivideByZero, Inexact], HandlerKind.NO_FLAG)
        assert context.handler(DivideByZero) == (HandlerKind.NO_FLAG, None)
        assert context.handler(Inexact) == (HandlerKind.NO_FLAG, None)
        assert context.handler(DivisionByZero) == (HandlerKind.DEFAULT, None)
        assert context.handler(LogBZero) == (HandlerKind.DEFAULT, None)
        assert context.handler(IEEEError) == (HandlerKind.DEFAULT, None)

    def test_set_handler_tuple(self):
        context = Context()
        context.set_handler((DivideByZero, Inexact), HandlerKind.NO_FLAG)
        context.set_handler(DivisionByZero, HandlerKind.MAYBE_FLAG)
        assert context.handler(DivideByZero) == (HandlerKind.NO_FLAG, None)
        assert context.handler(Inexact) == (HandlerKind.NO_FLAG, None)
        assert context.handler(DivisionByZero) == (HandlerKind.MAYBE_FLAG, None)
        assert context.handler(LogBZero) == (HandlerKind.MAYBE_FLAG, None)
        assert context.handler(IEEEError) == (HandlerKind.DEFAULT, None)

    @pytest.mark.parametrize('exc', (ZeroDivisionError, (LogBZero, NotImplementedError)))
    def test_set_handler_bad(self, exc):
        context = Context()
        with pytest.raises(TypeError) as e:
            context.set_handler(exc, HandlerKind.DEFAULT)
        assert 'of IEEEError' in str(e.value)

    @pytest.mark.parametrize('kind', (2, None))
    def test_set_handler_bad_kind(self, kind):
        context = Context()
        with pytest.raises(TypeError) as e:
            context.set_handler(DivideByZero, 2)
        assert 'HandlerKind instance' in str(e.value)

    @pytest.mark.parametrize('kind', (
        HandlerKind.NO_FLAG, HandlerKind.DEFAULT, HandlerKind.MAYBE_FLAG,
        HandlerKind.RECORD_EXCEPTION, HandlerKind.ABRUPT_UNDERFLOW, HandlerKind.RAISE))
    def test_set_handler_unwanted_handler(self, kind):
        context = Context()
        with pytest.raises(ValueError) as e:
            context.set_handler(Underflow, kind, from_string)
        assert 'handler given' in str(e.value)

    @pytest.mark.parametrize('kind', (HandlerKind.SUBSTITUTE_VALUE,
                                      HandlerKind.SUBSTITUTE_VALUE_XOR))
    def test_set_handler_missing_handler(self, kind):
        context = Context()
        with pytest.raises(ValueError) as e:
            context.set_handler(DivideByZero, kind)
        assert 'handler not given' in str(e.value)

    @pytest.mark.parametrize('exc', (Underflow, UnderflowExact, UnderflowInexact))
    def test_set_handler_abrupt_underflow(self, exc):
        context = Context()
        context.set_handler(exc, HandlerKind.ABRUPT_UNDERFLOW)

    @pytest.mark.parametrize('exc', (Overflow, Inexact, DivisionByZero, Invalid))
    def test_set_handler_abrupt_underflow_bad(self, exc):
        context = Context()
        with pytest.raises(TypeError) as e:
            context.set_handler(exc, HandlerKind.ABRUPT_UNDERFLOW)
        assert 'of Underflow' in str(e.value)


signs = [True, False]
zeroes = [fmt.make_zero(sign) for sign in signs for fmt in all_IEEE_fmts]


def divide_by_zero_testcase(dst_fmt, handler_class, zero):
    lhs = random.choice(all_IEEE_fmts).make_one(random.choice(signs))
    op_tuple = (OP_DIVIDE, lhs, zero)
    result = dst_fmt.make_infinity(lhs.sign ^ zero.sign)
    return result, DivideByZero, handler_class, op_tuple, partial(dst_fmt.divide, lhs, zero)


def logb_zero_testcase(dst_fmt, handler_class, sign):
    zero = dst_fmt.make_zero(sign)
    op_tuple = (OP_LOGB, zero)
    result = dst_fmt.make_infinity(True)
    return result, LogBZero, handler_class, op_tuple, zero.logb


div_by_zero_testcases = tuple(divide_by_zero_testcase(fmt, handler_class, zero)
                              for fmt in all_IEEE_fmts
                              for handler_class in (DivideByZero, DivisionByZero, IEEEError)
                              for zero in zeroes)
div_by_zero_testcases += tuple(logb_zero_testcase(fmt, handler_class, sign)
                               for fmt in all_IEEE_fmts
                               for handler_class in (LogBZero, DivisionByZero, IEEEError)
                               for sign in signs)


class TestDivisionByZero:

    def test_general(self):
        assert issubclass(DivisionByZero, IEEEError)
        assert issubclass(DivisionByZero, ZeroDivisionError)

    @pytest.mark.parametrize('testcase, kind', product(
        div_by_zero_testcases,
        (HandlerKind.DEFAULT, HandlerKind.NO_FLAG, HandlerKind.MAYBE_FLAG,
         HandlerKind.RECORD_EXCEPTION),
    ))
    def test_basic_kinds(self, testcase, kind, quiet_context):
        answer, exc_class, handler_class, op_tuple, div_by_zero_func = testcase

        context = quiet_context
        context.set_handler(handler_class, kind)

        result = div_by_zero_func(context)

        assert floats_equal(result, answer)
        assert context.flags == 0 if kind == HandlerKind.NO_FLAG else Flags.DIV_BY_ZERO
        if kind == HandlerKind.RECORD_EXCEPTION:
            assert len(context.exceptions) == 1
            exception = context.exceptions[0]
            assert isinstance(exception, exc_class)
            assert exception.op_tuple == op_tuple
        else:
            assert not context.exceptions

    @pytest.mark.parametrize('testcase', div_by_zero_testcases)
    def test_substitute_value(self, testcase, quiet_context):
        answer, exc_class, handler_class, op_tuple, div_by_zero_func = testcase

        context = quiet_context
        context.set_handler(handler_class, HandlerKind.SUBSTITUTE_VALUE, substitute_plus_zero)

        result = div_by_zero_func(context)

        assert floats_equal(result, answer.fmt.make_zero(False))
        assert context.flags == Flags.DIV_BY_ZERO
        assert not context.exceptions

    @pytest.mark.parametrize('testcase', div_by_zero_testcases)
    def test_substitute_value_xor(self, testcase, quiet_context):
        answer, exc_class, handler_class, op_tuple, div_by_zero_func = testcase

        context = quiet_context
        context.set_handler(handler_class, HandlerKind.SUBSTITUTE_VALUE_XOR, substitute_plus_zero)

        result = div_by_zero_func(context)

        if op_tuple[0] == OP_DIVIDE:
            assert floats_equal(result, answer.fmt.make_zero(answer.sign))
        else:
            assert floats_equal(result, answer)
        assert context.flags == Flags.DIV_BY_ZERO
        assert not context.exceptions

    @pytest.mark.parametrize('exc_class', (DivideByZero, DivisionByZero, IEEEError))
    def test_abrupt_underflow(self, exc_class, quiet_context):
        context = quiet_context
        with pytest.raises(TypeError) as e:
            context.set_handler(exc_class, HandlerKind.ABRUPT_UNDERFLOW)

        assert 'must be subclasses of Underflow' in str(e.value)

    @pytest.mark.parametrize('testcase', div_by_zero_testcases)
    def test_raise(self, testcase, quiet_context):
        answer, exc_class, handler_class, op_tuple, div_by_zero_func = testcase

        context = quiet_context
        context.set_handler(handler_class, HandlerKind.RAISE)

        with pytest.raises(exc_class) as e:
            div_by_zero_func(context)

        e = e.value
        assert e.op_tuple == op_tuple
        assert floats_equal(e.default_result, answer)
        assert context.flags == Flags.DIV_BY_ZERO
        assert not context.exceptions


payloads = [1, 20, 300]
no_snan_text_format = TextFormat(snan='')


def random_snan(fmt):
    return fmt.make_nan(random.choice(signs), True, random.choice(payloads))


def invalid_to_decimal_string(dst_fmt, index):
    value = random_snan(dst_fmt)
    sign = '-' if value.sign else ''
    result = f'{sign}NaN0x{value.nan_payload():x}'
    op_tuple = (OP_TO_DECIMAL_STRING, value, -1)
    handler_class = (InvalidToString, Invalid, IEEEError)[index]
    return (result, InvalidToString, handler_class, op_tuple,
            partial(value.to_decimal_string, -1, no_snan_text_format))


def invalid_to_string(dst_fmt, index):
    value = random_snan(dst_fmt)
    sign = '-' if value.sign else ''
    result = f'{sign}NaN0x{value.nan_payload():x}'
    op_tuple = (OP_TO_STRING, value)
    handler_class = (InvalidToString, Invalid, IEEEError)[index]
    return (result, InvalidToString, handler_class, op_tuple,
            partial(value.to_string, no_snan_text_format))


def invalid_convert(dst_fmt, index):
    value = random_snan(random.choice(all_IEEE_fmts))
    result = dst_fmt.make_nan(value.sign, False, value.nan_payload())
    op_tuple = (OP_CONVERT, value)
    handler_class = (SignallingNaNOperand, Invalid, IEEEError)[index]
    return result, SignallingNaNOperand, handler_class, op_tuple, partial(dst_fmt.convert, value)


def invalid_add(dst_fmt, index):
    lhs = random.choice(all_IEEE_fmts).make_infinity(random.choice(signs))
    rhs = random.choice(all_IEEE_fmts).make_infinity(not lhs.sign)
    result = dst_fmt.make_nan(False, False, 0)
    op_tuple = (OP_ADD, lhs, rhs)
    handler_class = (InvalidAdd, Invalid, IEEEError)[index]
    return result, InvalidAdd, handler_class, op_tuple, partial(dst_fmt.add, lhs, rhs)


def invalid_subtract(dst_fmt, index):
    lhs = random.choice(all_IEEE_fmts).make_infinity(random.choice(signs))
    rhs = random.choice(all_IEEE_fmts).make_infinity(lhs.sign)
    result = dst_fmt.make_nan(False, False, 0)
    op_tuple = (OP_SUBTRACT, lhs, rhs)
    handler_class = (InvalidAdd, Invalid, IEEEError)[index]
    return result, InvalidAdd, handler_class, op_tuple, partial(dst_fmt.subtract, lhs, rhs)


def invalid_multiply(dst_fmt, index):
    zero = random.choice(all_IEEE_fmts).make_zero(random.choice(signs))
    inf = random.choice(all_IEEE_fmts).make_infinity(random.choice(signs))
    lhs, rhs = random.choice(((zero, inf), (inf, zero)))
    result = dst_fmt.make_nan(False, False, 0)
    op_tuple = (OP_MULTIPLY, lhs, rhs)
    handler_class = (InvalidMultiply, Invalid, IEEEError)[index]
    return result, InvalidMultiply, handler_class, op_tuple, partial(dst_fmt.multiply, lhs, rhs)


def invalid_divide_zero(dst_fmt, index):
    lhs = random.choice(all_IEEE_fmts).make_zero(random.choice(signs))
    rhs = random.choice(all_IEEE_fmts).make_zero(random.choice(signs))
    result = dst_fmt.make_nan(False, False, 0)
    op_tuple = (OP_DIVIDE, lhs, rhs)
    handler_class = (InvalidDivide, Invalid, IEEEError)[index]
    return result, InvalidDivide, handler_class, op_tuple, partial(dst_fmt.divide, lhs, rhs)


def invalid_divide_inf(dst_fmt, index):
    lhs = random.choice(all_IEEE_fmts).make_infinity(random.choice(signs))
    rhs = random.choice(all_IEEE_fmts).make_infinity(random.choice(signs))
    result = dst_fmt.make_nan(False, False, 0)
    op_tuple = (OP_DIVIDE, lhs, rhs)
    handler_class = (InvalidDivide, Invalid, IEEEError)[index]
    return result, InvalidDivide, handler_class, op_tuple, partial(dst_fmt.divide, lhs, rhs)


def invalid_sqrt(dst_fmt, index):
    lhs = getattr(dst_fmt, random.choice(('make_one', 'make_infinity')))(True)
    result = dst_fmt.make_nan(False, False, 0)
    op_tuple = (OP_SQRT, lhs)
    handler_class = (InvalidSqrt, Invalid, IEEEError)[index]
    return result, InvalidSqrt, handler_class, op_tuple, partial(dst_fmt.sqrt, lhs)

def invalid_fma(dst_fmt, index):
    zero = random.choice(all_IEEE_fmts).make_zero(random.choice(signs))
    inf = random.choice(all_IEEE_fmts).make_infinity(random.choice(signs))
    lhs, rhs = random.choice(((zero, inf), (inf, zero)))
    addend = random.choice(all_IEEE_fmts).make_one(random.choice(signs))
    result = dst_fmt.make_nan(False, False, 0)
    op_tuple = (OP_FMA, lhs, rhs, addend)
    handler_class = (InvalidFMA, Invalid, IEEEError)[index]
    return result, InvalidFMA, handler_class, op_tuple, partial(dst_fmt.fma, lhs, rhs, addend)


def invalid_remainder(dst_fmt, index):
    if random.choice((0,1)):
        lhs = dst_fmt.make_infinity(random.choice(signs))
        rhs = dst_fmt.make_one(random.choice(signs))
    else:
        lhs = dst_fmt.make_one(random.choice(signs))
        rhs = dst_fmt.make_zero(random.choice(signs))
    op, op_name = random.choice(((lhs.remainder, OP_REMAINDER), (lhs.fmod, OP_FMOD)))
    result = dst_fmt.make_nan(False, False, 0)
    op_tuple = (op_name, lhs, rhs)
    handler_class = (InvalidRemainder, Invalid, IEEEError)[index]
    return result, InvalidRemainder, handler_class, op_tuple, partial(op, rhs)


def invalid_logb_integral(dst_fmt, index):
    kind = random.choice(range(3))
    if kind == 0:
        value = dst_fmt.make_nan(False, False, 0)
        result = dst_fmt.logb_nan
    elif kind == 1:
        value = dst_fmt.make_zero(random.choice(signs))
        result = dst_fmt.logb_zero
    else:
        value = dst_fmt.make_infinity(random.choice(signs))
        result = dst_fmt.logb_inf
    op_tuple = (OP_LOGB_INTEGRAL, value)
    handler_class = (InvalidLogBIntegral, Invalid, IEEEError)[index]
    return result, InvalidLogBIntegral, handler_class, op_tuple, value.logb_integral


def invalid_comparison(dst_fmt, index):
    lhs = dst_fmt.make_zero(random.choice(signs))
    rhs = dst_fmt.make_nan(False, False, 0)
    lhs, rhs = random.choice(((lhs, rhs), (rhs, lhs)))
    result = Compare.UNORDERED
    op_tuple = (OP_COMPARE, lhs, rhs)
    handler_class = (InvalidComparison, Invalid, IEEEError)[index]
    return result, InvalidComparison, handler_class, op_tuple, partial(lhs.compare_signal, rhs)


def invalid_convert_to_integer(kind, index):
    dst_fmt = random.choice(all_IEEE_fmts)
    if kind == 0:
        value = dst_fmt.make_nan(False, False, 0)
        result = 0
    elif kind == 1:
        value = dst_fmt.make_one(True)
        result = 0
    elif kind == 2:
        value = dst_fmt.make_infinity(False)
        result = 15
    else:
        value = dst_fmt.from_value(16)
        result = 15
    op_tuple = (OP_CONVERT_TO_INTEGER, value, 0, 15, ROUND_DOWN)
    handler_class = (InvalidConvertToInteger, Invalid, IEEEError)[index]
    return (result, InvalidConvertToInteger, handler_class, op_tuple,
            partial(value.convert_to_integer, 0, 15, ROUND_DOWN))


invalid_testcase_funcs = (invalid_to_decimal_string, invalid_to_string, invalid_convert,
                          invalid_add, invalid_subtract, invalid_multiply, invalid_divide_zero,
                          invalid_divide_inf, invalid_sqrt, invalid_fma, invalid_remainder,
                          invalid_logb_integral, invalid_comparison)

invalid_testcases = tuple(testcase_func(fmt, index)
                          for testcase_func in invalid_testcase_funcs
                          for index in range(3)
                          for fmt in all_IEEE_fmts)
invalid_testcases += tuple(invalid_convert_to_integer(kind, index)
                           for kind in range(4)
                           for index in range(3))


class TestInvalid:

    @pytest.mark.parametrize('testcase, kind', product(
        invalid_testcases,
        (HandlerKind.DEFAULT, HandlerKind.NO_FLAG, HandlerKind.MAYBE_FLAG,
         HandlerKind.RECORD_EXCEPTION),
    ))
    def test_basic_kinds(self, testcase, kind, quiet_context):
        answer, exc_class, handler_class, op_tuple, invalid_func = testcase

        context = quiet_context
        context.set_handler(handler_class, kind)

        result = invalid_func(context)
        assert values_equal(result, answer)
        assert context.flags == 0 if kind == HandlerKind.NO_FLAG else Flags.INVALID
        if kind == HandlerKind.RECORD_EXCEPTION:
            assert len(context.exceptions) == 1
            exception = context.exceptions[0]
            assert isinstance(exception, exc_class)
            assert exception.op_tuple == op_tuple
        else:
            assert not context.exceptions

    @pytest.mark.parametrize('testcase', invalid_testcases)
    def test_substitute_value(self, testcase, quiet_context):
        answer, exc_class, handler_class, op_tuple, invalid_func = testcase

        context = quiet_context
        context.set_handler(handler_class, HandlerKind.SUBSTITUTE_VALUE, substitute_plus_one)

        result = invalid_func(context)
        if isinstance(answer, str):
            assert result == '+1.0'
        elif isinstance(result, int):
            assert result == 1
        else:
            assert floats_equal(result, answer.fmt.make_one(False))
        assert context.flags == Flags.INVALID
        assert not context.exceptions

    @pytest.mark.parametrize('testcase', invalid_testcases)
    def test_substitute_value_xor(self, testcase, quiet_context):
        answer, exc_class, handler_class, op_tuple, invalid_func = testcase

        context = quiet_context
        context.set_handler(handler_class, HandlerKind.SUBSTITUTE_VALUE_XOR, substitute_plus_one)

        result = invalid_func(context)
        # Ignored unless one of these
        if op_tuple[0] in {OP_MULTIPLY, OP_DIVIDE}:
            sign = op_tuple[1].sign ^ op_tuple[2].sign
            assert floats_equal(result, answer.fmt.make_one(sign))
        else:
            assert values_equal(result, answer)
        assert context.flags == Flags.INVALID
        assert not context.exceptions

    @pytest.mark.parametrize('testcase', invalid_testcases)
    def test_abrupt_underflow(self, testcase, quiet_context):
        handler_class = testcase[2]
        context = quiet_context
        with pytest.raises(TypeError) as e:
            context.set_handler(handler_class, HandlerKind.ABRUPT_UNDERFLOW)

        assert 'must be subclasses of Underflow' in str(e.value)

    @pytest.mark.parametrize('testcase', invalid_testcases)
    def test_raise(self, testcase, quiet_context):
        answer, exc_class, handler_class, op_tuple, invalid_func = testcase

        context = quiet_context
        context.set_handler(handler_class, HandlerKind.RAISE)

        with pytest.raises(exc_class) as e:
            invalid_func(context)

        e = e.value
        assert e.op_tuple == op_tuple
        assert values_equal(e.default_result, answer)
        assert context.flags == Flags.INVALID
        assert not context.exceptions


def inexact_divide(dst_fmt, index):
    lhs = random.choice(all_IEEE_fmts).make_one(random.choice(signs))
    rhs = random.choice(all_IEEE_fmts).from_int(3)
    op_tuple = (OP_DIVIDE, lhs, rhs)
    handler_class = (Inexact, IEEEError)[index]
    return Inexact, handler_class, op_tuple, partial(dst_fmt.divide, lhs, rhs)


def inexact_from_string(dst_fmt, index):
    lhs = '0.2'
    op_tuple = (OP_FROM_STRING, lhs)
    handler_class = (Inexact, IEEEError)[index]
    return Inexact, handler_class, op_tuple, partial(dst_fmt.from_string, lhs)


def inexact_sqrt(dst_fmt, index):
    lhs = random.choice(all_IEEE_fmts).from_int(2)
    op_tuple = (OP_SQRT, lhs)
    handler_class = (Inexact, IEEEError)[index]
    return Inexact, handler_class, op_tuple, partial(dst_fmt.sqrt, lhs)


def inexact_round_to_integral(dst_fmt, index):
    lhs = dst_fmt.from_string('0.5')
    op_tuple = (OP_ROUND_TO_INTEGRAL_EXACT, lhs)
    handler_class = (Inexact, IEEEError)[index]
    return Inexact, handler_class, op_tuple, lhs.round_to_integral_exact


def inexact_convert_to_integer(dst_fmt, index):
    lhs = dst_fmt.from_string('0.5')
    op_tuple = (OP_CONVERT_TO_INTEGER_EXACT, lhs, 0, 15, ROUND_DOWN)
    handler_class = (Inexact, IEEEError)[index]
    return Inexact, handler_class, op_tuple, partial(lhs.convert_to_integer_exact,
                                                     0, 15, ROUND_DOWN)


def inexact_to_decimal_string(dst_fmt, index):
    lhs = dst_fmt.from_string('0x1p-14')
    op_tuple = (OP_TO_DECIMAL_STRING, lhs, 2)
    handler_class = (Inexact, IEEEError)[index]
    return Inexact, handler_class, op_tuple, partial(lhs.to_decimal_string, 2, None)



inexact_testcase_funcs = (inexact_divide, inexact_from_string, inexact_sqrt,
                          inexact_round_to_integral, inexact_convert_to_integer,
                          inexact_to_decimal_string)

inexact_testcases = tuple(testcase_func(fmt, index)
                          for testcase_func in inexact_testcase_funcs
                          for index in range(2)
                          for fmt in all_IEEE_fmts)

class TestInexact:

    @pytest.mark.parametrize('testcase, kind', product(
        inexact_testcases,
        (HandlerKind.DEFAULT, HandlerKind.NO_FLAG, HandlerKind.MAYBE_FLAG,
         HandlerKind.RECORD_EXCEPTION),
    ))
    def test_basic_kinds(self, testcase, kind, context):
        exc_class, handler_class, op_tuple, inexact_func = testcase

        context = get_context() if context is None else context
        context.set_handler(handler_class, kind)

        inexact_func(context)
        assert context.flags == 0 if kind == HandlerKind.NO_FLAG else Flags.INEXACT
        if kind == HandlerKind.RECORD_EXCEPTION:
            assert len(context.exceptions) == 1
            exception = context.exceptions[0]
            assert isinstance(exception, exc_class)
            assert exception.op_tuple == op_tuple
        else:
            assert not context.exceptions

    @pytest.mark.parametrize('testcase', inexact_testcases)
    def test_substitute_value(self, testcase, context):
        exc_class, handler_class, op_tuple, inexact_func = testcase

        context = get_context() if context is None else context
        context.set_handler(handler_class, HandlerKind.SUBSTITUTE_VALUE, substitute_plus_one)

        result = inexact_func(context)
        if isinstance(result, str):
            assert result == '+1.0'
        elif isinstance(result, int):
            assert result == 1
        else:
            assert floats_equal(result, result.fmt.make_one(False))
        assert context.flags == Flags.INEXACT
        assert not context.exceptions

    @pytest.mark.parametrize('testcase', inexact_testcases)
    def test_substitute_value_xor(self, testcase, context):
        exc_class, handler_class, op_tuple, inexact_func = testcase

        context = get_context() if context is None else context
        context.set_handler(handler_class, HandlerKind.SUBSTITUTE_VALUE_XOR, substitute_plus_one)

        result = inexact_func(context)
        # Ignored unless one of these
        if op_tuple[0] in {OP_MULTIPLY, OP_DIVIDE}:
            sign = op_tuple[1].sign ^ op_tuple[2].sign
            assert floats_equal(result, result.fmt.make_one(sign))
        assert context.flags == Flags.INEXACT
        assert not context.exceptions

    @pytest.mark.parametrize('testcase', inexact_testcases)
    def test_abrupt_underflow(self, testcase, context):
        handler_class = testcase[1]
        context = get_context() if context is None else context
        with pytest.raises(TypeError) as e:
            context.set_handler(handler_class, HandlerKind.ABRUPT_UNDERFLOW)

        assert 'must be subclasses of Underflow' in str(e.value)

    @pytest.mark.parametrize('testcase', inexact_testcases)
    def test_raise(self, testcase, context):
        exc_class, handler_class, op_tuple, inexact_func = testcase

        context = get_context() if context is None else context
        context.set_handler(handler_class, HandlerKind.RAISE)

        with pytest.raises(exc_class) as e:
            inexact_func(context)

        e = e.value
        assert e.op_tuple == op_tuple
        assert context.flags == Flags.INEXACT
        assert not context.exceptions


def overflow_multiply(dst_fmt, index):
    lhs = dst_fmt.make_largest_finite(random.choice(signs))
    rhs = random.choice(all_IEEE_fmts).from_value(2)
    op_tuple = (OP_MULTIPLY, lhs, rhs)
    handler_class = (Overflow, IEEEError)[index]
    return Overflow, handler_class, op_tuple, partial(dst_fmt.multiply, lhs, rhs)


def overflow_from_string(dst_fmt, index):
    lhs = '1e200000'
    op_tuple = (OP_FROM_STRING, lhs)
    handler_class = (Overflow, IEEEError)[index]
    return Overflow, handler_class, op_tuple, partial(dst_fmt.from_string, lhs)


overflow_testcase_funcs = (overflow_multiply, overflow_from_string)

overflow_testcases = tuple(testcase_func(fmt, index)
                           for testcase_func in overflow_testcase_funcs
                           for index in range(2)
                           for fmt in all_IEEE_fmts)

class TestOverflow:

    @pytest.mark.parametrize('testcase, kind, inexact_kind', product(
        overflow_testcases,
        (HandlerKind.DEFAULT, HandlerKind.NO_FLAG, HandlerKind.MAYBE_FLAG,
         HandlerKind.RECORD_EXCEPTION),
        (HandlerKind.DEFAULT, HandlerKind.NO_FLAG, HandlerKind.MAYBE_FLAG,
         HandlerKind.RECORD_EXCEPTION),
    ))
    def test_basic_kinds(self, testcase, kind, inexact_kind, quiet_context):
        exc_class, handler_class, op_tuple, overflow_func = testcase
        context = quiet_context
        context.set_handler(handler_class, kind)
        context.set_handler(Inexact, inexact_kind)

        overflow_func(context)

        flags = 0 if kind == HandlerKind.NO_FLAG else Flags.OVERFLOW
        flags |= 0 if inexact_kind == HandlerKind.NO_FLAG else Flags.INEXACT
        assert context.flags == flags

        record_count = ((kind == HandlerKind.RECORD_EXCEPTION) +
                        (inexact_kind == HandlerKind.RECORD_EXCEPTION))
        if record_count:
            assert len(context.exceptions) == record_count
            if kind == HandlerKind.RECORD_EXCEPTION:
                assert isinstance(context.exceptions[0], exc_class)
            if inexact_kind == HandlerKind.RECORD_EXCEPTION:
                assert isinstance(context.exceptions[-1], Inexact)
            assert (exception.op_tuple == op_tuple for exception in context.exceptions)
        else:
            assert not context.exceptions

    @pytest.mark.parametrize('testcase, inexact', product(overflow_testcases, (True, False)))
    def test_substitute_value(self, testcase, inexact, quiet_context):
        exc_class, handler_class, op_tuple, overflow_func = testcase

        context = quiet_context
        context.set_handler(handler_class, HandlerKind.SUBSTITUTE_VALUE, substitute_plus_one)
        if inexact:
            context.set_handler(Inexact, HandlerKind.SUBSTITUTE_VALUE, substitute_plus_zero)

        result = overflow_func(context)
        if inexact:
            assert floats_equal(result, result.fmt.make_zero(False))
        else:
            assert floats_equal(result, result.fmt.make_one(False))
        assert context.flags == Flags.OVERFLOW | Flags.INEXACT
        assert not context.exceptions

    @pytest.mark.parametrize('testcase', overflow_testcases)
    def test_substitute_value_xor(self, testcase, quiet_context):
        exc_class, handler_class, op_tuple, overflow_func = testcase

        context = quiet_context
        context.set_handler(handler_class, HandlerKind.SUBSTITUTE_VALUE_XOR, substitute_plus_one)

        result = overflow_func(context)
        # Ignored unless one of these
        if op_tuple[0] in {OP_MULTIPLY, OP_DIVIDE}:
            sign = op_tuple[1].sign ^ op_tuple[2].sign
            assert floats_equal(result, result.fmt.make_one(sign))
        else:
            assert floats_equal(result, result.fmt.make_infinity(False))
        assert context.flags == Flags.OVERFLOW | Flags.INEXACT
        assert not context.exceptions

    @pytest.mark.parametrize('testcase', overflow_testcases)
    def test_abrupt_underflow(self, testcase, quiet_context):
        handler_class = testcase[1]
        context = quiet_context
        with pytest.raises(TypeError) as e:
            context.set_handler(handler_class, HandlerKind.ABRUPT_UNDERFLOW)

        assert 'must be subclasses of Underflow' in str(e.value)

    @pytest.mark.parametrize('testcase', overflow_testcases)
    def test_raise(self, testcase, quiet_context):
        exc_class, handler_class, op_tuple, overflow_func = testcase

        context = quiet_context
        context.set_handler(handler_class, HandlerKind.RAISE)

        with pytest.raises(exc_class) as e:
            overflow_func(context)

        e = e.value
        assert e.op_tuple == op_tuple
        assert context.flags == Flags.OVERFLOW   # Inexact has not been signalled
        assert not context.exceptions

    @pytest.mark.parametrize('testcase', overflow_testcases)
    def test_raise_inexact(self, testcase, quiet_context):
        exc_class, handler_class, op_tuple, overflow_func = testcase

        context = quiet_context
        context.set_handler(Inexact, HandlerKind.RAISE)

        with pytest.raises(Inexact) as e:
            overflow_func(context)

        e = e.value
        assert e.op_tuple == op_tuple
        assert context.flags == Flags.OVERFLOW | Flags.INEXACT
        assert not context.exceptions


def underflow_multiply(dst_fmt, index):
    lhs = dst_fmt.make_smallest_normal(random.choice(signs))
    rhs = random.choice(all_IEEE_fmts).from_value(0.5)
    op_tuple = (OP_MULTIPLY, lhs, rhs)
    handler_class = (UnderflowExact, Underflow, IEEEError)[index]
    return UnderflowExact, lhs.sign, handler_class, op_tuple, partial(dst_fmt.multiply, lhs, rhs)


def underflow_from_string(dst_fmt, index):
    lhs = random.choice(('1e-200000', '-1e-200000'))
    op_tuple = (OP_FROM_STRING, lhs)
    handler_class = (UnderflowInexact, Underflow, IEEEError)[index]
    return (UnderflowInexact, lhs[0] == '-', handler_class, op_tuple,
            partial(dst_fmt.from_string, lhs))


underflow_testcase_funcs = (underflow_multiply, underflow_from_string)
underflow_testcases = tuple(testcase_func(fmt, index)
                            for testcase_func in underflow_testcase_funcs
                            for index in range(3)
                            for fmt in all_IEEE_fmts)


class TestUnderflow:

    @pytest.mark.parametrize('testcase, kind, inexact_kind', product(
        underflow_testcases,
        (HandlerKind.DEFAULT, HandlerKind.NO_FLAG, HandlerKind.MAYBE_FLAG,
         HandlerKind.RECORD_EXCEPTION),
        (HandlerKind.DEFAULT, HandlerKind.NO_FLAG, HandlerKind.MAYBE_FLAG,
         HandlerKind.RECORD_EXCEPTION),
    ))
    def test_basic_kinds(self, testcase, kind, inexact_kind, quiet_context):
        exc_class, _sign, handler_class, op_tuple, underflow_func = testcase
        context = quiet_context
        context.set_handler(handler_class, kind)
        context.set_handler(Inexact, inexact_kind)

        underflow_func(context)

        # Underflow handling is a bit messy; checking needs to be precise
        if exc_class is UnderflowExact or kind == HandlerKind.NO_FLAG:
            flags = 0
        else:
            flags = Flags.UNDERFLOW
        record_count = 1 if (flags and kind == HandlerKind.RECORD_EXCEPTION) else 0

        if exc_class is UnderflowInexact and inexact_kind != HandlerKind.NO_FLAG:
            flags |= Flags.INEXACT
            if inexact_kind == HandlerKind.RECORD_EXCEPTION:
                record_count += 1

        if record_count:
            if (flags & Flags.UNDERFLOW and kind == HandlerKind.RECORD_EXCEPTION):
                assert isinstance(context.exceptions[0], exc_class)
            if inexact_kind == HandlerKind.RECORD_EXCEPTION and flags & Flags.INEXACT:
                assert isinstance(context.exceptions[-1], Inexact)
            assert (exception.op_tuple == op_tuple for exception in context.exceptions)
        else:
            assert not context.exceptions

    @pytest.mark.parametrize('testcase, inexact', product(underflow_testcases, (True, False)))
    def test_substitute_value(self, testcase, inexact, quiet_context):
        exc_class, _sign, handler_class, op_tuple, underflow_func = testcase

        context = quiet_context
        context.set_handler(handler_class, HandlerKind.SUBSTITUTE_VALUE, substitute_plus_one)
        if inexact:
            context.set_handler(Inexact, HandlerKind.SUBSTITUTE_VALUE, substitute_plus_zero)

        result = underflow_func(context)

        if exc_class is UnderflowExact:
            assert floats_equal(result, result.fmt.make_one(False))
            assert context.flags == 0
        else:
            if inexact:
                assert floats_equal(result, result.fmt.make_zero(False))
            else:
                assert floats_equal(result, result.fmt.make_one(False))
            assert context.flags == Flags.UNDERFLOW | Flags.INEXACT
        assert not context.exceptions

    @pytest.mark.parametrize('testcase', underflow_testcases)
    def test_substitute_value_xor(self, testcase, quiet_context):
        exc_class, sign, handler_class, op_tuple, underflow_func = testcase

        context = quiet_context
        context.set_handler(handler_class, HandlerKind.SUBSTITUTE_VALUE_XOR, substitute_plus_one)

        result = underflow_func(context)
        # Ignored unless one of these
        if op_tuple[0] in {OP_MULTIPLY, OP_DIVIDE}:
            assert floats_equal(result, result.fmt.make_one(sign))
        else:
            assert result.is_subnormal() or result.is_zero()
            assert result.sign is sign
        if exc_class is UnderflowExact:
            assert context.flags == 0
        else:
            assert context.flags == Flags.UNDERFLOW | Flags.INEXACT
        assert not context.exceptions

    @pytest.mark.parametrize('testcase, rounding', product(underflow_testcases, all_roundings))
    def test_abrupt_underflow(self, testcase, rounding, quiet_context):
        exc_class, sign, handler_class, op_tuple, underflow_func = testcase
        context = quiet_context
        context.rounding = rounding

        if handler_class is IEEEError:
            with pytest.raises(TypeError) as e:
                context.set_handler(handler_class, HandlerKind.ABRUPT_UNDERFLOW)
        else:
            assert context.flags == 0
            context.set_handler(handler_class, HandlerKind.ABRUPT_UNDERFLOW)
            result = underflow_func(context)

            assert context.flags == Flags.UNDERFLOW | Flags.INEXACT
            if rounding == ROUND_CEILING:
                is_zero = sign
            elif rounding == ROUND_FLOOR:
                is_zero = not sign
            else:
                is_zero = rounding in {ROUND_HALF_EVEN, ROUND_HALF_UP, ROUND_HALF_DOWN, ROUND_DOWN}
            if is_zero:
                assert result.is_zero()
                assert result.sign is sign
            else:
                assert floats_equal(result, result.fmt.make_smallest_normal(sign))

        assert not context.exceptions

    @pytest.mark.parametrize('testcase', underflow_testcases)
    def test_raise(self, testcase, quiet_context):
        exc_class, _sign, handler_class, op_tuple, underflow_func = testcase

        context = quiet_context
        context.set_handler(handler_class, HandlerKind.RAISE)

        with pytest.raises(exc_class) as e:
            underflow_func(context)

        e = e.value
        assert e.op_tuple == op_tuple
        if exc_class is UnderflowExact:
            assert context.flags == 0
        else:
            assert context.flags == Flags.UNDERFLOW  # Inexact has not been signalled
        assert not context.exceptions

    @pytest.mark.parametrize('testcase', underflow_testcases)
    def test_raise_inexact(self, testcase, quiet_context):
        exc_class, _sign, handler_class, op_tuple, underflow_func = testcase

        context = quiet_context
        context.set_handler(Inexact, HandlerKind.RAISE)

        if exc_class is UnderflowExact:
            underflow_func(context)
            assert context.flags == 0
        else:
            with pytest.raises(Inexact) as e:
                underflow_func(context)

            e = e.value
            assert e.op_tuple == op_tuple
            assert context.flags == Flags.UNDERFLOW | Flags.INEXACT  # Both have been signalled
        assert not context.exceptions

    @pytest.mark.parametrize('fmt', all_IEEE_fmts)
    def test_ue_next_up(self, fmt, quiet_context):
        quiet_context.set_handler(Underflow, HandlerKind.RAISE)
        value = fmt.make_zero(True)
        with pytest.raises(UnderflowExact) as e:
            value.next_up(quiet_context)
        assert e.value.op_tuple == (OP_NEXT_UP, value)

        value = e.value.default_result
        assert value.is_subnormal()
        assert value.next_down(quiet_context).is_zero()

        def handler(exception, context):
            assert isinstance(exception, UnderflowExact)
            return fmt.make_infinity(False)
        quiet_context.set_handler(Underflow, HandlerKind.SUBSTITUTE_VALUE, handler)
        assert value.next_up().is_infinite()

    def test_ue_convert(self, quiet_context):
        quiet_context.set_handler(Underflow, HandlerKind.RAISE)
        value = IEEEdouble.from_string('0x1.000000p-127')
        with pytest.raises(UnderflowExact) as e:
            IEEEsingle.convert(value, quiet_context)
        assert e.value.op_tuple == (OP_CONVERT, value)

    @pytest.mark.parametrize('fmt', all_IEEE_fmts)
    def test_ue_to_decimal_string(self, fmt, quiet_context):
        value = fmt.divide(fmt.make_smallest_normal(False), fmt.from_int(2), quiet_context)
        assert quiet_context.flags == 0
        quiet_context.set_handler(Underflow, HandlerKind.RAISE)
        # Should not raise
        value.to_decimal_string(precision=0, context=quiet_context)

    @pytest.mark.parametrize('fmt', all_IEEE_fmts)
    def test_ue_to_hexadecimal_string(self, fmt, quiet_context):
        value = fmt.divide(fmt.make_smallest_normal(False), fmt.from_int(2), quiet_context)
        assert quiet_context.flags == 0
        quiet_context.set_handler(Underflow, HandlerKind.RAISE)
        with pytest.raises(UnderflowExact) as e:
            value.to_string(context=quiet_context)
        assert e.value.op_tuple[0] == OP_TO_STRING

    @pytest.mark.parametrize('string', ('0x1.000000p-15', '0.000030517578125'))
    def test_ue_from_string(self, string, quiet_context):
        quiet_context.set_handler(Underflow, HandlerKind.RAISE)
        with pytest.raises(UnderflowExact) as e:
            IEEEhalf.from_string(string)
        assert e.value.op_tuple == (OP_FROM_STRING, string)

    @pytest.mark.parametrize('fmt', (IEEEhalf, IEEEsingle, IEEEdouble))
    def test_ue_from_float(self, fmt, quiet_context):
        value =  pow(2, fmt.e_min - 1)
        quiet_context.set_handler(Underflow, HandlerKind.RAISE)
        with pytest.raises(UnderflowExact) as e:
            fmt.from_float(value)
        assert e.value.op_tuple == (OP_FROM_FLOAT, value)

    def test_ue_from_decimal(self, quiet_context):
        value = Decimal('0.000030517578125')
        quiet_context.set_handler(Underflow, HandlerKind.RAISE)
        with pytest.raises(UnderflowExact) as e:
            IEEEhalf.from_decimal(value)
        assert e.value.op_tuple == (OP_FROM_DECIMAL, value)

    @pytest.mark.parametrize('fmt', all_IEEE_fmts)
    def test_ue_from_fraction(self, fmt, quiet_context):
        value = Fraction(1, 1 << fmt.e_max)
        quiet_context.set_handler(Underflow, HandlerKind.RAISE)
        with pytest.raises(UnderflowExact) as e:
            fmt.from_fraction(value)
        assert e.value.op_tuple == (OP_FROM_FRACTION, value)

    def test_ue_unpack_value(self, quiet_context):
        value = bytes((0, 2))
        quiet_context.set_handler(Underflow, HandlerKind.RAISE)
        with pytest.raises(UnderflowExact) as e:
            IEEEhalf.unpack_value(value, None, quiet_context)
        assert e.value.op_tuple == (OP_UNPACK_VALUE, value, None)

    @pytest.mark.parametrize('fmt', all_IEEE_fmts)
    def test_ue_add(self, fmt, quiet_context):
        lhs = fmt.make_zero(False)
        rhs = fmt.make_smallest_subnormal(False)
        quiet_context.set_handler(Underflow, HandlerKind.RAISE)
        with pytest.raises(UnderflowExact) as e:
            fmt.add(lhs, rhs, quiet_context)
        assert e.value.op_tuple == (OP_ADD, lhs, rhs)

    @pytest.mark.parametrize('fmt', all_IEEE_fmts)
    def test_ue_subtract(self, fmt, quiet_context):
        lhs = fmt.make_smallest_subnormal(False)
        rhs = fmt.make_zero(False)
        quiet_context.set_handler(Underflow, HandlerKind.RAISE)
        with pytest.raises(UnderflowExact) as e:
            fmt.subtract(lhs, rhs, quiet_context)
        assert e.value.op_tuple == (OP_SUBTRACT, lhs, rhs)

    @pytest.mark.parametrize('fmt', all_IEEE_fmts)
    def test_ue_multiply(self, fmt, quiet_context):
        lhs = fmt.make_smallest_subnormal(False)
        rhs = fmt.make_one(False)
        quiet_context.set_handler(Underflow, HandlerKind.RAISE)
        with pytest.raises(UnderflowExact) as e:
            fmt.multiply(lhs, rhs, quiet_context)
        assert e.value.op_tuple == (OP_MULTIPLY, lhs, rhs)

    @pytest.mark.parametrize('fmt', all_IEEE_fmts)
    def test_ue_divide(self, fmt, quiet_context):
        lhs = fmt.make_smallest_normal(False)
        rhs = fmt.from_int(2)
        quiet_context.set_handler(Underflow, HandlerKind.RAISE)
        with pytest.raises(UnderflowExact) as e:
            fmt.divide(lhs, rhs, quiet_context)
        assert e.value.op_tuple == (OP_DIVIDE, lhs, rhs)

    @pytest.mark.parametrize('fmt', all_IEEE_fmts)
    def test_ue_fma(self, fmt, quiet_context):
        lhs = fmt.make_smallest_normal(False)
        rhs = fmt.make_one(False).scaleb(-1)
        addend = fmt.make_zero(True)
        quiet_context.set_handler(Underflow, HandlerKind.RAISE)
        with pytest.raises(UnderflowExact) as e:
            fmt.fma(lhs, rhs, addend, quiet_context)
        assert e.value.op_tuple == (OP_FMA, lhs, rhs, addend)

    @pytest.mark.parametrize('fmt', (IEEEhalf, IEEEsingle, IEEEdouble))
    def test_ue_sqrt(self, fmt, quiet_context):
        string = f'0x1.088p{fmt.e_min * 2 - 1}'
        value = IEEEquad.from_string(string, quiet_context)
        quiet_context.set_handler(Underflow, HandlerKind.RAISE)
        with pytest.raises(UnderflowExact) as e:
            fmt.sqrt(value, quiet_context)
        assert e.value.op_tuple == (OP_SQRT, value)


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

    @pytest.mark.parametrize('width, precision', (
        (160, 144),
        (192, 175),
        (224, 206),
        (256, 237),
    ))
    def test_IEEE(self, width, precision):
        assert BinaryFormat.from_IEEE(width).precision == precision

    @pytest.mark.parametrize('precision, e_max', (
        (32, 1023),
        (64, 16383),
        (128, 65535),
        (237, 1048575),
    ))
    def test_from_precision(self, precision, e_max):
        assert BinaryFormat.from_precision(precision).e_max == e_max

    @pytest.mark.parametrize('triple', (
        (2, 5, -5),
        (3, 1, -5),
        (3, 2, 0),
    ))
    def test_from_triple_bad(self, triple):
        with pytest.raises(ValueError):
            BinaryFormat.from_triple(*triple)

    @pytest.mark.parametrize('triple', (
        (16.0, 15, -15),
        (16, 15.0, -15),
        (16, 15, -15.0),
    ))
    def test_from_triple_bad_type(self, triple):
        with pytest.raises(TypeError):
            BinaryFormat.from_triple(*triple)

    @pytest.mark.parametrize('triple', (
        (3, 2, -1),
        (16, 15, -15),
    ))
    def test_from_triple_good(self, triple):
        fmt = BinaryFormat.from_triple(*triple)
        assert (fmt.precision, fmt.e_max, fmt.e_min) == triple

    @pytest.mark.parametrize('width', (-1, 0, 40, 80, 96))
    def test_IEEE_bad(self, width):
        with pytest.raises(ValueError):
            BinaryFormat.from_IEEE(width)

    def test_immutable(self):
        with pytest.raises(AttributeError):
            IEEEhalf.precision = 5

    @pytest.mark.parametrize('fmt, text', product(
        all_IEEE_fmts, ('0', '-0', 'Inf', '-Inf', 'NaN', '1', '-1',
                        '123.456', '1e300', '-2.65721e-310')))
    def test_from_float(self, fmt, text, quiet_context):
        # Work out what should happen
        with local_context() as ctx:
            answer = IEEEdouble.from_string(text)
            ctx.flags = 0
            answer = fmt.convert(answer)
            flags = ctx.flags

        py_value = float(text)
        result = fmt.from_float(py_value)
        assert result.fmt == fmt
        assert floats_equal(result, answer)
        assert get_context().flags == flags

    @pytest.mark.parametrize('text', ('+0', '-0', 'Inf', '-Inf', '1.1', '-1.25', 'NaN'))
    def test_from_decimal(self, text, context):
        for fmt in all_IEEE_fmts:
            with local_context(context) as ctx1:
                value1 = fmt.from_string(text)
            with local_context(context) as ctx2:
                value2 = fmt.from_decimal(Decimal(text))

            assert value1.fmt is fmt
            assert floats_equal(value1, value2)
            assert ctx1.flags == ctx2.flags

    @pytest.mark.parametrize('fmt, fraction, answer, flags', (
        (IEEEdouble, Fraction(1, 3), '0x1.5555555555555p-2', Flags.INEXACT),
        (IEEEdouble, Fraction(-1, 2), '-0.5', 0),
        (IEEEhalf, Fraction(65504, 1), '65504', 0),
        (IEEEhalf, Fraction(-65505, 1), '-65504', Flags.INEXACT),
        (IEEEhalf, Fraction(-65520, 1), '-Inf', Flags.OVERFLOW | Flags.INEXACT),
        # This test would fail if each integer were converted to IEEEdouble before dividing
        (IEEEdouble, Fraction(72057594037927941, 72057594037927933), '1.0000000000000002',
         Flags.INEXACT),
    ))
    def test_from_fraction(self, fmt, fraction, answer, flags, quiet_context):
        with local_context():
            answer = fmt.from_string(answer)
        result = fmt.from_fraction(fraction)
        assert result.fmt is fmt
        assert floats_equal(result, answer)
        assert quiet_context.flags == flags

    @pytest.mark.parametrize('fmt, value', product(
        all_IEEE_fmts,
        (-1, 0, 1, 123456 << 5000, -1.3, 1.25, 1.2e1000, '6.25', '-1.1', '-Inf', 'NaN2', 'sNaN',
         Decimal(1.2), Decimal(-15))))
    def test_from_value(self, fmt, value, quiet_context):
        with local_context() as ctx:
            if isinstance(value, int):
                answer = fmt.from_int(value)
            elif isinstance(value, float):
                answer = fmt.from_float(value)
            elif isinstance(value, Decimal):
                answer = fmt.from_decimal(value)
            else:
                answer = fmt.from_string(value)
            flags = ctx.flags

        result = fmt.from_value(value)
        assert result.fmt == fmt
        assert floats_equal(result, answer)
        assert get_context().flags == flags
        # Test binary is accepted too.
        assert floats_equal(fmt.from_value(answer.pack()), answer)

    def test_from_value_type(self):
        with pytest.raises(TypeError):
            IEEEdouble.from_value(complex(1, 2))

    def test_from_decimal_type(self):
        with pytest.raises(TypeError):
            IEEEdouble.from_decimal(1.2)

    def test_from_fraction_type(self):
        with pytest.raises(TypeError):
            IEEEdouble.from_fraction(1)

    def test_from_int_type(self):
        with pytest.raises(TypeError):
            IEEEdouble.from_int(1.0)

    def test_from_float_type(self):
        with pytest.raises(TypeError):
            IEEEdouble.from_float(1)

    def test_from_string_type(self):
        with pytest.raises(TypeError):
            IEEEdouble.from_string(b'')

    @pytest.mark.parametrize('fmt', all_IEEE_fmts)
    def test_from_string_stripping(self, fmt):
        assert fmt.from_string('  _1_00__ ') == fmt.from_int(100)
        assert fmt.from_string(' _0_X_1A_P0_ ') == fmt.from_int(26)

    @pytest.mark.parametrize('fmt, string', product(all_IEEE_fmts, ('_ 1 _', '0x0', 'l')))
    def test_from_string_bad(self, fmt, string, context):
        with pytest.raises(InvalidFromString) as e:
            fmt.from_string(string, context)
        assert e.value.op_tuple == (OP_FROM_STRING, string)

    @pytest.mark.parametrize('fmt', all_IEEE_fmts)
    def test_from_string_unicode(self, fmt):
        assert fmt.from_string(' ０x１＿０２p0 ') == fmt.from_int(0x102)

    def test_repr(self):
        assert repr(IEEEdouble) == 'BinaryFormat(precision=53, e_max=1023, e_min=-1022)'

    def test_eq(self):
        assert BinaryFormat.from_triple(8, 99, -99) == BinaryFormat.from_triple(8, 99, -99)
        assert BinaryFormat.from_triple(8, 99, -99) != BinaryFormat.from_triple(8, 99, -100)
        assert BinaryFormat.from_triple(8, 99, -99) != BinaryFormat.from_triple(8, 100, -99)
        assert BinaryFormat.from_triple(8, 99, -99) != BinaryFormat.from_triple(9, 99, -99)
        assert BinaryFormat.from_triple(8, 99, -99) != 1


class TestBinary:

    def test_constructor_type1(self):
        with pytest.raises(TypeError):
            Binary(1.0, False, 0, 0)

    def test_constructor_type2(self):
        with pytest.raises(TypeError):
            Binary(IEEEdouble, 0, 0, 0)

    def test_constructor_type3(self):
        with pytest.raises(TypeError):
            Binary(IEEEdouble, False, 0.0, 0)

    def test_constructor_type4(self):
        with pytest.raises(TypeError):
            Binary(IEEEdouble, False, 0, 0.0)

    def test_constructor_value1(self):
        with pytest.raises(ValueError):
            Binary(IEEEdouble, False, -1, 0)

    def test_constructor_value2(self):
        with pytest.raises(ValueError):
            Binary(IEEEdouble, False, IEEEdouble.e_max + IEEEdouble.e_bias + 1, 0)

    def test_constructor_value3(self):
        with pytest.raises(ValueError):
            Binary(IEEEdouble, False, 0, -1)

    def test_constructor_value4(self):
        with pytest.raises(ValueError):
            Binary(IEEEdouble, False, 0, IEEEdouble.int_bit)

    def test_constructor_value5(self):
        with pytest.raises(ValueError):
            Binary(IEEEdouble, False, 1, -1)

    def test_constructor_value6(self):
        with pytest.raises(ValueError):
            Binary(IEEEdouble, False, 1, IEEEdouble.max_significand + 1)

    def test_repr_str(self):
        d = IEEEdouble.from_string('1.25')
        assert repr(d) == '0x1.4000000000000p+0'
        assert str(d) == repr(d)

    def test_immutable(self):
        d = IEEEdouble.from_string('1.25')
        with pytest.raises(AttributeError):
            d.sign = True

    @pytest.mark.parametrize('fmt, sign', product(all_IEEE_fmts, signs))
    def test_as_integer_ratio_inf(self, fmt, sign):
        with pytest.raises(OverflowError):
            fmt.make_infinity(sign).as_integer_ratio()

    @pytest.mark.parametrize('fmt, is_signalling', product(all_IEEE_fmts, signs))
    def test_as_integer_ratio_nan(self, fmt, is_signalling):
        with pytest.raises(ValueError):
            fmt.make_nan(False, is_signalling, 0).as_integer_ratio()

    @pytest.mark.parametrize('fmt, testcase', product(
        all_IEEE_fmts, (
            ('1', 1, 1),
            ('-1', -1, 1),
            ('0', 0, 1),
            ('-0', 0, 1),
            ('-0.5', -1, 2),
            ('123.25', 493, 4),
            ('-0.000518798828125', -17, 32768),
        )))
    def test_as_integer_ratio_exact(self, fmt, testcase):
        text, n, d = testcase
        assert fmt.from_string(text).as_integer_ratio() == (n, d)

    def test_as_integer_ratio_inexact(self):
        pi =  IEEEdouble.from_string('3.141592653589793')
        assert pi.as_integer_ratio() == (884279719003555, 281474976710656)

    @pytest.mark.parametrize('fmt', all_IEEE_fmts)
    def test_copy_sign(self, fmt, context):
        minus_one = fmt.from_int(-1)
        plus_one = fmt.from_int(1)
        two = fmt.from_int(2)
        minus_two = fmt.from_int(-2)
        assert floats_equal(minus_one.copy_sign(two), plus_one)
        assert floats_equal(two.copy_sign(minus_one), minus_two)
        assert minus_one.copy_sign(minus_one) is minus_one
        assert two.copy_sign(two) is two

    @pytest.mark.parametrize('fmt', all_IEEE_fmts)
    def test_abs(self, fmt, context):
        d = fmt.from_int(1)
        assert abs(d) is d

        e = fmt.from_int(-1)
        assert floats_equal(abs(e), d)

        f = fmt.from_string('-NaN1')
        g = fmt.from_string('NaN1')
        assert abs(f) is f
        assert abs(g) is g

        assert context.flags == 0

        h = fmt.from_string('-sNaN')
        with pytest.raises(SignallingNaNOperand) as e:
            abs(h)
        assert e.value.op_tuple == (OP_ABS, h)
        # The NaN is quietened; sign is not changed
        assert floats_equal(e.value.default_result, f)
        assert context.flags == Flags.INVALID

        s = fmt.make_smallest_subnormal(False)
        context.set_handler(Underflow, HandlerKind.RAISE)
        with pytest.raises(UnderflowExact) as e:
            abs(s)
        assert e.value.op_tuple == (OP_ABS, s)
        assert e.value.default_result is s

    @pytest.mark.parametrize('fmt', all_IEEE_fmts)
    def test_abs_quiet(self, fmt, context):
        d = fmt.from_int(1)
        assert d.abs_quiet() is d

        e = fmt.from_int(-1)
        assert floats_equal(e.abs_quiet(), d)

        pi = fmt.make_infinity(False)
        ni = fmt.make_infinity(True)
        assert pi.abs_quiet() is pi
        assert floats_equal(ni.abs_quiet(), pi)

        f = fmt.from_string('-NaN')
        g = fmt.from_string('NaN')
        assert floats_equal(f.abs_quiet(), g)

        h = fmt.from_string('sNaN')
        assert h.abs_quiet() is h

        assert context.flags == 0

    @pytest.mark.parametrize('fmt', all_IEEE_fmts)
    def test_negate(self, fmt, context):
        d = fmt.from_int(1)
        e = fmt.from_int(-1)

        assert floats_equal(-d, e)
        assert floats_equal(-e, d)

        f = fmt.from_string('-NaN1')
        g = fmt.from_string('NaN1')
        assert -f is f
        assert -g is g

        assert context.flags == 0

        h = fmt.from_string('-sNaN')
        with pytest.raises(SignallingNaNOperand) as e:
            -h
        assert e.value.op_tuple == (OP_MINUS, h)
        # The NaN is quietened; sign is not changed
        assert floats_equal(e.value.default_result, f)
        assert context.flags == Flags.INVALID

        s = fmt.make_smallest_subnormal(False)
        context.set_handler(Underflow, HandlerKind.RAISE)
        with pytest.raises(UnderflowExact) as e:
            -s
        assert e.value.op_tuple == (OP_MINUS, s)

    @pytest.mark.parametrize('fmt', all_IEEE_fmts)
    def test_negate_quiet(self, fmt, context):
        d = fmt.from_int(1)
        e = fmt.from_int(-1)

        assert floats_equal(d.negate_quiet(), e)
        assert floats_equal(e.negate_quiet(), d)

        f = fmt.from_string('-NaN')
        g = fmt.from_string('NaN')
        assert floats_equal(f.negate_quiet(), g)
        assert floats_equal(g.negate_quiet(), f)

        h = fmt.from_string('sNaN')
        k = fmt.from_string('-sNaN')
        assert floats_equal(k, h.negate_quiet())
        assert floats_equal(h, k.negate_quiet())
        assert context.flags == 0

    @pytest.mark.parametrize('fmt', all_IEEE_fmts)
    def test_plus(self, fmt, context):
        d = fmt.from_int(1)
        e = fmt.from_int(-1)

        assert +d is d
        assert +e is e

        f = fmt.from_string('-NaN1')
        g = fmt.from_string('NaN1')
        assert +f is f
        assert +g is g

        assert context.flags == 0

        k = fmt.from_string('-sNaN')
        with pytest.raises(SignallingNaNOperand) as e:
            +k
        assert e.value.op_tuple == (OP_PLUS, k)
        # The NaN is quietened; sign is not changed
        assert floats_equal(e.value.default_result, f)
        assert context.flags == Flags.INVALID

        s = fmt.make_smallest_subnormal(False)
        context.set_handler(Underflow, HandlerKind.RAISE)
        with pytest.raises(UnderflowExact) as e:
            +s
        assert e.value.op_tuple == (OP_PLUS, s)
        assert e.value.default_result is s


    @pytest.mark.parametrize('text, rhs, compare', (
        # Comparisons of Infs
        ('1', Decimal('Inf'), Compare.LESS_THAN),
        ('1', Decimal('-Inf'), Compare.GREATER_THAN),
        ('-Inf', Decimal('-Inf'), Compare.EQUAL),
        ('Inf', Decimal('Inf'), Compare.EQUAL),
        ('-Inf', Decimal('0'), Compare.LESS_THAN),
        ('Inf', Decimal('0'), Compare.GREATER_THAN),
        ('Inf', 0, Compare.GREATER_THAN),
        ('-Inf', 0, Compare.LESS_THAN),
        ('Inf', 1.1, Compare.GREATER_THAN),
        ('-Inf', -1.1, Compare.LESS_THAN),
        ('-Inf', Fraction(1, 2), Compare.LESS_THAN),
        ('Inf', Fraction(1, 2), Compare.GREATER_THAN),
        # Comparison of NaNs
        ('1', Decimal('NaN'), Compare.UNORDERED),
        ('Inf', Decimal('-NaN'), Compare.UNORDERED),
        ('-NaN', Decimal('NaN'), Compare.UNORDERED),
        ('NaN', Decimal('1'), Compare.UNORDERED),
        ('-NaN', 1, Compare.UNORDERED),
        ('NaN', 1.0, Compare.UNORDERED),
        ('NaN', Fraction(1, 3), Compare.UNORDERED),
        # Comparison of one
        ('1', Decimal('-1'), Compare.GREATER_THAN),
        ('1', Decimal('1'), Compare.EQUAL),
        ('1', -1, Compare.GREATER_THAN),
        ('1', 1, Compare.EQUAL),
        ('1', -1.0, Compare.GREATER_THAN),
        ('1', 1.0, Compare.EQUAL),
        ('1', Fraction(-1, 1), Compare.GREATER_THAN),
        ('1', Fraction(1, 1), Compare.EQUAL),
        ('1', IEEEhalf.from_int(-1), Compare.GREATER_THAN),
        ('1', IEEEhalf.from_int(1), Compare.EQUAL),
    ))
    def test_comparisons_exact(self, text, rhs, compare, context):
        for fmt in all_IEEE_fmts:
            lhs = fmt.from_string(text)
            if compare == Compare.EQUAL:
                assert lhs == rhs
                assert not lhs != rhs
                assert lhs >= rhs
                assert not lhs > rhs
                assert lhs <= rhs
                assert not lhs < rhs
            elif compare == Compare.LESS_THAN:
                assert not lhs == rhs
                assert lhs != rhs
                assert not lhs >= rhs
                assert not lhs > rhs
                assert lhs <= rhs
                assert lhs < rhs
            elif compare == Compare.GREATER_THAN:
                assert not lhs == rhs
                assert lhs != rhs
                assert lhs >= rhs
                assert lhs > rhs
                assert not lhs <= rhs
                assert not lhs < rhs
            else:
                assert not lhs == rhs
                assert lhs != rhs
                assert not lhs >= rhs
                assert not lhs > rhs
                assert not lhs <= rhs
                assert not lhs < rhs
        assert context.flags == 0

    @pytest.mark.parametrize('fmt', all_IEEE_fmts)
    def test_comparisons_inexact(self, fmt, context):
        one = fmt.from_int(1)
        three = fmt.from_int(3)
        third = fmt.divide(one, three)
        assert context.flags == Flags.INEXACT
        context.flags = 0
        frac = Fraction(1, 3)
        if fmt in (IEEEhalf, IEEEdouble, IEEEquad):
            # 0x1.554p-2 0x1.5555555555555p-2 0x1.5555555555555555555555555555p-2
            assert third < frac
            assert -third > -frac
        else:
            # 0x1.555556p-2 for IEEEsingle
            assert third > frac
            assert -third < -frac
        assert context.flags == 0

    @pytest.mark.parametrize('fmt', all_IEEE_fmts)
    def test_comparisons_unimplemented(self, fmt, context):
        one = fmt.from_int(1)
        with pytest.raises(TypeError):
            one <= 'a'
        with pytest.raises(TypeError):
            one < 'a'
        with pytest.raises(TypeError):
            one > 'a'
        with pytest.raises(TypeError):
            one >= 'a'
        assert not one == 'a'
        assert one != 'a'

    @pytest.mark.parametrize('fmt, sign', product(all_IEEE_fmts, (False, True)))
    def test_bool(self, fmt, sign, context):
        assert not bool(fmt.make_zero(sign))
        assert bool(fmt.make_one(sign))
        assert bool(fmt.make_infinity(sign))
        assert bool(fmt.make_nan(sign, False, 0))
        assert bool(fmt.make_nan(sign, True, 0))
        assert context.flags == 0


# Test basic class functions before reading test files
class TestGeneralNonComputationalOps:

    @pytest.mark.parametrize('fmt, sign',
                             product(all_IEEE_fmts,
                                     (False, True)
                             ))
    def test_make_zero(self, fmt, sign):
        value = fmt.make_zero(sign)
        assert value.fmt is fmt
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
        assert not value.is_nan()
        assert not value.is_qnan()
        assert not value.is_snan()
        assert value.is_canonical()
        assert not value.is_finite_non_zero()
        assert value.radix() == 2

    @pytest.mark.parametrize('fmt, sign',
                             product(all_IEEE_fmts,
                                     (False, True)
                             ))
    def test_make_infinity(self, fmt, sign):
        value = fmt.make_infinity(sign)
        assert value.fmt is fmt
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
        assert not value.is_nan()
        assert not value.is_qnan()
        assert not value.is_snan()
        assert value.is_canonical()
        assert not value.is_finite_non_zero()
        assert value.radix() == 2

    @pytest.mark.parametrize('fmt, sign, is_signalling, payload',
                             product(all_IEEE_fmts,
                                     (False, True),
                                     (False, True),
                                     (0, 1, 24),
                             ))
    def test_make_nan(self, fmt, sign, is_signalling, payload):
        value = fmt.make_nan(sign, is_signalling, payload)
        assert value.fmt is fmt
        if payload == 0 and is_signalling:
            payload = 1
        if is_signalling:
            assert value.number_class() == 'sNaN'
        else:
            assert value.number_class() == 'NaN'
        if sign:
            assert value.is_negative()
        else:
            assert not value.is_negative()
        assert not value.is_normal()
        assert not value.is_finite()
        assert not value.is_subnormal()
        assert not value.is_infinite()
        assert value.is_nan()
        assert value.is_qnan() is not is_signalling
        assert value.is_snan() is is_signalling
        assert value.is_canonical()
        assert not value.is_finite_non_zero()
        assert value.radix() == 2
        assert value.as_tuple()[-1] == payload

        with pytest.raises(ValueError):
            fmt.make_nan(sign, is_signalling, -1)
        with pytest.raises(TypeError):
            fmt.make_nan(sign, is_signalling, 1.2)
        with pytest.raises(TypeError):
            fmt.make_nan(sign, is_signalling, 1.2)

    @pytest.mark.parametrize('fmt, sign',
                             product(all_IEEE_fmts, (False, True),
                             ))
    def test_normalize_MSB_set(self, fmt, sign):
        '''Test MSB set with various exponents.'''
        op_tuple = ('test', None)
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
            context = Context(rounding=ROUND_HALF_EVEN)
            value = fmt._normalize(sign, exponent, significand, op_tuple, context)
            assert value.fmt is fmt
            if exponent < fmt.e_min:
                assert context.flags == 0
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
            assert not value.is_nan()
            assert not value.is_qnan()
            assert not value.is_snan()
            assert value.is_canonical()
            assert value.is_finite_non_zero()
            assert value.radix() == 2

    @pytest.mark.parametrize('fmt, sign, exponent',
                             product(all_IEEE_fmts,
                                     (False, True),
                                     (-1, 0, 1, (1 << 200)),
                             ))
    def test_normalize_zero_significand(self, fmt, sign, exponent, context):
        # Test that a zero significand gives a zero regardless of exponent
        value = fmt._normalize(sign, exponent, 0, None, context)
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
    def test_normalize_underflow_to_zero(self, fmt, sign, two_bits, rounding):
        # Test that a value that loses two bits of precision underflows correctly
        context = Context(rounding=rounding)
        op_tuple = ('test', None)
        value = fmt._normalize(sign, fmt.e_min - 2 - (fmt.precision - 1), two_bits,
                               op_tuple, context)
        underflows_to_zero = (rounding in {ROUND_HALF_EVEN, ROUND_HALF_DOWN} and two_bits in (1, 2)
                              or (rounding == ROUND_HALF_UP and two_bits == 1)
                              or (rounding == ROUND_CEILING and sign)
                              or (rounding == ROUND_FLOOR and not sign)
                              or (rounding == ROUND_DOWN))
        assert context.flags == Flags.INEXACT | Flags.UNDERFLOW
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
        op_tuple = ('test', None)
        context = Context(rounding=rounding)
        # First test the exponent that doesn't overflow but that one more would
        exponent = fmt.e_max
        value = fmt._normalize(sign, exponent, 1, op_tuple, context)
        assert context.flags == 0
        assert value.is_normal()
        assert value.sign is sign
        assert value.fmt is fmt

        # Increment the exponent.  Overflow now depends on rounding mode
        exponent += 1
        value = fmt._normalize(sign, exponent, 1, op_tuple, context)
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
    def test_normalie_overflows_significand(self, fmt, sign, e_selector, two_bits, rounding):
        op_tuple = ('test', None)
        # Test cases where rounding away causes significand to overflow
        context = Context(rounding=rounding)
        # Minimimum good, maximum good, overflows to infinity
        exponent = [fmt.e_min - 2, fmt.e_max - 3, fmt.e_max - 2][e_selector]
        # two extra bits in the significand
        significand = two_bits + (fmt.max_significand << 2)
        value = fmt._normalize(sign, exponent - (fmt.precision - 1), significand,
                               op_tuple, context)
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
    def test_normalize_subnormal_to_normal(self, fmt, sign, two_bits, rounding):
        op_tuple = ('test', None)
        # Test cases where rounding away causes a subnormal to normalize
        context = Context(rounding=rounding)
        # an extra bit in the significand with two LSBs varying
        significand = two_bits + ((fmt.max_significand >> 1) << 2)
        value = fmt._normalize(sign, fmt.e_min - 2 - (fmt.precision - 1), significand,
                               op_tuple, context)
        rounds_away = (two_bits and
                       ((rounding == ROUND_HALF_EVEN and two_bits in (2, 3))
                        or rounding == ROUND_UP
                        or (rounding == ROUND_HALF_DOWN and two_bits == 3)
                        or (rounding == ROUND_HALF_UP and two_bits in (2, 3))
                        or (rounding == ROUND_CEILING and not sign)
                        or (rounding == ROUND_FLOOR and sign)))
        if rounds_away:
            assert context.flags == Flags.INEXACT
            assert value.is_normal()
            assert value.significand == fmt.int_bit
            assert value.e_biased == 1
        else:
            if two_bits == 0:
                assert context.flags == 0
            else:
                assert context.flags == Flags.INEXACT | Flags.UNDERFLOW
            assert value.is_subnormal()
            assert value.significand == fmt.int_bit - 1
            assert value.e_biased == 1
        assert value.sign is sign
        assert value.fmt is fmt


class TestUnaryOps:

    def test_from_string_overflow(self, quiet_context):
        fmt = BinaryFormat.from_triple(24, 66, -66)
        fmt.from_string('9' * 21, quiet_context)

    def test_from_string_exact_underflow(self, quiet_context):
        quiet_context.set_handler(UnderflowExact, HandlerKind.RAISE)
        with pytest.raises(UnderflowExact):
            IEEEhalf.from_string('.000030517578125')

    @pytest.mark.parametrize('line', read_lines('from_string.txt'))
    def test_from_string(self, line):
        parts = line.split()
        if len(parts) == 1:
            hex_str, = parts
            context = Context()
            context.set_handler(Invalid, HandlerKind.RAISE)
            with pytest.raises(InvalidFromString) as e:
                IEEEsingle.from_string(hex_str, context)
            assert e.value.op_tuple == (OP_FROM_STRING, hex_str)
        elif len(parts) in (5, 7):
            fmt, context, test_str, status = parts[:4]
            fmt = format_codes[fmt]
            context = rounding_string_to_context(context)
            result = fmt.from_string(test_str, context)
            assert result.fmt is fmt
            status = status_codes[status]

            if len(parts) == 5:
                answer = parts[-1]
                answer = from_string(fmt, answer)
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
        ('round', (-128, 127), (0, 255), (-(1 << 63), (1 << 63) - 1), (0, (1 << 64) - 1)),
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
                context = rounding_string_to_context(rounding)
                result = value.round_to_integral_exact(context)
                assert floats_equal(result, answer)
                assert context.flags == status
            else:
                context = Context()
                result = value.round_to_integral(rounding_codes[rounding], context)
                assert floats_equal(result, answer)
                assert context.flags == status & ~Flags.INEXACT
        else:
            min_int, max_int = kind
            context = Context()
            if answer.is_nan():
                status = Flags.INVALID
                answer = 0
            elif answer.is_infinite():
                status = Flags.INVALID
                answer = min_int if answer.sign else max_int
            else:
                answer = int(answer_str)
                if answer < min_int:
                    answer, status = min_int, Flags.INVALID
                elif answer > max_int:
                    answer, status = max_int, Flags.INVALID
            if exact:
                result = value.convert_to_integer_exact(min_int, max_int,
                                                        rounding_codes[rounding], context)
                assert isinstance(result, int)
                assert result == answer
                assert context.flags == status
            else:
                result = value.convert_to_integer(min_int, max_int, rounding_codes[rounding],
                                                  context)
                assert isinstance(result, int)
                assert result == answer
                assert context.flags == status & ~Flags.INEXACT


    @pytest.mark.parametrize('line', read_lines('convert.txt'))
    def test_convert(self, line):
        parts = line.split()
        if len(parts) != 6:
            assert False, f'bad line: {line}'
        src_fmt, context, src_value, dst_fmt, status, answer = parts
        src_fmt = format_codes[src_fmt]
        context = rounding_string_to_context(context)
        src_value = from_string(src_fmt, src_value)

        dst_fmt = format_codes[dst_fmt]
        status = status_codes[status]
        answer = from_string(dst_fmt, answer)

        result = dst_fmt.convert(src_value, context)
        assert result.fmt is dst_fmt
        assert floats_equal(result, answer)
        assert context.flags == status

    @pytest.mark.parametrize('line', read_lines('from_int.txt'))
    def test_from_int(self, line):
        parts = line.split()
        if len(parts) != 5:
            assert False, f'bad line: {line}'
        dst_fmt, context, src_value, status, answer = parts

        dst_fmt = format_codes[dst_fmt]
        context = rounding_string_to_context(context)
        value = int(src_value)
        assert str(value) == src_value
        status = status_codes[status]
        answer = dst_fmt.from_string(answer, context)
        assert context.flags == 0

        result = dst_fmt.from_int(value, context)
        assert result.fmt is dst_fmt
        assert floats_equal(result, answer)
        assert context.flags == status

    @pytest.mark.parametrize('line', read_lines('to_hex.txt'))
    def test_to_hex(self, line):
        parts = line.split()
        if len(parts) != 7:
            assert False, f'bad line: {line}'
        text_format, context, src_fmt, in_str, dst_fmt, status, answer = parts
        text_format = to_text_format(text_format)
        context = rounding_string_to_context(context)
        in_value = from_string(format_codes[src_fmt], in_str)
        dst_fmt = format_codes[dst_fmt]
        status = status_codes[status]

        result = dst_fmt.to_string(in_value, text_format, context)
        assert result == answer
        assert context.flags == status

    @pytest.mark.parametrize('force_point, answer', ((True, '0x1.0p+0'), (False, '0x1p+0')))
    def test_to_hex_force_point(self, force_point, answer):
        text_format = TextFormat(force_exp_sign=True, rstrip_zeroes=True, force_point=force_point)
        d = IEEEdouble.from_int(1)
        assert d.to_string(text_format) == answer

    @pytest.mark.parametrize('line', read_lines('to_decimal.txt'))
    def test_to_decimal(self, line):
        parts = line.split()
        if len(parts) != 6:
            assert False, f'bad line: {line}'
        context, fmt, precision, in_str, status, answer = parts
        fmt = format_codes[fmt]
        context = rounding_string_to_context(context)
        precision = int(precision)
        value = from_string(fmt, in_str)
        status = status_codes[status]
        text_format = TextFormat(exp_digits=-2, force_exp_sign=True)
        # Abuse meaning of rounding field for NaNs in this test only
        if value.is_nan() and context.rounding != ROUND_HALF_EVEN:
            text_format.snan = ''
        result = value.to_decimal_string(precision, text_format, context)
        assert result == answer
        assert context.flags == status

        if precision <= 0 and value.is_finite():
            # Confirm the round-trip: reading in the decimal value gives the same as the
            # hex value
            context.flags = 0
            dec_value = fmt.from_string(answer, context)
            assert floats_equal(dec_value, value)
            assert context.flags == status

            # Confirm Python prints the same.
            if fmt is IEEEdouble and precision == 0:
                if '0x' in in_str:
                    value = float.fromhex(in_str)
                else:
                    value = float(in_str)
                py_str = str(value)
                if py_str.endswith('.0'):
                    py_str = py_str[:-2]
                assert py_str == answer

    @pytest.mark.parametrize('line', read_lines('scaleb.txt'))
    def test_scaleb(self, line):
        parts = line.split()
        if len(parts) != 6:
            assert False, f'bad line: {line}'
        fmt, context, in_str, N_str, status, answer = parts
        fmt = format_codes[fmt]
        context = rounding_string_to_context(context)
        in_value = from_string(fmt, in_str)
        answer = from_string(fmt, answer)
        N = int(N_str)
        assert str(N) == N_str
        status = status_codes[status]

        result = in_value.scaleb(N, context)
        assert result.fmt is fmt
        assert floats_equal(result, answer)
        assert context.flags == status

    def test_scaleb_type(self):
        one = IEEEdouble.make_one(False)
        with pytest.raises(TypeError):
            one.scaleb(0.0)

    @pytest.mark.parametrize('fmt', all_IEEE_fmts)
    def test_logb_specials(self, fmt):
        # Test all 3 values are different
        values = {fmt.logb_zero, fmt.logb_inf, fmt.logb_nan}
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
        in_value = from_string(fmt, in_str)
        answer = from_string(fmt, answer)
        status = status_codes[status]

        context = Context()
        result = in_value.logb(context)
        assert result.fmt is fmt
        assert floats_equal(result, answer)
        assert context.flags == status

        # Now test logb_integral
        context.flags = 0
        result_integral = in_value.logb_integral(context)
        assert result.fmt is fmt
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
                assert result_integral == fmt.logb_nan
            assert context.flags == Flags.INVALID

    @pytest.mark.parametrize('line', read_lines('next_up.txt'))
    def test_next(self, line):
        # Tests next_up and next_down
        parts = line.split()
        if len(parts) != 4:
            assert False, f'bad line: {line}'
        context = Context()
        fmt, in_str, status, answer = parts
        fmt = format_codes[fmt]
        in_value = from_string(fmt, in_str)
        answer = from_string(fmt, answer)
        status = status_codes[status]

        result = in_value.next_up(context)
        assert result.fmt is fmt
        assert floats_equal(result, answer)
        assert context.flags == status

        # Now for next_down
        context.flags = 0
        in_value = in_value.negate_quiet()
        answer = answer.negate_quiet()
        result = in_value.next_down(context)
        assert result.fmt is fmt
        assert floats_equal(result, answer)
        assert context.flags == status

    @pytest.mark.parametrize('line', read_lines('payload.txt'))
    def test_payload(self, line):
        parts = line.split()
        if len(parts) != 3:
            assert False, f'bad line: {line}'
        fmt, value, answer = parts
        fmt = format_codes[fmt]
        value = from_string(fmt, value)
        answer = from_string(fmt, answer)

        get_context().flags = 0
        result = value.payload()
        assert result.fmt is fmt
        assert floats_equal(result, answer)
        assert get_context().flags == 0

    @pytest.mark.parametrize('line', read_lines('set_payload.txt'))
    def test_set_payload(self, line):
        parts = line.split()
        if len(parts) != 3:
            assert False, f'bad line: {line}'
        fmt, value, answer = parts
        fmt = format_codes[fmt]
        value = from_string(fmt, value)
        answer = from_string(fmt, answer)

        get_context().flags = 0
        result = value.set_payload()
        assert result.fmt is fmt
        assert floats_equal(result, answer)
        assert get_context().flags == 0

    @pytest.mark.parametrize('line', read_lines('set_payload_signalling.txt'))
    def test_set_payload_signalling(self, line):
        parts = line.split()
        if len(parts) != 3:
            assert False, f'bad line: {line}'
        fmt, value, answer = parts
        fmt = format_codes[fmt]
        value = from_string(fmt, value)
        answer = from_string(fmt, answer)

        result = value.set_payload_signalling()
        assert result.fmt is fmt
        assert floats_equal(result, answer)
        assert get_context().flags == 0

    @pytest.mark.parametrize('line', read_lines('sqrt.txt'))
    def test_sqrt(self, line):
        # Tests next_up and next_down
        parts = line.split()
        if len(parts) != 6:
            assert False, f'bad line: {line}'
        context, fmt, value, dst_fmt, status, answer = parts
        fmt = format_codes[fmt]
        dst_fmt = format_codes[dst_fmt]
        context = rounding_string_to_context(context)
        value = from_string(fmt, value)
        status = status_codes[status]
        answer = from_string(dst_fmt, answer)

        result = dst_fmt.sqrt(value, context)
        assert result.fmt is dst_fmt
        assert floats_equal(result, answer)
        assert context.flags == status

    @pytest.mark.parametrize('endianness', ('big', 'little', None))
    def test_pack_unpack_round_trip(self, endianness):
        for value in range(0, 65536):
            binary = value.to_bytes(2, endianness or native)
            parts = IEEEhalf.unpack(binary, endianness)
            packed_value = IEEEhalf.pack(*parts, endianness)
            assert binary == packed_value

    def test_x87_pseudos(self):
        # 3fff9180000000000000 is the canonical representation of 0x1.23p0.  Clear its
        # integer bit (making it an unnormal) and check
        for hex_str in ('3fff9180000000000000', '3fff1180000000000000'):
            value = x87extended.unpack_value(bytes.fromhex(hex_str), 'big')
            assert str(value) == '0x1.2300000000000000p+0'

        # 7fffc000000000000000 is the canonical representation of a NaN with integer bit
        # set.  Test clearing it (a pseudo-NaN) gives the right answer.
        for hex_str in ('7fffc000000000000000', '7fff4000000000000000'):
            value = x87extended.unpack_value(bytes.fromhex(hex_str), 'big')
            assert value.is_nan()
            assert not value.is_snan()
            assert value.nan_payload() == 0

        # 7fff8000000000000000 is the canonical representation of +Inf with integer bit
        # set.  Test clearing it gives the right answer.
        for hex_str in ('7fff8000000000000000', '7fff0000000000000000'):
            value = x87extended.unpack_value(bytes.fromhex(hex_str), 'big')
            assert str(value) == 'Infinity'

        # 00000000a03000000000 is the canonical representation of 0x1.85p-16400, or
        # 0x0.0000614p-16382, a subnormal with integer bit clear.  Test setting it gives
        # the right answer.
        answer = x87extended.from_string('0x1.85p-16400')
        for hex_str in ('0000000030a000000000', '0000800030a000000000'):
            value = x87extended.unpack_value(bytes.fromhex(hex_str), 'big')
            assert floats_equal(value, answer)

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
        assert floats_equal(value, value.fmt.unpack_value(result, 'big'))
        # Test little-endian unpacking
        assert floats_equal(value, value.fmt.unpack_value(le_packing, 'little'))
        assert floats_equal(value.fmt.unpack_value(le_packing, native),
                            value.fmt.unpack_value(le_packing, None))


def min_max_op(line, operation):
    parts = line.split()
    if len(parts) != 5:
        assert False, f'bad line: {line}'
    fmt, lhs, rhs, status, answer = parts
    fmt = format_codes[fmt]
    fmt = random.choice((fmt, IEEEquad))
    lhs = from_string(fmt, lhs)
    rhs = from_string(fmt, rhs)
    answer = from_string(fmt, answer)
    status = status_codes[status]

    operation = getattr(lhs, operation)
    context = Context()
    result = operation(rhs, context)
    assert result.fmt is fmt
    assert floats_equal(result, answer)
    assert context.flags == status


def binary_operation(line, operation):
    parts = line.split()
    if len(parts) != 8:
        assert False, f'bad line: {line}'
    context, lhs_fmt, lhs, rhs_fmt, rhs, dst_fmt, status, answer = parts
    context = rounding_string_to_context(context)
    dst_fmt = format_codes[dst_fmt]

    lhs = from_string(format_codes[lhs_fmt], lhs)
    rhs = from_string(format_codes[rhs_fmt], rhs)
    answer = from_string(dst_fmt, answer)
    status = status_codes[status]

    operation = getattr(dst_fmt, operation)
    result = operation(lhs, rhs, context)
    assert result.fmt is dst_fmt
    assert floats_equal(result, answer)
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
        # FIXME: subnormal tests
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

    @pytest.mark.parametrize('line', read_lines('max.txt'))
    def test_max(self, line):
        min_max_op(line, 'max')

    @pytest.mark.parametrize('line', read_lines('max_num.txt'))
    def test_max_num(self, line):
        min_max_op(line, 'max_num')

    @pytest.mark.parametrize('line', read_lines('max_mag.txt'))
    def test_max_mag(self, line):
        min_max_op(line, 'max_mag')

    @pytest.mark.parametrize('line', read_lines('max_mag_num.txt'))
    def test_max_mag_num(self, line):
        min_max_op(line, 'max_mag_num')

    @pytest.mark.parametrize('line', read_lines('min.txt'))
    def test_min(self, line):
        min_max_op(line, 'min')

    @pytest.mark.parametrize('line', read_lines('min_num.txt'))
    def test_min_num(self, line):
        min_max_op(line, 'min_num')

    @pytest.mark.parametrize('line', read_lines('min_mag.txt'))
    def test_min_mag(self, line):
        min_max_op(line, 'min_mag')

    @pytest.mark.parametrize('line', read_lines('min_mag_num.txt'))
    def test_min_mag_num(self, line):
        min_max_op(line, 'min_mag_num')

    @pytest.mark.parametrize('operation', ('remainder', 'fmod', 'mod', 'floordiv'))
    def test_diff_formats(self, operation, context):
        lhs = IEEEsingle.make_one(False)
        rhs = IEEEdouble.make_one(False)
        with pytest.raises(ValueError):
            getattr(lhs, operation)(rhs, context)

    @pytest.mark.parametrize('line', read_lines('remainder.txt'))
    def test_remainder(self, line, quiet_context):
        parts = line.split()
        if len(parts) != 5:
            assert False, f'bad line: {line}'
        fmt, lhs, rhs, status, answer = parts
        context = quiet_context

        fmt = format_codes[fmt]
        lhs = from_string(fmt, lhs)
        rhs = from_string(fmt, rhs)
        status = status_codes[status]
        answer = from_string(fmt, answer)

        context.set_handler(Underflow, HandlerKind.RAISE)
        if answer.is_subnormal():
            with pytest.raises(UnderflowExact) as e:
                lhs.remainder(rhs, context)
            result = e.value.default_result
        else:
            result = lhs.remainder(rhs, context)
        assert result.fmt is fmt
        assert floats_equal(result, answer)
        assert context.flags == status

    @pytest.mark.parametrize('line', read_lines('fmod.txt'))
    def test_fmod(self, line):
        parts = line.split()
        if len(parts) != 5:
            assert False, f'bad line: {line}'
        fmt, lhs, rhs, status, answer = parts

        fmt = format_codes[fmt]
        lhs = from_string(fmt, lhs)
        rhs = from_string(fmt, rhs)
        status = status_codes[status]
        answer = from_string(fmt, answer)

        context = Context()
        result = lhs.fmod(rhs, context)
        assert result.fmt is fmt
        assert floats_equal(result, answer)
        assert context.flags == status

    @pytest.mark.parametrize('line', read_lines('mod.txt'))
    def test_mod(self, line, quiet_context):
        parts = line.split()
        if len(parts) != 7:
            assert False, f'bad line: {line}'
        fmt, lhs, rhs, _, _, status, answer = parts
        context = quiet_context

        fmt = format_codes[fmt]
        lhs = from_string(fmt, lhs)
        rhs = from_string(fmt, rhs)
        status = status_codes[status]
        answer = from_string(fmt, answer)

        result = lhs.mod(rhs, context)
        assert result.fmt is fmt
        assert floats_equal(result, answer)
        assert context.flags == status

    @pytest.mark.parametrize('line', read_lines('mod.txt'))
    def test_floordiv(self, line, quiet_context):
        parts = line.split()
        if len(parts) != 7:
            assert False, f'bad line: {line}'
        fmt, lhs, rhs, status, answer, _, _ = parts
        context = quiet_context

        fmt = format_codes[fmt]
        lhs = from_string(fmt, lhs)
        rhs = from_string(fmt, rhs)
        status = status_codes[status]
        answer = from_string(fmt, answer)

        result = lhs.floordiv(rhs, context)
        assert result.fmt is fmt
        assert floats_equal(result, answer)
        assert context.flags == status

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
        context = Context()
        result = lhs.compare(rhs, context)
        assert result == answer
        assert context.flags == status

        # Compare singalling
        context = Context()
        result = lhs.compare_signal(rhs, context)
        op_status = Flags.INVALID if lhs.is_nan() or rhs.is_nan() else 0
        assert result == answer
        assert context.flags == op_status

        # Now check all the other comparison operations
        for op, true_set in comparison_ops.items():
            # Test the quiet form:
            op_name = f'compare_{op}'
            op_result = answer_code in true_set
            op_status = Flags.INVALID if lhs.is_snan() or rhs.is_snan() else 0
            context = Context()
            result = getattr(lhs, op_name)(rhs, context)
            assert result == op_result
            assert context.flags == op_status

            # Test the singalling form:
            if op not in {'un', 'or'}:
                op_name = f'compare_{op}_signal'
                op_status = Flags.INVALID if lhs.is_nan() or rhs.is_nan() else 0
                context = Context()
                result = getattr(lhs, op_name)(rhs, context)
                assert result == op_result
                assert context.flags == op_status

    @pytest.mark.parametrize('line', read_lines('compare_total.txt'))
    def test_compare_total(self, line):
        parts = line.split()
        if len(parts) != 4:
            assert False, f'bad line: {line}'
        fmt, lhs, rhs, answer_code = parts

        lhs = from_string(format_codes[fmt], lhs)
        rhs = from_string(format_codes[fmt], rhs)
        answer = boolean_codes[answer_code]

        assert lhs.compare_total(rhs) is answer
        lhs_abs = lhs.abs_quiet()
        rhs_abs = rhs.abs_quiet()

        assert lhs.compare_total_mag(rhs) is lhs_abs.compare_total(rhs_abs)

    def test_compare_formats(self):
        assert IEEEsingle.make_zero(True).compare_total(IEEEdouble.make_zero(False))
        assert not IEEEsingle.make_zero(False).compare_total(IEEEquad.make_zero(True))


class TestFMA:

    @pytest.mark.parametrize('line', read_lines('fma.txt'))
    def test_fma(self, line):
        # FIXME: subnormal tests
        parts = line.split()
        if len(parts) != 10:
            assert False, f'bad line: {line}'
        context, lhs_fmt, lhs, rhs_fmt, rhs, add_fmt, addend, dst_fmt, status, answer = parts
        context = rounding_string_to_context(context)
        dst_fmt = format_codes[dst_fmt]

        lhs = from_string(format_codes[lhs_fmt], lhs)
        rhs = from_string(format_codes[rhs_fmt], rhs)
        addend = from_string(format_codes[add_fmt], addend)
        answer = from_string(dst_fmt, answer)
        status = status_codes[status]

        result = dst_fmt.fma(lhs, rhs, addend, context)
        assert result.fmt is dst_fmt
        assert floats_equal(result, answer)
        assert context.flags == status

    def test_fma_op_tuple(self):
        # Test the op_tuple is an OP_FMA one and not an OP_ADD one
        context = Context()
        context.set_handler(Inexact, HandlerKind.RAISE)
        zero = IEEEhalf.make_zero(False)
        epsilon = IEEEdouble.make_smallest_normal(False)
        with pytest.raises(Inexact) as e:
            IEEEhalf.fma(zero, zero, epsilon, context)
        assert e.value.op_tuple == (OP_FMA, zero, zero, epsilon)

    def test_fma_sNaNs(self, context):
        snan = IEEEhalf.make_nan(False, True, 1)
        epsilon = IEEEhalf.make_smallest_normal(False)
        with pytest.raises(Invalid) as e:
            IEEEhalf.fma(epsilon, epsilon, snan, context)
        assert e.value.op_tuple == (OP_FMA, epsilon, epsilon, snan)
        assert e.value.default_result.is_qnan()
        # Not inexact
        assert context.flags == Flags.INVALID

        context.flags = 0
        with pytest.raises(Invalid) as e:
            IEEEhalf.fma(snan, epsilon, epsilon, context)
        assert e.value.op_tuple == (OP_FMA, snan, epsilon, epsilon)
        assert context.flags == Flags.INVALID
        assert e.value.default_result.is_qnan()

    def test_fma_invalid(self, context):
        def handler(exception, context):
            assert exception.op_tuple == (OP_FMA, zero, inf, one)
            assert exception.default_result.is_nan()
            return one

        context.set_handler(InvalidFMA, HandlerKind.SUBSTITUTE_VALUE, handler)
        zero = IEEEhalf.make_zero(False)
        one = IEEEhalf.make_one(False)
        inf = IEEEhalf.make_infinity(True)
        result = IEEEhalf.fma(zero, inf, one, context)
        # Result is -1; no addition happens
        assert context.flags == Flags.INVALID
        assert result is one

    def test_fma_invalid_xor(self, context):
        def handler(exception, context):
            assert False

        context.set_handler(InvalidFMA, HandlerKind.SUBSTITUTE_VALUE_XOR, handler)
        zero = IEEEhalf.make_zero(False)
        one = IEEEhalf.make_one(False)
        inf = IEEEhalf.make_infinity(True)
        result = IEEEhalf.fma(zero, inf, one, context)
        assert context.flags == Flags.INVALID
        assert result.is_nan()
