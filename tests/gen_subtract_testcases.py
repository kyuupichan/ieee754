import os
import random

from ieee754 import *


def random_significand(bits):
    size = (bits + 7) // 8
    value = int.from_bytes(os.urandom(size), 'big')
    value >>= size * 8 - bits
    return value | (1 << (bits - 1))


def large_significand(fmt):
    significand = random_significand(fmt.precision)
    exponent = random.randrange(fmt.e_min, fmt.e_max + 1)
    sign = random.choice((False, True))
    return Binary(fmt, sign, exponent + fmt.e_bias, significand)


def small_significand(fmt):
    significand = random_significand(min(11, fmt.precision))
    significand <<= fmt.precision - significand.bit_length()
    exponent = random.randrange(fmt.e_min, fmt.e_max + 1)
    sign = random.choice((False, True))
    return Binary(fmt, sign, exponent + fmt.e_bias, significand)


def subnormal(fmt):
    shift = random.randrange(1, fmt.precision)
    significand = random_significand(fmt.precision) >> shift
    exponent = fmt.e_min
    sign = random.choice((False, True))
    return Binary(fmt, sign, exponent + fmt.e_bias, significand)


fmts = {
    IEEEhalf: 'H',
    IEEEsingle: 'S',
    IEEEdouble: 'D',
    IEEEquad: 'Q',
}
values = [large_significand, small_significand, subnormal]
roundings = {
    ROUND_CEILING: 'C',
    ROUND_FLOOR: 'F',
    ROUND_DOWN: 'D',
    ROUND_UP: 'U',
    ROUND_HALF_EVEN: 'E',
    ROUND_HALF_DOWN: 'd',
    ROUND_HALF_UP: 'u',
}
flags = {
    Flags.INEXACT: 'I',
    Flags.OVERFLOW | Flags.INEXACT: 'VI',
    0: 'K',
    Flags.UNDERFLOW | Flags.INEXACT: 'U',
}

big_fmt = BinaryFormat.from_triple(50000, 50000, -50000)
roundings_lst = list(roundings)

for lhs_fmt in fmts:
    for lhs_value in values:
        lhs = lhs_value(lhs_fmt)
        for rhs_fmt in fmts:
            for rhs_value in values:
                rhs = rhs_value(rhs_fmt)
                for result_fmt in fmts:
                    rounding = random.choice(roundings_lst)
                    context = Context(rounding=rounding)
                    result = big_fmt.subtract(lhs, rhs, context)
                    assert context.flags == 0
                    result = result_fmt.convert(result, context)
                    print(roundings[rounding], fmts[lhs_fmt], lhs, fmts[rhs_fmt], rhs,
                          fmts[result_fmt], flags[context.flags], result)
