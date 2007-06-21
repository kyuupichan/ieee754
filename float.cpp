/*
   Copyright 2005-2007 Neil Booth.

   See the file "COPYING" for information about the copyright
   and warranty status of this software.
*/

#include <cassert>
#include "float.h"

using namespace llvm;

#define convolve(lhs, rhs) ((lhs) * 4 + (rhs))

/* Assumed in hexadecimal significand parsing.  */
compile_time_assert (t_integer_part_width % 4 == 0);

namespace llvm {

  /* Represents floating point arithmetic semantics.  */
  struct flt_semantics
  {
    /* The largest E such that 2^E is representable; this matches the
       definition of IEEE 754.  */
    exponent_t max_exponent;

    /* The smallest E such that 2^E is a normalized number; this
       matches the definition of IEEE 754.  */
    exponent_t min_exponent;

    /* Number of bits in the significand.  This includes the integer
       bit.  */
    unsigned char precision;
  };

  struct decimal_number
  {
    t_integer_part *parts;
    unsigned int part_count;
    int exponent;
  };

  const flt_semantics t_float::ieee_single = { 127, -126, 24 };
  const flt_semantics t_float::ieee_double = { 1023, -1022, 53 };
  const flt_semantics t_float::ieee_quad = { 16383, -16382, 113 };
  const flt_semantics t_float::x87_double_extended = { 16383, -16382, 64 };
}

/* Put a bunch of private, handy routines in an anonymous namespace.  */
namespace {

  inline unsigned int
  part_count_for_bits (unsigned int bits)
  {
    return ((bits) + t_integer_part_width - 1) / t_integer_part_width;
  }

  unsigned int
  digit_value (unsigned int c)
  {
    unsigned int r;

    r = c - '0';
    if (r <= 9)
      return r;

    return -1U;
  }

  unsigned int
  hex_digit_value (unsigned int c)
  {
    unsigned int r;

    r = c - '0';
    if (r <= 9)
      return r;

    r = c - 'A';
    if (r <= 5)
      return r + 10;

    r = c - 'a';
    if (r <= 5)
      return r + 10;

    return -1U;
  }

  /* This is ugly and needs cleaning up, but I don't immediately see
     how whilst remaining safe.  */
  static int
  total_exponent (const char *p, int exponent_adjustment)
  {
    t_integer_part unsigned_exponent;
    bool negative, overflow;
    long exponent;

    /* Move past the exponent letter and sign to the digits.  */
    p++;
    negative = *p == '-';
    if (*p == '-' || *p == '+')
      p++;

    unsigned_exponent = 0;
    overflow = false;
    for (;;)
      {
	unsigned int value;

	value = digit_value (*p);
	if (value == -1U)
	  break;

	p++;
	unsigned_exponent = unsigned_exponent * 10 + value;
	if (unsigned_exponent > 65535)
	  overflow = true;
      }

    if (exponent_adjustment > 65535 || exponent_adjustment < -65536)
      overflow = true;

    if (!overflow)
      {
	exponent = unsigned_exponent;
	if (negative)
	  exponent = -exponent;
	exponent += exponent_adjustment;
	if (exponent > 65535 || exponent < -65536)
	  overflow = true;
      }

    if (overflow)
      exponent = negative ? -65536: 65535;

    return exponent;
  }

  const char *
  skip_leading_zeroes_and_any_dot (const char *p, const char **dot)
  {
    *dot = 0;
    while (*p == '0')
      p++;

    if (*p == '.')
    {
      *dot = p++;
      while (*p == '0')
	p++;
    }

    return p;
  }

  /* Reads the full significand and exponent of a decimal number,
     converting them to binary.  Quite efficient.  If the significand
     is zero, return TRUE without filling NUMBER.  Otherwise fill out
     NUMBER with allocated memory and return FALSE.  */
  static bool
  read_decimal_number (const char *p, decimal_number *number)
  {
    unsigned int msb, part_count, parts_used;
    const char *dot, *first_significant_digit;
    t_integer_part *parts, val, max;
    int exponent;

    /* Skip leading zeroes and any decimal point.  */
    p = skip_leading_zeroes_and_any_dot (p, &dot);
    first_significant_digit = p;

    /* Do single-part arithmetic for as long as we can.  */
    max = (~ (t_integer_part) 0 - 9) / 10;
    val = 0;

    while (val <= max)
      {
	unsigned int value;

	if (*p == '.')
	  {
	    assert (dot == 0);
	    dot = p++;
	    continue;
	  }

	value = digit_value (*p);
	if (value == -1U)
	  break;

	val = val * 10 + value;
      }

    if (p == first_significant_digit)
      return true;

    /* Allocate space for 2 integer parts initially.  This will be
       almost always be enough.  */
    part_count = 2;
    parts = new t_integer_part[part_count];
    parts_used = 1;
    APInt::tc_set (parts, val, part_count);

    /* Now repeatedly do single-part arithmetic for as long as we can,
       before having to do a long multiplication.  */
    for (;;)
      {
	t_integer_part multiplier;
	unsigned int value;

	value = 0;
	val = 0;
	multiplier = 1;

	while (multiplier <= max)
	  {
	    if (*p == '.')
	      {
		assert (dot == 0);
		dot = p++;
		continue;
	      }

	    value = digit_value (*p);
	    if (value == -1U)
	      break;

	    multiplier *= 10;
	    val = val * 10 + value;
	  }

	APInt::tc_multiply_part (parts, parts, multiplier, val,
				 parts_used, parts_used + 1, false);

	if (value == -1U)
	  break;

	/* Note this is a conservative estimate but very likely
	   correct.  We calculate the correct value at the end.  */
	parts_used++;

	/* Allocate more space if necessary.  */
	if (parts_used == part_count)
	  {
	    t_integer_part *tmp;

	    part_count *= 2;
	    tmp = new t_integer_part[part_count];
	    APInt::tc_set (tmp, 0, part_count);
	    APInt::tc_assign (tmp, parts, parts_used);
	    delete [] parts;
	    parts = tmp;
	  }
      }

    /* Calculate the exponent adjustment implicit in the number of
       significant digits.  */
    if (!dot)
      dot = p;

    exponent = dot - first_significant_digit;
    if (exponent < 0)
      exponent++;
    if (*p == 'e' || *p == 'E')
      exponent = total_exponent (p, exponent);

    /* Calculate exactly how many parts we actually used.  Shift the
       significand left so the MSB is in the most significant bit.
       This helps our caller.  */
    msb = APInt::tc_msb (parts, parts_used);
    parts_used = msb + (t_integer_part_width - 1) / t_integer_part_width;

    if (msb %= t_integer_part_width)
      {
	msb = t_integer_part_width - msb;
	exponent -= msb;
	APInt::tc_shift_left (parts, parts_used, msb);
      }

    number->parts = parts;
    number->part_count = parts_used;
    number->exponent = exponent;

    return false;
  }

  /* Return the trailing fraction of a hexadecimal number.
     DIGIT_VALUE is the first hex digit of the fraction, P points to
     the next digit.  */
  e_lost_fraction
  trailing_hexadecimal_fraction (const char *p, unsigned int digit_value)
  {
    unsigned int hex_digit;

    /* If the first trailing digit isn't 0 or 8 we can work out the
       fraction immediately.  */
    if (digit_value > 8)
      return lf_more_than_half;
    else if (digit_value < 8 && digit_value > 0)
      return lf_less_than_half;

    /* Otherwise we need to find the first non-zero digit.  */
    while (*p == '0')
      p++;

    hex_digit = hex_digit_value (*p);

    /* If we ran off the end it is exactly zero or one-half, otherwise
       a little more.  */
    if (hex_digit == -1U)
      return digit_value == 0 ? lf_exactly_zero: lf_exactly_half;
    else
      return digit_value == 0 ? lf_less_than_half: lf_more_than_half;
  }

  /* Return the fraction lost were a bignum truncated.  */
  e_lost_fraction
  lost_fraction_through_truncation (t_integer_part *parts,
				    unsigned int part_count,
				    unsigned int bits)
  {
    unsigned int lsb;

    /* See if we would lose precision.  Fast-path two cases that would
       fail the generic logic.  */
    if (bits == 0 || (lsb = APInt::tc_lsb (parts, part_count)) == 0)
      return lf_exactly_zero;

    if (bits < lsb)
      return lf_exactly_zero;
    if (bits == lsb)
      return lf_exactly_half;
    if (bits <= part_count * t_integer_part_width
	&& APInt::tc_extract_bit (parts, bits))
      return lf_more_than_half;

    return lf_less_than_half;
  }

  /* Shift DST right BITS bits noting lost fraction.  */
  e_lost_fraction
  shift_right (t_integer_part *dst, unsigned int parts, unsigned int bits)
  {
    e_lost_fraction lost_fraction;

    lost_fraction = lost_fraction_through_truncation (dst, parts, bits);

    APInt::tc_shift_right (dst, parts, bits);

    return lost_fraction;
  }
}

/* Constructors.  */
void
t_float::initialize (const flt_semantics *our_semantics)
{
  unsigned int count;

  semantics = our_semantics;
  count = part_count ();
  if (count > 1)
    significand.parts = new t_integer_part[count];
}

void
t_float::free_significand ()
{
  if (part_count () > 1)
    delete [] significand.parts;
}

void
t_float::assign (const t_float &rhs)
{
  assert (semantics == rhs.semantics);

  sign = rhs.sign;
  category = rhs.category;
  exponent = rhs.exponent;
  if (category == fc_normal)
    copy_significand (rhs);
}

void
t_float::copy_significand (const t_float &rhs)
{
  assert (category == fc_normal);
  assert (rhs.part_count () >= part_count ());

  APInt::tc_assign (sig_parts_array(), rhs.sig_parts_array(),
		    part_count ());
}

t_float &
t_float::operator= (const t_float &rhs)
{
  if (this != &rhs)
    {
      if (semantics != rhs.semantics)
	{
	  free_significand ();
	  initialize (rhs.semantics);
	}
      assign (rhs);
    }

  return *this;
}

t_float::t_float (const flt_semantics &our_semantics, t_integer_part value)
{
  initialize (&our_semantics);
  sign = 0;
  zero_significand ();
  exponent = our_semantics.precision - 1;
  sig_parts_array ()[0] = value;
  normalize (frm_to_nearest, lf_exactly_zero);
}

t_float::t_float (const flt_semantics &our_semantics,
		  e_category our_category, bool negative)
{
  initialize (&our_semantics);
  category = our_category;
  sign = negative;
  if (category == fc_normal)
    category = fc_zero;
}

t_float::t_float (const flt_semantics &our_semantics, const char *text)
{
  initialize (&our_semantics);
  convert_from_string (text, frm_to_nearest);
}

t_float::t_float (const t_float &rhs)
{
  initialize (rhs.semantics);
  assign (rhs);
}

t_float::~t_float ()
{
  free_significand ();
}

unsigned int
t_float::part_count () const
{
  return part_count_for_bits (semantics->precision + 1);
}

unsigned int
t_float::semantics_precision (const flt_semantics &semantics)
{
  return semantics.precision;
}

const t_integer_part *
t_float::sig_parts_array () const
{
  return const_cast<t_float *>(this)->sig_parts_array ();
}

t_integer_part *
t_float::sig_parts_array ()
{
  assert (category == fc_normal);

  if (part_count () > 1)
    return significand.parts;
  else
    return &significand.part;
}

/* Combine the effect of two lost fractions.  */
e_lost_fraction
t_float::combine_lost_fractions (e_lost_fraction more_significant,
				 e_lost_fraction less_significant)
{
  if (less_significant != lf_exactly_zero)
    {
      if (more_significant == lf_exactly_zero)
	more_significant = lf_less_than_half;
      else if (more_significant == lf_exactly_half)
	more_significant = lf_more_than_half;
    }

  return more_significant;
}

void
t_float::zero_significand ()
{
  category = fc_normal;
  APInt::tc_set (sig_parts_array (), 0, part_count ());
}

/* Increment an fc_normal floating point number's significand.  */
void
t_float::increment_significand ()
{
  t_integer_part carry;

  carry = APInt::tc_increment (sig_parts_array (), part_count ());

  /* Our callers should never cause us to overflow.  */
  assert (carry == 0);
}

/* Add the significand of the RHS.  Returns the carry flag.  */
t_integer_part
t_float::add_significand (const t_float &rhs)
{
  t_integer_part *parts;

  parts = sig_parts_array ();

  assert (semantics == rhs.semantics);
  assert (exponent == rhs.exponent);

  return APInt::tc_add (parts, rhs.sig_parts_array (), 0, part_count ());
}

/* Subtract the significand of the RHS with a borrow flag.  Returns
   the borrow flag.  */
t_integer_part
t_float::subtract_significand (const t_float &rhs, t_integer_part borrow)
{
  t_integer_part *parts;

  parts = sig_parts_array ();

  assert (semantics == rhs.semantics);
  assert (exponent == rhs.exponent);

  return APInt::tc_subtract (parts, rhs.sig_parts_array (), borrow,
			     part_count ());
}

/* Multiply the significand of the RHS.  If ADDEND is non-NULL, add it
   on to the full-precision result of the multiplication.  Returns the
   lost fraction.  */
e_lost_fraction
t_float::multiply_significand (const t_float &rhs, const t_float *addend)
{
  unsigned int msb, parts_count, new_parts_count, precision;
  t_integer_part *lhs_significand;
  t_integer_part *full_significand;
  e_lost_fraction lost_fraction;

  assert (semantics == rhs.semantics);

  precision = semantics->precision;
  new_parts_count = part_count_for_bits (precision * 2);
  full_significand = new t_integer_part[new_parts_count];

  lhs_significand = sig_parts_array();
  parts_count = part_count ();

  APInt::tc_full_multiply (full_significand, lhs_significand,
			   rhs.sig_parts_array (), parts_count);

  lost_fraction = lf_exactly_zero;
  msb = APInt::tc_msb (full_significand, new_parts_count);
  exponent += rhs.exponent;

  /* This must be true because our input was normalized.  We rely on
     this if ADDEND is not NULL.  */
  assert (msb >= precision);

  if (addend)
    {
      Significand saved_significand = significand;
      const flt_semantics *saved_semantics = semantics;
      flt_semantics extended_semantics;
      unsigned int new_msb;
      e_status status;

      /* Create new semantics with precision that of our MSB.  That
	 way only ADDEND and not THIS needs to be normalized.  */
      extended_semantics = *semantics;
      extended_semantics.precision = msb;

      if (new_parts_count == 1)
	significand.part = full_significand[0];
      else
	significand.parts = full_significand;
      semantics = &extended_semantics;

      t_float extended_addend (*addend);
      status = extended_addend.convert (extended_semantics, frm_to_zero);
      assert (status == fs_ok);
      lost_fraction = add_or_subtract_significand (extended_addend, false);

      /* Restore our state.  */
      if (new_parts_count == 1)
	full_significand[0] = significand.part;
      significand = saved_significand;
      semantics = saved_semantics;

      msb = APInt::tc_msb (full_significand, new_parts_count);
    }

  exponent -= (precision - 1);

  if (msb > precision)
    {
      unsigned int bits, significant_parts;
      e_lost_fraction lf;

      bits = msb - precision;
      significant_parts = part_count_for_bits (msb);
      lf = shift_right (full_significand, significant_parts, bits);
      lost_fraction = combine_lost_fractions (lf, lost_fraction);
      exponent += bits;
    }

  APInt::tc_assign (lhs_significand, full_significand, parts_count);

  delete [] full_significand;

  return lost_fraction;
}

/* Multiply the significands of LHS and RHS to DST.  */
e_lost_fraction
t_float::divide_significand (const t_float &rhs)
{
  unsigned int bit, i, parts_count;
  const t_integer_part *rhs_significand;
  t_integer_part *lhs_significand, *dividend, *divisor;
  t_integer_part scratch[2];
  e_lost_fraction lost_fraction;

  assert (semantics == rhs.semantics);

  lhs_significand = sig_parts_array();
  rhs_significand = rhs.sig_parts_array();
  parts_count = part_count ();

  if (parts_count > 1)
    dividend = new t_integer_part[parts_count * 2];
  else
    dividend = scratch;

  divisor = dividend + parts_count;

  /* Copy the dividend and divisor as they will be modified in-place.  */
  for (i = 0; i < parts_count; i++)
    {
      dividend[i] = lhs_significand[i];
      divisor[i] = rhs_significand[i];
      lhs_significand[i] = 0;
    }

  exponent -= rhs.exponent;

  unsigned int precision = semantics->precision;

  /* Normalize the divisor.  */
  bit = precision - APInt::tc_msb (divisor, parts_count);
  if (bit)
    {
      exponent += bit;
      APInt::tc_shift_left (divisor, parts_count, bit);
    }

  /* Normalize the dividend.  */
  bit = precision - APInt::tc_msb (dividend, parts_count);
  if (bit)
    {
      exponent -= bit;
      APInt::tc_shift_left (dividend, parts_count, bit);
    }

  if (APInt::tc_compare (dividend, divisor, parts_count) < 0)
    {
      exponent--;
      APInt::tc_shift_left (dividend, parts_count, 1);
      assert (APInt::tc_compare (dividend, divisor, parts_count) >= 0);
    }

  /* Long division.  */
  for (bit = precision; bit; bit -= 1)
    {
      if (APInt::tc_compare (dividend, divisor, parts_count) >= 0)
	{
	  APInt::tc_subtract (dividend, divisor, 0, parts_count);
	  APInt::tc_set_bit (lhs_significand, bit);
	}

      APInt::tc_shift_left (dividend, parts_count, 1);
    }

  /* Figure out the lost fraction.  */
  int cmp = APInt::tc_compare (dividend, divisor, parts_count);

  if (cmp > 0)
    lost_fraction = lf_more_than_half;
  else if (cmp == 0)
    lost_fraction = lf_exactly_half;
  else if (APInt::tc_is_zero (dividend, parts_count))
    lost_fraction = lf_exactly_zero;
  else
    lost_fraction = lf_less_than_half;

  if (parts_count > 1)
    delete [] dividend;

  return lost_fraction;
}

unsigned int
t_float::significand_msb () const
{
  return APInt::tc_msb (sig_parts_array (), part_count ());
}

unsigned int
t_float::significand_lsb () const
{
  return APInt::tc_lsb (sig_parts_array (), part_count ());
}

/* Note that a zero result is NOT normalized to fc_zero.  */
e_lost_fraction
t_float::shift_significand_right (unsigned int bits)
{
  /* Our exponent should not overflow.  */
  assert ((exponent_t) (exponent + bits) >= exponent);

  exponent += bits;

  return shift_right (sig_parts_array (), part_count (), bits);
}

/* Shift the significand left BITS bits, subtract BITS from its exponent.  */
void
t_float::shift_significand_left (unsigned int bits)
{
  assert (bits < semantics->precision);

  if (bits)
    {
      unsigned int parts_count = part_count ();

      APInt::tc_shift_left (sig_parts_array (), parts_count, bits);
      exponent -= bits;

      assert (!APInt::tc_is_zero (sig_parts_array (), parts_count));
    }
}

t_float::e_comparison
t_float::compare_absolute_value (const t_float &rhs) const
{
  int compare;

  assert (semantics == rhs.semantics);
  assert (category == fc_normal);
  assert (rhs.category == fc_normal);

  compare = exponent - rhs.exponent;

  /* If exponents are equal, do an unsigned bignum comparison of the
     significands.  */
  if (compare == 0)
    compare = APInt::tc_compare (sig_parts_array (), rhs.sig_parts_array (),
				 part_count ());

  if (compare > 0)
    return fcmp_greater_than;
  else if (compare < 0)
    return fcmp_less_than;
  else
    return fcmp_equal;
}

/* Sign is preserved.  */
t_float::e_status
t_float::handle_overflow (e_rounding_mode rounding_mode)
{
  /* Test if we become an infinity.  */
  if (rounding_mode == frm_to_nearest
      || (rounding_mode == frm_to_plus_infinity && !sign)
      || (rounding_mode == frm_to_minus_infinity && sign))
    {
      category = fc_infinity;
      return (e_status) (fs_overflow | fs_inexact);
    }

  /* Otherwise we become the largest finite number.  */
  category = fc_normal;
  exponent = semantics->max_exponent;
  APInt::tc_set_least_significant_bits (sig_parts_array (), part_count (),
					semantics->precision);

  return fs_inexact;
}

/* This routine must work for fc_zero of both signs, and fc_normal
   numbers.  */
bool
t_float::round_away_from_zero (e_rounding_mode rounding_mode,
			       e_lost_fraction lost_fraction)
{
  /* NaNs and infinities should not have lost fractions.  */
  assert (category == fc_normal || category == fc_zero);

  /* Our caller has already handled this case.  */
  assert (lost_fraction != lf_exactly_zero);

  switch (rounding_mode)
    {
    default:
      assert (0);

    case frm_to_nearest:
      if (lost_fraction == lf_more_than_half)
	return true;

      /* Our zeroes don't have a significand to test.  */
      if (lost_fraction == lf_exactly_half && category != fc_zero)
	return sig_parts_array()[0] & 1;
	
      return false;

    case frm_to_zero:
      return false;

    case frm_to_plus_infinity:
      return sign == false;

    case frm_to_minus_infinity:
      return sign == true;
    }
}

t_float::e_status
t_float::normalize (e_rounding_mode rounding_mode,
		    e_lost_fraction lost_fraction)
{
  unsigned int msb;
  int exponent_change;

  if (category != fc_normal)
    return fs_ok;

  /* Before rounding normalize the exponent of fc_normal numbers.  */
  msb = significand_msb ();

  if (msb)
    {
      /* The MSB is numbered from 1.  We want to place it in the integer
	 bit numbered PRECISON if possible, with a compensating change in
	 the exponent.  */
      exponent_change = msb - semantics->precision;

      /* If the resulting exponent is too high, overflow according to
	 the rounding mode.  */
      if (exponent + exponent_change > semantics->max_exponent)
	return handle_overflow (rounding_mode);

      /* Subnormal numbers have exponent min_exponent, and their MSB
	 is forced based on that.  */
      if (exponent + exponent_change < semantics->min_exponent)
	exponent_change = semantics->min_exponent - exponent;

      /* Shifting left is easy as we don't lose precision.  */
      if (exponent_change < 0)
	{
	  assert (lost_fraction == lf_exactly_zero);

	  shift_significand_left (-exponent_change);

	  return fs_ok;
	}

      if (exponent_change > 0)
	{
	  e_lost_fraction lf;

	  /* Shift right and capture any new lost fraction.  */
	  lf = shift_significand_right (exponent_change);

	  lost_fraction = combine_lost_fractions (lf, lost_fraction);

	  /* Keep MSB up-to-date.  */
	  if (msb > exponent_change)
	    msb -= exponent_change;
	  else
	    msb = 0;
	}
    }

  /* Now round the number according to rounding_mode given the lost
     fraction.  */

  /* As specified in IEEE 754, since we do not trap we do not report
     underflow for exact results.  */
  if (lost_fraction == lf_exactly_zero)
    {
      /* Canonicalize zeroes.  */
      if (msb == 0)
	category = fc_zero;

      return fs_ok;
    }

  /* Increment the significand if we're rounding away from zero.  */
  if (round_away_from_zero (rounding_mode, lost_fraction))
    {
      if (msb == 0)
	exponent = semantics->min_exponent;

      increment_significand ();
      msb = significand_msb ();

      /* Did the significand increment overflow?  */
      if (msb == semantics->precision + 1)
	{
	  /* Renormalize by incrementing the exponent and shifting our
	     significand right one.  However if we already have the
	     maximum exponent we overflow to infinity.  */
	  if (exponent == semantics->max_exponent)
	    {
	      category = fc_infinity;

	      return (e_status) (fs_overflow | fs_inexact);
	    }

	  shift_significand_right (1);

	  return fs_inexact;
	}
    }

  /* The normal case - we were and are not denormal, and any
     significand increment above didn't overflow.  */
  if (msb == semantics->precision)
    return fs_inexact;

  /* We have a non-zero denormal.  */
  assert (msb < semantics->precision);
  assert (exponent == semantics->min_exponent);

  /* Canonicalize zeroes.  */
  if (msb == 0)
    category = fc_zero;

  /* The fc_zero case is a denormal that underflowed to zero.  */
  return (e_status) (fs_underflow | fs_inexact);
}

t_float::e_status
t_float::add_or_subtract_specials (const t_float &rhs, bool subtract)
{
  switch (convolve (category, rhs.category))
    {
    default:
      assert (0);

    case convolve (fc_nan, fc_zero):
    case convolve (fc_nan, fc_normal):
    case convolve (fc_nan, fc_infinity):
    case convolve (fc_nan, fc_nan):
    case convolve (fc_normal, fc_zero):
    case convolve (fc_infinity, fc_normal):
    case convolve (fc_infinity, fc_zero):
      return fs_ok;

    case convolve (fc_zero, fc_nan):
    case convolve (fc_normal, fc_nan):
    case convolve (fc_infinity, fc_nan):
      category = fc_nan;
      return fs_ok;

    case convolve (fc_normal, fc_infinity):
    case convolve (fc_zero, fc_infinity):
      category = fc_infinity;
      sign = rhs.sign ^ subtract;
      return fs_ok;

    case convolve (fc_zero, fc_normal):
      assign (rhs);
      sign = rhs.sign ^ subtract;
      return fs_ok;

    case convolve (fc_zero, fc_zero):
      /* Sign depends on rounding mode; handled by caller.  */
      return fs_ok;

    case convolve (fc_infinity, fc_infinity):
      /* Differently signed infinities can only be validly
	 subtracted.  */
      if (sign ^ rhs.sign != subtract)
	{
	  category = fc_nan;
	  return fs_invalid_op;
	}

      return fs_ok;

    case convolve (fc_normal, fc_normal):
      return fs_div_by_zero;
    }
}

/* Add or subtract two normal numbers.  */
e_lost_fraction
t_float::add_or_subtract_significand (const t_float &rhs, bool subtract)
{
  t_integer_part carry;
  e_lost_fraction lost_fraction;
  int bits;

  /* Determine if the operation on the absolute values is effectively
     an addition or subtraction.  */
  subtract ^= (sign ^ rhs.sign);

  /* Are we bigger exponent-wise than the RHS?  */
  bits = exponent - rhs.exponent;

  /* Subtraction is more subtle than one might naively expect.  */
  if (subtract)
    {
      t_float temp_rhs (rhs);
      bool reverse;

      if (bits == 0)
	{
	  reverse = compare_absolute_value (temp_rhs) == fcmp_less_than;
	  lost_fraction = lf_exactly_zero;
	}
      else if (bits > 0)
	{
	  lost_fraction = temp_rhs.shift_significand_right (bits - 1);
	  shift_significand_left (1);
	  reverse = false;
	}
      else if (bits < 0)
	{
	  lost_fraction = shift_significand_right (-bits - 1);
	  temp_rhs.shift_significand_left (1);
	  reverse = true;
	}

      if (reverse)
	{
	  carry = temp_rhs.subtract_significand
	    (*this, lost_fraction != lf_exactly_zero);
	  copy_significand (temp_rhs);
	  sign = !sign;
	}
      else
	carry = subtract_significand
	  (temp_rhs, lost_fraction != lf_exactly_zero);

      /* Invert the lost fraction - it was on the RHS and
	 subtracted.  */
      if (lost_fraction == lf_less_than_half)
	lost_fraction = lf_more_than_half;
      else if (lost_fraction == lf_more_than_half)
	lost_fraction = lf_less_than_half;

      /* The code above is intended to ensure that no borrow is
	 necessary.  */
      assert (!carry);
    }
  else
    {
      if (bits > 0)
	{
	  t_float temp_rhs (rhs);

	  lost_fraction = temp_rhs.shift_significand_right (bits);
	  carry = add_significand (temp_rhs);
	}
      else
	{
	  lost_fraction = shift_significand_right (-bits);
	  carry = add_significand (rhs);
	}

      /* We have a guard bit; generating a carry cannot happen.  */
      assert (!carry);
    }

  return lost_fraction;
}

t_float::e_status
t_float::multiply_specials (const t_float &rhs)
{
  switch (convolve (category, rhs.category))
    {
    default:
      assert (0);

    case convolve (fc_nan, fc_zero):
    case convolve (fc_nan, fc_normal):
    case convolve (fc_nan, fc_infinity):
    case convolve (fc_nan, fc_nan):
    case convolve (fc_zero, fc_nan):
    case convolve (fc_normal, fc_nan):
    case convolve (fc_infinity, fc_nan):
      category = fc_nan;
      return fs_ok;

    case convolve (fc_normal, fc_infinity):
    case convolve (fc_infinity, fc_normal):
    case convolve (fc_infinity, fc_infinity):
      category = fc_infinity;
      return fs_ok;

    case convolve (fc_zero, fc_normal):
    case convolve (fc_normal, fc_zero):
    case convolve (fc_zero, fc_zero):
      category = fc_zero;
      return fs_ok;

    case convolve (fc_zero, fc_infinity):
    case convolve (fc_infinity, fc_zero):
      category = fc_nan;
      return fs_invalid_op;

    case convolve (fc_normal, fc_normal):
      return fs_ok;
    }
}

t_float::e_status
t_float::divide_specials (const t_float &rhs)
{
  switch (convolve (category, rhs.category))
    {
    default:
      assert (0);

    case convolve (fc_nan, fc_zero):
    case convolve (fc_nan, fc_normal):
    case convolve (fc_nan, fc_infinity):
    case convolve (fc_nan, fc_nan):
    case convolve (fc_infinity, fc_zero):
    case convolve (fc_infinity, fc_normal):
    case convolve (fc_zero, fc_infinity):
    case convolve (fc_zero, fc_normal):
      return fs_ok;

    case convolve (fc_zero, fc_nan):
    case convolve (fc_normal, fc_nan):
    case convolve (fc_infinity, fc_nan):
      category = fc_nan;
      return fs_ok;

    case convolve (fc_normal, fc_infinity):
      category = fc_zero;
      return fs_ok;

    case convolve (fc_normal, fc_zero):
      category = fc_infinity;
      return fs_div_by_zero;

    case convolve (fc_infinity, fc_infinity):
    case convolve (fc_zero, fc_zero):
      category = fc_nan;
      return fs_invalid_op;

    case convolve (fc_normal, fc_normal):
      return fs_ok;
    }
}

/* Change sign.  */
void
t_float::change_sign ()
{
  /* Look mummy, this one's easy.  */
  sign = !sign;
}

/* Normalized addition or subtraction.  */
t_float::e_status
t_float::add_or_subtract (const t_float &rhs, e_rounding_mode rounding_mode,
			  bool subtract)
{
  e_status fs;

  fs = add_or_subtract_specials (rhs, subtract);

  /* This return code means it was not a simple case.  */
  if (fs == fs_div_by_zero)
    {
      e_lost_fraction lost_fraction;

      lost_fraction = add_or_subtract_significand (rhs, subtract);
      fs = normalize (rounding_mode, lost_fraction);

      /* Can only be zero if we lost no fraction.  */
      assert (category != fc_zero || lost_fraction == lf_exactly_zero);
    }

  /* If two numbers add (exactly) to zero, IEEE 754 decrees it is a
     positive zero unless rounding to minus infinity, except that
     adding two like-signed zeroes gives that zero.  */
  if (category == fc_zero)
    {
      if (rhs.category != fc_zero || (sign == rhs.sign) == subtract)
	sign = (rounding_mode == frm_to_minus_infinity);
    }      

  return fs;
}

/* Normalized addition.  */
t_float::e_status
t_float::add (const t_float &rhs, e_rounding_mode rounding_mode)
{
  return add_or_subtract (rhs, rounding_mode, false);
}

/* Normalized subtraction.  */
t_float::e_status
t_float::subtract (const t_float &rhs, e_rounding_mode rounding_mode)
{
  return add_or_subtract (rhs, rounding_mode, true);
}

/* Normalized multiply.  */
t_float::e_status
t_float::multiply (const t_float &rhs, e_rounding_mode rounding_mode)
{
  e_status fs;

  sign ^= rhs.sign;
  fs = multiply_specials (rhs);

  if (category == fc_normal)
    {
      e_lost_fraction lost_fraction = multiply_significand (rhs, 0);
      fs = normalize (rounding_mode, lost_fraction);
      if (lost_fraction != lf_exactly_zero)
	fs = (e_status) (fs | fs_inexact);
    }

  return fs;
}

/* Normalized divide.  */
t_float::e_status
t_float::divide (const t_float &rhs, e_rounding_mode rounding_mode)
{
  e_status fs;

  sign ^= rhs.sign;
  fs = divide_specials (rhs);

  if (category == fc_normal)
    {
      e_lost_fraction lost_fraction = divide_significand (rhs);
      fs = normalize (rounding_mode, lost_fraction);
      if (lost_fraction != lf_exactly_zero)
	fs = (e_status) (fs | fs_inexact);
    }

  return fs;
}

/* Normalized fused-multiply-add.  */
t_float::e_status
t_float::fused_multiply_add (const t_float &multiplicand,
			     const t_float &addend,
			     e_rounding_mode rounding_mode)
{
  e_status fs;

  /* Post-multiplication sign, before addition.  */
  sign ^= multiplicand.sign;

  /* If and only if all arguments are normal do we need to do an
     extended-precision calculation.  */
  if (category == fc_normal && multiplicand.category == fc_normal
      && addend.category == fc_normal)
    {
      e_lost_fraction lost_fraction;

      lost_fraction = multiply_significand (multiplicand, &addend);
      fs = normalize (rounding_mode, lost_fraction);
      if (lost_fraction != lf_exactly_zero)
	fs = (e_status) (fs | fs_inexact);

      /* If two numbers add (exactly) to zero, IEEE 754 decrees it is a
	 positive zero unless rounding to minus infinity, except that
	 adding two like-signed zeroes gives that zero.  */
      if (category == fc_zero && sign != addend.sign)
	sign = (rounding_mode == frm_to_minus_infinity);
    }
  else
    {
      fs = multiply_specials (multiplicand);

      /* FS can only be fs_ok or fs_invalid_op.  There is no more work
	 to do in the latter case.  The IEEE-754R standard says it is
	 implementation-defined in this case whether, if ADDEND is a
	 quiet NaN, we raise invalid op; this implementation does so.
	 
	 If we need to do the addition we can do so with normal
	 precision.  */
      if (fs == fs_ok)
	fs = add_or_subtract (addend, rounding_mode, false);
    }

  return fs;
}

/* Comparison requires normalized numbers.  */
t_float::e_comparison
t_float::compare (const t_float &rhs) const
{
  e_comparison comparison;

  assert (semantics == rhs.semantics);

  switch (convolve (category, rhs.category))
    {
    default:
      assert (0);

    case convolve (fc_nan, fc_zero):
    case convolve (fc_nan, fc_normal):
    case convolve (fc_nan, fc_infinity):
    case convolve (fc_nan, fc_nan):
    case convolve (fc_zero, fc_nan):
    case convolve (fc_normal, fc_nan):
    case convolve (fc_infinity, fc_nan):
      return fcmp_unordered;

    case convolve (fc_infinity, fc_normal):
    case convolve (fc_infinity, fc_zero):
    case convolve (fc_normal, fc_zero):
      if (sign)
	return fcmp_less_than;
      else
	return fcmp_greater_than;

    case convolve (fc_normal, fc_infinity):
    case convolve (fc_zero, fc_infinity):
    case convolve (fc_zero, fc_normal):
      if (rhs.sign)
	return fcmp_greater_than;
      else
	return fcmp_less_than;

    case convolve (fc_infinity, fc_infinity):
      if (sign == rhs.sign)
	return fcmp_equal;
      else if (sign)
	return fcmp_less_than;
      else
	return fcmp_greater_than;

    case convolve (fc_zero, fc_zero):
      return fcmp_equal;      

    case convolve (fc_normal, fc_normal):
      break;
    }

  /* Two normal numbers.  Do they have the same sign?  */
  if (sign != rhs.sign)
    {
      if (sign)
	comparison = fcmp_less_than;
      else
	comparison = fcmp_greater_than;
    }
  else
    {
      /* Compare absolute values; invert result if negative.  */
      comparison = compare_absolute_value (rhs);

      if (sign)
	{
	  if (comparison == fcmp_less_than)
	    comparison = fcmp_greater_than;
	  else if (comparison == fcmp_greater_than)
	    comparison = fcmp_less_than;
	}
    }

  return comparison;
}

t_float::e_status
t_float::convert (const flt_semantics &to_semantics,
		  e_rounding_mode rounding_mode)
{
  unsigned int new_part_count;
  e_status fs;

  new_part_count = part_count_for_bits (to_semantics.precision + 1);

  /* If our new form is wider, re-allocate our bit pattern into wider
     storage.  */ 
  if (new_part_count > part_count ())
    {
      t_integer_part *new_parts;

      new_parts = new t_integer_part[new_part_count];
      APInt::tc_set (new_parts, 0, new_part_count);
      APInt::tc_assign (new_parts, sig_parts_array (), part_count ());
      free_significand ();
      significand.parts = new_parts;
    }

  if (category == fc_normal)
    {
      /* Re-interpret our bit-pattern.  */
      exponent += to_semantics.precision - semantics->precision;
      semantics = &to_semantics;
      fs = normalize (rounding_mode, lf_exactly_zero);
    }
  else
    {
      semantics = &to_semantics;
      fs = fs_ok;
    }

  return fs;
}

/* Convert a floating point number to an integer according to the
   rounding mode.  If the rounded integer value is out of range this
   returns an invalid operation exception.  If the rounded value is in
   range but the floating point number is not the exact integer, the C
   standard doesn't require an inexact exception to be raised.  IEEE
   854 does require it so we do that.

   Note that for conversions to integer type the C standard requires
   round-to-zero to always be used.  */
t_float::e_status
t_float::convert_to_integer (t_integer_part *parts, unsigned int width,
			     bool is_signed,
			     e_rounding_mode rounding_mode) const
{
  e_lost_fraction lost_fraction;
  unsigned int msb, parts_count;
  int bits;

  /* Handle the three special cases first.  */
  if (category == fc_infinity || category == fc_nan)
    return fs_invalid_op;

  parts_count = part_count_for_bits (width);

  if (category == fc_zero)
    {
      APInt::tc_set (parts, 0, parts_count);
      return fs_ok;
    }

  /* Shift the bit pattern so the fraction is lost.  */
  t_float tmp (*this);

  bits = (int) semantics->precision - 1 - exponent;

  if (bits > 0)
    lost_fraction = tmp.shift_significand_right (bits);
  else
    {
      tmp.shift_significand_left (-bits);
      lost_fraction = lf_exactly_zero;
    }

  if (lost_fraction != lf_exactly_zero
      && tmp.round_away_from_zero (rounding_mode, lost_fraction))
    tmp.increment_significand ();

  msb = tmp.significand_msb ();

  /* Negative numbers cannot be represented as unsigned.  */
  if (!is_signed && tmp.sign && msb)
    return fs_invalid_op;

  /* It takes exponent + 1 bits to represent the truncated floating
     point number without its sign.  We lose a bit for the sign, but
     the maximally negative integer is a special case.  */
  if (msb > width)
    return fs_invalid_op;

  if (is_signed && msb == width
      && (!tmp.sign || tmp.significand_lsb () != msb))
    return fs_invalid_op;

  APInt::tc_assign (parts, tmp.sig_parts_array (), parts_count);

  if (tmp.sign)
    APInt::tc_negate (parts, parts_count);

  if (lost_fraction == lf_exactly_zero)
    return fs_ok;
  else
    return fs_inexact;
}

t_float::e_status
t_float::convert_from_unsigned_integer (t_integer_part *parts,
					unsigned int part_count,
					e_rounding_mode rounding_mode)
{
  unsigned int msb, precision;
  e_lost_fraction lost_fraction;

  msb = APInt::tc_msb (parts, part_count);
  precision = semantics->precision;

  category = fc_normal;
  exponent = precision - 1;

  if (msb > precision)
    {
      exponent += (msb - precision);
      lost_fraction = shift_right (parts, part_count, msb - precision);
      msb = precision;
    }
  else
    lost_fraction = lf_exactly_zero;

  /* Copy the bit image.  */
  zero_significand ();
  APInt::tc_assign (sig_parts_array (), parts, part_count_for_bits (msb));

  return normalize (rounding_mode, lost_fraction);
}

t_float::e_status
t_float::convert_from_integer (const t_integer_part *parts,
			       unsigned int part_count, bool is_signed,
			       e_rounding_mode rounding_mode)
{
  unsigned int width;
  e_status status;
  t_integer_part *copy;

  copy = new t_integer_part[part_count];
  APInt::tc_assign (copy, parts, part_count);

  width = part_count * t_integer_part_width;

  sign = false;
  if (is_signed && APInt::tc_extract_bit (parts, width))
    {
      sign = true;
      APInt::tc_negate (copy, part_count);
    }

  status = convert_from_unsigned_integer (copy, part_count, rounding_mode);
  delete [] copy;

  return status;
}

t_float::e_status
t_float::convert_from_hexadecimal_string (const char *p,
					  e_rounding_mode rounding_mode)
{
  e_lost_fraction lost_fraction;
  t_integer_part *significand;
  unsigned int bit_pos, parts_count;
  const char *dot, *first_significant_digit;

  zero_significand ();
  exponent = 0;
  category = fc_normal;

  significand = sig_parts_array ();
  parts_count = part_count ();
  bit_pos = parts_count * t_integer_part_width;

  /* Skip leading zeroes and any (hexa)decimal point.  */
  p = skip_leading_zeroes_and_any_dot (p, &dot);
  first_significant_digit = p;

  for (;;)
    {
      t_integer_part hex_value;

      if (*p == '.')
	{
	  assert (dot == 0);
	  dot = p++;
	}

      hex_value = hex_digit_value (*p);
      if (hex_value == -1U)
	{
	  lost_fraction = lf_exactly_zero;
	  break;
	}

      p++;
 
      /* Store the number whilst 4-bit nibbles remain.  */
      if (bit_pos)
	{
	  bit_pos -= 4;
	  hex_value <<= bit_pos % t_integer_part_width;
	  significand[bit_pos / t_integer_part_width] |= hex_value;
	}
      else
	{
	  lost_fraction = trailing_hexadecimal_fraction (p, hex_value);
	  while (hex_digit_value (*p) != -1U)
	    p++;
	  break;
	}
    }

  /* Hex floats require an exponent but not a hexadecimal point.  */
  assert (*p == 'p' || *p == 'P');

  /* Ignore the exponent if we are zero.  */
  if (p != first_significant_digit)
    {
      int exp_adjustment;

      /* Implicit hexadecimal point?  */
      if (!dot)
	dot = p;

      /* Calculate the exponent adjustment implicit in the number of
	 significant digits.  */
      exp_adjustment = dot - first_significant_digit;
      if (exp_adjustment < 0)
	exp_adjustment++;
      exp_adjustment = exp_adjustment * 4 - 1;

      /* Adjust for writing the significand starting at the most
	 significant nibble.  */
      exp_adjustment += semantics->precision;
      exp_adjustment -= parts_count * t_integer_part_width;

      /* Adjust for the given exponent.  */
      exponent = total_exponent (p, exp_adjustment);
    }

  return normalize (rounding_mode, lost_fraction);
}

t_float::e_status
t_float::attempt_decimal_to_binary_conversion (const decimal_number *number,
					       unsigned int part_count,
					       e_rounding_mode rounding_mode)
{
  t_integer_part *parts;
  unsigned int half_ulps_error;

  parts = new t_integer_part[part_count];
  APInt::tc_set (parts, 0, part_count);

  if (part_count > number->part_count)
    {
      APInt::tc_assign (parts, number->parts, number->part_count);
      half_ulps_error = 0;
    }
  else
    {
      APInt::tc_assign (parts, number->parts, part_count);
      half_ulps_error = 2;
    }
#if 0
  if (exponent)
    {
      t_float tenth_power;
      c_part exponent_half_ulps_error;
      unsigned int absolute_exponent;

      if (number->exponent < 0)
	absolute_exponent = -number->exponent;
      else
	absolute_exponent = number->exponent;

      exponent_half_ulps_error = power_of_ten (&tenth_power,
					       absolute_exponent,
					       &read_decimal_semantics);
      if (exponent < 0)
	as = t_float_div (flt, flt, &tenth_power, &read_decimal_semantics);
      else
	as = t_float_mul (flt, flt, &tenth_power, &read_decimal_semantics);

      half_ulps_error = 2 * (exponent_half_ulps_error + half_ulps_error);
      if (as & as_flt_inexact)
	half_ulps_error++;

      semantically_normalize (flt, &read_decimal_semantics, lf_exactly_zero);
    }
#endif


  category = fc_zero;

  return fs_ok;
}

t_float::e_status
t_float::convert_from_decimal_string (const char *p,
				      e_rounding_mode rounding_mode)
{
  e_status status;
  decimal_number number;

  if (read_decimal_number (p, &number))
    {
      category = fc_zero;
      status = fs_ok;
    }
  else
    {
      category = fc_normal;

      for (unsigned int count = part_count ();; count++)
	{
	  status = attempt_decimal_to_binary_conversion (&number, count,
							 rounding_mode);
	  if (status != fs_invalid_op)
	    break;
	}

      delete [] number.parts;
    }

  return status;
}

t_float::e_status
t_float::convert_from_string (const char *p, e_rounding_mode rounding_mode)
{
  /* Handle a leading minus sign.  */
  if (*p == '-')
    sign = 1, p++;
  else
    sign = 0;

  if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X'))
    return convert_from_hexadecimal_string (p + 2, rounding_mode);
  else
    return convert_from_decimal_string (p, rounding_mode);
}

#if 0
static c_part power_of_ten (t_float *dst, unsigned int exponent,
			    const c_float_semantics *semantics);
static unsigned int first_exponent_losing_precision = -1U;

static c_part
part_ulps_from_half (c_part part, unsigned int bits)
{
  c_part half;

  assert_cheap (bits != 0 && bits <= c_part_width);

  part &= ~(c_part) 0 >> (c_part_width - bits);
  half = (c_part) 1 << (bits - 1);

  if (part >= half)
    return part - half;
  else
    return half - part;
}

static c_part
multi_part_ulps_from_half (c_part *significand, unsigned int excess_precision)
{
  unsigned int part_precision;
  unsigned int index;
  c_part part, half;

  index = (excess_precision - 1) / c_part_width;
  part_precision = (excess_precision - 1) % c_part_width + 1;

  part = significand[index];
  if (part_precision != c_part_width)
    part &= ~(c_part) 0 >> (c_part_width - part_precision);
  half = (c_part) 1 << (part_precision - 1);

  if (part > half || part < half - 1)
    return ~(c_part) 0;

  /* The difference is significand[0] or -significand[0] unless there
     are non-zero parts in-between.  */
  if (part == half)
    {
      while (--index)
	if (significand[index])
	  return ~(c_part) 0;

      return significand[0];
    }
  else
    {
      while (--index)
	if (~significand[index])
	  return ~(c_part) 0;

      return -significand[0];
    }
}

static c_part
ulps_from_half (c_part *significand, unsigned int excess_precision)
{
  if (excess_precision <= c_part_width)
    return part_ulps_from_half (significand[0], excess_precision);
  else
    return multi_part_ulps_from_half (significand, excess_precision);
}

static void
calc_power_of_ten (t_float *dst, unsigned int exponent,
		   const c_float_semantics *semantics)
{
  static c_part tens[10] = { 0, 10, 100, 1000, 10000, 100000, 1000000,
			     10000000, 100000000, 1000000000 };

  t_float tmp;
  e_arith_status as;

  if (exponent <= 9)
    float_from_part (dst, tens[exponent], semantics);
  else if (exponent <= 15)
    {
      power_of_ten (dst, 8, semantics);
      power_of_ten (&tmp, exponent - 8, semantics);
      as = t_float_mul (dst, dst, &tmp, semantics);
      assert_cheap (as == 0);
    }
  else
    {
      c_part half_ulps_error, ulps_room;
      unsigned int excess_precision;

      assert_cheap ((exponent & (exponent - 1)) == 0);
      power_of_ten (&tmp, exponent / 2, semantics);
      t_float_convert (&tmp, semantics, &power_of_ten_semantics);

      as = t_float_mul (dst, &tmp, &tmp, &power_of_ten_semantics);

      half_ulps_error = 2 * 2 * (exponent > first_exponent_losing_precision);
      if (as & as_flt_inexact)
	{
	  if (first_exponent_losing_precision == -1U)
	    first_exponent_losing_precision = exponent;
	  half_ulps_error++;
	}

      excess_precision = (power_of_ten_semantics.precision
			  - semantics->precision);
      ulps_room = ulps_from_half (dst->significand, excess_precision);

      if (half_ulps_error > ulps_room)
	{
	  ulps_room <<= 1;
	  assert_cheap (half_ulps_error < ulps_room);
	}

      t_float_convert (dst, &power_of_ten_semantics, semantics);
    }
}

static c_part
power_of_ten (t_float *dst, unsigned int exponent,
	      const c_float_semantics *semantics)
{
  /* 10 to the powers 1, 2, ..., 15.  */
  static t_float unit_tens[15];

  /* 10 to the powers 16, 32, 64, 128, 256, 512, 1024, 2048, 4096.  */
  static t_float higher_tens[9];

  t_float *ten;
  c_part half_ulps_error;
  unsigned int power;
  bool store;

  assert_cheap (exponent != 0);

  if (exponent >= 4952)
    {
      dst->category = fc_infinity;
      dst->sign = false;
      return 0;
    }

  power = exponent & 15;
  if (power)
    {
      ten = &unit_tens[power];
      if (ten->category != fc_normal)
	calc_power_of_ten (ten, power, semantics);
      *dst = *ten;
      store = false;
    }
  else
    store = true;

  half_ulps_error = 0;
  power = 16;
  ten = higher_tens;
  for (exponent >>= 4; exponent; exponent >>= 1, ten++, power <<= 1)
    {
      if (! (exponent & 1))
	continue;

      if (ten->category != fc_normal)
	calc_power_of_ten (ten, power, semantics);

      if (store)
	{
	  store = false;
	  *dst = *ten;
	  half_ulps_error = power >= first_exponent_losing_precision;
	}
      else
	{
	  e_arith_status as;

	  as = t_float_mul (dst, dst, ten, semantics);
	  half_ulps_error = 2 * (half_ulps_error + 1);
	  if (as & as_flt_inexact)
	    half_ulps_error++;
	}
    }

  return half_ulps_error;
}

static e_arith_status
read_decimal_float (t_float *flt, const c_number *number,
		    const c_float_semantics *semantics)
{
  e_lost_fraction lost_fraction;
  e_arith_status as;
  c_part half_ulps_error, ulps_room;
  unsigned int excess_precision;
  int exponent;

  lost_fraction = read_decimal_significand (number, flt,
					    read_decimal_semantics.precision,
					    &exponent);

  as = semantically_normalize (flt, &read_decimal_semantics, lost_fraction);

  /* Exit early for zeroes; we don't care about exponent etc.  */
  if (flt->category == fc_zero)
    return as;

  t_float tmp (fsk_ieee_quad, fc_zero, false);

  tmp.convert_from_integer (const t_integer_part *, unsigned int, bool,
			    e_rounding_mode);


  half_ulps_error = (as & as_flt_inexact) != 0;

  if (exponent)
    {
      t_float tenth_power;
      c_part exponent_half_ulps_error;
      unsigned int absolute_exponent;

      if (exponent < 0)
	absolute_exponent = -exponent;
      else
	absolute_exponent = exponent;

      exponent_half_ulps_error = power_of_ten (&tenth_power,
					       absolute_exponent,
					       &read_decimal_semantics);
      if (exponent < 0)
	as = t_float_div (flt, flt, &tenth_power, &read_decimal_semantics);
      else
	as = t_float_mul (flt, flt, &tenth_power, &read_decimal_semantics);

      half_ulps_error = 2 * (exponent_half_ulps_error + half_ulps_error);
      if (as & as_flt_inexact)
	half_ulps_error++;

      semantically_normalize (flt, &read_decimal_semantics, lf_exactly_zero);
    }

  excess_precision = read_decimal_semantics.precision - semantics->precision;

  if (half_ulps_error)
    {
      as = as_flt_inexact;
      ulps_room = ulps_from_half (flt->significand, excess_precision);

      if (half_ulps_error > ulps_room)
	{
	  ulps_room <<= 1;
	  if (half_ulps_error >= ulps_room)
	    as |= as_flt_cst_rounding;
	}
    }
  else
    as = as_ok;

  flt->exponent -= excess_precision;

  as |= semantically_normalize (flt, semantics, lf_exactly_zero);

  return as;
}
#endif
