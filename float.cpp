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

/* Represents floating point arithmetic semantics.  */
struct llvm::flt_semantics
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
} all_semantics [] = {
  /* fsk_ieee_single */
  { 127, -126, 24 },
  /* fsk_ieee_double */
  { 1023, -1022, 53 },
  /* fsk_ieee_quad */
  { 16383, -16382, 113 },
  /* fsk_x87_double_extended */
  { 16383, -16382, 64 },
};

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
}

const flt_semantics &
t_float::semantics_for_kind (e_semantics_kind kind)
{
  return all_semantics[(int) kind];
}

unsigned int
t_float::precision_for_kind (e_semantics_kind kind)
{
  return semantics_for_kind (kind).precision;
}

unsigned int
t_float::part_count_for_kind (e_semantics_kind kind)
{
  return part_count_for_bits (precision_for_kind (kind) + 1);
}

/* Constructors.  */
void
t_float::initialize (e_semantics_kind semantics_kind)
{
  unsigned int count;

  kind = semantics_kind;

  count = part_count_for_kind (semantics_kind);
  is_wide = (count > 1);
  if (is_wide)
    significand.parts = new t_integer_part[count];
}

void
t_float::free_significand ()
{
  if (is_wide)
    delete [] significand.parts;
}

void
t_float::assign (const t_float &rhs)
{
  assert (kind == rhs.kind);

  sign = rhs.sign;
  category = rhs.category;
  exponent = rhs.exponent;

  if (category == fc_normal)
    APInt::tc_assign (sig_parts_array(), rhs.sig_parts_array(),
		      part_count_for_kind (kind));
}

t_float &
t_float::operator= (const t_float &rhs)
{
  if (this != &rhs)
    {
      if (kind != rhs.kind)
	{
	  free_significand ();
	  initialize (kind);
	}

      assign (rhs);
    }

  return *this;
}

t_float::t_float (e_semantics_kind kind, t_integer_part value)
{
  initialize (kind);
  sign = 0;
  zero_significand ();
  exponent = precision_for_kind (kind) - 1;
  sig_parts_array ()[0] = value;
  normalize (frm_to_nearest, lf_exactly_zero);
}

t_float::t_float (e_semantics_kind kind, e_category c, bool negative)
{
  initialize (kind);
  category = c;
  sign = negative;
  if (category == fc_normal)
    category = fc_zero;
}

t_float::t_float (e_semantics_kind kind, const char *text)
{
  initialize (kind);
  convert_from_string (text, frm_to_nearest);
}

t_float::t_float (const t_float &rhs)
{
  initialize (rhs.kind);
  assign (rhs);
}

t_float::~t_float ()
{
  free_significand ();
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

  if (is_wide)
    return significand.parts;
  else
    return &significand.part;
}

bool
t_float::is_significand_zero ()
{
  return APInt::tc_is_zero (sig_parts_array (), part_count_for_kind (kind));
}

/* Combine the effect of two lost fractions.  */
t_float::e_lost_fraction
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
  APInt::tc_set (sig_parts_array (), 0, part_count_for_kind (kind));
}

/* Increment a floating point number's significand.  */
void
t_float::increment_significand ()
{
  t_integer_part carry;

  if (category == fc_zero)
    zero_significand ();

  carry = APInt::tc_increment (sig_parts_array (), part_count_for_kind (kind));

  /* Our callers should never cause us to overflow.  */
  assert (carry == 0);
}

/* Increment a floating point number's significand.  */
void
t_float::negate_significand ()
{
  APInt::tc_negate (sig_parts_array (), part_count_for_kind (kind));
}

/* Shift the significand left BITS bits, subtract BITS from its exponent.  */
void
t_float::logical_left_shift_significand (unsigned int bits)
{
  assert (bits < precision_for_kind (kind));

  if (bits)
    {
      t_integer_part *parts;

      parts = sig_parts_array ();
      APInt::tc_left_shift (parts, parts, part_count_for_kind (kind), bits);
      exponent -= bits;

      assert (!is_significand_zero ());
    }
}

/* Add or subtract the significand of the RHS.  Returns the carry /
   borrow flag.  */
t_integer_part
t_float::add_or_subtract_significands (const t_float &rhs, bool subtract)
{
  t_integer_part *parts;

  parts = sig_parts_array ();

  assert (kind == rhs.kind);
  assert (exponent == rhs.exponent);

  return (subtract ? APInt::tc_subtract: APInt::tc_add)
    (parts, const_cast<const t_integer_part *>(parts),
     rhs.sig_parts_array (), 0, part_count_for_kind (kind));
}

/* Multiply the significand of the RHS.  Returns the lost fraction.  */
t_float::e_lost_fraction
t_float::multiply_significand (const t_float &rhs)
{
  unsigned int i, msb, parts_count, precision;
  t_integer_part *lhs_significand;
  t_integer_part scratch[2], *full_significand;
  e_lost_fraction lost_fraction;

  assert (kind == rhs.kind);

  lhs_significand = sig_parts_array();
  parts_count = part_count_for_kind (kind);

  if (parts_count > 1)
    full_significand = new t_integer_part[parts_count * 2];
  else
    full_significand = scratch;

  APInt::tc_full_multiply (full_significand, lhs_significand,
			   rhs.sig_parts_array (), parts_count);

  msb = APInt::tc_msb (full_significand, parts_count * 2);
  assert (msb != 0);

  precision = precision_for_kind (kind);
  if (msb > precision)
    {
      unsigned int bits, significant_parts;

      bits = msb - precision;
      significant_parts = part_count_for_bits (msb);
      lost_fraction = right_shift (full_significand, significant_parts, bits);
      exponent += bits;
    }
  else
    lost_fraction = lf_exactly_zero;

  for (i = 0; i < parts_count; i++)
    lhs_significand[i] = full_significand[i];

  if (parts_count > 1)
    delete [] full_significand;

  return lost_fraction;
}

/* Multiply the significands of LHS and RHS to DST.  */
t_float::e_lost_fraction
t_float::divide_significand (const t_float &rhs)
{
  unsigned int bit, i, parts_count;
  const t_integer_part *rhs_significand;
  t_integer_part *lhs_significand, *dividend, *divisor;
  t_integer_part scratch[2];
  e_lost_fraction lost_fraction;

  assert (kind == rhs.kind);

  lhs_significand = sig_parts_array();
  rhs_significand = rhs.sig_parts_array();
  parts_count = part_count_for_kind (kind);

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

  unsigned int precision = precision_for_kind (kind);

  /* Normalize the divisor.  */
  bit = precision - APInt::tc_msb (divisor, parts_count);
  if (bit)
    {
      exponent += bit;
      APInt::tc_left_shift (divisor, divisor, parts_count, bit);
    }

  /* Normalize the dividend.  */
  bit = precision - APInt::tc_msb (dividend, parts_count);
  if (bit)
    {
      exponent -= bit;
      APInt::tc_left_shift (dividend, dividend, parts_count, bit);
    }

  /* Long division.  */
  unsigned int set = 0;
  for (bit = precision; bit; bit -= set)
    {
      if (APInt::tc_compare (dividend, divisor, parts_count) >= 0)
	{
	  APInt::tc_subtract (dividend, dividend, divisor, 0, parts_count);
	  APInt::tc_set_bit (lhs_significand, bit);
	  set = 1;
	}
      else if (!set)
	exponent--;

      APInt::tc_left_shift (dividend, dividend, parts_count, 1);
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

/* Shift DST right COUNT bits noting lost fraction.  */
t_float::e_lost_fraction
t_float::right_shift (t_integer_part *dst, unsigned int parts,
		      unsigned int count)
{
  e_lost_fraction lost_fraction;
  unsigned int lsb;

  /* Before shifting see if we would lose precision.  Fast-path two
     cases that would fail the generic logic.  */
  if (count == 0 || (lsb = APInt::tc_lsb (dst, parts)) == 0)
    lost_fraction = lf_exactly_zero;
  else
    {
      if (lsb == 0 || count < lsb)
	lost_fraction = lf_exactly_zero;
      else if (count == lsb)
	lost_fraction = lf_exactly_half;
      else if (count <= parts * t_integer_part_width
	       && APInt::tc_extract_bit (dst, count))
	lost_fraction = lf_more_than_half;
      else
	lost_fraction = lf_less_than_half;

      APInt::tc_right_shift (dst, dst, parts, count);
    }

  return lost_fraction;
}

unsigned int
t_float::significand_msb ()
{
  return APInt::tc_msb (sig_parts_array (), part_count_for_kind (kind));
}

unsigned int
t_float::significand_lsb () const
{
  return APInt::tc_lsb (sig_parts_array (), part_count_for_kind (kind));
}

/* Note that a zero result is NOT normalized to fc_zero.  */
t_float::e_lost_fraction
t_float::rescale_significand_right (unsigned int bits)
{
  /* Our exponent should not overflow.  */
  assert ((exponent_t) (exponent + bits) >= exponent);

  exponent += bits;

  return right_shift (sig_parts_array (), part_count_for_kind (kind), bits);
}

t_float::e_comparison
t_float::compare_absolute_value (const t_float &rhs) const
{
  int compare;

  assert (kind == rhs.kind);
  assert (category == fc_normal);
  assert (rhs.category == fc_normal);

  compare = exponent - rhs.exponent;

  /* If exponents are equal, do an unsigned bignum comparison of the
     significands.  */
  if (compare == 0)
    compare = APInt::tc_compare (sig_parts_array (), rhs.sig_parts_array (),
				 part_count_for_kind (kind));

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
  const flt_semantics &our_semantics = semantics_for_kind (kind);

  category = fc_normal;
  exponent = our_semantics.max_exponent;
  APInt::tc_set_lsbs (sig_parts_array (), part_count_for_kind (kind),
		      our_semantics.precision);

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
  const flt_semantics &our_semantics = semantics_for_kind (kind);
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
      exponent_change = msb - our_semantics.precision;

      /* If the resulting exponent is too high, overflow according to
	 the rounding mode.  */
      if (exponent + exponent_change > our_semantics.max_exponent)
	return handle_overflow (rounding_mode);

      /* Subnormal numbers have exponent min_exponent, and their MSB
	 is forced based on that.  */
      if (exponent + exponent_change < our_semantics.min_exponent)
	exponent_change = our_semantics.min_exponent - exponent;

      /* Shifting left is easy as we don't lose precision.  */
      if (exponent_change < 0)
	{
	  assert (lost_fraction == lf_exactly_zero);

	  logical_left_shift_significand (-exponent_change);

	  return fs_ok;
	}

      if (exponent_change > 0)
	{
	  e_lost_fraction lf;

	  /* Shift right and capture any new lost fraction.  */
	  lf = rescale_significand_right (exponent_change);

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
	exponent = our_semantics.min_exponent;

      increment_significand ();
      msb = significand_msb ();

      /* Did the significand increment overflow?  */
      if (msb == our_semantics.precision + 1)
	{
	  /* Renormalize by incrementing the exponent and shifting our
	     significand right one.  However if we already have the
	     maximum exponent we overflow to infinity.  */
	  if (exponent == our_semantics.max_exponent)
	    {
	      category = fc_infinity;

	      return (e_status) (fs_overflow | fs_inexact);
	    }

	  rescale_significand_right (1);

	  return fs_inexact;
	}
    }

  /* The normal case - we were and are not denormal, and any
     significand increment above didn't overflow.  */
  if (msb == our_semantics.precision)
    return fs_inexact;

  /* We have a non-zero denormal.  */
  assert (msb < our_semantics.precision);
  assert (exponent == our_semantics.min_exponent);

  /* Canonicalize zeroes.  */
  if (msb == 0)
    category = fc_zero;

  /* The fc_zero case is a denormal that underflowed to zero.  */
  return (e_status) (fs_underflow | fs_inexact);
}

/* Unfortunately for IEEE semantics a rounding mode is needed
   here.  */
t_float::e_status
t_float::unnormalized_add_or_subtract (const t_float &rhs, bool subtract,
				       e_rounding_mode rounding_mode,
				       e_lost_fraction *lost_fraction_ptr)
{
  t_integer_part carry;

  *lost_fraction_ptr = lf_exactly_zero;

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
      /* Sign handled by caller.  */
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
      break;
    }

  e_lost_fraction lost_fraction;
  int bits;

  /* Determine if the operation on the absolute values is effectively
     an addition or subtraction.  */
  subtract ^= (sign ^ rhs.sign);

  /* Shift the significand of one operand right, losing precision, so
     they have same exponent.  Capture the lost fraction.  */
  bits = exponent - rhs.exponent;

  if (bits > 0)
    {
      t_float temp_rhs (rhs);

      lost_fraction = temp_rhs.rescale_significand_right (bits);
      carry = add_or_subtract_significands (temp_rhs, subtract);
    }
  else
    {
      lost_fraction = rescale_significand_right (-bits);
      carry = add_or_subtract_significands (rhs, subtract);
    }

  if (subtract)
    {
      if (carry)
	{
	  /* If bits > 0 we've right-shifted the RHS, and so to carry
	     means we'd be denormal.  But then our exponent is
	     minimal, so bits <= 0, a contradiction.  */
	  assert (bits <= 0);

	  negate_significand ();
	  sign = !sign;

	  /* Correct the lost fraction - it was the result of the
	     reverse subtraction.  */
	  if (lost_fraction == lf_less_than_half)
	    lost_fraction = lf_more_than_half;
	  else if (lost_fraction == lf_more_than_half)
	    lost_fraction = lf_less_than_half;
	}
      else
	{
	  /* If bits < 0, then we were shifted right, meaning we would
	     generate a carry unless the RHS were denormal.  But if
	     the RHS is denormal bits < 0 is impossible.  */
	  assert (bits >= 0);
	}
    }
  else
    {
      /* We have a guard bit; generating a carry cannot happen.  */
      assert (!carry);
    }

  *lost_fraction_ptr = lost_fraction;

  return fs_ok;
}

t_float::e_status
t_float::unnormalized_multiply (const t_float &rhs,
				e_lost_fraction *lost_fraction)
{
  sign ^= rhs.sign;
  *lost_fraction = lf_exactly_zero;

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
      break;
    }

  exponent += rhs.exponent - (precision_for_kind (kind) - 1);
  *lost_fraction = multiply_significand (rhs);

  if (*lost_fraction == lf_exactly_zero)
    return fs_ok;
  else
    return fs_inexact;
}

t_float::e_status
t_float::unnormalized_divide (const t_float &rhs,
			      e_lost_fraction *lost_fraction)
{
  sign ^= rhs.sign;
  *lost_fraction = lf_exactly_zero;

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
      break;
    }

  exponent -= rhs.exponent;
  *lost_fraction = divide_significand (rhs);

  if (*lost_fraction == lf_exactly_zero)
    return fs_ok;
  else
    return fs_inexact;
}

/* Change sign.  */
void
t_float::change_sign ()
{
  /* Look mummy, this one's easy.  */
  sign = !sign;
}

/* Normalized addition.  */
t_float::e_status
t_float::add (const t_float &rhs, e_rounding_mode rounding_mode)
{
  e_status fs;
  e_lost_fraction lost_fraction;

  fs = unnormalized_add_or_subtract (rhs, false, rounding_mode,
				     &lost_fraction);

  /* We return normalized numbers.  */
  fs = (e_status) (fs | normalize (rounding_mode, lost_fraction));

  /* If two numbers add (exactly) to zero, IEEE 754 decrees it is a
     positive zero unless rounding to minus infinity, except that
     adding two like-signed zeroes gives that zero.  */
  if (category == fc_zero)
    {
      assert (lost_fraction == lf_exactly_zero);

      if (rhs.category != fc_zero || sign != rhs.sign)
	sign = (rounding_mode == frm_to_minus_infinity);
    }      

  return fs;
}

/* Normalized subtraction.  */
t_float::e_status
t_float::subtract (const t_float &rhs, e_rounding_mode rounding_mode)
{
  e_status fs;
  e_lost_fraction lost_fraction;

  fs = unnormalized_add_or_subtract (rhs, true, rounding_mode, &lost_fraction);

  /* We return normalized numbers.  */
  fs = (e_status) (fs | normalize (rounding_mode, lost_fraction));

  /* If two numbers add (exactly) to zero, IEEE 754 decrees it is a
     positive zero unless rounding to minus infinity, except that
     adding two like-signed zeroes gives that zero.  */
  if (category == fc_zero)
    {
      assert (lost_fraction == lf_exactly_zero);

      if (rhs.category != fc_zero || sign == rhs.sign)
	sign = (rounding_mode == frm_to_minus_infinity);
    }      

  return fs;
}

/* Normalized multiply.  */
t_float::e_status
t_float::multiply (const t_float &rhs, e_rounding_mode rounding_mode)
{
  e_status fs;
  e_lost_fraction lost_fraction;

  fs = unnormalized_multiply (rhs, &lost_fraction);

  /* We return normalized numbers.  */
  fs = (e_status) (fs | normalize (rounding_mode, lost_fraction));

  return fs;
}

/* Normalized divide.  */
t_float::e_status
t_float::divide (const t_float &rhs, e_rounding_mode rounding_mode)
{
  e_status fs;
  e_lost_fraction lost_fraction;

  fs = unnormalized_divide (rhs, &lost_fraction);

  /* We return normalized numbers.  */
  fs = (e_status) (fs | normalize (rounding_mode, lost_fraction));

  return fs;
}

/* Comparison requires normalized numbers.  */
t_float::e_comparison
t_float::compare (const t_float &rhs) const
{
  e_comparison comparison;

  assert (kind == rhs.kind);

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
t_float::convert (e_semantics_kind to_kind, e_rounding_mode rounding_mode)
{
  if (category != fc_normal)
    {
      kind = to_kind;
      return fs_ok;
    }

  /* Reinterpret the bit pattern.  */
  exponent += precision_for_kind (to_kind) - precision_for_kind (kind);
  kind = to_kind;

  return normalize (rounding_mode, lf_exactly_zero);
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

  bits = (int) precision_for_kind (kind) - 1 - exponent;

  if (bits > 0)
    lost_fraction = tmp.rescale_significand_right (bits);
  else
    {
      tmp.logical_left_shift_significand (-bits);
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
  precision = precision_for_kind (kind);

  category = fc_normal;
  exponent = precision - 1;

  if (msb > precision)
    {
      exponent += (msb - precision);
      lost_fraction = right_shift (parts, part_count, msb - precision);
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

t_float::e_lost_fraction
t_float::trailing_hexadecimal_fraction (const char *p,
					unsigned int digit_value)
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

  dot = 0;
  significand = sig_parts_array ();
  parts_count = part_count_for_kind (kind);
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
      exp_adjustment += precision_for_kind (kind);
      exp_adjustment -= parts_count * t_integer_part_width;

      /* Adjust for the given exponent.  */
      exponent = total_exponent (p, exp_adjustment);
    }

  return normalize (rounding_mode, lost_fraction);
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
    assert (0);
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

/* Returns the number of digits P is to the left of the (possibly
   implicit) first fraction digit, ignoring the decimal point.  In the
   number 8.62, this returns 1 for the 8 and -1 for the 2.  */
static int
digits_left_of_fraction (const c_number *number, const char *p)
{
  int digits;

  if (number->text.decimal_point)
    {
      digits = number->text.decimal_point - p;
      if (digits < 0)
	digits++;
    }
  else
    {
      assert_cheap (number->text.suffix != NULL);

      /* Decimal floats may not have an exponent; hexadecimal ones are
	 guaranteed to.  SUFFIX is always set.  */
      if (number->text.exponent)
	digits = number->text.exponent - p;
      else
	digits = number->text.suffix - p;

      assert_cheap (digits >= 0);
    }

  return digits;
}

static bool
remainder_non_zero (const char *p)
{
  unsigned int digit_value;

  do
    {
      if (*p == '.')
	p++;

      digit_value = host_digit_value (*p);
      p++;
    }
  while (digit_value == 0);

  return digit_value != -1U;
}

static e_lost_fraction
read_decimal_significand (const c_number *number, t_float *flt,
			  unsigned int precision, int *p_exponent)
{
  const char *p;
  c_part significand[tf_parts * 2];
  unsigned int i, part_count;
  unsigned int digit_value, msb;
  int exponent;
  e_lost_fraction lost_fraction;

  part_count = (precision + (c_part_width - 1)) / c_part_width;
  assert_cheap (part_count + 1 <= ARRAY_SIZE (significand));

  tc_set (significand, 0, part_count + 1);
  p = skip_leading_zeroes (number->pp_token->spelling);
  msb = -1U;

  /* Result is in SRC on exit.  */
  for (;;)
    {
      if (*p == '.')
	{
	  p++;
	  continue;
	}

      digit_value = host_digit_value (*p);
      if (digit_value == -1U)
	break;

      p++;

      /* FIXME: this is inefficient.  */
      tc_multiply_part (significand, significand, 10, digit_value,
			part_count, part_count + 1, false);

      /* FIXME: and this.  */
      msb = tc_msb (significand, part_count + 1);
      if (msb >= precision)
	break;
    }

  exponent = digits_left_of_fraction (number, p);
  if (number->text.exponent)
    exponent = total_exponent (number->text.exponent, exponent);
  *p_exponent = exponent;

  flt->sign = 0;
  zero_significand (flt);
  for (i = 0; i < part_count; i++)
    flt->significand[i] = significand[i];

  lost_fraction = lf_exactly_zero;
  if (!zero_check_significand (flt))
    {
      flt->exponent = precision - 1;

      /* Check MSB was set.  Setting it above quiets GCC.  */
      assert_cheap (msb != -1U);

      if (msb >= precision)
	{
	  lost_fraction = rescale_significand_right (flt, msb - precision);
	  if (lost_fraction == lf_exactly_half && remainder_non_zero (p))
	    lost_fraction = lf_more_than_half;
	  else if (lost_fraction == lf_exactly_zero && remainder_non_zero (p))
	    lost_fraction = lf_less_than_half;
	}
    }

  return lost_fraction;
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
