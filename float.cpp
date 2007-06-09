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

  /* Number of bits of precision in the IEEE 754 sense; this should
     include the integer bit whether implicit or explicit.  */
  unsigned char precision;
} all_semantics [] = {
  /* fsem_ieee_single */
  { 127, -126, 24 },
  /* fsem_ieee_double */
  { 1023, -1022, 53 },
  /* fsem_ieee_double_extended */
  { 16383, -16382, 64 },
  /* fsem_read_decimal */
  { 16383, -16382, 88 },
  /* fsem_power_ten */
  { 16383, -16382, 95 },
};

const flt_semantics &
t_float::semantics_for_kind (e_semantics_kind kind)
{
  return all_semantics[(int) kind];
}

unsigned int
t_float::part_count_for_kind (e_semantics_kind kind)
{
  return 1 + (semantics_for_kind (kind).precision / t_integer_part_width);
}

/* Constructors.  */
void
t_float::initialize (e_semantics_kind semantics_kind)
{
  unsigned int count;

  count = part_count_for_kind (semantics_kind);

  kind = semantics_kind;
  sign = 0;
  category = fc_zero;
  exponent = 0;
  is_wide = (count > 1);

  if (is_wide)
    significand.parts = new t_integer_part[count];
}

t_float::t_float (e_semantics_kind kind, t_integer_part value,
		  e_rounding_mode rounding_mode, e_status *status)
{
  initialize (kind);
  zero_significand ();
  exponent = semantics_for_kind (kind).precision - 1;
  sig_parts_array ()[0] = value;
  *status = normalize (rounding_mode, lf_exactly_zero);
}

t_float::t_float (e_semantics_kind kind, e_category c, bool negative)
{
  initialize (kind);
  category = c;
  sign = negative;
  if (category == fc_normal)
    category = fc_zero;
}

t_float::t_float (const t_float &rhs)
{
  unsigned int i, parts_count;
  const t_integer_part *src;
  t_integer_part *dst;

  initialize (rhs.kind);
  sign = rhs.sign;
  category = rhs.category;
  exponent = rhs.exponent;

  dst = sig_parts_array ();
  src = rhs.sig_parts_array ();
  parts_count = part_count_for_kind (kind);
  for (i = 0; i < parts_count; i++)
    dst[i] = src[i];
}

t_float::~t_float ()
{
  if (is_wide)
    delete [] significand.parts;
}

const t_integer_part *
t_float::sig_parts_array () const
{
  if (is_wide)
    return significand.parts;
  else
    return &significand.part;
}

t_integer_part *
t_float::sig_parts_array ()
{
  if (is_wide)
    return significand.parts;
  else
    return &significand.part;
}

bool
t_float::normalize_zeroes ()
{
  assert (category == fc_normal);

  if (APInt::tc_is_zero (sig_parts_array (), part_count_for_kind (kind)))
    {
      category = fc_zero;
      return true;
    }

  return false;
}

void
t_float::zero_significand ()
{
  APInt::tc_set (sig_parts_array (), 0, part_count_for_kind (kind));
  category = fc_normal;
}

/* Increment a floating point number's significand.  */
void
t_float::increment_significand ()
{
  t_integer_part carry;

  if (category == fc_zero)
    zero_significand ();

  assert (category == fc_normal);

  carry = APInt::tc_increment (sig_parts_array (), part_count_for_kind (kind));

  /* Our callers should never cause us to overflow.  */
  assert (carry == 0);
}

/* Shift the significand left BITS bits, subtract BITS from its exponent.  */
void
t_float::logical_left_shift_significand (unsigned int bits)
{
  assert (bits < semantics_for_kind (kind).precision);

  if (bits)
    {
      t_integer_part *parts;

      parts = sig_parts_array ();
      APInt::tc_left_shift (parts, parts, part_count_for_kind (kind), bits);
      exponent -= bits;

      assert (!normalize_zeroes ());
    }
}

/* Adds the significand of the RHS.  Returns 1 if there was a carry.  */
t_integer_part
t_float::add_significand (const t_float &rhs)
{
  t_integer_part *parts;

  parts = sig_parts_array ();

  assert (kind == rhs.kind);
  assert (category == fc_normal);
  assert (rhs.category == fc_normal);
  assert (exponent == rhs.exponent);

  return APInt::tc_add (parts, const_cast<const t_integer_part *>(parts),
			rhs.sig_parts_array (), 0,
			part_count_for_kind (kind));
}

/* Subtract the significand of RHS.  Returns the borrow flag.  */
t_integer_part
t_float::subtract_significand (const t_float &rhs, t_integer_part carry)
{
  t_integer_part *parts;

  parts = sig_parts_array ();

  assert (kind == rhs.kind);
  assert (category == fc_normal);
  assert (rhs.category == fc_normal);

  return APInt::tc_subtract (parts, const_cast<const t_integer_part *>(parts),
			     rhs.sig_parts_array (), carry,
			     part_count_for_kind (kind));
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
  assert (category == fc_normal);
  assert (rhs.category == fc_normal);

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

  precision = semantics_for_kind (kind).precision;
  if (msb > precision)
    {
      unsigned int bits, significant_parts;

      bits = msb - precision;
      significant_parts = ((msb + t_integer_part_width - 1)
			   / t_integer_part_width);
      lost_fraction = right_shift_lost_fraction
	(full_significand, significant_parts, bits);
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
  assert (category == fc_normal);
  assert (rhs.category == fc_normal);

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

  unsigned int precision = semantics_for_kind(kind).precision;

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
t_float::right_shift_lost_fraction (t_integer_part *dst, unsigned int parts,
				    unsigned int count)
{
  e_lost_fraction lost_fraction;

  /* Fast-path this trivial case.  It also ensures we don't have to
     worry about it for the count == lsb test below.  */
  if (count == 0)
    lost_fraction = lf_exactly_zero;
  else
    {
      unsigned int lsb;

      /* Before shifting see if we would lose precision.  */
      lsb = APInt::tc_lsb (dst, parts);

      if (count < lsb)
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

/* Note that a zero result is NOT normalized to fc_zero.  */
t_float::e_lost_fraction
t_float::rescale_significand_right (unsigned int bits)
{
  /* Our exponent should not overflow.  */
  assert ((exponent_t) (exponent + bits) >= exponent);

  exponent += bits;

  return right_shift_lost_fraction (sig_parts_array (),
				    part_count_for_kind (kind), bits);
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

  /* Before rounding normalize the exponent of fc_normal numbers.  */
  if (category == fc_normal)
    {
      unsigned int msb;
      int exponent_change;

      msb = significand_msb ();

      /* The MSB is numbered from 1.  We want to place it in the
	 integer bit numbered PRECISON if possible, with a
	 compensating change in the exponent.  */
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
	  e_lost_fraction new_lost_fraction;

	  /* Shift right and capture any new lost fraction.  */
	  new_lost_fraction = rescale_significand_right (exponent_change);
	  normalize_zeroes ();

	  /* Combine the effect of the lost fractions.  The newly lost
	     fraction is more significant than any prior one.  */
	  if (lost_fraction == lf_exactly_zero)
	    lost_fraction = new_lost_fraction;
	  else if (new_lost_fraction == lf_exactly_zero)
	    lost_fraction = lf_less_than_half;
	  else if (new_lost_fraction == lf_exactly_half)
	    lost_fraction = lf_more_than_half;
	}
    }

  /* Now round the number according to rounding_mode given the lost
     fraction in lost_fraction.  */

  /* As specified in IEEE 754, since we do not trap we do not report
     underflow for exact results.  */
  if (lost_fraction == lf_exactly_zero)
    return fs_ok;

  /* Increment the significand if we're rounding away from zero.  */
  if (round_away_from_zero (rounding_mode, lost_fraction))
    {
      /* FIXME: are we handling -0 correctly?  */
      if (category == fc_zero)
	{
	  zero_significand ();
	  exponent = our_semantics.min_exponent;
	}

      increment_significand ();
    }

  /* We now have the correctly-rounded value, but it might not be
     normalized.  Normalize fc_normal values.  */
  if (category == fc_normal)
    {
      unsigned int msb;

      msb = significand_msb ();

      /* The normal case - we were and are not denormal, and any
	 significand increment above didn't overflow.  */
      if (msb == our_semantics.precision)
	return fs_inexact;
      
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

      /* We have a non-zero denormal.  */
      assert (msb < our_semantics.precision);
      assert (exponent == our_semantics.min_exponent);
    }

  /* The fc_zero case is a denormal that underflowed to zero.  */
  return (e_status) (fs_underflow | fs_inexact);
}

#if 0
/* Unfortunately for IEEE semantics a rounding mode is needed
   here.  */
t_float::e_arith_status
t_float::unnormalized_add_or_subtract (const t_float &rhs, bool subtract,
				       e_rounding_mode rounding_mode,
				       e_lost_fraction *lost_fraction)
{
  /* Canonicalize to SIGN ( this SUBTRACT rhs ) where sign and
     subtract are booleans, and we ignore the true signs of this and
     RHS, and assume for now that this > RHS.  */
  subtract ^= (sign ^ rhs.sign);
  *lost_fraction = lf_exactly_zero;

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
      set_kind (rhs.kind);
      return fs_ok;

    case convolve (fc_normal, fc_infinity):
    case convolve (fc_zero, fc_infinity):
    case convolve (fc_zero, fc_normal):
      set_kind (rhs.kind);
      sign ^= subtract;
      return fs_ok;

    case convolve (fc_zero, fc_zero):
      /* Canonical form -(0 + 0), such as -0 + -0.  */
      sign = sign && !subtract;
      return fs_ok;

    case convolve (fc_infinity, fc_infinity):
      if (subtract)
	{
	  category = fc_nan;
	  return fs_invalid_op;
	}
      return fs_ok;

    case convolve (fc_normal, fc_normal):
      break;
    }

  unsigned int parts_count;
  int bits;
  t_integer_part carry, scratch, *tmp_significand;
  t_float tmp_float;

  parts_count = sig_parts_count ();
  if (parts_count > 1)
    tmp_significand = new t_integer_part[parts_count];
  else
    tmp_significand = &scratch;

  /* Shift significands so LHS and RHS have same exponent.  */
  bits = exponent - rhs.exponent;

  if (bits >= 0)
    {
      tmp_float = *rhs;
      *lost_fraction = rescale_significand_right (&tmp_float, bits);
    }
  else
    {
      tmp_float = *lhs;
      *lost_fraction = rescale_significand_right (&tmp_float, -bits);

      lhs = rhs;
      sign ^= subtract;
    }

  /* Now the canonical form is SIGN (lhs SUBTRACT tmp_float) with the
     lost fraction to the "right" of tmp_float.  Add or subtract the
     significands.  */
  if (subtract)
    {
      /* Avoid carry.  */
      if (compare_absolute_values (lhs, &tmp_float) == fcmp_greater_than)
	{
	  carry = subtract_significands (dst, lhs, &tmp_float,
					 *lost_fraction != lf_exactly_zero);

	  /* Correct the lost fraction - it belonged to the RHS of the
	     subtraction.  */
	  if (*lost_fraction == lf_less_than_half)
	    *lost_fraction = lf_more_than_half;
	  else if (*lost_fraction == lf_more_than_half)
	    *lost_fraction = lf_less_than_half;
	}
      else
	{
	  carry = subtract_significands (dst, &tmp_float, lhs, false);
	  sign = !sign;
	}
    }
  else
    carry = add_significands (dst, lhs, &tmp_float);

  /* By design we should not carry, but use the guard bit.  */
  assert (!carry);

  if (parts_count > 1)
    delete [] tmp_significand;

  dst->exponent = tmp_float.exponent;

  /* The case of two zeroes is correctly handled above.  Otherwise, if
     two numbers add to zero, IEEE 754 decrees it is a positive zero
     unless rounding to minus infinity.  */
  if (canonicalize_zeroes ())
    sign = (rounding_mode == frm_to_minus_infinity);

  return fs_ok;
}
#endif

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

  exponent += rhs.exponent - (semantics_for_kind (kind).precision - 1);
  *lost_fraction = multiply_significand (rhs);

  /* Canonicalize zeroes.  */
  normalize_zeroes ();

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

  /* Canonicalize zeroes.  */
  normalize_zeroes ();

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
  exponent += (semantics_for_kind (to_kind).precision
	       - semantics_for_kind (kind).precision);
  kind = to_kind;

  return normalize (rounding_mode, lf_exactly_zero);
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

static e_arith_status
float_add (t_float *dst, const t_float *lhs, const t_float *rhs,
	   bool subtract, bool round_to_minus_infinity,
	   e_lost_fraction *lost_fraction)
{
  bool sign;
  int bits;
  c_part carry;
  t_float tmp_float;

  /* Canonicalize to SIGN ( lhs SUBTRACT rhs ) where sign and subtract
     are booleans, and we ignore the true signs of LHS and RHS, and
     assume for now that LHS > RHS.  */
  sign = lhs->sign;
  subtract = subtract ^ (lhs->sign ^ rhs->sign);
  *lost_fraction = lf_exactly_zero;

  switch (convolve (lhs->category, rhs->category))
    {
    case convolve (fc_nan, fc_zero):
    case convolve (fc_nan, fc_normal):
    case convolve (fc_nan, fc_infinity):
    case convolve (fc_nan, fc_nan):
    case convolve (fc_normal, fc_zero):
    case convolve (fc_infinity, fc_normal):
    case convolve (fc_infinity, fc_zero):
      *dst = *lhs;
      return as_ok;

    case convolve (fc_zero, fc_nan):
    case convolve (fc_normal, fc_nan):
    case convolve (fc_infinity, fc_nan):
      *dst = *rhs;
      return as_ok;

    case convolve (fc_normal, fc_infinity):
    case convolve (fc_zero, fc_infinity):
    case convolve (fc_zero, fc_normal):
      *dst = *rhs;
      dst->sign = subtract ^ sign;
      return as_ok;

    case convolve (fc_zero, fc_zero):
      /* Canonical form -(0 + 0), such as -0 + -0.  */
      dst->category = fc_zero;
      dst->sign = sign && !subtract;
      return as_ok;

    case convolve (fc_infinity, fc_infinity):
      if (subtract)
	{
	  dst->category = fc_nan;
	  return as_flt_invalid_op;
	}

      dst->category = fc_infinity;
      dst->sign = sign;
      return as_ok;

    case convolve (fc_normal, fc_normal):
      break;

    default:
      assert_unreachable ();
    }

  /* Shift significands so LHS and RHS have same exponent.  */
  bits = exponent - rhs.exponent;

  if (bits >= 0)
    {
      tmp_float = *rhs;
      *lost_fraction = rescale_significand_right (&tmp_float, bits);
    }
  else
    {
      tmp_float = *lhs;
      *lost_fraction = rescale_significand_right (&tmp_float, -bits);

      lhs = rhs;
      sign ^= subtract;
    }

  /* Now the canonical form is SIGN (lhs SUBTRACT tmp_float) with the
     lost fraction to the "right" of tmp_float.  Add or subtract the
     significands.  */
  if (subtract)
    {
      /* Avoid carry.  */
      if (compare_absolute_values (lhs, &tmp_float) == fcmp_greater_than)
	{
	  carry = subtract_significands (dst, lhs, &tmp_float,
					 *lost_fraction != lf_exactly_zero);

	  /* Correct the lost fraction - it belonged to the RHS of the
	     subtraction.  */
	  if (*lost_fraction == lf_less_than_half)
	    *lost_fraction = lf_more_than_half;
	  else if (*lost_fraction == lf_more_than_half)
	    *lost_fraction = lf_less_than_half;
	}
      else
	{
	  carry = subtract_significands (dst, &tmp_float, lhs, false);
	  sign = !sign;
	}
    }
  else
    carry = add_significands (dst, lhs, &tmp_float);

  /* By design we should not carry, but use the guard bit.  */
  assert (!carry);

  dst->exponent = tmp_float.exponent;
  dst->category = fc_normal;
  dst->sign = sign;

  /* The case of two zeroes is correctly handled above.  Otherwise, if
     two numbers add to zero, IEEE 754 decrees it is a positive zero
     unless rounding to minus infinity.  */
  if (zero_check_significand (dst))
    dst->sign = round_to_minus_infinity;

  return as_ok;
}

e_arith_status
t_float_add (t_float *dst, const t_float *lhs, const t_float *rhs,
	     const c_float_semantics *semantics)
{
  e_arith_status as;
  e_lost_fraction lost_fraction;

  as = float_add (dst, lhs, rhs, false,
		  semantics->rounding_mode == rm_to_minus_infinity,
		  &lost_fraction);

  /* We return normalized numbers.  */
  as |= semantically_normalize (dst, semantics, lost_fraction);

  return as;
}

e_arith_status
t_float_sub (t_float *dst, const t_float *lhs, const t_float *rhs,
	     const c_float_semantics *semantics)
{
  e_arith_status as;
  e_lost_fraction lost_fraction;

  as = float_add (dst, lhs, rhs, true,
		  semantics->rounding_mode == rm_to_minus_infinity,
		  &lost_fraction);

  /* We return normalized numbers.  */
  as |= semantically_normalize (dst, semantics, lost_fraction);

  return as;
}

/* Convert a floating point number to an integer.  The C standard
   requires this to round to zero regardless of the rounding mode.  If
   the rounded integer value is out of range, this should raise the
   invalid operation exception.  If the rounded value is in range but
   the floating point number is not the exact integer, the C standard
   doesn't require an inexact exception to be raised.  IEEE 854 does
   require it so we do that for quality-of-implementation.  */
e_arith_status
t_float_to_int_convert (const t_float *flt, t_integer *value,
			unsigned int width, bool is_signed,
			const c_float_semantics *float_semantics)
{
  e_lost_fraction lost_fraction;
  t_float tmp;
  unsigned int target_width;
  int count;

  /* Put something there.  */
  *value = integer_zero;

  if (flt->category == fc_infinity || flt->category == fc_nan)
    return as_flt_invalid_op;

  if (flt->category == fc_zero)
    return as_ok;

  assert_cheap (flt->category == fc_normal);

  /* Catch abs (flt) < 1 case.  */
  if (flt->exponent < 0)
    return as_flt_inexact;

  /* Remaining negative numbers cannot be represented as unsigned.  */
  if (flt->sign && !is_signed)
    return as_flt_invalid_op;

  /* It takes flt->exponent + 1 bits to represent the truncated
     floating point number without its sign.  We lose a bit for the
     sign, but the maximally negative integer is a special case; it
     takes the full width to represent as an unsigned number.  Catch
     the negative number issues below.  */
  target_width = width;
  if (is_signed && !flt->sign)
    target_width--;

  if (flt->exponent >= target_width)
    return as_flt_invalid_op;

  /* We want to shift the bit pattern so that we lose all decimal
     places.  */
  tmp = *flt;
  count = float_semantics->precision - 1 - flt->exponent;

  if (count > 0)
    lost_fraction = rescale_significand_right (&tmp, count);
  else
    {
      logical_lshift_significand (&tmp, -count);
      lost_fraction = lf_exactly_zero;
    }

  /* We shouldn't have become zero.  */
  assert_cheap (float_lsb (&tmp) != 0);

  /* Now catch the tricky signed case - amongst the set of numbers
     with exponent + 1 the target width, only the one with just the
     msb set can be represented.  */
  if (is_signed && flt->exponent + 1 == width && float_lsb (&tmp) != width)
    return as_flt_invalid_op;

  tc_assign (integer_parts (value), tmp.significand, t_integer_part_count);

  if (flt->sign)
    integer_change_sign (value, value, width, false);

  if (lost_fraction == lf_exactly_zero)
    return as_ok;
  else
    return as_flt_inexact;
}

e_arith_status
t_int_to_float_convert (t_float *flt, const t_integer *value,
			bool is_signed, const c_float_semantics *semantics)
{
  t_integer absolute;
  unsigned int width_required, parts;

  absolute = *value;
  flt->sign = false;

  if (integer_compare (value, &integer_zero, is_signed) < 0)
    {
      integer_change_sign (&absolute, &absolute, t_integer_width, is_signed);
      flt->sign = true;
    }

  width_required = integer_width (&absolute);

  /* If it's too big, we give up immediately.  Note that this codepath
     is not currently exercised as none of our integer types are large
     enough.  */
  if (width_required > t_float_width)
    {
      flt->category = fc_infinity;
      return as_flt_overflow | as_flt_inexact;
    }

  flt->category = fc_normal;
  flt->exponent = semantics->precision - 1;

  /* Copy the bit image.  */
  parts = (width_required + c_part_width - 1) / c_part_width;
  tc_assign (flt->significand, integer_parts (&absolute), parts);
  while (parts < tf_parts)
    flt->significand[parts++] = 0;
  zero_check_significand (flt);

  /* Return a normalized result.  */
  return semantically_normalize (flt, semantics, lf_exactly_zero);
}

e_arith_status
t_float_constant (t_float *flt, t_integer_part value,
		  const c_float_semantics *semantics)
{
  unsigned int i;

  flt->sign = 0;
  flt->category = fc_normal;
  flt->exponent = semantics->precision - 1;
  flt->significand[0] = value;
  for (i = 1; i < tf_parts; i++)
    flt->significand[i] = 0;

  /* Return a normalized result.  */
  return semantically_normalize (flt, semantics, lf_exactly_zero);
}


bool
t_float_eq (const t_float *lhs, const t_float *rhs)
{
  return t_float_compare (lhs, rhs) == fcmp_equal;
}

bool
t_float_ne (const t_float *lhs, const t_float *rhs)
{
  return t_float_compare (lhs, rhs) != fcmp_equal;
}

bool
t_float_zr (const t_float *flt)
{
  return flt->category == fc_zero;
}

bool
t_float_nz (const t_float *flt)
{
  return !t_float_zr (flt);
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
      unsigned int digit_value;

      digit_value = host_digit_value (*p);
      if (digit_value == -1U)
	break;

      p++;
      unsigned_exponent = unsigned_exponent * 10 + digit_value;
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

/* Skip leading zeroes.  We know from the syntax that floating point
   strings cannot be merely zeroes, so we don't need to worry about
   going off the end.  */
static const char *
skip_leading_zeroes (const char *p)
{
  while (*p == '0' || *p == '.')
    p++;

  return p;
}

/* Returns true iff exact.  */
static bool
read_hexadecimal_significand (t_float *flt, const char *p)
{
  unsigned int bit_pos;

  zero_significand (flt);

  bit_pos = t_float_width;

  for (;;)
    {
      t_integer_part hex_value;

      if (*p == '.')
	p++;

      hex_value = host_hex_value (*p);
      if (hex_value == -1U)
	return true;

      p++;
 
      /* Store the number whilst 4-bit nibbles remain.  Otherwise scan
	 the remainder of the string: if we encounter a non-hex number
	 before a non-zero number it is exact, otherwise inexact.  */
      if (bit_pos)
	{
	  bit_pos -= 4;
	  hex_value <<= bit_pos % c_part_width;
	  flt->significand[bit_pos / c_part_width] |= hex_value;
	}
      else if (hex_value != 0)
	return false;
    }
}

static e_arith_status
read_hexadecimal_float (t_float *flt, const c_number *number,
			const c_float_semantics *semantics)
{
  const char *p;
  bool exact;

  p = skip_leading_zeroes (number->pp_token->spelling + 2);
  exact = read_hexadecimal_significand (flt, p);

  if (!zero_check_significand (flt))
    {
      int exponent;

      exponent = digits_left_of_fraction (number, p);
      exponent = exponent * 4 - 1;
      exponent += semantics->precision - t_float_width;

      flt->category = fc_normal;
      flt->exponent = total_exponent (number->text.exponent, exponent);
    }

  flt->sign = 0;

  return semantically_normalize (flt, semantics,
				 exact ? lf_exactly_zero: lf_less_than_half);
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

e_arith_status
t_float_read (t_float *flt, const c_number *number,
	      const c_float_semantics *semantics)
{
  if (number->base == 16)
    return read_hexadecimal_float (flt, number, semantics);
  else
    return read_decimal_float (flt, number, semantics);
}
#endif
