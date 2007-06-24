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
compileTimeAssert (integerPartWidth % 4 == 0);

namespace llvm {

  /* Represents floating point arithmetic semantics.  */
  struct fltSemantics
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

    /* If the target format has an implicit integer bit.  */
    bool implicit_integer_bit;
  };

  struct decimal_number
  {
    integerPart *parts;
    unsigned int part_count;
    int exponent;
  };

  const fltSemantics APFloat::ieee_single = { 127, -126, 24, true };
  const fltSemantics APFloat::ieee_double = { 1023, -1022, 53, true };
  const fltSemantics APFloat::ieee_quad = { 16383, -16382, 113, true };
  const fltSemantics APFloat::x87_double_extended = { 16383, -16382, 64,
						      false };
}

/* Put a bunch of private, handy routines in an anonymous namespace.  */
namespace {

  inline unsigned int
  part_count_for_bits (unsigned int bits)
  {
    return ((bits) + integerPartWidth - 1) / integerPartWidth;
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
    integerPart unsigned_exponent;
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
  lost_fraction_through_truncation (integerPart *parts,
				    unsigned int part_count,
				    unsigned int bits)
  {
    unsigned int lsb;

    /* Fast-path two cases that would fail the generic logic.  */
    if (bits == 0 || (lsb = APInt::tcLSB (parts, part_count)) == 0)
      return lf_exactly_zero;

    if (bits < lsb)
      return lf_exactly_zero;
    if (bits == lsb)
      return lf_exactly_half;
    if (bits <= part_count * integerPartWidth
	&& APInt::tcExtractBit (parts, bits))
      return lf_more_than_half;

    return lf_less_than_half;
  }

  /* Shift DST right BITS bits noting lost fraction.  */
  e_lost_fraction
  shift_right (integerPart *dst, unsigned int parts, unsigned int bits)
  {
    e_lost_fraction lost_fraction;

    lost_fraction = lost_fraction_through_truncation (dst, parts, bits);

    APInt::tcShiftRight (dst, parts, bits);

    return lost_fraction;
  }
}

/* Constructors.  */
void
APFloat::initialize (const fltSemantics *our_semantics)
{
  unsigned int count;

  semantics = our_semantics;
  count = partCount ();
  if (count > 1)
    significand.parts = new integerPart[count];
}

void
APFloat::freeSignificand ()
{
  if (partCount () > 1)
    delete [] significand.parts;
}

void
APFloat::assign (const APFloat &rhs)
{
  assert (semantics == rhs.semantics);

  sign = rhs.sign;
  category = rhs.category;
  exponent = rhs.exponent;
  if (category == fcNormal)
    copySignificand (rhs);
}

void
APFloat::copySignificand (const APFloat &rhs)
{
  assert (category == fcNormal);
  assert (rhs.partCount () >= partCount ());

  APInt::tcAssign (significandParts(), rhs.significandParts(),
		    partCount ());
}

APFloat &
APFloat::operator= (const APFloat &rhs)
{
  if (this != &rhs)
    {
      if (semantics != rhs.semantics)
	{
	  freeSignificand ();
	  initialize (rhs.semantics);
	}
      assign (rhs);
    }

  return *this;
}

APFloat::APFloat (const fltSemantics &our_semantics, integerPart value)
{
  initialize (&our_semantics);
  sign = 0;
  zeroSignificand ();
  exponent = our_semantics.precision - 1;
  significandParts ()[0] = value;
  normalize (rmNearestTiesToEven, lf_exactly_zero);
}

APFloat::APFloat (const fltSemantics &our_semantics,
		  fltCategory our_category, bool negative)
{
  initialize (&our_semantics);
  category = our_category;
  sign = negative;
  if (category == fcNormal)
    category = fcZero;
}

APFloat::APFloat (const fltSemantics &our_semantics, const char *text)
{
  initialize (&our_semantics);
  convertFromString (text, rmNearestTiesToEven);
}

APFloat::APFloat (const APFloat &rhs)
{
  initialize (rhs.semantics);
  assign (rhs);
}

APFloat::~APFloat ()
{
  freeSignificand ();
}

unsigned int
APFloat::partCount () const
{
  return part_count_for_bits (semantics->precision + 1);
}

unsigned int
APFloat::semanticsPrecision (const fltSemantics &semantics)
{
  return semantics.precision;
}

const integerPart *
APFloat::significandParts () const
{
  return const_cast<APFloat *>(this)->significandParts ();
}

integerPart *
APFloat::significandParts ()
{
  assert (category == fcNormal);

  if (partCount () > 1)
    return significand.parts;
  else
    return &significand.part;
}

/* Combine the effect of two lost fractions.  */
e_lost_fraction
APFloat::combineLostFractions (e_lost_fraction more_significant,
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
APFloat::zeroSignificand ()
{
  category = fcNormal;
  APInt::tcSet (significandParts (), 0, partCount ());
}

/* Increment an fcNormal floating point number's significand.  */
void
APFloat::incrementSignificand ()
{
  integerPart carry;

  carry = APInt::tcIncrement (significandParts (), partCount ());

  /* Our callers should never cause us to overflow.  */
  assert (carry == 0);
}

/* Add the significand of the RHS.  Returns the carry flag.  */
integerPart
APFloat::addSignificand (const APFloat &rhs)
{
  integerPart *parts;

  parts = significandParts ();

  assert (semantics == rhs.semantics);
  assert (exponent == rhs.exponent);

  return APInt::tcAdd (parts, rhs.significandParts (), 0, partCount ());
}

/* Subtract the significand of the RHS with a borrow flag.  Returns
   the borrow flag.  */
integerPart
APFloat::subtractSignificand (const APFloat &rhs, integerPart borrow)
{
  integerPart *parts;

  parts = significandParts ();

  assert (semantics == rhs.semantics);
  assert (exponent == rhs.exponent);

  return APInt::tcSubtract (parts, rhs.significandParts (), borrow,
			     partCount ());
}

/* Multiply the significand of the RHS.  If ADDEND is non-NULL, add it
   on to the full-precision result of the multiplication.  Returns the
   lost fraction.  */
e_lost_fraction
APFloat::multiplySignificand (const APFloat &rhs, const APFloat *addend)
{
  unsigned int msb, parts_count, new_parts_count, precision;
  integerPart *lhs_significand;
  integerPart scratch[4];
  integerPart *full_significand;
  e_lost_fraction lost_fraction;

  assert (semantics == rhs.semantics);

  precision = semantics->precision;
  new_parts_count = part_count_for_bits (precision * 2);

  if (new_parts_count > 4)
    full_significand = new integerPart[new_parts_count];
  else
    full_significand = scratch;

  lhs_significand = significandParts();
  parts_count = partCount ();

  APInt::tcFullMultiply (full_significand, lhs_significand,
			 rhs.significandParts (), parts_count);

  lost_fraction = lf_exactly_zero;
  msb = APInt::tcMSB (full_significand, new_parts_count);
  exponent += rhs.exponent;

  /* This must be true because our input was normalized.  We rely on
     this if ADDEND is not NULL.  */
  assert (msb >= precision);

  if (addend)
    {
      Significand saved_significand = significand;
      const fltSemantics *saved_semantics = semantics;
      fltSemantics extended_semantics;
      unsigned int new_msb;
      opStatus status;

      /* Create new semantics with precision that of our MSB.  That
	 way only ADDEND and not THIS needs to be normalized.  */
      extended_semantics = *semantics;
      extended_semantics.precision = msb;

      if (new_parts_count == 1)
	significand.part = full_significand[0];
      else
	significand.parts = full_significand;
      semantics = &extended_semantics;

      APFloat extended_addend (*addend);
      status = extended_addend.convert (extended_semantics, rmTowardZero);
      assert (status == opOK);
      lost_fraction = addOrSubtractSignificand (extended_addend, false);

      /* Restore our state.  */
      if (new_parts_count == 1)
	full_significand[0] = significand.part;
      significand = saved_significand;
      semantics = saved_semantics;

      msb = APInt::tcMSB (full_significand, new_parts_count);
    }

  exponent -= (precision - 1);

  if (msb > precision)
    {
      unsigned int bits, significant_parts;
      e_lost_fraction lf;

      bits = msb - precision;
      significant_parts = part_count_for_bits (msb);
      lf = shift_right (full_significand, significant_parts, bits);
      lost_fraction = combineLostFractions (lf, lost_fraction);
      exponent += bits;
    }

  APInt::tcAssign (lhs_significand, full_significand, parts_count);

  if (new_parts_count > 4)
    delete [] full_significand;

  return lost_fraction;
}

/* Multiply the significands of LHS and RHS to DST.  */
e_lost_fraction
APFloat::divideSignificand (const APFloat &rhs)
{
  unsigned int bit, i, parts_count;
  const integerPart *rhs_significand;
  integerPart *lhs_significand, *dividend, *divisor;
  integerPart scratch[4];
  e_lost_fraction lost_fraction;

  assert (semantics == rhs.semantics);

  lhs_significand = significandParts();
  rhs_significand = rhs.significandParts();
  parts_count = partCount ();

  if (parts_count > 2)
    dividend = new integerPart[parts_count * 2];
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
  bit = precision - APInt::tcMSB (divisor, parts_count);
  if (bit)
    {
      exponent += bit;
      APInt::tcShiftLeft (divisor, parts_count, bit);
    }

  /* Normalize the dividend.  */
  bit = precision - APInt::tcMSB (dividend, parts_count);
  if (bit)
    {
      exponent -= bit;
      APInt::tcShiftLeft (dividend, parts_count, bit);
    }

  if (APInt::tcCompare (dividend, divisor, parts_count) < 0)
    {
      exponent--;
      APInt::tcShiftLeft (dividend, parts_count, 1);
      assert (APInt::tcCompare (dividend, divisor, parts_count) >= 0);
    }

  /* Long division.  */
  for (bit = precision; bit; bit -= 1)
    {
      if (APInt::tcCompare (dividend, divisor, parts_count) >= 0)
	{
	  APInt::tcSubtract (dividend, divisor, 0, parts_count);
	  APInt::tcSetBit (lhs_significand, bit);
	}

      APInt::tcShiftLeft (dividend, parts_count, 1);
    }

  /* Figure out the lost fraction.  */
  int cmp = APInt::tcCompare (dividend, divisor, parts_count);

  if (cmp > 0)
    lost_fraction = lf_more_than_half;
  else if (cmp == 0)
    lost_fraction = lf_exactly_half;
  else if (APInt::tcIsZero (dividend, parts_count))
    lost_fraction = lf_exactly_zero;
  else
    lost_fraction = lf_less_than_half;

  if (parts_count > 2)
    delete [] dividend;

  return lost_fraction;
}

unsigned int
APFloat::significandMSB () const
{
  return APInt::tcMSB (significandParts (), partCount ());
}

unsigned int
APFloat::significandLSB () const
{
  return APInt::tcLSB (significandParts (), partCount ());
}

/* Note that a zero result is NOT normalized to fcZero.  */
e_lost_fraction
APFloat::shiftSignificandRight (unsigned int bits)
{
  /* Our exponent should not overflow.  */
  assert ((exponent_t) (exponent + bits) >= exponent);

  exponent += bits;

  return shift_right (significandParts (), partCount (), bits);
}

/* Shift the significand left BITS bits, subtract BITS from its exponent.  */
void
APFloat::shiftSignificandLeft (unsigned int bits)
{
  assert (bits < semantics->precision);

  if (bits)
    {
      unsigned int parts_count = partCount ();

      APInt::tcShiftLeft (significandParts (), parts_count, bits);
      exponent -= bits;

      assert (!APInt::tcIsZero (significandParts (), parts_count));
    }
}

APFloat::cmpResult
APFloat::compareAbsoluteValue (const APFloat &rhs) const
{
  int compare;

  assert (semantics == rhs.semantics);
  assert (category == fcNormal);
  assert (rhs.category == fcNormal);

  compare = exponent - rhs.exponent;

  /* If exponents are equal, do an unsigned bignum comparison of the
     significands.  */
  if (compare == 0)
    compare = APInt::tcCompare (significandParts (), rhs.significandParts (),
				partCount ());

  if (compare > 0)
    return cmpGreaterThan;
  else if (compare < 0)
    return cmpLessThan;
  else
    return cmpEqual;
}

/* Handle overflow.  Sign is preserved.  We either become infinity or
   the largest finite number.  */
APFloat::opStatus
APFloat::handleOverflow (roundingMode rounding_mode)
{
  /* Infinity?  */
  if (rounding_mode == rmNearestTiesToEven
      || rounding_mode == rmNearestTiesToAway
      || (rounding_mode == rmTowardPositive && !sign)
      || (rounding_mode == rmTowardNegative && sign))
    {
      category = fcInfinity;
      return (opStatus) (opOverflow | opInexact);
    }

  /* Otherwise we become the largest finite number.  */
  category = fcNormal;
  exponent = semantics->max_exponent;
  APInt::tcSetLeastSignificantBits (significandParts (), partCount (),
				    semantics->precision);

  return opInexact;
}

/* This routine must work for fcZero of both signs, and fcNormal
   numbers.  */
bool
APFloat::roundAwayFromZero (roundingMode rounding_mode,
			    e_lost_fraction lost_fraction)
{
  /* NaNs and infinities should not have lost fractions.  */
  assert (category == fcNormal || category == fcZero);

  /* Our caller has already handled this case.  */
  assert (lost_fraction != lf_exactly_zero);

  switch (rounding_mode)
    {
    default:
      assert (0);

    case rmNearestTiesToAway:
      return (lost_fraction == lf_exactly_half
	      || lost_fraction == lf_more_than_half);

    case rmNearestTiesToEven:
      if (lost_fraction == lf_more_than_half)
	return true;

      /* Our zeroes don't have a significand to test.  */
      if (lost_fraction == lf_exactly_half && category != fcZero)
	return significandParts()[0] & 1;
	
      return false;

    case rmTowardZero:
      return false;

    case rmTowardPositive:
      return sign == false;

    case rmTowardNegative:
      return sign == true;
    }
}

APFloat::opStatus
APFloat::normalize (roundingMode rounding_mode,
		    e_lost_fraction lost_fraction)
{
  unsigned int msb;
  int exponent_change;

  if (category != fcNormal)
    return opOK;

  /* Before rounding normalize the exponent of fcNormal numbers.  */
  msb = significandMSB ();

  if (msb)
    {
      /* The MSB is numbered from 1.  We want to place it in the integer
	 bit numbered PRECISON if possible, with a compensating change in
	 the exponent.  */
      exponent_change = msb - semantics->precision;

      /* If the resulting exponent is too high, overflow according to
	 the rounding mode.  */
      if (exponent + exponent_change > semantics->max_exponent)
	return handleOverflow (rounding_mode);

      /* Subnormal numbers have exponent min_exponent, and their MSB
	 is forced based on that.  */
      if (exponent + exponent_change < semantics->min_exponent)
	exponent_change = semantics->min_exponent - exponent;

      /* Shifting left is easy as we don't lose precision.  */
      if (exponent_change < 0)
	{
	  assert (lost_fraction == lf_exactly_zero);

	  shiftSignificandLeft (-exponent_change);

	  return opOK;
	}

      if (exponent_change > 0)
	{
	  e_lost_fraction lf;

	  /* Shift right and capture any new lost fraction.  */
	  lf = shiftSignificandRight (exponent_change);

	  lost_fraction = combineLostFractions (lf, lost_fraction);

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
	category = fcZero;

      return opOK;
    }

  /* Increment the significand if we're rounding away from zero.  */
  if (roundAwayFromZero (rounding_mode, lost_fraction))
    {
      if (msb == 0)
	exponent = semantics->min_exponent;

      incrementSignificand ();
      msb = significandMSB ();

      /* Did the significand increment overflow?  */
      if (msb == semantics->precision + 1)
	{
	  /* Renormalize by incrementing the exponent and shifting our
	     significand right one.  However if we already have the
	     maximum exponent we overflow to infinity.  */
	  if (exponent == semantics->max_exponent)
	    {
	      category = fcInfinity;

	      return (opStatus) (opOverflow | opInexact);
	    }

	  shiftSignificandRight (1);

	  return opInexact;
	}
    }

  /* The normal case - we were and are not denormal, and any
     significand increment above didn't overflow.  */
  if (msb == semantics->precision)
    return opInexact;

  /* We have a non-zero denormal.  */
  assert (msb < semantics->precision);
  assert (exponent == semantics->min_exponent);

  /* Canonicalize zeroes.  */
  if (msb == 0)
    category = fcZero;

  /* The fcZero case is a denormal that underflowed to zero.  */
  return (opStatus) (opUnderflow | opInexact);
}

APFloat::opStatus
APFloat::addOrSubtractSpecials (const APFloat &rhs, bool subtract)
{
  switch (convolve (category, rhs.category))
    {
    default:
      assert (0);

    case convolve (fcNaN, fcZero):
    case convolve (fcNaN, fcNormal):
    case convolve (fcNaN, fcInfinity):
    case convolve (fcNaN, fcNaN):
    case convolve (fcNormal, fcZero):
    case convolve (fcInfinity, fcNormal):
    case convolve (fcInfinity, fcZero):
      return opOK;

    case convolve (fcZero, fcNaN):
    case convolve (fcNormal, fcNaN):
    case convolve (fcInfinity, fcNaN):
      category = fcNaN;
      return opOK;

    case convolve (fcNormal, fcInfinity):
    case convolve (fcZero, fcInfinity):
      category = fcInfinity;
      sign = rhs.sign ^ subtract;
      return opOK;

    case convolve (fcZero, fcNormal):
      assign (rhs);
      sign = rhs.sign ^ subtract;
      return opOK;

    case convolve (fcZero, fcZero):
      /* Sign depends on rounding mode; handled by caller.  */
      return opOK;

    case convolve (fcInfinity, fcInfinity):
      /* Differently signed infinities can only be validly
	 subtracted.  */
      if (sign ^ rhs.sign != subtract)
	{
	  category = fcNaN;
	  return opInvalidOp;
	}

      return opOK;

    case convolve (fcNormal, fcNormal):
      return opDivByZero;
    }
}

/* Add or subtract two normal numbers.  */
e_lost_fraction
APFloat::addOrSubtractSignificand (const APFloat &rhs, bool subtract)
{
  integerPart carry;
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
      APFloat temp_rhs (rhs);
      bool reverse;

      if (bits == 0)
	{
	  reverse = compareAbsoluteValue (temp_rhs) == cmpLessThan;
	  lost_fraction = lf_exactly_zero;
	}
      else if (bits > 0)
	{
	  lost_fraction = temp_rhs.shiftSignificandRight (bits - 1);
	  shiftSignificandLeft (1);
	  reverse = false;
	}
      else if (bits < 0)
	{
	  lost_fraction = shiftSignificandRight (-bits - 1);
	  temp_rhs.shiftSignificandLeft (1);
	  reverse = true;
	}

      if (reverse)
	{
	  carry = temp_rhs.subtractSignificand
	    (*this, lost_fraction != lf_exactly_zero);
	  copySignificand (temp_rhs);
	  sign = !sign;
	}
      else
	carry = subtractSignificand
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
	  APFloat temp_rhs (rhs);

	  lost_fraction = temp_rhs.shiftSignificandRight (bits);
	  carry = addSignificand (temp_rhs);
	}
      else
	{
	  lost_fraction = shiftSignificandRight (-bits);
	  carry = addSignificand (rhs);
	}

      /* We have a guard bit; generating a carry cannot happen.  */
      assert (!carry);
    }

  return lost_fraction;
}

APFloat::opStatus
APFloat::multiplySpecials (const APFloat &rhs)
{
  switch (convolve (category, rhs.category))
    {
    default:
      assert (0);

    case convolve (fcNaN, fcZero):
    case convolve (fcNaN, fcNormal):
    case convolve (fcNaN, fcInfinity):
    case convolve (fcNaN, fcNaN):
    case convolve (fcZero, fcNaN):
    case convolve (fcNormal, fcNaN):
    case convolve (fcInfinity, fcNaN):
      category = fcNaN;
      return opOK;

    case convolve (fcNormal, fcInfinity):
    case convolve (fcInfinity, fcNormal):
    case convolve (fcInfinity, fcInfinity):
      category = fcInfinity;
      return opOK;

    case convolve (fcZero, fcNormal):
    case convolve (fcNormal, fcZero):
    case convolve (fcZero, fcZero):
      category = fcZero;
      return opOK;

    case convolve (fcZero, fcInfinity):
    case convolve (fcInfinity, fcZero):
      category = fcNaN;
      return opInvalidOp;

    case convolve (fcNormal, fcNormal):
      return opOK;
    }
}

APFloat::opStatus
APFloat::divideSpecials (const APFloat &rhs)
{
  switch (convolve (category, rhs.category))
    {
    default:
      assert (0);

    case convolve (fcNaN, fcZero):
    case convolve (fcNaN, fcNormal):
    case convolve (fcNaN, fcInfinity):
    case convolve (fcNaN, fcNaN):
    case convolve (fcInfinity, fcZero):
    case convolve (fcInfinity, fcNormal):
    case convolve (fcZero, fcInfinity):
    case convolve (fcZero, fcNormal):
      return opOK;

    case convolve (fcZero, fcNaN):
    case convolve (fcNormal, fcNaN):
    case convolve (fcInfinity, fcNaN):
      category = fcNaN;
      return opOK;

    case convolve (fcNormal, fcInfinity):
      category = fcZero;
      return opOK;

    case convolve (fcNormal, fcZero):
      category = fcInfinity;
      return opDivByZero;

    case convolve (fcInfinity, fcInfinity):
    case convolve (fcZero, fcZero):
      category = fcNaN;
      return opInvalidOp;

    case convolve (fcNormal, fcNormal):
      return opOK;
    }
}

/* Change sign.  */
void
APFloat::changeSign ()
{
  /* Look mummy, this one's easy.  */
  sign = !sign;
}

/* Normalized addition or subtraction.  */
APFloat::opStatus
APFloat::addOrSubtract (const APFloat &rhs, roundingMode rounding_mode,
			bool subtract)
{
  opStatus fs;

  fs = addOrSubtractSpecials (rhs, subtract);

  /* This return code means it was not a simple case.  */
  if (fs == opDivByZero)
    {
      e_lost_fraction lost_fraction;

      lost_fraction = addOrSubtractSignificand (rhs, subtract);
      fs = normalize (rounding_mode, lost_fraction);

      /* Can only be zero if we lost no fraction.  */
      assert (category != fcZero || lost_fraction == lf_exactly_zero);
    }

  /* If two numbers add (exactly) to zero, IEEE 754 decrees it is a
     positive zero unless rounding to minus infinity, except that
     adding two like-signed zeroes gives that zero.  */
  if (category == fcZero)
    {
      if (rhs.category != fcZero || (sign == rhs.sign) == subtract)
	sign = (rounding_mode == rmTowardNegative);
    }      

  return fs;
}

/* Normalized addition.  */
APFloat::opStatus
APFloat::add (const APFloat &rhs, roundingMode rounding_mode)
{
  return addOrSubtract (rhs, rounding_mode, false);
}

/* Normalized subtraction.  */
APFloat::opStatus
APFloat::subtract (const APFloat &rhs, roundingMode rounding_mode)
{
  return addOrSubtract (rhs, rounding_mode, true);
}

/* Normalized multiply.  */
APFloat::opStatus
APFloat::multiply (const APFloat &rhs, roundingMode rounding_mode)
{
  opStatus fs;

  sign ^= rhs.sign;
  fs = multiplySpecials (rhs);

  if (category == fcNormal)
    {
      e_lost_fraction lost_fraction = multiplySignificand (rhs, 0);
      fs = normalize (rounding_mode, lost_fraction);
      if (lost_fraction != lf_exactly_zero)
	fs = (opStatus) (fs | opInexact);
    }

  return fs;
}

/* Normalized divide.  */
APFloat::opStatus
APFloat::divide (const APFloat &rhs, roundingMode rounding_mode)
{
  opStatus fs;

  sign ^= rhs.sign;
  fs = divideSpecials (rhs);

  if (category == fcNormal)
    {
      e_lost_fraction lost_fraction = divideSignificand (rhs);
      fs = normalize (rounding_mode, lost_fraction);
      if (lost_fraction != lf_exactly_zero)
	fs = (opStatus) (fs | opInexact);
    }

  return fs;
}

/* Normalized fused-multiply-add.  */
APFloat::opStatus
APFloat::fusedMultiplyAdd (const APFloat &multiplicand,
			   const APFloat &addend,
			   roundingMode rounding_mode)
{
  opStatus fs;

  /* Post-multiplication sign, before addition.  */
  sign ^= multiplicand.sign;

  /* If and only if all arguments are normal do we need to do an
     extended-precision calculation.  */
  if (category == fcNormal && multiplicand.category == fcNormal
      && addend.category == fcNormal)
    {
      e_lost_fraction lost_fraction;

      lost_fraction = multiplySignificand (multiplicand, &addend);
      fs = normalize (rounding_mode, lost_fraction);
      if (lost_fraction != lf_exactly_zero)
	fs = (opStatus) (fs | opInexact);

      /* If two numbers add (exactly) to zero, IEEE 754 decrees it is a
	 positive zero unless rounding to minus infinity, except that
	 adding two like-signed zeroes gives that zero.  */
      if (category == fcZero && sign != addend.sign)
	sign = (rounding_mode == rmTowardNegative);
    }
  else
    {
      fs = multiplySpecials (multiplicand);

      /* FS can only be opOK or opInvalidOp.  There is no more work
	 to do in the latter case.  The IEEE-754R standard says it is
	 implementation-defined in this case whether, if ADDEND is a
	 quiet NaN, we raise invalid op; this implementation does so.
	 
	 If we need to do the addition we can do so with normal
	 precision.  */
      if (fs == opOK)
	fs = addOrSubtract (addend, rounding_mode, false);
    }

  return fs;
}

/* Comparison requires normalized numbers.  */
APFloat::cmpResult
APFloat::compare (const APFloat &rhs) const
{
  cmpResult result;

  assert (semantics == rhs.semantics);

  switch (convolve (category, rhs.category))
    {
    default:
      assert (0);

    case convolve (fcNaN, fcZero):
    case convolve (fcNaN, fcNormal):
    case convolve (fcNaN, fcInfinity):
    case convolve (fcNaN, fcNaN):
    case convolve (fcZero, fcNaN):
    case convolve (fcNormal, fcNaN):
    case convolve (fcInfinity, fcNaN):
      return cmpUnordered;

    case convolve (fcInfinity, fcNormal):
    case convolve (fcInfinity, fcZero):
    case convolve (fcNormal, fcZero):
      if (sign)
	return cmpLessThan;
      else
	return cmpGreaterThan;

    case convolve (fcNormal, fcInfinity):
    case convolve (fcZero, fcInfinity):
    case convolve (fcZero, fcNormal):
      if (rhs.sign)
	return cmpGreaterThan;
      else
	return cmpLessThan;

    case convolve (fcInfinity, fcInfinity):
      if (sign == rhs.sign)
	return cmpEqual;
      else if (sign)
	return cmpLessThan;
      else
	return cmpGreaterThan;

    case convolve (fcZero, fcZero):
      return cmpEqual;      

    case convolve (fcNormal, fcNormal):
      break;
    }

  /* Two normal numbers.  Do they have the same sign?  */
  if (sign != rhs.sign)
    {
      if (sign)
	result = cmpLessThan;
      else
	result = cmpGreaterThan;
    }
  else
    {
      /* Compare absolute values; invert result if negative.  */
      result = compareAbsoluteValue (rhs);

      if (sign)
	{
	  if (result == cmpLessThan)
	    result = cmpGreaterThan;
	  else if (result == cmpGreaterThan)
	    result = cmpLessThan;
	}
    }

  return result;
}

APFloat::opStatus
APFloat::convert (const fltSemantics &to_semantics,
		  roundingMode rounding_mode)
{
  unsigned int new_part_count;
  opStatus fs;

  new_part_count = part_count_for_bits (to_semantics.precision + 1);

  /* If our new form is wider, re-allocate our bit pattern into wider
     storage.  */ 
  if (new_part_count > partCount ())
    {
      integerPart *new_parts;

      new_parts = new integerPart[new_part_count];
      APInt::tcSet (new_parts, 0, new_part_count);
      APInt::tcAssign (new_parts, significandParts (), partCount ());
      freeSignificand ();
      significand.parts = new_parts;
    }

  if (category == fcNormal)
    {
      /* Re-interpret our bit-pattern.  */
      exponent += to_semantics.precision - semantics->precision;
      semantics = &to_semantics;
      fs = normalize (rounding_mode, lf_exactly_zero);
    }
  else
    {
      semantics = &to_semantics;
      fs = opOK;
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
APFloat::opStatus
APFloat::convertToInteger (integerPart *parts, unsigned int width,
			   bool is_signed,
			   roundingMode rounding_mode) const
{
  e_lost_fraction lost_fraction;
  unsigned int msb, parts_count;
  int bits;

  /* Handle the three special cases first.  */
  if (category == fcInfinity || category == fcNaN)
    return opInvalidOp;

  parts_count = part_count_for_bits (width);

  if (category == fcZero)
    {
      APInt::tcSet (parts, 0, parts_count);
      return opOK;
    }

  /* Shift the bit pattern so the fraction is lost.  */
  APFloat tmp (*this);

  bits = (int) semantics->precision - 1 - exponent;

  if (bits > 0)
    lost_fraction = tmp.shiftSignificandRight (bits);
  else
    {
      tmp.shiftSignificandLeft (-bits);
      lost_fraction = lf_exactly_zero;
    }

  if (lost_fraction != lf_exactly_zero
      && tmp.roundAwayFromZero (rounding_mode, lost_fraction))
    tmp.incrementSignificand ();

  msb = tmp.significandMSB ();

  /* Negative numbers cannot be represented as unsigned.  */
  if (!is_signed && tmp.sign && msb)
    return opInvalidOp;

  /* It takes exponent + 1 bits to represent the truncated floating
     point number without its sign.  We lose a bit for the sign, but
     the maximally negative integer is a special case.  */
  if (msb > width)
    return opInvalidOp;

  if (is_signed && msb == width
      && (!tmp.sign || tmp.significandLSB () != msb))
    return opInvalidOp;

  APInt::tcAssign (parts, tmp.significandParts (), parts_count);

  if (tmp.sign)
    APInt::tcNegate (parts, parts_count);

  if (lost_fraction == lf_exactly_zero)
    return opOK;
  else
    return opInexact;
}

APFloat::opStatus
APFloat::convertFromUnsignedInteger (integerPart *parts,
				     unsigned int part_count,
				     roundingMode rounding_mode)
{
  unsigned int msb, precision;
  e_lost_fraction lost_fraction;

  msb = APInt::tcMSB (parts, part_count);
  precision = semantics->precision;

  category = fcNormal;
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
  zeroSignificand ();
  APInt::tcAssign (significandParts (), parts, part_count_for_bits (msb));

  return normalize (rounding_mode, lost_fraction);
}

APFloat::opStatus
APFloat::convertFromInteger (const integerPart *parts,
			     unsigned int part_count, bool is_signed,
			     roundingMode rounding_mode)
{
  unsigned int width;
  opStatus status;
  integerPart *copy;

  copy = new integerPart[part_count];
  APInt::tcAssign (copy, parts, part_count);

  width = part_count * integerPartWidth;

  sign = false;
  if (is_signed && APInt::tcExtractBit (parts, width))
    {
      sign = true;
      APInt::tcNegate (copy, part_count);
    }

  status = convertFromUnsignedInteger (copy, part_count, rounding_mode);
  delete [] copy;

  return status;
}

APFloat::opStatus
APFloat::convertFromHexadecimalString (const char *p,
				       roundingMode rounding_mode)
{
  e_lost_fraction lost_fraction;
  integerPart *significand;
  unsigned int bit_pos, parts_count;
  const char *dot, *first_significant_digit;

  zeroSignificand ();
  exponent = 0;
  category = fcNormal;

  significand = significandParts ();
  parts_count = partCount ();
  bit_pos = parts_count * integerPartWidth;

  /* Skip leading zeroes and any (hexa)decimal point.  */
  p = skip_leading_zeroes_and_any_dot (p, &dot);
  first_significant_digit = p;

  for (;;)
    {
      integerPart hex_value;

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
	  hex_value <<= bit_pos % integerPartWidth;
	  significand[bit_pos / integerPartWidth] |= hex_value;
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
      exp_adjustment -= parts_count * integerPartWidth;

      /* Adjust for the given exponent.  */
      exponent = total_exponent (p, exp_adjustment);
    }

  return normalize (rounding_mode, lost_fraction);
}

APFloat::opStatus
APFloat::convertFromString (const char *p, roundingMode rounding_mode)
{
  /* Handle a leading minus sign.  */
  if (*p == '-')
    sign = 1, p++;
  else
    sign = 0;

  if (p[0] == '0' && (p[1] == 'x' || p[1] == 'X'))
    return convertFromHexadecimalString (p + 2, rounding_mode);
  else
    assert (0);
}
