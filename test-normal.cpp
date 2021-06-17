/*
   Copyright 2007 Neil Booth.

   See the file "COPYING" for information about the copyright
   and warranty status of this software.
*/

#include <cassert>
#include <cstdio>
#include "APFloat.h"

using namespace llvm;

#define inexact APFloat::opInexact
#define underflow (APFloat::opStatus) \
	(APFloat::opUnderflow | APFloat::opInexact)
#define overflow (APFloat::opStatus) \
	(APFloat::opOverflow | APFloat::opInexact)

const fltSemantics *all_semantics[] = {
  &APFloat::IEEEsingle,
  &APFloat::IEEEdouble,
  &APFloat::IEEEquad,
  &APFloat::x87DoubleExtended,
};

static bool
compare (const APFloat &lhs, const APFloat &rhs)
{
  return (lhs.compare (rhs) == APFloat::cmpEqual
	  && lhs.isNegative() == rhs.isNegative() );
}

static bool
convertFromInteger_parts (integerPart *value, unsigned int count,
			    bool is_signed,
			    APFloat::roundingMode rounding_mode,
			    const fltSemantics &semantics, const char *a,
			    APFloat::opStatus status)
{
  APFloat number (semantics, APFloat::fcZero);
  APFloat result (semantics, a);

  if (number.convertFromSignExtendedInteger (value, count, is_signed,
                                             rounding_mode) != status)
    return false;

  return compare (number, result);
}

static bool
convertFromInteger_parts (integerPart *value, unsigned int count,
			    bool is_signed,
			    APFloat::roundingMode rounding_mode,
			    const fltSemantics &semantics,
			    const APFloat &result,
			    APFloat::opStatus status)
{
  APFloat number (semantics, APFloat::fcZero);

  if (number.convertFromSignExtendedInteger (value, count, is_signed,
                                             rounding_mode) != status)
    return false;

  return compare (number, result);
}

static bool
convertFromInteger (integerPart value, bool is_signed,
		      APFloat::roundingMode rounding_mode,
		      const fltSemantics &kind,
		      const char *a,
		      APFloat::opStatus status)
{
  APFloat number (kind, APFloat::fcZero);
  APFloat result (kind, a);

  if (number.convertFromSignExtendedInteger (&value, 1, is_signed,
                                             rounding_mode) != status)
    return false;

  return compare (number, result);
}

static bool
convertToInteger (const char *a, unsigned int width, bool is_signed,
		    APFloat::roundingMode rounding_mode,
		    const fltSemantics &kind,
		    integerPart result, APFloat::opStatus status)
{
  APFloat number (kind, a);
  integerPart part;

  if (number.convertToInteger
      (&part, width, is_signed, rounding_mode) != status)
    return false;

  return status == APFloat::opInvalidOp || part == result;
}

static bool
add (const char *a, const char *b, const char *c,
     APFloat::roundingMode rounding_mode,
     const fltSemantics &kind,
     APFloat::opStatus status)
{
  APFloat lhs (kind, a);
  APFloat rhs (kind, b);
  APFloat result (kind, c);

  if (lhs.add (rhs, rounding_mode) != status)
    return false;

  return compare (lhs, result);
}

static bool
add (const char *a, const char *b, const APFloat &result,
     APFloat::roundingMode rounding_mode,
     const fltSemantics &kind,
     APFloat::opStatus status)
{
  APFloat lhs (kind, a);
  APFloat rhs (kind, b);

  if (lhs.add (rhs, rounding_mode) != status)
    return false;

  return compare (lhs, result);
}

static bool
subtract (const char *a, const char *b, const char *c,
	  APFloat::roundingMode rounding_mode,
	  const fltSemantics &kind,
	  APFloat::opStatus status)
{
  APFloat lhs (kind, a);
  APFloat rhs (kind, b);
  APFloat result (kind, c);

  if (lhs.subtract (rhs, rounding_mode) != status)
    return false;

  return compare (lhs, result);
}

static bool
multiply (const char *a, const char *b, const char *c,
	  APFloat::roundingMode rounding_mode,
	  const fltSemantics &kind,
	  APFloat::opStatus status)
{
  APFloat lhs (kind, a);
  APFloat rhs (kind, b);
  APFloat result (kind, c);

  if (lhs.multiply (rhs, rounding_mode) != status)
    return false;

  return compare (lhs, result);
}

static bool
fma (const char *a, const char *b, const char *c, const char *d,
     APFloat::roundingMode rounding_mode, const fltSemantics &kind,
     APFloat::opStatus status)
{
  APFloat lhs (kind, a);
  APFloat multiplicand (kind, b);
  APFloat addend (kind, c);
  APFloat result (kind, d);

  if (lhs.fusedMultiplyAdd (multiplicand, addend, rounding_mode) != status)
    return false;

  return compare (lhs, result);
}

static bool
divide (const char *a, const char *b, const char *c,
	APFloat::roundingMode rounding_mode,
	const fltSemantics &kind,
	APFloat::opStatus status)
{
  APFloat lhs (kind, a);
  APFloat rhs (kind, b);
  APFloat result (kind, c);

  if (lhs.divide (rhs, rounding_mode) != status)
    return false;

  return compare (lhs, result);
}

#if 0
static bool
fmod (const char *a, const char *b, const char *c,
      const fltSemantics &kind)
{
  APFloat lhs (kind, a);
  APFloat rhs (kind, b);
  APFloat result (kind, c);

  if (lhs.fmod (rhs) != APFloat::opOK)
    return false;

  return compare (lhs, result);
}

static bool
remainder (const char *a, const char *b, const char *c,
	   const fltSemantics &kind)
{
  APFloat lhs (kind, a);
  APFloat rhs (kind, b);
  APFloat result (kind, c);

  if (lhs.remainder (rhs) != APFloat::opOK)
    return false;

  return compare (lhs, result);
}
#endif

static bool
compare (const char *a, const char *b, const fltSemantics &kind,
         APFloat::roundingMode rounding_mode)
{
  APFloat lhs (kind, APFloat::fcZero, false);
  APFloat rhs (kind, APFloat::fcZero, false);

  lhs.convertFromString(a, rounding_mode);
  rhs.convertFromString(b, rounding_mode);

  return compare (lhs, rhs);
}

int main (void)
{
  APFloat d_nan (APFloat::IEEEdouble, APFloat::fcNaN, false);
  APFloat d_pos_infinity (APFloat::IEEEdouble, APFloat::fcInfinity, false);
  APFloat d_neg_infinity (APFloat::IEEEdouble, APFloat::fcInfinity, true);
  APFloat f_pos_infinity (APFloat::IEEEsingle, APFloat::fcInfinity, false);
  APFloat f_neg_infinity (APFloat::IEEEsingle, APFloat::fcInfinity, true);
  APFloat f_pos_zero (APFloat::IEEEsingle, APFloat::fcZero, false);
  APFloat f_neg_zero (APFloat::IEEEsingle, APFloat::fcZero, true);

  /* Test floating-point exact divisions.  */
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      for (int j = 0; j < 4; j++)
	{
	  const fltSemantics &kind = *all_semantics[j];

	  assert (divide ("0x1f865cp0", "0x944p0", "0x367p0",
			  rm, kind, APFloat::opOK));
	  assert (divide ("0xFb320p-4", "-0xd9.4p2", "-0x25.0p1",
			  rm, kind, APFloat::opOK));

	  if (APFloat::semanticsPrecision (kind) >= 53)
	    assert (divide ("0x0.8b6064570fa168p800", "0x0.badeadf1p500",
			    "0x0.beefe8p300",rm, kind, APFloat::opOK));

	  if (APFloat::semanticsPrecision (kind) >= 62)
	    assert (divide ("-0x0.a84f5a4693e1a774p-2", "-0x0.badeadf1p-16000",
			    "0x0.39a4beadp16000",rm, kind, APFloat::opOK));
	}
    }

  /* Division leaves a fraction of one-half and an odd number.  */
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);
      const fltSemantics &kind = APFloat::IEEEsingle;

      if (rm == APFloat::rmNearestTiesToEven || rm == APFloat::rmTowardPositive)
	assert (divide ("0x1.000006p-126", "0x2p0", "0x1.000008p-127",
			rm, kind, underflow));
      else
	assert (divide ("0x1.000006p-126", "0x2p0", "0x1.000004p-127",
			rm, kind, underflow));
    }

  /* Same, but negative.  */
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);
      const fltSemantics &kind = APFloat::IEEEsingle;

      if (rm == APFloat::rmNearestTiesToEven
	  || rm == APFloat::rmTowardNegative)
	assert (divide ("0x1.000006p-126", "-0x2p0", "-0x1.000008p-127",
			rm, kind, underflow));
      else
	assert (divide ("0x1.000006p-126", "-0x2p0", "-0x1.000004p-127",
			rm, kind, underflow));
    }

  /* Division leaves a fraction of one-half and an even number.  */
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);
      const fltSemantics &kind = APFloat::IEEEsingle;

      if (rm != APFloat::rmTowardPositive)
	assert (divide ("0x1.000002p-126", "0x2p0", "0x1.000000p-127",
			rm, kind, underflow));
      else
	assert (divide ("0x1.000002p-126", "0x2p0", "0x1.000004p-127",
			rm, kind, underflow));
    }

  /* Same, but negative.  */
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);
      const fltSemantics &kind = APFloat::IEEEsingle;

      if (rm != APFloat::rmTowardNegative)
	assert (divide ("-0x1.000002p-126", "0x2p0", "-0x1.000000p-127",
			rm, kind, underflow));
      else
	assert (divide ("-0x1.000002p-126", "0x2p0", "-0x1.000004p-127",
			rm, kind, underflow));
    }

  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);
      const fltSemantics &kind = APFloat::IEEEdouble;

      bool up = (rm == APFloat::rmTowardPositive
		 || rm == APFloat::rmNearestTiesToEven);
      bool inf = (rm == APFloat::rmTowardPositive);

      assert( divide ("0x1.567abc234109ep156", "0x1.0478fedbca987p-56",
		      up ? "0x1.50994033508bcp+212": "0x1.50994033508bbp+212",
		      rm, kind, APFloat::opInexact));

      assert( multiply ("0x1.567abc234109ep156", "0x1.0478fedbca987p-56",
		      inf ? "0x1.5c76858fcf474p+100": "0x1.5c76858fcf473p+100",
		      rm, kind, APFloat::opInexact));
    }

  /* Test rounding of multiplication to zero.  */
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);
      const fltSemantics &kind = APFloat::IEEEsingle;

      if (rm == APFloat::rmTowardPositive)
	assert (multiply ("0x1.0p-100", "0x1.0p-100", "0x1.0p-149f",
			rm, kind, underflow));
      else
	assert (multiply ("0x1.0p-100", "0x1.0p-100", "0x0p0",
			rm, kind, underflow));

      if (rm == APFloat::rmTowardNegative)
	assert (multiply ("-0x1.0p-100", "0x1.0p-100", "-0x1.0p-149f",
			rm, kind, underflow));
      else
	assert (multiply ("-0x1.0p-100", "0x1.0p-100", "-0x0p0",
			rm, kind, underflow));
    }

  /* Test floating-point exact additions, same sign.  */
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      for (int j = 0; j < 4; j++)
	{
	  const fltSemantics &kind = *all_semantics[j];

	  assert (add ("-0x4p0", "-0x4p0", "-0x8p0",
		       rm, kind, APFloat::opOK));
	  assert (add ("0x1p5", "0x1.3456p1", "0x22.68acp0",
		       rm, kind, APFloat::opOK));
	  assert (add ("0x1.3456p1", "0x1p5", "0x22.68acp0",
		       rm, kind, APFloat::opOK));
	  assert (add ("0x1.000002p0", "0x0.000002p0", "0x1.000004p0",
		       rm, kind, APFloat::opOK));

 	  /* This case is exact except for IEEEsingle.  */
	  if (APFloat::semanticsPrecision (kind) > 24)
	    {
	      assert (add ("0x1.234562p0", "0x1.234562p-1", "0x1.b4e813p0",
			   rm, kind, APFloat::opOK));
	      assert (add ("0x1.234562p-1", "0x1.234562p0", "0x1.b4e813p0",
			   rm, kind, APFloat::opOK));
	    }
	  else if (rm == APFloat::rmTowardZero
		   || rm == APFloat::rmTowardNegative)
	    {
	      assert (add ("0x1.234562p0", "0x1.234562p-1", "0x1.b4e812p0",
			   rm, kind, APFloat::opInexact));
	      assert (add ("0x1.234562p-1", "0x1.234562p0", "0x1.b4e812p0",
			   rm, kind, APFloat::opInexact));
	    }
	  else
	    {
	      assert (add ("0x1.234562p0", "0x1.234562p-1", "0x1.b4e814p0",
			   rm, kind, APFloat::opInexact));
	      assert (add ("0x1.234562p-1", "0x1.234562p0", "0x1.b4e814p0",
			   rm, kind, APFloat::opInexact));
	    }
	}
    }

  /* Test rounding on floating-point additions, same sign.  These
     lose a fraction of exactly one half.  */
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);
      const fltSemantics &kind = APFloat::IEEEsingle;

      bool up = (rm == APFloat::rmTowardPositive
		 || rm == APFloat::rmNearestTiesToEven);
      bool inf = (rm == APFloat::rmTowardPositive);

      assert (add ("0x1.000002p0", "0x0.000001p0",
		   up ? "0x1.000004p0": "0x1.000002p0",
		   rm, kind, APFloat::opInexact));
      assert (add ("0x0.000001p0", "0x1.000002p0",
		   up ? "0x1.000004p0": "0x1.000002p0",
		   rm, kind, APFloat::opInexact));
      assert (add ("0x1.000000p0", "0x0.000001p0",
		   rm == APFloat::rmTowardPositive
		   ? "0x1.000002p0" : "0x1.000000p0",
		   rm, kind, APFloat::opInexact));
      assert (add ("0x0.000001p0", "0x1.000000p0",
		   rm == APFloat::rmTowardPositive
		   ? "0x1.000002p0" : "0x1.000000p0",
		   rm, kind, APFloat::opInexact));

      assert (add ("0x1.fffffep0", "0x0.00000201p0",
		   inf ? "0x2.000004p0" : "0x2.000000p0",
		   rm, kind, APFloat::opInexact));
      assert (add ("0x1.fffffep0", "0x0.00000300p0",
		   inf ? "0x2.000004p0" : "0x2.000000p0",
		   rm, kind, APFloat::opInexact));
      assert (add ("0x1.fffffep0", "0x0.00000301p0",
		   inf ? "0x2.000004p0" : "0x2.000000p0",
		   rm, kind, APFloat::opInexact));
      assert (add ("0x1.fffffep0", "0x0.00000401p0",
		   up ? "0x2.000004p0" : "0x2.000000p0",
		   rm, kind, APFloat::opInexact));
      assert (add ("0x1.fffffep0", "0x0.00000500p0",
		   up ? "0x2.000004p0" : "0x2.000000p0",
		   rm, kind, APFloat::opInexact));
      assert (add ("0x1.fffffep0", "0x0.00000501p0",
		   up ? "0x2.000004p0" : "0x2.000000p0",
		   rm, kind, APFloat::opInexact));
      assert (add ("0x1.fffffep0", "0x0.00000700p0",
		   inf ? "0x2.000008p0" : "0x2.000004p0",
		   rm, kind, APFloat::opInexact));

      assert (add ("-0x1.000006p-2", "0x1.000006p0",
		   inf ? "0x1.80000ap-1" : "0x1.800008p-1",
		   rm, kind, APFloat::opInexact));
      assert (add ("-0x1.000006p-3", "0x1.000006p0",
		   inf ? "0x1.c0000cp-1" : "0x1.c0000ap-1",
		   rm, kind, APFloat::opInexact));
      assert (add ("-0x1.000006p-4", "0x1.000006p0",
		   up ? "0x1.e0000cp-1" : "0x1.e0000ap-1",
		   rm, kind, APFloat::opInexact));

      assert (add ("0x1.000006p0", "-0x1.000006p-2",
		   inf ? "0x1.80000ap-1" : "0x1.800008p-1",
		   rm, kind, APFloat::opInexact));
      assert (add ("0x1.000006p0", "-0x1.000006p-3",
		   inf ? "0x1.c0000cp-1" : "0x1.c0000ap-1",
		   rm, kind, APFloat::opInexact));
      assert (add ("0x1.000006p0", "-0x1.000006p-4",
		   up ? "0x1.e0000cp-1" : "0x1.e0000ap-1",
		   rm, kind, APFloat::opInexact));

      assert (multiply ("0x1.2p-126", "0x1.abcdep-3",
			up ? "0x1.e147a0p-129": "0x1.e14790p-129",
			rm, kind, underflow));
      assert (multiply ("0x1.22p-126", "0x1.abcdep-2",
			up ? "0x1.e49f38p-128": "0x1.e49f30p-128",
			rm, kind, underflow));
      assert (multiply ("0x1.22p-126", "0x1.abcdep-3",
			inf ? "0x1.e49f4p-129": "0x1.e49f3p-129",
			rm, kind, underflow));
      assert (multiply ("0x1.221p-126", "0x1.abcdep-3",
			inf ? "0x1.e4ba0p-129": "0x1.e4b9fp-129",
			rm, kind, underflow));
      assert (multiply ("0x1.221p-126", "0x1.abcdep-4",
			up ? "0x1.e4ba0p-130": "0x1.e4b9ep-130",
			rm, kind, underflow));
    }

  /* DBL_MAX.  */
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);
      const fltSemantics &kind = APFloat::IEEEdouble;

      assert (add ("0x1.ffffffffffff0p1023", "0x0.fp975",
		   "0x1.fffffffffffffp1023", rm, kind, APFloat::opOK));
      assert (add ("0x0.fp975", "0x1.ffffffffffff0p1023",
		   "0x1.fffffffffffffp1023", rm, kind, APFloat::opOK));

      /* Sub-half overflow.  */
      if (rm == APFloat::rmTowardPositive)
	{
	  assert (add ("0x1.ffffffffffff0p1023", "0x0.f000000000001p975",
		       d_pos_infinity, rm, kind, overflow));
	  assert (add ("0x0.f000000000001p975", "0x1.ffffffffffff0p1023",
		       d_pos_infinity, rm, kind, overflow));
	}
      else
	{
	  assert (add ("0x1.ffffffffffff0p1023", "0x0.f000000000001p975",
		       "0x1.fffffffffffffp1023", rm, kind, inexact));
	  assert (add ("0x0.f000000000001p975", "0x1.ffffffffffff0p1023",
		       "0x1.fffffffffffffp1023", rm, kind, inexact));
	}

      /* Exactly half overflow.  */
      if (rm != APFloat::rmTowardPositive && rm != APFloat::rmTowardZero)
	{
	  assert (add ("-0x1.ffffffffffff0p1023", "-0x0.f8p975",
		       d_neg_infinity, rm, kind, overflow));
	  assert (add ( "-0x0.f8p975", "-0x1.ffffffffffff0p1023",
		       d_neg_infinity, rm, kind, overflow));
	}
      else
	{
	  assert (add ("-0x1.ffffffffffff0p1023", "-0x0.f8p975",
		       "-0x1.fffffffffffffp1023", rm, kind, inexact));
	  assert (add ("-0x0.f8p975", "-0x1.ffffffffffff0p1023",
		       "-0x1.fffffffffffffp1023", rm, kind, inexact));
	}

      /* Unit overflow.  */
      if (rm != APFloat::rmTowardNegative && rm != APFloat::rmTowardZero)
	assert (add ("0x1.ffffffffffff0p1023", "0x1.0p975",
		     d_pos_infinity, rm, kind, overflow));
      else
	assert (add ("0x1.ffffffffffff0p1023", "0x1.0p975",
		     "0x1.fffffffffffffp1023", rm, kind, inexact));

      if (rm != APFloat::rmTowardPositive && rm != APFloat::rmTowardZero)
	assert (add ("-0x1.0p975", "-0x1.ffffffffffff0p1023",
		     d_neg_infinity, rm, kind, overflow));
      else
	assert (add ("-0x1.0p975", "-0x1.ffffffffffff0p1023",
		     "-0x1.fffffffffffffp1023", rm, kind, inexact));

      /* Add the biggest doule to itself.  */
      if (rm == APFloat::rmTowardNegative || rm == APFloat::rmTowardZero)
	assert (add ("0x1.fffffffffffffp1023", "0x1.fffffffffffffp1023",
		     "0x1.fffffffffffffp1023", rm, kind, inexact));
      else
	assert (add ("0x1.fffffffffffffp1023", "0x1.fffffffffffffp1023",
		     d_pos_infinity, rm, kind, overflow));
    }

  /* Denormal addition, epsilon and complete loss of precision.  */
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);
      const fltSemantics &kind = APFloat::IEEEdouble;

      assert (add ("0x0.0000000000001p-1022", "0x0.0000000000001p-1022",
		   "0x0.0000000000001p-1021", rm, kind, APFloat::opOK));
      assert (add ("0x0.9000000000001p-1022", "0x0.8abcdef012345p-1022",
		   "0x1.1abcdef012346p-1022", rm, kind, APFloat::opOK));
      /* Two denormals add to a normal.  */
      assert (add ("0x0.8abcdef012345p-1022", "0x0.9000000000001p-1022",
		   "0x1.1abcdef012346p-1022", rm, kind, APFloat::opOK));

      /* Epsilon.  */
      assert (add ("0x1.ap0", "0x1.0p-52",
		   "0x1.a000000000001p0", rm, kind, APFloat::opOK));
      assert (add ("0x1.0p-52","0x1.ap0",
		   "0x1.a000000000001p0", rm, kind, APFloat::opOK));

      /* Loss of precision.  */
      if (rm == APFloat::rmTowardPositive)
	{
	  assert (add ("0x1.ap0", "0x1.0p-53",
		       "0x1.a000000000001p0", rm, kind, APFloat::opInexact));
	  assert (add ("0x1.0p-53","0x1.ap0",
		       "0x1.a000000000001p0", rm, kind, APFloat::opInexact));
	}
      else
	{
	  assert (add ("0x1.ap0", "0x1.0p-53",
		       "0x1.ap0", rm, kind, APFloat::opInexact));
	  assert (add ("0x1.0p-53","0x1.ap0",
		       "0x1.ap0", rm, kind, APFloat::opInexact));
	}
    }


  /* Subtraction, exact.  */
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      for (int j = 0; j < 4; j++)
	{
	  const fltSemantics &kind = *all_semantics[j];

	  assert (add ("-0x4p0", "0x5p0", "0x1p0",
		       rm, kind, APFloat::opOK));
	  assert (add ("0x5p0", "-0x4p0", "0x1p0",
		       rm, kind, APFloat::opOK));
	  assert (add ("-0x4p0", "0x8p0", "0x4p0",
		       rm, kind, APFloat::opOK));
	  assert (add ("0x8p0", "-0x4p0", "0x4p0",
		       rm, kind, APFloat::opOK));

	  assert (add ("-0x4p0", "0x3p0", "-0x1p0",
		       rm, kind, APFloat::opOK));
	  assert (add ("0x3p0", "-0x4p0", "-0x1p0",
		       rm, kind, APFloat::opOK));
	  assert (add ("0x4p0", "-0x3p0", "0x1p0",
		       rm, kind, APFloat::opOK));
	  assert (add ("-0x3p0", "0x4p0", "0x1p0",
		       rm, kind, APFloat::opOK));

	  assert (subtract ("0x4p0", "0x5p0", "-0x1p0",
			    rm, kind, APFloat::opOK));
	  assert (subtract ("0x5p0", "0x4p0", "0x1p0",
			    rm, kind, APFloat::opOK));
	  assert (subtract ("-0x4p0", "-0x8p0", "0x4p0",
			    rm, kind, APFloat::opOK));
	  assert (subtract ("0x8p0", "0x4p0", "0x4p0",
			    rm, kind, APFloat::opOK));

	  assert (subtract ("-0x4p0", "0x3p0", "-0x7p0",
			    rm, kind, APFloat::opOK));
	  assert (subtract ("0x3p0", "-0x4p0", "0x7p0",
			    rm, kind, APFloat::opOK));
	  assert (subtract ("0x4p0", "-0x3p0", "0x7p0",
			    rm, kind, APFloat::opOK));
	  assert (subtract ("-0x3p0", "0x4p0", "-0x7p0",
			    rm, kind, APFloat::opOK));
	}
    }

  /* Test positive / negative zero rule.  */
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      for (int j = 0; j < 4; j++)
	{
	  const fltSemantics &kind = *all_semantics[j];

  	  if (rm == APFloat::rmTowardNegative)
	    {
	      assert (subtract ("0x3p0", "0x3p0", "-0x0p0",
				rm, kind, APFloat::opOK));
	      assert (add ("-0x3p0", "0x3p0", "-0x0p0",
			   rm, kind, APFloat::opOK));
	    }
	  else
	    {
	      assert (subtract ("0x3p0", "0x3p0", "0x0p0",
				rm, kind, APFloat::opOK));
	      assert (add ("-0x3p0", "0x3p0", "0x0p0",
			   rm, kind, APFloat::opOK));
	    }
	}
    }

  /* Test conversion to integer.  */
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);
      integerPart tmp;

      assert (d_nan.convertToInteger (&tmp, 5, true, rm)
	      == APFloat::opInvalidOp);
      assert (d_pos_infinity.convertToInteger (&tmp, 5, true, rm)
	      == APFloat::opInvalidOp);
      assert (d_neg_infinity.convertToInteger (&tmp, 5, true, rm)
	      == APFloat::opInvalidOp);

      for (int j = 0; j < 4; j++)
	{
	  const fltSemantics &kind = *all_semantics[j];

	  assert (convertToInteger ("0x0p0", 5, false, rm, kind,
				      0, APFloat::opOK));
	  assert (convertToInteger ("-0x0p0", 5, false, rm, kind,
				      0, APFloat::opOK));
	  assert (convertToInteger ("0x0p0", 5, true, rm, kind,
				      0, APFloat::opOK));
	  assert (convertToInteger ("-0x0p0", 5, true, rm, kind,
				      0, APFloat::opOK));

	  assert (convertToInteger ("0x1p0", 5, true, rm, kind,
				      1, APFloat::opOK));
	  assert (convertToInteger ("0xfp0", 5, true, rm, kind,
				      15, APFloat::opOK));
	  assert (convertToInteger ("0x10p0", 5, true, rm, kind,
				      0, APFloat::opInvalidOp));
	  assert (convertToInteger ("0x1fp0", 5, false, rm, kind,
				      31, APFloat::opOK));
	  assert (convertToInteger ("0x20p0", 5, false, rm, kind,
				      0, APFloat::opInvalidOp));

	  assert (convertToInteger ("-0x1p0", 5, true, rm, kind,
				      -1, APFloat::opOK));
	  assert (convertToInteger ("-0x1p0", 5, false, rm, kind,
				      0, APFloat::opInvalidOp));
	  assert (convertToInteger ("-0xfp0", 5, true, rm, kind,
				      -15, APFloat::opOK));
	  assert (convertToInteger ("-0x10p0", 5, true, rm, kind,
				      -16, APFloat::opOK));
	  assert (convertToInteger ("0x11p0", 5, true, rm, kind,
				      0, APFloat::opInvalidOp));
	  assert (convertToInteger ("-15.5", 4, true, rm, kind,
				      0, APFloat::opInvalidOp));

	  assert (convertToInteger ("0x1p63", 64, false, rm, kind,
                                    1ULL << 63, APFloat::opOK));
	  assert (convertToInteger ("0x1p63", 64, true, rm, kind,
                                    0, APFloat::opInvalidOp));
	  assert (convertToInteger ("0x1p64", 64, false, rm, kind,
                                    0, APFloat::opInvalidOp));
          assert (convertToInteger ("-0x1p63", 64,
                                    true, rm, kind,
                                    1ULL << 63, APFloat::opOK));

	  for (int k = 0; k <= 1; k++)
	    {
	      assert (convertToInteger ("0x1p-1", 5, k, rm, kind,
					  rm == APFloat::rmTowardPositive
					  ? 1: 0, APFloat::opInexact));
	      assert (convertToInteger ("0x1p-2", 5, k, rm, kind,
					  rm == APFloat::rmTowardPositive
					  ? 1: 0, APFloat::opInexact));
	      assert (convertToInteger ("0x3p-2", 5, k, rm, kind,
					  rm == APFloat::rmTowardPositive
					  || rm == APFloat::rmNearestTiesToEven
					  ? 1: 0, APFloat::opInexact));
	      assert (convertToInteger ("0x3p-1", 5, k, rm, kind,
					  rm == APFloat::rmTowardPositive
					  || rm == APFloat::rmNearestTiesToEven
					  ? 2: 1, APFloat::opInexact));
	      assert (convertToInteger ("0x7p-2", 5, k, rm, kind,
					  rm == APFloat::rmTowardPositive
					  || rm == APFloat::rmNearestTiesToEven
					  ? 2: 1, APFloat::opInexact));
	    }

          if (&kind == &APFloat::IEEEquad)
            {
              if (rm == APFloat::rmTowardNegative
                  || rm == APFloat::rmNearestTiesToEven)
                assert (convertToInteger ("-0x1.ffffffffffffffffp62", 64,
                                          true, rm, kind,
                                          1ULL << 63, APFloat::opInexact));

              if (rm == APFloat::rmTowardNegative)
                {
                  assert (convertToInteger ("-0x1.8000000000000000p63", 64,
                                            true, rm, kind,
                                            0, APFloat::opInvalidOp));
                  assert (convertToInteger ("-0x1.0000000000000001p63", 64,
                                            true, rm, kind,
                                            0, APFloat::opInvalidOp));
                }

              if (rm == APFloat::rmTowardPositive
                  || rm == APFloat::rmNearestTiesToEven)
                {
                  assert (convertToInteger ("0x1.ffffffffffffffffp63", 64,
                                            false, rm, kind,
                                            0, APFloat::opInvalidOp));
                  assert (convertToInteger ("0x1.ffffffffffffffffp62", 64,
                                            true, rm, kind,
                                            0, APFloat::opInvalidOp));
                }
              else
                {
                  assert (convertToInteger ("0x1.ffffffffffffffffp63", 64,
                                            false, rm, kind,
                                            ~0ULL, APFloat::opInexact));
                  assert (convertToInteger ("0x1.ffffffffffffffffp62", 64,
                                            true, rm, kind,
                                            ~0ULL >> 1, APFloat::opInexact));
                }
            }

	  if (rm == APFloat::rmTowardPositive
	      || rm == APFloat::rmNearestTiesToEven)
	    assert (convertToInteger ("0x1fp-1", 5, true, rm, kind,
					0, APFloat::opInvalidOp));
	  else
	    assert (convertToInteger ("0x1fp-1", 5, true, rm, kind,
					15, APFloat::opInexact));

	  if (rm == APFloat::rmTowardNegative)
	    assert (convertToInteger ("-0x1p-1", 5, false, rm, kind,
					0, APFloat::opInvalidOp));
	  else
	    assert (convertToInteger ("-0x1p-1", 5, false, rm, kind,
					0, APFloat::opInexact));

	  if (rm == APFloat::rmTowardNegative
	      || rm == APFloat::rmNearestTiesToEven)
	    assert (convertToInteger ("-0x3p-2", 5, false, rm, kind,
					0, APFloat::opInvalidOp));
	  else
	    assert (convertToInteger ("-0x3p-2", 5, false, rm, kind,
					0, APFloat::opInexact));
	}
    }

  /* Test conversion from integer.  */
  for (int i = 0; i < 4; i++)
    {
      static integerPart flt_max[]= { 0, ((1ULL << 24) - 1) << 40, 0 };
      static integerPart flt_max2[]= { 0, (((1ULL << 24) - 1) << 40) + 1, 0 };

      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      for (int j = 0; j < 4; j++)
	{
	  const fltSemantics &semantics = *all_semantics[j];

	  assert (convertFromInteger (5, true, rm, semantics, "0x5p0",
					APFloat::opOK));
	  assert (convertFromInteger (-1, true, rm, semantics, "-0x1p0",
					APFloat::opOK));
	  assert (convertFromInteger_parts (flt_max, 3, false, rm, semantics,
					      "0x1.fffffep127",
					      APFloat::opOK));
	}

      if (rm == APFloat::rmTowardPositive)
	assert (convertFromInteger_parts (flt_max2, 3, false, rm,
					    APFloat::IEEEsingle,
					    f_pos_infinity, overflow));
      else
	assert (convertFromInteger_parts (flt_max2, 3, false, rm,
					    APFloat::IEEEsingle,
					    "0x1.fffffep127",
					    APFloat::opInexact));

      assert (convertFromInteger (0x1fffffe, false, rm,
				    APFloat::IEEEsingle, "0x1fffffep0",
				    APFloat::opOK));

      if (rm == APFloat::rmTowardZero
	  || rm == APFloat::rmTowardNegative)
	assert (convertFromInteger (0x1ffffff, false, rm,
				      APFloat::IEEEsingle, "0x1fffffep0",
				      APFloat::opInexact));
      else
	assert (convertFromInteger (0x1ffffff, false, rm,
				      APFloat::IEEEsingle, "0x2000000p0",
				      APFloat::opInexact));

      if (rm == APFloat::rmTowardZero
	  || rm == APFloat::rmTowardPositive)
	assert (convertFromInteger (-0x1ffffff, true, rm,
				      APFloat::IEEEsingle, "-0x1fffffep0",
				      APFloat::opInexact));
      else
	assert (convertFromInteger (-0x1ffffff, true, rm,
				      APFloat::IEEEsingle, "-0x2000000p0",
				      APFloat::opInexact));


    }

  /* Subtraction, exact.  */
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      bool up = (rm == APFloat::rmTowardPositive
		 || rm == APFloat::rmNearestTiesToEven);
      bool inf = (rm == APFloat::rmTowardPositive);

      const fltSemantics &kind = APFloat::IEEEsingle;

      assert (fma ("-0x4p0", "0x5p0", "0x5p0", "-0xfp0",
		   rm, kind, APFloat::opOK));
      assert (fma ("0x3p0", "-0x2p0", "0x8p0", "0x2p0",
		   rm, kind, APFloat::opOK));
      assert (fma ("0x4p0", "0x9p0", "0xfp0", "0x33p0",
		   rm, kind, APFloat::opOK));

      assert( fma ("0x1.7e5p0", "0x1.3bbp0", "0x1.0p-24",
		   "0x1.d77348p0", rm, kind, APFloat::opOK));
      assert( fma ("0x1.7e5p0", "0x1.3bbp0", "-0x1.0p-24",
		   "0x1.d77346p0", rm, kind, APFloat::opOK));
      assert( fma ("0x1.7e5p0", "0x1.3bbp0", "-0x1.0p-25",
		   inf ? "0x1.d77348p0" : "0x1.d77346p0",
		   rm, kind, APFloat::opInexact));
      assert( fma ("0x1.7e5p0", "0x1.3bbp0", "0x1.0p-25",
		   up ? "0x1.d77348p0" : "0x1.d77346p0",
		   rm, kind, APFloat::opInexact));
      assert( fma ("0x1p-128", "0x1p-128", "0x1p0",
		   inf ? "0x1.000002p0" : "0x1.0p0",
		   rm, kind, APFloat::opInexact));
    }

#if 0
  assert (remainder ("1.2e234", "0.13", "0x1.97b55ff72df4p-6",
                     APFloat::IEEEdouble));

  /* Remainder.  */
  for (int j = 0; j < 4; j++)
    {
      const fltSemantics &kind = *all_semantics[j];

      assert (fmod ("0x1p0", "0x4p0", "0x1p0", kind));
      assert (fmod ("0x2p0", "0x4p0", "0x2p0", kind));
      assert (fmod ("0x3p0", "0x4p0", "0x3p0", kind));
      assert (fmod ("0x4p0", "0x4p0", "0x0p0", kind));
      assert (fmod ("0x5p0", "0x4p0", "0x1p0", kind));
      assert (fmod ("0x6p0", "0x4p0", "0x2p0", kind));
      assert (fmod ("0x7p0", "0x4p0", "0x3p0", kind));

      assert (fmod ("-0x1p0", "0x4p0", "-0x1p0", kind));
      assert (fmod ("-0x2p0", "0x4p0", "-0x2p0", kind));
      assert (fmod ("-0x3p0", "0x4p0", "-0x3p0", kind));
      assert (fmod ("-0x4p0", "0x4p0", "-0x0p0", kind));
      assert (fmod ("-0x5p0", "0x4p0", "-0x1p0", kind));
      assert (fmod ("-0x6p0", "0x4p0", "-0x2p0", kind));
      assert (fmod ("-0x7p0", "0x4p0", "-0x3p0", kind));

      assert (remainder ("0x1p0", "0x4p0", "0x1p0", kind));
      assert (remainder ("0x2p0", "0x4p0", "0x2p0", kind));
      assert (remainder ("0x3p0", "0x4p0", "-0x1p0", kind));
      assert (remainder ("0x4p0", "0x4p0", "0x0p0", kind));
      assert (remainder ("0x5p0", "0x4p0", "0x1p0", kind));
      assert (remainder ("0x6p0", "0x4p0", "-0x2p0", kind));
      assert (remainder ("0x7p0", "0x4p0", "-0x1p0", kind));

      assert (remainder ("-0x1p0", "0x4p0", "-0x1p0", kind));
      assert (remainder ("-0x2p0", "0x4p0", "-0x2p0", kind));
      assert (remainder ("-0x3p0", "0x4p0", "0x1p0", kind));
      assert (remainder ("-0x4p0", "0x4p0", "-0x0p0", kind));
      assert (remainder ("-0x5p0", "0x4p0", "-0x1p0", kind));
      assert (remainder ("-0x6p0", "0x4p0", "0x2p0", kind));
      assert (remainder ("-0x7p0", "0x4p0", "0x1p0", kind));
    }

  assert (remainder ("1.2", "0.13", "0x1.eb851eb851eap-6",
                     APFloat::IEEEdouble));
  assert (remainder ("1.255921e234", "-4.56e-24", "-0x1.1d69f880ec1ep-82",
                     APFloat::IEEEdouble));
  assert (fmod ("1.255921e234", "-4.56e-24", "0x1.4ef97b6aac0cp-78",
                     APFloat::IEEEdouble));
#endif

  /* Decimal to binary conversion on IEEE single-precision.  */
  assert (compare ("1.2e32", "0x1.7aa73ap+106", APFloat::IEEEsingle,
                   APFloat::rmNearestTiesToEven));
  assert (compare ("9.87654321e12", "0x1.1f71fcp+43", APFloat::IEEEsingle,
                   APFloat::rmNearestTiesToEven));
  assert (compare ("4483519178866687", "0x1.fdb794p+51", APFloat::IEEEsingle,
                   APFloat::rmNearestTiesToEven));
  assert (compare ("5.2E1", "520E-1", APFloat::IEEEsingle,
                   APFloat::rmNearestTiesToEven));
  assert (compare ("5E2", "500", APFloat::IEEEsingle,
                   APFloat::rmNearestTiesToEven));
  assert (compare ("0x5p0", "5", APFloat::IEEEsingle,
                   APFloat::rmNearestTiesToEven));
  assert (compare ("7.006492321624085354618e-100", "0x0p0",
                   APFloat::IEEEsingle, APFloat::rmNearestTiesToEven));
  assert (compare ("7.7071415537864938900805e-45", "0x1.4p-147",
                   APFloat::IEEEsingle, APFloat::rmNearestTiesToEven));
  assert (compare ("7.7071415537864938900806e-45", "0x1.8p-147",
                   APFloat::IEEEsingle, APFloat::rmNearestTiesToEven));

  /* This number lies on a half-boundary for single-precision.  */
  assert (compare ("308105110354283262570921984", "0x1.fdb798p+87",
                   APFloat::IEEEsingle, APFloat::rmNearestTiesToEven));
  assert (compare ("308105110354283262570921983", "0x1.fdb796p+87",
                   APFloat::IEEEsingle, APFloat::rmNearestTiesToEven));

  /* Alternative roundings.  */
  assert (compare ("308105110354283262570921984", "0x1.fdb796p+87",
                   APFloat::IEEEsingle, APFloat::rmTowardZero));
  assert (compare ("308105110354283262570921983", "0x1.fdb798p+87",
                   APFloat::IEEEsingle, APFloat::rmTowardPositive));

  /* This is FLT_MAX, first most closely, then widest, then
     overflowing.  */
  assert (compare ("3.40282347E+38F", "0x1.fffffep+127",
                   APFloat::IEEEsingle, APFloat::rmNearestTiesToEven));
  assert (compare ("3.40282356E+38F", "0x1.fffffep+127",
                   APFloat::IEEEsingle, APFloat::rmNearestTiesToEven));
  assert (compare ("3.40282357E+38F", "0x1.ffffffp+127",
                   APFloat::IEEEsingle, APFloat::rmNearestTiesToEven));

  /* Test exponent overflow.  */
  assert(compare(APFloat(APFloat::IEEEsingle, "0.0e99999"), f_pos_zero));
  assert(compare(APFloat(APFloat::IEEEsingle, "-0.0e99999"), f_neg_zero));
  assert(compare(APFloat(APFloat::IEEEsingle, "1.0e39"), f_pos_infinity));
  assert(compare(APFloat(APFloat::IEEEsingle, "1.0e51085"), f_pos_infinity));
  assert(compare(APFloat(APFloat::IEEEsingle, "0x1p-18446744073709551615"), f_pos_zero));
  assert(compare(APFloat(APFloat::IEEEsingle, "1.0e99999"), f_pos_infinity));
  assert(compare(APFloat(APFloat::IEEEsingle, "1.0e99999999999999999999999999"), f_pos_infinity));
  assert(compare(APFloat(APFloat::IEEEsingle, "-1.0e51085"), f_neg_infinity));

  /* Test exponent underflow.  */
  assert(APFloat(APFloat::IEEEsingle, "1.0e-45").getCategory() != APFloat::fcZero);
  assert(compare(APFloat(APFloat::IEEEsingle, "1.0e-46"), f_pos_zero));
  assert(compare(APFloat(APFloat::IEEEsingle, "-1.0e-46"), f_neg_zero));
  assert(compare(APFloat(APFloat::IEEEsingle, "1.0e-51085"), f_pos_zero));
  assert(compare(APFloat(APFloat::IEEEsingle, "-1.0e-51085"), f_neg_zero));
  assert(compare(APFloat(APFloat::IEEEsingle, "1.0e-99999"), f_pos_zero));
  assert(compare(APFloat(APFloat::IEEEsingle, "1.0e-99999999999999999999999999"), f_pos_zero));

  /* This is FLT_MIN, first most closely, then narrowest, then
     denormal.  */
  assert (compare ("1.17549435E-38F", "0x1p-126",
                   APFloat::IEEEsingle, APFloat::rmNearestTiesToEven));
  assert (compare ("1.17549429E-38F", "0x1p-126",
                   APFloat::IEEEsingle, APFloat::rmNearestTiesToEven));
  assert (compare ("1.17549428E-38F", "0x1.fffffcp-127",
                   APFloat::IEEEsingle, APFloat::rmNearestTiesToEven));

  /* This is FLT_DENORM_MIN, first most closely, then midge's dick
     from zero, then zero.  */
  assert (compare ("1.40129846e-45F", "0x1p-149",
                   APFloat::IEEEsingle, APFloat::rmNearestTiesToEven));
  assert (compare ("7.006492321624085354619e-46F", "0x1p-149",
                   APFloat::IEEEsingle, APFloat::rmNearestTiesToEven));
  assert (compare ("7.006492321624085354618e-46F", "0x0p0",
                   APFloat::IEEEsingle, APFloat::rmNearestTiesToEven));

  assert (compare ("-7.006492321624085354619e-46F", "-0x0p0",
                   APFloat::IEEEsingle, APFloat::rmTowardZero));
  assert (compare ("-7.006492321624085354618e-46F", "-0x1p-149",
                   APFloat::IEEEsingle, APFloat::rmTowardNegative));

  assert (compare ("1.4e-47F", "0x1p-149",
                   APFloat::IEEEsingle, APFloat::rmTowardPositive));
  assert (compare ("1.4e-47F", "0x0p0",
                   APFloat::IEEEsingle, APFloat::rmTowardZero));

  /* Decimal to binary conversion on IEEE double-precision, then hard
     cases.  */
  assert (compare ("1.2e234", "0x1.82780b8bbd6b7p+777", APFloat::IEEEdouble,
                   APFloat::rmNearestTiesToEven));
  assert (compare (".13", "0x1.0a3d70a3d70a4p-3", APFloat::IEEEdouble,
                   APFloat::rmNearestTiesToEven));
  assert (compare ("834548641e-46", "0x1.c65f1a8ed60c4p-124", APFloat::IEEEdouble,
                   APFloat::rmNearestTiesToEven));
  assert (compare ("412413848938563e-27", "0x1.d05632e531e79p-42",
                   APFloat::IEEEdouble, APFloat::rmNearestTiesToEven));
  assert (compare ("5592117679628511e-48", "0x1.d09330a4597fep-108",
                   APFloat::IEEEdouble, APFloat::rmNearestTiesToEven));
  assert (compare ("83881765194427665e-50", "0x1.16beb6c9027ffp-110",
                   APFloat::IEEEdouble, APFloat::rmNearestTiesToEven));
  assert (compare ("356645068918103229683e-42", "0xd.79426bd75b3c68p-75",
                   APFloat::x87DoubleExtended, APFloat::rmNearestTiesToEven));
  assert (compare ("4891559871276714924261e222", "0x1.6ecaf7694a3c7p+809",
                   APFloat::IEEEdouble, APFloat::rmNearestTiesToEven));

  /* This is DBL_MAX closely, then widest, then overflowing.  */
  assert (compare ("1.7976931348623157E+308", "0x1.fffffffffffffp+1023",
                   APFloat::IEEEdouble, APFloat::rmNearestTiesToEven));

  assert (compare ("1.79769313486231580793728971405303415079934132710037826936173778980444968292764750946649017977587207096330286416692887910946555547851940402630657488671505820681908902000708383676273854845817711531764475730270069855571366959622842914819860834936475292719074168444365510704342711559699508093042880e+308", "0x1.fffffffffffffp+1023",
                   APFloat::IEEEdouble, APFloat::rmNearestTiesToEven));

  assert (compare ("1.79769313486231580793728971405303415079934132710037826936173778980444968292764750946649017977587207096330286416692887910946555547851940402630657488671505820681908902000708383676273854845817711531764475730270069855571366959622842914819860834936475292719074168444365510704342711559699508093042881e+308", "0x2.0p+1023",
                   APFloat::IEEEdouble, APFloat::rmNearestTiesToEven));

  /* Now DBL_MIN closely.  */
  assert (compare ("2.2250738585072014E-308", "0x1p-1022",
                   APFloat::IEEEdouble, APFloat::rmNearestTiesToEven));

  /* Now a denormal double.  */
  assert (compare ("1.0864618449742194253370276940099629219820390529601229641373669027769552904364834940694741987365776e-310", "0x1.4p-1030",
                   APFloat::IEEEdouble, APFloat::rmNearestTiesToEven));

  return 0;
}
