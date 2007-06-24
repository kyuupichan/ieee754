/*
   Copyright 2007 Neil Booth.

   See the file "COPYING" for information about the copyright
   and warranty status of this software.
*/

#include <cassert>
#include <cstdio>
#include "float.h"

using namespace llvm;

#define inexact APFloat::fs_inexact
#define underflow (APFloat::e_status) \
	(APFloat::fs_underflow | APFloat::fs_inexact)
#define overflow (APFloat::e_status) \
	(APFloat::fs_overflow | APFloat::fs_inexact)

const fltSemantics *all_semantics[] = {
  &APFloat::ieee_single,
  &APFloat::ieee_double,
  &APFloat::ieee_quad,
  &APFloat::x87_double_extended,
};

static bool
compare (const APFloat &lhs, const APFloat &rhs)
{
  return (lhs.compare (rhs) == APFloat::cmpEqual
	  && lhs.is_negative() == rhs.is_negative() );
}

static bool
convert_from_integer_parts (integerPart *value, unsigned int count,
			    bool is_signed,
			    APFloat::roundingMode rounding_mode,
			    const fltSemantics &semantics, const char *a,
			    APFloat::e_status status)
{
  APFloat number (semantics, APFloat::fc_zero);
  APFloat result (semantics, a);

  if (number.convert_from_integer (value, count, is_signed, rounding_mode)
      != status)
    return false;

  return compare (number, result);
}

static bool
convert_from_integer_parts (integerPart *value, unsigned int count,
			    bool is_signed,
			    APFloat::roundingMode rounding_mode,
			    const fltSemantics &semantics,
			    const APFloat &result,
			    APFloat::e_status status)
{
  APFloat number (semantics, APFloat::fc_zero);

  if (number.convert_from_integer (value, count, is_signed, rounding_mode)
      != status)
    return false;

  return compare (number, result);
}

static bool
convert_from_integer (integerPart value, bool is_signed,
		      APFloat::roundingMode rounding_mode,
		      const fltSemantics &kind,
		      const char *a,
		      APFloat::e_status status)
{
  APFloat number (kind, APFloat::fc_zero);
  APFloat result (kind, a);

  if (number.convert_from_integer (&value, 1, is_signed, rounding_mode)
      != status)
    return false;

  return compare (number, result);
}

static bool
convert_to_integer (const char *a, unsigned int width, bool is_signed,
		    APFloat::roundingMode rounding_mode,
		    const fltSemantics &kind,
		    integerPart result, APFloat::e_status status)
{
  APFloat number (kind, a);
  integerPart part;

  if (number.convert_to_integer
      (&part, width, is_signed, rounding_mode) != status)
    return false;

  return status == APFloat::fs_invalid_op || part == result;
}

static bool
add (const char *a, const char *b, const char *c,
     APFloat::roundingMode rounding_mode,
     const fltSemantics &kind,
     APFloat::e_status status)
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
     APFloat::e_status status)
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
	  APFloat::e_status status)
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
	  APFloat::e_status status)
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
     APFloat::e_status status)
{
  APFloat lhs (kind, a);
  APFloat multiplicand (kind, b);
  APFloat addend (kind, c);
  APFloat result (kind, d);

  if (lhs.fused_multiply_add (multiplicand, addend, rounding_mode) != status)
    return false;

  return compare (lhs, result);
}

static bool
divide (const char *a, const char *b, const char *c,
	APFloat::roundingMode rounding_mode,
	const fltSemantics &kind,
	APFloat::e_status status)
{
  APFloat lhs (kind, a);
  APFloat rhs (kind, b);
  APFloat result (kind, c);

  if (lhs.divide (rhs, rounding_mode) != status)
    return false;

  return compare (lhs, result);
}

int main (void)
{
  APFloat d_nan (APFloat::ieee_double, APFloat::fc_nan, false);
  APFloat d_pos_infinity (APFloat::ieee_double, APFloat::fc_infinity, false);
  APFloat d_neg_infinity (APFloat::ieee_double, APFloat::fc_infinity, true);
  APFloat f_pos_infinity (APFloat::ieee_single, APFloat::fc_infinity, false);

  /* Test floating-point exact divisions.  */
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      for (int j = 0; j < 4; j++)
	{
	  const fltSemantics &kind = *all_semantics[j];

	  assert (divide ("0x1f865cp0", "0x944p0", "0x367p0",
			  rm, kind, APFloat::fs_ok));
	  assert (divide ("0xFb320p-4", "-0xd9.4p2", "-0x25.0p1",
			  rm, kind, APFloat::fs_ok));

	  if (APFloat::semantics_precision (kind) >= 53)
	    assert (divide ("0x0.8b6064570fa168p800", "0x0.badeadf1p500",
			    "0x0.beefe8p300",rm, kind, APFloat::fs_ok));

	  if (APFloat::semantics_precision (kind) >= 62)
	    assert (divide ("-0x0.a84f5a4693e1a774p-2", "-0x0.badeadf1p-16000",
			    "0x0.39a4beadp16000",rm, kind, APFloat::fs_ok));
	}
    }

  /* Division leaves a fraction of one-half and an odd number.  */
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);
      const fltSemantics &kind = APFloat::ieee_single;

      if (rm == APFloat::frm_to_nearest || rm == APFloat::frm_to_plus_infinity)
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
      const fltSemantics &kind = APFloat::ieee_single;

      if (rm == APFloat::frm_to_nearest
	  || rm == APFloat::frm_to_minus_infinity)
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
      const fltSemantics &kind = APFloat::ieee_single;

      if (rm != APFloat::frm_to_plus_infinity)
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
      const fltSemantics &kind = APFloat::ieee_single;

      if (rm != APFloat::frm_to_minus_infinity)
	assert (divide ("-0x1.000002p-126", "0x2p0", "-0x1.000000p-127",
			rm, kind, underflow));
      else
	assert (divide ("-0x1.000002p-126", "0x2p0", "-0x1.000004p-127",
			rm, kind, underflow));
    }

  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);
      const fltSemantics &kind = APFloat::ieee_double;

      bool up = (rm == APFloat::frm_to_plus_infinity
		 || rm == APFloat::frm_to_nearest);
      bool inf = (rm == APFloat::frm_to_plus_infinity);

      assert( divide ("0x1.567abc234109ep156", "0x1.0478fedbca987p-56",
		      up ? "0x1.50994033508bcp+212": "0x1.50994033508bbp+212",
		      rm, kind, APFloat::fs_inexact));

      assert( multiply ("0x1.567abc234109ep156", "0x1.0478fedbca987p-56",
		      inf ? "0x1.5c76858fcf474p+100": "0x1.5c76858fcf473p+100",
		      rm, kind, APFloat::fs_inexact));
    }

  /* Test rounding of multiplication to zero.  */
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);
      const fltSemantics &kind = APFloat::ieee_single;

      if (rm == APFloat::frm_to_plus_infinity)
	assert (multiply ("0x1.0p-100", "0x1.0p-100", "0x1.0p-149f",
			rm, kind, underflow));
      else
	assert (multiply ("0x1.0p-100", "0x1.0p-100", "0x0p0",
			rm, kind, underflow));

      if (rm == APFloat::frm_to_minus_infinity)
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
		       rm, kind, APFloat::fs_ok));
	  assert (add ("0x1p5", "0x1.3456p1", "0x22.68acp0",
		       rm, kind, APFloat::fs_ok));
	  assert (add ("0x1.3456p1", "0x1p5", "0x22.68acp0",
		       rm, kind, APFloat::fs_ok));
	  assert (add ("0x1.000002p0", "0x0.000002p0", "0x1.000004p0",
		       rm, kind, APFloat::fs_ok));

 	  /* This case is exact except for ieee_single.  */
	  if (APFloat::semantics_precision (kind) > 24)
	    {
	      assert (add ("0x1.234562p0", "0x1.234562p-1", "0x1.b4e813p0",
			   rm, kind, APFloat::fs_ok));
	      assert (add ("0x1.234562p-1", "0x1.234562p0", "0x1.b4e813p0",
			   rm, kind, APFloat::fs_ok));
	    }
	  else if (rm == APFloat::frm_to_zero
		   || rm == APFloat::frm_to_minus_infinity)
	    {
	      assert (add ("0x1.234562p0", "0x1.234562p-1", "0x1.b4e812p0",
			   rm, kind, APFloat::fs_inexact));
	      assert (add ("0x1.234562p-1", "0x1.234562p0", "0x1.b4e812p0",
			   rm, kind, APFloat::fs_inexact));
	    }
	  else
	    {
	      assert (add ("0x1.234562p0", "0x1.234562p-1", "0x1.b4e814p0",
			   rm, kind, APFloat::fs_inexact));
	      assert (add ("0x1.234562p-1", "0x1.234562p0", "0x1.b4e814p0",
			   rm, kind, APFloat::fs_inexact));
	    }
	}
    }

  /* Test rounding on floating-point additions, same sign.  These
     lose a fraction of exactly one half.  */
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);
      const fltSemantics &kind = APFloat::ieee_single;

      bool up = (rm == APFloat::frm_to_plus_infinity
		 || rm == APFloat::frm_to_nearest);
      bool inf = (rm == APFloat::frm_to_plus_infinity);

      assert (add ("0x1.000002p0", "0x0.000001p0",
		   up ? "0x1.000004p0": "0x1.000002p0",
		   rm, kind, APFloat::fs_inexact));
      assert (add ("0x0.000001p0", "0x1.000002p0",
		   up ? "0x1.000004p0": "0x1.000002p0",
		   rm, kind, APFloat::fs_inexact));
      assert (add ("0x1.000000p0", "0x0.000001p0",
		   rm == APFloat::frm_to_plus_infinity
		   ? "0x1.000002p0" : "0x1.000000p0",
		   rm, kind, APFloat::fs_inexact));
      assert (add ("0x0.000001p0", "0x1.000000p0",
		   rm == APFloat::frm_to_plus_infinity
		   ? "0x1.000002p0" : "0x1.000000p0",
		   rm, kind, APFloat::fs_inexact));

      assert (add ("0x1.fffffep0", "0x0.00000201p0",
		   inf ? "0x2.000004p0" : "0x2.000000p0",
		   rm, kind, APFloat::fs_inexact));
      assert (add ("0x1.fffffep0", "0x0.00000300p0",
		   inf ? "0x2.000004p0" : "0x2.000000p0",
		   rm, kind, APFloat::fs_inexact));
      assert (add ("0x1.fffffep0", "0x0.00000301p0",
		   inf ? "0x2.000004p0" : "0x2.000000p0",
		   rm, kind, APFloat::fs_inexact));
      assert (add ("0x1.fffffep0", "0x0.00000401p0",
		   up ? "0x2.000004p0" : "0x2.000000p0",
		   rm, kind, APFloat::fs_inexact));
      assert (add ("0x1.fffffep0", "0x0.00000500p0",
		   up ? "0x2.000004p0" : "0x2.000000p0",
		   rm, kind, APFloat::fs_inexact));
      assert (add ("0x1.fffffep0", "0x0.00000501p0",
		   up ? "0x2.000004p0" : "0x2.000000p0",
		   rm, kind, APFloat::fs_inexact));
      assert (add ("0x1.fffffep0", "0x0.00000700p0",
		   inf ? "0x2.000008p0" : "0x2.000004p0",
		   rm, kind, APFloat::fs_inexact));

      assert (add ("-0x1.000006p-2", "0x1.000006p0",
		   inf ? "0x1.80000ap-1" : "0x1.800008p-1",
		   rm, kind, APFloat::fs_inexact));
      assert (add ("-0x1.000006p-3", "0x1.000006p0",
		   inf ? "0x1.c0000cp-1" : "0x1.c0000ap-1",
		   rm, kind, APFloat::fs_inexact));
      assert (add ("-0x1.000006p-4", "0x1.000006p0",
		   up ? "0x1.e0000cp-1" : "0x1.e0000ap-1",
		   rm, kind, APFloat::fs_inexact));

      assert (add ("0x1.000006p0", "-0x1.000006p-2",
		   inf ? "0x1.80000ap-1" : "0x1.800008p-1",
		   rm, kind, APFloat::fs_inexact));
      assert (add ("0x1.000006p0", "-0x1.000006p-3",
		   inf ? "0x1.c0000cp-1" : "0x1.c0000ap-1",
		   rm, kind, APFloat::fs_inexact));
      assert (add ("0x1.000006p0", "-0x1.000006p-4",
		   up ? "0x1.e0000cp-1" : "0x1.e0000ap-1",
		   rm, kind, APFloat::fs_inexact));

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
      const fltSemantics &kind = APFloat::ieee_double;

      assert (add ("0x1.ffffffffffff0p1023", "0x0.fp975",
		   "0x1.fffffffffffffp1023", rm, kind, APFloat::fs_ok));
      assert (add ("0x0.fp975", "0x1.ffffffffffff0p1023",
		   "0x1.fffffffffffffp1023", rm, kind, APFloat::fs_ok));

      /* Sub-half overflow.  */
      if (rm == APFloat::frm_to_plus_infinity)
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
      if (rm != APFloat::frm_to_plus_infinity && rm != APFloat::frm_to_zero)
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
      if (rm != APFloat::frm_to_minus_infinity && rm != APFloat::frm_to_zero)
	assert (add ("0x1.ffffffffffff0p1023", "0x1.0p975",
		     d_pos_infinity, rm, kind, overflow));
      else
	assert (add ("0x1.ffffffffffff0p1023", "0x1.0p975",
		     "0x1.fffffffffffffp1023", rm, kind, inexact));

      if (rm != APFloat::frm_to_plus_infinity && rm != APFloat::frm_to_zero)
	assert (add ("-0x1.0p975", "-0x1.ffffffffffff0p1023",
		     d_neg_infinity, rm, kind, overflow));
      else
	assert (add ("-0x1.0p975", "-0x1.ffffffffffff0p1023",
		     "-0x1.fffffffffffffp1023", rm, kind, inexact));

      /* Add the biggest doule to itself.  */
      if (rm == APFloat::frm_to_minus_infinity || rm == APFloat::frm_to_zero)
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
      const fltSemantics &kind = APFloat::ieee_double;

      assert (add ("0x0.0000000000001p-1022", "0x0.0000000000001p-1022",
		   "0x0.0000000000001p-1021", rm, kind, APFloat::fs_ok));
      assert (add ("0x0.9000000000001p-1022", "0x0.8abcdef012345p-1022",
		   "0x1.1abcdef012346p-1022", rm, kind, APFloat::fs_ok));
      /* Two denormals add to a normal.  */
      assert (add ("0x0.8abcdef012345p-1022", "0x0.9000000000001p-1022",
		   "0x1.1abcdef012346p-1022", rm, kind, APFloat::fs_ok));

      /* Epsilon.  */
      assert (add ("0x1.ap0", "0x1.0p-52",
		   "0x1.a000000000001p0", rm, kind, APFloat::fs_ok));
      assert (add ("0x1.0p-52","0x1.ap0",
		   "0x1.a000000000001p0", rm, kind, APFloat::fs_ok));

      /* Loss of precision.  */
      if (rm == APFloat::frm_to_plus_infinity)
	{
	  assert (add ("0x1.ap0", "0x1.0p-53",
		       "0x1.a000000000001p0", rm, kind, APFloat::fs_inexact));
	  assert (add ("0x1.0p-53","0x1.ap0",
		       "0x1.a000000000001p0", rm, kind, APFloat::fs_inexact));
	}
      else
	{
	  assert (add ("0x1.ap0", "0x1.0p-53",
		       "0x1.ap0", rm, kind, APFloat::fs_inexact));
	  assert (add ("0x1.0p-53","0x1.ap0",
		       "0x1.ap0", rm, kind, APFloat::fs_inexact));
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
		       rm, kind, APFloat::fs_ok));
	  assert (add ("0x5p0", "-0x4p0", "0x1p0",
		       rm, kind, APFloat::fs_ok));
	  assert (add ("-0x4p0", "0x8p0", "0x4p0",
		       rm, kind, APFloat::fs_ok));
	  assert (add ("0x8p0", "-0x4p0", "0x4p0",
		       rm, kind, APFloat::fs_ok));

	  assert (add ("-0x4p0", "0x3p0", "-0x1p0",
		       rm, kind, APFloat::fs_ok));
	  assert (add ("0x3p0", "-0x4p0", "-0x1p0",
		       rm, kind, APFloat::fs_ok));
	  assert (add ("0x4p0", "-0x3p0", "0x1p0",
		       rm, kind, APFloat::fs_ok));
	  assert (add ("-0x3p0", "0x4p0", "0x1p0",
		       rm, kind, APFloat::fs_ok));

	  assert (subtract ("0x4p0", "0x5p0", "-0x1p0",
			    rm, kind, APFloat::fs_ok));
	  assert (subtract ("0x5p0", "0x4p0", "0x1p0",
			    rm, kind, APFloat::fs_ok));
	  assert (subtract ("-0x4p0", "-0x8p0", "0x4p0",
			    rm, kind, APFloat::fs_ok));
	  assert (subtract ("0x8p0", "0x4p0", "0x4p0",
			    rm, kind, APFloat::fs_ok));

	  assert (subtract ("-0x4p0", "0x3p0", "-0x7p0",
			    rm, kind, APFloat::fs_ok));
	  assert (subtract ("0x3p0", "-0x4p0", "0x7p0",
			    rm, kind, APFloat::fs_ok));
	  assert (subtract ("0x4p0", "-0x3p0", "0x7p0",
			    rm, kind, APFloat::fs_ok));
	  assert (subtract ("-0x3p0", "0x4p0", "-0x7p0",
			    rm, kind, APFloat::fs_ok));
	}
    }

  /* Test positive / negative zero rule.  */
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      for (int j = 0; j < 4; j++)
	{
	  const fltSemantics &kind = *all_semantics[j];

  	  if (rm == APFloat::frm_to_minus_infinity)
	    {
	      assert (subtract ("0x3p0", "0x3p0", "-0x0p0",
				rm, kind, APFloat::fs_ok));
	      assert (add ("-0x3p0", "0x3p0", "-0x0p0",
			   rm, kind, APFloat::fs_ok));
	    }
	  else
	    {
	      assert (subtract ("0x3p0", "0x3p0", "0x0p0",
				rm, kind, APFloat::fs_ok));
	      assert (add ("-0x3p0", "0x3p0", "0x0p0",
			   rm, kind, APFloat::fs_ok));
	    }
	}
    }

  /* Test conversion to integer.  */
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);
      integerPart tmp;

      assert (d_nan.convert_to_integer (&tmp, 5, true, rm)
	      == APFloat::fs_invalid_op);
      assert (d_pos_infinity.convert_to_integer (&tmp, 5, true, rm)
	      == APFloat::fs_invalid_op);
      assert (d_neg_infinity.convert_to_integer (&tmp, 5, true, rm)
	      == APFloat::fs_invalid_op);

      for (int j = 0; j < 4; j++)
	{
	  const fltSemantics &kind = *all_semantics[j];

	  assert (convert_to_integer ("0x0p0", 5, false, rm, kind,
				      0, APFloat::fs_ok));
	  assert (convert_to_integer ("-0x0p0", 5, false, rm, kind,
				      0, APFloat::fs_ok));
	  assert (convert_to_integer ("0x0p0", 5, true, rm, kind,
				      0, APFloat::fs_ok));
	  assert (convert_to_integer ("-0x0p0", 5, true, rm, kind,
				      0, APFloat::fs_ok));

	  assert (convert_to_integer ("0x1p0", 5, true, rm, kind,
				      1, APFloat::fs_ok));
	  assert (convert_to_integer ("0xfp0", 5, true, rm, kind,
				      15, APFloat::fs_ok));
	  assert (convert_to_integer ("0x10p0", 5, true, rm, kind,
				      0, APFloat::fs_invalid_op));
	  assert (convert_to_integer ("0x1fp0", 5, false, rm, kind,
				      31, APFloat::fs_ok));
	  assert (convert_to_integer ("0x20p0", 5, false, rm, kind,
				      0, APFloat::fs_invalid_op));

	  assert (convert_to_integer ("-0x1p0", 5, true, rm, kind,
				      -1, APFloat::fs_ok));
	  assert (convert_to_integer ("-0x1p0", 5, false, rm, kind,
				      0, APFloat::fs_invalid_op));
	  assert (convert_to_integer ("-0xfp0", 5, true, rm, kind,
				      -15, APFloat::fs_ok));
	  assert (convert_to_integer ("-0x10p0", 5, true, rm, kind,
				      -16, APFloat::fs_ok));
	  assert (convert_to_integer ("0x11p0", 5, true, rm, kind,
				      0, APFloat::fs_invalid_op));

	  for (int k = 0; k <= 1; k++)
	    {
	      assert (convert_to_integer ("0x1p-1", 5, k, rm, kind,
					  rm == APFloat::frm_to_plus_infinity
					  ? 1: 0, APFloat::fs_inexact));
	      assert (convert_to_integer ("0x1p-2", 5, k, rm, kind,
					  rm == APFloat::frm_to_plus_infinity
					  ? 1: 0, APFloat::fs_inexact));
	      assert (convert_to_integer ("0x3p-2", 5, k, rm, kind,
					  rm == APFloat::frm_to_plus_infinity
					  || rm == APFloat::frm_to_nearest
					  ? 1: 0, APFloat::fs_inexact));
	      assert (convert_to_integer ("0x3p-1", 5, k, rm, kind,
					  rm == APFloat::frm_to_plus_infinity
					  || rm == APFloat::frm_to_nearest
					  ? 2: 1, APFloat::fs_inexact));
	      assert (convert_to_integer ("0x7p-2", 5, k, rm, kind,
					  rm == APFloat::frm_to_plus_infinity
					  || rm == APFloat::frm_to_nearest
					  ? 2: 1, APFloat::fs_inexact));
	    }

	  if (rm == APFloat::frm_to_plus_infinity
	      || rm == APFloat::frm_to_nearest)
	    assert (convert_to_integer ("0x1fp-1", 5, true, rm, kind,
					0, APFloat::fs_invalid_op));
	  else
	    assert (convert_to_integer ("0x1fp-1", 5, true, rm, kind,
					15, APFloat::fs_inexact));

	  if (rm == APFloat::frm_to_minus_infinity)
	    assert (convert_to_integer ("-0x1p-1", 5, false, rm, kind,
					0, APFloat::fs_invalid_op));
	  else
	    assert (convert_to_integer ("-0x1p-1", 5, false, rm, kind,
					0, APFloat::fs_inexact));

	  if (rm == APFloat::frm_to_minus_infinity
	      || rm == APFloat::frm_to_nearest)
	    assert (convert_to_integer ("-0x3p-2", 5, false, rm, kind,
					0, APFloat::fs_invalid_op));
	  else
	    assert (convert_to_integer ("-0x3p-2", 5, false, rm, kind,
					0, APFloat::fs_inexact));
	}
    }

  /* Test conversion from integer.  */
  for (int i = 0; i < 4; i++)
    {
      static integerPart flt_max[]= { 0, ((1ULL << 24) - 1) << 40 };
      static integerPart flt_max2[]= { 0, (((1ULL << 24) - 1) << 40) + 1 };
				   
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      for (int j = 0; j < 4; j++)
	{
	  const fltSemantics &semantics = *all_semantics[j];

	  assert (convert_from_integer (5, true, rm, semantics, "0x5p0",
					APFloat::fs_ok));
	  assert (convert_from_integer (-1, true, rm, semantics, "-0x1p0",
					APFloat::fs_ok));
	  assert (convert_from_integer_parts (flt_max, 2, false, rm, semantics,
					      "0x1.fffffep127",
					      APFloat::fs_ok));
	}

      if (rm == APFloat::frm_to_plus_infinity)
	assert (convert_from_integer_parts (flt_max2, 2, false, rm,
					    APFloat::ieee_single,
					    f_pos_infinity, overflow));
      else
	assert (convert_from_integer_parts (flt_max2, 2, false, rm,
					    APFloat::ieee_single,
					    "0x1.fffffep127",
					    APFloat::fs_inexact));

      assert (convert_from_integer (0x1fffffe, false, rm,
				    APFloat::ieee_single, "0x1fffffep0",
				    APFloat::fs_ok));

      if (rm == APFloat::frm_to_zero
	  || rm == APFloat::frm_to_minus_infinity)
	assert (convert_from_integer (0x1ffffff, false, rm,
				      APFloat::ieee_single, "0x1fffffep0",
				      APFloat::fs_inexact));
      else
	assert (convert_from_integer (0x1ffffff, false, rm,
				      APFloat::ieee_single, "0x2000000p0",
				      APFloat::fs_inexact));

      if (rm == APFloat::frm_to_zero
	  || rm == APFloat::frm_to_plus_infinity)
	assert (convert_from_integer (-0x1ffffff, true, rm,
				      APFloat::ieee_single, "-0x1fffffep0",
				      APFloat::fs_inexact));
      else
	assert (convert_from_integer (-0x1ffffff, true, rm,
				      APFloat::ieee_single, "-0x2000000p0",
				      APFloat::fs_inexact));

      
    }

  /* Subtraction, exact.  */
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      bool up = (rm == APFloat::frm_to_plus_infinity
		 || rm == APFloat::frm_to_nearest);
      bool inf = (rm == APFloat::frm_to_plus_infinity);

      const fltSemantics &kind = APFloat::ieee_single;

      assert (fma ("-0x4p0", "0x5p0", "0x5p0", "-0xfp0",
		   rm, kind, APFloat::fs_ok));
      assert (fma ("0x3p0", "-0x2p0", "0x8p0", "0x2p0",
		   rm, kind, APFloat::fs_ok));
      assert (fma ("0x4p0", "0x9p0", "0xfp0", "0x33p0",
		   rm, kind, APFloat::fs_ok));

      assert( fma ("0x1.7e5p0", "0x1.3bbp0", "0x1.0p-24",
		   "0x1.d77348p0", rm, kind, APFloat::fs_ok));
      assert( fma ("0x1.7e5p0", "0x1.3bbp0", "-0x1.0p-24",
		   "0x1.d77346p0", rm, kind, APFloat::fs_ok));
      assert( fma ("0x1.7e5p0", "0x1.3bbp0", "-0x1.0p-25",
		   inf ? "0x1.d77348p0" : "0x1.d77346p0",
		   rm, kind, APFloat::fs_inexact));
      assert( fma ("0x1.7e5p0", "0x1.3bbp0", "0x1.0p-25",
		   up ? "0x1.d77348p0" : "0x1.d77346p0",
		   rm, kind, APFloat::fs_inexact));
    }

  return 0;
}
