/*
   Copyright 2007 Neil Booth.

   See the file "COPYING" for information about the copyright
   and warranty status of this software.
*/

#include <cassert>
#include "float.h"

using namespace llvm;

static bool
good_compare (const APFloat &lhs, const APFloat &rhs)
{
  if (lhs.get_category () == APFloat::fc_nan)
    return rhs.get_category () == APFloat::fc_nan;

  return (lhs.compare (rhs) == APFloat::cmpEqual
	  && lhs.is_negative() == rhs.is_negative());
}

static APFloat::cmpResult
compare (const APFloat &lhs, const APFloat &rhs)
{
  return lhs.compare (rhs);
}

static bool
convert (const APFloat &lhs, const APFloat &result,
	APFloat::roundingMode rounding_mode, APFloat::e_status status)
{
  APFloat tmp (lhs);

  if (tmp.convert (result.get_semantics (), rounding_mode) != status)
    return false;

  return good_compare (tmp, result);
}

static bool
divide (const APFloat &lhs, const APFloat &rhs, const APFloat &result,
	APFloat::roundingMode rounding_mode, APFloat::e_status status)
{
  APFloat tmp (lhs);

  if (tmp.divide (rhs, rounding_mode) != status)
    return false;

  return good_compare (tmp, result);
}

static bool
mult (const APFloat &lhs, const APFloat &rhs, const APFloat &result,
      APFloat::roundingMode rounding_mode, APFloat::e_status status)
{
  APFloat tmp (lhs);

  if (tmp.multiply (rhs, rounding_mode) != status)
    return false;

  return good_compare (tmp, result);
}

static bool
add (const APFloat &lhs, const APFloat &rhs, const APFloat &result,
     APFloat::roundingMode rounding_mode, APFloat::e_status status)
{
  APFloat tmp (lhs);

  if (tmp.add (rhs, rounding_mode) != status)
    return false;

  return good_compare (tmp, result);
}

static bool
sub (const APFloat &lhs, const APFloat &rhs, const APFloat &result,
     APFloat::roundingMode rounding_mode, APFloat::e_status status)
{
  APFloat tmp (lhs);

  if (tmp.subtract (rhs, rounding_mode) != status)
    return false;

  return good_compare (tmp, result);
}

static bool
fma (const APFloat &lhs, const APFloat &m, const APFloat &a,
     const APFloat &result, APFloat::roundingMode rounding_mode,
     APFloat::e_status status)
{
  APFloat tmp (lhs);

  if (tmp.fused_multiply_add (m, a, rounding_mode) != status)
    return false;

  return good_compare (tmp, result);
}

int main (void)
{
  APFloat f_pos_infinity (APFloat::ieee_single, APFloat::fc_infinity, false);
  APFloat f_neg_infinity (APFloat::ieee_single, APFloat::fc_infinity, true);
  APFloat f_pos_zero (APFloat::ieee_single, APFloat::fc_zero, false);
  APFloat f_neg_zero (APFloat::ieee_single, APFloat::fc_zero, true);
  APFloat f_nan (APFloat::ieee_single, APFloat::fc_nan, false);
  APFloat f_one (APFloat::ieee_single, 1);
  APFloat d_one (APFloat::ieee_double, 1);
  APFloat f_two (APFloat::ieee_single, 2);
  APFloat f_neg_one (f_one);
  APFloat f_neg_two (f_two);
  APFloat d_neg_one (f_one);

  f_neg_one.change_sign ();
  f_neg_two.change_sign ();
  d_neg_one.change_sign ();

  // Conversions.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (convert (f_one, d_one, rm, APFloat::fs_ok));
      assert (convert (f_neg_one, d_neg_one, rm, APFloat::fs_ok));
      assert (convert (d_one, f_one, rm, APFloat::fs_ok));
      assert (convert (d_neg_one, f_neg_one, rm, APFloat::fs_ok));
    }

  // Comparisons; pos_infinity lhs.
  assert (compare (f_pos_infinity, f_pos_infinity) == APFloat::cmpEqual);
  assert (compare (f_pos_infinity, f_neg_infinity)
	  == APFloat::cmpGreaterThan);
  assert (compare (f_pos_infinity, f_pos_zero) == APFloat::cmpGreaterThan);
  assert (compare (f_pos_infinity, f_neg_zero) == APFloat::cmpGreaterThan);
  assert (compare (f_pos_infinity, f_one) == APFloat::cmpGreaterThan);
  assert (compare (f_pos_infinity, f_neg_one) == APFloat::cmpGreaterThan);
  assert (compare (f_pos_infinity, f_nan) == APFloat::cmpUnordered);

  // Comparisons; neg_infinity lhs.
  assert (compare (f_neg_infinity, f_pos_infinity) == APFloat::cmpLessThan);
  assert (compare (f_neg_infinity, f_neg_infinity) == APFloat::cmpEqual);
  assert (compare (f_neg_infinity, f_pos_zero) == APFloat::cmpLessThan);
  assert (compare (f_neg_infinity, f_neg_zero) == APFloat::cmpLessThan);
  assert (compare (f_neg_infinity, f_one) == APFloat::cmpLessThan);
  assert (compare (f_neg_infinity, f_neg_one) == APFloat::cmpLessThan);
  assert (compare (f_neg_infinity, f_nan) == APFloat::cmpUnordered);

  // Comparisons; pos_zero lhs.
  assert (compare (f_pos_zero, f_pos_infinity) == APFloat::cmpLessThan);
  assert (compare (f_pos_zero, f_neg_infinity) == APFloat::cmpGreaterThan);
  assert (compare (f_pos_zero, f_pos_zero) == APFloat::cmpEqual);
  assert (compare (f_pos_zero, f_neg_zero) == APFloat::cmpEqual);
  assert (compare (f_pos_zero, f_one) == APFloat::cmpLessThan);
  assert (compare (f_pos_zero, f_neg_one) == APFloat::cmpGreaterThan);
  assert (compare (f_pos_zero, f_nan) == APFloat::cmpUnordered);

  // Comparisons; neg_zero lhs.
  assert (compare (f_neg_zero, f_pos_infinity) == APFloat::cmpLessThan);
  assert (compare (f_neg_zero, f_neg_infinity) == APFloat::cmpGreaterThan);
  assert (compare (f_neg_zero, f_pos_zero) == APFloat::cmpEqual);
  assert (compare (f_neg_zero, f_neg_zero) == APFloat::cmpEqual);
  assert (compare (f_neg_zero, f_one) == APFloat::cmpLessThan);
  assert (compare (f_neg_zero, f_neg_one) == APFloat::cmpGreaterThan);
  assert (compare (f_neg_zero, f_nan) == APFloat::cmpUnordered);

  // Comparisons; NAN lhs.
  assert (compare (f_nan, f_pos_infinity) == APFloat::cmpUnordered);
  assert (compare (f_nan, f_neg_infinity) == APFloat::cmpUnordered);
  assert (compare (f_nan, f_pos_zero) == APFloat::cmpUnordered);
  assert (compare (f_nan, f_neg_zero) == APFloat::cmpUnordered);
  assert (compare (f_nan, f_one) == APFloat::cmpUnordered);
  assert (compare (f_nan, f_neg_one) == APFloat::cmpUnordered);
  assert (compare (f_nan, f_nan) == APFloat::cmpUnordered);

  // Comparisons; one lhs.
  assert (compare (f_one, f_pos_infinity) == APFloat::cmpLessThan);
  assert (compare (f_one, f_neg_infinity) == APFloat::cmpGreaterThan);
  assert (compare (f_one, f_pos_zero) == APFloat::cmpGreaterThan);
  assert (compare (f_one, f_neg_zero) == APFloat::cmpGreaterThan);
  assert (compare (f_one, f_one) == APFloat::cmpEqual);
  assert (compare (f_one, f_neg_one) == APFloat::cmpGreaterThan);
  assert (compare (f_one, f_nan) == APFloat::cmpUnordered);

  // Comparisons; negative one lhs.
  assert (compare (f_neg_one, f_pos_infinity) == APFloat::cmpLessThan);
  assert (compare (f_neg_one, f_neg_infinity) == APFloat::cmpGreaterThan);
  assert (compare (f_neg_one, f_pos_zero) == APFloat::cmpLessThan);
  assert (compare (f_neg_one, f_neg_zero) == APFloat::cmpLessThan);
  assert (compare (f_neg_one, f_one) == APFloat::cmpLessThan);
  assert (compare (f_neg_one, f_neg_one) == APFloat::cmpEqual);
  assert (compare (f_neg_one, f_nan) == APFloat::cmpUnordered);



  // Divisions; pos_infinity lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (divide (f_pos_infinity, f_pos_infinity, f_nan, rm,
		      APFloat::fs_invalid_op));
      assert (divide (f_pos_infinity, f_neg_infinity, f_nan, rm,
		      APFloat::fs_invalid_op));
      assert (divide (f_pos_infinity, f_pos_zero, f_pos_infinity, rm,
		      APFloat::fs_ok));
      assert (divide (f_pos_infinity, f_neg_zero, f_neg_infinity, rm,
		      APFloat::fs_ok));
      assert (divide (f_pos_infinity, f_one, f_pos_infinity, rm,
		      APFloat::fs_ok));
      assert (divide (f_pos_infinity, f_neg_one, f_neg_infinity, rm,
		      APFloat::fs_ok));
      assert (divide (f_pos_infinity, f_nan, f_nan, rm,
		      APFloat::fs_ok));
    }

  // Divisions; neg_infinity lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (divide (f_neg_infinity, f_pos_infinity, f_nan, rm,
		      APFloat::fs_invalid_op));
      assert (divide (f_neg_infinity, f_neg_infinity, f_nan, rm,
		      APFloat::fs_invalid_op));
      assert (divide (f_neg_infinity, f_pos_zero, f_neg_infinity, rm,
		      APFloat::fs_ok));
      assert (divide (f_neg_infinity, f_neg_zero, f_pos_infinity, rm,
		      APFloat::fs_ok));
      assert (divide (f_neg_infinity, f_one, f_neg_infinity, rm,
		      APFloat::fs_ok));
      assert (divide (f_neg_infinity, f_neg_one, f_pos_infinity, rm,
		      APFloat::fs_ok));
      assert (divide (f_neg_infinity, f_nan, f_nan, rm,
		      APFloat::fs_ok));
    }

  // Divisions; pos_zero lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (divide (f_pos_zero, f_pos_infinity, f_pos_zero, rm,
		      APFloat::fs_ok));
      assert (divide (f_pos_zero, f_neg_infinity, f_neg_zero, rm,
		      APFloat::fs_ok));
      assert (divide (f_pos_zero, f_pos_zero, f_nan, rm,
		      APFloat::fs_invalid_op));
      assert (divide (f_pos_zero, f_neg_zero, f_nan, rm,
		      APFloat::fs_invalid_op));
      assert (divide (f_pos_zero, f_one, f_pos_zero, rm,
		      APFloat::fs_ok));
      assert (divide (f_pos_zero, f_neg_one, f_neg_zero, rm,
		      APFloat::fs_ok));
      assert (divide (f_pos_zero, f_nan, f_nan, rm,
		      APFloat::fs_ok));
    }

  // Divisions; neg_zero lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (divide (f_neg_zero, f_pos_infinity, f_neg_zero, rm,
		      APFloat::fs_ok));
      assert (divide (f_neg_zero, f_neg_infinity, f_pos_zero, rm,
		      APFloat::fs_ok));
      assert (divide (f_neg_zero, f_pos_zero, f_nan, rm,
		      APFloat::fs_invalid_op));
      assert (divide (f_neg_zero, f_neg_zero, f_nan, rm,
		      APFloat::fs_invalid_op));
      assert (divide (f_neg_zero, f_one, f_neg_zero, rm,
		      APFloat::fs_ok));
      assert (divide (f_neg_zero, f_neg_one, f_pos_zero, rm,
		      APFloat::fs_ok));
      assert (divide (f_neg_zero, f_nan, f_nan, rm,
		      APFloat::fs_ok));
    }

  // Divisions; nan lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (divide (f_nan, f_pos_infinity, f_nan, rm,
		      APFloat::fs_ok));
      assert (divide (f_nan, f_neg_infinity, f_nan, rm,
		      APFloat::fs_ok));
      assert (divide (f_nan, f_pos_zero, f_nan, rm,
		      APFloat::fs_ok));
      assert (divide (f_nan, f_neg_zero, f_nan, rm,
		      APFloat::fs_ok));
      assert (divide (f_nan, f_one, f_nan, rm,
		      APFloat::fs_ok));
      assert (divide (f_nan, f_neg_one, f_nan, rm,
		      APFloat::fs_ok));
      assert (divide (f_nan, f_nan, f_nan, rm,
		      APFloat::fs_ok));
    }

  // Divisions; one lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (divide (f_one, f_pos_infinity, f_pos_zero, rm,
		      APFloat::fs_ok));
      assert (divide (f_one, f_neg_infinity, f_neg_zero, rm,
		      APFloat::fs_ok));
      assert (divide (f_one, f_pos_zero, f_pos_infinity, rm,
		      APFloat::fs_div_by_zero));
      assert (divide (f_one, f_neg_zero, f_neg_infinity, rm,
		      APFloat::fs_div_by_zero));
      assert (divide (f_one, f_one, f_one, rm,
		      APFloat::fs_ok));
      assert (divide (f_one, f_neg_one, f_neg_one, rm,
		      APFloat::fs_ok));
      assert (divide (f_one, f_nan, f_nan, rm,
		      APFloat::fs_ok));
    }

  // Divisions; neg_one lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (divide (f_neg_one, f_pos_infinity, f_neg_zero, rm,
		      APFloat::fs_ok));
      assert (divide (f_neg_one, f_neg_infinity, f_pos_zero, rm,
		      APFloat::fs_ok));
      assert (divide (f_neg_one, f_pos_zero, f_neg_infinity, rm,
		      APFloat::fs_div_by_zero));
      assert (divide (f_neg_one, f_neg_zero, f_pos_infinity, rm,
		      APFloat::fs_div_by_zero));
      assert (divide (f_neg_one, f_one, f_neg_one, rm,
		      APFloat::fs_ok));
      assert (divide (f_neg_one, f_neg_one, f_one, rm,
		      APFloat::fs_ok));
      assert (divide (f_neg_one, f_nan, f_nan, rm,
		      APFloat::fs_ok));
    }



  // Multiplications; pos_infinity lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (mult (f_pos_infinity, f_pos_infinity, f_pos_infinity, rm,
		    APFloat::fs_ok));
      assert (mult (f_pos_infinity, f_neg_infinity, f_neg_infinity, rm,
		    APFloat::fs_ok));
      assert (mult (f_pos_infinity, f_pos_zero, f_nan, rm,
		    APFloat::fs_invalid_op));
      assert (mult (f_pos_infinity, f_neg_zero, f_nan, rm,
		    APFloat::fs_invalid_op));
      assert (mult (f_pos_infinity, f_one, f_pos_infinity, rm,
		    APFloat::fs_ok));
      assert (mult (f_pos_infinity, f_neg_one, f_neg_infinity, rm,
		    APFloat::fs_ok));
      assert (mult (f_pos_infinity, f_nan, f_nan, rm,
		    APFloat::fs_ok));
    }

  // Multiplications; neg_infinity lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (mult (f_neg_infinity, f_pos_infinity, f_neg_infinity, rm,
		    APFloat::fs_ok));
      assert (mult (f_neg_infinity, f_neg_infinity, f_pos_infinity, rm,
		    APFloat::fs_ok));
      assert (mult (f_neg_infinity, f_pos_zero, f_nan, rm,
		    APFloat::fs_invalid_op));
      assert (mult (f_neg_infinity, f_neg_zero, f_nan, rm,
		    APFloat::fs_invalid_op));
      assert (mult (f_neg_infinity, f_one, f_neg_infinity, rm,
		    APFloat::fs_ok));
      assert (mult (f_neg_infinity, f_neg_one, f_pos_infinity, rm,
		    APFloat::fs_ok));
      assert (mult (f_neg_infinity, f_nan, f_nan, rm,
		    APFloat::fs_ok));
    }

  // Multiplications; pos_zero lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (mult (f_pos_zero, f_pos_infinity, f_nan, rm,
		    APFloat::fs_invalid_op));
      assert (mult (f_pos_zero, f_neg_infinity, f_nan, rm,
		    APFloat::fs_invalid_op));
      assert (mult (f_pos_zero, f_pos_zero, f_pos_zero, rm,
		    APFloat::fs_ok));
      assert (mult (f_pos_zero, f_neg_zero, f_neg_zero, rm,
		    APFloat::fs_ok));
      assert (mult (f_pos_zero, f_one, f_pos_zero, rm,
		    APFloat::fs_ok));
      assert (mult (f_pos_zero, f_neg_one, f_neg_zero, rm,
		    APFloat::fs_ok));
      assert (mult (f_pos_zero, f_nan, f_nan, rm,
		    APFloat::fs_ok));
    }

  // Multiplications; neg_zero lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (mult (f_neg_zero, f_pos_infinity, f_nan, rm,
		    APFloat::fs_invalid_op));
      assert (mult (f_neg_zero, f_neg_infinity, f_nan, rm,
		    APFloat::fs_invalid_op));
      assert (mult (f_neg_zero, f_pos_zero, f_neg_zero, rm,
		    APFloat::fs_ok));
      assert (mult (f_neg_zero, f_neg_zero, f_pos_zero, rm,
		    APFloat::fs_ok));
      assert (mult (f_neg_zero, f_one, f_neg_zero, rm,
		    APFloat::fs_ok));
      assert (mult (f_neg_zero, f_neg_one, f_pos_zero, rm,
		    APFloat::fs_ok));
      assert (mult (f_neg_zero, f_nan, f_nan, rm,
		    APFloat::fs_ok));
    }

  // Multiplications; nan lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (mult (f_nan, f_pos_infinity, f_nan, rm,
		    APFloat::fs_ok));
      assert (mult (f_nan, f_neg_infinity, f_nan, rm,
		    APFloat::fs_ok));
      assert (mult (f_nan, f_pos_zero, f_nan, rm,
		    APFloat::fs_ok));
      assert (mult (f_nan, f_neg_zero, f_nan, rm,
		    APFloat::fs_ok));
      assert (mult (f_nan, f_one, f_nan, rm,
		    APFloat::fs_ok));
      assert (mult (f_nan, f_neg_one, f_nan, rm,
		    APFloat::fs_ok));
      assert (mult (f_nan, f_nan, f_nan, rm,
		    APFloat::fs_ok));
    }

  // Multiplications; one lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (mult (f_one, f_pos_infinity, f_pos_infinity, rm,
		    APFloat::fs_ok));
      assert (mult (f_one, f_neg_infinity, f_neg_infinity, rm,
		    APFloat::fs_ok));
      assert (mult (f_one, f_pos_zero, f_pos_zero, rm,
		    APFloat::fs_ok));
      assert (mult (f_one, f_neg_zero, f_neg_zero, rm,
		    APFloat::fs_ok));
      assert (mult (f_one, f_one, f_one, rm,
		    APFloat::fs_ok));
      assert (mult (f_one, f_neg_one, f_neg_one, rm,
		    APFloat::fs_ok));
      assert (mult (f_one, f_nan, f_nan, rm,
		    APFloat::fs_ok));
    }

  // Multiplications; neg_one lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (mult (f_neg_one, f_pos_infinity, f_neg_infinity, rm,
		    APFloat::fs_ok));
      assert (mult (f_neg_one, f_neg_infinity, f_pos_infinity, rm,
		    APFloat::fs_ok));
      assert (mult (f_neg_one, f_pos_zero, f_neg_zero, rm,
		    APFloat::fs_ok));
      assert (mult (f_neg_one, f_neg_zero, f_pos_zero, rm,
		    APFloat::fs_ok));
      assert (mult (f_neg_one, f_one, f_neg_one, rm,
		    APFloat::fs_ok));
      assert (mult (f_neg_one, f_neg_one, f_one, rm,
		    APFloat::fs_ok));
      assert (mult (f_neg_one, f_nan, f_nan, rm,
		    APFloat::fs_ok));
    }



  // Additions; pos_infinity lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (add (f_pos_infinity, f_pos_infinity, f_pos_infinity, rm,
		   APFloat::fs_ok));
      assert (add (f_pos_infinity, f_neg_infinity, f_nan, rm,
		   APFloat::fs_invalid_op));
      assert (add (f_pos_infinity, f_pos_zero, f_pos_infinity, rm,
		   APFloat::fs_ok));
      assert (add (f_pos_infinity, f_neg_zero, f_pos_infinity, rm,
		   APFloat::fs_ok));
      assert (add (f_pos_infinity, f_one, f_pos_infinity, rm,
		   APFloat::fs_ok));
      assert (add (f_pos_infinity, f_neg_one, f_pos_infinity, rm,
		   APFloat::fs_ok));
      assert (add (f_pos_infinity, f_nan, f_nan, rm,
		   APFloat::fs_ok));
    }

  // Additions; neg_infinity lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (add (f_neg_infinity, f_pos_infinity, f_nan, rm,
		   APFloat::fs_invalid_op));
      assert (add (f_neg_infinity, f_neg_infinity, f_neg_infinity, rm,
		   APFloat::fs_ok));
      assert (add (f_neg_infinity, f_pos_zero, f_neg_infinity, rm,
		   APFloat::fs_ok));
      assert (add (f_neg_infinity, f_neg_zero, f_neg_infinity, rm,
		   APFloat::fs_ok));
      assert (add (f_neg_infinity, f_one, f_neg_infinity, rm,
		   APFloat::fs_ok));
      assert (add (f_neg_infinity, f_neg_one, f_neg_infinity, rm,
		   APFloat::fs_ok));
      assert (add (f_neg_infinity, f_nan, f_nan, rm,
		   APFloat::fs_ok));
    }

  // Additions; pos_zero lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (add (f_pos_zero, f_pos_infinity, f_pos_infinity, rm,
		   APFloat::fs_ok));
      assert (add (f_pos_zero, f_neg_infinity, f_neg_infinity, rm,
		   APFloat::fs_ok));
      assert (add (f_pos_zero, f_pos_zero, f_pos_zero, rm,
		   APFloat::fs_ok));
      assert (add (f_pos_zero, f_neg_zero, rm == APFloat::frm_to_minus_infinity
		   ? f_neg_zero: f_pos_zero, rm, APFloat::fs_ok));
      assert (add (f_pos_zero, f_one, f_one, rm,
		   APFloat::fs_ok));
      assert (add (f_pos_zero, f_neg_one, f_neg_one, rm,
		   APFloat::fs_ok));
      assert (add (f_pos_zero, f_nan, f_nan, rm,
		   APFloat::fs_ok));
    }

  // Additions; neg_zero lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (add (f_neg_zero, f_pos_infinity, f_pos_infinity, rm,
		   APFloat::fs_ok));
      assert (add (f_neg_zero, f_neg_infinity, f_neg_infinity, rm,
		   APFloat::fs_ok));
      assert (add (f_neg_zero, f_pos_zero, rm == APFloat::frm_to_minus_infinity
		   ? f_neg_zero: f_pos_zero, rm, APFloat::fs_ok));
      assert (add (f_neg_zero, f_neg_zero, f_neg_zero, rm,
		   APFloat::fs_ok));
      assert (add (f_neg_zero, f_one, f_one, rm,
		   APFloat::fs_ok));
      assert (add (f_neg_zero, f_neg_one, f_neg_one, rm,
		   APFloat::fs_ok));
      assert (add (f_neg_zero, f_nan, f_nan, rm,
		   APFloat::fs_ok));
    }

  // Additions; nan lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (add (f_nan, f_pos_infinity, f_nan, rm,
		   APFloat::fs_ok));
      assert (add (f_nan, f_neg_infinity, f_nan, rm,
		   APFloat::fs_ok));
      assert (add (f_nan, f_pos_zero, f_nan, rm,
		   APFloat::fs_ok));
      assert (add (f_nan, f_neg_zero, f_nan, rm,
		   APFloat::fs_ok));
      assert (add (f_nan, f_one, f_nan, rm,
		   APFloat::fs_ok));
      assert (add (f_nan, f_neg_one, f_nan, rm,
		   APFloat::fs_ok));
      assert (add (f_nan, f_nan, f_nan, rm,
		   APFloat::fs_ok));
    }

  // Additions; one lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (add (f_one, f_pos_infinity, f_pos_infinity, rm,
		   APFloat::fs_ok));
      assert (add (f_one, f_neg_infinity, f_neg_infinity, rm,
		   APFloat::fs_ok));
      assert (add (f_one, f_pos_zero, f_one, rm,
		   APFloat::fs_ok));
      assert (add (f_one, f_neg_zero, f_one, rm,
		   APFloat::fs_ok));
      assert (add (f_one, f_one, f_two, rm,
		   APFloat::fs_ok));
      assert (add (f_one, f_neg_one, rm == APFloat::frm_to_minus_infinity
		   ? f_neg_zero: f_pos_zero, rm, APFloat::fs_ok));
      assert (add (f_one, f_nan, f_nan, rm,
		   APFloat::fs_ok));
    }

  // Additions; neg_one lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (add (f_neg_one, f_pos_infinity, f_pos_infinity, rm,
		   APFloat::fs_ok));
      assert (add (f_neg_one, f_neg_infinity, f_neg_infinity, rm,
		   APFloat::fs_ok));
      assert (add (f_neg_one, f_pos_zero, f_neg_one, rm,
		   APFloat::fs_ok));
      assert (add (f_neg_one, f_neg_zero, f_neg_one, rm,
		   APFloat::fs_ok));
      assert (add (f_neg_one, f_one, rm == APFloat::frm_to_minus_infinity
		   ? f_neg_zero: f_pos_zero, rm, APFloat::fs_ok));
      assert (add (f_neg_one, f_neg_one, f_neg_two, rm,
		   APFloat::fs_ok));
      assert (add (f_neg_one, f_nan, f_nan, rm,
		   APFloat::fs_ok));
    }



  // Subtractions; pos_infinity lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (sub (f_pos_infinity, f_pos_infinity, f_nan, rm,
		   APFloat::fs_invalid_op));
      assert (sub (f_pos_infinity, f_neg_infinity, f_pos_infinity, rm,
		   APFloat::fs_ok));
      assert (sub (f_pos_infinity, f_pos_zero, f_pos_infinity, rm,
		   APFloat::fs_ok));
      assert (sub (f_pos_infinity, f_neg_zero, f_pos_infinity, rm,
		   APFloat::fs_ok));
      assert (sub (f_pos_infinity, f_one, f_pos_infinity, rm,
		   APFloat::fs_ok));
      assert (sub (f_pos_infinity, f_neg_one, f_pos_infinity, rm,
		   APFloat::fs_ok));
      assert (sub (f_pos_infinity, f_nan, f_nan, rm,
		   APFloat::fs_ok));
    }

  // Subtractions; neg_infinity lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (sub (f_neg_infinity, f_pos_infinity, f_neg_infinity, rm,
		   APFloat::fs_ok));
      assert (sub (f_neg_infinity, f_neg_infinity, f_nan, rm,
		   APFloat::fs_invalid_op));
      assert (sub (f_neg_infinity, f_pos_zero, f_neg_infinity, rm,
		   APFloat::fs_ok));
      assert (sub (f_neg_infinity, f_neg_zero, f_neg_infinity, rm,
		   APFloat::fs_ok));
      assert (sub (f_neg_infinity, f_one, f_neg_infinity, rm,
		   APFloat::fs_ok));
      assert (sub (f_neg_infinity, f_neg_one, f_neg_infinity, rm,
		   APFloat::fs_ok));
      assert (sub (f_neg_infinity, f_nan, f_nan, rm,
		   APFloat::fs_ok));
    }

  // Subtractions; pos_zero lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (sub (f_pos_zero, f_pos_infinity, f_neg_infinity, rm,
		   APFloat::fs_ok));
      assert (sub (f_pos_zero, f_neg_infinity, f_pos_infinity, rm,
		   APFloat::fs_ok));
      assert (sub (f_pos_zero, f_pos_zero, rm == APFloat::frm_to_minus_infinity
		   ? f_neg_zero: f_pos_zero, rm, APFloat::fs_ok));
      assert (sub (f_pos_zero, f_neg_zero, f_pos_zero, rm,
		   APFloat::fs_ok));
      assert (sub (f_pos_zero, f_one, f_neg_one, rm,
		   APFloat::fs_ok));
      assert (sub (f_pos_zero, f_neg_one, f_one, rm,
		   APFloat::fs_ok));
      assert (sub (f_pos_zero, f_nan, f_nan, rm,
		   APFloat::fs_ok));
    }

  // Subtractions; neg_zero lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (sub (f_neg_zero, f_pos_infinity, f_neg_infinity, rm,
		   APFloat::fs_ok));
      assert (sub (f_neg_zero, f_neg_infinity, f_pos_infinity, rm,
		   APFloat::fs_ok));
      assert (sub (f_neg_zero, f_pos_zero, f_neg_zero, rm,
		   APFloat::fs_ok));
      assert (sub (f_neg_zero, f_neg_zero, rm == APFloat::frm_to_minus_infinity
		   ? f_neg_zero: f_pos_zero, rm, APFloat::fs_ok));
      assert (sub (f_neg_zero, f_one, f_neg_one, rm,
		   APFloat::fs_ok));
      assert (sub (f_neg_zero, f_neg_one, f_one, rm,
		   APFloat::fs_ok));
      assert (sub (f_neg_zero, f_nan, f_nan, rm,
		   APFloat::fs_ok));
    }

  // Subtractions; nan lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (sub (f_nan, f_pos_infinity, f_nan, rm,
		   APFloat::fs_ok));
      assert (sub (f_nan, f_neg_infinity, f_nan, rm,
		   APFloat::fs_ok));
      assert (sub (f_nan, f_pos_zero, f_nan, rm,
		   APFloat::fs_ok));
      assert (sub (f_nan, f_neg_zero, f_nan, rm,
		   APFloat::fs_ok));
      assert (sub (f_nan, f_one, f_nan, rm,
		   APFloat::fs_ok));
      assert (sub (f_nan, f_neg_one, f_nan, rm,
		   APFloat::fs_ok));
      assert (sub (f_nan, f_nan, f_nan, rm,
		   APFloat::fs_ok));
    }

  // Subtractions; one lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (sub (f_one, f_pos_infinity, f_neg_infinity, rm,
		   APFloat::fs_ok));
      assert (sub (f_one, f_neg_infinity, f_pos_infinity, rm,
		   APFloat::fs_ok));
      assert (sub (f_one, f_pos_zero, f_one, rm,
		   APFloat::fs_ok));
      assert (sub (f_one, f_neg_zero, f_one, rm,
		   APFloat::fs_ok));
      assert (sub (f_one, f_one, rm == APFloat::frm_to_minus_infinity
		   ? f_neg_zero: f_pos_zero, rm, APFloat::fs_ok));
      assert (sub (f_one, f_neg_one, f_two, rm,
		   APFloat::fs_ok));
      assert (sub (f_one, f_nan, f_nan, rm,
		   APFloat::fs_ok));
    }

  // Subtractions; neg_one lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (sub (f_neg_one, f_pos_infinity, f_neg_infinity, rm,
		   APFloat::fs_ok));
      assert (sub (f_neg_one, f_neg_infinity, f_pos_infinity, rm,
		   APFloat::fs_ok));
      assert (sub (f_neg_one, f_pos_zero, f_neg_one, rm,
		   APFloat::fs_ok));
      assert (sub (f_neg_one, f_neg_zero, f_neg_one, rm,
		   APFloat::fs_ok));
      assert (sub (f_neg_one, f_one, f_neg_two, rm,
		   APFloat::fs_ok));
      assert (sub (f_neg_one, f_neg_one, rm == APFloat::frm_to_minus_infinity
		   ? f_neg_zero: f_pos_zero, rm, APFloat::fs_ok));
      assert (sub (f_neg_one, f_nan, f_nan, rm,
		   APFloat::fs_ok));
    }

  // FMA, NaN somewhere.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (fma (f_pos_infinity, f_pos_infinity, f_nan, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_pos_infinity, f_neg_infinity, f_nan, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_neg_infinity, f_pos_infinity, f_nan, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_neg_infinity, f_neg_infinity, f_nan, f_nan, rm,
		   APFloat::fs_ok));

      assert (fma (f_pos_infinity, f_nan, f_pos_infinity, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_pos_infinity, f_nan, f_neg_infinity, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_neg_infinity, f_nan, f_pos_infinity, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_neg_infinity, f_nan, f_neg_infinity, f_nan, rm,
		   APFloat::fs_ok));

      assert (fma (f_nan, f_pos_infinity, f_pos_infinity, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_nan, f_pos_infinity, f_neg_infinity, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_nan, f_neg_infinity, f_pos_infinity, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_nan, f_neg_infinity, f_neg_infinity, f_nan, rm,
		   APFloat::fs_ok));

      assert (fma (f_pos_zero, f_pos_zero, f_nan, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_pos_zero, f_neg_zero, f_nan, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_neg_zero, f_pos_zero, f_nan, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_neg_zero, f_neg_zero, f_nan, f_nan, rm,
		   APFloat::fs_ok));

      assert (fma (f_pos_zero, f_nan, f_pos_zero, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_pos_zero, f_nan, f_neg_zero, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_neg_zero, f_nan, f_pos_zero, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_neg_zero, f_nan, f_neg_zero, f_nan, rm,
		   APFloat::fs_ok));

      assert (fma (f_nan, f_pos_zero, f_pos_zero, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_nan, f_pos_zero, f_neg_zero, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_nan, f_neg_zero, f_pos_zero, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_nan, f_neg_zero, f_neg_zero, f_nan, rm,
		   APFloat::fs_ok));

      assert (fma (f_pos_infinity, f_pos_zero, f_nan, f_nan, rm,
		   APFloat::fs_invalid_op));
      assert (fma (f_pos_infinity, f_neg_zero, f_nan, f_nan, rm,
		   APFloat::fs_invalid_op));
      assert (fma (f_neg_zero, f_pos_infinity, f_nan, f_nan, rm,
		   APFloat::fs_invalid_op));
      assert (fma (f_neg_zero, f_neg_infinity, f_nan, f_nan, rm,
		   APFloat::fs_invalid_op));
      assert (fma (f_neg_zero, f_neg_zero, f_nan, f_nan, rm,
		   APFloat::fs_ok));

      assert (fma (f_pos_infinity, f_nan, f_pos_zero, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_pos_infinity, f_nan, f_neg_zero, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_neg_zero, f_nan, f_pos_infinity, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_neg_zero, f_nan, f_neg_zero, f_nan, rm,
		   APFloat::fs_ok));

      assert (fma (f_nan, f_pos_infinity, f_pos_zero, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_nan, f_pos_infinity, f_neg_zero, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_nan, f_neg_zero, f_pos_infinity, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_nan, f_neg_zero, f_neg_zero, f_nan, rm,
		   APFloat::fs_ok));

      assert (fma (f_pos_infinity, f_one, f_nan, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_pos_infinity, f_neg_one, f_nan, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_neg_one, f_pos_infinity, f_nan, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_neg_one, f_neg_one, f_nan, f_nan, rm,
		   APFloat::fs_ok));

      assert (fma (f_pos_infinity, f_nan, f_one, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_pos_infinity, f_nan, f_neg_one, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_neg_one, f_nan, f_pos_infinity, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_neg_one, f_nan, f_neg_one, f_nan, rm,
		   APFloat::fs_ok));

      assert (fma (f_nan, f_pos_infinity, f_one, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_nan, f_pos_infinity, f_neg_one, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_nan, f_neg_one, f_pos_infinity, f_nan, rm,
		   APFloat::fs_ok));
      assert (fma (f_nan, f_neg_one, f_neg_one, f_nan, rm,
		   APFloat::fs_ok));
    }

  // FMA, non-NaN +inf first.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (fma (f_pos_infinity, f_pos_infinity, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_pos_infinity, f_pos_infinity, f_neg_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_pos_infinity, f_pos_infinity, f_pos_zero,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_pos_infinity, f_pos_infinity, f_neg_zero,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_pos_infinity, f_pos_infinity, f_one,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_pos_infinity, f_pos_infinity, f_neg_one,
		   f_pos_infinity, rm, APFloat::fs_ok));

      assert (fma (f_pos_infinity, f_neg_infinity, f_pos_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_pos_infinity, f_neg_infinity, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_pos_infinity, f_neg_infinity, f_pos_zero,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_pos_infinity, f_neg_infinity, f_neg_zero,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_pos_infinity, f_neg_infinity, f_one,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_pos_infinity, f_neg_infinity, f_neg_one,
		   f_neg_infinity, rm, APFloat::fs_ok));

      assert (fma (f_pos_infinity, f_pos_zero, f_pos_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_pos_infinity, f_pos_zero, f_neg_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_pos_infinity, f_pos_zero, f_pos_zero,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_pos_infinity, f_pos_zero, f_neg_zero,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_pos_infinity, f_pos_zero, f_one,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_pos_infinity, f_pos_zero, f_neg_one,
		   f_nan, rm, APFloat::fs_invalid_op));

      assert (fma (f_pos_infinity, f_neg_zero, f_pos_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_pos_infinity, f_neg_zero, f_neg_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_pos_infinity, f_neg_zero, f_pos_zero,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_pos_infinity, f_neg_zero, f_neg_zero,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_pos_infinity, f_neg_zero, f_one,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_pos_infinity, f_neg_zero, f_neg_one,
		   f_nan, rm, APFloat::fs_invalid_op));

      assert (fma (f_pos_infinity, f_neg_one, f_pos_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_pos_infinity, f_neg_one, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_pos_infinity, f_neg_one, f_pos_zero,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_pos_infinity, f_neg_one, f_neg_zero,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_pos_infinity, f_neg_one, f_one,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_pos_infinity, f_neg_one, f_neg_one,
		   f_neg_infinity, rm, APFloat::fs_ok));
    }

  // FMA, non-NaN -inf first.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (fma (f_neg_infinity, f_pos_infinity, f_pos_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_neg_infinity, f_pos_infinity, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_infinity, f_pos_infinity, f_pos_zero,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_infinity, f_pos_infinity, f_neg_zero,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_infinity, f_pos_infinity, f_one,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_infinity, f_pos_infinity, f_neg_one,
		   f_neg_infinity, rm, APFloat::fs_ok));

      assert (fma (f_neg_infinity, f_neg_infinity, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_infinity, f_neg_infinity, f_neg_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_neg_infinity, f_neg_infinity, f_pos_zero,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_infinity, f_neg_infinity, f_neg_zero,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_infinity, f_neg_infinity, f_one,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_infinity, f_neg_infinity, f_neg_one,
		   f_pos_infinity, rm, APFloat::fs_ok));

      assert (fma (f_neg_infinity, f_pos_zero, f_pos_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_neg_infinity, f_pos_zero, f_neg_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_neg_infinity, f_pos_zero, f_pos_zero,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_neg_infinity, f_pos_zero, f_neg_zero,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_neg_infinity, f_pos_zero, f_one,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_neg_infinity, f_pos_zero, f_neg_one,
		   f_nan, rm, APFloat::fs_invalid_op));

      assert (fma (f_neg_infinity, f_neg_zero, f_pos_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_neg_infinity, f_neg_zero, f_neg_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_neg_infinity, f_neg_zero, f_pos_zero,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_neg_infinity, f_neg_zero, f_neg_zero,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_neg_infinity, f_neg_zero, f_one,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_neg_infinity, f_neg_zero, f_neg_one,
		   f_nan, rm, APFloat::fs_invalid_op));

      assert (fma (f_neg_infinity, f_neg_one, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_infinity, f_neg_one, f_neg_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_neg_infinity, f_neg_one, f_pos_zero,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_infinity, f_neg_one, f_neg_zero,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_infinity, f_neg_one, f_one,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_infinity, f_neg_one, f_neg_one,
		   f_pos_infinity, rm, APFloat::fs_ok));
    }

  // FMA, non-NaN +zero first.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      bool down = rm == APFloat::frm_to_minus_infinity;

      assert (fma (f_pos_zero, f_pos_infinity, f_pos_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_pos_zero, f_pos_infinity, f_neg_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_pos_zero, f_pos_infinity, f_pos_zero,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_pos_zero, f_pos_infinity, f_neg_zero,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_pos_zero, f_pos_infinity, f_one,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_pos_zero, f_pos_infinity, f_neg_one,
		   f_nan, rm, APFloat::fs_invalid_op));

      assert (fma (f_pos_zero, f_neg_infinity, f_pos_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_pos_zero, f_neg_infinity, f_neg_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_pos_zero, f_neg_infinity, f_pos_zero,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_pos_zero, f_neg_infinity, f_neg_zero,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_pos_zero, f_neg_infinity, f_one,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_pos_zero, f_neg_infinity, f_neg_one,
		   f_nan, rm, APFloat::fs_invalid_op));

      assert (fma (f_pos_zero, f_pos_zero, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_pos_zero, f_pos_zero, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_pos_zero, f_pos_zero, f_pos_zero,
		   f_pos_zero, rm, APFloat::fs_ok));
      assert (fma (f_pos_zero, f_pos_zero, f_neg_zero,
		   down ? f_neg_zero: f_pos_zero, rm, APFloat::fs_ok));
      assert (fma (f_pos_zero, f_pos_zero, f_one,
		   f_one, rm, APFloat::fs_ok));
      assert (fma (f_pos_zero, f_pos_zero, f_neg_one,
		   f_neg_one, rm, APFloat::fs_ok));

      assert (fma (f_pos_zero, f_neg_zero, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_pos_zero, f_neg_zero, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_pos_zero, f_neg_zero, f_pos_zero,
		   down ? f_neg_zero: f_pos_zero, rm, APFloat::fs_ok));
      assert (fma (f_pos_zero, f_neg_zero, f_neg_zero,
		   f_neg_zero, rm, APFloat::fs_ok));
      assert (fma (f_pos_zero, f_neg_zero, f_one,
		   f_one, rm, APFloat::fs_ok));
      assert (fma (f_pos_zero, f_neg_zero, f_neg_one,
		   f_neg_one, rm, APFloat::fs_ok));

      assert (fma (f_pos_zero, f_neg_one, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_pos_zero, f_neg_one, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_pos_zero, f_neg_one, f_pos_zero,
		   down ? f_neg_zero: f_pos_zero, rm, APFloat::fs_ok));
      assert (fma (f_pos_zero, f_neg_one, f_neg_zero,
		   f_neg_zero, rm, APFloat::fs_ok));
      assert (fma (f_pos_zero, f_neg_one, f_one,
		   f_one, rm, APFloat::fs_ok));
      assert (fma (f_pos_zero, f_neg_one, f_neg_one,
		   f_neg_one, rm, APFloat::fs_ok));
    }

  // FMA, non-NaN -zero first.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      bool down = rm == APFloat::frm_to_minus_infinity;

      assert (fma (f_neg_zero, f_pos_infinity, f_pos_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_neg_zero, f_pos_infinity, f_neg_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_neg_zero, f_pos_infinity, f_pos_zero,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_neg_zero, f_pos_infinity, f_neg_zero,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_neg_zero, f_pos_infinity, f_one,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_neg_zero, f_pos_infinity, f_neg_one,
		   f_nan, rm, APFloat::fs_invalid_op));

      assert (fma (f_neg_zero, f_neg_infinity, f_pos_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_neg_zero, f_neg_infinity, f_neg_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_neg_zero, f_neg_infinity, f_pos_zero,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_neg_zero, f_neg_infinity, f_neg_zero,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_neg_zero, f_neg_infinity, f_one,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_neg_zero, f_neg_infinity, f_neg_one,
		   f_nan, rm, APFloat::fs_invalid_op));

      assert (fma (f_neg_zero, f_pos_zero, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_zero, f_pos_zero, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_zero, f_pos_zero, f_pos_zero,
		   down ? f_neg_zero: f_pos_zero, rm, APFloat::fs_ok));
      assert (fma (f_neg_zero, f_pos_zero, f_neg_zero,
		   f_neg_zero, rm, APFloat::fs_ok));
      assert (fma (f_neg_zero, f_pos_zero, f_one,
		   f_one, rm, APFloat::fs_ok));
      assert (fma (f_neg_zero, f_pos_zero, f_neg_one,
		   f_neg_one, rm, APFloat::fs_ok));

      assert (fma (f_neg_zero, f_neg_zero, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_zero, f_neg_zero, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_zero, f_neg_zero, f_pos_zero,
		   f_pos_zero, rm, APFloat::fs_ok));
      assert (fma (f_neg_zero, f_neg_zero, f_neg_zero,
		   down ? f_neg_zero: f_pos_zero, rm, APFloat::fs_ok));
      assert (fma (f_neg_zero, f_neg_zero, f_one,
		   f_one, rm, APFloat::fs_ok));
      assert (fma (f_neg_zero, f_neg_zero, f_neg_one,
		   f_neg_one, rm, APFloat::fs_ok));

      assert (fma (f_neg_zero, f_neg_one, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_zero, f_neg_one, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_zero, f_neg_one, f_pos_zero,
		   f_pos_zero, rm, APFloat::fs_ok));
      assert (fma (f_neg_zero, f_neg_one, f_neg_zero,
		   down ? f_neg_zero: f_pos_zero, rm, APFloat::fs_ok));
      assert (fma (f_neg_zero, f_neg_one, f_one,
		   f_one, rm, APFloat::fs_ok));
      assert (fma (f_neg_zero, f_neg_one, f_neg_one,
		   f_neg_one, rm, APFloat::fs_ok));
    }

  // FMA, non-NaN +one first.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      bool down = rm == APFloat::frm_to_minus_infinity;

      assert (fma (f_one, f_pos_infinity, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_one, f_pos_infinity, f_neg_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_one, f_pos_infinity, f_pos_zero,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_one, f_pos_infinity, f_neg_zero,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_one, f_pos_infinity, f_one,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_one, f_pos_infinity, f_neg_one,
		   f_pos_infinity, rm, APFloat::fs_ok));

      assert (fma (f_one, f_neg_infinity, f_pos_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_one, f_neg_infinity, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_one, f_neg_infinity, f_pos_zero,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_one, f_neg_infinity, f_neg_zero,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_one, f_neg_infinity, f_one,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_one, f_neg_infinity, f_neg_one,
		   f_neg_infinity, rm, APFloat::fs_ok));

      assert (fma (f_one, f_pos_zero, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_one, f_pos_zero, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_one, f_pos_zero, f_pos_zero,
		   f_pos_zero, rm, APFloat::fs_ok));
      assert (fma (f_one, f_pos_zero, f_neg_zero,
		   down ? f_neg_zero: f_pos_zero, rm, APFloat::fs_ok));
      assert (fma (f_one, f_pos_zero, f_one,
		   f_one, rm, APFloat::fs_ok));
      assert (fma (f_one, f_pos_zero, f_neg_one,
		   f_neg_one, rm, APFloat::fs_ok));

      assert (fma (f_one, f_neg_zero, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_one, f_neg_zero, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_one, f_neg_zero, f_pos_zero,
		   down ? f_neg_zero: f_pos_zero, rm, APFloat::fs_ok));
      assert (fma (f_one, f_neg_zero, f_neg_zero,
		   f_neg_zero, rm, APFloat::fs_ok));
      assert (fma (f_one, f_neg_zero, f_one,
		   f_one, rm, APFloat::fs_ok));
      assert (fma (f_one, f_neg_zero, f_neg_one,
		   f_neg_one, rm, APFloat::fs_ok));

      assert (fma (f_one, f_neg_one, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_one, f_neg_one, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_one, f_neg_one, f_pos_zero,
		   f_neg_one, rm, APFloat::fs_ok));
      assert (fma (f_one, f_neg_one, f_neg_zero,
		   f_neg_one, rm, APFloat::fs_ok));
      assert (fma (f_one, f_neg_one, f_one,
		   down ? f_neg_zero: f_pos_zero, rm, APFloat::fs_ok));
      assert (fma (f_one, f_neg_one, f_neg_one,
		   f_neg_two, rm, APFloat::fs_ok));
    }

  // FMA, non-NaN -one first.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      bool down = rm == APFloat::frm_to_minus_infinity;

      assert (fma (f_neg_one, f_pos_infinity, f_pos_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_neg_one, f_pos_infinity, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_one, f_pos_infinity, f_pos_zero,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_one, f_pos_infinity, f_neg_zero,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_one, f_pos_infinity, f_one,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_one, f_pos_infinity, f_neg_one,
		   f_neg_infinity, rm, APFloat::fs_ok));

      assert (fma (f_neg_one, f_neg_infinity, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_one, f_neg_infinity, f_neg_infinity,
		   f_nan, rm, APFloat::fs_invalid_op));
      assert (fma (f_neg_one, f_neg_infinity, f_pos_zero,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_one, f_neg_infinity, f_neg_zero,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_one, f_neg_infinity, f_one,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_one, f_neg_infinity, f_neg_one,
		   f_pos_infinity, rm, APFloat::fs_ok));

      assert (fma (f_neg_one, f_pos_zero, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_one, f_pos_zero, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_one, f_pos_zero, f_pos_zero,
		   down ? f_neg_zero: f_pos_zero, rm, APFloat::fs_ok));
      assert (fma (f_neg_one, f_pos_zero, f_neg_zero,
		   f_neg_zero, rm, APFloat::fs_ok));
      assert (fma (f_neg_one, f_pos_zero, f_one,
		   f_one, rm, APFloat::fs_ok));
      assert (fma (f_neg_one, f_pos_zero, f_neg_one,
		   f_neg_one, rm, APFloat::fs_ok));

      assert (fma (f_neg_one, f_neg_zero, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_one, f_neg_zero, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_one, f_neg_zero, f_pos_zero,
		   f_pos_zero, rm, APFloat::fs_ok));
      assert (fma (f_neg_one, f_neg_zero, f_neg_zero,
		   down ? f_neg_zero: f_pos_zero, rm, APFloat::fs_ok));
      assert (fma (f_neg_one, f_neg_zero, f_one,
		   f_one, rm, APFloat::fs_ok));
      assert (fma (f_neg_one, f_neg_zero, f_neg_one,
		   f_neg_one, rm, APFloat::fs_ok));

      assert (fma (f_neg_one, f_neg_one, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_one, f_neg_one, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::fs_ok));
      assert (fma (f_neg_one, f_neg_one, f_pos_zero,
		   f_one, rm, APFloat::fs_ok));
      assert (fma (f_neg_one, f_neg_one, f_neg_zero,
		   f_one, rm, APFloat::fs_ok));
      assert (fma (f_neg_one, f_neg_one, f_one,
		   f_two, rm, APFloat::fs_ok));
      assert (fma (f_neg_one, f_neg_one, f_neg_one,
		   down ? f_neg_zero: f_pos_zero, rm, APFloat::fs_ok));
    }

  return 0;
}
