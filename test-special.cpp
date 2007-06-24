/*
   Copyright 2007 Neil Booth.

   See the file "COPYING" for information about the copyright
   and warranty status of this software.
*/

#include <cassert>
#include "APFloat.h"

using namespace llvm;

static bool
good_compare (const APFloat &lhs, const APFloat &rhs)
{
  if (lhs.getCategory () == APFloat::fcQNaN)
    return rhs.getCategory () == APFloat::fcQNaN;

  return (lhs.compare (rhs) == APFloat::cmpEqual
	  && lhs.isNegative() == rhs.isNegative());
}

static APFloat::cmpResult
compare (const APFloat &lhs, const APFloat &rhs)
{
  return lhs.compare (rhs);
}

static bool
convert (const APFloat &lhs, const APFloat &result,
	APFloat::roundingMode rounding_mode, APFloat::opStatus status)
{
  APFloat tmp (lhs);

  if (tmp.convert (result.getSemantics (), rounding_mode) != status)
    return false;

  return good_compare (tmp, result);
}

static bool
divide (const APFloat &lhs, const APFloat &rhs, const APFloat &result,
	APFloat::roundingMode rounding_mode, APFloat::opStatus status)
{
  APFloat tmp (lhs);

  if (tmp.divide (rhs, rounding_mode) != status)
    return false;

  return good_compare (tmp, result);
}

static bool
mult (const APFloat &lhs, const APFloat &rhs, const APFloat &result,
      APFloat::roundingMode rounding_mode, APFloat::opStatus status)
{
  APFloat tmp (lhs);

  if (tmp.multiply (rhs, rounding_mode) != status)
    return false;

  return good_compare (tmp, result);
}

static bool
add (const APFloat &lhs, const APFloat &rhs, const APFloat &result,
     APFloat::roundingMode rounding_mode, APFloat::opStatus status)
{
  APFloat tmp (lhs);

  if (tmp.add (rhs, rounding_mode) != status)
    return false;

  return good_compare (tmp, result);
}

static bool
sub (const APFloat &lhs, const APFloat &rhs, const APFloat &result,
     APFloat::roundingMode rounding_mode, APFloat::opStatus status)
{
  APFloat tmp (lhs);

  if (tmp.subtract (rhs, rounding_mode) != status)
    return false;

  return good_compare (tmp, result);
}

static bool
fma (const APFloat &lhs, const APFloat &m, const APFloat &a,
     const APFloat &result, APFloat::roundingMode rounding_mode,
     APFloat::opStatus status)
{
  APFloat tmp (lhs);

  if (tmp.fusedMultiplyAdd (m, a, rounding_mode) != status)
    return false;

  return good_compare (tmp, result);
}

int main (void)
{
  APFloat f_pos_infinity (APFloat::IEEEsingle, APFloat::fcInfinity, false);
  APFloat f_neg_infinity (APFloat::IEEEsingle, APFloat::fcInfinity, true);
  APFloat f_pos_zero (APFloat::IEEEsingle, APFloat::fcZero, false);
  APFloat f_neg_zero (APFloat::IEEEsingle, APFloat::fcZero, true);
  APFloat f_nan (APFloat::IEEEsingle, APFloat::fcQNaN, false);
  APFloat f_one (APFloat::IEEEsingle, 1);
  APFloat d_one (APFloat::IEEEdouble, 1);
  APFloat f_two (APFloat::IEEEsingle, 2);
  APFloat f_neg_one (f_one);
  APFloat f_neg_two (f_two);
  APFloat d_neg_one (f_one);

  f_neg_one.changeSign ();
  f_neg_two.changeSign ();
  d_neg_one.changeSign ();

  // Conversions.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (convert (f_one, d_one, rm, APFloat::opOK));
      assert (convert (f_neg_one, d_neg_one, rm, APFloat::opOK));
      assert (convert (d_one, f_one, rm, APFloat::opOK));
      assert (convert (d_neg_one, f_neg_one, rm, APFloat::opOK));
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
		      APFloat::opInvalidOp));
      assert (divide (f_pos_infinity, f_neg_infinity, f_nan, rm,
		      APFloat::opInvalidOp));
      assert (divide (f_pos_infinity, f_pos_zero, f_pos_infinity, rm,
		      APFloat::opOK));
      assert (divide (f_pos_infinity, f_neg_zero, f_neg_infinity, rm,
		      APFloat::opOK));
      assert (divide (f_pos_infinity, f_one, f_pos_infinity, rm,
		      APFloat::opOK));
      assert (divide (f_pos_infinity, f_neg_one, f_neg_infinity, rm,
		      APFloat::opOK));
      assert (divide (f_pos_infinity, f_nan, f_nan, rm,
		      APFloat::opOK));
    }

  // Divisions; neg_infinity lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (divide (f_neg_infinity, f_pos_infinity, f_nan, rm,
		      APFloat::opInvalidOp));
      assert (divide (f_neg_infinity, f_neg_infinity, f_nan, rm,
		      APFloat::opInvalidOp));
      assert (divide (f_neg_infinity, f_pos_zero, f_neg_infinity, rm,
		      APFloat::opOK));
      assert (divide (f_neg_infinity, f_neg_zero, f_pos_infinity, rm,
		      APFloat::opOK));
      assert (divide (f_neg_infinity, f_one, f_neg_infinity, rm,
		      APFloat::opOK));
      assert (divide (f_neg_infinity, f_neg_one, f_pos_infinity, rm,
		      APFloat::opOK));
      assert (divide (f_neg_infinity, f_nan, f_nan, rm,
		      APFloat::opOK));
    }

  // Divisions; pos_zero lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (divide (f_pos_zero, f_pos_infinity, f_pos_zero, rm,
		      APFloat::opOK));
      assert (divide (f_pos_zero, f_neg_infinity, f_neg_zero, rm,
		      APFloat::opOK));
      assert (divide (f_pos_zero, f_pos_zero, f_nan, rm,
		      APFloat::opInvalidOp));
      assert (divide (f_pos_zero, f_neg_zero, f_nan, rm,
		      APFloat::opInvalidOp));
      assert (divide (f_pos_zero, f_one, f_pos_zero, rm,
		      APFloat::opOK));
      assert (divide (f_pos_zero, f_neg_one, f_neg_zero, rm,
		      APFloat::opOK));
      assert (divide (f_pos_zero, f_nan, f_nan, rm,
		      APFloat::opOK));
    }

  // Divisions; neg_zero lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (divide (f_neg_zero, f_pos_infinity, f_neg_zero, rm,
		      APFloat::opOK));
      assert (divide (f_neg_zero, f_neg_infinity, f_pos_zero, rm,
		      APFloat::opOK));
      assert (divide (f_neg_zero, f_pos_zero, f_nan, rm,
		      APFloat::opInvalidOp));
      assert (divide (f_neg_zero, f_neg_zero, f_nan, rm,
		      APFloat::opInvalidOp));
      assert (divide (f_neg_zero, f_one, f_neg_zero, rm,
		      APFloat::opOK));
      assert (divide (f_neg_zero, f_neg_one, f_pos_zero, rm,
		      APFloat::opOK));
      assert (divide (f_neg_zero, f_nan, f_nan, rm,
		      APFloat::opOK));
    }

  // Divisions; nan lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (divide (f_nan, f_pos_infinity, f_nan, rm,
		      APFloat::opOK));
      assert (divide (f_nan, f_neg_infinity, f_nan, rm,
		      APFloat::opOK));
      assert (divide (f_nan, f_pos_zero, f_nan, rm,
		      APFloat::opOK));
      assert (divide (f_nan, f_neg_zero, f_nan, rm,
		      APFloat::opOK));
      assert (divide (f_nan, f_one, f_nan, rm,
		      APFloat::opOK));
      assert (divide (f_nan, f_neg_one, f_nan, rm,
		      APFloat::opOK));
      assert (divide (f_nan, f_nan, f_nan, rm,
		      APFloat::opOK));
    }

  // Divisions; one lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (divide (f_one, f_pos_infinity, f_pos_zero, rm,
		      APFloat::opOK));
      assert (divide (f_one, f_neg_infinity, f_neg_zero, rm,
		      APFloat::opOK));
      assert (divide (f_one, f_pos_zero, f_pos_infinity, rm,
		      APFloat::opDivByZero));
      assert (divide (f_one, f_neg_zero, f_neg_infinity, rm,
		      APFloat::opDivByZero));
      assert (divide (f_one, f_one, f_one, rm,
		      APFloat::opOK));
      assert (divide (f_one, f_neg_one, f_neg_one, rm,
		      APFloat::opOK));
      assert (divide (f_one, f_nan, f_nan, rm,
		      APFloat::opOK));
    }

  // Divisions; neg_one lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (divide (f_neg_one, f_pos_infinity, f_neg_zero, rm,
		      APFloat::opOK));
      assert (divide (f_neg_one, f_neg_infinity, f_pos_zero, rm,
		      APFloat::opOK));
      assert (divide (f_neg_one, f_pos_zero, f_neg_infinity, rm,
		      APFloat::opDivByZero));
      assert (divide (f_neg_one, f_neg_zero, f_pos_infinity, rm,
		      APFloat::opDivByZero));
      assert (divide (f_neg_one, f_one, f_neg_one, rm,
		      APFloat::opOK));
      assert (divide (f_neg_one, f_neg_one, f_one, rm,
		      APFloat::opOK));
      assert (divide (f_neg_one, f_nan, f_nan, rm,
		      APFloat::opOK));
    }



  // Multiplications; pos_infinity lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (mult (f_pos_infinity, f_pos_infinity, f_pos_infinity, rm,
		    APFloat::opOK));
      assert (mult (f_pos_infinity, f_neg_infinity, f_neg_infinity, rm,
		    APFloat::opOK));
      assert (mult (f_pos_infinity, f_pos_zero, f_nan, rm,
		    APFloat::opInvalidOp));
      assert (mult (f_pos_infinity, f_neg_zero, f_nan, rm,
		    APFloat::opInvalidOp));
      assert (mult (f_pos_infinity, f_one, f_pos_infinity, rm,
		    APFloat::opOK));
      assert (mult (f_pos_infinity, f_neg_one, f_neg_infinity, rm,
		    APFloat::opOK));
      assert (mult (f_pos_infinity, f_nan, f_nan, rm,
		    APFloat::opOK));
    }

  // Multiplications; neg_infinity lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (mult (f_neg_infinity, f_pos_infinity, f_neg_infinity, rm,
		    APFloat::opOK));
      assert (mult (f_neg_infinity, f_neg_infinity, f_pos_infinity, rm,
		    APFloat::opOK));
      assert (mult (f_neg_infinity, f_pos_zero, f_nan, rm,
		    APFloat::opInvalidOp));
      assert (mult (f_neg_infinity, f_neg_zero, f_nan, rm,
		    APFloat::opInvalidOp));
      assert (mult (f_neg_infinity, f_one, f_neg_infinity, rm,
		    APFloat::opOK));
      assert (mult (f_neg_infinity, f_neg_one, f_pos_infinity, rm,
		    APFloat::opOK));
      assert (mult (f_neg_infinity, f_nan, f_nan, rm,
		    APFloat::opOK));
    }

  // Multiplications; pos_zero lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (mult (f_pos_zero, f_pos_infinity, f_nan, rm,
		    APFloat::opInvalidOp));
      assert (mult (f_pos_zero, f_neg_infinity, f_nan, rm,
		    APFloat::opInvalidOp));
      assert (mult (f_pos_zero, f_pos_zero, f_pos_zero, rm,
		    APFloat::opOK));
      assert (mult (f_pos_zero, f_neg_zero, f_neg_zero, rm,
		    APFloat::opOK));
      assert (mult (f_pos_zero, f_one, f_pos_zero, rm,
		    APFloat::opOK));
      assert (mult (f_pos_zero, f_neg_one, f_neg_zero, rm,
		    APFloat::opOK));
      assert (mult (f_pos_zero, f_nan, f_nan, rm,
		    APFloat::opOK));
    }

  // Multiplications; neg_zero lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (mult (f_neg_zero, f_pos_infinity, f_nan, rm,
		    APFloat::opInvalidOp));
      assert (mult (f_neg_zero, f_neg_infinity, f_nan, rm,
		    APFloat::opInvalidOp));
      assert (mult (f_neg_zero, f_pos_zero, f_neg_zero, rm,
		    APFloat::opOK));
      assert (mult (f_neg_zero, f_neg_zero, f_pos_zero, rm,
		    APFloat::opOK));
      assert (mult (f_neg_zero, f_one, f_neg_zero, rm,
		    APFloat::opOK));
      assert (mult (f_neg_zero, f_neg_one, f_pos_zero, rm,
		    APFloat::opOK));
      assert (mult (f_neg_zero, f_nan, f_nan, rm,
		    APFloat::opOK));
    }

  // Multiplications; nan lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (mult (f_nan, f_pos_infinity, f_nan, rm,
		    APFloat::opOK));
      assert (mult (f_nan, f_neg_infinity, f_nan, rm,
		    APFloat::opOK));
      assert (mult (f_nan, f_pos_zero, f_nan, rm,
		    APFloat::opOK));
      assert (mult (f_nan, f_neg_zero, f_nan, rm,
		    APFloat::opOK));
      assert (mult (f_nan, f_one, f_nan, rm,
		    APFloat::opOK));
      assert (mult (f_nan, f_neg_one, f_nan, rm,
		    APFloat::opOK));
      assert (mult (f_nan, f_nan, f_nan, rm,
		    APFloat::opOK));
    }

  // Multiplications; one lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (mult (f_one, f_pos_infinity, f_pos_infinity, rm,
		    APFloat::opOK));
      assert (mult (f_one, f_neg_infinity, f_neg_infinity, rm,
		    APFloat::opOK));
      assert (mult (f_one, f_pos_zero, f_pos_zero, rm,
		    APFloat::opOK));
      assert (mult (f_one, f_neg_zero, f_neg_zero, rm,
		    APFloat::opOK));
      assert (mult (f_one, f_one, f_one, rm,
		    APFloat::opOK));
      assert (mult (f_one, f_neg_one, f_neg_one, rm,
		    APFloat::opOK));
      assert (mult (f_one, f_nan, f_nan, rm,
		    APFloat::opOK));
    }

  // Multiplications; neg_one lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (mult (f_neg_one, f_pos_infinity, f_neg_infinity, rm,
		    APFloat::opOK));
      assert (mult (f_neg_one, f_neg_infinity, f_pos_infinity, rm,
		    APFloat::opOK));
      assert (mult (f_neg_one, f_pos_zero, f_neg_zero, rm,
		    APFloat::opOK));
      assert (mult (f_neg_one, f_neg_zero, f_pos_zero, rm,
		    APFloat::opOK));
      assert (mult (f_neg_one, f_one, f_neg_one, rm,
		    APFloat::opOK));
      assert (mult (f_neg_one, f_neg_one, f_one, rm,
		    APFloat::opOK));
      assert (mult (f_neg_one, f_nan, f_nan, rm,
		    APFloat::opOK));
    }



  // Additions; pos_infinity lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (add (f_pos_infinity, f_pos_infinity, f_pos_infinity, rm,
		   APFloat::opOK));
      assert (add (f_pos_infinity, f_neg_infinity, f_nan, rm,
		   APFloat::opInvalidOp));
      assert (add (f_pos_infinity, f_pos_zero, f_pos_infinity, rm,
		   APFloat::opOK));
      assert (add (f_pos_infinity, f_neg_zero, f_pos_infinity, rm,
		   APFloat::opOK));
      assert (add (f_pos_infinity, f_one, f_pos_infinity, rm,
		   APFloat::opOK));
      assert (add (f_pos_infinity, f_neg_one, f_pos_infinity, rm,
		   APFloat::opOK));
      assert (add (f_pos_infinity, f_nan, f_nan, rm,
		   APFloat::opOK));
    }

  // Additions; neg_infinity lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (add (f_neg_infinity, f_pos_infinity, f_nan, rm,
		   APFloat::opInvalidOp));
      assert (add (f_neg_infinity, f_neg_infinity, f_neg_infinity, rm,
		   APFloat::opOK));
      assert (add (f_neg_infinity, f_pos_zero, f_neg_infinity, rm,
		   APFloat::opOK));
      assert (add (f_neg_infinity, f_neg_zero, f_neg_infinity, rm,
		   APFloat::opOK));
      assert (add (f_neg_infinity, f_one, f_neg_infinity, rm,
		   APFloat::opOK));
      assert (add (f_neg_infinity, f_neg_one, f_neg_infinity, rm,
		   APFloat::opOK));
      assert (add (f_neg_infinity, f_nan, f_nan, rm,
		   APFloat::opOK));
    }

  // Additions; pos_zero lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (add (f_pos_zero, f_pos_infinity, f_pos_infinity, rm,
		   APFloat::opOK));
      assert (add (f_pos_zero, f_neg_infinity, f_neg_infinity, rm,
		   APFloat::opOK));
      assert (add (f_pos_zero, f_pos_zero, f_pos_zero, rm,
		   APFloat::opOK));
      assert (add (f_pos_zero, f_neg_zero, rm == APFloat::rmTowardNegative
		   ? f_neg_zero: f_pos_zero, rm, APFloat::opOK));
      assert (add (f_pos_zero, f_one, f_one, rm,
		   APFloat::opOK));
      assert (add (f_pos_zero, f_neg_one, f_neg_one, rm,
		   APFloat::opOK));
      assert (add (f_pos_zero, f_nan, f_nan, rm,
		   APFloat::opOK));
    }

  // Additions; neg_zero lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (add (f_neg_zero, f_pos_infinity, f_pos_infinity, rm,
		   APFloat::opOK));
      assert (add (f_neg_zero, f_neg_infinity, f_neg_infinity, rm,
		   APFloat::opOK));
      assert (add (f_neg_zero, f_pos_zero, rm == APFloat::rmTowardNegative
		   ? f_neg_zero: f_pos_zero, rm, APFloat::opOK));
      assert (add (f_neg_zero, f_neg_zero, f_neg_zero, rm,
		   APFloat::opOK));
      assert (add (f_neg_zero, f_one, f_one, rm,
		   APFloat::opOK));
      assert (add (f_neg_zero, f_neg_one, f_neg_one, rm,
		   APFloat::opOK));
      assert (add (f_neg_zero, f_nan, f_nan, rm,
		   APFloat::opOK));
    }

  // Additions; nan lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (add (f_nan, f_pos_infinity, f_nan, rm,
		   APFloat::opOK));
      assert (add (f_nan, f_neg_infinity, f_nan, rm,
		   APFloat::opOK));
      assert (add (f_nan, f_pos_zero, f_nan, rm,
		   APFloat::opOK));
      assert (add (f_nan, f_neg_zero, f_nan, rm,
		   APFloat::opOK));
      assert (add (f_nan, f_one, f_nan, rm,
		   APFloat::opOK));
      assert (add (f_nan, f_neg_one, f_nan, rm,
		   APFloat::opOK));
      assert (add (f_nan, f_nan, f_nan, rm,
		   APFloat::opOK));
    }

  // Additions; one lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (add (f_one, f_pos_infinity, f_pos_infinity, rm,
		   APFloat::opOK));
      assert (add (f_one, f_neg_infinity, f_neg_infinity, rm,
		   APFloat::opOK));
      assert (add (f_one, f_pos_zero, f_one, rm,
		   APFloat::opOK));
      assert (add (f_one, f_neg_zero, f_one, rm,
		   APFloat::opOK));
      assert (add (f_one, f_one, f_two, rm,
		   APFloat::opOK));
      assert (add (f_one, f_neg_one, rm == APFloat::rmTowardNegative
		   ? f_neg_zero: f_pos_zero, rm, APFloat::opOK));
      assert (add (f_one, f_nan, f_nan, rm,
		   APFloat::opOK));
    }

  // Additions; neg_one lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (add (f_neg_one, f_pos_infinity, f_pos_infinity, rm,
		   APFloat::opOK));
      assert (add (f_neg_one, f_neg_infinity, f_neg_infinity, rm,
		   APFloat::opOK));
      assert (add (f_neg_one, f_pos_zero, f_neg_one, rm,
		   APFloat::opOK));
      assert (add (f_neg_one, f_neg_zero, f_neg_one, rm,
		   APFloat::opOK));
      assert (add (f_neg_one, f_one, rm == APFloat::rmTowardNegative
		   ? f_neg_zero: f_pos_zero, rm, APFloat::opOK));
      assert (add (f_neg_one, f_neg_one, f_neg_two, rm,
		   APFloat::opOK));
      assert (add (f_neg_one, f_nan, f_nan, rm,
		   APFloat::opOK));
    }



  // Subtractions; pos_infinity lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (sub (f_pos_infinity, f_pos_infinity, f_nan, rm,
		   APFloat::opInvalidOp));
      assert (sub (f_pos_infinity, f_neg_infinity, f_pos_infinity, rm,
		   APFloat::opOK));
      assert (sub (f_pos_infinity, f_pos_zero, f_pos_infinity, rm,
		   APFloat::opOK));
      assert (sub (f_pos_infinity, f_neg_zero, f_pos_infinity, rm,
		   APFloat::opOK));
      assert (sub (f_pos_infinity, f_one, f_pos_infinity, rm,
		   APFloat::opOK));
      assert (sub (f_pos_infinity, f_neg_one, f_pos_infinity, rm,
		   APFloat::opOK));
      assert (sub (f_pos_infinity, f_nan, f_nan, rm,
		   APFloat::opOK));
    }

  // Subtractions; neg_infinity lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (sub (f_neg_infinity, f_pos_infinity, f_neg_infinity, rm,
		   APFloat::opOK));
      assert (sub (f_neg_infinity, f_neg_infinity, f_nan, rm,
		   APFloat::opInvalidOp));
      assert (sub (f_neg_infinity, f_pos_zero, f_neg_infinity, rm,
		   APFloat::opOK));
      assert (sub (f_neg_infinity, f_neg_zero, f_neg_infinity, rm,
		   APFloat::opOK));
      assert (sub (f_neg_infinity, f_one, f_neg_infinity, rm,
		   APFloat::opOK));
      assert (sub (f_neg_infinity, f_neg_one, f_neg_infinity, rm,
		   APFloat::opOK));
      assert (sub (f_neg_infinity, f_nan, f_nan, rm,
		   APFloat::opOK));
    }

  // Subtractions; pos_zero lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (sub (f_pos_zero, f_pos_infinity, f_neg_infinity, rm,
		   APFloat::opOK));
      assert (sub (f_pos_zero, f_neg_infinity, f_pos_infinity, rm,
		   APFloat::opOK));
      assert (sub (f_pos_zero, f_pos_zero, rm == APFloat::rmTowardNegative
		   ? f_neg_zero: f_pos_zero, rm, APFloat::opOK));
      assert (sub (f_pos_zero, f_neg_zero, f_pos_zero, rm,
		   APFloat::opOK));
      assert (sub (f_pos_zero, f_one, f_neg_one, rm,
		   APFloat::opOK));
      assert (sub (f_pos_zero, f_neg_one, f_one, rm,
		   APFloat::opOK));
      assert (sub (f_pos_zero, f_nan, f_nan, rm,
		   APFloat::opOK));
    }

  // Subtractions; neg_zero lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (sub (f_neg_zero, f_pos_infinity, f_neg_infinity, rm,
		   APFloat::opOK));
      assert (sub (f_neg_zero, f_neg_infinity, f_pos_infinity, rm,
		   APFloat::opOK));
      assert (sub (f_neg_zero, f_pos_zero, f_neg_zero, rm,
		   APFloat::opOK));
      assert (sub (f_neg_zero, f_neg_zero, rm == APFloat::rmTowardNegative
		   ? f_neg_zero: f_pos_zero, rm, APFloat::opOK));
      assert (sub (f_neg_zero, f_one, f_neg_one, rm,
		   APFloat::opOK));
      assert (sub (f_neg_zero, f_neg_one, f_one, rm,
		   APFloat::opOK));
      assert (sub (f_neg_zero, f_nan, f_nan, rm,
		   APFloat::opOK));
    }

  // Subtractions; nan lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (sub (f_nan, f_pos_infinity, f_nan, rm,
		   APFloat::opOK));
      assert (sub (f_nan, f_neg_infinity, f_nan, rm,
		   APFloat::opOK));
      assert (sub (f_nan, f_pos_zero, f_nan, rm,
		   APFloat::opOK));
      assert (sub (f_nan, f_neg_zero, f_nan, rm,
		   APFloat::opOK));
      assert (sub (f_nan, f_one, f_nan, rm,
		   APFloat::opOK));
      assert (sub (f_nan, f_neg_one, f_nan, rm,
		   APFloat::opOK));
      assert (sub (f_nan, f_nan, f_nan, rm,
		   APFloat::opOK));
    }

  // Subtractions; one lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (sub (f_one, f_pos_infinity, f_neg_infinity, rm,
		   APFloat::opOK));
      assert (sub (f_one, f_neg_infinity, f_pos_infinity, rm,
		   APFloat::opOK));
      assert (sub (f_one, f_pos_zero, f_one, rm,
		   APFloat::opOK));
      assert (sub (f_one, f_neg_zero, f_one, rm,
		   APFloat::opOK));
      assert (sub (f_one, f_one, rm == APFloat::rmTowardNegative
		   ? f_neg_zero: f_pos_zero, rm, APFloat::opOK));
      assert (sub (f_one, f_neg_one, f_two, rm,
		   APFloat::opOK));
      assert (sub (f_one, f_nan, f_nan, rm,
		   APFloat::opOK));
    }

  // Subtractions; neg_one lhs.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (sub (f_neg_one, f_pos_infinity, f_neg_infinity, rm,
		   APFloat::opOK));
      assert (sub (f_neg_one, f_neg_infinity, f_pos_infinity, rm,
		   APFloat::opOK));
      assert (sub (f_neg_one, f_pos_zero, f_neg_one, rm,
		   APFloat::opOK));
      assert (sub (f_neg_one, f_neg_zero, f_neg_one, rm,
		   APFloat::opOK));
      assert (sub (f_neg_one, f_one, f_neg_two, rm,
		   APFloat::opOK));
      assert (sub (f_neg_one, f_neg_one, rm == APFloat::rmTowardNegative
		   ? f_neg_zero: f_pos_zero, rm, APFloat::opOK));
      assert (sub (f_neg_one, f_nan, f_nan, rm,
		   APFloat::opOK));
    }

  // FMA, QNaN somewhere.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (fma (f_pos_infinity, f_pos_infinity, f_nan, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_pos_infinity, f_neg_infinity, f_nan, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_neg_infinity, f_pos_infinity, f_nan, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_neg_infinity, f_neg_infinity, f_nan, f_nan, rm,
		   APFloat::opOK));

      assert (fma (f_pos_infinity, f_nan, f_pos_infinity, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_pos_infinity, f_nan, f_neg_infinity, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_neg_infinity, f_nan, f_pos_infinity, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_neg_infinity, f_nan, f_neg_infinity, f_nan, rm,
		   APFloat::opOK));

      assert (fma (f_nan, f_pos_infinity, f_pos_infinity, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_nan, f_pos_infinity, f_neg_infinity, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_nan, f_neg_infinity, f_pos_infinity, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_nan, f_neg_infinity, f_neg_infinity, f_nan, rm,
		   APFloat::opOK));

      assert (fma (f_pos_zero, f_pos_zero, f_nan, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_pos_zero, f_neg_zero, f_nan, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_neg_zero, f_pos_zero, f_nan, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_neg_zero, f_neg_zero, f_nan, f_nan, rm,
		   APFloat::opOK));

      assert (fma (f_pos_zero, f_nan, f_pos_zero, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_pos_zero, f_nan, f_neg_zero, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_neg_zero, f_nan, f_pos_zero, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_neg_zero, f_nan, f_neg_zero, f_nan, rm,
		   APFloat::opOK));

      assert (fma (f_nan, f_pos_zero, f_pos_zero, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_nan, f_pos_zero, f_neg_zero, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_nan, f_neg_zero, f_pos_zero, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_nan, f_neg_zero, f_neg_zero, f_nan, rm,
		   APFloat::opOK));

      assert (fma (f_pos_infinity, f_pos_zero, f_nan, f_nan, rm,
		   APFloat::opInvalidOp));
      assert (fma (f_pos_infinity, f_neg_zero, f_nan, f_nan, rm,
		   APFloat::opInvalidOp));
      assert (fma (f_neg_zero, f_pos_infinity, f_nan, f_nan, rm,
		   APFloat::opInvalidOp));
      assert (fma (f_neg_zero, f_neg_infinity, f_nan, f_nan, rm,
		   APFloat::opInvalidOp));
      assert (fma (f_neg_zero, f_neg_zero, f_nan, f_nan, rm,
		   APFloat::opOK));

      assert (fma (f_pos_infinity, f_nan, f_pos_zero, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_pos_infinity, f_nan, f_neg_zero, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_neg_zero, f_nan, f_pos_infinity, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_neg_zero, f_nan, f_neg_zero, f_nan, rm,
		   APFloat::opOK));

      assert (fma (f_nan, f_pos_infinity, f_pos_zero, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_nan, f_pos_infinity, f_neg_zero, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_nan, f_neg_zero, f_pos_infinity, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_nan, f_neg_zero, f_neg_zero, f_nan, rm,
		   APFloat::opOK));

      assert (fma (f_pos_infinity, f_one, f_nan, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_pos_infinity, f_neg_one, f_nan, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_neg_one, f_pos_infinity, f_nan, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_neg_one, f_neg_one, f_nan, f_nan, rm,
		   APFloat::opOK));

      assert (fma (f_pos_infinity, f_nan, f_one, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_pos_infinity, f_nan, f_neg_one, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_neg_one, f_nan, f_pos_infinity, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_neg_one, f_nan, f_neg_one, f_nan, rm,
		   APFloat::opOK));

      assert (fma (f_nan, f_pos_infinity, f_one, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_nan, f_pos_infinity, f_neg_one, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_nan, f_neg_one, f_pos_infinity, f_nan, rm,
		   APFloat::opOK));
      assert (fma (f_nan, f_neg_one, f_neg_one, f_nan, rm,
		   APFloat::opOK));
    }

  // FMA, non-QNaN +inf first.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (fma (f_pos_infinity, f_pos_infinity, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_pos_infinity, f_pos_infinity, f_neg_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_pos_infinity, f_pos_infinity, f_pos_zero,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_pos_infinity, f_pos_infinity, f_neg_zero,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_pos_infinity, f_pos_infinity, f_one,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_pos_infinity, f_pos_infinity, f_neg_one,
		   f_pos_infinity, rm, APFloat::opOK));

      assert (fma (f_pos_infinity, f_neg_infinity, f_pos_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_pos_infinity, f_neg_infinity, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_pos_infinity, f_neg_infinity, f_pos_zero,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_pos_infinity, f_neg_infinity, f_neg_zero,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_pos_infinity, f_neg_infinity, f_one,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_pos_infinity, f_neg_infinity, f_neg_one,
		   f_neg_infinity, rm, APFloat::opOK));

      assert (fma (f_pos_infinity, f_pos_zero, f_pos_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_pos_infinity, f_pos_zero, f_neg_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_pos_infinity, f_pos_zero, f_pos_zero,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_pos_infinity, f_pos_zero, f_neg_zero,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_pos_infinity, f_pos_zero, f_one,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_pos_infinity, f_pos_zero, f_neg_one,
		   f_nan, rm, APFloat::opInvalidOp));

      assert (fma (f_pos_infinity, f_neg_zero, f_pos_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_pos_infinity, f_neg_zero, f_neg_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_pos_infinity, f_neg_zero, f_pos_zero,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_pos_infinity, f_neg_zero, f_neg_zero,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_pos_infinity, f_neg_zero, f_one,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_pos_infinity, f_neg_zero, f_neg_one,
		   f_nan, rm, APFloat::opInvalidOp));

      assert (fma (f_pos_infinity, f_neg_one, f_pos_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_pos_infinity, f_neg_one, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_pos_infinity, f_neg_one, f_pos_zero,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_pos_infinity, f_neg_one, f_neg_zero,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_pos_infinity, f_neg_one, f_one,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_pos_infinity, f_neg_one, f_neg_one,
		   f_neg_infinity, rm, APFloat::opOK));
    }

  // FMA, non-QNaN -inf first.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      assert (fma (f_neg_infinity, f_pos_infinity, f_pos_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_neg_infinity, f_pos_infinity, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_infinity, f_pos_infinity, f_pos_zero,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_infinity, f_pos_infinity, f_neg_zero,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_infinity, f_pos_infinity, f_one,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_infinity, f_pos_infinity, f_neg_one,
		   f_neg_infinity, rm, APFloat::opOK));

      assert (fma (f_neg_infinity, f_neg_infinity, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_infinity, f_neg_infinity, f_neg_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_neg_infinity, f_neg_infinity, f_pos_zero,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_infinity, f_neg_infinity, f_neg_zero,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_infinity, f_neg_infinity, f_one,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_infinity, f_neg_infinity, f_neg_one,
		   f_pos_infinity, rm, APFloat::opOK));

      assert (fma (f_neg_infinity, f_pos_zero, f_pos_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_neg_infinity, f_pos_zero, f_neg_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_neg_infinity, f_pos_zero, f_pos_zero,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_neg_infinity, f_pos_zero, f_neg_zero,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_neg_infinity, f_pos_zero, f_one,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_neg_infinity, f_pos_zero, f_neg_one,
		   f_nan, rm, APFloat::opInvalidOp));

      assert (fma (f_neg_infinity, f_neg_zero, f_pos_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_neg_infinity, f_neg_zero, f_neg_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_neg_infinity, f_neg_zero, f_pos_zero,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_neg_infinity, f_neg_zero, f_neg_zero,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_neg_infinity, f_neg_zero, f_one,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_neg_infinity, f_neg_zero, f_neg_one,
		   f_nan, rm, APFloat::opInvalidOp));

      assert (fma (f_neg_infinity, f_neg_one, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_infinity, f_neg_one, f_neg_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_neg_infinity, f_neg_one, f_pos_zero,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_infinity, f_neg_one, f_neg_zero,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_infinity, f_neg_one, f_one,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_infinity, f_neg_one, f_neg_one,
		   f_pos_infinity, rm, APFloat::opOK));
    }

  // FMA, non-QNaN +zero first.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      bool down = rm == APFloat::rmTowardNegative;

      assert (fma (f_pos_zero, f_pos_infinity, f_pos_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_pos_zero, f_pos_infinity, f_neg_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_pos_zero, f_pos_infinity, f_pos_zero,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_pos_zero, f_pos_infinity, f_neg_zero,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_pos_zero, f_pos_infinity, f_one,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_pos_zero, f_pos_infinity, f_neg_one,
		   f_nan, rm, APFloat::opInvalidOp));

      assert (fma (f_pos_zero, f_neg_infinity, f_pos_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_pos_zero, f_neg_infinity, f_neg_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_pos_zero, f_neg_infinity, f_pos_zero,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_pos_zero, f_neg_infinity, f_neg_zero,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_pos_zero, f_neg_infinity, f_one,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_pos_zero, f_neg_infinity, f_neg_one,
		   f_nan, rm, APFloat::opInvalidOp));

      assert (fma (f_pos_zero, f_pos_zero, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_pos_zero, f_pos_zero, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_pos_zero, f_pos_zero, f_pos_zero,
		   f_pos_zero, rm, APFloat::opOK));
      assert (fma (f_pos_zero, f_pos_zero, f_neg_zero,
		   down ? f_neg_zero: f_pos_zero, rm, APFloat::opOK));
      assert (fma (f_pos_zero, f_pos_zero, f_one,
		   f_one, rm, APFloat::opOK));
      assert (fma (f_pos_zero, f_pos_zero, f_neg_one,
		   f_neg_one, rm, APFloat::opOK));

      assert (fma (f_pos_zero, f_neg_zero, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_pos_zero, f_neg_zero, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_pos_zero, f_neg_zero, f_pos_zero,
		   down ? f_neg_zero: f_pos_zero, rm, APFloat::opOK));
      assert (fma (f_pos_zero, f_neg_zero, f_neg_zero,
		   f_neg_zero, rm, APFloat::opOK));
      assert (fma (f_pos_zero, f_neg_zero, f_one,
		   f_one, rm, APFloat::opOK));
      assert (fma (f_pos_zero, f_neg_zero, f_neg_one,
		   f_neg_one, rm, APFloat::opOK));

      assert (fma (f_pos_zero, f_neg_one, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_pos_zero, f_neg_one, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_pos_zero, f_neg_one, f_pos_zero,
		   down ? f_neg_zero: f_pos_zero, rm, APFloat::opOK));
      assert (fma (f_pos_zero, f_neg_one, f_neg_zero,
		   f_neg_zero, rm, APFloat::opOK));
      assert (fma (f_pos_zero, f_neg_one, f_one,
		   f_one, rm, APFloat::opOK));
      assert (fma (f_pos_zero, f_neg_one, f_neg_one,
		   f_neg_one, rm, APFloat::opOK));
    }

  // FMA, non-QNaN -zero first.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      bool down = rm == APFloat::rmTowardNegative;

      assert (fma (f_neg_zero, f_pos_infinity, f_pos_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_neg_zero, f_pos_infinity, f_neg_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_neg_zero, f_pos_infinity, f_pos_zero,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_neg_zero, f_pos_infinity, f_neg_zero,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_neg_zero, f_pos_infinity, f_one,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_neg_zero, f_pos_infinity, f_neg_one,
		   f_nan, rm, APFloat::opInvalidOp));

      assert (fma (f_neg_zero, f_neg_infinity, f_pos_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_neg_zero, f_neg_infinity, f_neg_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_neg_zero, f_neg_infinity, f_pos_zero,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_neg_zero, f_neg_infinity, f_neg_zero,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_neg_zero, f_neg_infinity, f_one,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_neg_zero, f_neg_infinity, f_neg_one,
		   f_nan, rm, APFloat::opInvalidOp));

      assert (fma (f_neg_zero, f_pos_zero, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_zero, f_pos_zero, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_zero, f_pos_zero, f_pos_zero,
		   down ? f_neg_zero: f_pos_zero, rm, APFloat::opOK));
      assert (fma (f_neg_zero, f_pos_zero, f_neg_zero,
		   f_neg_zero, rm, APFloat::opOK));
      assert (fma (f_neg_zero, f_pos_zero, f_one,
		   f_one, rm, APFloat::opOK));
      assert (fma (f_neg_zero, f_pos_zero, f_neg_one,
		   f_neg_one, rm, APFloat::opOK));

      assert (fma (f_neg_zero, f_neg_zero, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_zero, f_neg_zero, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_zero, f_neg_zero, f_pos_zero,
		   f_pos_zero, rm, APFloat::opOK));
      assert (fma (f_neg_zero, f_neg_zero, f_neg_zero,
		   down ? f_neg_zero: f_pos_zero, rm, APFloat::opOK));
      assert (fma (f_neg_zero, f_neg_zero, f_one,
		   f_one, rm, APFloat::opOK));
      assert (fma (f_neg_zero, f_neg_zero, f_neg_one,
		   f_neg_one, rm, APFloat::opOK));

      assert (fma (f_neg_zero, f_neg_one, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_zero, f_neg_one, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_zero, f_neg_one, f_pos_zero,
		   f_pos_zero, rm, APFloat::opOK));
      assert (fma (f_neg_zero, f_neg_one, f_neg_zero,
		   down ? f_neg_zero: f_pos_zero, rm, APFloat::opOK));
      assert (fma (f_neg_zero, f_neg_one, f_one,
		   f_one, rm, APFloat::opOK));
      assert (fma (f_neg_zero, f_neg_one, f_neg_one,
		   f_neg_one, rm, APFloat::opOK));
    }

  // FMA, non-QNaN +one first.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      bool down = rm == APFloat::rmTowardNegative;

      assert (fma (f_one, f_pos_infinity, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_one, f_pos_infinity, f_neg_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_one, f_pos_infinity, f_pos_zero,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_one, f_pos_infinity, f_neg_zero,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_one, f_pos_infinity, f_one,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_one, f_pos_infinity, f_neg_one,
		   f_pos_infinity, rm, APFloat::opOK));

      assert (fma (f_one, f_neg_infinity, f_pos_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_one, f_neg_infinity, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_one, f_neg_infinity, f_pos_zero,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_one, f_neg_infinity, f_neg_zero,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_one, f_neg_infinity, f_one,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_one, f_neg_infinity, f_neg_one,
		   f_neg_infinity, rm, APFloat::opOK));

      assert (fma (f_one, f_pos_zero, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_one, f_pos_zero, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_one, f_pos_zero, f_pos_zero,
		   f_pos_zero, rm, APFloat::opOK));
      assert (fma (f_one, f_pos_zero, f_neg_zero,
		   down ? f_neg_zero: f_pos_zero, rm, APFloat::opOK));
      assert (fma (f_one, f_pos_zero, f_one,
		   f_one, rm, APFloat::opOK));
      assert (fma (f_one, f_pos_zero, f_neg_one,
		   f_neg_one, rm, APFloat::opOK));

      assert (fma (f_one, f_neg_zero, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_one, f_neg_zero, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_one, f_neg_zero, f_pos_zero,
		   down ? f_neg_zero: f_pos_zero, rm, APFloat::opOK));
      assert (fma (f_one, f_neg_zero, f_neg_zero,
		   f_neg_zero, rm, APFloat::opOK));
      assert (fma (f_one, f_neg_zero, f_one,
		   f_one, rm, APFloat::opOK));
      assert (fma (f_one, f_neg_zero, f_neg_one,
		   f_neg_one, rm, APFloat::opOK));

      assert (fma (f_one, f_neg_one, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_one, f_neg_one, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_one, f_neg_one, f_pos_zero,
		   f_neg_one, rm, APFloat::opOK));
      assert (fma (f_one, f_neg_one, f_neg_zero,
		   f_neg_one, rm, APFloat::opOK));
      assert (fma (f_one, f_neg_one, f_one,
		   down ? f_neg_zero: f_pos_zero, rm, APFloat::opOK));
      assert (fma (f_one, f_neg_one, f_neg_one,
		   f_neg_two, rm, APFloat::opOK));
    }

  // FMA, non-QNaN -one first.
  for (int i = 0; i < 4; i++)
    {
      APFloat::roundingMode rm ((APFloat::roundingMode) i);

      bool down = rm == APFloat::rmTowardNegative;

      assert (fma (f_neg_one, f_pos_infinity, f_pos_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_neg_one, f_pos_infinity, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_one, f_pos_infinity, f_pos_zero,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_one, f_pos_infinity, f_neg_zero,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_one, f_pos_infinity, f_one,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_one, f_pos_infinity, f_neg_one,
		   f_neg_infinity, rm, APFloat::opOK));

      assert (fma (f_neg_one, f_neg_infinity, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_one, f_neg_infinity, f_neg_infinity,
		   f_nan, rm, APFloat::opInvalidOp));
      assert (fma (f_neg_one, f_neg_infinity, f_pos_zero,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_one, f_neg_infinity, f_neg_zero,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_one, f_neg_infinity, f_one,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_one, f_neg_infinity, f_neg_one,
		   f_pos_infinity, rm, APFloat::opOK));

      assert (fma (f_neg_one, f_pos_zero, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_one, f_pos_zero, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_one, f_pos_zero, f_pos_zero,
		   down ? f_neg_zero: f_pos_zero, rm, APFloat::opOK));
      assert (fma (f_neg_one, f_pos_zero, f_neg_zero,
		   f_neg_zero, rm, APFloat::opOK));
      assert (fma (f_neg_one, f_pos_zero, f_one,
		   f_one, rm, APFloat::opOK));
      assert (fma (f_neg_one, f_pos_zero, f_neg_one,
		   f_neg_one, rm, APFloat::opOK));

      assert (fma (f_neg_one, f_neg_zero, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_one, f_neg_zero, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_one, f_neg_zero, f_pos_zero,
		   f_pos_zero, rm, APFloat::opOK));
      assert (fma (f_neg_one, f_neg_zero, f_neg_zero,
		   down ? f_neg_zero: f_pos_zero, rm, APFloat::opOK));
      assert (fma (f_neg_one, f_neg_zero, f_one,
		   f_one, rm, APFloat::opOK));
      assert (fma (f_neg_one, f_neg_zero, f_neg_one,
		   f_neg_one, rm, APFloat::opOK));

      assert (fma (f_neg_one, f_neg_one, f_pos_infinity,
		   f_pos_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_one, f_neg_one, f_neg_infinity,
		   f_neg_infinity, rm, APFloat::opOK));
      assert (fma (f_neg_one, f_neg_one, f_pos_zero,
		   f_one, rm, APFloat::opOK));
      assert (fma (f_neg_one, f_neg_one, f_neg_zero,
		   f_one, rm, APFloat::opOK));
      assert (fma (f_neg_one, f_neg_one, f_one,
		   f_two, rm, APFloat::opOK));
      assert (fma (f_neg_one, f_neg_one, f_neg_one,
		   down ? f_neg_zero: f_pos_zero, rm, APFloat::opOK));
    }

  return 0;
}
