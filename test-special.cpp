/*
   Copyright 2007 Neil Booth.

   See the file "COPYING" for information about the copyright
   and warranty status of this software.
*/

#include <cassert>
#include "float.h"

using namespace llvm;

static bool
good_compare (const t_float &lhs, const t_float &rhs)
{
  if (lhs.get_category () == t_float::fc_nan)
    return rhs.get_category () == t_float::fc_nan;

  return (lhs.compare (rhs) == t_float::fcmp_equal
	  && lhs.is_negative() == rhs.is_negative());
}

static t_float::e_comparison
compare (const t_float &lhs, const t_float &rhs)
{
  return lhs.compare (rhs);
}

static bool
divide (const t_float &lhs, const t_float &rhs, const t_float &result,
	t_float::e_rounding_mode rounding_mode, t_float::e_status status)
{
  t_float tmp (lhs);

  if (tmp.divide (rhs, rounding_mode) != status)
    return false;

  return good_compare (tmp, result);
}

static bool
mult (const t_float &lhs, const t_float &rhs, const t_float &result,
      t_float::e_rounding_mode rounding_mode, t_float::e_status status)
{
  t_float tmp (lhs);

  if (tmp.multiply (rhs, rounding_mode) != status)
    return false;

  return good_compare (tmp, result);
}

static bool
add (const t_float &lhs, const t_float &rhs, const t_float &result,
     t_float::e_rounding_mode rounding_mode, t_float::e_status status)
{
  t_float tmp (lhs);

  if (tmp.add (rhs, rounding_mode) != status)
    return false;

  return good_compare (tmp, result);
}

static bool
sub (const t_float &lhs, const t_float &rhs, const t_float &result,
     t_float::e_rounding_mode rounding_mode, t_float::e_status status)
{
  t_float tmp (lhs);

  if (tmp.subtract (rhs, rounding_mode) != status)
    return false;

  return good_compare (tmp, result);
}

int main (void)
{
  t_float f_pos_infinity (t_float::ieee_single, t_float::fc_infinity, false);
  t_float f_neg_infinity (t_float::ieee_single, t_float::fc_infinity, true);
  t_float f_pos_zero (t_float::ieee_single, t_float::fc_zero, false);
  t_float f_neg_zero (t_float::ieee_single, t_float::fc_zero, true);
  t_float f_nan (t_float::ieee_single, t_float::fc_nan, false);
  t_float f_one (t_float::ieee_single, 1);
  t_float f_two (t_float::ieee_single, 2);
  t_float f_neg_one (f_one);
  t_float f_neg_two (f_two);

  f_neg_one.change_sign ();
  f_neg_two.change_sign ();

  // Comparisons; pos_infinity lhs.
  assert (compare (f_pos_infinity, f_pos_infinity) == t_float::fcmp_equal);
  assert (compare (f_pos_infinity, f_neg_infinity)
	  == t_float::fcmp_greater_than);
  assert (compare (f_pos_infinity, f_pos_zero) == t_float::fcmp_greater_than);
  assert (compare (f_pos_infinity, f_neg_zero) == t_float::fcmp_greater_than);
  assert (compare (f_pos_infinity, f_one) == t_float::fcmp_greater_than);
  assert (compare (f_pos_infinity, f_neg_one) == t_float::fcmp_greater_than);
  assert (compare (f_pos_infinity, f_nan) == t_float::fcmp_unordered);

  // Comparisons; neg_infinity lhs.
  assert (compare (f_neg_infinity, f_pos_infinity) == t_float::fcmp_less_than);
  assert (compare (f_neg_infinity, f_neg_infinity) == t_float::fcmp_equal);
  assert (compare (f_neg_infinity, f_pos_zero) == t_float::fcmp_less_than);
  assert (compare (f_neg_infinity, f_neg_zero) == t_float::fcmp_less_than);
  assert (compare (f_neg_infinity, f_one) == t_float::fcmp_less_than);
  assert (compare (f_neg_infinity, f_neg_one) == t_float::fcmp_less_than);
  assert (compare (f_neg_infinity, f_nan) == t_float::fcmp_unordered);

  // Comparisons; pos_zero lhs.
  assert (compare (f_pos_zero, f_pos_infinity) == t_float::fcmp_less_than);
  assert (compare (f_pos_zero, f_neg_infinity) == t_float::fcmp_greater_than);
  assert (compare (f_pos_zero, f_pos_zero) == t_float::fcmp_equal);
  assert (compare (f_pos_zero, f_neg_zero) == t_float::fcmp_equal);
  assert (compare (f_pos_zero, f_one) == t_float::fcmp_less_than);
  assert (compare (f_pos_zero, f_neg_one) == t_float::fcmp_greater_than);
  assert (compare (f_pos_zero, f_nan) == t_float::fcmp_unordered);

  // Comparisons; neg_zero lhs.
  assert (compare (f_neg_zero, f_pos_infinity) == t_float::fcmp_less_than);
  assert (compare (f_neg_zero, f_neg_infinity) == t_float::fcmp_greater_than);
  assert (compare (f_neg_zero, f_pos_zero) == t_float::fcmp_equal);
  assert (compare (f_neg_zero, f_neg_zero) == t_float::fcmp_equal);
  assert (compare (f_neg_zero, f_one) == t_float::fcmp_less_than);
  assert (compare (f_neg_zero, f_neg_one) == t_float::fcmp_greater_than);
  assert (compare (f_neg_zero, f_nan) == t_float::fcmp_unordered);

  // Comparisons; NAN lhs.
  assert (compare (f_nan, f_pos_infinity) == t_float::fcmp_unordered);
  assert (compare (f_nan, f_neg_infinity) == t_float::fcmp_unordered);
  assert (compare (f_nan, f_pos_zero) == t_float::fcmp_unordered);
  assert (compare (f_nan, f_neg_zero) == t_float::fcmp_unordered);
  assert (compare (f_nan, f_one) == t_float::fcmp_unordered);
  assert (compare (f_nan, f_neg_one) == t_float::fcmp_unordered);
  assert (compare (f_nan, f_nan) == t_float::fcmp_unordered);

  // Comparisons; one lhs.
  assert (compare (f_one, f_pos_infinity) == t_float::fcmp_less_than);
  assert (compare (f_one, f_neg_infinity) == t_float::fcmp_greater_than);
  assert (compare (f_one, f_pos_zero) == t_float::fcmp_greater_than);
  assert (compare (f_one, f_neg_zero) == t_float::fcmp_greater_than);
  assert (compare (f_one, f_one) == t_float::fcmp_equal);
  assert (compare (f_one, f_neg_one) == t_float::fcmp_greater_than);
  assert (compare (f_one, f_nan) == t_float::fcmp_unordered);

  // Comparisons; negative one lhs.
  assert (compare (f_neg_one, f_pos_infinity) == t_float::fcmp_less_than);
  assert (compare (f_neg_one, f_neg_infinity) == t_float::fcmp_greater_than);
  assert (compare (f_neg_one, f_pos_zero) == t_float::fcmp_less_than);
  assert (compare (f_neg_one, f_neg_zero) == t_float::fcmp_less_than);
  assert (compare (f_neg_one, f_one) == t_float::fcmp_less_than);
  assert (compare (f_neg_one, f_neg_one) == t_float::fcmp_equal);
  assert (compare (f_neg_one, f_nan) == t_float::fcmp_unordered);



  // Divisions; pos_infinity lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (divide (f_pos_infinity, f_pos_infinity, f_nan, rm,
		      t_float::fs_invalid_op));
      assert (divide (f_pos_infinity, f_neg_infinity, f_nan, rm,
		      t_float::fs_invalid_op));
      assert (divide (f_pos_infinity, f_pos_zero, f_pos_infinity, rm,
		      t_float::fs_ok));
      assert (divide (f_pos_infinity, f_neg_zero, f_neg_infinity, rm,
		      t_float::fs_ok));
      assert (divide (f_pos_infinity, f_one, f_pos_infinity, rm,
		      t_float::fs_ok));
      assert (divide (f_pos_infinity, f_neg_one, f_neg_infinity, rm,
		      t_float::fs_ok));
      assert (divide (f_pos_infinity, f_nan, f_nan, rm,
		      t_float::fs_ok));
    }

  // Divisions; neg_infinity lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (divide (f_neg_infinity, f_pos_infinity, f_nan, rm,
		      t_float::fs_invalid_op));
      assert (divide (f_neg_infinity, f_neg_infinity, f_nan, rm,
		      t_float::fs_invalid_op));
      assert (divide (f_neg_infinity, f_pos_zero, f_neg_infinity, rm,
		      t_float::fs_ok));
      assert (divide (f_neg_infinity, f_neg_zero, f_pos_infinity, rm,
		      t_float::fs_ok));
      assert (divide (f_neg_infinity, f_one, f_neg_infinity, rm,
		      t_float::fs_ok));
      assert (divide (f_neg_infinity, f_neg_one, f_pos_infinity, rm,
		      t_float::fs_ok));
      assert (divide (f_neg_infinity, f_nan, f_nan, rm,
		      t_float::fs_ok));
    }

  // Divisions; pos_zero lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (divide (f_pos_zero, f_pos_infinity, f_pos_zero, rm,
		      t_float::fs_ok));
      assert (divide (f_pos_zero, f_neg_infinity, f_neg_zero, rm,
		      t_float::fs_ok));
      assert (divide (f_pos_zero, f_pos_zero, f_nan, rm,
		      t_float::fs_invalid_op));
      assert (divide (f_pos_zero, f_neg_zero, f_nan, rm,
		      t_float::fs_invalid_op));
      assert (divide (f_pos_zero, f_one, f_pos_zero, rm,
		      t_float::fs_ok));
      assert (divide (f_pos_zero, f_neg_one, f_neg_zero, rm,
		      t_float::fs_ok));
      assert (divide (f_pos_zero, f_nan, f_nan, rm,
		      t_float::fs_ok));
    }

  // Divisions; neg_zero lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (divide (f_neg_zero, f_pos_infinity, f_neg_zero, rm,
		      t_float::fs_ok));
      assert (divide (f_neg_zero, f_neg_infinity, f_pos_zero, rm,
		      t_float::fs_ok));
      assert (divide (f_neg_zero, f_pos_zero, f_nan, rm,
		      t_float::fs_invalid_op));
      assert (divide (f_neg_zero, f_neg_zero, f_nan, rm,
		      t_float::fs_invalid_op));
      assert (divide (f_neg_zero, f_one, f_neg_zero, rm,
		      t_float::fs_ok));
      assert (divide (f_neg_zero, f_neg_one, f_pos_zero, rm,
		      t_float::fs_ok));
      assert (divide (f_neg_zero, f_nan, f_nan, rm,
		      t_float::fs_ok));
    }

  // Divisions; nan lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (divide (f_nan, f_pos_infinity, f_nan, rm,
		      t_float::fs_ok));
      assert (divide (f_nan, f_neg_infinity, f_nan, rm,
		      t_float::fs_ok));
      assert (divide (f_nan, f_pos_zero, f_nan, rm,
		      t_float::fs_ok));
      assert (divide (f_nan, f_neg_zero, f_nan, rm,
		      t_float::fs_ok));
      assert (divide (f_nan, f_one, f_nan, rm,
		      t_float::fs_ok));
      assert (divide (f_nan, f_neg_one, f_nan, rm,
		      t_float::fs_ok));
      assert (divide (f_nan, f_nan, f_nan, rm,
		      t_float::fs_ok));
    }

  // Divisions; one lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (divide (f_one, f_pos_infinity, f_pos_zero, rm,
		      t_float::fs_ok));
      assert (divide (f_one, f_neg_infinity, f_neg_zero, rm,
		      t_float::fs_ok));
      assert (divide (f_one, f_pos_zero, f_pos_infinity, rm,
		      t_float::fs_div_by_zero));
      assert (divide (f_one, f_neg_zero, f_neg_infinity, rm,
		      t_float::fs_div_by_zero));
      assert (divide (f_one, f_one, f_one, rm,
		      t_float::fs_ok));
      assert (divide (f_one, f_neg_one, f_neg_one, rm,
		      t_float::fs_ok));
      assert (divide (f_one, f_nan, f_nan, rm,
		      t_float::fs_ok));
    }

  // Divisions; neg_one lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (divide (f_neg_one, f_pos_infinity, f_neg_zero, rm,
		      t_float::fs_ok));
      assert (divide (f_neg_one, f_neg_infinity, f_pos_zero, rm,
		      t_float::fs_ok));
      assert (divide (f_neg_one, f_pos_zero, f_neg_infinity, rm,
		      t_float::fs_div_by_zero));
      assert (divide (f_neg_one, f_neg_zero, f_pos_infinity, rm,
		      t_float::fs_div_by_zero));
      assert (divide (f_neg_one, f_one, f_neg_one, rm,
		      t_float::fs_ok));
      assert (divide (f_neg_one, f_neg_one, f_one, rm,
		      t_float::fs_ok));
      assert (divide (f_neg_one, f_nan, f_nan, rm,
		      t_float::fs_ok));
    }



  // Multiplications; pos_infinity lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (mult (f_pos_infinity, f_pos_infinity, f_pos_infinity, rm,
		    t_float::fs_ok));
      assert (mult (f_pos_infinity, f_neg_infinity, f_neg_infinity, rm,
		    t_float::fs_ok));
      assert (mult (f_pos_infinity, f_pos_zero, f_nan, rm,
		    t_float::fs_invalid_op));
      assert (mult (f_pos_infinity, f_neg_zero, f_nan, rm,
		    t_float::fs_invalid_op));
      assert (mult (f_pos_infinity, f_one, f_pos_infinity, rm,
		    t_float::fs_ok));
      assert (mult (f_pos_infinity, f_neg_one, f_neg_infinity, rm,
		    t_float::fs_ok));
      assert (mult (f_pos_infinity, f_nan, f_nan, rm,
		    t_float::fs_ok));
    }

  // Multiplications; neg_infinity lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (mult (f_neg_infinity, f_pos_infinity, f_neg_infinity, rm,
		    t_float::fs_ok));
      assert (mult (f_neg_infinity, f_neg_infinity, f_pos_infinity, rm,
		    t_float::fs_ok));
      assert (mult (f_neg_infinity, f_pos_zero, f_nan, rm,
		    t_float::fs_invalid_op));
      assert (mult (f_neg_infinity, f_neg_zero, f_nan, rm,
		    t_float::fs_invalid_op));
      assert (mult (f_neg_infinity, f_one, f_neg_infinity, rm,
		    t_float::fs_ok));
      assert (mult (f_neg_infinity, f_neg_one, f_pos_infinity, rm,
		    t_float::fs_ok));
      assert (mult (f_neg_infinity, f_nan, f_nan, rm,
		    t_float::fs_ok));
    }

  // Multiplications; pos_zero lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (mult (f_pos_zero, f_pos_infinity, f_nan, rm,
		    t_float::fs_invalid_op));
      assert (mult (f_pos_zero, f_neg_infinity, f_nan, rm,
		    t_float::fs_invalid_op));
      assert (mult (f_pos_zero, f_pos_zero, f_pos_zero, rm,
		    t_float::fs_ok));
      assert (mult (f_pos_zero, f_neg_zero, f_neg_zero, rm,
		    t_float::fs_ok));
      assert (mult (f_pos_zero, f_one, f_pos_zero, rm,
		    t_float::fs_ok));
      assert (mult (f_pos_zero, f_neg_one, f_neg_zero, rm,
		    t_float::fs_ok));
      assert (mult (f_pos_zero, f_nan, f_nan, rm,
		    t_float::fs_ok));
    }

  // Multiplications; neg_zero lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (mult (f_neg_zero, f_pos_infinity, f_nan, rm,
		    t_float::fs_invalid_op));
      assert (mult (f_neg_zero, f_neg_infinity, f_nan, rm,
		    t_float::fs_invalid_op));
      assert (mult (f_neg_zero, f_pos_zero, f_neg_zero, rm,
		    t_float::fs_ok));
      assert (mult (f_neg_zero, f_neg_zero, f_pos_zero, rm,
		    t_float::fs_ok));
      assert (mult (f_neg_zero, f_one, f_neg_zero, rm,
		    t_float::fs_ok));
      assert (mult (f_neg_zero, f_neg_one, f_pos_zero, rm,
		    t_float::fs_ok));
      assert (mult (f_neg_zero, f_nan, f_nan, rm,
		    t_float::fs_ok));
    }

  // Multiplications; nan lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (mult (f_nan, f_pos_infinity, f_nan, rm,
		    t_float::fs_ok));
      assert (mult (f_nan, f_neg_infinity, f_nan, rm,
		    t_float::fs_ok));
      assert (mult (f_nan, f_pos_zero, f_nan, rm,
		    t_float::fs_ok));
      assert (mult (f_nan, f_neg_zero, f_nan, rm,
		    t_float::fs_ok));
      assert (mult (f_nan, f_one, f_nan, rm,
		    t_float::fs_ok));
      assert (mult (f_nan, f_neg_one, f_nan, rm,
		    t_float::fs_ok));
      assert (mult (f_nan, f_nan, f_nan, rm,
		    t_float::fs_ok));
    }

  // Multiplications; one lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (mult (f_one, f_pos_infinity, f_pos_infinity, rm,
		    t_float::fs_ok));
      assert (mult (f_one, f_neg_infinity, f_neg_infinity, rm,
		    t_float::fs_ok));
      assert (mult (f_one, f_pos_zero, f_pos_zero, rm,
		    t_float::fs_ok));
      assert (mult (f_one, f_neg_zero, f_neg_zero, rm,
		    t_float::fs_ok));
      assert (mult (f_one, f_one, f_one, rm,
		    t_float::fs_ok));
      assert (mult (f_one, f_neg_one, f_neg_one, rm,
		    t_float::fs_ok));
      assert (mult (f_one, f_nan, f_nan, rm,
		    t_float::fs_ok));
    }

  // Multiplications; neg_one lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (mult (f_neg_one, f_pos_infinity, f_neg_infinity, rm,
		    t_float::fs_ok));
      assert (mult (f_neg_one, f_neg_infinity, f_pos_infinity, rm,
		    t_float::fs_ok));
      assert (mult (f_neg_one, f_pos_zero, f_neg_zero, rm,
		    t_float::fs_ok));
      assert (mult (f_neg_one, f_neg_zero, f_pos_zero, rm,
		    t_float::fs_ok));
      assert (mult (f_neg_one, f_one, f_neg_one, rm,
		    t_float::fs_ok));
      assert (mult (f_neg_one, f_neg_one, f_one, rm,
		    t_float::fs_ok));
      assert (mult (f_neg_one, f_nan, f_nan, rm,
		    t_float::fs_ok));
    }



  // Additions; pos_infinity lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (add (f_pos_infinity, f_pos_infinity, f_pos_infinity, rm,
		   t_float::fs_ok));
      assert (add (f_pos_infinity, f_neg_infinity, f_nan, rm,
		   t_float::fs_invalid_op));
      assert (add (f_pos_infinity, f_pos_zero, f_pos_infinity, rm,
		   t_float::fs_ok));
      assert (add (f_pos_infinity, f_neg_zero, f_pos_infinity, rm,
		   t_float::fs_ok));
      assert (add (f_pos_infinity, f_one, f_pos_infinity, rm,
		   t_float::fs_ok));
      assert (add (f_pos_infinity, f_neg_one, f_pos_infinity, rm,
		   t_float::fs_ok));
      assert (add (f_pos_infinity, f_nan, f_nan, rm,
		   t_float::fs_ok));
    }

  // Additions; neg_infinity lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (add (f_neg_infinity, f_pos_infinity, f_nan, rm,
		   t_float::fs_invalid_op));
      assert (add (f_neg_infinity, f_neg_infinity, f_neg_infinity, rm,
		   t_float::fs_ok));
      assert (add (f_neg_infinity, f_pos_zero, f_neg_infinity, rm,
		   t_float::fs_ok));
      assert (add (f_neg_infinity, f_neg_zero, f_neg_infinity, rm,
		   t_float::fs_ok));
      assert (add (f_neg_infinity, f_one, f_neg_infinity, rm,
		   t_float::fs_ok));
      assert (add (f_neg_infinity, f_neg_one, f_neg_infinity, rm,
		   t_float::fs_ok));
      assert (add (f_neg_infinity, f_nan, f_nan, rm,
		   t_float::fs_ok));
    }

  // Additions; pos_zero lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (add (f_pos_zero, f_pos_infinity, f_pos_infinity, rm,
		   t_float::fs_ok));
      assert (add (f_pos_zero, f_neg_infinity, f_neg_infinity, rm,
		   t_float::fs_ok));
      assert (add (f_pos_zero, f_pos_zero, f_pos_zero, rm,
		   t_float::fs_ok));
      assert (add (f_pos_zero, f_neg_zero, rm == t_float::frm_to_minus_infinity
		   ? f_neg_zero: f_pos_zero, rm, t_float::fs_ok));
      assert (add (f_pos_zero, f_one, f_one, rm,
		   t_float::fs_ok));
      assert (add (f_pos_zero, f_neg_one, f_neg_one, rm,
		   t_float::fs_ok));
      assert (add (f_pos_zero, f_nan, f_nan, rm,
		   t_float::fs_ok));
    }

  // Additions; neg_zero lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (add (f_neg_zero, f_pos_infinity, f_pos_infinity, rm,
		   t_float::fs_ok));
      assert (add (f_neg_zero, f_neg_infinity, f_neg_infinity, rm,
		   t_float::fs_ok));
      assert (add (f_neg_zero, f_pos_zero, rm == t_float::frm_to_minus_infinity
		   ? f_neg_zero: f_pos_zero, rm, t_float::fs_ok));
      assert (add (f_neg_zero, f_neg_zero, f_neg_zero, rm,
		   t_float::fs_ok));
      assert (add (f_neg_zero, f_one, f_one, rm,
		   t_float::fs_ok));
      assert (add (f_neg_zero, f_neg_one, f_neg_one, rm,
		   t_float::fs_ok));
      assert (add (f_neg_zero, f_nan, f_nan, rm,
		   t_float::fs_ok));
    }

  // Additions; nan lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (add (f_nan, f_pos_infinity, f_nan, rm,
		   t_float::fs_ok));
      assert (add (f_nan, f_neg_infinity, f_nan, rm,
		   t_float::fs_ok));
      assert (add (f_nan, f_pos_zero, f_nan, rm,
		   t_float::fs_ok));
      assert (add (f_nan, f_neg_zero, f_nan, rm,
		   t_float::fs_ok));
      assert (add (f_nan, f_one, f_nan, rm,
		   t_float::fs_ok));
      assert (add (f_nan, f_neg_one, f_nan, rm,
		   t_float::fs_ok));
      assert (add (f_nan, f_nan, f_nan, rm,
		   t_float::fs_ok));
    }

  // Additions; one lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (add (f_one, f_pos_infinity, f_pos_infinity, rm,
		   t_float::fs_ok));
      assert (add (f_one, f_neg_infinity, f_neg_infinity, rm,
		   t_float::fs_ok));
      assert (add (f_one, f_pos_zero, f_one, rm,
		   t_float::fs_ok));
      assert (add (f_one, f_neg_zero, f_one, rm,
		   t_float::fs_ok));
      assert (add (f_one, f_one, f_two, rm,
		   t_float::fs_ok));
      assert (add (f_one, f_neg_one, rm == t_float::frm_to_minus_infinity
		   ? f_neg_zero: f_pos_zero, rm, t_float::fs_ok));
      assert (add (f_one, f_nan, f_nan, rm,
		   t_float::fs_ok));
    }

  // Additions; neg_one lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (add (f_neg_one, f_pos_infinity, f_pos_infinity, rm,
		   t_float::fs_ok));
      assert (add (f_neg_one, f_neg_infinity, f_neg_infinity, rm,
		   t_float::fs_ok));
      assert (add (f_neg_one, f_pos_zero, f_neg_one, rm,
		   t_float::fs_ok));
      assert (add (f_neg_one, f_neg_zero, f_neg_one, rm,
		   t_float::fs_ok));
      assert (add (f_neg_one, f_one, rm == t_float::frm_to_minus_infinity
		   ? f_neg_zero: f_pos_zero, rm, t_float::fs_ok));
      assert (add (f_neg_one, f_neg_one, f_neg_two, rm,
		   t_float::fs_ok));
      assert (add (f_neg_one, f_nan, f_nan, rm,
		   t_float::fs_ok));
    }



  // Subtractions; pos_infinity lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (sub (f_pos_infinity, f_pos_infinity, f_nan, rm,
		   t_float::fs_invalid_op));
      assert (sub (f_pos_infinity, f_neg_infinity, f_pos_infinity, rm,
		   t_float::fs_ok));
      assert (sub (f_pos_infinity, f_pos_zero, f_pos_infinity, rm,
		   t_float::fs_ok));
      assert (sub (f_pos_infinity, f_neg_zero, f_pos_infinity, rm,
		   t_float::fs_ok));
      assert (sub (f_pos_infinity, f_one, f_pos_infinity, rm,
		   t_float::fs_ok));
      assert (sub (f_pos_infinity, f_neg_one, f_pos_infinity, rm,
		   t_float::fs_ok));
      assert (sub (f_pos_infinity, f_nan, f_nan, rm,
		   t_float::fs_ok));
    }

  // Subtractions; neg_infinity lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (sub (f_neg_infinity, f_pos_infinity, f_neg_infinity, rm,
		   t_float::fs_ok));
      assert (sub (f_neg_infinity, f_neg_infinity, f_nan, rm,
		   t_float::fs_invalid_op));
      assert (sub (f_neg_infinity, f_pos_zero, f_neg_infinity, rm,
		   t_float::fs_ok));
      assert (sub (f_neg_infinity, f_neg_zero, f_neg_infinity, rm,
		   t_float::fs_ok));
      assert (sub (f_neg_infinity, f_one, f_neg_infinity, rm,
		   t_float::fs_ok));
      assert (sub (f_neg_infinity, f_neg_one, f_neg_infinity, rm,
		   t_float::fs_ok));
      assert (sub (f_neg_infinity, f_nan, f_nan, rm,
		   t_float::fs_ok));
    }

  // Subtractions; pos_zero lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (sub (f_pos_zero, f_pos_infinity, f_neg_infinity, rm,
		   t_float::fs_ok));
      assert (sub (f_pos_zero, f_neg_infinity, f_pos_infinity, rm,
		   t_float::fs_ok));
      assert (sub (f_pos_zero, f_pos_zero, rm == t_float::frm_to_minus_infinity
		   ? f_neg_zero: f_pos_zero, rm, t_float::fs_ok));
      assert (sub (f_pos_zero, f_neg_zero, f_pos_zero, rm,
		   t_float::fs_ok));
      assert (sub (f_pos_zero, f_one, f_neg_one, rm,
		   t_float::fs_ok));
      assert (sub (f_pos_zero, f_neg_one, f_one, rm,
		   t_float::fs_ok));
      assert (sub (f_pos_zero, f_nan, f_nan, rm,
		   t_float::fs_ok));
    }

  // Subtractions; neg_zero lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (sub (f_neg_zero, f_pos_infinity, f_neg_infinity, rm,
		   t_float::fs_ok));
      assert (sub (f_neg_zero, f_neg_infinity, f_pos_infinity, rm,
		   t_float::fs_ok));
      assert (sub (f_neg_zero, f_pos_zero, f_neg_zero, rm,
		   t_float::fs_ok));
      assert (sub (f_neg_zero, f_neg_zero, rm == t_float::frm_to_minus_infinity
		   ? f_neg_zero: f_pos_zero, rm, t_float::fs_ok));
      assert (sub (f_neg_zero, f_one, f_neg_one, rm,
		   t_float::fs_ok));
      assert (sub (f_neg_zero, f_neg_one, f_one, rm,
		   t_float::fs_ok));
      assert (sub (f_neg_zero, f_nan, f_nan, rm,
		   t_float::fs_ok));
    }

  // Subtractions; nan lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (sub (f_nan, f_pos_infinity, f_nan, rm,
		   t_float::fs_ok));
      assert (sub (f_nan, f_neg_infinity, f_nan, rm,
		   t_float::fs_ok));
      assert (sub (f_nan, f_pos_zero, f_nan, rm,
		   t_float::fs_ok));
      assert (sub (f_nan, f_neg_zero, f_nan, rm,
		   t_float::fs_ok));
      assert (sub (f_nan, f_one, f_nan, rm,
		   t_float::fs_ok));
      assert (sub (f_nan, f_neg_one, f_nan, rm,
		   t_float::fs_ok));
      assert (sub (f_nan, f_nan, f_nan, rm,
		   t_float::fs_ok));
    }

  // Subtractions; one lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (sub (f_one, f_pos_infinity, f_neg_infinity, rm,
		   t_float::fs_ok));
      assert (sub (f_one, f_neg_infinity, f_pos_infinity, rm,
		   t_float::fs_ok));
      assert (sub (f_one, f_pos_zero, f_one, rm,
		   t_float::fs_ok));
      assert (sub (f_one, f_neg_zero, f_one, rm,
		   t_float::fs_ok));
      assert (sub (f_one, f_one, rm == t_float::frm_to_minus_infinity
		   ? f_neg_zero: f_pos_zero, rm, t_float::fs_ok));
      assert (sub (f_one, f_neg_one, f_two, rm,
		   t_float::fs_ok));
      assert (sub (f_one, f_nan, f_nan, rm,
		   t_float::fs_ok));
    }

  // Subtractions; neg_one lhs.
  for (int i = 0; i < 4; i++)
    {
      t_float::e_rounding_mode rm ((t_float::e_rounding_mode) i);

      assert (sub (f_neg_one, f_pos_infinity, f_neg_infinity, rm,
		   t_float::fs_ok));
      assert (sub (f_neg_one, f_neg_infinity, f_pos_infinity, rm,
		   t_float::fs_ok));
      assert (sub (f_neg_one, f_pos_zero, f_neg_one, rm,
		   t_float::fs_ok));
      assert (sub (f_neg_one, f_neg_zero, f_neg_one, rm,
		   t_float::fs_ok));
      assert (sub (f_neg_one, f_one, f_neg_two, rm,
		   t_float::fs_ok));
      assert (sub (f_neg_one, f_neg_one, rm == t_float::frm_to_minus_infinity
		   ? f_neg_zero: f_pos_zero, rm, t_float::fs_ok));
      assert (sub (f_neg_one, f_nan, f_nan, rm,
		   t_float::fs_ok));
    }

  return 0;
}
