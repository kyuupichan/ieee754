/*
   Copyright 2004-2007 Neil Booth.

   See the file "COPYING" for information about the copyright
   and warranty status of this software.
*/

#ifndef LLVM_FLOAT_H
#define LLVM_FLOAT_H

#define HOST_CHAR_BIT 8
#define compile_time_assert(cond) extern int CTAssert[(cond) ? 1 : -1]
#define t_integer_part_width (HOST_CHAR_BIT * sizeof (llvm::t_integer_part))

namespace llvm {

  /* The most convenient unsigned host type.  */
   __extension__ typedef unsigned long long t_integer_part;

  /* Exponents are stored as signed numbers.  */
  typedef signed short exponent_t;

  struct flt_semantics;

class APInt {
 public:
  /* Sets the least significant part of a bignum to the input value,
     and zeroes out higher parts.  */
  static void tc_set (t_integer_part *, t_integer_part, unsigned int);

  /* Assign one bignum to another.  */
  static void tc_assign (t_integer_part *, const t_integer_part *,
			 unsigned int);

  /* Returns true if a bignum is zero, false otherwise.  */
  static bool tc_is_zero (const t_integer_part *, unsigned int);

  /* Extract the given bit of a bignum; returns 0 or 1.  BIT cannot be
     zero.  */
  static int tc_extract_bit (const t_integer_part *, unsigned int bit);

  /* Set the given bit of a bignum.  BIT cannot be zero.  */
  static void tc_set_bit (t_integer_part *, unsigned int bit);

  /* Returns the bit  number of the least or  most significant set bit
     of  a number.   If  the input  number  has no  bits  set zero  is
     returned.  */
  static unsigned int tc_lsb (const t_integer_part *, unsigned int);
  static unsigned int tc_msb (const t_integer_part *, unsigned int);

  /* Negate a bignum in-place.  */
  static void tc_negate (t_integer_part *, unsigned int);

  /* DST += RHS + CARRY where CARRY is zero or one.  Returns the carry
     flag.  */
  static t_integer_part tc_add (t_integer_part *, const t_integer_part *,
				t_integer_part carry, unsigned);

  /* DST -= RHS + CARRY where CARRY is zero or one.  Returns the carry
     flag.  */
  static t_integer_part tc_subtract (t_integer_part *, const t_integer_part *,
				     t_integer_part carry, unsigned);

  /*  DST += SRC * MULTIPLIER + PART   if add is true
      DST  = SRC * MULTIPLIER + PART   if add is false

      Requires 0 <= DST_PARTS <= SRC_PARTS + 1.  If DST overlaps SRC
      they must start at the same point, i.e. DST == SRC.

      If DST_PARTS == SRC_PARTS + 1 no overflow occurs and zero is
      returned.  Otherwise DST is filled with the least significant
      DST_PARTS parts of the result, and if all of the omitted higher
      parts were zero return zero, otherwise overflow occurred and
      return one.  */
  static int tc_multiply_part (t_integer_part *dst, const t_integer_part *src,
			       t_integer_part multiplier, t_integer_part carry,
			       unsigned int src_parts, unsigned int dst_parts,
			       bool add);

  /* DST = LHS * RHS, where DST has the same width as the operands and
     is filled with the least significant parts of the result.
     Returns one if overflow occurred, otherwise zero.  DST must be
     disjoint from both operands.  */
  static int tc_multiply (t_integer_part *, const t_integer_part *,
			  const t_integer_part *, unsigned);

  /* DST = LHS * RHS, where DST has twice the width as the operands.  No
     overflow occurs.  DST must be disjoint from both operands.  */
  static void tc_full_multiply (t_integer_part *, const t_integer_part *,
				const t_integer_part *, unsigned);

  /* If RHS is zero LHS and REMAINDER are left unchanged, return one.
     Otherwise set LHS to LHS / RHS with the fractional part
     discarded, set REMAINDER to the remainder, return zero.  i.e.

       OLD_LHS = RHS * LHS + REMAINDER

     SCRATCH is a bignum of the same size as the operands and result
     for use by the routine; its contents need not be initialized and
     are destroyed.  LHS, REMAINDER and SCRATCH must be distinct.  */
  static int tc_divide (t_integer_part *lhs, const t_integer_part *rhs,
			t_integer_part *remainder, t_integer_part *scratch,
			unsigned int parts);

  /* Shift a bignum left COUNT bits.  Shifted in bits are zero.  There
     are no restrictions on COUNT.  */
  static void tc_shift_left (t_integer_part *, unsigned int parts,
			     unsigned int count);

  /* Shift a bignum right COUNT bits.  Shifted in bits are zero.
     There are no restrictions on COUNT.  */
  static void tc_shift_right (t_integer_part *, unsigned int parts,
			      unsigned int count);

  /* The obvious AND, OR and XOR and complement operations.  */
  static void tc_and (t_integer_part *, const t_integer_part *, unsigned int);
  static void tc_or (t_integer_part *, const t_integer_part *, unsigned int);
  static void tc_xor (t_integer_part *, const t_integer_part *, unsigned int);
  static void tc_complement (t_integer_part *, unsigned int);
  
  /* Comparison (unsigned) of two bignums.  */
  static int tc_compare (const t_integer_part *, const t_integer_part *,
			 unsigned int);

  /* Increment a bignum in-place.  Return the carry flag.  */
  static t_integer_part tc_increment (t_integer_part *, unsigned int);

  /* Set the least significant BITS and clear the rest.  */
  static void tc_set_least_significant_bits (t_integer_part *, unsigned int,
					     unsigned int bits);
};

class t_float {
 public:

  /* Floating point numbers have a four-state comparison relation.  */
  enum e_comparison {
    fcmp_less_than,
    fcmp_equal,
    fcmp_greater_than,
    fcmp_unordered
  };

  /* IEEE gives four possible rounding modes.  */
  enum e_rounding_mode {
    frm_to_nearest,
    frm_to_plus_infinity,
    frm_to_minus_infinity,
    frm_to_zero
  };

  /* We support the following floating point semantics.  */
  enum e_semantics_kind {
    fsk_ieee_single,
    fsk_ieee_double,
    fsk_ieee_quad,
    fsk_x87_double_extended,
  };

  /* Operation status.  fs_underflow or fs_overflow are always
     returned or-ed with fs_inexact.  */
  enum e_status {
    fs_ok             = 0x00,
    fs_invalid_op     = 0x01,
    fs_div_by_zero    = 0x02,
    fs_overflow       = 0x04,
    fs_underflow      = 0x08,
    fs_inexact        = 0x10
  };

  /* Category of internally-represented number.  */
  enum e_category {
    fc_infinity,
    fc_nan,
    fc_normal,
    fc_zero
  };

  /* Constructors.  */
  t_float (e_semantics_kind, const char *);
  t_float (e_semantics_kind, t_integer_part);
  t_float (e_semantics_kind, e_category, bool negative);
  t_float (const t_float &);
  ~t_float ();

  /* Arithmetic.  */
  e_status add (const t_float &, e_rounding_mode);
  e_status subtract (const t_float &, e_rounding_mode);
  e_status multiply (const t_float &, e_rounding_mode);
  e_status divide (const t_float &, e_rounding_mode);
  void change_sign ();

  /* Conversions.  */
  e_status convert (e_semantics_kind, e_rounding_mode);
  e_status convert_to_integer (t_integer_part *, unsigned int, bool,
			       e_rounding_mode) const;
  e_status convert_from_integer (const t_integer_part *, unsigned int, bool,
				 e_rounding_mode);
  e_status convert_from_string (const char *, e_rounding_mode);

  /* Comparison with another floating point number.  */
  e_comparison compare (const t_float &) const;

  /* Simple queries.  */
  e_category get_category () const { return category; }
  bool is_zero () const { return category == fc_zero; }
  bool is_non_zero () const { return category != fc_zero; }
  bool is_negative () const { return sign; }
  static unsigned int precision_for_kind (e_semantics_kind);

  t_float& operator= (const t_float &);

 private:

  enum e_lost_fraction {
    lf_exactly_zero,
    lf_less_than_half,
    lf_exactly_half,
    lf_more_than_half
  };

  /* Trivial queries.  */
  t_integer_part *sig_parts_array ();
  const t_integer_part *sig_parts_array () const;

  /* Significand operations.  */
  t_integer_part add_or_subtract_significands (const t_float &, bool subtract);
  e_lost_fraction divide_significand (const t_float &);
  e_lost_fraction multiply_significand (const t_float &);
  void increment_significand ();
  void initialize (e_semantics_kind);
  bool is_significand_zero ();
  void shift_significand_left (unsigned int bits);
  e_lost_fraction shift_significand_right (unsigned int bits);
  void negate_significand ();
  unsigned int significand_lsb () const;
  unsigned int significand_msb ();
  void zero_significand ();

  /* Right shift a bignum but return the lost fraction.  */
  static e_lost_fraction shift_right (t_integer_part *, unsigned int parts,
				      unsigned int bits);
  static e_lost_fraction trailing_hexadecimal_fraction (const char *,
							unsigned int);

  /* Non-normalized arithmetic.  */
  e_status unnormalized_add_or_subtract (const t_float &, bool,
					 e_rounding_mode, e_lost_fraction *);
  e_status unnormalized_divide (const t_float &, e_lost_fraction *);
  e_status unnormalized_multiply (const t_float &, e_lost_fraction *);

  /* Normalization.  */
  e_status normalize (e_rounding_mode, e_lost_fraction);

  e_comparison compare_absolute_value (const t_float &) const;
  e_status handle_overflow (e_rounding_mode);
  bool round_away_from_zero (e_rounding_mode, e_lost_fraction);

  /* Miscellany.  */
  e_status convert_from_unsigned_integer (t_integer_part *, unsigned int,
					  e_rounding_mode);
  static unsigned int part_count_for_kind (e_semantics_kind);
  static const flt_semantics &semantics_for_kind (e_semantics_kind);
  e_lost_fraction combine_lost_fractions (e_lost_fraction, e_lost_fraction);
  e_status convert_from_hexadecimal_string (const char *, e_rounding_mode);
  void assign (const t_float &);
  void free_significand ();

  /* Significand - the fraction with an explicit integer bit.  Must be
     at least one bit wider than the target precision.  */
  union
  {
    t_integer_part part;
    t_integer_part *parts;
  } significand;

  /* The exponent - a signed number.  */
  exponent_t exponent;

  /* What kind of semantics does this value obey?  */
  e_semantics_kind kind: 8;

  /* What kind of floating point number this is.  */
  e_category category: 2;

  /* The sign bit of this number.  */
  unsigned int sign: 1;

  /* If the significand uses multiple parts.  */
  unsigned int is_wide: 1;
};

}

#endif /* LLVM_FLOAT_H */
