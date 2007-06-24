/*
   Copyright 2004-2007 Neil Booth.

   See the file "COPYING" for information about the copyright
   and warranty status of this software.
*/

#ifndef LLVM_FLOAT_H
#define LLVM_FLOAT_H

#define HOST_CHAR_BIT 8
#define compileTimeAssert(cond) extern int CTAssert[(cond) ? 1 : -1]
#define integerPartWidth (HOST_CHAR_BIT * sizeof (llvm::integerPart))

namespace llvm {

  /* The most convenient unsigned host type.  */
   __extension__ typedef unsigned long long integerPart;

  /* Exponents are stored as signed numbers.  */
  typedef signed short exponent_t;

  struct flt_semantics;
  struct decimal_number;

  enum e_lost_fraction {
    lf_exactly_zero,
    lf_less_than_half,
    lf_exactly_half,
    lf_more_than_half
  };

class APInt {
 public:
  /* Sets the least significant part of a bignum to the input value,
     and zeroes out higher parts.  */
  static void tcSet (integerPart *, integerPart, unsigned int);

  /* Assign one bignum to another.  */
  static void tcAssign (integerPart *, const integerPart *, unsigned int);

  /* Returns true if a bignum is zero, false otherwise.  */
  static bool tcIsZero (const integerPart *, unsigned int);

  /* Extract the given bit of a bignum; returns 0 or 1.  BIT cannot be
     zero.  */
  static int tcExtractBit (const integerPart *, unsigned int bit);

  /* Set the given bit of a bignum.  BIT cannot be zero.  */
  static void tcSetBit (integerPart *, unsigned int bit);

  /* Returns the bit  number of the least or  most significant set bit
     of  a number.   If  the input  number  has no  bits  set zero  is
     returned.  */
  static unsigned int tcLSB (const integerPart *, unsigned int);
  static unsigned int tcMSB (const integerPart *, unsigned int);

  /* Negate a bignum in-place.  */
  static void tcNegate (integerPart *, unsigned int);

  /* DST += RHS + CARRY where CARRY is zero or one.  Returns the carry
     flag.  */
  static integerPart tcAdd (integerPart *, const integerPart *,
			    integerPart carry, unsigned);

  /* DST -= RHS + CARRY where CARRY is zero or one.  Returns the carry
     flag.  */
  static integerPart tcSubtract (integerPart *, const integerPart *,
				 integerPart carry, unsigned);

  /*  DST += SRC * MULTIPLIER + PART   if add is true
      DST  = SRC * MULTIPLIER + PART   if add is false

      Requires 0 <= DSTPARTS <= SRCPARTS + 1.  If DST overlaps SRC
      they must start at the same point, i.e. DST == SRC.

      If DSTPARTS == SRC_PARTS + 1 no overflow occurs and zero is
      returned.  Otherwise DST is filled with the least significant
      DSTPARTS parts of the result, and if all of the omitted higher
      parts were zero return zero, otherwise overflow occurred and
      return one.  */
  static int tcMultiplyPart (integerPart *dst, const integerPart *src,
			     integerPart multiplier, integerPart carry,
			     unsigned int srcParts, unsigned int dstParts,
			     bool add);

  /* DST = LHS * RHS, where DST has the same width as the operands and
     is filled with the least significant parts of the result.
     Returns one if overflow occurred, otherwise zero.  DST must be
     disjoint from both operands.  */
  static int tcMultiply (integerPart *, const integerPart *,
			 const integerPart *, unsigned);

  /* DST = LHS * RHS, where DST has twice the width as the operands.  No
     overflow occurs.  DST must be disjoint from both operands.  */
  static void tcFullMultiply (integerPart *, const integerPart *,
			      const integerPart *, unsigned);

  /* If RHS is zero LHS and REMAINDER are left unchanged, return one.
     Otherwise set LHS to LHS / RHS with the fractional part
     discarded, set REMAINDER to the remainder, return zero.  i.e.

       OLD_LHS = RHS * LHS + REMAINDER

     SCRATCH is a bignum of the same size as the operands and result
     for use by the routine; its contents need not be initialized and
     are destroyed.  LHS, REMAINDER and SCRATCH must be distinct.  */
  static int tcDivide (integerPart *lhs, const integerPart *rhs,
		       integerPart *remainder, integerPart *scratch,
		       unsigned int parts);

  /* Shift a bignum left COUNT bits.  Shifted in bits are zero.  There
     are no restrictions on COUNT.  */
  static void tcShiftLeft (integerPart *, unsigned int parts,
			   unsigned int count);

  /* Shift a bignum right COUNT bits.  Shifted in bits are zero.
     There are no restrictions on COUNT.  */
  static void tcShiftRight (integerPart *, unsigned int parts,
			    unsigned int count);

  /* The obvious AND, OR and XOR and complement operations.  */
  static void tcAnd (integerPart *, const integerPart *, unsigned int);
  static void tcOr (integerPart *, const integerPart *, unsigned int);
  static void tcXor (integerPart *, const integerPart *, unsigned int);
  static void tcComplement (integerPart *, unsigned int);
  
  /* Comparison (unsigned) of two bignums.  */
  static int tcCompare (const integerPart *, const integerPart *,
			unsigned int);

  /* Increment a bignum in-place.  Return the carry flag.  */
  static integerPart tcIncrement (integerPart *, unsigned int);

  /* Set the least significant BITS and clear the rest.  */
  static void tcSetLeastSignificantBits (integerPart *, unsigned int,
					 unsigned int bits);
};

class t_float {
 public:

  /* We support the following floating point semantics.  */
  static const flt_semantics ieee_single;
  static const flt_semantics ieee_double;
  static const flt_semantics ieee_quad;
  static const flt_semantics x87_double_extended;

  static unsigned int semantics_precision (const flt_semantics &);

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
  t_float (const flt_semantics &, const char *);
  t_float (const flt_semantics &, integerPart);
  t_float (const flt_semantics &, e_category, bool negative);
  t_float (const t_float &);
  ~t_float ();

  /* Arithmetic.  */
  e_status add (const t_float &, e_rounding_mode);
  e_status subtract (const t_float &, e_rounding_mode);
  e_status multiply (const t_float &, e_rounding_mode);
  e_status divide (const t_float &, e_rounding_mode);
  e_status fused_multiply_add (const t_float &, const t_float &,
			       e_rounding_mode);
  void change_sign ();

  /* Conversions.  */
  e_status convert (const flt_semantics &, e_rounding_mode);
  e_status convert_to_integer (integerPart *, unsigned int, bool,
			       e_rounding_mode) const;
  e_status convert_from_integer (const integerPart *, unsigned int, bool,
				 e_rounding_mode);
  e_status convert_from_string (const char *, e_rounding_mode);

  /* Return the value as an IEEE double.  */
  double getAsDouble () const;

  /* Comparison with another floating point number.  */
  e_comparison compare (const t_float &) const;

  /* Simple queries.  */
  e_category get_category () const { return category; }
  const flt_semantics &get_semantics () const { return *semantics; }
  bool is_zero () const { return category == fc_zero; }
  bool is_non_zero () const { return category != fc_zero; }
  bool is_negative () const { return sign; }

  t_float& operator= (const t_float &);

 private:

  /* Trivial queries.  */
  integerPart *sig_parts_array ();
  const integerPart *sig_parts_array () const;
  unsigned int part_count () const;

  /* Significand operations.  */
  integerPart add_significand (const t_float &);
  integerPart subtract_significand (const t_float &, integerPart);
  e_lost_fraction add_or_subtract_significand (const t_float &, bool subtract);
  e_lost_fraction multiply_significand (const t_float &, const t_float *);
  e_lost_fraction divide_significand (const t_float &);
  void increment_significand ();
  void initialize (const flt_semantics *);
  void shift_significand_left (unsigned int);
  e_lost_fraction shift_significand_right (unsigned int);
  unsigned int significand_lsb () const;
  unsigned int significand_msb () const;
  void zero_significand ();

  /* Arithmetic on special values.  */
  e_status add_or_subtract_specials (const t_float &, bool subtract);
  e_status divide_specials (const t_float &);
  e_status multiply_specials (const t_float &);

  /* Miscellany.  */
  e_status normalize (e_rounding_mode, e_lost_fraction);
  e_status add_or_subtract (const t_float &, e_rounding_mode, bool subtract);
  e_comparison compare_absolute_value (const t_float &) const;
  e_status handle_overflow (e_rounding_mode);
  bool round_away_from_zero (e_rounding_mode, e_lost_fraction);
  e_status convert_from_unsigned_integer (integerPart *, unsigned int,
					  e_rounding_mode);
  e_lost_fraction combine_lost_fractions (e_lost_fraction, e_lost_fraction);
  e_status convert_from_hexadecimal_string (const char *, e_rounding_mode);

  void assign (const t_float &);
  void copy_significand (const t_float &);
  void free_significand ();

  /* What kind of semantics does this value obey?  */
  const flt_semantics *semantics;

  /* Significand - the fraction with an explicit integer bit.  Must be
     at least one bit wider than the target precision.  */
  union Significand
  {
    integerPart part;
    integerPart *parts;
  } significand;

  /* The exponent - a signed number.  */
  exponent_t exponent;

  /* What kind of floating point number this is.  */
  e_category category: 2;

  /* The sign bit of this number.  */
  unsigned int sign: 1;
};

}

#endif /* LLVM_FLOAT_H */
